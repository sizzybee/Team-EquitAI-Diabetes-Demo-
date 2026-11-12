"""
Sex gaps, stratified by race/ethnicity, for a set of conditions in
NAMCS 2019 (office visits) and NHAMCS 2022 (ED visits). Adults only.

Usage:
    .venv/Scripts/python.exe audit_by_race.py                 # everything
    .venv/Scripts/python.exe audit_by_race.py namcs           # NAMCS checks only
    .venv/Scripts/python.exe audit_by_race.py nhamcs          # NHAMCS checks only

Race groups (both files; neither separates South Asian from East Asian):
    hispanic  ETHIM = 1, any race
    white     non-Hispanic white
    black     non-Hispanic Black
    asian     non-Hispanic, race REPORTED as Asian (RACEUN = 3). The imputed race
              variable folds Asian into "other", so visits with race left blank
              (~19-26%) cannot be counted as Asian.
    other     everything else

For each check the summary table shows, per outcome:
    F% / M%      survey-weighted % (PATWT)
    gap          F% - M% (percentage points)
    OR [CI] p    adjusted odds ratio for female (OR < 1 = women less likely)
    <race> gap   F% - M% within that race; "*" = fewer than 30 records per sex
                 (unreliable), "-" = no records
    interact_p   does the sex gap differ by race? (female x race Wald test)
Long-format results are saved to CSV for every check.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import nhamcs_filter as nf
import focus_analyses as fa
import namcs_focus as nm
from namcs_diabetes_race import add_race, RACES

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

NAMCS_COVARS = ["age", "C(race)", "C(payer)", "C(msa)", "C(specialty)", "seen_before",
                "n_chronic", "cad", "chf", "copd", "diabetes", "htn", "depression"]
NHAMCS_COVARS = ["age", "C(race)", "C(payer)", "C(region)", "n_chronic", "cad", "chf", "copd",
                 "diabetes", "htn", "ems", "abnormal_vitals"]

ANXIETY = ["F41"]
DEPRESSION = ["F32", "F33"]
SOMATIC = ["F45"]
AUTOIMMUNE = {"lupus": ["M32"], "rheumatoid_arthritis": ["M05", "M06"], "multiple_sclerosis": ["G35"],
              "sjogren": ["M350"], "ibd": ["K50", "K51"], "hashimoto_graves": ["E063", "E050"],
              "psoriatic_ankylosing": ["L405", "M45"]}


# ------------------------------------------------------------- engine ----
def wpct(x, outcome, sex):
    m = x["sex"].eq(sex) & x[outcome].notna()
    if m.sum() == 0:
        return np.nan, 0, 0
    y = x.loc[m, outcome].astype(float)
    return 100 * (x.loc[m, "w"] * y).sum() / x.loc[m, "w"].sum(), int(m.sum()), int(y.sum())


def interaction_p(x, outcome, covars):
    covars = [c for c in covars if c != "C(race)"]
    cols = [outcome] + [c.replace("C(", "").replace(")", "") for c in covars]
    x = x.dropna(subset=cols).copy()
    x[outcome] = x[outcome].astype(float)
    keep = x.groupby("race5")[outcome].transform("sum") >= 5      # drop races with almost no events
    x = x[keep]
    if x["race5"].nunique() < 2 or x[outcome].sum() < 30:
        return np.nan
    try:
        if "YEAR" in x.columns and x["YEAR"].nunique() > 1:
            covars = covars + ["C(YEAR)"]
        r = sm.GLM.from_formula(f"{outcome} ~ female * C(race5) + " + " + ".join(covars), x,
                                family=sm.families.Binomial(),
                                var_weights=x["w"] / x["w"].mean()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(x["CPSUM"])[0]})
        names = [n for n in r.params.index if n.startswith("female:")]
        return round(float(r.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True).pvalue), 3)
    except Exception:
        return np.nan


def analyse(name, x, outcomes, covars, extra=(), drop=()):
    """Sex gap overall + within each race for every outcome."""
    fa.BASE_COVARS = covars
    nF, nM = int(x.female.sum()), int((1 - x.female).sum())
    counts = pd.crosstab(x.race5, x.sex).reindex(RACES).fillna(0).astype(int)
    print("\n" + "=" * 110 + f"\n{name}   ({len(x):,} visits: {nF} women, {nM} men)\n" + "=" * 110)
    print("records by race:  " + "   ".join(f"{r}: {counts.loc[r].get('Female', 0)}F/{counts.loc[r].get('Male', 0)}M"
                                            for r in RACES))
    summary, long = [], []
    for out in outcomes:
        if x[out].notna().sum() == 0:
            continue
        f, fn, fy = wpct(x, out, "Female")
        m, mn, my = wpct(x, out, "Male")
        adj = fa.adjusted(x, out, list(extra), out, drop=list(drop))
        row = {"outcome": out, "F%": round(f, 1), "M%": round(m, 1), "gap": round(f - m, 1),
               "F_yes": fy, "M_yes": my,
               "OR [95% CI]": (f"{adj['OR_female']:.2f} [{adj['CI_low']:.2f}-{adj['CI_high']:.2f}]"
                               if pd.notna(adj.get("OR_female")) and pd.notna(adj.get("CI_low"))
                               and adj["CI_high"] < 1000 else
                               ("too few" if pd.isna(adj.get("OR_female")) else "unstable")),
               "p": adj.get("p", np.nan)}
        long.append({"check": name, "race": "all", "outcome": out, "F_n": fn, "F_yes": fy, "F_%": f,
                     "M_n": mn, "M_yes": my, "M_%": m, "gap": f - m, **{k: adj.get(k) for k in
                     ("OR_female", "CI_low", "CI_high", "p")}})
        for r in RACES:
            xr = x[x.race5 == r]
            fr, frn, fry = wpct(xr, out, "Female")
            mr, mrn, mry = wpct(xr, out, "Male")
            if frn == 0 or mrn == 0:
                row[f"{r}"] = "-"
            else:
                flag = "" if min(frn, mrn) >= 30 else "*"
                row[f"{r}"] = f"{fr - mr:+.1f}{flag}"
            long.append({"check": name, "race": r, "outcome": out, "F_n": frn, "F_yes": fry, "F_%": fr,
                         "M_n": mrn, "M_yes": mry, "M_%": mr, "gap": fr - mr})
        row["interact_p"] = interaction_p(x, out, [c for c in covars if c not in drop] + list(extra))
        summary.append(row)
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    return pd.DataFrame(long)


def label_symptom(x, sx):
    x = x.copy()
    x["symptom"] = "other"
    for nm_, codes in reversed(list(sx.items())):
        x.loc[fa.has_rfv(x, codes), "symptom"] = nm_
    return x


# ------------------------------------------------------------- NAMCS -----
def namcs_checks(path):
    d = add_race(nm.prep(nf.load(path)))
    d = d[nf.num(d, "INJURY") != 1]
    d["dx_anxiety"] = fa.has_dx(d, ANXIETY).astype(int)
    d["dx_depression"] = fa.has_dx(d, DEPRESSION).astype(int)
    d["dx_somatic"] = fa.has_dx(d, SOMATIC).astype(int)
    d["dx_anxiety_or_somatic"] = d[["dx_anxiety", "dx_somatic"]].max(axis=1)
    d["dx_any_autoimmune"] = fa.has_dx(d, sum(AUTOIMMUNE.values(), [])).astype(int)
    d["dx_thyroid"] = fa.has_dx(d, ["E0"]).astype(int)
    d["dx_hypothyroid"] = fa.has_dx(d, ["E01", "E02", "E03", "E063"]).astype(int)
    d["blood_tests"] = nf.yes(d, ["CBC", "CMP", "BMP"]).astype(int)
    d["any_imaging"] = nf.yes(d, "ANYIMAGE").astype(int)
    d["xray"] = nf.yes(d, "XRAY").astype(int)
    d["spirometry"] = nf.yes(d, ["SPIRO", "PEAK"]).astype(int)
    d["dx_asthma_copd"] = fa.has_dx(d, ["J44", "J43", "J45"]).astype(int)
    d["sent_to_ed"] = nf.yes(d, "ERADMHOS").astype(int)
    d["return_appt"] = nf.yes(d, ["RETAPPT1", "RETAPPT2", "RETAPPT3", "RETUNSP"]).astype(int)
    results = []

    # 1. Autoimmune -----------------------------------------------------------
    print("\n\n######## NAMCS 1. AUTOIMMUNE ########")
    print("Visits with an autoimmune diagnosis listed (women / men):")
    for k, p in AUTOIMMUNE.items():
        m = fa.has_dx(d, p)
        print(f"  {k:22s} {int((m & d.female.eq(1)).sum()):4d} / {int((m & d.female.eq(0)).sum()):4d}")
    print("Too few to compare care for diagnosed patients. Instead: visits for symptoms that autoimmune")
    print("disease often first shows up as (fatigue, numbness/tingling, joint/muscle symptoms without injury).")
    sx = {"fatigue": [1015], "numbness/tingling": [1220], "joint/muscle (unspecified)": [1965, 1970, 1980],
          "general weakness": [1020]}
    a = label_symptom(d[fa.has_rfv(d, sum(sx.values(), []))], sx)
    results.append(analyse("NAMCS autoimmune-type symptoms", a,
                           ["blood_tests", "thyroid_test", "referred_other_md", "dx_any_autoimmune",
                            "dx_symptom_code_only", "dx_anxiety", "dx_depression", "dx_anxiety_or_somatic"],
                           NAMCS_COVARS, ["C(symptom)"]))

    # 2. Thyroid -------------------------------------------------------------
    print("\n\n######## NAMCS 2. THYROID ########")
    print("Thyroid disease is several times more common in women, so a higher diagnosis rate in women")
    print("is expected. The question is whether TESTING follows symptoms equally.")
    sx = {"fatigue": [1015], "weight gain": [1040], "weight loss": [1045], "palpitations": [1260],
          "anxiety/nervousness": [1100], "depression": [1110]}
    t = label_symptom(d[fa.has_rfv(d, sum(sx.values(), []))], sx)
    results.append(analyse("NAMCS thyroid-type symptoms: thyroid test ordered", t,
                           ["thyroid_test", "dx_thyroid", "dx_anxiety", "dx_depression"],
                           NAMCS_COVARS, ["C(symptom)"]))
    thy = d[d.dx_thyroid == 1]
    results.append(analyse("NAMCS thyroid disease diagnosed: monitoring", thy,
                           ["thyroid_test", "blood_tests", "referred_other_md", "return_appt"],
                           NAMCS_COVARS))

    # 3-4. Shortness of breath, chest pain -----------------------------------
    print("\n\n######## NAMCS 3-4. SHORTNESS OF BREATH / CHEST PAIN ########")
    sob = d[fa.has_rfv(d, [1415])]
    results.append(analyse("NAMCS shortness of breath", sob,
                           ["ecg", "echo", "stress_test", "xray", "spirometry", "blood_tests", "any_cardiac_test",
                            "dx_any_cardiac", "dx_asthma_copd", "dx_anxiety", "dx_symptom_code_only",
                            "referred_other_md", "sent_to_ed"], NAMCS_COVARS))
    cp = d[fa.has_rfv(d, [1050])]
    results.append(analyse("NAMCS chest pain", cp,
                           ["ecg", "echo", "stress_test", "any_cardiac_test", "xray", "blood_tests",
                            "dx_any_cardiac", "dx_anxiety", "dx_symptom_code_only", "referred_other_md",
                            "sent_to_ed"], NAMCS_COVARS))

    # 5. Physical symptoms labelled as anxiety --------------------------------
    print("\n\n######## NAMCS 5. PHYSICAL SYMPTOMS -> ANXIETY LABEL ########")
    print("Visits where the patient came in for a PHYSICAL symptom (not anxiety/depression).")
    print("An anxiety diagnosis is not proof of misdiagnosis -- but a sex gap in it, at the same")
    print("symptoms and same depression history, is the pattern that would hide missed disease.")
    sx = {"chest pain": [1050], "short of breath": [1415], "palpitations": [1260], "dizziness": [1225],
          "fatigue": [1015], "headache": [1210], "abdominal pain": [1545], "numbness": [1220],
          "nausea": [1525], "fainting": [1030], "weakness": [1020]}
    p = label_symptom(d[fa.has_rfv(d, sum(sx.values(), [])) & ~fa.has_rfv(d, [1100, 1110])], sx)
    results.append(analyse("NAMCS physical symptoms", p,
                           ["dx_anxiety", "dx_somatic", "dx_anxiety_or_somatic", "dx_depression",
                            "anxiolytic_listed", "mental_health_referral", "dx_symptom_code_only",
                            "blood_tests", "any_imaging", "any_cardiac_test", "referred_other_md"],
                           NAMCS_COVARS, ["C(symptom)"]))
    print("\nAnxiety diagnosis by symptom (weighted %, women vs men):")
    print(fa.raw(p, "dx_anxiety", "symptom", "anxiety dx").to_string(index=False))
    return results


# ------------------------------------------------------------ NHAMCS -----
def nhamcs_checks(path):
    d = add_race(fa.prep(nf.load(path)))
    d = d[d.left_early == 0]
    for k, col in {"ecg": "EKG", "troponin": "CARDENZ", "cardiac_monitor": "CARDMON", "bnp": "BNP",
                   "d_dimer": "DDIMER", "xray": "XRAY", "ct_chest": "CTCHEST", "cbc": "CBC"}.items():
        d[k] = nf.yes(d, col).astype(int)
    d["dx_mi"] = fa.has_dx(d, ["I21", "I22"]).astype(int)
    d["dx_acs"] = fa.has_dx(d, ["I20", "I21", "I22", "I23", "I24"]).astype(int)
    d["dx_any_cardiac"] = fa.has_dx(d, ["I2", "I3", "I4", "I50", "I51"]).astype(int)
    d["dx_heart_failure"] = fa.has_dx(d, ["I50"]).astype(int)
    d["dx_pe"] = fa.has_dx(d, ["I26"]).astype(int)
    d["dx_asthma_copd"] = fa.has_dx(d, ["J44", "J43", "J45"]).astype(int)
    d["dx_pneumonia"] = fa.has_dx(d, ["J12", "J13", "J14", "J15", "J16", "J17", "J18"]).astype(int)
    d["dx_anxiety"] = fa.has_dx(d, ANXIETY).astype(int)
    diag = d[nf.DIAG_COLS].apply(nf.clean_diag)
    real = diag.apply(lambda s: s.str.match(r"^[A-QS-Z]\d"))
    d["dx_symptom_code_only"] = (diag["DIAG1"].str.startswith("R") & ~real.any(axis=1)).astype(int)
    d["long_wait"] = (d.wait > 30).where(d.wait.notna())
    d["chest_pain_complaint"] = fa.has_rfv(d, [1050]).astype(int)
    d["sob_complaint"] = fa.has_rfv(d, [1415]).astype(int)
    results = []
    workup = ["triage_urgent", "long_wait", "ecg", "troponin", "cardiac_monitor", "bnp", "d_dimer",
              "xray", "ct_chest", "med_aspirin", "med_pain_relief", "admitted"]

    # 1. Heart attack ---------------------------------------------------------
    print("\n\n######## NHAMCS 1. HEART ATTACK (diagnosed MI, I21-I22) ########")
    mi = d[d.dx_mi == 1]
    print(f"Only {len(mi)} MI visits -> sex comparison only; race splits are shown but are not interpretable.")
    results.append(analyse("NHAMCS diagnosed heart attack (MI)", mi,
                           ["chest_pain_complaint", "sob_complaint", "triage_urgent", "long_wait", "ecg",
                            "troponin", "med_aspirin", "med_nitro", "admitted"], NHAMCS_COVARS))
    acs = d[d.dx_acs == 1]
    results.append(analyse("NHAMCS acute coronary syndrome (I20-I24, wider net)", acs,
                           ["chest_pain_complaint", "triage_urgent", "long_wait", "ecg", "troponin",
                            "med_aspirin", "admitted"], NHAMCS_COVARS))

    # 2-3. Shortness of breath, chest pain ------------------------------------
    print("\n\n######## NHAMCS 2-3. SHORTNESS OF BREATH / CHEST PAIN ########")
    base = d[d.injury == 0]
    results.append(analyse("NHAMCS shortness of breath", base[fa.has_rfv(base, [1415])],
                           workup + ["dx_any_cardiac", "dx_heart_failure", "dx_asthma_copd", "dx_pneumonia",
                                     "dx_pe", "dx_anxiety", "dx_symptom_code_only"], NHAMCS_COVARS, ["pain"]))
    results.append(analyse("NHAMCS chest pain", base[fa.has_rfv(base, [1050])],
                           workup + ["dx_acs", "dx_any_cardiac", "dx_pe", "dx_anxiety", "dx_symptom_code_only"],
                           NHAMCS_COVARS, ["pain"]))

    # 4. Triage ---------------------------------------------------------------
    print("\n\n######## NHAMCS 4. TRIAGE ########")
    tri = d[d.triage.notna()].copy()
    tri["triage_semi_or_nonurgent"] = (tri.triage >= 4).astype(int)
    results.append(analyse("NHAMCS triage, all adults", tri,
                           ["triage_urgent", "triage_semi_or_nonurgent", "long_wait"], NHAMCS_COVARS,
                           ["pain", "injury"]))
    results.append(analyse("NHAMCS triage, abnormal vital signs", tri[tri.abnormal_vitals == 1],
                           ["triage_urgent", "long_wait"], NHAMCS_COVARS, ["pain", "injury"],
                           drop=["abnormal_vitals"]))
    results.append(analyse("NHAMCS triage, patients later admitted", tri[tri.admitted == 1],
                           ["triage_urgent", "long_wait"], NHAMCS_COVARS, ["pain", "injury"]))
    young = tri[tri.age < 50]
    results.append(analyse("NHAMCS triage, adults under 50", young,
                           ["triage_urgent", "long_wait"], NHAMCS_COVARS, ["pain", "injury"]))
    return results


def main(which):
    out = []
    if "namcs" in which:
        out += namcs_checks(os.environ.get("NAMCS_PATH", "data/namcs2019.pkl"))
    if "nhamcs" in which:
        out += nhamcs_checks(os.environ.get("NHAMCS_PATH", "data/nhamcs2022.pkl"))
    pd.concat(out, ignore_index=True).round(2).to_csv(f"audit_by_race_{'_'.join(which)}.csv", index=False)
    print(f"\nSaved audit_by_race_{'_'.join(which)}.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or ["namcs", "nhamcs"])
