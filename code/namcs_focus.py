"""
Palpitations and diabetes sex-gap checks on NAMCS 2019 (office visits).

Usage:
    .venv/Scripts/python.exe namcs_focus.py data/namcs2019.pkl              # both
    .venv/Scripts/python.exe namcs_focus.py data/namcs2019.pkl diabetes     # one

Why NAMCS for these two:
  - Diabetes is diagnosed and managed in offices, not EDs, and NAMCS 2019 records
    the patient's most recent HbA1c and fasting glucose results (past 12 months).
    That lets us compare the chart ("has diabetes?") against the lab values.
  - Palpitations / anxiety: office visits show the outpatient workup (ECG, echo,
    stress test) and referrals, not just ED rule-outs.

Codings: SEX 1=Female 2=Male. Lab fields (inferred from the data -- the 2019
codebook PDF is no longer online, verify if you get a copy):
  A1C / FBG    1 = result from past 12 months recorded, 2 = none, -7 = lab
               section not applicable for this visit, -9 = blank
  A1CRES       HbA1c, %        FBGRES  fasting glucose, mg/dL
Medications in NAMCS are every drug listed at the visit (new + continued).

Same output format as focus_analyses.py: raw weighted % women vs men, then the
odds ratio for being female from a weighted logistic regression (OR < 1 =
women less likely). SEs clustered on CPSUM -- approximate.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import nhamcs_filter as nf
import focus_analyses as fa

warnings.filterwarnings("ignore")

# Covariates available in NAMCS (no vitals-based acuity, no region in the PUF)
fa.BASE_COVARS = ["age", "C(race)", "C(payer)", "C(msa)", "C(specialty)", "seen_before",
                  "n_chronic", "cad", "chf", "copd", "diabetes", "htn", "depression"]

DM_DX = ["E08", "E09", "E10", "E11", "E13"]
STATINS = ["d00280", "d00348", "d00746", "d04105", "d04851", "d07637"]


def meds_listed(d, cats=(), drug_ids=()):
    hit = pd.Series(False, index=d.index)
    for i in range(1, 31):
        if f"DRUGID{i}" not in d.columns:
            break
        if drug_ids:
            hit |= d[f"DRUGID{i}"].isin(drug_ids)
        if cats:
            cols = [f"RX{i}V{j}C{k}" for j in (1, 2, 3) for k in range(1, 5) if f"RX{i}V{j}C{k}" in d.columns]
            hit |= d[cols].astype(str).isin(cats).any(axis=1)
    return hit


def prep(df):
    d = df[nf.num(df, "AGE") >= 18].copy()
    num = lambda c: nf.num(d, c)
    d["female"] = num("SEX").eq(1).astype(int)
    d["sex"] = num("SEX").map(nf.SEX_LABELS)
    d["w"] = num("PATWT").fillna(0)
    d["age"] = num("AGE")
    d["age_band"] = pd.cut(d["age"], [17, 34, 49, 64, 120], labels=["18-34", "35-49", "50-64", "65+"])
    d["race"] = num("RACERETH").map({1: "white", 2: "black", 3: "hispanic", 4: "other"})
    d["payer"] = num("PAYTYPER").fillna(-9).astype(int).astype(str)
    d["msa"] = num("MSA").fillna(-9).astype(int).astype(str)
    d["specialty"] = num("SPECCAT").map({1: "primary", 2: "surgical", 3: "medical"})
    d["seen_before"] = num("SENBEFOR").eq(1).astype(int)
    d["n_chronic"] = num("TOTCHRON").clip(lower=0).fillna(0)
    for c in ["CAD", "CHF", "COPD", "HTN", "OBESITY", "HYPLIPID"]:
        d[c.lower()] = num(c).eq(1).astype(int)
    d["depression"] = num("DEPRN").eq(1).astype(int)
    bmi = num("BMI").where(num("BMI") > 0)
    d["bmi"] = bmi
    d["overweight"] = (bmi >= 25).where(bmi.notna())

    dm_chart = nf.yes(d, ["DIABTYP1", "DIABTYP2", "DIABTYP0"])
    d["dm_dx_today"] = fa.has_dx(d, DM_DX)
    d["diabetes"] = (dm_chart | d["dm_dx_today"]).astype(int)

    # Labs
    d["lab_section"] = num("A1C").isin([1, 2]) | num("FBG").isin([1, 2])
    a1c = num("A1CRES").where(num("A1CRES").between(3, 20))
    fbg = num("FBGRES").where(num("FBGRES") > 0)
    d["a1c"], d["fbg"] = a1c, fbg
    d["has_a1c_result"] = num("A1C").eq(1).astype(int)
    d["has_glucose_or_a1c"] = (num("A1C").eq(1) | num("FBG").eq(1)).astype(int)
    d["lab_diabetes_range"] = ((a1c >= 6.5) | (fbg >= 126)).astype(int)
    d["lab_prediabetes_range"] = (((a1c >= 5.7) & (a1c < 6.5)) | ((fbg >= 100) & (fbg < 126))).astype(int)
    d["a1c_ordered_today"] = nf.yes(d, ["HGBA", "GLUCOSE"]).astype(int)
    d["screened"] = (d["has_glucose_or_a1c"].astype(bool) | d["a1c_ordered_today"].astype(bool)).astype(int)

    # Diabetes care
    d["foot_exam"] = nf.yes(d, "FOOT").astype(int)
    d["retinal_exam"] = nf.yes(d, "RETINAL").astype(int)
    d["diabetes_education"] = nf.yes(d, "DIAEDUC").astype(int)
    d["diet_counselling"] = nf.yes(d, "DIETNUTR").astype(int)
    d["statin"] = meds_listed(d, drug_ids=STATINS).astype(int)
    d["diabetes_med"] = meds_listed(d, cats=["099"]).astype(int)
    d["a1c_under_7"] = (a1c < 7).where(a1c.notna())

    # Heart-type workup
    d["ecg"] = nf.yes(d, "EKG").astype(int)
    d["echo"] = nf.yes(d, "ECHOCARD").astype(int)
    d["stress_test"] = nf.yes(d, "CARDIAC").astype(int)
    d["any_cardiac_test"] = d[["ecg", "echo", "stress_test"]].max(axis=1)
    d["thyroid_test"] = nf.yes(d, "THYROID").astype(int)
    d["mental_health_referral"] = nf.yes(d, ["MENTAL", "PSYCHOTH", "STRESMGT"]).astype(int)
    d["referred_other_md"] = nf.yes(d, "REFOTHMD").astype(int)
    d["anxiolytic_listed"] = meds_listed(d, cats=["067"]).astype(int)
    d["dx_anxiety"] = fa.has_dx(d, ["F41"]).astype(int)
    d["dx_arrhythmia"] = fa.has_dx(d, ["I47", "I48", "I49"]).astype(int)
    d["dx_any_cardiac"] = fa.has_dx(d, ["I2", "I3", "I4", "I50", "I51"]).astype(int)
    diag = d[nf.DIAG_COLS].apply(nf.clean_diag)
    real = diag.apply(lambda s: s.str.match(r"^[A-QS-Z]\d"))
    d["dx_symptom_code_only"] = (diag["DIAG1"].str.startswith("R") & ~real.any(axis=1)).astype(int)
    return d


# --------------------------------------------------------------- checks ---
def check_diabetes(d):
    t, a = [], []
    labs = d[d.lab_section]
    print(f"Adults with the lab section filled in: {len(labs):,} of {len(d):,} adult visits")

    # 1. Chart vs labs: lab values in the diabetes range but no diabetes in the chart
    hi = labs[labs.lab_diabetes_range == 1].assign(not_in_chart=lambda x: 1 - x.diabetes)
    t.append(fa.raw(hi, "not_in_chart", None, "labs in diabetes range, NOT recorded as diabetes"))
    pre = labs[(labs.lab_prediabetes_range == 1) & (labs.diabetes == 0)]
    pre = pre.assign(prediabetes_dx=fa.has_dx(pre, ["R730", "R739"]).astype(int))
    t.append(fa.raw(pre, "prediabetes_dx", None, "labs in prediabetes range: prediabetes coded today"))

    # 2. Screening: adults 35-70 who are overweight and not known to have diabetes
    #    (USPSTF 2021 screening group). Was there any HbA1c / glucose in the past year?
    elig = labs[(labs.age.between(35, 70)) & (labs.overweight == 1) & (labs.diabetes == 0)]
    t += [fa.raw(elig, "screened", None, "screening group (35-70, BMI>=25, no DM): A1c/glucose in past yr"),
          fa.raw(elig, "screened", "race", "screening group: screened, by race"),
          fa.raw(elig, "screened", "age_band", "screening group: screened, by age")]
    a.append(fa.adjusted(elig, "screened", ["bmi"], "screening group: screened", drop=["diabetes"]))
    # Same question without the BMI rule (some groups develop diabetes at lower BMI)
    elig2 = labs[(labs.age.between(35, 70)) & (labs.diabetes == 0)]
    t.append(fa.raw(elig2, "screened", None, "all 35-70 without DM: A1c/glucose in past yr"))
    a.append(fa.adjusted(elig2, "screened", [], "35-70 without DM: screened", drop=["diabetes"]))

    # 3. Care for people who do have diabetes
    dm = d[d.diabetes == 1]
    for c in ["has_a1c_result", "a1c_ordered_today", "foot_exam", "retinal_exam",
              "diabetes_education", "diet_counselling", "statin", "diabetes_med"]:
        t.append(fa.raw(dm, c, None, f"has diabetes: {c}"))
    t.append(fa.raw(dm, "a1c_under_7", None, "has diabetes: last A1c < 7% (if result)"))
    for c in ["has_a1c_result", "foot_exam", "statin"]:
        a.append(fa.adjusted(dm, c, [], f"has diabetes: {c}", drop=["diabetes"]))
    return t, a


HEART_SX = {"palpitations": [1260], "chest pain": [1050], "short of breath": [1415],
            "dizziness": [1225], "fatigue": [1015], "fainting": [1030]}


def check_palpitations(d):
    x = d[nf.num(d, "INJURY") != 1].copy()
    p = x[fa.has_rfv(x, [1260])]
    print(f"Palpitations visits: {len(p)} ({int(p.female.sum())} women, {int((1 - p.female).sum())} men)")
    outs = ["ecg", "echo", "stress_test", "any_cardiac_test", "thyroid_test", "dx_arrhythmia",
            "dx_anxiety", "dx_symptom_code_only", "anxiolytic_listed", "mental_health_referral",
            "referred_other_md"]
    t = [fa.raw(p, c, None, f"palpitations: {c}") for c in outs]

    h = x[fa.has_rfv(x, sum(HEART_SX.values(), []))].copy()
    h["symptom"] = "other"
    for name, codes in reversed(list(HEART_SX.items())):
        h.loc[fa.has_rfv(h, codes), "symptom"] = name
    print(f"Any heart-type symptom: {len(h)} ({int(h.female.sum())} women, {int((1 - h.female).sum())} men)")
    t += [fa.raw(h, c, None, f"heart-type sx: {c}") for c in outs]
    t.append(fa.raw(h, "dx_anxiety", "symptom", "heart-type sx: anxiety dx, by symptom"))
    t.append(fa.raw(h, "any_cardiac_test", "symptom", "heart-type sx: cardiac test, by symptom"))
    a = [fa.adjusted(h, c, ["C(symptom)"], f"heart-type sx: {c}")
         for c in ["any_cardiac_test", "ecg", "dx_anxiety", "anxiolytic_listed", "dx_symptom_code_only"]]
    return t, a


CHECKS = {"diabetes": ("DIABETES (NAMCS 2019 office visits)", check_diabetes),
          "palp": ("PALPITATIONS / HEART-TYPE SYMPTOMS (NAMCS 2019 office visits)", check_palpitations)}


def main(path, which):
    d = prep(nf.load(path))
    print(f"Adults: {len(d):,} visits ({int(d.female.sum()):,} women, {int((1 - d.female).sum()):,} men)")
    for key in which:
        title, fn = CHECKS[key]
        print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
        t, a = fn(d)
        fa.show(title, t, a)
        fa.save(f"namcs_{key}", t, a)
    print("\nSaved focus_namcs_<check>_raw.csv and focus_namcs_<check>_adjusted.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python namcs_focus.py <namcs2019.pkl> [diabetes|palp]")
    main(sys.argv[1], sys.argv[2:] or list(CHECKS))
