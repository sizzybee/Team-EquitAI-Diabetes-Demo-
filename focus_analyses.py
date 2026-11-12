"""
Five focused sex-gap checks on NHAMCS 2022 (adult ED visits).

Usage:
    .venv/Scripts/python.exe focus_analyses.py data/nhamcs2022.pkl            # all
    .venv/Scripts/python.exe focus_analyses.py data/nhamcs2022.pkl pain       # one

  pain       Same pain score -> same pain relief?
  triage     Same vitals / same eventual outcome -> same triage urgency?
  diabetes   What the ED data can (and can't) say about diabetes
  palp       Palpitations: arrhythmia vs anxiety diagnosis
  brain      Brain imaging (CT head / MRI) for neurological symptoms

Every check prints:
  raw    survey-weighted % for women vs men (PATWT), with record counts
  adj    odds ratio for being female from a weighted logistic regression that
         holds age, race, payer, chronic conditions, vitals, pain, ambulance
         arrival, and region constant. OR < 1 = women less likely.
         Standard errors are clustered on the survey PSU (CPSUM) -- this is an
         approximation of the full NCHS design, good enough to rank findings,
         not to publish.

All codings were checked against the 2022 NHAMCS public-use documentation.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import nhamcs_filter as nf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

MIN_AGE = 18


# ------------------------------------------------------------------ prep ---
def prep(df):
    """Adult ED visits with the covariates every model uses."""
    d = df[nf.num(df, "AGE") >= MIN_AGE].copy()
    num = lambda c: nf.num(d, c)
    d["female"] = num("SEX").eq(1).astype(int)
    d["sex"] = num("SEX").map(nf.SEX_LABELS)
    d["w"] = num("PATWT").fillna(0)
    d["age"] = num("AGE")
    d["age_band"] = pd.cut(d["age"], [17, 34, 49, 64, 120], labels=["18-34", "35-49", "50-64", "65+"])

    # Vitals (negative codes = missing). TEMPF is stored as tenths of a degree F.
    temp = num("TEMPF").where(num("TEMPF") > 0)
    temp = np.where(temp > 200, temp / 10, temp)
    hr, sbp, rr, spo2 = (num(c).where(num(c) > 0) for c in ["PULSE", "BPSYS", "RESPR", "POPCT"])
    d["abnormal_vitals"] = ((hr > 100) | (sbp < 90) | (rr > 22) | (spo2 < 94)
                            | (pd.Series(temp, index=d.index) >= 100.4)).astype(int)
    d["hr"], d["sbp"], d["spo2"] = hr, sbp, spo2

    pain = num("PAINSCALE").where(num("PAINSCALE").between(0, 10))
    d["pain"] = pain
    d["pain_band"] = pd.cut(pain, [-1, 3, 6, 10], labels=["0-3", "4-6", "7-10"])
    imm = num("IMMEDR").where(num("IMMEDR").between(1, 5))
    d["triage"] = imm
    d["triage_urgent"] = (imm <= 2).where(imm.notna())   # immediate or emergent

    d["ems"] = num("ARREMS").eq(1).astype(int)
    d["injury"] = num("INJURY").eq(1).astype(int)
    d["diabetes"] = nf.yes(d, ["DIABTYP1", "DIABTYP2", "DIABTYP0"]).astype(int)
    for c in ["CAD", "CHF", "COPD", "HTN", "ASTHMA", "CKD", "DEPRN", "OBESITY"]:
        d[c.lower()] = num(c).eq(1).astype(int)
    d["n_chronic"] = num("TOTCHRON").clip(lower=0).fillna(0)
    d["race"] = num("RACERETH").map({1: "white", 2: "black", 3: "hispanic", 4: "other"})
    d["payer"] = num("PAYTYPER").fillna(-9).astype(int).astype(str)
    d["region"] = num("REGION").astype(int).astype(str)
    d["left_early"] = nf.yes(d, ["LWBS", "LBTC", "LEFTAMA", "DOA"]).astype(int)
    d["admitted"] = nf.yes(d, ["ADMITHOS", "OBSHOS", "TRANOTH"]).astype(int)
    d["wait"] = num("WAITTIME").where(num("WAITTIME") >= 0)

    for k, cats in nf.MED_CATS.items():
        d[f"med_{k}"] = nf.meds_given(d, cats).astype(int)
    d["med_pain_relief"] = d[[f"med_{k}" for k in nf.PAIN_RELIEF]].any(axis=1).astype(int)
    d["brain_imaging"] = nf.yes(d, ["CTHEAD", "MRI"]).astype(int)
    d["ct_head"] = nf.yes(d, "CTHEAD").astype(int)
    d["mri"] = nf.yes(d, "MRI").astype(int)
    rfv = d[nf.RFV3_COLS].apply(pd.to_numeric, errors="coerce")
    d["_rfv"] = list(rfv.values)
    d["diag"] = list(d[nf.DIAG_COLS].apply(nf.clean_diag).values)
    return d


def has_rfv(d, codes):
    rfv = d[nf.RFV3_COLS].apply(pd.to_numeric, errors="coerce")
    return rfv.isin(codes).any(axis=1)


def has_dx(d, prefixes):
    diag = d[nf.DIAG_COLS].apply(nf.clean_diag)
    pat = "^(" + "|".join(prefixes) + ")"
    return diag.apply(lambda s: s.str.match(pat)).any(axis=1)


# ------------------------------------------------------------- reporting ---
def raw(d, outcome, by=None, label=None):
    """Weighted % women vs men, optionally within strata of `by`."""
    groups = [("all", d)] if by is None else [(str(g), x) for g, x in d.groupby(by, observed=True)]
    rows = []
    for g, x in groups:
        x = x[x[outcome].notna()]
        row = {"check": label or outcome, "stratum": g}
        for s in ("Female", "Male"):
            m = x["sex"].eq(s)
            yes = x.loc[m, outcome].astype(float)
            row[f"{s[0]}_n"] = int(m.sum())
            row[f"{s[0]}_yes"] = int(yes.sum())
            row[f"{s[0]}_%"] = round(100 * (x.loc[m, "w"] * yes).sum() / max(x.loc[m, "w"].sum(), 1e-9), 1)
        row["gap"] = round(row["F_%"] - row["M_%"], 1)
        row["ok_n"] = min(row["F_yes"], row["M_yes"]) >= 30
        rows.append(row)
    return pd.DataFrame(rows)


BASE_COVARS = ["age", "C(race)", "C(payer)", "C(region)", "n_chronic", "cad", "chf", "copd",
               "diabetes", "htn", "ems", "abnormal_vitals"]


def adjusted(d, outcome, extra=(), label=None, drop=()):
    """Weighted logistic regression; returns OR for female with 95% CI."""
    covars = [c for c in BASE_COVARS + list(extra) if c not in drop]
    if "YEAR" in d.columns and d["YEAR"].nunique() > 1:
        covars = covars + ["C(YEAR)"]   # pooled years: year fixed effect
    x = d.dropna(subset=[outcome] + [c.replace("C(", "").replace(")", "") for c in covars]).copy()
    x[outcome] = x[outcome].astype(float)
    if x[outcome].sum() < 30 or x[outcome].nunique() < 2:
        return {"check": label or outcome, "n": len(x), "OR_female": np.nan, "note": "too few events"}
    formula = f"{outcome} ~ female + " + " + ".join(covars)
    try:
        m = sm.GLM.from_formula(formula, x, family=sm.families.Binomial(),
                                var_weights=x["w"] / x["w"].mean())
        r = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["CPSUM"])[0]})
        b, se = r.params["female"], r.bse["female"]
        return {"check": label or outcome, "n": len(x),
                "OR_female": round(np.exp(b), 2),
                "CI_low": round(np.exp(b - 1.96 * se), 2), "CI_high": round(np.exp(b + 1.96 * se), 2),
                "p": round(r.pvalues["female"], 3)}
    except Exception as e:  # small strata can fail to converge
        return {"check": label or outcome, "n": len(x), "OR_female": np.nan, "note": str(e)[:60]}


def show(title, raw_tables, adj_rows):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    if raw_tables:
        print(pd.concat(raw_tables, ignore_index=True).to_string(index=False))
    if adj_rows:
        print("\nAdjusted (OR < 1 = women less likely, all else equal):")
        print(pd.DataFrame(adj_rows).to_string(index=False))


def save(name, raw_tables, adj_rows):
    out = pd.concat(raw_tables, ignore_index=True) if raw_tables else pd.DataFrame()
    out.to_csv(f"focus_{name}_raw.csv", index=False)
    pd.DataFrame(adj_rows).to_csv(f"focus_{name}_adjusted.csv", index=False)


# --------------------------------------------------------------- checks ---
def check_pain(d):
    x = d[(d.left_early == 0) & d.pain.notna()]
    sev = x[x.pain >= 7]
    tables = [
        raw(x, "med_pain_relief", "pain_band", "any pain relief, by pain score"),
        raw(x, "med_opioid", "pain_band", "opioid, by pain score"),
        raw(sev, "med_pain_relief", "age_band", "pain 7-10: any relief, by age"),
    ]
    # Severe pain within the most common complaints
    for name, codes in {"abdominal pain": [1545], "back pain": [1905, 1910], "headache": [1210],
                        "chest pain": [1050], "limb/joint pain": [1915, 1920, 1925, 1930, 1935, 1940, 1945]}.items():
        tables.append(raw(sev[has_rfv(sev, codes)], "med_opioid", None, f"pain 7-10 + {name}: opioid"))
    adj = [
        adjusted(x[x.pain >= 4], "med_pain_relief", ["pain", "injury"], "pain 4-10: any relief"),
        adjusted(x[x.pain >= 4], "med_opioid", ["pain", "injury"], "pain 4-10: opioid"),
        adjusted(sev, "med_opioid", ["pain", "injury"], "pain 7-10: opioid"),
    ]
    return tables, adj


def check_triage(d):
    x = d[d.triage.notna()]
    tables = [
        raw(x, "triage_urgent", None, "triaged immediate/emergent"),
        raw(x, "triage_urgent", "abnormal_vitals", "triaged urgent, by abnormal vitals (0/1)"),
        raw(x, "triage_urgent", "age_band", "triaged urgent, by age"),
        raw(x[x.admitted == 1], "triage_urgent", None, "triaged urgent | later admitted"),
        raw(x[x.admitted == 1], "triage_urgent", "age_band", "triaged urgent | admitted, by age"),
    ]
    x = x.assign(long_wait=(x.wait > 30).where(x.wait.notna()))
    tables.append(raw(x[x.admitted == 1], "long_wait", None, "waited > 30 min | later admitted"))
    adj = [
        adjusted(x, "triage_urgent", ["pain", "injury"], "triaged urgent (all adults)"),
        adjusted(x[x.admitted == 1], "triage_urgent", ["pain", "injury"], "triaged urgent | admitted"),
        adjusted(x[x.abnormal_vitals == 1], "triage_urgent", ["pain", "injury"],
                 "triaged urgent | abnormal vitals", drop=["abnormal_vitals"]),
    ]
    return tables, adj


def check_diabetes(d):
    x = d[d.left_early == 0].copy()
    x["glucose_test"] = nf.yes(x, ["GLUCOSE", "BMP", "CMP"]).astype(int)
    x["ecg"] = nf.yes(x, "EKG").astype(int)
    x["troponin"] = nf.yes(x, "CARDENZ").astype(int)
    tables = [raw(x, "diabetes", "race", "diabetes recorded in chart, by race")]
    dm = x[x.diabetes == 1]
    tables.append(raw(dm, "admitted", None, "diabetes: admitted"))
    # Diabetic patients often have 'atypical' heart attack symptoms. Do diabetic women with
    # chest pain, breathlessness, fatigue or weakness get the same cardiac workup?
    vague = dm[has_rfv(dm, [1050, 1415, 1015, 1020])]
    tables += [raw(vague, "ecg", None, "diabetes + chest pain/SOB/fatigue/weakness: ECG"),
               raw(vague, "troponin", None, "diabetes + chest pain/SOB/fatigue/weakness: troponin")]
    tables.append(raw(x[has_rfv(x, [1015, 1020])], "troponin", None, "any adult fatigue/weakness: troponin"))
    adj = [adjusted(vague, "troponin", [], "diabetes + vague cardiac sx: troponin", drop=["diabetes"]),
           adjusted(vague, "ecg", [], "diabetes + vague cardiac sx: ECG", drop=["diabetes"])]
    return tables, adj


def check_palpitations(d):
    x = d[(d.left_early == 0) & (d.injury == 0)].copy()
    x["dx_anxiety"] = has_dx(x, ["F41"]).astype(int)
    x["dx_arrhythmia"] = has_dx(x, ["I47", "I48", "I49"]).astype(int)
    x["dx_symptom_only"] = has_dx(x, ["R00"]).astype(int)
    x["ecg"] = nf.yes(x, "EKG").astype(int)
    x["monitor"] = nf.yes(x, "CARDMON").astype(int)
    x["troponin"] = nf.yes(x, "CARDENZ").astype(int)
    p = x[has_rfv(x, [1260])]
    tables = [raw(p, c, None, f"palpitations: {c}")
              for c in ["ecg", "monitor", "troponin", "dx_arrhythmia", "dx_anxiety",
                        "dx_symptom_only", "med_anxiolytic", "admitted"]]
    # Anxiety label across all 'heart-type' complaints, where it can mask a cardiac cause
    cardiac_sx = x[has_rfv(x, [1050, 1260, 1415, 1225, 1030])].copy()
    cardiac_sx["symptom"] = np.select(
        [has_rfv(cardiac_sx, [1260]), has_rfv(cardiac_sx, [1050]), has_rfv(cardiac_sx, [1415]),
         has_rfv(cardiac_sx, [1225]), has_rfv(cardiac_sx, [1030])],
        ["palpitations", "chest pain", "short of breath", "dizziness", "fainting"], default="other")
    tables.append(raw(cardiac_sx, "dx_anxiety", "symptom", "anxiety diagnosis, by heart-type symptom"))
    tables.append(raw(cardiac_sx, "med_anxiolytic", None, "heart-type symptom: anxiolytic given"))
    adj = [adjusted(cardiac_sx, "dx_anxiety", ["C(symptom)", "deprn"], "heart-type sx: anxiety dx"),
           adjusted(cardiac_sx, "dx_arrhythmia", ["C(symptom)"], "heart-type sx: arrhythmia dx"),
           adjusted(p, "monitor", [], "palpitations: cardiac monitor")]
    return tables, adj


NEURO = {"headache": [1210], "dizziness": [1225], "numbness/tingling": [1220],
         "neuro weakness": [1230], "speech": [1235], "vision": [1305],
         "fainting": [1030], "seizure": [1205], "confusion/memory": [1215]}


def check_brain(d):
    x = d[(d.left_early == 0) & (d.injury == 0)].copy()
    allcodes = sum(NEURO.values(), [])
    n = x[has_rfv(x, allcodes)].copy()
    n["symptom"] = "other"
    for name, codes in reversed(list(NEURO.items())):   # first listed RFV wins below
        n.loc[has_rfv(n, codes), "symptom"] = name
    n["main"] = pd.to_numeric(n["RFV13D"], errors="coerce")
    for name, codes in NEURO.items():                 # prefer the main reason for visit
        n.loc[n.main.isin(codes), "symptom"] = name
    n["stroke_symptom"] = has_rfv(n, [1220, 1230, 1235, 1305]).astype(int)  # FAST-type
    tables = [
        raw(n, "brain_imaging", None, "neuro symptoms: CT head or MRI"),
        raw(n, "brain_imaging", "symptom", "brain imaging, by symptom"),
        raw(n, "brain_imaging", "age_band", "brain imaging, by age"),
        raw(n, "brain_imaging", "triage", "brain imaging, by triage level"),
        raw(n, "mri", None, "neuro symptoms: MRI"),
        raw(n[n.stroke_symptom == 1], "brain_imaging", None, "stroke-type symptom: brain imaging"),
    ]
    adj = [adjusted(n, "brain_imaging", ["C(symptom)", "pain"], "neuro sx: brain imaging"),
           adjusted(n, "brain_imaging", ["C(symptom)", "pain", "C(triage)"],
                    "neuro sx: brain imaging (+triage)"),
           adjusted(n, "mri", ["C(symptom)"], "neuro sx: MRI"),
           adjusted(n[n.symptom == "headache"], "ct_head", ["pain"], "headache: CT head"),
           adjusted(n[n.symptom == "dizziness"], "brain_imaging", [], "dizziness: brain imaging")]
    return tables, adj


CHECKS = {"pain": ("1. PAIN: same pain score, same relief?", check_pain),
          "triage": ("2. TRIAGE: same presentation, same urgency?", check_triage),
          "diabetes": ("3. DIABETES (ED view)", check_diabetes),
          "palp": ("4. PALPITATIONS / ANXIETY LABEL", check_palpitations),
          "brain": ("5. BRAIN IMAGING for neurological symptoms", check_brain)}


def main(path, which):
    d = prep(nf.load(path))
    print(f"Adults: {len(d):,} visits ({int(d.female.sum()):,} women, {int((1 - d.female).sum()):,} men)")
    for key in which:
        title, fn = CHECKS[key]
        t, a = fn(d)
        show(title, t, a)
        save(key, t, a)
    print("\nSaved focus_<check>_raw.csv and focus_<check>_adjusted.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python focus_analyses.py <nhamcs2022.pkl> [pain|triage|diabetes|palp|brain]")
    main(sys.argv[1], sys.argv[2:] or list(CHECKS))
