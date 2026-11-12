"""
NAMCS 2019 sex-gap audit — filtering script (matched to the real 2019 PUF columns)
---------------------------------------------------------------------------------
Usage (from your project folder):
    python namcs_filter.py data/namcs2019.pkl
    python namcs_filter.py data/namcs2019_sas.sas7bdat
 
What it does:
  1. Keeps visits where a chosen symptom is any of the 5 reasons for visit.
  2. Flags tests ordered (yes/no checkbox columns in the 2019 file),
     diagnoses given (ICD-10 prefixes), "left with only a symptom code",
     and what happened next (disposition columns).
  3. Compares female vs male: weighted % (PATWT) and raw counts.
     NCHS rule: estimates based on < 30 sample records are unreliable.
  4. Saves the filtered cohort + comparison table as CSV.
 
Codes still marked VERIFY: the reason-for-visit symptom codes. The script
prints how many visits each code matches so you can sanity-check them.
"""
 
import sys
import pandas as pd
 
# ---------------------------------------------------------------- settings --
# Reason-for-visit (RVC) codes as stored in RFV1-RFV5 (e.g. 14150.0).
SYMPTOMS = {
    "shortness_of_breath": [14150],          # S415.0  VERIFY in codebook
    "chest_pain":          [10500, 10501],   # S050.x  VERIFY in codebook
}
COHORT = "shortness_of_breath"               # which symptom to study
 
# Tests ordered: yes/no columns in the 2019 file (1 = yes).
TESTS = {
    "spirometry":    "SPIRO",
    "peak_flow":     "PEAK",
    "xray":          "XRAY",
    "ct_scan":       "CATSCAN",
    "ecg":           "EKG",
    "echo":          "ECHOCARD",
    "cardiac_stress": "CARDIAC",
    "any_imaging":   "ANYIMAGE",
    "cbc":           "CBC",
}
 
# What happened at the end of the visit (1 = yes).
DISPOSITIONS = {
    "sent_to_ED_or_admitted": "ERADMHOS",
    "referred_other_doctor":  "REFOTHMD",
    "return_to_referring":    "RETREFPHY",
    "return_appt":            ["RETAPPT1", "RETAPPT2", "RETAPPT3", "RETUNSP"],
    "return_as_needed":       "RETNEED",
    "no_follow_up_planned":   "NODISP",
}
 
# Diagnoses: start of DIAG1-DIAG5 (ICD-10-CM, no decimal, e.g. 'J459').
DIAGNOSES = {
    "asthma":         ["J45"],
    "copd":           ["J44", "J43"],
    "anxiety":        ["F41"],
    "heart_failure":  ["I50"],
    "ischemic_heart": ["I20", "I21", "I24", "I25"],
}
 
RFV_COLS  = [f"RFV{i}" for i in range(1, 6)]
DIAG_COLS = [f"DIAG{i}" for i in range(1, 6)]
SEX_LABELS = {1: "Female", 2: "Male"}        # confirmed: 4,609 female visits
 
 
# ------------------------------------------------------------------ load ---
def load(path):
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    elif path.endswith(".sas7bdat"):
        df = pd.read_sas(path, encoding="latin-1")
    elif path.endswith(".dta"):
        df = pd.read_stata(path, convert_categoricals=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    print(f"Loaded {len(df):,} visits, {df.shape[1]} columns.")
    return df
 
 
# --------------------------------------------------------------- helpers ---
def yes(df, cols):
    """True where any of the given columns == 1."""
    cols = [cols] if isinstance(cols, str) else cols
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").eq(1).any(axis=1)
 
 
def clean_diag(s):
    """'I10-' -> 'I10', 'J459' stays, blanks become ''."""
    return (s.astype(str).str.strip().str.upper()
             .str.replace("-", "", regex=False).str.replace(".", "", regex=False))
 
 
# -------------------------------------------------------------- pipeline ---
def check_symptom_codes(df):
    rfv = df[RFV_COLS].apply(pd.to_numeric, errors="coerce")
    print("\nVisits matching each symptom code (sanity check):")
    for name, codes in SYMPTOMS.items():
        for c in codes:
            print(f"  {name:22s} {c}: {int(rfv.eq(c).any(axis=1).sum()):,} visits")
 
 
def build_cohort(df):
    rfv = df[RFV_COLS].apply(pd.to_numeric, errors="coerce")
    has_symptom = rfv.isin(SYMPTOMS[COHORT]).any(axis=1)
    cohort = df[has_symptom].copy()
    cohort["symptom_is_main_reason"] = rfv.loc[has_symptom, "RFV1"].isin(SYMPTOMS[COHORT])
    cohort["sex"] = pd.to_numeric(cohort["SEX"], errors="coerce").map(SEX_LABELS)
 
    for name, col in TESTS.items():
        cohort[f"test_{name}"] = yes(cohort, col)
    for name, col in DISPOSITIONS.items():
        cohort[f"next_{name}"] = yes(cohort, col)
 
    diag = cohort[DIAG_COLS].apply(clean_diag)
    for name, prefixes in DIAGNOSES.items():
        pat = "^(" + "|".join(prefixes) + ")"
        cohort[f"dx_{name}"] = diag.apply(lambda s: s.str.match(pat)).any(axis=1)
 
    # Left with only a symptom code: first diagnosis is an R-code and no
    # other listed diagnosis is a real (non-R) ICD-10 code.
    real_dx = diag.apply(lambda s: s.str.match(r"^[A-QS-Z]\d"))
    cohort["symptom_code_only"] = diag["DIAG1"].str.startswith("R") & ~real_dx.any(axis=1)
    return cohort
 
 
def compare_by_sex(cohort):
    outcomes = [c for c in cohort.columns
                if c.startswith(("test_", "next_", "dx_"))] + ["symptom_code_only"]
    w = pd.to_numeric(cohort["PATWT"], errors="coerce").fillna(0)
 
    rows = []
    for out in outcomes:
        row = {"outcome": out}
        for sex in ("Female", "Male"):
            m = cohort["sex"].eq(sex)
            n_yes = int(cohort.loc[m, out].sum())
            row[f"{sex}_n"] = int(m.sum())
            row[f"{sex}_yes"] = n_yes
            row[f"{sex}_wtd_%"] = round(100 * (w[m] * cohort.loc[m, out]).sum() / w[m].sum(), 1)
        row["gap_F-M"] = round(row["Female_wtd_%"] - row["Male_wtd_%"], 1)
        row["reliable"] = min(row["Female_yes"], row["Male_yes"]) >= 30
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def main(path):
    df = load(path)
    check_symptom_codes(df)
    cohort = build_cohort(df)
    print(f"\nCohort '{COHORT}': {len(cohort):,} visits "
          f"{cohort['sex'].value_counts().to_dict()}")
 
    table = compare_by_sex(cohort)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))
 
    cohort.to_csv(f"cohort_{COHORT}.csv", index=False)
    table.to_csv(f"sex_comparison_{COHORT}.csv", index=False)
    print(f"\nSaved cohort_{COHORT}.csv and sex_comparison_{COHORT}.csv")
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python namcs_filter.py <namcs2019 .pkl / .sas7bdat>")
    main(sys.argv[1])
 
