"""
NHAMCS 2022 (ED) sex-gap audit -- filtering script
--------------------------------------------------
Usage (from the project folder):
    .venv/Scripts/python.exe nhamcs_filter.py data/nhamcs2022.pkl
    .venv/Scripts/python.exe nhamcs_filter.py data/nhamcs2022.pkl chest_pain

What it does:
  1. Cohort: adult ED visits where the chosen symptom is any of the 5 reasons
     for visit. Drops injury visits and visits where the patient left before
     the workup finished (left without being seen, left before treatment
     complete, left against advice, dead on arrival).
  2. Flags outcomes in four groups:
       triage_  how urgently they were triaged, how long they waited
       test_    tests / imaging ordered
       med_     medications given in the ED (pain relief, aspirin, nitro)
       next_    disposition (admitted, observation, sent home)
       dx_      what diagnosis they left with
  3. Compares female vs male: survey-weighted % (PATWT) + raw counts.
     NCHS rule: estimates based on < 30 sample records are unreliable.
  4. Pain relief is also compared within the same pain-score band, since
     "same pain, same treatment?" is a cleaner question than overall rates.
  5. Saves cohort + comparison tables as CSV.

These are RAW (unadjusted) gaps -- age, history, and vitals are not yet
accounted for. A gap here is a reason to look closer, not proof of bias.

Codings confirmed against the 2022 file:
  SEX 1=Female 2=Male | IMMEDR 1=immediate .. 5=nonurgent, 0=no triage,
  7=no nurse triage, -8/-9=missing | GPMED 1=given in ED, 2=Rx at discharge,
  3=both | RFVn3D = 3-digit reason-for-visit code (1050 = chest pain).
"""

import sys
import pandas as pd

# ---------------------------------------------------------------- settings --
# Symptom cohorts: 3-digit reason-for-visit codes (RFV13D..RFV53D) plus the
# diagnoses worth tracking for that symptom (ICD-10-CM prefixes, no decimal).
# Add more conditions here to compare them.
CARDIAC = {
    "acs_mi":        ["I20", "I21", "I22", "I23", "I24"],
    "any_cardiac":   ["I2", "I3", "I4", "I50", "I51"],  # heart disease, not hypertension
}
STROKE = {
    "stroke_tia":    ["I60", "I61", "I62", "I63", "G45"],
}
COHORTS = {
    "chest_pain": {"codes": [1050], "dx": {
        **CARDIAC,
        "pulmonary_embolism": ["I26"],
        "anxiety_panic":   ["F41"],
        "gerd_gi":         ["K21", "K20", "K29", "K30"],
        "musculoskeletal": ["M94", "R0789"],
    }},
    "shortness_of_breath": {"codes": [1415], "dx": {
        **CARDIAC,
        "heart_failure":   ["I50"],
        "pulmonary_embolism": ["I26"],
        "asthma":          ["J45"],
        "copd":            ["J44", "J43"],
        "pneumonia":       ["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
        "anxiety_panic":   ["F41"],
    }},
    "headache": {"codes": [1210], "dx": {
        **STROKE,
        "subarachnoid":    ["I60"],
        "migraine":        ["G43"],
        "tension_headache": ["G44"],
        "anxiety_panic":   ["F41"],
    }},
    "dizziness": {"codes": [1225], "dx": {
        **STROKE,
        "vertigo_inner_ear": ["H81", "H82"],
        "arrhythmia":      ["I47", "I48", "I49"],
        "syncope":         ["R55"],
        "anxiety_panic":   ["F41"],
    }},
    "palpitations": {"codes": [1260], "dx": {
        "arrhythmia":      ["I47", "I48", "I49"],
        "afib_flutter":    ["I48"],
        "any_cardiac":     ["I2", "I3", "I4", "I50", "I51"],
        "palpitations_code": ["R002"],
        "anxiety_panic":   ["F41"],
        "thyroid":         ["E05"],
    }},
    "neuro_any": {"codes": [1030, 1205, 1210, 1215, 1220, 1225, 1230, 1235, 1305], "dx": {
        **STROKE,
        "migraine":        ["G43"],
        "seizure":         ["G40", "R56"],
        "vertigo_inner_ear": ["H81", "H82"],
        "anxiety_panic":   ["F41"],
        "conversion_functional": ["F44", "F45"],
    }},
    "abdominal_pain": {"codes": [1545], "dx": {
        "appendicitis":    ["K35", "K36", "K37"],
        "gallbladder":     ["K80", "K81", "K82"],
        "ectopic_ovarian_gyn": ["O00", "N83", "N70", "N73"],
        "urinary_kidney":  ["N39", "N20", "N10"],
        "gastritis_ibs_constipation": ["K29", "K30", "K58", "K59"],
        "anxiety_panic":   ["F41"],
    }},
}
DEFAULT_COHORT = "chest_pain"
MIN_AGE = 18

# Tests ordered (1 = yes)
TESTS = {
    "ecg":            "EKG",
    "troponin":       "CARDENZ",   # cardiac enzymes
    "cardiac_monitor": "CARDMON",
    "bnp":            "BNP",
    "d_dimer":        "DDIMER",
    "xray":           "XRAY",
    "ct_chest":       "CTCHEST",
    "ct_head":        "CTHEAD",
    "ct_abdomen":     "CTAB",
    "mri":            "MRI",
    "ultrasound":     "ULTRASND",
    "any_imaging":    "ANYIMAGE",
    "cbc":            "CBC",
    "lipase_liver":   "LFT",
    "urinalysis":     "URINE",
    "pregnancy_test": "PREGTEST",
    "lumbar_puncture": "LUMBAR",
}

# Disposition (1 = yes)
DISPOSITIONS = {
    "admitted":        "ADMITHOS",
    "observation":     ["OBSHOS", "OBSDIS"],
    "transferred":     "TRANOTH",
    "no_follow_up":    "NOFU",
    "return_to_ed_prn": "RETRNED",
    "referred_fu":     "RETREFFU",
}

# Multum drug categories (RXnVjCk), for drugs given in the ED (GPMED 1 or 3)
MED_CATS = {   # confirmed against the 2022 codebook drug category list
    "aspirin":      ["062"],          # salicylates
    "nitro":        ["045"],          # antianginal agents
    "opioid":       ["060", "191"],   # narcotic analgesics (+ combinations)
    "nsaid":        ["061", "278"],   # NSAIDs, Cox-2 inhibitors
    "other_analgesic": ["059", "063"],  # misc analgesics (acetaminophen), combinations
    "anxiolytic":   ["067"],          # anxiolytics, sedatives, hypnotics (all subtypes)
}
PAIN_RELIEF = ["opioid", "nsaid", "other_analgesic"]   # aspirin excluded on purpose

RFV3_COLS = [f"RFV{i}3D" for i in range(1, 6)]
DIAG_COLS = [f"DIAG{i}" for i in range(1, 6)]
SEX_LABELS = {1: "Female", 2: "Male"}


# ------------------------------------------------------------------ load ---
def load(path):
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    elif path.endswith(".sas7bdat"):
        df = pd.read_sas(path, encoding="latin-1")
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    print(f"Loaded {len(df):,} visits, {df.shape[1]} columns.")
    return df


# --------------------------------------------------------------- helpers ---
def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def yes(df, cols):
    cols = [cols] if isinstance(cols, str) else cols
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").eq(1).any(axis=1)


def clean_diag(s):
    return (s.astype(str).str.strip().str.upper()
             .str.replace("-", "", regex=False).str.replace(".", "", regex=False))


def meds_given(df, cats):
    """True if any drug given in the ED belongs to one of the Multum categories."""
    hit = pd.Series(False, index=df.index)
    for i in range(1, 31):
        given = num(df, f"GPMED{i}").isin([1, 3])
        cols = [f"RX{i}V{j}C{k}" for j in (1, 2, 3) for k in range(1, 5)
                if f"RX{i}V{j}C{k}" in df.columns]
        in_cat = df[cols].astype(str).isin(cats).any(axis=1)
        hit |= given & in_cat
    return hit


# -------------------------------------------------------------- pipeline ---
def build_cohort(df, name):
    codes = COHORTS[name]["codes"]
    rfv = df[RFV3_COLS].apply(pd.to_numeric, errors="coerce")
    has_symptom = rfv.isin(codes).any(axis=1)
    steps = [("symptom in any reason for visit", has_symptom)]
    keep = has_symptom & (num(df, "AGE") >= MIN_AGE)
    steps.append((f"age >= {MIN_AGE}", keep))
    keep = keep & (num(df, "INJURY") != 1)
    steps.append(("not an injury visit", keep))
    left_early = yes(df, ["LWBS", "LBTC", "LEFTAMA", "DOA"])
    keep = keep & ~left_early
    steps.append(("stayed for full workup", keep))

    print(f"\nCohort '{name}' (RFV codes {codes}):")
    for label, mask in steps:
        print(f"  {label:34s} {int(mask.sum()):>6,}")

    c = df[keep].copy()
    c["sex"] = num(c, "SEX").map(SEX_LABELS)
    c["symptom_is_main_reason"] = num(c, "RFV13D").isin(codes)

    # Triage / timing (missing codes -> NaN so they drop out of the denominator)
    imm = num(c, "IMMEDR").where(num(c, "IMMEDR").between(1, 5))
    c["triage_level"] = imm
    c["triage_immediate_or_emergent"] = (imm <= 2).where(imm.notna())
    wait = num(c, "WAITTIME").where(num(c, "WAITTIME") >= 0)
    c["wait_minutes"] = wait
    c["triage_wait_over_30min"] = (wait > 30).where(wait.notna())
    pain = num(c, "PAINSCALE").where(num(c, "PAINSCALE").between(0, 10))
    c["pain_score"] = pain
    c["pain_band"] = pd.cut(pain, [-1, 3, 6, 10], labels=["0-3", "4-6", "7-10"])

    for k, col in TESTS.items():
        c[f"test_{k}"] = yes(c, col)

    for k, cats in MED_CATS.items():
        c[f"med_{k}"] = meds_given(c, cats)
    c["med_any_pain_relief"] = c[[f"med_{k}" for k in PAIN_RELIEF]].any(axis=1)

    for k, col in DISPOSITIONS.items():
        c[f"next_{k}"] = yes(c, col)

    diag = c[DIAG_COLS].apply(clean_diag)
    for k, prefixes in COHORTS[name]["dx"].items():
        pat = "^(" + "|".join(prefixes) + ")"
        c[f"dx_{k}"] = diag.apply(lambda s: s.str.match(pat)).any(axis=1)
    real_dx = diag.apply(lambda s: s.str.match(r"^[A-QS-Z]\d"))
    c["dx_symptom_code_only"] = diag["DIAG1"].str.startswith("R") & ~real_dx.any(axis=1)

    # Kept for the adjusted model later
    c["age"] = num(c, "AGE")
    c["arrived_by_ems"] = num(c, "ARREMS").eq(1)
    return c


def wpct(flag, w):
    ok = flag.notna()
    if w[ok].sum() == 0:
        return float("nan")
    return 100 * (w[ok] * flag[ok].astype(float)).sum() / w[ok].sum()


def compare_by_sex(c, outcomes, label=""):
    w = num(c, "PATWT").fillna(0)
    rows = []
    for out in outcomes:
        row = {"group": label, "outcome": out}
        for sex in ("Female", "Male"):
            m = c["sex"].eq(sex) & c[out].notna()
            row[f"{sex}_n"] = int(m.sum())
            row[f"{sex}_yes"] = int(c.loc[m, out].astype(bool).sum())
            row[f"{sex}_wtd_%"] = round(wpct(c.loc[m, out], w[m]), 1)
        row["gap_F-M"] = round(row["Female_wtd_%"] - row["Male_wtd_%"], 1)
        row["reliable"] = min(row["Female_yes"], row["Male_yes"]) >= 30
        rows.append(row)
    return pd.DataFrame(rows)


def describe_cohort(c):
    w = num(c, "PATWT").fillna(0)
    print("\nWho is in the cohort (weighted means / %):")
    for sex in ("Female", "Male"):
        m = c["sex"].eq(sex)
        print(f"  {sex:6s} n={int(m.sum()):4d}  age={c.loc[m,'age'].mean():.0f}  "
              f"CAD hx={wpct(c.loc[m,'CAD'].eq(1), w[m]):.0f}%  "
              f"diabetes={wpct(yes(c[m], ['DIABTYP1','DIABTYP2','DIABTYP0']), w[m]):.0f}%  "
              f"EMS={wpct(c.loc[m,'arrived_by_ems'], w[m]):.0f}%  "
              f"median wait={c.loc[m,'wait_minutes'].median():.0f} min  "
              f"median pain={c.loc[m,'pain_score'].median():.0f}")


def main(path, name):
    df = load(path)
    c = build_cohort(df, name)
    describe_cohort(c)

    outcomes = ([x for x in c.columns if x.startswith(("triage_", "test_", "med_", "next_", "dx_"))])
    outcomes = [x for x in outcomes if x != "triage_level"]
    table = compare_by_sex(c, outcomes, "all")

    # Same pain, same relief?
    band_tables = [compare_by_sex(c[c["pain_band"].eq(b)],
                                  ["med_any_pain_relief", "med_opioid"], f"pain {b}")
                   for b in ["0-3", "4-6", "7-10"]]
    band = pd.concat(band_tables, ignore_index=True)

    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))
    print("\nPain relief within the same pain-score band:")
    print(band.to_string(index=False))

    c.to_csv(f"nhamcs_cohort_{name}.csv", index=False)
    pd.concat([table, band]).to_csv(f"nhamcs_sex_comparison_{name}.csv", index=False)
    print(f"\nSaved nhamcs_cohort_{name}.csv and nhamcs_sex_comparison_{name}.csv")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python nhamcs_filter.py <nhamcs2022.pkl> [cohort]  "
                 f"cohorts: {', '.join(COHORTS)}")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else DEFAULT_COHORT)
