"""
Predict physician actions at adult NAMCS 2018-2019 office visits from patient
characteristics, then score the patients in patients.csv.

Outputs (next to this script):
  predictions_round1.csv  patient_id + predicted probability (%) per outcome x model
  cv_auc_round1.csv       cross-validated AUC per outcome x model
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 2026

# --------------------------------------------------------------------------
# Reason-for-visit categories. NAMCS RFV codes are stored x10 (3100.0 -> 31000).
# Each category is a set of [lo, hi] code ranges, matched against RFV1-RFV5.
# --------------------------------------------------------------------------
RFV_RANGES = {
    "rfv_general_exam":   [(31000, 31099)],                  # general medical exam
    "rfv_progress":       [(48000, 48099)],                  # progress / follow-up visit
    "rfv_diabetes":       [(22050, 22099)],                  # diabetes mellitus
    "rfv_urinary":        [(16400, 16899)],                  # urinary symptoms
    "rfv_genital":        [(16900, 18299), (20250, 20299)],  # genital symptoms, fungal infection
    "rfv_respiratory":    [(14400, 14499), (14550, 14599)],  # cough, sore throat
    "rfv_fatigue_thirst": [(10150, 10199), (10900, 10999)],  # tiredness; fluid/thirst symptoms
    "rfv_musculoskeletal": [(19000, 19999)],                 # back, knee, other MSK symptoms
}
# Keyword -> category mapping for the free-text reason_for_visit in patients.csv
RFV_KEYWORDS = {
    "check-up": ["rfv_general_exam"],
    "follow-up": ["rfv_progress"],
    "diabetes": ["rfv_diabetes"],
    "urinat": ["rfv_urinary"],
    "urinary": ["rfv_urinary"],
    "genital": ["rfv_genital"],
    "yeast": ["rfv_genital"],
    "cough": ["rfv_respiratory"],
    "sore throat": ["rfv_respiratory"],
    "tired": ["rfv_fatigue_thirst"],
    "thirst": ["rfv_fatigue_thirst"],
    "back pain": ["rfv_musculoskeletal"],
    "knee pain": ["rfv_musculoskeletal"],
}

# Multum drug categories (RXnCATk) and drug IDs
METFORMIN_IDS = {"d03807"}      # sole drug in Multum cat 214 (biguanides)
CAT_INSULIN = "215"
CAT_GLP1 = "373"
CAT_SGLT2 = "442"
CAT_STATIN = "173"              # HMG-CoA reductase inhibitors

FEATURES = [
    "age", "female", "race_black", "race_hispanic", "race_other", "bmi",
    "high_blood_pressure", "high_cholesterol", "heart_disease", "kidney_disease",
    "has_diabetes", "visits_past_year", "seen_before", "primary_care_visit",
    *RFV_RANGES.keys(),
    "lab_items_collected", "hba1c_pct", "fasting_glucose_mg_dl",
    "on_metformin", "on_insulin",
]

# Outcome definitions: (population filter column or None, features to drop).
# Features dropped are those built from the same NAMCS fields as the outcome
# (or constant within the population), so the model cannot just read the answer.
OUTCOMES = {
    "diabetes":          (None,           ["has_diabetes"]),
    "glycemic_test":     ("pop_glyc",     ["has_diabetes", "hba1c_pct",
                                           "fasting_glucose_mg_dl", "lab_items_collected"]),
    "lifestyle_counsel": (None,           []),
    "metformin":         ("pop_diabetes", ["has_diabetes", "on_metformin", "on_insulin"]),
    "glp1_sglt2":        ("pop_diabetes", ["has_diabetes", "on_metformin", "on_insulin"]),
    "statin":            ("pop_diabetes", ["has_diabetes"]),
}


def any_in_ranges(frame, ranges):
    out = np.zeros(len(frame), dtype=bool)
    for lo, hi in ranges:
        out |= ((frame >= lo) & (frame <= hi)).any(axis=1).to_numpy()
    return out


def build_namcs():
    df = pd.read_pickle(DATA / "namcs_2018_2019.pkl")
    df = df[df["AGE"] >= 18].copy()

    def neg_to_nan(s):
        return s.where(s >= 0)

    X = pd.DataFrame(index=df.index)
    X["age"] = df["AGE"]
    X["female"] = (df["SEX"] == 1).astype(float)
    X["race_black"] = (df["RACERETH"] == 2).astype(float)
    X["race_hispanic"] = (df["RACERETH"] == 3).astype(float)
    X["race_other"] = (df["RACERETH"] == 4).astype(float)
    X["bmi"] = neg_to_nan(df["BMI"])
    X["high_blood_pressure"] = df["HTN"]
    X["high_cholesterol"] = df["HYPLIPID"]
    X["heart_disease"] = ((df["CAD"] == 1) | (df["CHF"] == 1)).astype(float)
    X["kidney_disease"] = ((df["CKD"] == 1) | (df["ESRD"] == 1)).astype(float)
    dm_chart = (df[["DIABTYP0", "DIABTYP1", "DIABTYP2"]] == 1).any(axis=1)
    X["has_diabetes"] = dm_chart.astype(float)
    # PASTVIS is -7 (not applicable) for new patients -> 0 prior visits
    X["visits_past_year"] = neg_to_nan(df["PASTVIS"].replace(-7, 0))
    X["seen_before"] = (df["SENBEFOR"] == 1).astype(float)
    X["primary_care_visit"] = df["PRIMCARE"].map({1: 1.0, 2: 0.0})

    rfv = df[[f"RFV{i}" for i in range(1, 6)]]
    for name, ranges in RFV_RANGES.items():
        X[name] = any_in_ranges(rfv, ranges).astype(float)

    # Lab items were only collected by primary-care-specialty physicians (-7 otherwise)
    X["lab_items_collected"] = (df["A1C"] != -7).astype(float)
    X["hba1c_pct"] = df["A1CRES"].where((df["A1C"] == 1) & (df["A1CRES"] >= 0))
    X["fasting_glucose_mg_dl"] = df["FBGRES"].where((df["FBG"] == 1) & (df["FBGRES"] >= 0))

    drug = df[[f"DRUGID{n}" for n in range(1, 31)]]

    def has_cat(code):
        m = np.zeros(len(df), dtype=bool)
        for n in range(1, 31):
            m |= df[[f"RX{n}CAT{k}" for k in range(1, 5)]].eq(code).any(axis=1).to_numpy()
        return m

    metformin = drug.isin(METFORMIN_IDS).any(axis=1).to_numpy()
    insulin = has_cat(CAT_INSULIN)
    X["on_metformin"] = metformin.astype(float)
    X["on_insulin"] = insulin.astype(float)

    # ---- outcomes ----
    diag = df[[f"DIAG{i}" for i in range(1, 6)]].astype(str)
    dm_diag = diag.apply(lambda s: s.str.match(r"^E(0[89]|1[013])")).any(axis=1)
    diabetes = (dm_chart | dm_diag).to_numpy()

    Y = pd.DataFrame(index=df.index)
    Y["diabetes"] = diabetes.astype(int)
    Y["glycemic_test"] = ((df["HGBA"] == 1) | (df["GLUCOSE"] == 1)
                          | (df["A1C"] == 1) | (df["FBG"] == 1)).astype(int)
    Y["lifestyle_counsel"] = ((df["DIETNUTR"] == 1) | (df["EXERCISE"] == 1)
                              | (df["WTREDUC"] == 1)).astype(int)
    Y["metformin"] = metformin.astype(int)
    Y["glp1_sglt2"] = (has_cat(CAT_GLP1) | has_cat(CAT_SGLT2)).astype(int)
    Y["statin"] = has_cat(CAT_STATIN).astype(int)

    Y["pop_diabetes"] = diabetes
    # "On record" is only observable where the lab items were collected
    Y["pop_glyc"] = (~diabetes) & (df["A1C"] != -7).to_numpy()

    return X, Y, df["PATWT"].to_numpy(), df["PHYCODE"].to_numpy()


def build_patients():
    p = pd.read_csv(DATA / "patients.csv")
    X = pd.DataFrame(index=p.index)
    X["age"] = p["age"]
    X["female"] = (p["sex"].str.lower() == "female").astype(float)
    race = p["race"].str.lower()
    known = {"white", "black", "hispanic", "asian", "other"}
    assert race.isin(known).all(), f"unmapped race: {set(race) - known}"
    X["race_black"] = (race == "black").astype(float)
    X["race_hispanic"] = (race == "hispanic").astype(float)
    X["race_other"] = race.isin(["asian", "other"]).astype(float)
    X["bmi"] = p["bmi"]
    for c in ["high_blood_pressure", "high_cholesterol", "heart_disease",
              "kidney_disease", "has_diabetes", "visits_past_year",
              "seen_before", "primary_care_visit"]:
        X[c] = p[c].astype(float)

    for name in RFV_RANGES:
        X[name] = 0.0
    for i, text in p["reason_for_visit"].str.lower().items():
        hits = sorted({cat for kw, cats in RFV_KEYWORDS.items() if kw in text for cat in cats})
        assert hits, f"unmapped reason_for_visit: {text!r}"
        X.loc[i, hits] = 1.0

    X["lab_items_collected"] = 1.0  # primary-care visits with lab fields available
    X["hba1c_pct"] = p["hba1c_pct"]
    X["fasting_glucose_mg_dl"] = p["fasting_glucose_mg_dl"]
    meds = p["diabetes_medicines"].fillna("none").str.lower()
    X["on_metformin"] = meds.str.contains("metformin").astype(float)
    X["on_insulin"] = meds.str.contains("insulin").astype(float)
    return p, X[FEATURES]


def make_models():
    lr = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0),
    )
    gb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
    )
    return {"lr": lr, "gb": gb}


def fit(model, X, y, w):
    key = "logisticregression__sample_weight" if hasattr(model, "steps") else "sample_weight"
    return model.fit(X, y, **{key: w})


def main():
    X_all, Y_all, w_all, groups_all = build_namcs()
    patients, X_pat = build_patients()

    out = pd.DataFrame({"patient_id": patients["patient_id"]})
    auc_rows = []

    for outcome, (pop_col, drop) in OUTCOMES.items():
        feats = [f for f in FEATURES if f not in drop]
        mask = np.ones(len(X_all), bool) if pop_col is None else Y_all[pop_col].to_numpy()
        X, y = X_all.loc[mask, feats], Y_all.loc[mask, outcome].to_numpy()
        w = w_all[mask] / w_all[mask].mean()
        groups = groups_all[mask]

        # Patients outside the model's population get no prediction
        if pop_col == "pop_diabetes":
            pat_mask = X_pat["has_diabetes"] == 1
        elif pop_col == "pop_glyc":
            pat_mask = X_pat["has_diabetes"] == 0
        else:
            pat_mask = pd.Series(True, index=X_pat.index)

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for name in ["lr", "gb"]:
            oof = np.zeros(len(y))
            fold_aucs = []
            for tr, te in cv.split(X, y, groups):
                m = fit(make_models()[name], X.iloc[tr], y[tr], w[tr])
                oof[te] = m.predict_proba(X.iloc[te])[:, 1]
                fold_aucs.append(roc_auc_score(y[te], oof[te], sample_weight=w[te]))
            auc_rows.append({
                "outcome": outcome, "model": name,
                "n_visits": len(y), "n_events": int(y.sum()),
                "weighted_prevalence_pct": round(100 * np.average(y, weights=w), 1),
                "cv_auc_pooled": round(roc_auc_score(y, oof, sample_weight=w), 3),
                "cv_auc_fold_mean": round(np.mean(fold_aucs), 3),
                "cv_auc_fold_sd": round(np.std(fold_aucs), 3),
            })

            final = fit(make_models()[name], X, y, w)
            col = f"{outcome}_{name}_pct"
            out[col] = np.nan
            if pat_mask.any():
                out.loc[pat_mask, col] = 100 * final.predict_proba(X_pat.loc[pat_mask, feats])[:, 1]
            out[col] = out[col].round(1)

    assert out["patient_id"].tolist() == patients["patient_id"].tolist()
    out.to_csv(ROOT / "predictions_round1.csv", index=False)
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(ROOT / "cv_auc_round1.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 100)
    print(auc.to_string(index=False))
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
