"""
How often do the "diabetes recorded" models predict < 50% for people whose labs
meet diabetes thresholds (HbA1c >= 6.5% or fasting glucose >= 126 mg/dL)?

Part 1: NAMCS 2018-2019 adult visits, 5-fold cross-validated predictions
        (same features, folds and models as model_predictions.py).
Part 2: NHANES 2017-March 2020 adults 20+, not pregnant, scored with models
        refit on all adult NAMCS visits using only the fields NHANES shares.

Outputs: underprediction_namcs_cv.csv, underprediction_nhanes.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from model_predictions import (FEATURES, OUTCOMES, ROOT, DATA, SEED,
                               build_namcs, fit, make_models)

MODELS = ["lr", "gb"]
FULL_FEATS = [f for f in FEATURES if f not in OUTCOMES["diabetes"][1]]
# Fields available in both NAMCS and NHANES. lab_items_collected is a NAMCS
# design indicator (labs were only asked of primary-care physicians); NHANES
# participants all had labs drawn, so it is set to 1 for them.
SHARED_FEATS = ["age", "female", "race_black", "race_hispanic", "race_other", "bmi",
                "high_blood_pressure", "high_cholesterol",
                "hba1c_pct", "fasting_glucose_mg_dl", "lab_items_collected"]
SMALL_N = 30


def cv_predictions(X, y, w, groups):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(y)) for m in MODELS}
    for tr, te in cv.split(X, y, groups):
        for m in MODELS:
            model = fit(make_models()[m], X.iloc[tr], y[tr], w[tr])
            oof[m][te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def summarise(frame, preds, weight):
    """frame has hba1c, glucose, bmi, age, female. Returns one row per group x sex."""
    a1c_hi = frame["hba1c_pct"] >= 6.5
    glu_hi = frame["fasting_glucose_mg_dl"] >= 126
    high = a1c_hi | glu_hi
    groups = {
        "All with high HbA1c or glucose": high,
        "Only fasting glucose high": glu_hi & ~a1c_hi,
        "Only HbA1c high": a1c_hi & ~glu_hi,
        "BMI < 25": high & (frame["bmi"] < 25),
        "Age 35-49": high & frame["age"].between(35, 49),
    }
    rows = []
    for gname, gmask in groups.items():
        for sex, smask in [("Women", frame["female"] == 1), ("Men", frame["female"] == 0)]:
            m = (gmask & smask).to_numpy()
            n = int(m.sum())
            row = {"group": gname, "sex": sex, "n": n,
                   "flag": f"n<{SMALL_N}" if n < SMALL_N else ""}
            for name, p in preds.items():
                under = p[m] < 0.5
                row[f"{name}_pct_under50_weighted"] = (
                    round(100 * np.average(under, weights=weight[m]), 1) if n else np.nan)
                row[f"{name}_pct_under50_unweighted"] = round(100 * under.mean(), 1) if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_nhanes():
    rd = lambda f: pd.read_sas(DATA / "nhanes" / f"{f}.xpt").set_index("SEQN")
    demo, bmx, bpq, ghb, glu = (rd(f) for f in ["P_DEMO", "P_BMX", "P_BPQ", "P_GHB", "P_GLU"])
    d = demo.join([bmx[["BMXBMI"]], bpq[["BPQ020", "BPQ080"]], ghb[["LBXGH"]],
                   glu[["LBXGLU", "WTSAFPRP"]]], how="left")
    d = d[(d["RIDAGEYR"] >= 20) & (d["RIDEXPRG"] != 1) & (d["RIDSTATR"] == 2)]

    X = pd.DataFrame(index=d.index)
    X["age"] = d["RIDAGEYR"]
    X["female"] = (d["RIAGENDR"] == 2).astype(float)
    X["race_black"] = (d["RIDRETH3"] == 4).astype(float)
    X["race_hispanic"] = d["RIDRETH3"].isin([1, 2]).astype(float)
    X["race_other"] = d["RIDRETH3"].isin([6, 7]).astype(float)
    X["bmi"] = d["BMXBMI"]
    # "Ever told by a doctor": 1 = yes; no / refused / don't know / missing -> 0
    X["high_blood_pressure"] = (d["BPQ020"] == 1).astype(float)
    X["high_cholesterol"] = (d["BPQ080"] == 1).astype(float)
    X["hba1c_pct"] = d["LBXGH"]
    X["fasting_glucose_mg_dl"] = d["LBXGLU"]   # fasting subsample only
    X["lab_items_collected"] = 1.0
    return X, d["WTMECPRP"].to_numpy()


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def main():
    X_all, Y_all, w_all, groups = build_namcs()
    y = Y_all["diabetes"].to_numpy()
    w = w_all / w_all.mean()

    # ---- Part 1: NAMCS cross-validated predictions ----
    oof_full = cv_predictions(X_all[FULL_FEATS], y, w, groups)
    oof_shared = cv_predictions(X_all[SHARED_FEATS], y, w, groups)
    preds1 = {"lr": oof_full["lr"], "gb": oof_full["gb"],
              "lr_shared": oof_shared["lr"], "gb_shared": oof_shared["gb"]}
    part1 = summarise(X_all, preds1, w_all)
    # How many "only X high" cases had the other test recorded at all
    a1c_hi, glu_hi = X_all["hba1c_pct"] >= 6.5, X_all["fasting_glucose_mg_dl"] >= 126
    print("NAMCS only-glucose-high with an HbA1c recorded:",
          int((glu_hi & ~a1c_hi & X_all["hba1c_pct"].notna()).sum()), "of", int((glu_hi & ~a1c_hi).sum()))
    print("NAMCS only-HbA1c-high with a glucose recorded:",
          int((a1c_hi & ~glu_hi & X_all["fasting_glucose_mg_dl"].notna()).sum()), "of", int((a1c_hi & ~glu_hi).sum()))

    # ---- Part 2: NHANES, models refit on all NAMCS adults with shared fields ----
    X_nh, w_nh = load_nhanes()
    preds2 = {}
    for m in MODELS:
        model = fit(make_models()[m], X_all[SHARED_FEATS], y, w)
        preds2[m] = model.predict_proba(X_nh[SHARED_FEATS])[:, 1]
    part2 = summarise(X_nh, preds2, w_nh)
    a1c_hi, glu_hi = X_nh["hba1c_pct"] >= 6.5, X_nh["fasting_glucose_mg_dl"] >= 126
    print("NHANES adults 20+, not pregnant, examined:", len(X_nh),
          "| HbA1c measured:", int(X_nh["hba1c_pct"].notna().sum()),
          "| fasting glucose measured:", int(X_nh["fasting_glucose_mg_dl"].notna().sum()))
    print("NHANES only-HbA1c-high with a fasting glucose measured:",
          int((a1c_hi & ~glu_hi & X_nh["fasting_glucose_mg_dl"].notna()).sum()), "of", int((a1c_hi & ~glu_hi).sum()))

    part1.to_csv(ROOT / "underprediction_namcs_cv.csv", index=False)
    part2.to_csv(ROOT / "underprediction_nhanes.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    show("Part 1: NAMCS adult visits, 5-fold CV (weights = PATWT)", part1)
    show("Part 2: NHANES 2017-Mar 2020 adults 20+ (weights = WTMECPRP)", part2)


if __name__ == "__main__":
    main()
