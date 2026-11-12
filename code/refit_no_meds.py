"""
Refit the blind-run "diabetes recorded" logistic models WITHOUT diabetes-medicine
fields (on_metformin, on_insulin), and report:

  Q1  sex odds ratio (female vs male), full and shared-fields models
  Q2  same models + sex x BMI<25 interaction
  Q3  NHANES miss rate (predicted < 50% among HbA1c >= 6.5 or fasting glucose >= 126),
      BMI < 25 vs 25+, women vs men -- NHANES 2015-16 pooled with 2017-Mar 2020,
      exam (MEC) weights
  Q4  same as Q3, only people with BOTH HbA1c and fasting glucose, fasting-subsample weights

Models, features, imputation and penalty are those of model_predictions.py /
diabetes_underprediction.py (LogisticRegression C=1 on standardised features).
Odds ratios are converted back to the unstandardised scale (per 1 unit of the 0/1 variable).

95% CIs: percentile bootstrap, B reps. Each rep resamples NAMCS physicians (PHYCODE,
as in the blind run's CV folds) and refits every model; NHANES quantities in the same
rep also resample NHANES PSUs within strata. So NHANES CIs include model uncertainty.

NHANES pooling: weights multiplied by cycle length / total (2 / 5.2 for 2015-16,
3.2 / 5.2 for 2017-Mar 2020), per NCHS guidance for combining cycles.

Usage:  .venv/Scripts/python.exe refit_no_meds.py [--boot 500]
Output: results_refit/refit_no_meds.csv  (+ log printed)
"""
import sys
import numpy as np
import pandas as pd

from model_predictions import FEATURES, DATA, ROOT, build_namcs, fit, make_models

B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 500
RNG = np.random.default_rng(2026)
OUT = ROOT / "results_refit"
OUT.mkdir(exist_ok=True)

DROP = ["has_diabetes", "on_metformin", "on_insulin"]
FULL = [f for f in FEATURES if f not in DROP]
SHARED = ["age", "female", "race_black", "race_hispanic", "race_other", "bmi",
          "high_blood_pressure", "high_cholesterol",
          "hba1c_pct", "fasting_glucose_mg_dl", "lab_items_collected"]
INTER = ["bmi_lt25", "female_x_bmi_lt25"]


# ------------------------------------------------------------------ data ---
def add_interaction(X):
    X = X.copy()
    lt = (X["bmi"] < 25).astype(float).where(X["bmi"].notna())   # NaN when BMI missing -> imputed + flagged
    X["bmi_lt25"] = lt
    X["female_x_bmi_lt25"] = X["female"] * lt
    return X


def load_nhanes():
    def rd(f):
        return pd.read_sas(DATA / "nhanes" / f"{f}.xpt", format="xport").set_index("SEQN")
    frames = []
    for cyc, files, wmec, wfast, years in [
        ("2015-16", ["DEMO_I", "BMX_I", "BPQ_I", "GHB_I", "GLU_I"], "WTMEC2YR", "WTSAF2YR", 2.0),
        ("2017-20", ["P_DEMO", "P_BMX", "P_BPQ", "P_GHB", "P_GLU"], "WTMECPRP", "WTSAFPRP", 3.2),
    ]:
        demo, bmx, bpq, ghb, glu = (rd(f) for f in files)
        d = demo.join([bmx[["BMXBMI"]], bpq[["BPQ020", "BPQ080"]], ghb[["LBXGH"]],
                       glu[["LBXGLU", wfast]]], how="left")
        d = d[(d["RIDAGEYR"] >= 20) & (d["RIDEXPRG"] != 1) & (d["RIDSTATR"] == 2)].copy()
        d["cycle"] = cyc
        d["w_mec"] = d[wmec] * years / 5.2
        d["w_fast"] = d[wfast] * years / 5.2
        frames.append(d)
    d = pd.concat(frames)
    X = pd.DataFrame(index=d.index)
    X["age"] = d["RIDAGEYR"]
    X["female"] = (d["RIAGENDR"] == 2).astype(float)
    X["race_black"] = (d["RIDRETH3"] == 4).astype(float)
    X["race_hispanic"] = d["RIDRETH3"].isin([1, 2]).astype(float)
    X["race_other"] = d["RIDRETH3"].isin([6, 7]).astype(float)
    X["bmi"] = d["BMXBMI"]
    X["high_blood_pressure"] = (d["BPQ020"] == 1).astype(float)
    X["high_cholesterol"] = (d["BPQ080"] == 1).astype(float)
    X["hba1c_pct"] = d["LBXGH"]
    X["fasting_glucose_mg_dl"] = d["LBXGLU"]
    X["lab_items_collected"] = 1.0
    meta = pd.DataFrame({"cycle": d["cycle"], "w_mec": d["w_mec"], "w_fast": d["w_fast"],
                         "stratum": d["cycle"] + "_" + d["SDMVSTRA"].astype(int).astype(str),
                         "psu": d["cycle"] + "_" + d["SDMVSTRA"].astype(int).astype(str) + "_" +
                                d["SDMVPSU"].astype(int).astype(str)}, index=d.index)
    return X, meta


# ------------------------------------------------------------- helpers ---
def lr_or(model, feats, term):
    """Odds ratio for `term` per 1 unit on the original scale."""
    sc = model.named_steps["standardscaler"]
    coef = model.named_steps["logisticregression"].coef_[0]
    i = feats.index(term)                    # imputer keeps column order; indicators appended after
    return float(np.exp(coef[i] / sc.scale_[i]))


def fit_lr(X, y, w):
    return fit(make_models()["lr"], X, y, w)


def miss_cells(p_hat, X, meta, w_col, both_only):
    """Weighted % predicted < 50% among high-lab people, by BMI group x sex."""
    a1c, glu = X["hba1c_pct"], X["fasting_glucose_mg_dl"]
    base = X["bmi"].notna()
    if both_only:
        base &= a1c.notna() & glu.notna() & (meta["w_fast"] > 0)
    high = base & ((a1c >= 6.5) | (glu >= 126))
    w = meta[w_col].to_numpy()
    out = {}
    for bname, bm in (("BMI<25", X["bmi"] < 25), ("BMI25+", X["bmi"] >= 25)):
        for sname, sm in (("women", X["female"] == 1), ("men", X["female"] == 0)):
            m = (high & bm & sm).to_numpy()
            out[(bname, sname)] = (m, w)
    return out


def miss_rates(cells, under, wmult):
    res = {}
    for k, (m, w) in cells.items():
        ww = w[m] * wmult[m]
        res[k] = 100 * np.sum(ww * under[m]) / np.sum(ww) if ww.sum() > 0 else np.nan
    for b in ("BMI<25", "BMI25+"):
        res[(b, "women-men")] = res[(b, "women")] - res[(b, "men")]
    return res


# ----------------------------------------------------------------- run ---
def main():
    X_all, Y_all, w_all, phys = build_namcs()
    y = Y_all["diabetes"].to_numpy()
    w = w_all / w_all.mean()
    Xi = add_interaction(X_all)
    X_nh, meta = load_nhanes()
    Xnh_i = add_interaction(X_nh)

    specs = {  # name: (frame, features)
        "full": (Xi, FULL), "full_int": (Xi, FULL + INTER),
        "shared": (Xi, SHARED), "shared_int": (Xi, SHARED + INTER),
    }

    def estimate(idx_namcs, nh_mult):
        r = {}
        models = {}
        for name, (F, feats) in specs.items():
            models[name] = fit_lr(F.iloc[idx_namcs][feats], y[idx_namcs], w[idx_namcs])
            r[(name, "OR_female")] = lr_or(models[name], feats, "female")
        for name in ("full_int", "shared_int"):
            feats = specs[name][1]
            of = r[(name, "OR_female")]
            oi = lr_or(models[name], feats, "female_x_bmi_lt25")
            r[(name, "OR_interaction")] = oi
            r[(name, "OR_female_BMI25plus")] = of
            r[(name, "OR_female_BMIlt25")] = of * oi
        p = models["shared"].predict_proba(X_nh[SHARED])[:, 1]
        under = (p < 0.5).astype(float)
        for q, wcol, both in (("Q3", "w_mec", False), ("Q4", "w_fast", True)):
            mr = miss_rates(miss_cells(p, X_nh, meta, wcol, both), under, nh_mult)
            for k, v in mr.items():
                r[(q,) + k] = v
        return r

    n_all = len(y)
    point = estimate(np.arange(n_all), np.ones(len(X_nh)))

    # cluster indices
    phys_codes, phys_u = pd.factorize(pd.Series(phys))
    phys_idx = [np.flatnonzero(phys_codes == k) for k in range(len(phys_u))]
    strata = meta["stratum"].to_numpy()
    psu = meta["psu"].to_numpy()
    strata_psus = {s: np.unique(psu[strata == s]) for s in np.unique(strata)}
    psu_rows = {u: np.flatnonzero(psu == u) for u in np.unique(psu)}

    def one_rep(seed):
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(phys_u), len(phys_u))
        idx = np.concatenate([phys_idx[k] for k in pick])
        mult = np.zeros(len(X_nh))
        for s, us in strata_psus.items():                 # resample PSUs within strata
            for u in rng.choice(us, size=len(us), replace=True):
                mult[psu_rows[u]] += 1
        return estimate(idx, mult)

    from joblib import Parallel, delayed
    seeds = RNG.integers(0, 2**31 - 1, B)
    boots = Parallel(n_jobs=-1, verbose=5)(delayed(one_rep)(int(sd)) for sd in seeds)

    bt = pd.DataFrame(boots)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(bt[k], [2.5, 97.5])
        rows.append({"key": " | ".join(map(str, k)), "estimate": v, "CI_low": lo, "CI_high": hi})
    res = pd.DataFrame(rows)

    # group sizes for Q3/Q4
    sizes = []
    for q, wcol, both in (("Q3", "w_mec", False), ("Q4", "w_fast", True)):
        for k, (m, _) in miss_cells(None, X_nh, meta, wcol, both).items():
            sizes.append({"key": " | ".join((q,) + k), "n": int(m.sum()),
                          "n_2015_16": int((m & (meta["cycle"] == "2015-16").to_numpy()).sum()),
                          "n_2017_20": int((m & (meta["cycle"] == "2017-20").to_numpy()).sum())})
    res = res.merge(pd.DataFrame(sizes), on="key", how="left")
    res.to_csv(OUT / "refit_no_meds.csv", index=False)

    print(f"\nNAMCS adult visits: {n_all:,} (diabetes recorded: {int(y.sum()):,}); physicians: {len(phys_u)}")
    print(f"NHANES adults 20+, examined, not pregnant (2015-16 + 2017-Mar 2020): {len(X_nh):,}")
    print(f"Bootstrap reps: {B}\n")
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
