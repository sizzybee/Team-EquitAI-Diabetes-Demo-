"""
Type 2 diabetes sex-gap audit -- NAMCS 2018-19 (office visits) + NHANES.

Usage:
    .venv/Scripts/python.exe diabetes_audit.py            # everything
    .venv/Scripts/python.exe diabetes_audit.py namcs      # office visits only
    .venv/Scripts/python.exe diabetes_audit.py nhanes     # NHANES only

Outputs go to results_diabetes/ (one CSV per check + all_results.csv).

DATA
  NAMCS 2018-19   data/namcs_2018_2019.pkl (built by data/pool_years.py).
                  Visits by adults 18+. Labs = most recent HbA1c / fasting glucose
                  in the past 12 months (lab section filled for ~1/3 of visits).
  NHANES 2017-Mar 2020 (pre-pandemic, "P_" files) -- adults 20+, not pregnant.
                  Everyone gets HbA1c; a morning subsample gets fasting glucose.
  NHANES 2015-16  ("_I" files) -- the latest cycle with an oral glucose tolerance
                  test (2-hour glucose), used only for the test-comparison check.
  NHANES sex coding: RIAGENDR 1 = male, 2 = female (opposite of NAMCS).

THRESHOLDS (ADA)
  Diabetes:     HbA1c >= 6.5%  or  fasting glucose >= 126 mg/dL  or  2-h OGTT >= 200
  Prediabetes:  HbA1c 5.7-6.4  or  fasting glucose 100-125       or  2-h OGTT 140-199
  (WHO uses 110-125 for impaired fasting glucose -- compared below.)

STATISTICS
  raw   survey-weighted % women vs men
  adj   odds ratio for women from a weighted logistic regression (OR < 1 = women
        less likely), SEs clustered on survey PSU -- approximate design.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 300)
OUT = "results_diabetes"
os.makedirs(OUT, exist_ok=True)
RESULTS = []          # every row printed is also saved here


# ===================================================================== helpers
def wpct(x, col, w="w"):
    x = x[x[col].notna()]
    if len(x) == 0 or x[w].sum() == 0:
        return np.nan, 0, 0
    return 100 * (x[w] * x[col].astype(float)).sum() / x[w].sum(), len(x), int(x[col].sum())


def compare(source, section, label, x, col, w="w", by=None):
    """Weighted % women vs men (optionally within strata)."""
    groups = [("all", x)] if by is None else [(str(g), s) for g, s in x.groupby(by, observed=True)]
    rows = []
    for g, s in groups:
        f, fn, fy = wpct(s[s.female == 1], col, w)
        m, mn, my = wpct(s[s.female == 0], col, w)
        rows.append({"source": source, "section": section, "check": label, "stratum": g,
                     "women_%": round(f, 1), "men_%": round(m, 1), "gap": round(f - m, 1),
                     "women_n": fn, "women_yes": fy, "men_n": mn, "men_yes": my,
                     "reliable": min(fy, my) >= 30 and min(fn - fy, mn - my) >= 30 or min(fy, my) >= 30})
    RESULTS.extend(rows)
    return pd.DataFrame(rows)


def adjusted(source, section, label, x, col, covars, w="w", cluster="psu"):
    names = [c.replace("C(", "").replace(")", "") for c in covars]
    x = x.dropna(subset=[col, w, cluster] + names).copy()
    x[col] = x[col].astype(float)
    row = {"source": source, "section": section, "check": label + " (adjusted)", "stratum": "all",
           "n": len(x), "events": int(x[col].sum())}
    if x[col].sum() < 20 or (len(x) - x[col].sum()) < 20 or x.female.nunique() < 2:
        row["note"] = "too few events"
    else:
        try:
            r = sm.GLM.from_formula(f"{col} ~ female + " + " + ".join(covars), x,
                                    family=sm.families.Binomial(),
                                    var_weights=x[w] / x[w].mean()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(x[cluster])[0]})
            b, se = r.params["female"], r.bse["female"]
            row.update({"OR_women": round(np.exp(b), 2), "CI_low": round(np.exp(b - 1.96 * se), 2),
                        "CI_high": round(np.exp(b + 1.96 * se), 2), "p": round(r.pvalues["female"], 4)})
        except Exception as e:
            row["note"] = str(e)[:60]
    RESULTS.append(row)
    return row


def show(title, tables=(), adj=()):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
    for t in tables:
        if len(t):
            print(t.drop(columns=["source", "section", "reliable"]).to_string(index=False))
    if adj:
        print("  adjusted (OR < 1 = women less likely):")
        print(pd.DataFrame(adj).drop(columns=["source", "section"]).to_string(index=False))


# ===================================================================== NAMCS
def namcs():
    import nhamcs_filter as nf
    import namcs_focus as nm
    import focus_analyses as fa
    from namcs_diabetes_race import add_race

    src = "NAMCS 2018-19"
    d = add_race(nm.prep(nf.load(os.environ.get("NAMCS_PATH", "data/namcs_2018_2019.pkl"))))
    d["psu"] = d["CPSUM"]
    yes = lambda cols: nf.yes(d, cols).astype(int)
    meds = lambda cats: nm.meds_listed(d, cats=cats).astype(int)

    # extra flags
    d["blood_drawn"] = yes(["CBC", "CMP", "BMP", "CHOLEST"])
    d["lipid_panel"] = yes("CHOLEST")
    d["metabolic_panel"] = yes(["CMP", "BMP"])            # these include a (non-fasting) glucose
    d["a1c_or_glucose_today"] = yes(["HGBA", "GLUCOSE"])
    d["any_glucose_info"] = ((d.screened == 1) | (d.metabolic_panel == 1)).astype(int)
    d["no_diabetes_test"] = 1 - d.screened
    d["a1c_used"] = ((nf.num(d, "A1C") == 1) | (yes("HGBA") == 1)).astype(int)
    d["fbg_only"] = ((nf.num(d, "FBG") == 1) & (d.a1c_used == 0)).astype(int)
    d["counselled_lifestyle"] = yes(["DIETNUTR", "EXERCISE", "WTREDUC"])
    d["weight_counsel"] = yes("WTREDUC")
    d["dx_abnormal_glucose"] = fa.has_dx(d, ["R730", "R739"]).astype(int)   # incl. prediabetes R73.03
    d["metformin"] = meds(["214"])
    d["sulfonylurea"] = meds(["213"])
    d["insulin"] = meds(["215"])
    d["dpp4"] = meds(["371"])
    d["glp1"] = meds(["373"])
    d["sglt2"] = meds(["458"])
    d["newer_agent"] = d[["glp1", "sglt2"]].max(axis=1)        # heart/kidney-protective classes
    d["ace_arb"] = meds(["042", "056"])
    d["referred"] = yes("REFOTHMD")
    d["bmi_band"] = pd.cut(d.bmi, [0, 25, 30, 35, 100], labels=["<25", "25-29.9", "30-34.9", "35+"], right=False)
    d["no_dx_today"] = 1 - d["dm_dx_today"].astype(int)

    cov = ["age", "C(race)", "C(payer)", "C(msa)", "C(specialty)", "seen_before",
           "n_chronic", "cad", "chf", "copd", "htn", "depression", "C(YEAR)"]
    labs = d[d.lab_section]
    elig = labs[labs.age.between(35, 70) & (labs.diabetes == 0)].copy()

    # ---------------------------------------------------------- 1 screening
    sec = "Screening"
    t = [compare(src, sec, "35-70, no diabetes: HbA1c/glucose in past yr", elig, "screened"),
         compare(src, sec, "35-70, no diabetes, BMI>=25: screened", elig[elig.overweight == 1], "screened"),
         compare(src, sec, "screened, by age", elig, "screened", by="age_band"),
         compare(src, sec, "counting metabolic panels (contain glucose) as screening", elig, "any_glucose_info")]
    a = [adjusted(src, sec, "screened", elig, "screened", cov),
         adjusted(src, sec, "screened, primary care only", elig[elig.specialty == "primary"], "screened",
                  [c for c in cov if c != "C(specialty)"]),
         adjusted(src, sec, "screened, incl. metabolic panels", elig, "any_glucose_info", cov)]
    show("NAMCS 1. SCREENING", t, a)

    # ---------------------------------------------------------- 2 missed chances
    sec = "Missed chances"
    drawn = elig[elig.blood_drawn == 1]
    lipid = elig[elig.lipid_panel == 1]
    t = [compare(src, sec, "blood drawn at visit, but no HbA1c/glucose in past yr", drawn, "no_diabetes_test"),
         compare(src, sec, "cholesterol test at visit, but no HbA1c/glucose in past yr", lipid, "no_diabetes_test"),
         compare(src, sec, "established patient (seen before), not screened", elig[elig.seen_before == 1], "no_diabetes_test")]
    a = [adjusted(src, sec, "blood drawn but no diabetes test", drawn, "no_diabetes_test", cov),
         adjusted(src, sec, "cholesterol test but no diabetes test", lipid, "no_diabetes_test", cov)]
    show("NAMCS 2. MISSED CHANCES TO SCREEN", t, a)

    # ---------------------------------------------------------- 3 which test
    sec = "Which test"
    scr = labs[(labs.diabetes == 0) & (labs.screened == 1) & labs.age.between(18, 120)]
    both = labs[(nf.num(labs, "A1C") == 1) & (nf.num(labs, "FBG") == 1)].copy()
    both["a1c_dm_fbg_normal"] = ((both.a1c >= 6.5) & (both.fbg < 126)).astype(int)
    both["fbg_dm_a1c_normal"] = ((both.fbg >= 126) & (both.a1c < 6.5)).astype(int)
    both["a1c_pre_fbg_normal"] = ((both.a1c >= 5.7) & (both.fbg < 100)).astype(int)
    both["fbg_pre_a1c_normal"] = ((both.fbg >= 100) & (both.a1c < 5.7)).astype(int)
    t = [compare(src, sec, "no-diabetes adults tested: HbA1c used", scr, "a1c_used"),
         compare(src, sec, "no-diabetes adults tested: fasting glucose ONLY", scr, "fbg_only"),
         compare(src, sec, "both results: HbA1c in DM range but fasting glucose normal", both, "a1c_dm_fbg_normal"),
         compare(src, sec, "both results: fasting glucose in DM range but HbA1c normal", both, "fbg_dm_a1c_normal"),
         compare(src, sec, "both results: HbA1c prediabetic, fasting glucose normal (<100)", both, "a1c_pre_fbg_normal"),
         compare(src, sec, "both results: fasting glucose prediabetic, HbA1c normal (<5.7)", both, "fbg_pre_a1c_normal")]
    show("NAMCS 3. WHICH TEST: HbA1c vs FASTING GLUCOSE", t)

    # ---------------------------------------------------------- 4 not in chart
    sec = "Missing from chart"
    hi = labs[labs.lab_diabetes_range == 1].copy()
    hi["not_in_chart"] = 1 - hi.diabetes
    hi["flag_test"] = np.select([(hi.a1c >= 6.5) & (hi.fbg >= 126), hi.a1c >= 6.5, hi.fbg >= 126],
                                ["both high", "HbA1c only", "fasting glucose only"], "other")
    hi["level"] = np.where((hi.a1c >= 8) | (hi.fbg >= 180), "clearly high (A1c>=8 or FBG>=180)",
                           "just over threshold")
    t = [compare(src, sec, "labs in diabetes range: diabetes NOT in chart", hi, "not_in_chart"),
         compare(src, sec, "  by which test was high", hi, "not_in_chart", by="flag_test"),
         compare(src, sec, "  by how high", hi, "not_in_chart", by="level"),
         compare(src, sec, "  by specialty", hi, "not_in_chart", by="specialty"),
         compare(src, sec, "  by BMI", hi, "not_in_chart", by="bmi_band"),
         compare(src, sec, "  by age", hi, "not_in_chart", by="age_band")]
    a = [adjusted(src, sec, "labs in DM range, not in chart", hi, "not_in_chart",
                  [c for c in cov if c not in ("C(msa)",)])]
    show("NAMCS 4. LABS IN DIABETES RANGE, DIABETES MISSING FROM CHART", t, a)

    # ---------------------------------------------------------- 5 prediabetes
    sec = "Prediabetes"
    pre = labs[(labs.lab_prediabetes_range == 1) & (labs.diabetes == 0)]
    t = [compare(src, sec, "prediabetes-range labs: glucose problem coded today", pre, "dx_abnormal_glucose"),
         compare(src, sec, "prediabetes-range labs: diet/exercise/weight counselling", pre, "counselled_lifestyle"),
         compare(src, sec, "prediabetes-range labs: weight-loss counselling", pre, "weight_counsel"),
         compare(src, sec, "prediabetes-range labs: metformin listed", pre, "metformin")]
    a = [adjusted(src, sec, "prediabetes: lifestyle counselling", pre, "counselled_lifestyle", cov)]
    show("NAMCS 5. PREDIABETES", t, a)

    # ---------------------------------------------------------- 6 BMI
    sec = "BMI"
    dm = d[d.diabetes == 1]
    print("\n" + "=" * 110 + "\nNAMCS 6. BMI\n" + "=" * 110)
    rows = []
    for name, x in {"diabetes in chart": dm, "labs in DM range, NOT in chart": hi[hi.not_in_chart == 1],
                    "labs in DM range, in chart": hi[hi.not_in_chart == 0]}.items():
        for s, lab in ((1, "women"), (0, "men")):
            g = x[(x.female == s) & x.bmi.notna()]
            if len(g):
                wm = np.average(g.bmi, weights=g.w)
                rows.append({"source": src, "section": sec, "check": f"mean BMI: {name}", "stratum": lab,
                             "value": round(wm, 1), "n": len(g),
                             "pct_BMI_under_30": round(100 * np.average(g.bmi < 30, weights=g.w), 1),
                             "pct_BMI_under_25": round(100 * np.average(g.bmi < 25, weights=g.w), 1)})
    print(pd.DataFrame(rows).drop(columns=["source", "section"]).to_string(index=False))
    RESULTS.extend(rows)
    ad = labs[labs.age.between(35, 70)]
    t = [compare(src, sec, "35-70: diabetes in chart, by BMI", ad, "diabetes", by="bmi_band")]
    show("", t)

    # ---------------------------------------------------------- 7 warning signs
    sec = "Warning signs"
    nodm = d[d.diabetes == 0].copy()
    nodm["tested_today"] = ((nodm.a1c_or_glucose_today == 1) | (nodm.metabolic_panel == 1)).astype(int)
    nodm["tested_past_yr_or_today"] = ((nodm.tested_today == 1) | (nodm.has_glucose_or_a1c == 1)).astype(int)
    sx = {"excessive thirst": ("rfv5", [10353]), "frequent urination": ("rfv3", [1645]),
          "tiredness": ("rfv3", [1015]), "weight loss": ("rfv3", [1045]),
          "blurred/diminished vision": ("rfv5", [13052])}
    rfv5 = nodm[[f"RFV{i}" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce")
    rfv3 = nodm[[f"RFV{i}3D" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce")
    t = []
    nodm["any_hyper_sx"] = 0
    for name, (kind, codes) in sx.items():
        m = (rfv5 if kind == "rfv5" else rfv3).isin(codes).any(axis=1)
        nodm.loc[m, "any_hyper_sx"] = 1
        t.append(compare(src, sec, f"{name}: blood sugar tested today", nodm[m], "tested_today"))
    hs = nodm[nodm.any_hyper_sx == 1]
    t.insert(0, compare(src, sec, "ANY high-sugar symptom: blood sugar tested today", hs, "tested_today"))
    hs2 = hs[~rfv3.loc[hs.index].isin([1645]).any(axis=1)]
    t.insert(1, compare(src, sec, "  same, excluding frequent urination (prostate in men)", hs2, "tested_today"))
    # genital / urinary infections
    inf_dx = fa.has_dx(nodm, ["B37", "N76", "N390", "N30"])
    inf_rfv = rfv3.isin([1650, 1760, 1765, 1770]).any(axis=1)
    nodm["infection"] = (inf_dx | inf_rfv).astype(int)
    nodm["yeast_vaginal"] = (fa.has_dx(nodm, ["B37", "N76"]) | rfv3.isin([1760, 1765, 1770]).any(axis=1)).astype(int)
    nodm["uti"] = (fa.has_dx(nodm, ["N390", "N30"]) | rfv3.isin([1650]).any(axis=1)).astype(int)
    t += [compare(src, sec, "UTI / painful urination: blood sugar tested today", nodm[nodm.uti == 1], "tested_today"),
          compare(src, sec, "UTI / painful urination: tested today or in past yr", nodm[nodm.uti == 1], "tested_past_yr_or_today")]
    w_ = nodm[nodm.female == 1]
    yv = w_[w_.yeast_vaginal == 1]
    base = w_[(w_.yeast_vaginal == 0) & (w_.uti == 0)]
    rows = []
    for name, x in {"women with yeast/vaginal infection or symptoms": yv, "all other women's visits": base}.items():
        for col in ["tested_today", "tested_past_yr_or_today"]:
            p, n, y = wpct(x, col)
            rows.append({"source": src, "section": sec, "check": f"{name}: {col}", "stratum": "women",
                         "value": round(p, 1), "n": n, "yes": y})
    a = [adjusted(src, sec, "high-sugar symptom: tested today", hs, "tested_today", cov)]
    show("NAMCS 7. WARNING SIGNS OF HIGH BLOOD SUGAR", t, a)
    print(pd.DataFrame(rows).drop(columns=["source", "section"]).to_string(index=False))
    RESULTS.extend(rows)

    # ---------------------------------------------------------- 8 after diagnosis
    sec = "After diagnosis"
    dmv = d[d.diabetes == 1].copy()
    dmv["htn_on_ace_arb"] = dmv.ace_arb.where(dmv.htn == 1)
    outs = ["diabetes_med", "metformin", "sulfonylurea", "insulin", "dpp4", "glp1", "sglt2", "newer_agent",
            "statin", "htn_on_ace_arb", "has_a1c_result", "a1c_ordered_today", "foot_exam", "retinal_exam",
            "diabetes_education", "diet_counselling", "counselled_lifestyle", "referred", "a1c_under_7"]
    t = [compare(src, sec, f"has diabetes: {o}", dmv, o) for o in outs]
    covd = [c for c in cov if c != "diabetes"]
    a = [adjusted(src, sec, f"has diabetes: {o}", dmv, o, [c for c in covd if not (o == "htn_on_ace_arb" and c == "htn")])
         for o in ["diabetes_med", "metformin", "insulin", "newer_agent", "statin", "htn_on_ace_arb",
                   "has_a1c_result", "diabetes_education", "counselled_lifestyle", "a1c_under_7"]]
    show("NAMCS 8. AFTER DIAGNOSIS", t, a)

    # ---------------------------------------------------------- 9 race
    sec = "Race"
    t = [compare(src, sec, "screened (35-70, no DM), by race", elig, "screened", by="race5"),
         compare(src, sec, "labs in DM range, not in chart, by race", hi, "not_in_chart", by="race5")]
    show("NAMCS 9. RACE", t)

    # ---------------------------------------------------------- 10 counts of rare groups
    print("\nRare groups in NAMCS 2018-19 (adult visits):")
    for name, p in {"gestational diabetes in pregnancy (O24.4)": ["O244"],
                    "history of endocrine disease incl. gestational diabetes (Z86.3)": ["Z863"],
                    "PCOS (E28.2)": ["E282"], "screening for diabetes coded (Z13.1)": ["Z131"]}.items():
        m = fa.has_dx(d, p)
        print(f"  {name:62s} {int((m & (d.female == 1)).sum())} women / {int((m & (d.female == 0)).sum())} men")


# ===================================================================== NHANES
def read(f):
    return pd.read_sas(f"data/nhanes/{f}.xpt", format="xport")


def nhanes_base(cycle="P"):
    if cycle == "P":
        files = ["P_DEMO", "P_GHB", "P_GLU", "P_DIQ", "P_BMX", "P_RHQ", "P_HIQ", "P_HUQ"]
        wmec, wfast = "WTMECPRP", "WTSAFPRP"
    else:
        files = ["DEMO_I", "GHB_I", "GLU_I", "OGTT_I", "DIQ_I", "BMX_I", "RHQ_I"]
        wmec, wfast = "WTMEC2YR", "WTSAF2YR"
    d = read(files[0])
    for f in files[1:]:
        d = d.merge(read(f), on="SEQN", how="left")
    d = d[(d.RIDAGEYR >= 20) & (d.RIDEXPRG != 1)].copy()
    d["female"] = (d.RIAGENDR == 2).astype(int)
    d["age"] = d.RIDAGEYR
    d["age_band"] = pd.cut(d.age, [19, 34, 49, 64, 120], labels=["20-34", "35-49", "50-64", "65+"])
    d["race"] = d.RIDRETH3.map({1: "hispanic", 2: "hispanic", 3: "white", 4: "black", 6: "asian", 7: "other"})
    d["w"] = d[wmec]
    d["wf"] = d[wfast] if wfast in d else np.nan
    d["psu"] = d.SDMVSTRA * 10 + d.SDMVPSU
    d["a1c"], d["fpg"], d["bmi"] = d.get("LBXGH"), d.get("LBXGLU"), d.get("BMXBMI")
    d["told_dm"] = (d.DIQ010 == 1).astype(int)
    d["a1c_dm"] = (d.a1c >= 6.5).astype(float).where(d.a1c.notna())
    d["fpg_dm"] = (d.fpg >= 126).astype(float).where(d.fpg.notna())
    d["educ"] = d.get("DMDEDUC2", pd.Series(np.nan, index=d.index)).fillna(-1).astype(int).astype(str)
    d["pir"] = d.INDFMPIR.fillna(d.INDFMPIR.median())
    d["insured"] = (d.get("HIQ011", pd.Series(np.nan, index=d.index)) == 1).astype(int)
    d["usual_place"] = d.get("HUQ030", pd.Series(np.nan, index=d.index)).isin([1, 3]).astype(int)
    d["visits_2plus"] = (d.get("HUQ051", pd.Series(np.nan, index=d.index)) >= 2).astype(int)   # 2+ visits/yr
    d["bmi_band"] = pd.cut(d.bmi, [0, 25, 30, 35, 100], labels=["<25", "25-29.9", "30-34.9", "35+"], right=False)
    d["overweight"] = np.where(d.race == "asian", d.bmi >= 23, d.bmi >= 25)
    d["tested_3yr"] = (d.DIQ180 == 1).astype(float).where(d.DIQ180.isin([1, 2]))
    return d


def nhanes():
    src = "NHANES 2017-20"
    d = nhanes_base("P")
    cov = ["age", "C(race)", "C(educ)", "pir", "insured", "usual_place", "visits_2plus", "bmi"]

    # ---------------------------------------------------------- A undiagnosed
    sec = "Missing from chart"
    tot = d[(d.told_dm == 1) | (d.a1c_dm == 1)].copy()
    tot["undiagnosed"] = ((tot.told_dm == 0) & (tot.a1c_dm == 1)).astype(int)
    fs = d[d.wf > 0].copy()
    fs["w2"] = fs.wf
    totf = fs[(fs.told_dm == 1) | (fs.a1c_dm == 1) | (fs.fpg_dm == 1)].copy()
    totf["undiagnosed"] = ((totf.told_dm == 0) & ((totf.a1c_dm == 1) | (totf.fpg_dm == 1))).astype(int)
    t = [compare(src, sec, "people with diabetes (told or HbA1c>=6.5): NOT diagnosed", tot, "undiagnosed"),
         compare(src, sec, "  by age", tot, "undiagnosed", by="age_band"),
         compare(src, sec, "  by race", tot, "undiagnosed", by="race"),
         compare(src, sec, "  by BMI", tot, "undiagnosed", by="bmi_band"),
         compare(src, sec, "fasting subsample (HbA1c or fasting glucose): NOT diagnosed", totf, "undiagnosed", w="w2")]
    a = [adjusted(src, sec, "undiagnosed among people with diabetes", tot, "undiagnosed", cov),
         adjusted(src, sec, "undiagnosed (fasting subsample)", totf, "undiagnosed", cov, w="w2")]
    show("NHANES A. UNDIAGNOSED DIABETES", t, a)
    # never tested among undiagnosed
    und = tot[tot.undiagnosed == 1]
    t = [compare(src, sec, "undiagnosed diabetes: had a diabetes blood test in past 3 yrs", und, "tested_3yr"),
         compare(src, sec, "undiagnosed diabetes: has a usual place of care", und, "usual_place"),
         compare(src, sec, "undiagnosed diabetes: 2+ health care visits last yr", und, "visits_2plus")]
    show("NHANES A2. PEOPLE WITH UNDIAGNOSED DIABETES -- WERE THEY IN CARE?", t)

    # ---------------------------------------------------------- B screening
    sec = "Screening"
    nd = d[d.told_dm == 0]
    el = nd[nd.age.between(35, 70) & nd.overweight]
    t = [compare(src, sec, "no diabetes, 35-70, overweight*: blood tested for diabetes in past 3 yrs", el, "tested_3yr"),
         compare(src, sec, "  by race", el, "tested_3yr", by="race"),
         compare(src, sec, "  by age", el, "tested_3yr", by="age_band"),
         compare(src, sec, "all adults without diabetes: tested in past 3 yrs", nd, "tested_3yr")]
    a = [adjusted(src, sec, "screening group: tested in past 3 yrs", el, "tested_3yr", cov),
         adjusted(src, sec, "all adults without diabetes: tested", nd, "tested_3yr", cov)]
    show("NHANES B. SCREENING (*BMI>=25, >=23 for Asian adults)", t, a)
    sec = "Missed chances"
    mc = el[el.visits_2plus == 1].assign(not_tested=lambda x: 1 - x.tested_3yr)
    t = [compare(src, sec, "screening group with 2+ visits last yr: NOT tested in 3 yrs", mc, "not_tested")]
    a = [adjusted(src, sec, "2+ visits but not tested", mc, "not_tested", [c for c in cov if c != "visits_2plus"])]
    show("NHANES B2. MISSED CHANCES", t, a)

    # ---------------------------------------------------------- C which test
    sec = "Which test"
    und_f = totf[totf.undiagnosed == 1].copy()
    und_f["found_by_a1c_only"] = ((und_f.a1c_dm == 1) & (und_f.fpg_dm != 1)).astype(int)
    und_f["found_by_fpg_only"] = ((und_f.fpg_dm == 1) & (und_f.a1c_dm != 1)).astype(int)
    und_f["missed_by_a1c"] = (und_f.a1c_dm != 1).astype(int)
    und_f["missed_by_fpg"] = (und_f.fpg_dm != 1).astype(int)
    t = [compare(src, sec, "undiagnosed diabetes: HbA1c alone would miss", und_f, "missed_by_a1c", w="w2"),
         compare(src, sec, "undiagnosed diabetes: fasting glucose alone would miss", und_f, "missed_by_fpg", w="w2")]
    show("NHANES C. WHICH TEST FINDS UNDIAGNOSED DIABETES (2017-20, fasting subsample)", t)

    # ---------------------------------------------------------- D prediabetes
    sec = "Prediabetes"
    pre = nd.copy()
    pre["pre_a1c"] = pre.a1c.between(5.7, 6.4, inclusive="both").astype(float).where(pre.a1c.notna())
    p_any = pre[(pre.pre_a1c == 1) & (pre.a1c_dm != 1)].copy()
    p_any["aware"] = (p_any.DIQ160 == 1).astype(float).where(p_any.DIQ160.isin([1, 2]))
    t = [compare(src, sec, "no diabetes: HbA1c in prediabetes range (5.7-6.4)", pre[pre.a1c_dm != 1], "pre_a1c"),
         compare(src, sec, "HbA1c prediabetes: told they have prediabetes", p_any, "aware"),
         compare(src, sec, "  by race", p_any, "aware", by="race")]
    fsn = fs[(fs.told_dm == 0) & (fs.a1c < 6.5) & (fs.fpg < 126)].copy()
    fsn["ifg_ada_100"] = (fsn.fpg >= 100).astype(int)
    fsn["ifg_who_110"] = (fsn.fpg >= 110).astype(int)
    fsn["pre_a1c"] = (fsn.a1c >= 5.7).astype(int)
    fsn["pre_by_fpg_only"] = ((fsn.fpg >= 100) & (fsn.a1c < 5.7)).astype(int)
    fsn["pre_by_a1c_only"] = ((fsn.a1c >= 5.7) & (fsn.fpg < 100)).astype(int)
    t += [compare(src, sec, "fasting glucose 100-125 (ADA cut-off)", fsn, "ifg_ada_100", w="w2"),
          compare(src, sec, "fasting glucose 110-125 (WHO cut-off)", fsn, "ifg_who_110", w="w2"),
          compare(src, sec, "HbA1c 5.7-6.4", fsn, "pre_a1c", w="w2"),
          compare(src, sec, "prediabetes by fasting glucose ONLY (HbA1c normal)", fsn, "pre_by_fpg_only", w="w2"),
          compare(src, sec, "prediabetes by HbA1c ONLY (fasting glucose normal)", fsn, "pre_by_a1c_only", w="w2")]
    a = [adjusted(src, sec, "HbA1c prediabetes: told prediabetes", p_any, "aware", cov)]
    show("NHANES D. PREDIABETES", t, a)

    # ---------------------------------------------------------- E BMI
    sec = "BMI"
    print("\n" + "=" * 110 + "\nNHANES E. BMI AND DIABETES\n" + "=" * 110)
    rows = []
    for name, x in {"diagnosed diabetes": d[d.told_dm == 1], "undiagnosed diabetes": tot[tot.undiagnosed == 1],
                    "no diabetes": d[(d.told_dm == 0) & (d.a1c_dm == 0)]}.items():
        for s, lab in ((1, "women"), (0, "men")):
            g = x[(x.female == s) & x.bmi.notna() & (x.w > 0)]
            rows.append({"source": src, "section": sec, "check": f"mean BMI: {name}", "stratum": lab,
                         "value": round(np.average(g.bmi, weights=g.w), 1), "n": len(g),
                         "pct_BMI_under_30": round(100 * np.average(g.bmi < 30, weights=g.w), 1),
                         "pct_BMI_under_25": round(100 * np.average(g.bmi < 25, weights=g.w), 1)})
    dx = d[(d.told_dm == 1) & d.DID040.between(1, 90)]
    for s, lab in ((1, "women"), (0, "men")):
        g = dx[dx.female == s]
        rows.append({"source": src, "section": sec, "check": "mean age when told diabetes", "stratum": lab,
                     "value": round(np.average(g.DID040, weights=g.w), 1), "n": len(g)})
    print(pd.DataFrame(rows).drop(columns=["source", "section"]).to_string(index=False))
    RESULTS.extend(rows)
    d["has_dm"] = ((d.told_dm == 1) | (d.a1c_dm == 1)).astype(int)
    t = [compare(src, sec, "diabetes (told or HbA1c>=6.5), by BMI", d[d.a1c.notna()], "has_dm", by="bmi_band")]
    bmi_a = d[d.a1c.notna() & d.bmi.notna()].copy()
    rows = []
    for s, lab in ((1, "women"), (0, "men")):
        g = bmi_a[bmi_a.female == s]
        r = sm.GLM.from_formula("has_dm ~ age + bmi + C(race)", g, family=sm.families.Binomial(),
                                var_weights=g.w / g.w.mean()).fit()
        # BMI at which a 50-year-old white adult reaches the same diabetes risk as at BMI 30 for men
        rows.append((lab, r))
    rm = rows[1][1]
    base = pd.DataFrame({"age": [50], "bmi": [30], "race": ["white"]})
    target = float(rm.predict(base).iloc[0])
    rw = rows[0][1]
    grid = pd.DataFrame({"age": 50, "bmi": np.arange(18, 50, 0.1), "race": "white"})
    pw = rw.predict(grid).values
    eq = float(grid.bmi.iloc[np.argmin(np.abs(pw - target))])
    msg = (f"Diabetes risk for a 50-year-old white MAN at BMI 30 = {100*target:.1f}%. "
           f"A 50-year-old white WOMAN reaches the same risk at BMI ~{eq:.1f}.")
    print(msg)
    RESULTS.append({"source": src, "section": sec, "check": "equal-risk BMI (age 50, white)", "stratum": "women vs men",
                    "value": eq, "note": msg})
    show("", t)

    # ---------------------------------------------------------- F gestational diabetes
    sec = "Gestational diabetes"
    wmn = d[(d.female == 1) & (d.RHQ131 == 1) & (d.told_dm == 0)].copy()   # ever pregnant, no diabetes dx
    wmn["gdm_history"] = (wmn.RHQ162 == 1).astype(int)
    wmn["a1c_pre_or_dm"] = (wmn.a1c >= 5.7).astype(float).where(wmn.a1c.notna())
    wmn["a1c_dm_undx"] = (wmn.a1c >= 6.5).astype(float).where(wmn.a1c.notna())
    wmn["aware_pre"] = (wmn.DIQ160 == 1).astype(float).where(wmn.DIQ160.isin([1, 2]))
    rows = []
    for col, lab in [("tested_3yr", "blood tested for diabetes in past 3 yrs"),
                     ("a1c_pre_or_dm", "HbA1c >= 5.7 today"), ("a1c_dm_undx", "HbA1c >= 6.5 today (undiagnosed)"),
                     ("aware_pre", "told they have prediabetes")]:
        for g, x in {"past gestational diabetes": wmn[wmn.gdm_history == 1],
                     "no gestational diabetes": wmn[wmn.gdm_history == 0]}.items():
            for agebin, xx in {"all ages": x, "age 20-49": x[x.age < 50]}.items():
                p, n, y = wpct(xx, col)
                rows.append({"source": src, "section": sec, "check": lab, "stratum": f"{g}, {agebin}",
                             "value": round(p, 1), "n": n, "yes": y})
    print("\n" + "=" * 110 + "\nNHANES F. HISTORY OF GESTATIONAL DIABETES (women ever pregnant, no diabetes diagnosis)\n" + "=" * 110)
    print(pd.DataFrame(rows).drop(columns=["source", "section"]).to_string(index=False))
    RESULTS.extend(rows)
    gd = wmn[wmn.gdm_history == 1]
    gd_pre = gd[(gd.a1c >= 5.7)]
    print(f"Women with past gestational diabetes and HbA1c >= 5.7 today: {len(gd_pre)}; "
          f"of these, told prediabetes/diabetes risk: {int((gd_pre.DIQ160 == 1).sum())}")
    wmn_cov = ["age", "C(race)", "C(educ)", "pir", "insured", "usual_place", "visits_2plus", "bmi"]
    wmn2 = wmn.copy(); wmn2["female"] = wmn2.gdm_history        # reuse helper: 'female' = GDM history here
    r = adjusted(src, sec, "past GDM vs none: tested in 3 yrs (OR = past GDM)", wmn2, "tested_3yr", wmn_cov)
    print("  adjusted:", r)

    # ---------------------------------------------------------- G after diagnosis
    sec = "After diagnosis"
    dg = d[d.told_dm == 1].copy()
    dg["insulin"] = (dg.DIQ050 == 1).astype(float).where(dg.DIQ050.isin([1, 2]))
    dg["pills"] = (dg.DIQ070 == 1).astype(float).where(dg.DIQ070.isin([1, 2]))
    dg["any_med"] = ((dg.insulin == 1) | (dg.pills == 1)).astype(int)
    dg["a1c_under_7"] = (dg.a1c < 7).astype(float).where(dg.a1c.notna())
    dg["a1c_9_plus"] = (dg.a1c >= 9).astype(float).where(dg.a1c.notna())
    dg["a1c_checked_yr"] = (dg.DIQ275 == 1).astype(float).where(dg.DIQ275.isin([1, 2]))
    dg["eye_exam_yr"] = dg.DIQ360.isin([1, 2]).astype(float).where(dg.DIQ360.isin([1, 2, 3, 4, 5]))
    dg["foot_check_yr"] = (dg.DID350 >= 1).astype(float).where(dg.DID350.between(0, 300))
    dg["saw_doctor_for_dm"] = (dg.DID341 >= 1).astype(float).where(dg.DID341.between(0, 300))
    outs = ["any_med", "pills", "insulin", "a1c_checked_yr", "eye_exam_yr", "foot_check_yr", "saw_doctor_for_dm",
            "a1c_under_7", "a1c_9_plus"]
    t = [compare(src, sec, f"diagnosed diabetes: {o}", dg, o) for o in outs]
    a = [adjusted(src, sec, f"diagnosed: {o}", dg, o, cov) for o in outs]
    show("NHANES G. AFTER DIAGNOSIS", t, a)

    # ---------------------------------------------------------- H OGTT (2015-16)
    src2 = "NHANES 2015-16 OGTT"
    sec = "Which test"
    o = nhanes_base("I")
    o = o[(o.told_dm == 0) & o.LBXGLT.notna() & (o.WTSOG2YR > 0)].copy()
    o["w"] = o.WTSOG2YR
    o["ogtt_dm"] = (o.LBXGLT >= 200).astype(int)
    o["ogtt_pre"] = o.LBXGLT.between(140, 199).astype(int)
    o["fpg_dm"] = (o.fpg >= 126).astype(int)
    o["a1c_dm"] = (o.a1c >= 6.5).astype(int)
    o["any_dm"] = o[["ogtt_dm", "fpg_dm", "a1c_dm"]].max(axis=1)
    o["ifg"] = o.fpg.between(100, 125).astype(int)
    o["a1c_pre"] = o.a1c.between(5.7, 6.4).astype(int)
    t = [compare(src2, sec, "undiagnosed diabetes by 2-hour OGTT (>=200)", o, "ogtt_dm"),
         compare(src2, sec, "undiagnosed diabetes by fasting glucose (>=126)", o, "fpg_dm"),
         compare(src2, sec, "undiagnosed diabetes by HbA1c (>=6.5)", o, "a1c_dm"),
         compare(src2, sec, "undiagnosed diabetes by ANY test", o, "any_dm")]
    dmx = o[o.any_dm == 1].copy()
    dmx["missed_without_ogtt"] = ((dmx.ogtt_dm == 1) & (dmx.fpg_dm == 0) & (dmx.a1c_dm == 0)).astype(int)
    dmx["missed_by_fpg"] = (dmx.fpg_dm == 0).astype(int)
    dmx["missed_by_a1c"] = (dmx.a1c_dm == 0).astype(int)
    t += [compare(src2, sec, "of those with undiagnosed diabetes: found ONLY by OGTT", dmx, "missed_without_ogtt"),
          compare(src2, sec, "of those with undiagnosed diabetes: fasting glucose misses", dmx, "missed_by_fpg"),
          compare(src2, sec, "of those with undiagnosed diabetes: HbA1c misses", dmx, "missed_by_a1c"),
          compare(src2, sec, "impaired glucose tolerance (2-h 140-199)", o, "ogtt_pre"),
          compare(src2, sec, "impaired fasting glucose (100-125)", o, "ifg"),
          compare(src2, sec, "HbA1c prediabetes (5.7-6.4)", o, "a1c_pre")]
    show("NHANES H. ORAL GLUCOSE TOLERANCE TEST (2015-16): WHICH TEST MISSES WOMEN?", t)


if __name__ == "__main__":
    which = sys.argv[1:] or ["namcs", "nhanes"]
    if "namcs" in which:
        namcs()
    if "nhanes" in which:
        nhanes()
    out = pd.DataFrame(RESULTS)
    out.to_csv(f"{OUT}/all_results.csv", index=False)
    for sec, g in out.groupby("section"):
        g.to_csv(f"{OUT}/{sec.lower().replace(' ', '_')}.csv", index=False)
    print(f"\nSaved {len(out)} rows to {OUT}/all_results.csv (+ one CSV per section)")
