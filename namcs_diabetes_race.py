"""
Diabetes sex gap by race/ethnicity -- NAMCS 2019 office visits (adults).

Usage:
    .venv/Scripts/python.exe namcs_diabetes_race.py data/namcs2019.pkl

Race groups (NAMCS has no South Asian / East Asian split):
  hispanic   ETHIM = 1 (any race)
  white      non-Hispanic, RACERETH = 1
  black      non-Hispanic, RACERETH = 2
  asian      non-Hispanic, RACEUN = 3 (race as REPORTED; NCHS imputes race for
             the ~26% of visits where it was left blank, and the imputed file
             folds Asian patients into "other", so Asian here = reported only)
  other      everyone else (NHOPI, AIAN, multiple, imputed-other)

Checks, each split by sex within race:
  1. Screening  -- adults 35-70 without diabetes: HbA1c/glucose in past year
     1b. same, using the overweight rule from US guidance (BMI >= 25, or
         BMI >= 23 for Asian patients, per ADA)
  2. Unrecognised -- labs in diabetes range (A1c >= 6.5 or FBG >= 126) but
     diabetes not in the chart / not diagnosed at this visit
  3. Diabetes recorded among adults 35-70 (how often the chart says diabetes)
Then the adjusted female odds ratio within each race, and a test of whether
the sex gap differs by race (female x race interaction).
"""

import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import nhamcs_filter as nf
import focus_analyses as fa
import namcs_focus as nm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

RACES = ["white", "black", "hispanic", "asian", "other"]


def add_race(d):
    hisp = nf.num(d, "ETHIM").eq(1)
    rr = nf.num(d, "RACERETH")
    ru = nf.num(d, "RACEUN")
    d["race5"] = np.select([hisp, rr.eq(1) & ~ru.eq(3), rr.eq(2), ru.eq(3)],
                           ["hispanic", "white", "black", "asian"], default="other")
    d["group"] = d["race5"] + " " + d["sex"].str.lower()
    return d


def by_race(x, outcome, label):
    t = fa.raw(x, outcome, "race5", label)
    t["stratum"] = pd.Categorical(t["stratum"], RACES)
    return t.sort_values("stratum")


def adjusted_by_race(x, outcome, label, extra=()):
    rows = []
    for r in RACES:
        res = fa.adjusted(x[x.race5 == r], outcome, list(extra), f"{label} | {r}",
                          drop=["diabetes", "C(race)"])
        rows.append(res)
    return rows


def interaction_test(x, outcome):
    """Does the female effect differ by race? Likelihood-ratio-style Wald test."""
    covars = [c for c in fa.BASE_COVARS if c not in ("diabetes", "C(race)")]
    x = x.dropna(subset=[outcome]).copy()
    x[outcome] = x[outcome].astype(float)
    if "YEAR" in x.columns and x["YEAR"].nunique() > 1:
        covars = covars + ["C(YEAR)"]
    f = f"{outcome} ~ female * C(race5) + " + " + ".join(covars)
    try:
        r = sm.GLM.from_formula(f, x, family=sm.families.Binomial(),
                                var_weights=x["w"] / x["w"].mean()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(x["CPSUM"])[0]})
        names = [n for n in r.params.index if n.startswith("female:")]
        w = r.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
        return round(float(w.pvalue), 3)
    except Exception as e:
        return f"failed: {str(e)[:50]}"


def main(path):
    d = add_race(nm.prep(nf.load(path)))
    labs = d[d.lab_section]
    print(f"Adults: {len(d):,} | with lab section: {len(labs):,}")
    print("\nAdults with lab section, by race and sex (records):")
    print(pd.crosstab(labs.race5, labs.sex).reindex(RACES).to_string())

    age_ok = labs.age.between(35, 70)
    elig = labs[age_ok & (labs.diabetes == 0)].copy()
    bmi_cut = np.where(elig.race5.eq("asian"), 23, 25)
    elig_bmi = elig[elig.bmi >= bmi_cut]
    hi = labs[labs.lab_diabetes_range == 1].assign(not_in_chart=lambda x: 1 - x.diabetes)
    adults = labs[age_ok]

    tables = [
        by_race(elig, "screened", "1. no DM, 35-70: A1c/glucose in past yr"),
        by_race(elig_bmi, "screened", "1b. no DM, 35-70, overweight*: screened"),
        by_race(hi, "not_in_chart", "2. labs in DM range, NOT in chart"),
        by_race(adults, "diabetes", "3. 35-70: diabetes recorded"),
    ]
    out = pd.concat(tables, ignore_index=True)
    print("\n" + out.to_string(index=False))
    print("  * overweight = BMI >= 25, or >= 23 for Asian patients (ADA)")

    adj = (adjusted_by_race(elig, "screened", "screened (35-70, no DM)")
           + adjusted_by_race(elig_bmi, "screened", "screened (overweight)", ["bmi"]))
    adj = pd.DataFrame(adj)
    print("\nAdjusted female OR within each race (OR < 1 = women less likely):")
    print(adj.to_string(index=False))

    p1 = interaction_test(elig, "screened")
    print(f"\nDoes the sex gap in screening differ by race? interaction p = {p1}")

    # Women of each group vs white men -- the comparison a model would 'see'
    print("\nScreening rate (35-70, no DM) for each race x sex group, weighted %:")
    g = (elig.groupby("group").apply(lambda x: pd.Series({
        "records": len(x), "screened": int(x.screened.sum()),
        "wtd_%": round(100 * (x.w * x.screened).sum() / x.w.sum(), 1)})))
    print(g.sort_values("wtd_%").to_string())

    out.to_csv("namcs_diabetes_by_race_raw.csv", index=False)
    adj.to_csv("namcs_diabetes_by_race_adjusted.csv", index=False)
    g.to_csv("namcs_diabetes_screening_by_group.csv")
    print("\nSaved namcs_diabetes_by_race_raw.csv, namcs_diabetes_by_race_adjusted.csv, "
          "namcs_diabetes_screening_by_group.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python namcs_diabetes_race.py <namcs2019.pkl>")
    main(sys.argv[1])
