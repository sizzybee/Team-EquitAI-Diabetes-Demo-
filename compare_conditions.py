"""
Run the NHAMCS sex-gap audit for every condition in nhamcs_filter.COHORTS
and put the results side by side.

Usage:
    .venv/Scripts/python.exe compare_conditions.py data/nhamcs2022.pkl

Outputs:
    nhamcs_all_conditions.csv    every outcome x condition (full table)
    nhamcs_top_gaps.csv          reliable outcomes, largest F-M gaps first
"""

import sys
import contextlib
import io
import pandas as pd
import nhamcs_filter as nf

OUTCOME_PREFIXES = ("triage_", "test_", "med_", "next_", "dx_")
MIN_RATE = 2.0   # skip outcomes that are near-zero for both sexes (e.g. MRI for chest pain)


def run(path):
    df = nf.load(path)
    tables, sizes = [], []
    for name in nf.COHORTS:
        with contextlib.redirect_stdout(io.StringIO()):   # keep the per-cohort chatter quiet
            c = nf.build_cohort(df, name)
        outcomes = [x for x in c.columns if x.startswith(OUTCOME_PREFIXES) and x != "triage_level"]
        t = nf.compare_by_sex(c, outcomes, name).rename(columns={"group": "condition"})
        tables.append(t)
        sizes.append({"condition": name,
                      "women": int(c["sex"].eq("Female").sum()),
                      "men": int(c["sex"].eq("Male").sum()),
                      "median_wait_F": c.loc[c.sex.eq("Female"), "wait_minutes"].median(),
                      "median_wait_M": c.loc[c.sex.eq("Male"), "wait_minutes"].median()})
    full = pd.concat(tables, ignore_index=True)
    full = full[(full["Female_wtd_%"] >= MIN_RATE) | (full["Male_wtd_%"] >= MIN_RATE)]

    top = (full[full["reliable"]]
           .assign(abs_gap=lambda d: d["gap_F-M"].abs())
           .sort_values("abs_gap", ascending=False)
           .drop(columns="abs_gap"))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print("\nCohort sizes:")
    print(pd.DataFrame(sizes).to_string(index=False))

    cols = ["condition", "outcome", "Female_yes", "Female_wtd_%", "Male_yes", "Male_wtd_%", "gap_F-M"]
    print("\nLargest reliable gaps (>= 30 records per sex), all conditions:")
    print(top[cols].head(30).to_string(index=False))

    dx = full[full["outcome"].str.startswith("dx_")]
    print("\nDiagnoses by condition (reliable column shows whether n >= 30 per sex):")
    print(dx[cols + ["reliable"]].to_string(index=False))

    full.to_csv("nhamcs_all_conditions.csv", index=False)
    top.to_csv("nhamcs_top_gaps.csv", index=False)
    print("\nSaved nhamcs_all_conditions.csv and nhamcs_top_gaps.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python compare_conditions.py <nhamcs2022.pkl>")
    run(sys.argv[1])
