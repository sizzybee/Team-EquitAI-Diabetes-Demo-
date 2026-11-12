# Catching diagnostic label gaps before AI training

**Worked example:** type 2 diabetes in women, using US national survey data (NAMCS 2018–19, checked against NHANES).

A diagnosis in a health record shows who was tested and charted, not only who is sick. When a group is tested less, or tested with a test that misses them, their records say "healthy" when the truth is "unknown" or "sick". A model trained on those records learns the gap as fact. This repo is a datacard that travels with the dataset, like the leaflet inside a medicine box. It tells AI builders, as they work with the data, where records can be trusted, where they can’t, and what to check before release.

---

## The three tangibles

| # | Tangible | File | What it is |
|---|---|---|---|
| 1 | **Fill-in template** | [`datacard_TEMPLATE.md`](datacard_TEMPLATE.md) | The rules on their own, ready to reuse: when to accept a chart diagnosis, how to treat missing tests, feature rules, leakage and checks before release. |
| 2 | **Decision tree** | [`diabetes_label_decision_tree.png`](diabetes_label_decision_tree.png) ([SVG](diabetes_label_decision_tree.svg)) | One page showing how each record labelled "no diabetes" gets a new label: SET_ASIDE, UNKNOWN, DIABETES_TREATED, PROBABLE / CONFIRMED, BORDERLINE, PARTLY_TESTED or HEALTHY. |
| 3 | **Worked example** | [`NAMCS_2018_2019_diabetes_datacard.md`](NAMCS_2018_2019_diabetes_datacard.md) | The template applied to NAMCS 2018–19, with the evidence behind each rule (sex-gap audit, model run, refit). Each number is tagged **[Audit]**, **[Blind]**, **[Refit]** or **[Drug check]**, which map to the scripts below. |

**Suggested reading order:** 2, then 3, then 1. The tree gives the idea in a minute, the worked example shows that the problem is real, and the template is what another team would take away.

---

## Code

All scripts are in [`code/`](code/). The tag in brackets matches the source tags in the worked example.

### Scripts behind the worked example

| Script | Tag | What it does |
|---|---|---|
| `diabetes_audit.py` | [Audit] | The full type 2 diabetes audit in NAMCS 2018–19 (15,885 adult visits), with NHANES 2017–March 2020 and 2015–16 as an outside check. Covers 12 areas, including screening, labs in the diabetes range but not in the chart, which test was used, prescriptions and missed chances. The source of most numbers in the worked example. Writes to `results_diabetes/`. |
| `audit_by_race.py` | [Audit] | Sex gaps split by race for every condition, in both NAMCS (office visits) and NHAMCS (emergency departments). The source of the NHAMCS D-dimer figure. |
| `model_predictions.py` | [Blind] | The blind-run models (diabetes, testing, counselling and medicines). **Written by a separate AI session that never saw the data card**, so it shows what a model learns from the original labels. |
| `diabetes_underprediction.py` | [Blind] | How often the blind-run models score people with diabetes-range labs below 50%, in NAMCS and NHANES. Found the fasting-glucose-only and lean-women patterns. |
| `refit_no_meds.py` | [Refit] | Refits the blind-run models without diabetes-medicine fields. Gives the sex odds ratio of 0.81 (95% CI 0.68–0.98), the female × BMI under 25 interaction, and pooled NHANES miss rates. Writes to `results_refit/`. |
| `verify_drug_classes.py` | [Drug check] | Lists the five most common drug names under each drug class code. Confirmed all seven diabetes-related codes, and that class 442 holds TNF blockers, not SGLT2 drugs. |

### Helper scripts (used by the scripts above)

| Script | What it does |
|---|---|
| `namcs_filter.py` | Filters office visits (NAMCS) by symptom and compares women and men. Reused by other scripts as a helper. |
| `namcs_focus.py` | Office-visit checks for diabetes (chart vs HbA1c and glucose values) and palpitations. Also holds shared helpers, such as the medication lookups. |
| `nhamcs_filter.py` | Filters emergency department visits (NHAMCS) by symptom and compares women and men on triage, waits, tests, medicines, disposition and diagnosis. |

### How we chose diabetes (exploration)

Before narrowing to diabetes, we screened several conditions for sex gaps. These scripts are in [`code/exploration/`](code/exploration/).

| Script | What it does |
|---|---|
| `compare_conditions.py` | Runs the emergency department comparison on five conditions side by side, ranking the largest female–male gaps first. |
| `focus_analyses.py` | Five focused emergency department checks: pain medication, triage, diabetes, palpitations and brain imaging (e.g. same pain score, same pain relief?). |
| `matched_presentations.py` | Compares women and men only within identical presentation profiles (reason for visit × age × vitals), to answer "same presentation, different care?". |
| `namcs_diabetes_race.py` | The diabetes checks split by sex within each race group. Found that NAMCS has only one Asian race code. |

---

## Data

All data is public. It is not included in this repo; download it from the official sources:

- **NAMCS 2018 and 2019** (office visits) and **NHAMCS** (hospital emergency departments): CDC / National Center for Health Statistics, Ambulatory Health Care Data – <https://www.cdc.gov/nchs/ahcd/datasets_documentation_related.htm>
- **NHANES 2017–March 2020 pre-pandemic** and **NHANES 2015–16**: CDC / NCHS, National Health and Nutrition Examination Survey – <https://wwwn.cdc.gov/nchs/nhanes/Default.aspx>

See section 2 of the worked example for important notes on these files: one row is one visit, survey weights must be halved when pooling years, and lab fields are only collected for primary-care visits.

---

## Limits

The data can't show diagnostic delay or follow-up, can't separate South Asian from East Asian patients, and has only a binary sex field. It is US data only. The full list is in section 8 of the worked example.
