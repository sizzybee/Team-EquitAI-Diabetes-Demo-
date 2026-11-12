"""
Same presentation, different care?  Matched ("twin") comparison of women vs men.

Usage:
    .venv/Scripts/python.exe matched_presentations.py            # ED scan + heart attack + office scan
    .venv/Scripts/python.exe matched_presentations.py ed mi      # pick sections: ed, mi, office

How the matching works (coarsened exact matching):
  Every visit gets a "presentation profile". In the ED (NHAMCS 2022):
      main reason for visit  x  age band  x  abnormal vital signs (yes/no)
      x  pain score band  x  arrived by ambulance (yes/no)
  In the office (NAMCS 2019):
      main reason for visit  x  age band  x  new vs chronic problem  x  seen before
  Women and men are only compared INSIDE the same profile. Profiles that contain
  only one sex are dropped. The matched gap is the average of (women% - men%)
  across profiles, weighted by how many women (survey-weighted) are in each --
  i.e. "for the women who came in, how different would their care have been if
  men with the same presentation had been treated the same way?"

  95% CIs come from resampling survey PSUs (CPSUM) 300 times.
  Negative gap = women received LESS of that outcome than matched men.

Triage, tests, consults, admission and diagnosis are outcomes -- never part of
the profile -- because they are what we are checking.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import nhamcs_filter as nf
import focus_analyses as fa
import namcs_focus as nm
from namcs_diabetes_race import add_race

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
RNG = np.random.default_rng(2026)
N_BOOT = 1000

ED_SYMPTOMS = {   # main reason for visit (RFV13D); sex-specific complaints left out on purpose
    1545: "abdominal pain", 1050: "chest pain", 1415: "shortness of breath", 1210: "headache",
    1055: "pain, site unspecified", 1905: "back pain", 1910: "low back pain", 1440: "cough",
    1920: "leg symptoms", 1225: "dizziness", 1455: "throat symptoms", 1020: "general weakness",
    1525: "nausea", 1530: "vomiting", 1925: "knee symptoms", 1010: "fever", 1935: "foot/toe",
    1940: "shoulder", 1260: "palpitations", 1030: "fainting", 1220: "numbness/tingling",
    1230: "neuro weakness", 1945: "arm symptoms", 1015: "tiredness",
}
OFFICE_SYMPTOMS = {
    1545: "abdominal pain", 1050: "chest pain", 1415: "shortness of breath", 1210: "headache",
    1055: "pain, site unspecified", 1905: "back pain", 1910: "low back pain", 1440: "cough",
    1920: "leg symptoms", 1225: "dizziness", 1455: "throat symptoms", 1925: "knee symptoms",
    1940: "shoulder", 1915: "hip", 1900: "neck", 1860: "skin rash", 1305: "vision",
    1015: "tiredness", 1260: "palpitations", 1220: "numbness/tingling", 1400: "nasal congestion",
    1355: "earache",
}


# --------------------------------------------------------------- engine ---
def _gap_arrays(sid, fem, w, y, n_strata):
    """Women-weighted average of within-stratum (F% - M%), vectorised."""
    cell = sid * 2 + fem
    W = np.bincount(cell, weights=w, minlength=2 * n_strata).reshape(n_strata, 2)
    WY = np.bincount(cell, weights=w * y, minlength=2 * n_strata).reshape(n_strata, 2)
    ok = (W[:, 0] > 0) & (W[:, 1] > 0)
    if not ok.any():
        return np.nan, np.nan, np.nan
    pm, pf, wf = WY[ok, 0] / W[ok, 0], WY[ok, 1] / W[ok, 1], W[ok, 1]
    return (100 * (wf * (pf - pm)).sum() / wf.sum(),
            100 * (wf * pf).sum() / wf.sum(), 100 * (wf * pm).sum() / wf.sum())


def matched(df, strata, outcome, boot=N_BOOT):
    x = df[df[outcome].notna()]
    if x[outcome].sum() < 10:
        return None
    sid, uniq = pd.factorize(pd.MultiIndex.from_frame(x[strata].astype(str)))
    n_strata = len(uniq)
    fem = x["female"].values.astype(int)
    w = x["w"].values.astype(float)
    y = x[outcome].values.astype(float)
    gap, pf, pm = _gap_arrays(sid, fem, w, y, n_strata)
    has_f = np.bincount(sid, weights=fem, minlength=n_strata) > 0
    has_m = np.bincount(sid, weights=1 - fem, minlength=n_strata) > 0
    both = (has_f & has_m)[sid]
    nF, nM = int((both & (fem == 1)).sum()), int((both & (fem == 0)).sum())
    yF, yM = int(y[both & (fem == 1)].sum()), int(y[both & (fem == 0)].sum())
    # PSU bootstrap: resample PSUs, multiply each visit's weight by how often its PSU was drawn
    psu, psu_u = pd.factorize(x["CPSUM"])
    boots = []
    for _ in range(boot):
        draws = np.bincount(RNG.integers(0, len(psu_u), len(psu_u)), minlength=len(psu_u))
        b = _gap_arrays(sid, fem, w * draws[psu], y, n_strata)[0]
        if pd.notna(b):
            boots.append(b)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 50 else (np.nan, np.nan))
    return {"outcome": outcome, "F_matched": nF, "M_matched": nM, "F_yes": yF, "M_yes": yM,
            "F%": round(pf, 1), "M%_same_profile": round(pm, 1), "gap": round(gap, 1),
            "CI": f"[{lo:+.1f}, {hi:+.1f}]", "sig": bool(pd.notna(lo) and (lo > 0 or hi < 0)),
            "ok_n": min(yF, yM) >= 30}


def run_block(title, df, strata, outcomes, boot=N_BOOT):
    rows = []
    for o in outcomes:
        r = matched(df, strata, o, boot)
        if r:
            rows.append(r)
    t = pd.DataFrame(rows)
    print(f"\n--- {title}  ({int(df.female.sum())} women, {int((1 - df.female).sum())} men before matching)")
    if not t.empty:
        print(t.to_string(index=False))
    return t.assign(block=title)


# ---------------------------------------------------------------- data ----
def ed_data():
    d = add_race(fa.prep(nf.load(os.environ.get("NHAMCS_PATH", "data/nhamcs2022.pkl"))))
    d = d[(d.left_early == 0) & (d.injury == 0)].copy()
    d["symptom"] = pd.to_numeric(d["RFV13D"], errors="coerce").map(ED_SYMPTOMS)
    d["pain_band"] = d["pain_band"].astype(str).replace("nan", "unknown")
    d["age10"] = pd.cut(d.age, [17, 34, 49, 64, 79, 120]).astype(str)
    for k, col in {"ecg": "EKG", "troponin": "CARDENZ", "cardiac_monitor": "CARDMON",
                   "ct_any": "CATSCAN", "xray": "XRAY", "ultrasound": "ULTRASND", "mri": "MRI",
                   "consult": "CONSULT"}.items():
        d[k] = nf.yes(d, col).astype(int)
    d["blood_tests"] = nf.yes(d, ["CBC", "BMP", "CMP"]).astype(int)
    d["any_imaging"] = nf.yes(d, "ANYIMAGE").astype(int)
    d["tests_5plus"] = (nf.num(d, "TOTDIAG") >= 5).where(nf.num(d, "TOTDIAG") >= 0)
    d["long_wait"] = (d.wait > 30).where(d.wait.notna())
    diag = d[nf.DIAG_COLS].apply(nf.clean_diag)
    real = diag.apply(lambda s: s.str.match(r"^[A-QS-Z]\d"))
    d["symptom_code_only"] = (diag["DIAG1"].str.startswith("R") & ~real.any(axis=1)).astype(int)
    d["dx_mi"] = fa.has_dx(d, ["I21", "I22"]).astype(int)
    d["dx_acs"] = fa.has_dx(d, ["I20", "I21", "I22", "I23", "I24"]).astype(int)
    d["dx_any_cardiac"] = fa.has_dx(d, ["I2", "I3", "I4", "I50", "I51"]).astype(int)
    d["dx_anxiety"] = fa.has_dx(d, ["F41"]).astype(int)
    d["pain_relief_if_pain4"] = d["med_pain_relief"].where(d.pain >= 4)
    return d


ED_STRATA = ["symptom", "age10", "abnormal_vitals", "pain_band", "ems"]
ED_OUTCOMES = ["triage_urgent", "long_wait", "blood_tests", "tests_5plus", "any_imaging", "ct_any",
               "ecg", "consult", "admitted", "symptom_code_only", "pain_relief_if_pain4"]


def office_data():
    d = add_race(nm.prep(nf.load(os.environ.get("NAMCS_PATH", "data/namcs2019.pkl"))))
    d = d[nf.num(d, "INJURY") != 1].copy()
    d["symptom"] = pd.to_numeric(d["RFV13D"], errors="coerce").map(OFFICE_SYMPTOMS)
    d["age10"] = pd.cut(d.age, [17, 34, 49, 64, 79, 120]).astype(str)
    d["problem"] = nf.num(d, "MAJOR").map({1: "new", 2: "chronic routine", 3: "chronic flare",
                                           4: "pre/post surgery", 5: "preventive", 6: "preventive"}).fillna("unknown")
    d["blood_tests"] = nf.yes(d, ["CBC", "BMP", "CMP"]).astype(int)
    d["imaging"] = nf.yes(d, ["XRAY", "CATSCAN", "MRI", "OTHULTRA", "ECHOCARD"]).astype(int)  # no mammo/bone density
    d["any_test_or_imaging"] = (d.blood_tests | d.imaging | d.ecg | nf.yes(d, ["URINE", "THYROID"])).astype(int)
    d["procedure_or_therapy"] = nf.yes(d, ["PT", "OCCUPY", "WOUND", "EXCISION", "BIOPSY"]).astype(int)
    d["sent_to_ed"] = nf.yes(d, "ERADMHOS").astype(int)
    d["age3"] = pd.cut(d.age, [17, 44, 64, 120]).astype(str)
    d["new_problem"] = d["problem"].eq("new").astype(int)
    return d


OFFICE_STRATA = ["symptom", "age3", "new_problem"]
OFFICE_OUTCOMES = ["any_test_or_imaging", "blood_tests", "imaging", "ecg", "referred_other_md",
                   "procedure_or_therapy", "dx_symptom_code_only", "dx_anxiety", "mental_health_referral",
                   "anxiolytic_listed"]


# ------------------------------------------------------------ sections ----
def section_ed():
    print("\n" + "#" * 110 + "\nED (NHAMCS 2022): SAME PRESENTATION, DIFFERENT CARE?\n" + "#" * 110)
    print("Profile = main reason x age band x abnormal vitals x pain band x ambulance arrival")
    d = ed_data()
    p = d[d.symptom.notna()]
    out = [run_block("ALL listed presentations pooled", p, ED_STRATA, ED_OUTCOMES)]
    for sym in ED_SYMPTOMS.values():
        s = p[p.symptom == sym]
        if s.female.sum() >= 60 and (1 - s.female).sum() >= 60:
            out.append(run_block(sym, s, ED_STRATA, ED_OUTCOMES, boot=1000))
    t = pd.concat(out, ignore_index=True)
    t.to_csv("matched_ed.csv", index=False)
    flag = t[(t.gap < 0) & t.sig]
    print("\n>>> ED: outcomes where women got significantly LESS than matched men:")
    print(flag[["block", "outcome", "F_yes", "M_yes", "F%", "M%_same_profile", "gap", "CI", "ok_n"]].to_string(index=False)
          if not flag.empty else "   none")
    flag2 = t[(t.gap > 0) & t.sig]
    print("\n>>> ED: outcomes where women got significantly MORE than matched men:")
    print(flag2[["block", "outcome", "F%", "M%_same_profile", "gap", "CI"]].to_string(index=False)
          if not flag2.empty else "   none")
    return d


def section_mi(d=None):
    print("\n" + "#" * 110 + "\nHEART ATTACK: same presentation -> are men more often diagnosed?\n" + "#" * 110)
    d = ed_data() if d is None else d
    cardiac_sx = [1050, 1415, 1260, 1020, 1525, 1030, 1945, 1015, 1225]
    cp = d[fa.has_rfv(d, [1050])]
    strata = ["age10", "abnormal_vitals", "pain_band", "ems"]
    out = [run_block("chest pain (any reason-for-visit slot)", cp, strata,
                     ["troponin", "ecg", "dx_mi", "dx_acs", "dx_any_cardiac", "dx_anxiety",
                      "symptom_code_only", "admitted"])]
    out.append(run_block("chest pain, age 50+", cp[cp.age >= 50], strata,
                         ["troponin", "dx_mi", "dx_acs", "dx_any_cardiac", "admitted"]))
    tested = cp[cp.troponin == 1]
    out.append(run_block("chest pain AND troponin ordered (same workup)", tested, strata,
                         ["dx_mi", "dx_acs", "dx_any_cardiac", "admitted"]))
    wide = d[fa.has_rfv(d, cardiac_sx)]
    wide = wide.assign(sx=np.where(fa.has_rfv(wide, [1050]), "chest pain", "other cardiac-type"))
    out.append(run_block("any heart-type symptom (chest pain, SOB, palpitations, weakness, nausea, "
                         "fainting, arm, tiredness, dizziness)", wide, ["sx"] + strata,
                         ["troponin", "ecg", "dx_mi", "dx_acs", "dx_any_cardiac", "admitted"]))
    # Among people who WERE diagnosed with MI: did they come in with the same complaints?
    mi = d[d.dx_mi == 1]
    print(f"\nDiagnosed MI visits: {len(mi)} ({int(mi.female.sum())} women, {int((1 - mi.female).sum())} men)."
          " How they presented (weighted %):")
    pres = {"chest pain": [1050], "shortness of breath": [1415], "weakness/tiredness": [1020, 1015],
            "nausea/vomiting": [1525, 1530], "dizziness/fainting": [1225, 1030], "arm/back/neck": [1945, 1905, 1900]}
    rows = []
    for k, c in pres.items():
        mi[f"p_{k}"] = fa.has_rfv(mi, c).astype(int)
        rows.append(fa.raw(mi, f"p_{k}", None, k))
    print(pd.concat(rows).drop(columns=["stratum"]).to_string(index=False))
    t = pd.concat(out, ignore_index=True)
    t.to_csv("matched_mi.csv", index=False)


def section_office():
    print("\n" + "#" * 110 + "\nOFFICE (NAMCS 2019): SAME PRESENTATION, DIFFERENT CARE?\n" + "#" * 110)
    print("Profile = main reason x age (18-44 / 45-64 / 65+) x new problem (yes/no)")
    d = office_data()
    p = d[d.symptom.notna()]
    out = [run_block("ALL listed presentations pooled", p, OFFICE_STRATA, OFFICE_OUTCOMES)]
    for sym in OFFICE_SYMPTOMS.values():
        s = p[p.symptom == sym]
        if s.female.sum() >= 40 and (1 - s.female).sum() >= 40:
            out.append(run_block(sym, s, OFFICE_STRATA, OFFICE_OUTCOMES, boot=1000))
    t = pd.concat(out, ignore_index=True)
    t.to_csv("matched_office.csv", index=False)
    flag = t[(t.gap < 0) & t.sig]
    print("\n>>> OFFICE: outcomes where women got significantly LESS than matched men:")
    print(flag[["block", "outcome", "F_yes", "M_yes", "F%", "M%_same_profile", "gap", "CI", "ok_n"]].to_string(index=False)
          if not flag.empty else "   none")
    flag2 = t[(t.gap > 0) & t.sig]
    print("\n>>> OFFICE: outcomes where women got significantly MORE than matched men:")
    print(flag2[["block", "outcome", "F%", "M%_same_profile", "gap", "CI"]].to_string(index=False)
          if not flag2.empty else "   none")


if __name__ == "__main__":
    which = sys.argv[1:] or ["ed", "mi", "office"]
    d = section_ed() if "ed" in which else None
    if "mi" in which:
        section_mi(d)
    if "office" in which:
        section_office()
    print("\nSaved matched_ed.csv, matched_mi.csv, matched_office.csv (whichever sections ran)")
