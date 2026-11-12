"""Verify Multum drug classes in NAMCS 2018-19: top drug names per class.
Names from NHANES prescription files (P_RXQ_RX, RXQ_RX_I) in data/multum/.
Usage: .venv/Scripts/python.exe verify_drug_classes.py"""
import pandas as pd, collections
names = {}
for f in ["data/multum/P_RXQ_RX.xpt", "data/multum/RXQ_RX_I.xpt"]:
    r = pd.read_sas(f, format="xport", encoding="latin1")
    r = r.dropna(subset=["RXDDRGID"])
    names.update(dict(zip(r.RXDDRGID.str.strip(), r.RXDDRUG.str.strip())))
d = pd.read_pickle("data/namcs_2018_2019.pkl")
d = d[pd.to_numeric(d.AGE, errors="coerce") >= 18]
cats = ["173","213","214","215","371","373","458","442","314"]
cnt = {c: collections.Counter() for c in cats}
for i in range(1, 31):
    if f"DRUGID{i}" not in d.columns: break
    cols = [f"RX{i}V{j}C{k}" for j in (1,2,3) for k in range(1,5) if f"RX{i}V{j}C{k}" in d.columns]
    sub = d[cols].astype(str)
    for c in cats:
        m = sub.isin([c]).any(axis=1)
        cnt[c].update(d.loc[m, f"DRUGID{i}"].astype(str))
for c in cats:
    tot = sum(cnt[c].values())
    print(f"\n== class {c}: {tot} mentions, {len(cnt[c])} distinct drugs")
    for k, v in cnt[c].most_common(5):
        print(f"   {k:8s} {names.get(k, '(not in NHANES lookup)'):45s} {v}")
