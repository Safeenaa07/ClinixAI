"""Bias check: performance by sex and age group.
CSV columns: label,score,sex,age   (label 0/1, score = model probability for ONE finding)
Reports n, AUC, sensitivity and specificity (threshold 0.5) per group and the largest gap.
Usage: python evaluation/subgroup_metrics.py predictions.csv [threshold]"""
import csv
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

rows = list(csv.DictReader(open(sys.argv[1])))
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


def age_bin(a):
    a = float(a)
    return "age 0-40" if a <= 40 else "age 41-60" if a <= 60 else "age 61+"


def stats(sub):
    y = np.array([int(r["label"]) for r in sub]); s = np.array([float(r["score"]) for r in sub])
    pred = s >= thr
    auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
    sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1) if (y == 1).any() else float("nan")
    spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1) if (y == 0).any() else float("nan")
    return len(sub), auc, sens, spec


groups = {}
for r in rows:
    groups.setdefault("sex " + r["sex"].strip().upper(), []).append(r)
    groups.setdefault(age_bin(r["age"]), []).append(r)
print(f"{'group':12s} {'n':>6s} {'AUC':>7s} {'sens':>7s} {'spec':>7s}")
res = {}
for g in sorted(groups):
    res[g] = stats(groups[g])
    n, a, se, sp = res[g]
    print(f"{g:12s} {n:6d} {a:7.3f} {se:7.3f} {sp:7.3f}")
for fam in ("sex", "age"):
    keys = [g for g in res if g.startswith(fam)]
    for i, name in ((1, "AUC"), (2, "sensitivity")):
        v = [res[k][i] for k in keys if not np.isnan(res[k][i])]
        if len(v) > 1:
            print(f"largest {fam} gap in {name}: {max(v) - min(v):.3f}")
print("Small groups give noisy numbers. Report n with every result.")
