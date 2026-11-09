"""Doctor-only vs AI-only vs doctor+AI.
CSV columns: case_id,truth,reader_id,condition,answer,seconds[,ai_answer]
  condition: doctor_only | ai_only | doctor_ai     truth/answer/ai_answer: 0 or 1
  ai_only rows: use reader_id "AI". doctor_ai rows need ai_answer (what the AI suggested).
Prints accuracy, sensitivity, specificity, mean seconds per condition, over-trust
(reader followed a WRONG AI answer) and correction rate (reader fixed a wrong AI answer),
and an exact McNemar test for doctor_only vs doctor_ai on the same reader-case pairs.
Use real readers if you can. If you simulate readers, say so in the report.
Usage: python evaluation/reader_study.py study.csv"""
import csv
import sys

import numpy as np
from scipy.stats import binomtest

rows = list(csv.DictReader(open(sys.argv[1])))
if any(r.get("simulated") == "1" for r in rows):
    print("*** SIMULATED DATA: invented readers, not a clinical result. Label it that way in the report. ***\n")
by = {}
for r in rows:
    by.setdefault(r["condition"], []).append(r)
print(f"{'condition':12s} {'n':>5s} {'acc':>6s} {'sens':>6s} {'spec':>6s} {'sec':>6s}")
for c in ("doctor_only", "ai_only", "doctor_ai"):
    sub = by.get(c, [])
    if not sub:
        continue
    t = np.array([int(r["truth"]) for r in sub]); a = np.array([int(r["answer"]) for r in sub])
    sec = np.mean([float(r["seconds"]) for r in sub if r.get("seconds")] or [np.nan])
    sens = ((a == 1) & (t == 1)).sum() / max((t == 1).sum(), 1)
    spec = ((a == 0) & (t == 0)).sum() / max((t == 0).sum(), 1)
    print(f"{c:12s} {len(sub):5d} {(a == t).mean():6.3f} {sens:6.3f} {spec:6.3f} {sec:6.1f}")

da = by.get("doctor_ai", [])
wrong = [r for r in da if r.get("ai_answer") not in (None, "") and int(r["ai_answer"]) != int(r["truth"])]
if wrong:
    follow = np.mean([int(r["answer"]) == int(r["ai_answer"]) for r in wrong])
    fixed = np.mean([int(r["answer"]) == int(r["truth"]) for r in wrong])
    print(f"when the AI was wrong ({len(wrong)} cases): reader followed it {follow:.3f}, corrected it {fixed:.3f}")

key = lambda r: (r["reader_id"], r["case_id"])  # noqa: E731
d0 = {key(r): int(r["answer"]) == int(r["truth"]) for r in by.get("doctor_only", [])}
d1 = {key(r): int(r["answer"]) == int(r["truth"]) for r in da}
pairs = [k for k in d0 if k in d1]
b = sum(d0[k] and not d1[k] for k in pairs); c = sum(d1[k] and not d0[k] for k in pairs)
if pairs and b + c:
    p = binomtest(min(b, c), b + c, 0.5).pvalue
    print(f"paired cases {len(pairs)}: only-doctor-correct {b}, only-doctor+AI-correct {c}, McNemar exact p = {p:.4f}")
