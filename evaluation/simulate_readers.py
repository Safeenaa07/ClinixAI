"""SIMULATED doctor study, for when real readers are not available. It is a simulation,
not a clinical result, and the output file says so (column simulated=1).

It takes the real AI scores for labelled cases and invents readers:
  - doctor_only: each reader is right with probability --doctor-acc
  - doctor_ai: the reader sees the AI answer; follows it with probability --follow, otherwise
    answers on their own (right with --doctor-acc). Over-trust and the gain from AI therefore
    come from YOUR assumptions about the readers, while the AI error rates are real.
Input CSV columns: label (0/1), score (AI score). Optional: case_id.
Usage: python evaluation/simulate_readers.py predictions.csv --readers 5 --threshold 0.5 --out study_simulated.csv"""
import argparse
import csv

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--readers", type=int, default=5)
ap.add_argument("--doctor-acc", type=float, default=0.75); ap.add_argument("--follow", type=float, default=0.6)
ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="study_simulated.csv")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
rows = list(csv.DictReader(open(a.csv)))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "truth", "reader_id", "condition", "answer", "seconds", "ai_answer", "simulated"])
    for i, r in enumerate(rows):
        cid, truth = r.get("case_id") or i, int(r["label"])
        ai = int(float(r["score"]) >= a.threshold)
        w.writerow([cid, truth, "AI", "ai_only", ai, 1, "", 1])
        for k in range(a.readers):
            own = lambda: truth if rng.random() < a.doctor_acc else 1 - truth  # noqa: E731
            w.writerow([cid, truth, f"sim{k + 1}", "doctor_only", own(), 60, "", 1])
            ans = ai if rng.random() < a.follow else own()
            w.writerow([cid, truth, f"sim{k + 1}", "doctor_ai", ans, 45, ai, 1])
print("wrote", a.out, "(SIMULATED readers; say so in the report)")
