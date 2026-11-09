"""Fit the temperature T for calibration on a held-out VALIDATION set (never the test set).
CSV columns: logit,label   (one row per image-and-finding pair; label is 0 or 1)
Prints T, calibration error (ECE) and Brier score before and after.
Then set it:  PowerShell  $env:CLINIX_TEMPERATURE="1.7"
Usage: python evaluation/fit_temperature.py validation_logits.csv"""
import csv
import sys

import numpy as np
from scipy.optimize import minimize_scalar

rows = list(csv.DictReader(open(sys.argv[1])))
z = np.array([float(r["logit"]) for r in rows]); y = np.array([int(r["label"]) for r in rows])
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def nll(t):
    p = np.clip(sig(z / t), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
for name, t in (("before (T=1.0)", 1.0), (f"after (T={T:.3f})", T)):
    p = sig(z / t)
    print(f"{name:18s} ECE {ece(p):.4f}  Brier {np.mean((p - y) ** 2):.4f}")
print(f'Set CLINIX_TEMPERATURE="{T:.3f}"')
