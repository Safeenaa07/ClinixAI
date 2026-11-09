"""External evaluation of the X-ray model on the NIH ChestX-ray14 sample (Kaggle "nih-chest-xrays/sample").
Default weights are PadChest-trained, so NIH images are an EXTERNAL test set (do not use
"densenet121-res224-all" or "-nih" here: they saw NIH images).

Produces: per-finding AUC, any-finding AUC and operating points, calibration (temperature fit on
the validation half, scored on the test half), sex/age subgroup results, and how often
low-certainty predictions are wrong compared with high-certainty ones.
Patients are split val/test by patient id, so one patient never appears in both.
Usage:
  python evaluation/eval_xray_nih.py --csv sample_labels.csv --images sample/images
Options: --weights NAME  --min-n 10  --max-images N  --out evaluation/results_xray.json"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

import _common  # noqa: F401
from config import LOW_RELIABILITY
from services.xray_model import TorchXrayVisionModel

NIH14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
         "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--images", required=True)
ap.add_argument("--weights", default=None); ap.add_argument("--min-n", type=int, default=10)
ap.add_argument("--max-images", type=int, default=0)
ap.add_argument("--out", default=str(Path(__file__).parent / "results_xray.json"))
a = ap.parse_args()

paths = {}
for root, _, files in os.walk(a.images):
    for f in files:
        paths[f] = os.path.join(root, f)
rows = []
for r in csv.DictReader(open(a.csv, encoding="utf-8-sig")):
    name = r["Image Index"]
    age = re.findall(r"\d+", r.get("Patient Age", ""))
    if name in paths and age:
        rows.append({"name": name, "labels": set(r["Finding Labels"].split("|")), "pid": int(re.sub(r"\D", "", r["Patient ID"]) or 0),
                     "age": int(age[0]), "sex": r.get("Patient Gender", "?").strip().upper()})
if a.max_images:
    rows = rows[:a.max_images]
print(f"{len(rows)} images with labels found")
if not rows:
    raise SystemExit("No images matched the CSV. Check --images points at the folder with the .png files.")

model = TorchXrayVisionModel(a.weights)
print("model:", model.name)
grays = lambda chunk: [cv2.imread(paths[r["name"]], cv2.IMREAD_GRAYSCALE) for r in chunk]  # noqa: E731
outs = []
for i in range(0, len(rows), 64):
    outs.append(model.batch_logits(grays(rows[i:i + 64])))
    if (i // 64) % 10 == 0:
        print(f"  {i}/{len(rows)}")
out = np.concatenate(outs)                       # operating-point-normalised scores
labels = [l for l in model.labels if l and l in NIH14]
col = {l: model.labels.index(l) for l in labels}
P = np.stack([sig(out[:, col[l]]) if out.min() < 0 or out.max() > 1 else out[:, col[l]] for l in labels], 1)
Y = np.array([[l in r["labels"] for l in labels] for r in rows], int)
any_y = np.array([r["labels"] != {"No Finding"} for r in rows], int)
split = np.array([r["pid"] % 2 for r in rows])   # 0 = validation, 1 = test (by patient)
val, test = split == 0, split == 1
print(f"validation {val.sum()}  test {test.sum()}")


def auc(y, s):
    return float(roc_auc_score(y, s)) if y.sum() >= 1 and (1 - y).sum() >= 1 else float("nan")


res = {"model": model.name, "n_images": len(rows), "n_val": int(val.sum()), "n_test": int(test.sum()), "per_finding": {}}
for j, l in enumerate(labels):
    pos, neg = int(Y[test, j].sum()), int((1 - Y[test, j]).sum())
    if pos >= a.min_n and neg >= a.min_n:
        pred = P[test, j] >= 0.5
        yy = Y[test, j]
        res["per_finding"][l] = {"n_pos": pos, "auc": round(auc(yy, P[test, j]), 4),
                                 "sens@0.5": round(float((pred & (yy == 1)).sum() / pos), 4),
                                 "spec@0.5": round(float((~pred & (yy == 0)).sum() / neg), 4)}
aucs = [v["auc"] for v in res["per_finding"].values() if not np.isnan(v["auc"])]
res["macro_auc"] = round(float(np.mean(aucs)), 4) if aucs else None

score = P.max(1)                                  # any-finding score
res["any_finding"] = {"auc": round(auc(any_y[test], score[test]), 4)}
cands = np.linspace(0.3, 0.95, 66)
youden = [(score[val][any_y[val] == 1] >= t).mean() - (score[val][any_y[val] == 0] >= t).mean()
          if (any_y[val] == 1).any() and (any_y[val] == 0).any() else -9 for t in cands]
thr = float(cands[int(np.argmax(youden))])
res["any_finding"]["threshold_from_validation"] = round(thr, 3)


def op(mask, t):
    y, p = any_y[mask], score[mask] >= t
    return {"n": int(mask.sum()), "sens": round(float((p & (y == 1)).sum() / max((y == 1).sum(), 1)), 4),
            "spec": round(float((~p & (y == 0)).sum() / max((y == 0).sum(), 1)), 4)}


res["any_finding"]["test_at_validation_threshold"] = op(test, thr)

# calibration: fit T on validation, score on test (per finding-and-image pair)
zv, yv = logit(P[val]).ravel(), Y[val].ravel()
zt, yt = logit(P[test]).ravel(), Y[test].ravel()
nll = lambda t: -np.mean(yv * np.log(np.clip(sig(zv / t), 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - sig(zv / t), 1e-7, 1)))  # noqa: E731
Tm = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x) if len(zv) else 1.0


def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        e += m.mean() * abs(p[m].mean() - y[m].mean()) if m.any() else 0
    return float(e)


p0, p1 = sig(zt), sig(zt / Tm)
res["calibration"] = {"temperature": round(Tm, 3), "ece_before": round(ece(p0, yt), 4), "ece_after": round(ece(p1, yt), 4),
                      "brier_before": round(float(np.mean((p0 - yt) ** 2)), 4), "brier_after": round(float(np.mean((p1 - yt) ** 2)), 4)}

# bias: any-finding performance by sex and age (test half)
ages = np.array([r["age"] for r in rows]); sexes = np.array([r["sex"] for r in rows])
groups = {"sex F": sexes == "F", "sex M": sexes == "M", "age 0-40": ages <= 40,
          "age 41-60": (ages > 40) & (ages <= 60), "age 61+": ages > 60}
res["subgroups"] = {}
for g, m in groups.items():
    m = m & test
    if m.sum() >= 2:
        d = op(m, thr); d["auc"] = round(auc(any_y[m], score[m]), 4); res["subgroups"][g] = d
for fam in ("sex", "age"):
    v = [d["auc"] for k, d in res["subgroups"].items() if k.startswith(fam) and not np.isnan(d["auc"]) and d["n"] >= a.min_n]
    if len(v) > 1:
        res[f"{fam}_auc_gap"] = round(max(v) - min(v), 4)

# does low certainty mark wrong predictions? (certainty only; the app also uses stability)
cert = np.abs(2 * P[test] - 1); err = ((P[test] >= 0.5).astype(int) != Y[test])
lowm = cert < LOW_RELIABILITY
res["reliability_check"] = {"pairs": int(lowm.size), "low_certainty_share": round(float(lowm.mean()), 4),
                            "error_rate_low_certainty": round(float(err[lowm].mean()), 4) if lowm.any() else None,
                            "error_rate_high_certainty": round(float(err[~lowm].mean()), 4) if (~lowm).any() else None}

Path(a.out).write_text(json.dumps(res, indent=1))
print("\n=== RESULTS (paste into the report) ===")
print(f"macro AUC over {len(aucs)} findings (test): {res['macro_auc']}")
for l, v in res["per_finding"].items():
    print(f"  {l:20s} AUC {v['auc']:.3f}  sens {v['sens@0.5']:.3f}  spec {v['spec@0.5']:.3f}  (positives {v['n_pos']})")
print("any finding:", res["any_finding"])
print("calibration:", res["calibration"])
for g, d in res["subgroups"].items():
    print(f"  {g:10s} {d}")
print("gaps:", {k: v for k, v in res.items() if k.endswith("auc_gap")})
print("reliability check:", res["reliability_check"])
print("saved", a.out)
