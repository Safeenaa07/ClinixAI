"""Train the ML quality classifier and the OOD detector from a folder of GOOD scans.
  - Quality classifier: logistic regression, good vs synthetically degraded copies
    (blur, noise, darkening, brightening, low contrast, shrinking).
  - OOD detector: Isolation Forest fitted on the good scans only.
Images are split into train/test BEFORE degrading, so test images are never seen in training.
Limit: it learns synthetic defects, not every real-world defect. Say so in your report.
Usage: python evaluation/train_quality_ood.py <folder> --modality xray   (100+ scans recommended)"""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import _common  # noqa: F401
from config import MODEL_DIR
from services import quality


def degrade(g, rng):
    kind = int(rng.integers(0, 6))
    if kind == 0:
        k = int(rng.choice([9, 15, 21, 31]))
        return cv2.GaussianBlur(g, (k, k), 0)
    if kind == 1:
        return np.clip(g + rng.normal(0, rng.uniform(25, 60), g.shape), 0, 255).astype(np.uint8)
    if kind == 2:
        return (g * rng.uniform(0.15, 0.4)).astype(np.uint8)
    if kind == 3:
        return np.clip(g.astype(np.float32) * 0.3 + rng.uniform(170, 210), 0, 255).astype(np.uint8)
    if kind == 4:
        return np.clip((g.astype(np.float32) - 128) * rng.uniform(0.05, 0.2) + 128, 0, 255).astype(np.uint8)
    s = int(rng.integers(60, 200))
    return cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA)


def build(files, rng, per_image=3):
    good, X, y = [], [], []
    for f in files:
        g, c = quality.load_image(f.read_bytes(), f.suffix.lower())
        fg = quality.features(g, c)
        good.append(fg)
        X.append(fg); y.append(1)
        for _ in range(per_image):
            X.append(quality.features(degrade(g, rng), c)); y.append(0)
    return np.array(good), np.array(X), np.array(y)


ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("--modality", default="xray")
args = ap.parse_args()
files = [f for f in sorted(Path(args.folder).iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".dcm"}]
if len(files) < 20:
    raise SystemExit(f"Only {len(files)} usable scans. Use at least 20 (100+ is better).")
rng = np.random.default_rng(0)
order = rng.permutation(len(files)); cut = int(0.8 * len(files))
train = [files[i] for i in order[:cut]]; test = [files[i] for i in order[cut:]]
g_tr, X_tr, y_tr = build(train, rng); g_te, X_te, y_te = build(test, rng)

scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_tr), y_tr)
iso = IsolationForest(contamination=0.02, random_state=0).fit(scaler.transform(g_tr))
acc = (clf.predict(scaler.transform(X_te)) == y_te).mean()
out_rate = (iso.predict(scaler.transform(g_te)) == -1).mean()
print(f"train scans {len(train)}, test scans {len(test)}")
print(f"quality classifier accuracy on held-out scans (synthetic defects): {acc:.3f}")
print(f"OOD detector flags {out_rate:.3f} of held-out GOOD scans (should be small)")
dest = MODEL_DIR / f"quality_{args.modality}.joblib"
joblib.dump({"scaler": scaler, "clf": clf, "iso": iso, "features": quality.FEATURE_NAMES}, dest)
print("saved", dest)
