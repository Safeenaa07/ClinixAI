"""Print quality-feature percentiles for a folder of GOOD scans, so you can set thresholds
in backend/services/quality.py from your own data.
Usage: python evaluation/quality_stats.py <folder>"""
import sys
from pathlib import Path

import numpy as np

import _common  # noqa: F401
from services import quality

folder = Path(sys.argv[1])
rows = []
for f in sorted(folder.iterdir()):
    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".dcm"}:
        try:
            g, c = quality.load_image(f.read_bytes(), f.suffix.lower())
            rows.append(quality.features(g, c))
        except Exception as e:  # noqa: BLE001
            print("skip", f.name, e)
a = np.array(rows)
print(f"{len(a)} images")
for i, name in enumerate(quality.FEATURE_NAMES):
    print(f"{name:14s} min {a[:, i].min():8.2f}  p5 {np.percentile(a[:, i], 5):8.2f}  "
          f"median {np.median(a[:, i]):8.2f}  p95 {np.percentile(a[:, i], 95):8.2f}  max {a[:, i].max():8.2f}")
print("blur = expm1(log_blur). Set BLUR_MIN a little below the 5th percentile of good scans.")
