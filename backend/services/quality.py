"""Module 2: scan quality + out-of-distribution (OOD) gate.

Two layers:
  1. Rules on simple image measures (blur, brightness, contrast, noise, size, colour).
  2. Optional trained ML models (quality classifier + anomaly detector) loaded from
     models/quality_<scan_type>.joblib. Create them with evaluation/train_quality_ood.py.
All thresholds are starting values. Tune them on your own scans
(see evaluation/quality_stats.py) and report the tuned values."""
import io

import cv2
import numpy as np

from config import MODEL_DIR

MIN_SIDE = 224
BLUR_MIN = 12.0          # variance of Laplacian on a 512x512 copy; lower = blurrier
BRIGHT_RANGE = (40, 220)
CONTRAST_MIN = 25.0
NOISE_MAX = 30.0
COLOR_MAX = 25.0         # colourful photos are not radiographs

FEATURE_NAMES = ["log_blur", "brightness", "contrast", "noise", "log_min_side", "colorfulness"]
_bundles: dict = {}


def load_image(data: bytes, ext: str):
    """Return (gray uint8 image, colorfulness). Raises ValueError on unreadable files.
    DICOM is converted to pixels only, so no patient metadata is kept."""
    if ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            arr = arr[arr.shape[0] // 2]          # multi-frame: middle slice
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255 - arr
        return arr.astype(np.uint8), 0.0
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File could not be read as an image")
    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    color = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), color


def _noise(img) -> float:
    """Immerkaer fast noise-level estimate."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    c = cv2.filter2D(img.astype(np.float64), -1, k)
    h, w = img.shape
    return float(np.sqrt(0.5 * np.pi) * np.abs(c[1:-1, 1:-1]).sum() / (6.0 * (w - 2) * (h - 2)))


def features(gray: np.ndarray, colorfulness: float = 0.0) -> np.ndarray:
    h, w = gray.shape
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    blur = cv2.Laplacian(small, cv2.CV_64F).var()
    return np.array([np.log1p(blur), small.mean(), small.std(), _noise(small),
                     np.log(min(h, w)), colorfulness])


def _bundle(scan_type: str):
    if scan_type not in _bundles:
        path = MODEL_DIR / f"quality_{scan_type}.joblib"
        if path.exists():
            import joblib
            _bundles[scan_type] = joblib.load(path)
        else:
            _bundles[scan_type] = None
    return _bundles[scan_type]


def check_quality(gray: np.ndarray, colorfulness: float, scan_type: str) -> dict:
    f = features(gray, colorfulness)
    h, w = gray.shape
    blur, bright, contrast, noise = float(np.expm1(f[0])), float(f[1]), float(f[2]), float(f[3])
    checks = {
        "resolution": {"value": f"{w}x{h}", "ok": min(h, w) >= MIN_SIDE},
        "blur": {"value": round(blur, 1), "ok": blur >= BLUR_MIN},
        "brightness": {"value": round(bright, 1), "ok": BRIGHT_RANGE[0] <= bright <= BRIGHT_RANGE[1]},
        "contrast": {"value": round(contrast, 1), "ok": contrast >= CONTRAST_MIN},
        "noise": {"value": round(noise, 1), "ok": noise <= NOISE_MAX},
        "colour": {"value": round(colorfulness, 1), "ok": colorfulness <= COLOR_MAX},
    }
    failed = [k for k, v in checks.items() if not v["ok"] and k != "colour"]
    ood_reasons = []
    if not checks["colour"]["ok"]:
        ood_reasons.append("colour image: does not look like a radiograph")

    ml = None
    bundle = _bundle(scan_type)
    if bundle is not None:
        x = bundle["scaler"].transform([f])
        p_good = float(bundle["clf"].predict_proba(x)[0, 1])
        is_outlier = int(bundle["iso"].predict(x)[0]) == -1
        ml = {"p_good": round(p_good, 3), "outlier": is_outlier}
        if p_good < 0.5 and "ml_quality" not in failed:
            failed.append("ml_quality")
        if is_outlier:
            ood_reasons.append(f"unlike the {scan_type} scans the system was trained to expect")

    passed = not failed and not ood_reasons
    parts = []
    if ood_reasons:
        parts.append("Unfamiliar scan, manual review needed: " + "; ".join(ood_reasons) + ".")
    if failed:
        parts.append("Upload a better scan. Failed: " + ", ".join(failed))
    reason = " ".join(parts)
    return {"passed": passed, "reason": reason, "checks": checks,
            "ood": bool(ood_reasons), "ml": ml,
            "method": "rules+ml" if ml else "rules only"}
