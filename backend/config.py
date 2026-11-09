import os
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
MODEL_DIR = BASE / "models"
KNOWLEDGE_DIR = BASE / "knowledge"
for _p in (DATA, UPLOADS, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "clinix.db"

# Set CLINIX_MOCK_MODEL=1 to test the pipeline without torch. Output is FAKE.
USE_MOCK_MODEL = os.getenv("CLINIX_MOCK_MODEL", "0") == "1"

# Temperature scaling. Stays 1.0 (= uncalibrated) until you fit it on a held-out
# validation set with evaluation/fit_temperature.py and set CLINIX_TEMPERATURE.
_t = os.getenv("CLINIX_TEMPERATURE")
TEMPERATURE = float(_t) if _t else 1.0
TEMPERATURE_FITTED = _t is not None

# Decision thresholds. Starting values, NOT validated. Tune on validation data.
# With the X-ray model, 0.5 is the library's operating point, so "possible" starts there and
# "detected" needs a clear margin. Sweep these on validation data (evaluation/eval_xray_nih.py).
DETECT_THRESHOLD = float(os.getenv("CLINIX_DETECT", "0.60"))
POSSIBLE_THRESHOLD = float(os.getenv("CLINIX_POSSIBLE", "0.50"))
LOW_RELIABILITY = 0.60
CRITICAL_FINDINGS = {"Pneumothorax"}
MAX_UPLOAD_MB = 25
