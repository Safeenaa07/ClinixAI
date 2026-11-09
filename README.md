# CLINIX-AI: Adaptive AI-driven Medical Scan Analysis

Research prototype. Decision support only, not a diagnosis. Not clinically validated.

## 1. Run it (Windows PowerShell)
```
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
Open http://localhost:8000/ and log in: `dr_junior` or `dr_specialist`, password `clinix123`
(demo accounts, change them in `db.py` before sharing).

## 2. Get a real X-ray model (needs torch)
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install torchxrayvision
```
First analysis downloads pretrained DenseNet121 weights (needs internet, then cached).
Default weights are `densenet121-res224-pc` (PadChest-trained, 15 findings). Change with
`$env:CLINIX_XRV_WEIGHTS="densenet121-res224-all"` for a stronger demo model, but do NOT evaluate
that one on NIH data (it was trained on NIH).

If torch is blocked on your laptop, test the pipeline with FAKE output: `$env:CLINIX_MOCK_MODEL="1"`.
A red banner shows whenever mock mode is on. Never show mock numbers as results.

## 3. Get CT and MRI models (one hour on Colab)
Open `colab/CLINIX_AI_Colab.ipynb` in Google Colab (GPU). It downloads the Kaggle datasets, trains
ResNet18 slice classifiers (`evaluation/train_classifier.py`), fits calibration, and runs the external
X-ray evaluation. Copy the resulting `backend/models/*.pt` and `*_meta.json` back into this project and
restart. CT and MRI then appear in the app. These classify single 2D images, not 3D volumes.

## 4. What is real
| Part | Status |
|---|---|
| Upload; DICOM to pixels only (no patient tags kept) | Real |
| Quality rules (blur, brightness, contrast, noise, size, colour) | Real, thresholds need tuning on your data |
| ML quality classifier + OOD detector | Real code; train with `evaluation/train_quality_ood.py`; learns synthetic defects only |
| X-ray model (DenseNet121, torchxrayvision) + Grad-CAM | Real, tested here on public chest X-rays |
| CT / MRI models (ResNet18, 2D) + Grad-CAM | Code real and tested on synthetic data; needs your Colab training run |
| Reliability: certainty + stability under jittered inputs | Real, thresholds not validated |
| Calibration (temperature scaling) | Code real; stays uncalibrated until fitted (report says so) |
| Evidence check (TF-IDF over `backend/knowledge/*.md`) | Real mechanics; the seed notes are UNVERIFIED, replace sources before demo |
| Adaptive report (rule-based, reliability-aware, evidence/guidance toggles) | Real |
| Optional LLM reword (`GROQ_API_KEY`) | Guarded (rejected if it changes numbers); untested |
| Accept / edit / challenge / reject + audit trail (SQLite) | Real |
| Docker | Files provided, not built here |
| Bias, calibration, reader-study evaluation | Scripts real; results need your data |

## 5. Evaluation scripts (`evaluation/`)
- `eval_xray_nih.py`: external X-ray test on the NIH sample: per-finding AUC, calibration (T fitted on a patient-split validation half), sex/age subgroup gaps, and how often low-certainty predictions are wrong
- `train_classifier.py`: CT/MRI training, calibration, accuracy, macro-F1, confusion matrix, ECE, Brier
- `train_quality_ood.py`, `quality_stats.py`: ML quality/OOD models and threshold statistics
- `fit_temperature.py`, `subgroup_metrics.py`: generic calibration and bias scripts for your own CSVs
- `simulate_readers.py`, `reader_study.py`: doctor-only vs AI-only vs doctor+AI. `simulate_readers.py` makes SIMULATED readers and marks the file as such. Use real readers if you can.

## 6. Settings (environment variables)
`CLINIX_MOCK_MODEL`, `CLINIX_XRV_WEIGHTS`, `CLINIX_TEMPERATURE` (after fitting), `CLINIX_DETECT`
(default 0.60), `CLINIX_POSSIBLE` (default 0.50), `GROQ_API_KEY`, `GROQ_MODEL`.

## 7. API
`POST /api/login` `GET /api/system` `POST /api/scans/upload` (includes quality + OOD check)
`POST /api/scans/analyze` `POST /api/reports/generate` `POST /api/reports/review`
`POST /api/reports/challenge` `GET /api/reports/history` `GET /api/analytics/decisions`

## 8. Limits
Not clinically validated. Public datasets only. Seed medical notes unverified. Scores from the X-ray model
are operating-point-normalised outputs, not probabilities. Public CT/MRI datasets may put the same patient in
train and test. The bias check covers sex and age only. SQLite for the prototype (PostgreSQL not implemented).
