import json
import secrets
import uuid
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
from config import KNOWLEDGE_DIR, MAX_UPLOAD_MB, TEMPERATURE_FITTED, UPLOADS, USE_MOCK_MODEL
from services import adaptive, explain, quality, rag, reliability, scan_router

app = FastAPI(title="CLINIX-AI")
db.init()
kb = rag.KnowledgeBase(KNOWLEDGE_DIR)

SCAN_TYPES = {"xray", "ct", "mri"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".dcm"}
TOKENS: dict[str, dict] = {}


def current_user(x_token: str = Header(None)):
    user = TOKENS.get(x_token)
    if not user:
        raise HTTPException(401, "Log in first")
    return user


class LoginIn(BaseModel):
    username: str
    password: str


class GenerateIn(BaseModel):
    analysis_id: str
    level: str
    evidence: bool | None = None     # None = automatic by level
    guidance: bool | None = None


class ReviewIn(BaseModel):
    analysis_id: str
    action: str                 # accept | edit | reject
    final_text: str | None = None
    reason: str | None = None


class ChallengeIn(BaseModel):
    analysis_id: str
    finding: str
    reason: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/login")
def login(body: LoginIn):
    user = db.check_login(body.username, body.password)
    if not user:
        db.audit(body.username, "login_failed")
        raise HTTPException(401, "Wrong username or password")
    token = secrets.token_urlsafe(24)
    TOKENS[token] = user
    db.audit(user["username"], "login")
    return {"token": token, "user": user}


@app.get("/api/system")
def system(user=Depends(current_user)):
    return {"mock_model": USE_MOCK_MODEL, "temperature_fitted": TEMPERATURE_FITTED,
            "knowledge_entries": len(kb.entries), "scan_types": sorted(SCAN_TYPES),
            "models_available": scan_router.available(),
            "levels": sorted(adaptive.LEVELS)}


@app.post("/api/scans/upload")
def upload(scan_type: str = Form(...), file: UploadFile = File(...), user=Depends(current_user)):
    if scan_type not in SCAN_TYPES:
        raise HTTPException(400, f"scan_type must be one of {sorted(SCAN_TYPES)}")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}. Use {sorted(ALLOWED_EXT)}")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
    try:
        gray, color = quality.load_image(data, ext)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read file: {e}")
    scan_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{scan_id}.png"
    cv2.imwrite(str(path), gray)            # pixels only: original filename and DICOM tags are dropped
    q = quality.check_quality(gray, color, scan_type)
    db.add_scan(scan_id, scan_type, path, q, user["username"])
    db.audit(user["username"], "upload", {"scan_id": scan_id, "scan_type": scan_type, "passed": q["passed"]})
    return {"scan_id": scan_id, "scan_type": scan_type, "quality": q}


@app.post("/api/scans/analyze")
def analyze(scan_id: str = Form(...), level: str = Form(None), user=Depends(current_user)):
    scan = db.get_scan(scan_id)
    if not scan:
        raise HTTPException(404, "Unknown scan_id")
    q = json.loads(scan["quality"])
    if not q["passed"]:
        raise HTTPException(409, "Scan did not pass the quality gate. " + q["reason"])
    try:
        model = scan_router.get_model(scan["scan_type"])
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if model is None:
        db.audit(user["username"], "analyze_no_model", {"scan_id": scan_id})
        return {"status": "no_model",
                "message": f"No trained {scan['scan_type'].upper()} model is installed yet. Quality check only. "
                           "Train one with evaluation/train_classifier.py (see README)."}

    gray = cv2.imread(scan["path"], cv2.IMREAD_GRAYSCALE)
    summary = reliability.assess(model.predict(gray))
    top_label = summary["top"]["label"]
    labels = {f["label"] for f in summary["findings"]} | {top_label}
    evidence = {lab: kb.verify(lab) for lab in labels}
    heatmap, heat_note = None, None
    try:
        heatmap = explain.overlay(gray, model.gradcam(gray, top_label))
    except Exception as e:  # noqa: BLE001
        heat_note = f"Heatmap unavailable: {type(e).__name__}"
    result = {"scan_type": scan["scan_type"], "model": {"name": model.name, "mock": model.mock},
              "quality": q, "summary": summary, "evidence": evidence}
    chosen = level or user["level"]
    report = adaptive.build_report(result, chosen, kb)
    analysis_id = uuid.uuid4().hex[:12]
    db.add_analysis(analysis_id, scan_id, result, report, chosen, user["username"])
    db.audit(user["username"], "analyze", {"analysis_id": analysis_id, "model": model.name,
                                           "reliability": summary["reliability"]["level"]})
    return {"status": "ok", "analysis_id": analysis_id, "result": result, "report": report,
            "original": explain.original(gray), "heatmap": heatmap, "heatmap_label": top_label,
            "heatmap_note": heat_note}


def _own_analysis(analysis_id, user):
    row = db.get_analysis(analysis_id)
    if not row or row["user"] != user["username"]:
        raise HTTPException(404, "Unknown analysis_id")
    return row


@app.post("/api/reports/generate")
def generate(body: GenerateIn, user=Depends(current_user)):
    if body.level not in adaptive.LEVELS:
        raise HTTPException(400, f"level must be one of {sorted(adaptive.LEVELS)}")
    row = _own_analysis(body.analysis_id, user)
    report = adaptive.build_report(json.loads(row["result"]), body.level, kb,
                                   {"evidence": body.evidence, "guidance": body.guidance})
    db.update_report(body.analysis_id, report, body.level)
    db.audit(user["username"], "regenerate", {"analysis_id": body.analysis_id, "level": body.level})
    return report


@app.post("/api/reports/review")
def review(body: ReviewIn, user=Depends(current_user)):
    row = _own_analysis(body.analysis_id, user)
    ai_text = json.loads(row["report"])["text"]
    if body.action not in {"accept", "edit", "reject"}:
        raise HTTPException(400, "action must be accept, edit or reject")
    if body.action == "reject" and not (body.reason or "").strip():
        raise HTTPException(400, "Give a reason when rejecting")
    final = body.final_text if body.final_text is not None else ai_text
    if body.action == "edit" and final.strip() == ai_text.strip():
        raise HTTPException(400, "No changes to save. Edit the text first, or choose accept")
    db.add_decision(body.analysis_id, user["username"], body.action,
                    None if body.action == "reject" else final, body.reason)
    db.audit(user["username"], "decision", {"analysis_id": body.analysis_id, "action": body.action})
    return {"saved": True, "action": body.action}


@app.post("/api/reports/challenge")
def challenge(body: ChallengeIn, user=Depends(current_user)):
    row = _own_analysis(body.analysis_id, user)
    if len(body.reason.strip()) < 3:
        raise HTTPException(400, "Say what you disagree with")
    summary = json.loads(row["result"])["summary"]
    mine = next((r for r in summary["all"] if r["label"] == body.finding), None)
    if not mine:
        raise HTTPException(400, "Unknown finding")
    own = kb.lookup(body.finding)
    hits = ([dict(own)] if own else []) + [h for h in kb.retrieve(f"{body.finding} {body.reason}", 3)
                                          if not own or h["title"] != own["title"]][:1]
    db.add_decision(body.analysis_id, user["username"], "challenge", None, body.reason, body.finding)
    db.audit(user["username"], "challenge", {"analysis_id": body.analysis_id, "finding": body.finding})
    return {"model_view": mine, "reliability": summary["reliability"],
            "evidence": [{"title": h["title"], "what": h["what"], "matters": h["matters"],
                          "source": h["source"]} for h in hits],
            "message": "Challenge recorded. The system answers only from the model output and the "
                       "reference notes above; it does not change its prediction. You decide."}


@app.get("/api/reports/history")
def history(user=Depends(current_user)):
    return db.history(user["username"])


@app.get("/api/analytics/decisions")
def analytics(user=Depends(current_user)):
    return db.analytics()


# Web page at http://localhost:8000/  (keep LAST so /api routes win)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
