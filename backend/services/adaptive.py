"""Module 7: reliability-aware adaptive assistance (rule-based, not learned).

Inputs: the doctor's chosen level, the reliability result, the quality result.
  junior     -> detailed (explanations, evidence, suggestions)
  specialist -> concise  (short summary)
  concise | standard | detailed -> as named
Rules: low reliability never gets a bare summary (concise is raised to standard and a
warning is shown); a critical finding is always put first.
Text is built from structured data only, so it cannot invent findings. An optional LLM
step (GROQ_API_KEY) may only reword it, and is rejected if it changes any number."""
import json
import os
import re
import urllib.request

from config import CRITICAL_FINDINGS

ALIASES = {"junior": "detailed", "specialist": "concise"}
LEVELS = {"junior", "specialist", "concise", "standard", "detailed"}


def resolve_level(level: str, reliability_level: str):
    base = ALIASES.get(level, level)
    if base not in {"concise", "standard", "detailed"}:
        base = "standard"
    forced = reliability_level == "low" and base == "concise"
    return ("standard" if forced else base), forced


def _pct(p):
    return f"{round(p * 100)}%"


def build_report(result: dict, level: str, kb, options: dict | None = None) -> dict:
    """options: {"evidence": True/False/None, "guidance": True/False/None}; None = automatic by level."""
    options = options or {}
    summary = result["summary"]
    rel = summary["reliability"]
    used, forced = resolve_level(level, rel["level"])
    ev = result["evidence"]
    findings = summary["findings"]
    top = summary["top"]
    ev_on = options.get("evidence")
    ev_on = (used != "concise") if ev_on is None else bool(ev_on)
    guide_on = options.get("guidance")
    guide_on = (used == "detailed") if guide_on is None else bool(guide_on)
    pretty = lambda x: x.replace("_", " ")  # noqa: E731

    L = ["AI-ASSISTED DRAFT. Decision support only, not a diagnosis. The doctor makes the final decision."]
    if result["model"]["mock"]:
        L.append("MOCK MODEL OUTPUT: fake numbers for pipeline testing. Do not interpret.")

    for f in findings:
        if f["label"] in CRITICAL_FINDINGS and f["status"] == "detected":
            if rel["level"] == "low":
                L.append(f"ATTENTION: possible {f['label']} at LOW confidence ({_pct(f['prob'])}). Check the image yourself.")
            else:
                L.append(f"ATTENTION, review first: possible {f['label']} ({_pct(f['prob'])}).")

    if findings:
        main = findings[0]
        word = "Main finding" if main["status"] == "detected" else "Possible finding (borderline score)"
        L.append(f"{word}: {pretty(main['label'])}, model score {_pct(main['prob'])}, "
                 f"reliability {rel['level']} ({rel['score']:.2f}).")
    else:
        L.append(f"No finding above the review threshold. Highest score: {pretty(top['label'])} "
                 f"{_pct(top['prob'])}, reliability {rel['level']} ({rel['score']:.2f}).")
    if rel["level"] == "low":
        L.append("WARNING: reliability is LOW. The model output is uncertain or unstable. "
                 "Look at the image directly before relying on this draft.")
    if forced:
        L.append("(Concise view was expanded to Standard because reliability is low.)")

    if used == "concise":
        shown = findings[:1] if (ev_on or guide_on) else []
    elif used == "standard":
        shown = findings[:3]
    else:
        shown = findings
    explain = used in ("standard", "detailed")
    for f in shown:
        e = kb.lookup(f["label"])
        L.append("")
        L.append(f"{pretty(f['label'])} ({_pct(f['prob'])}, {f['status']})")
        if explain and e:
            L.append(f"  What it is: {e['what']}")
            L.append(f"  Why it may matter: {e['matters']}")
        if ev_on:
            v = ev.get(f["label"])
            L.append(f"  Reference: {v['source']}" if v and v["supported"]
                     else "  No supporting reference in the knowledge base. Verify manually.")

    if guide_on:
        L.append("")
        L.append("Suggestions for the doctor (from reference notes, not orders):")
        any_next = False
        for f in (shown or findings[:1]):
            e = kb.lookup(f["label"])
            if e and e["next"]:
                L.append(f"  - {pretty(f['label'])}: {e['next']}")
                any_next = True
        if not any_next:
            L.append("  - None from the knowledge base.")

    if used == "detailed":
        q = result["quality"]
        L.append("")
        L.append(f"Scan quality: {'passed' if q['passed'] else 'failed'} ({q['method']}).")
        L.append("Limits:")
        L.append("  - Research prototype, not clinically validated.")
        L.append("  - Model was trained on public datasets; it may not generalise to this patient or scanner.")
        if not rel["calibrated"]:
            L.append("  - Scores are NOT calibrated (temperature = 1.0). Treat them as rough.")
        if rel.get("kind") != "softmax":
            L.append("  - Scores are the model's operating-point-normalised outputs, not probabilities.")
        L.append("  - The heatmap shows where the model looked, not a lesion boundary.")

    text = "\n".join(L)
    by = "template"
    polished = _llm_polish(text)
    if polished:
        text, by = polished, "template+llm"
    return {"level_requested": level, "level_used": used, "forced_upgrade": forced,
            "evidence_shown": ev_on, "guidance_shown": guide_on, "text": text, "generated_by": by}


def _llm_polish(text: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        body = json.dumps({
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Reword this radiology decision-support draft for clarity. "
                 "Do not add, remove or change any finding, number, reference or warning. Output only the text."},
                {"role": "user", "content": text}]}).encode()
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", body,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=20).read())["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001
        return None
    nums = lambda s: sorted(re.findall(r"\d+(?:\.\d+)?", s))  # noqa: E731
    if nums(out) != nums(text) or "not a diagnosis" not in out.lower():
        return None            # reject any rewrite that changes numbers or drops the disclaimer
    return out
