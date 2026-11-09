"""Module 4: calibrated scores + reliability.

Two kinds of model output are supported:
  sigmoid  (X-ray, one score per finding): temperature scaling on the logit.
  softmax  (CT, MRI, one class per scan): temperature scaling on the logits.
Temperature stays 1.0 (uncalibrated) until you fit it on a held-out validation set
(evaluation/fit_temperature.py, or the training script for CT/MRI), and the report says so.

Reliability per label = min(certainty, stability):
  certainty  sigmoid: |2p - 1| (distance from the decision point)
             softmax: 1 - entropy / log(K)
  stability  1 - std(score over jittered inputs) / 0.15, floored at 0
Thresholds are starting values and are NOT clinically validated."""
import numpy as np

from config import (DETECT_THRESHOLD, LOW_RELIABILITY, POSSIBLE_THRESHOLD,
                    TEMPERATURE, TEMPERATURE_FITTED)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _status(prob, thr):
    return ("detected" if prob >= thr["detect"]
            else "possible" if prob >= thr["possible"] else "not_detected")


def assess(pred: dict) -> dict:
    thr = {"detect": DETECT_THRESHOLD, "possible": POSSIBLE_THRESHOLD}
    thr.update(pred.get("thresholds", {}))
    keep = [i for i, name in enumerate(pred["labels"]) if name]
    labels = [pred["labels"][i] for i in keep]
    logits = np.asarray(pred["logits"], dtype=float)[keep]
    tta = np.asarray(pred["tta_logits"], dtype=float)[:, keep]
    softmax = pred.get("kind") == "softmax"
    T = float(pred.get("temperature") or TEMPERATURE)
    calibrated = bool(pred.get("temperature")) or TEMPERATURE_FITTED
    normal = set(pred.get("normal", []))

    if softmax:
        p = _softmax(logits / T)
        std = _softmax(tta / T, axis=1).std(axis=0)
        certainty_all = float(1 - (-(p * np.log(p + 1e-12)).sum()) / np.log(len(p)))
    else:
        p = _sigmoid(logits / T)
        std = _sigmoid(tta / T).std(axis=0)

    rows = []
    for j, label in enumerate(labels):
        prob, s = float(p[j]), float(std[j])
        certainty = certainty_all if softmax else abs(2 * prob - 1)
        stability = max(0.0, 1 - s / 0.15)
        status = "not_detected" if label in normal else _status(prob, thr)
        rows.append({"label": label, "prob": round(prob, 4), "status": status,
                     "certainty": round(certainty, 3), "stability": round(stability, 3),
                     "reliability": round(min(certainty, stability), 3)})
    rows.sort(key=lambda r: r["prob"], reverse=True)
    top = rows[0]
    score = top["reliability"]
    return {
        "top": top,
        "findings": [r for r in rows if r["status"] != "not_detected"],
        "all": rows,
        "reliability": {"score": score,
                        "level": "adequate" if score >= LOW_RELIABILITY else "low",
                        "certainty": top["certainty"], "stability": top["stability"],
                        "temperature": T, "calibrated": calibrated,
                        "kind": "softmax" if softmax else "sigmoid"},
    }
