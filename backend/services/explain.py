"""Module 5: Grad-CAM overlay as a base64 PNG."""
import base64

import cv2
import numpy as np


def _png_uri(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def overlay(gray: np.ndarray, cam: np.ndarray) -> str:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _png_uri(cv2.addWeighted(base, 0.6, heat, 0.4, 0))


def original(gray: np.ndarray) -> str:
    return _png_uri(gray)
