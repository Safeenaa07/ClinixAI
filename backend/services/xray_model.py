"""X-ray model. Real mode uses torchxrayvision's DenseNet121 (pretrained on public chest
X-ray datasets). Mock mode returns FAKE numbers so the pipeline can be tested without torch."""
import cv2
import numpy as np

import os

from config import USE_MOCK_MODEL

# PadChest-trained weights by default: they cover all 14 NIH labels plus Fracture and were NOT
# trained on NIH, so evaluating on the NIH sample is an external test. Use
# "densenet121-res224-all" for a stronger demo model (it saw NIH, so do not evaluate on NIH).
XRV_WEIGHTS = os.getenv("CLINIX_XRV_WEIGHTS", "densenet121-res224-pc")

XRV_LABELS = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
              "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
              "Mass", "Hernia", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _tta_images(gray, n=8, seed=0):
    """Small shifts and brightness/contrast jitter, used to measure prediction stability."""
    rng = np.random.default_rng(seed)
    h, w = gray.shape
    out = []
    for _ in range(n):
        dx = int(rng.integers(-(w // 30), w // 30 + 1))
        dy = int(rng.integers(-(h // 30), h // 30 + 1))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        im = cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        a, b = rng.uniform(0.9, 1.1), rng.uniform(-10, 10)
        out.append(np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8))
    return out


class TorchXrayVisionModel:
    """Outputs are the library's operating-point-normalised scores: 0.5 is the library's
    operating point for that finding. They are NOT clinical probabilities."""
    mock = False

    def __init__(self, weights=None):
        import torch
        import torchxrayvision as xrv
        self.torch, self.xrv = torch, xrv
        self.weights = weights or XRV_WEIGHTS
        self.name = f"torchxrayvision DenseNet121 ({self.weights})"
        self.model = xrv.models.DenseNet(weights=self.weights).eval()
        self.labels = list(self.model.pathologies)

    def batch_logits(self, grays, batch=32):
        """Single pass (no jitter) for many images. Used by evaluation scripts."""
        out = []
        for i in range(0, len(grays), batch):
            x = self.torch.cat([self._tensor(g) for g in grays[i:i + batch]])
            out.append(self._logits(x))
        return np.concatenate(out)

    def _tensor(self, gray):
        img = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        img = self.xrv.datasets.normalize(img, 255)
        return self.torch.from_numpy(img)[None, None]

    def _logits(self, batch):
        with self.torch.no_grad():
            out = self.model(batch).numpy()
        # The library may return probabilities or raw logits depending on version.
        return out if (out.min() < 0 or out.max() > 1) else _logit(out)

    def predict(self, gray):
        logits = self._logits(self._tensor(gray))[0]
        tta = self.torch.cat([self._tensor(g) for g in _tta_images(gray)])
        return {"labels": self.labels, "logits": logits, "tta_logits": self._logits(tta)}

    def gradcam(self, gray, label):
        import torch.nn.functional as F
        t = self.torch
        idx = self.labels.index(label)
        store = {}
        with t.enable_grad():
            feats = F.relu(self.model.features(self._tensor(gray)))
            feats.register_hook(lambda g: store.__setitem__("g", g))
            out = self.model.classifier(F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1))
            self.model.zero_grad()
            out[0, idx].backward()
        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1))[0].detach().numpy()
        cam = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
        cam -= cam.min()
        return cam / (cam.max() + 1e-8)


class MockXrayModel:
    name = "MOCK MODEL (fake output, pipeline test only)"
    mock = True
    labels = XRV_LABELS

    def predict(self, gray):
        rng = np.random.default_rng(int(gray.mean() * 1000) % (2 ** 32))
        logits = rng.normal(-1.5, 1.6, len(self.labels))
        tta = logits + rng.normal(0, 0.35, (8, len(self.labels)))
        return {"labels": self.labels, "logits": logits, "tta_logits": tta}

    def gradcam(self, gray, label):
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cam = np.exp(-(((xx - w * 0.6) / (w * 0.15)) ** 2 + ((yy - h * 0.45) / (h * 0.15)) ** 2))
        return cam / cam.max()


_model = None


def get_xray_model():
    global _model
    if _model is None:
        if USE_MOCK_MODEL:
            _model = MockXrayModel()
        else:
            try:
                _model = TorchXrayVisionModel()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "X-ray model could not be loaded (" + type(e).__name__ + ": " + str(e)[:120] +
                    "). Install requirements-model.txt, or set CLINIX_MOCK_MODEL=1 to test the "
                    "pipeline with FAKE output.") from e
    return _model
