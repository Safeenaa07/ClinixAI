"""CT and MRI: fine-tuned torchvision ResNet18 slice classifiers (2D images, one class per scan).
Weights come from evaluation/train_classifier.py (or the Colab notebook):
    models/<scan_type>.pt         state_dict
    models/<scan_type>_meta.json  labels, normal labels, temperature, test metrics
Honest scope: these classify a single 2D image or slice. They are not 3D models."""
import json

import cv2
import numpy as np

from config import MODEL_DIR, USE_MOCK_MODEL

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _tta(gray, n=8, seed=0):
    from services.xray_model import _tta_images
    return _tta_images(gray, n, seed)


class TorchSliceClassifier:
    mock = False

    def __init__(self, scan_type):
        import torch
        import torchvision
        self.torch = torch
        self.scan_type = scan_type
        meta = json.loads((MODEL_DIR / f"{scan_type}_meta.json").read_text())
        self.labels = meta["labels"]
        self.normal = meta.get("normal", [])
        self.temperature = meta.get("temperature")
        self.size = meta.get("img_size", 224)
        self.name = f"ResNet18 {scan_type.upper()} slice classifier (fine-tuned, 2D)"
        net = torchvision.models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(self.labels))
        net.load_state_dict(torch.load(MODEL_DIR / f"{scan_type}.pt", map_location="cpu"))
        self.model = net.eval()

    def _tensor(self, gray):
        img = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        img = (np.stack([img] * 3, -1) - MEAN) / STD
        return self.torch.from_numpy(img.transpose(2, 0, 1).copy())[None]

    def _logits(self, batch):
        with self.torch.no_grad():
            return self.model(batch).numpy()

    def predict(self, gray):
        logits = self._logits(self._tensor(gray))[0]
        tta = self.torch.cat([self._tensor(g) for g in _tta(gray)])
        return {"kind": "softmax", "labels": self.labels, "normal": self.normal,
                "temperature": self.temperature, "logits": logits, "tta_logits": self._logits(tta)}

    def gradcam(self, gray, label):
        import torch.nn.functional as F
        t = self.torch
        idx = self.labels.index(label)
        store = {}
        h = self.model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        try:
            with t.enable_grad():
                out = self.model(self._tensor(gray))
                a = store["a"]
                g = t.autograd.grad(out[0, idx], a)[0]
        finally:
            h.remove()
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1))[0].detach().numpy()
        cam = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
        cam -= cam.min()
        return cam / (cam.max() + 1e-8)


MOCK_LABELS = {"ct": (["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"], ["Normal"]),
               "mri": (["Glioma", "Meningioma", "No tumor", "Pituitary tumor"], ["No tumor"])}


class MockSliceClassifier:
    mock = True

    def __init__(self, scan_type):
        self.labels, self.normal = MOCK_LABELS[scan_type]
        self.name = f"MOCK {scan_type.upper()} MODEL (fake output, pipeline test only)"

    def predict(self, gray):
        rng = np.random.default_rng(int(gray.mean() * 1000) % (2 ** 32))
        logits = rng.normal(0, 1.5, len(self.labels))
        return {"kind": "softmax", "labels": self.labels, "normal": self.normal, "temperature": None,
                "logits": logits, "tta_logits": logits + rng.normal(0, 0.3, (8, len(self.labels)))}

    def gradcam(self, gray, label):
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cam = np.exp(-(((xx - w * 0.5) / (w * 0.2)) ** 2 + ((yy - h * 0.5) / (h * 0.2)) ** 2))
        return cam / cam.max()


def weights_present(scan_type):
    return (MODEL_DIR / f"{scan_type}.pt").exists() and (MODEL_DIR / f"{scan_type}_meta.json").exists()


_cache = {}


def get_classifier(scan_type):
    """Returns the model, or None if no trained weights exist (and mock mode is off)."""
    if scan_type not in _cache:
        if USE_MOCK_MODEL:
            _cache[scan_type] = MockSliceClassifier(scan_type)
        elif weights_present(scan_type):
            try:
                _cache[scan_type] = TorchSliceClassifier(scan_type)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"{scan_type} model could not be loaded ({type(e).__name__}: {str(e)[:120]})") from e
        else:
            return None
    return _cache[scan_type]
