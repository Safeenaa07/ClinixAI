"""Modality router: one model per scan type.
  xray -> torchxrayvision DenseNet121 (multi-label findings)
  ct, mri -> fine-tuned ResNet18 slice classifiers, available once models/<type>.pt exists
Add another scan type by registering a factory that returns an object with .name, .mock,
.predict(gray) and .gradcam(gray, label)."""
from services import image_classifier
from services.xray_model import get_xray_model

_REGISTRY = {
    "xray": get_xray_model,
    "ct": lambda: image_classifier.get_classifier("ct"),
    "mri": lambda: image_classifier.get_classifier("mri"),
}


def get_model(scan_type: str):
    factory = _REGISTRY.get(scan_type)
    return factory() if factory else None


def available() -> dict:
    from config import USE_MOCK_MODEL
    return {"xray": True,
            "ct": USE_MOCK_MODEL or image_classifier.weights_present("ct"),
            "mri": USE_MOCK_MODEL or image_classifier.weights_present("mri")}
