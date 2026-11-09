"""Train a CT or MRI slice classifier (ResNet18, 2D) from image folders.

Folder layout (torchvision ImageFolder): <dir>/<class_name>/*.jpg
  python evaluation/train_classifier.py --name mri --train Training --test Testing
  python evaluation/train_classifier.py --name ct  --train train --val valid --test test

Without --val, 15% of --train is held out for validation. Validation is used for model
selection and to fit the calibration temperature; --test is touched once at the end.
Writes backend/models/<name>.pt and <name>_meta.json (labels, normal labels, temperature,
test metrics) and evaluation/results_<name>.json. Pretrained ImageNet weights are used if they
can be downloaded; use --no-pretrained to skip.
Honest limit: images from the same patient may appear in both train and test folders in public
datasets, which can make results look better than they are."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from scipy.optimize import minimize_scalar
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

import _common  # noqa: F401
from config import MODEL_DIR

CLEAN = {
    "ct": [("adenocarcinoma", "Adenocarcinoma"), ("large.cell", "Large cell carcinoma"),
           ("large_cell", "Large cell carcinoma"), ("squamous", "Squamous cell carcinoma"), ("normal", "Normal")],
    "mri": [("glioma", "Glioma"), ("meningioma", "Meningioma"), ("pituitary", "Pituitary tumor"),
            ("notumor", "No tumor"), ("no_tumor", "No tumor"), ("normal", "No tumor")],
}
NORMAL = {"Normal", "No tumor"}


def clean(name, folder):
    for key, label in CLEAN.get(name, []):
        if key in folder.lower():
            return label
    return folder


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def brier(p, y):
    return float(np.mean(((p - np.eye(p.shape[1])[y]) ** 2).sum(1)))


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, choices=["ct", "mri"])
ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--val"); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.RandomRotation(10),
                      T.ColorJitter(0.2, 0.2), T.ToTensor(), norm])
tf_eval = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor(), norm])

full_train = torchvision.datasets.ImageFolder(a.train, tf_train)
classes = full_train.classes
labels = [clean(a.name, c) for c in classes]
if len(set(labels)) != len(labels):
    raise SystemExit(f"Class names collapse to duplicates: {labels}")
print("classes:", dict(zip(classes, labels)))
if a.val:
    train_ds, val_ds = full_train, torchvision.datasets.ImageFolder(a.val, tf_eval)
    assert val_ds.classes == classes, "train and val class folders differ"
else:
    idx = np.random.permutation(len(full_train)); cut = int(0.85 * len(idx))
    train_ds = Subset(full_train, idx[:cut].tolist())
    val_ds = Subset(torchvision.datasets.ImageFolder(a.train, tf_eval), idx[cut:].tolist())
test_ds = torchvision.datasets.ImageFolder(a.test, tf_eval)
assert test_ds.classes == classes, "train and test class folders differ"
print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}  device {dev}")

try:
    net = torchvision.models.resnet18(weights=None if a.no_pretrained else "IMAGENET1K_V1")
except Exception as e:  # noqa: BLE001
    print("Pretrained weights unavailable (", type(e).__name__, "), training from scratch.")
    net = torchvision.models.resnet18(weights=None)
net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
net.to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch, shuffle=sh, num_workers=0)  # noqa: E731


def run_logits(ds):
    net.eval(); zs, ys = [], []
    with torch.no_grad():
        for x, y in mk(ds, False):
            zs.append(net(x.to(dev)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


best, best_state = -1, None
for ep in range(a.epochs):
    net.train(); tot = 0
    for x, y in mk(train_ds, True):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x.to(dev)), y.to(dev))
        loss.backward(); opt.step(); tot += loss.item() * len(y)
    zv, yv = run_logits(val_ds)
    acc = float((zv.argmax(1) == yv).mean())
    print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(train_ds):.4f} val acc {acc:.4f}")
    if acc >= best:
        best, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)

zv, yv = run_logits(val_ds)
nll = lambda t: -np.mean(np.log(softmax(zv / t)[np.arange(len(yv)), yv] + 1e-12))  # noqa: E731
Tm = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)

zt, yt = run_logits(test_ds)
p0, p1 = softmax(zt), softmax(zt / Tm)
pred = p1.argmax(1)
res = {
    "name": a.name, "labels": labels, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
    "temperature": round(Tm, 4),
    "accuracy": round(float((pred == yt).mean()), 4),
    "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
    "ece_before": round(ece(p0, yt), 4), "ece_after": round(ece(p1, yt), 4),
    "brier_before": round(brier(p0, yt), 4), "brier_after": round(brier(p1, yt), 4),
    "per_class": classification_report(yt, pred, target_names=labels, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(yt, pred).tolist(),
}
print(classification_report(yt, pred, target_names=labels, zero_division=0))
print("confusion matrix (rows = true):\n", np.array(res["confusion_matrix"]))
print(f"accuracy {res['accuracy']}  macro-F1 {res['macro_f1']}  T {res['temperature']}")
print(f"ECE {res['ece_before']} -> {res['ece_after']}   Brier {res['brier_before']} -> {res['brier_after']}")

MODEL_DIR.mkdir(exist_ok=True)
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, MODEL_DIR / f"{a.name}.pt")
(MODEL_DIR / f"{a.name}_meta.json").write_text(json.dumps({
    "labels": labels, "normal": [l for l in labels if l in NORMAL], "arch": "resnet18", "img_size": 224,
    "temperature": res["temperature"], "test_accuracy": res["accuracy"], "test_macro_f1": res["macro_f1"]}, indent=1))
(Path(__file__).parent / f"results_{a.name}.json").write_text(json.dumps(res, indent=1))
print("saved", MODEL_DIR / f"{a.name}.pt")
