"""Shared helpers: data discovery, splits, transforms, model, evaluation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Cat", "Dog"]
MEAN, STD, IMG = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
EXT = {".jpg", ".jpeg", ".png"}
WEIGHTS = ROOT / "models" / "catdog_mobilenetv2.pth"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_data_dir(arg=None):
    if arg:
        return arg
    f = ROOT / "data" / "dataset_path.txt"
    if f.exists():
        return f.read_text().strip()
    raise SystemExit("No dataset found. Run: python src/download_data.py  (or pass --data_dir)")


def _valid(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def make_splits(data_dir, seed=42, max_per_class=None):
    """Deterministic stratified 70/15/15 split. Corrupt images are dropped."""
    items = []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() in EXT:
            n = p.parent.name.lower()
            if n in ("cat", "cats"):
                items.append((str(p), 0))
            elif n in ("dog", "dogs"):
                items.append((str(p), 1))
    if not items:
        raise SystemExit(f"No Cat/Dog images found under {data_dir}")
    if max_per_class:
        items = [x for c in (0, 1) for x in [i for i in items if i[1] == c][:max_per_class]]
    items = [x for x in items if _valid(x[0])]
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    p_tr, p_te, l_tr, l_te = train_test_split(paths, labels, test_size=0.15, stratify=labels, random_state=seed)
    p_tr, p_va, l_tr, l_va = train_test_split(p_tr, l_tr, test_size=0.15 / 0.85, stratify=l_tr, random_state=seed)
    return {"train": list(zip(p_tr, l_tr)), "val": list(zip(p_va, l_va)), "test": list(zip(p_te, l_te))}


def train_tf():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_tf():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


class ImageDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:  # unreadable file: fall back to the next one
            return self[(i + 1) % len(self)]
        return self.tf(img), label


def make_loader(items, tf, bs=32, shuffle=False):
    return DataLoader(ImageDS(items, tf), batch_size=bs, shuffle=shuffle, num_workers=2)


def build_model(pretrained=True):
    w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=w)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, len(CLASSES)))
    return m


def load_model(weights=WEIGHTS, device=None):
    device = device or get_device()
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return np.array(ys), np.array(ps)


def compute_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(pr.mean()), "recall": float(rc.mean()), "f1": float(f1.mean()),
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


def plot_confusion(cm, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(h["train_loss"], label="train"); ax[0].plot(h["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(h["train_acc"], label="train"); ax[1].plot(h["val_acc"], label="val"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
