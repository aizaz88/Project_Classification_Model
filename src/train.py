"""Train MobileNetV2 (transfer learning) on Cat vs Dog, evaluate, save weights, export ONNX."""
import argparse
import copy
import json
import shutil
import time

import torch
import torch.nn as nn

from common import (CLASSES, ROOT, WEIGHTS, build_model, compute_metrics, eval_tf, get_device, make_loader,
                    make_splits, plot_confusion, plot_history, predict_loader, resolve_data_dir, train_tf)
from export_onnx import export


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--epochs_head", type=int, default=3, help="epochs with frozen backbone")
    ap.add_argument("--epochs_ft", type=int, default=4, help="epochs fine-tuning all layers")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_per_class", type=int, default=None, help="use a subset for a quick test run")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = get_device()
    print("Device:", device)
    splits = make_splits(resolve_data_dir(a.data_dir), a.seed, a.max_per_class)
    print({k: len(v) for k, v in splits.items()})
    dl = {"train": make_loader(splits["train"], train_tf(), a.batch_size, True),
          "val": make_loader(splits["val"], eval_tf(), a.batch_size),
          "test": make_loader(splits["test"], eval_tf(), a.batch_size)}

    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best, best_acc = None, 0.0

    phases = [("head", a.epochs_head, 1e-3), ("finetune", a.epochs_ft, 1e-4)]
    for name, epochs, lr in phases:
        for p in model.features.parameters():
            p.requires_grad = name == "finetune"
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            t0 = time.time()
            tl, ta = run_epoch(model, dl["train"], device, loss_fn, opt)
            vl, va = run_epoch(model, dl["val"], device, loss_fn)
            for k, v in zip(hist, (tl, vl, ta, va)):
                hist[k].append(v)
            if va > best_acc:
                best_acc, best = va, copy.deepcopy(model.state_dict())
            print(f"[{name}] epoch {ep + 1}/{epochs}  train {ta:.4f}/{tl:.4f}  val {va:.4f}/{vl:.4f}  ({time.time() - t0:.0f}s)")

    model.load_state_dict(best)
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    print("Saved weights ->", WEIGHTS)

    y, p = predict_loader(model, dl["test"], device)
    m = compute_metrics(y, p)
    m.update(n_train=len(splits["train"]), n_val=len(splits["val"]), n_test=len(splits["test"]),
             epochs=a.epochs_head + a.epochs_ft, epochs_head=a.epochs_head, epochs_ft=a.epochs_ft, best_val_accuracy=best_acc)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(m, indent=2))
    plot_confusion(m["confusion_matrix"], ROOT / "results" / "confusion_matrix.png")
    plot_history(hist, ROOT / "results" / "training_curves.png")
    (ROOT / "models" / "class_names.json").write_text(json.dumps(CLASSES))
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))

    export(WEIGHTS)
    for f in ("confusion_matrix.png", "training_curves.png"):
        shutil.copy(ROOT / "results" / f, ROOT / "web" / "assets" / f)
    shutil.copy(ROOT / "models" / "class_names.json", ROOT / "web" / "model" / "class_names.json")
    print("Done. Next: python scripts/finalize.py --github <your repo url> ...")


if __name__ == "__main__":
    main()
