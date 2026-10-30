"""Re-evaluate saved weights on the held-out test split (same seed => same split)."""
import argparse
import json

from common import (ROOT, WEIGHTS, compute_metrics, eval_tf, get_device, load_model, make_loader, make_splits,
                    plot_confusion, predict_loader, resolve_data_dir)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir"); ap.add_argument("--weights", default=str(WEIGHTS))
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max_per_class", type=int, default=None)
a = ap.parse_args()

device = get_device()
test = make_splits(resolve_data_dir(a.data_dir), a.seed, a.max_per_class)["test"]
model = load_model(a.weights, device)
y, p = predict_loader(model, make_loader(test, eval_tf()), device)
m = compute_metrics(y, p)
print(json.dumps(m, indent=2))
plot_confusion(m["confusion_matrix"], ROOT / "results" / "confusion_matrix_eval.png")
