"""Predict a single image:  python src/predict.py path/to/image.jpg"""
import argparse
import json

import torch
from PIL import Image

from common import CLASSES, WEIGHTS, eval_tf, get_device, load_model

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("--weights", default=str(WEIGHTS))
a = ap.parse_args()

device = get_device()
model = load_model(a.weights, device)
x = eval_tf()(Image.open(a.image).convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    probs = torch.softmax(model(x), 1)[0].cpu().tolist()
print(json.dumps({"prediction": CLASSES[int(torch.tensor(probs).argmax())],
                  "probabilities": dict(zip(CLASSES, [round(p, 4) for p in probs]))}, indent=2))
