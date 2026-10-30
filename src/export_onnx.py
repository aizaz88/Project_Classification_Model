"""Export trained PyTorch weights to ONNX for in-browser inference (onnxruntime-web)."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import IMG, ROOT, WEIGHTS, load_model


def export(weights=WEIGHTS, out=ROOT / "web" / "model" / "catdog.onnx"):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(weights, torch.device("cpu"))
    dummy = torch.randn(1, 3, IMG, IMG)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(model, dummy, str(out), **kw)
    # sanity check: ONNX output must match PyTorch
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    a = sess.run(None, {"input": dummy.numpy()})[0]
    b = model(dummy).detach().numpy()
    assert np.allclose(a, b, atol=1e-3), "ONNX and PyTorch outputs differ!"
    print(f"Exported {out} ({out.stat().st_size / 1e6:.1f} MB) - outputs match PyTorch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    export(weights=ap.parse_args().weights)
