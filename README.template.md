# Cat vs Dog Classifier (Deep Learning Semester Project)

| | |
|---|---|
| Live demo (Vercel) | __VERCEL_URL__ |
| Project report | __REPORT_URL__ |
| Presentation slides | __SLIDES_URL__ |
| Presentation video | __YOUTUBE_URL__ |
| LinkedIn post | __LINKEDIN_URL__ |
| Dataset | __DATASET_URL__ |

**Task:** binary image classification (cat / dog). **Model:** MobileNetV2 (ImageNet pre-trained) fine-tuned with PyTorch.
**Test results:** accuracy __ACC__%, precision __PREC__%, recall __REC__%, F1 __F1__% on __N_TEST__ unseen images.

## Repository layout
```
src/            training, evaluation, prediction, ONNX export
models/         trained weights: catdog_mobilenetv2.pth, class_names.json
web/            static website (index.html, app.js, style.css, model/catdog.onnx)
results/        metrics.json, confusion matrix, training curves
report/         REPORT.md
presentation/   slides (.pptx) and voice-over script
data/           dataset download instructions (data is not committed)
scripts/        finalize.py (fills links/metrics), make_pptx.py
```

## Run it yourself
```bash
pip install -r requirements.txt
python src/download_data.py            # downloads the Kaggle dataset
python src/train.py                    # trains, saves weights, exports ONNX
python src/evaluate.py                 # test-set metrics from saved weights
python src/predict.py path/to/pet.jpg  # single image prediction
```

## Use the trained weights directly
```python
import sys; sys.path.insert(0, "src")
from common import load_model, eval_tf, CLASSES
import torch
from PIL import Image
model = load_model("models/catdog_mobilenetv2.pth")
x = eval_tf()(Image.open("pet.jpg").convert("RGB")).unsqueeze(0)
print(CLASSES[model(x).argmax(1).item()])
```
Input: RGB image, resized to 256, centre-cropped to 224, normalised with ImageNet mean/std. Output: logits for [Cat, Dog].
The browser model is `web/model/catdog.onnx` (input `input` 1x3x224x224, output `logits`).

## Deployment
Static site in `web/`, deployed on Vercel with Root Directory set to `web`. Inference runs in the browser with ONNX Runtime Web.
