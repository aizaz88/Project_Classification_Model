# Cat vs Dog Image Classification with Transfer Learning

**Course:** Deep Learning  |  **Author:** YOUR NAME (ROLL NO)  |  **Date:** __DATE__

Links: [GitHub](__GITHUB_URL__) · [Live demo](__VERCEL_URL__) · [Video](__YOUTUBE_URL__) · [Slides](__SLIDES_URL__)

## 1. Abstract
This project builds an image classifier that tells cats from dogs. A MobileNetV2 network pre-trained on ImageNet is fine-tuned on about 25,000 photos and reaches __ACC__% accuracy on a held-out test set of __N_TEST__ images. The trained model is exported to ONNX and deployed as a web page where it runs entirely in the browser.

## 2. Problem statement
Given a colour photograph, predict whether it shows a cat or a dog. This is a binary image classification task. It is a standard benchmark for studying transfer learning and has practical uses in pet-care apps, shelter databases and photo organisation.

## 3. Dataset
- **Source:** Dog and Cat Classification Dataset, Kaggle (secondary dataset): __DATASET_URL__
- **Size:** about 25,000 JPEG images, two classes (Cat, Dog), roughly balanced.
- **Cleaning:** files that cannot be opened are removed.
- **Split:** stratified 70 / 15 / 15 with seed 42: __N_TRAIN__ train, __N_VAL__ validation, __N_TEST__ test.
- **Augmentation (train only):** RandomResizedCrop(224, scale 0.7-1.0), horizontal flip, colour jitter (0.2). Evaluation images use Resize(256) and CenterCrop(224). All images are normalised with ImageNet mean and standard deviation.

## 4. Model architecture
MobileNetV2 is a light CNN built from inverted residual blocks with depthwise separable convolutions, which keeps the model small (about 2.2M parameters, 9 MB) and fast enough for the browser. The ImageNet classifier is replaced with Dropout(0.3) followed by Linear(1280, 2).

Transfer learning is used because ImageNet features (edges, textures, fur, eyes) transfer well to pets, so far less data and compute are needed than training from scratch.

## 5. Training details
| Setting | Value |
|---|---|
| Framework | PyTorch / torchvision |
| Phase 1 | __EPOCHS_HEAD__ epochs, backbone frozen, lr 1e-3 |
| Phase 2 | __EPOCHS_FT__ epochs, all layers trainable, lr 1e-4 |
| Optimiser | AdamW, weight decay 1e-4 |
| Loss | Cross-entropy |
| Batch size | 32 |
| Model selection | Best validation accuracy |
| Seed | 42 |

![Training curves](../results/training_curves.png)

## 6. Evaluation and results
Metrics on the untouched test set (precision, recall and F1 are macro averages):

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| __ACC__% | __PREC__% | __REC__% | __F1__% |

![Confusion matrix](../results/confusion_matrix.png)

**Discussion.** Add 3-5 sentences here: which class is confused more often, whether the training and validation curves show over-fitting, and what kinds of images fail (cropped faces, unusual poses, cartoons, several animals in one photo).

## 7. Deployment
The PyTorch weights (`models/catdog_mobilenetv2.pth`) are exported to ONNX and checked against the PyTorch output. The web page (`web/`) is a static site on Vercel. It loads the ONNX model with ONNX Runtime Web, so predictions run on the visitor's device and no server is needed.

## 8. Limitations and future work
The model knows only two classes, so any other animal gets a cat or dog answer. Future work: add an "other" class, try larger backbones such as EfficientNet, apply test-time augmentation, and extend to breed classification.

## 9. Reproducibility
```
pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/evaluate.py
python src/predict.py some_image.jpg
```

## 10. References
1. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
2. Deng et al., "ImageNet: A large-scale hierarchical image database", CVPR 2009.
3. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019.
4. Kaggle dataset: __DATASET_URL__
