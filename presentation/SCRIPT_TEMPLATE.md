# Slides and voice-over script (about 5 minutes)

Record each block while showing the matching slide. Speak calmly, about 130 words per minute.

**Slide 1 - Title.** "Hello, I am YOUR NAME. This is my Deep Learning semester project: a classifier that tells cats from dogs, deployed as a live web app."

**Slide 2 - Problem.** "The task is binary image classification. Given a photo, the model must say cat or dog. It is a classic way to study transfer learning, and it has real uses in pet apps and shelter databases."

**Slide 3 - Dataset.** "I used a secondary dataset from Kaggle with about 25,000 photos, balanced between the two classes. I removed unreadable files and split the data 70, 15, 15 into __N_TRAIN__ training, __N_VAL__ validation and __N_TEST__ test images, with a fixed seed so the split is reproducible."

**Slide 4 - Model.** "I chose MobileNetV2 pre-trained on ImageNet. It is small, about 9 megabytes, so it runs inside a browser. I replaced the last layer with dropout and a two-output linear layer."

**Slide 5 - Training.** "Training had two phases. First I trained only the new head for __EPOCHS_HEAD__ epochs while the backbone was frozen. Then I fine-tuned all layers for __EPOCHS_FT__ epochs with a smaller learning rate. I used AdamW, data augmentation, and kept the best epoch by validation accuracy."

**Slide 6 - Results.** "On the test set, the model reaches __ACC__ percent accuracy, with precision __PREC__, recall __REC__ and F1 __F1__ percent. The confusion matrix shows where the few mistakes happen."

**Slide 7 - Deployment.** "I exported the weights to ONNX and built a static website on Vercel. The model runs in the browser with ONNX Runtime Web, so no server is needed and photos never leave the user's device."

**Slide 8 - Links and conclusion.** "Everything is open: code, weights and dataset link are on GitHub, and the report, slides and this video are linked from the website. Future work is adding an 'other' class and breed recognition. Thank you."
