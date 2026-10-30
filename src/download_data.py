"""Download the Kaggle Cat/Dog dataset and remember where it lives."""
import kagglehub
from common import ROOT

path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "dataset_path.txt").write_text(str(path))
print("Dataset downloaded to:", path)
