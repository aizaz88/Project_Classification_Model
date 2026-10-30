"""Fill links and metrics into the README, website, report, script and LinkedIn post.

Example:
  python scripts/finalize.py --github https://github.com/USER/REPO --vercel https://REPO.vercel.app \
      --youtube https://youtu.be/XXXX --linkedin https://www.linkedin.com/posts/XXXX
Run it again whenever a link changes (templates are never modified).
"""
import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--github", required=True)
ap.add_argument("--vercel", default="#")
ap.add_argument("--youtube", default="#")
ap.add_argument("--linkedin", default="#")
ap.add_argument("--report")
ap.add_argument("--slides")
ap.add_argument("--dataset", default="https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset")
a = ap.parse_args()

gh = a.github.rstrip("/")
m = json.loads((ROOT / "results" / "metrics.json").read_text())
pct = lambda v: f"{v * 100:.2f}"
tokens = {
    "__GITHUB_URL__": gh, "__VERCEL_URL__": a.vercel, "__YOUTUBE_URL__": a.youtube,
    "__LINKEDIN_URL__": a.linkedin, "__DATASET_URL__": a.dataset,
    "__REPORT_URL__": a.report or f"{gh}/blob/main/report/REPORT.md",
    "__SLIDES_URL__": a.slides or f"{gh}/blob/main/presentation/Project_Presentation.pptx",
    "__ACC__": pct(m["accuracy"]), "__PREC__": pct(m["precision"]), "__REC__": pct(m["recall"]), "__F1__": pct(m["f1"]),
    "__N_TRAIN__": str(m["n_train"]), "__N_VAL__": str(m["n_val"]), "__N_TEST__": str(m["n_test"]),
    "__EPOCHS_HEAD__": str(m.get("epochs_head", 3)), "__EPOCHS_FT__": str(m.get("epochs_ft", 4)),
    "__DATE__": datetime.date.today().strftime("%d %B %Y"),
}
jobs = [("web/index.template.html", "web/index.html"), ("README.template.md", "README.md"),
        ("report/REPORT_TEMPLATE.md", "report/REPORT.md"), ("presentation/SCRIPT_TEMPLATE.md", "presentation/SCRIPT.md"),
        ("presentation/LINKEDIN_TEMPLATE.txt", "presentation/LINKEDIN_POST.txt")]
for src, dst in jobs:
    text = (ROOT / src).read_text(encoding="utf-8")
    for k, v in tokens.items():
        text = text.replace(k, v)
    (ROOT / dst).write_text(text, encoding="utf-8")
    print("wrote", dst)
