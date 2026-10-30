"""Build presentation/Project_Presentation.pptx from results/metrics.json (run finalize.py first is not required)."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results" / "metrics.json").read_text())
pct = lambda v: f"{v * 100:.2f}%"
INK, TEAL = RGBColor(0x12, 0x22, 0x2B), RGBColor(0x0B, 0x7A, 0x75)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def slide(title, bullets=(), image=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.7), Inches(0.5), Inches(12), Inches(1.1)).text_frame
    t.text = title
    t.paragraphs[0].font.size, t.paragraphs[0].font.bold, t.paragraphs[0].font.color.rgb = Pt(38), True, TEAL
    width = Inches(6.2) if image else Inches(12)
    b = s.shapes.add_textbox(Inches(0.7), Inches(1.9), width, Inches(5)).text_frame
    b.word_wrap = True
    for i, line in enumerate(bullets):
        p = b.paragraphs[0] if i == 0 else b.add_paragraph()
        p.text, p.font.size, p.font.color.rgb = "• " + line, Pt(22), INK
        p.space_after = Pt(10)
    if image and Path(image).exists():
        s.shapes.add_picture(str(image), Inches(7.3), Inches(1.8), height=Inches(4.8))


slide("Cat vs Dog Classifier", ["Deep Learning semester project", "Transfer learning with MobileNetV2, deployed on Vercel", "Your Name | Roll No"])
slide("Problem", ["Binary image classification: cat or dog", "A standard benchmark for transfer learning", "Uses: pet apps, shelter databases, photo organising"])
slide("Dataset", ["Kaggle Dog and Cat Classification Dataset, about 25,000 images",
                  f"Split 70/15/15: {m['n_train']} train, {m['n_val']} validation, {m['n_test']} test",
                  "Corrupt files removed; augmentation on training images only"])
slide("Model", ["MobileNetV2 pre-trained on ImageNet", "New head: Dropout(0.3) + Linear(1280, 2)", "Small (9 MB) so it runs in the browser"])
slide("Training", [f"Phase 1: {m.get('epochs_head', 3)} epochs, backbone frozen, lr 1e-3",
                   f"Phase 2: {m.get('epochs_ft', 4)} epochs, all layers, lr 1e-4",
                   "AdamW, cross-entropy, best epoch by validation accuracy"], ROOT / "results" / "training_curves.png")
slide("Results (test set)", [f"Accuracy {pct(m['accuracy'])}", f"Precision {pct(m['precision'])}", f"Recall {pct(m['recall'])}",
                             f"F1 {pct(m['f1'])}"], ROOT / "results" / "confusion_matrix.png")
slide("Deployment", ["PyTorch weights exported to ONNX", "Static website on Vercel", "ONNX Runtime Web runs the model on the user's device"])
slide("Links", ["GitHub: add URL", "Live demo: add URL", "Report and video: add URLs", "Thank you"])
out = ROOT / "presentation" / "Project_Presentation.pptx"
prs.save(out)
print("saved", out)
