import sys
sys.path.insert(0, "/home/herzzentrum/CiedOCR")
import pdf2image, torch
from vqa import load_model
from extractors.abbott_merlin import _vqa

PROBE = "What type of report section is this page? Name the main sections/headings visible. Be brief (2-3 lines)."

model, processor = load_model("cuda")

for name, path, npages in [
    ("Entrant VR", "/home/herzzentrum/CiedOCR/samples/abbott/entrant_vr/SummaryReport/001/report.pdf", 4),
    ("Quadra Assura", "/home/herzzentrum/CiedOCR/samples/abbott/quadra_assura/SummaryReport/001/report.pdf", 5),
]:
    print(f"\n{'='*50}\n=== {name} ({npages} pages) ===")
    imgs = pdf2image.convert_from_path(path, dpi=150)
    for i, img in enumerate(imgs):
        raw = _vqa(model, processor, img, PROBE)
        print(f"  Page {i+1}: {raw.strip()}")
