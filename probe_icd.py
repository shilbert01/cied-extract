import sys
sys.path.insert(0, "/home/herzzentrum/CiedOCR")
import pdf2image, torch
from vqa import load_model
from extractors.abbott_merlin import _vqa

PROBE = """You are looking at page 1 of an Abbott Merlin PCS FastPath Uebersicht report for an ICD device.
List ALL section headings and the German label/value pairs you can read, grouped by section.
Include every label and value even if the value is "Aus", "Nicht durchgefuhrt", or a number.
Format:
SECTION: <heading>
  <label>: <value>
"""

model, processor = load_model("cuda")

for name, path in [
    ("Entrant VR", "/home/herzzentrum/CiedOCR/samples/abbott/entrant_vr/SummaryReport/001/report.pdf"),
    ("Quadra Assura", "/home/herzzentrum/CiedOCR/samples/abbott/quadra_assura/SummaryReport/001/report.pdf"),
]:
    imgs = pdf2image.convert_from_path(path, dpi=200, first_page=1, last_page=1)
    raw = _vqa(model, processor, imgs[0], PROBE)
    print(f"\n{'='*60}\n=== {name} ===\n{'='*60}")
    print(raw)
