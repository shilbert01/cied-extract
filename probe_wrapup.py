import sys
sys.path.insert(0, "/home/herzzentrum/CiedOCR")
import pdf2image, torch
from vqa import load_model
from extractors.abbott_merlin import _vqa

PROBE = """List ALL section headings and the German label/value pairs you can read on this page, grouped by section.
Include every label and value even if "Aus", "Nicht durchgeführt", or a number.
Format:
SECTION: <heading>
  <label>: <value>
"""

model, processor = load_model("cuda")

# Probe Quadra Assura page 5 (Wrap-up) and Entrant VR page 1 parameters
path_qa = "/home/herzzentrum/CiedOCR/samples/abbott/quadra_assura/SummaryReport/001/report.pdf"
imgs = pdf2image.convert_from_path(path_qa, dpi=200)
print("\n=== Quadra Assura — Page 5 (Wrap-up) ===")
print(_vqa(model, processor, imgs[4], PROBE))
