"""
Abbott Merlin PCS — 27B VQA extractor (Qwen3-VL-27B via HuggingFace transformers).

Drop-in alternative to the 8B VQA backend: loads Qwen3-VL-27B-Instruct with 4-bit
quantization and device_map="auto" so it spreads across all available GPUs.
No PaddleOCR, no regex — the model extracts all fields from the PDF page images.

Model ID is taken from the VQA_MODEL_27B environment variable (default:
Qwen/Qwen3-VL-27B-Instruct).
"""
from pathlib import Path

from extractors.abbott_merlin import (
    detect_report_type,
    extract,
)


def load_model():
    """Load Qwen3-VL-27B with 4-bit quantization spread across all GPUs."""
    from vqa import load_model_27b
    return load_model_27b("cuda")


def extract_with_model(pdf_path: str | Path, model, processor) -> dict:
    """Extract using a pre-loaded 27B model/processor pair."""
    pdf_path = Path(pdf_path)
    report_type = detect_report_type(pdf_path)
    return extract(pdf_path, model, processor, report_type=report_type)
