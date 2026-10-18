#!/usr/bin/env python3
"""
CiedOCR — Pacemaker interrogation report parser.

Usage:
    python main.py report.pdf
    python main.py report.pdf --output result.json
    python main.py report.pdf --no-model          # dry-run using pdfplumber fallback
    python main.py *.pdf --output-dir ./results/
"""
import argparse
import json
import sys
from pathlib import Path

import extractor


def parse_args():
    p = argparse.ArgumentParser(
        description="OCR and extract parameters from pacemaker interrogation PDFs."
    )
    p.add_argument("pdfs", nargs="+", metavar="PDF", help="Input PDF file(s)")
    p.add_argument("--output", "-o", metavar="FILE",
                   help="Write JSON output to FILE (single PDF only)")
    p.add_argument("--output-dir", "-d", metavar="DIR",
                   help="Write one JSON per PDF into this directory")
    p.add_argument("--dpi", type=int, default=200,
                   help="DPI for PDF→image conversion (default: 200)")
    p.add_argument("--device", choices=["cuda", "cpu"],
                   help="Force device (default: auto-detect)")
    p.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6",
                   help="HuggingFace model ID or local path")
    p.add_argument("--cache-dir", metavar="DIR",
                   help="HuggingFace model cache directory")
    p.add_argument("--no-model", action="store_true",
                   help="Skip OCR; extract text from PDF directly (fast, no GPU needed)")
    p.add_argument("--raw", action="store_true",
                   help="Also print raw OCR text for debugging")
    return p.parse_args()


def ocr_pdf(pdf_path: Path, args) -> list[str]:
    """Return a list of OCR text strings, one per PDF page."""
    if args.no_model:
        return _pdfplumber_extract(pdf_path)

    from ocr_engine import OCREngine
    engine = OCREngine(
        model_id=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    return engine.ocr_pdf(pdf_path, dpi=args.dpi)


def _pdfplumber_extract(pdf_path: Path) -> list[str]:
    """Fallback: extract embedded text from PDF (no model required)."""
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text(layout=True) or ""
                pages.append(text)
            return pages
    except ImportError:
        print("pdfplumber not installed — install it or remove --no-model", file=sys.stderr)
        sys.exit(1)


def process_pdf(pdf_path: Path, args) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")

    ocr_pages = ocr_pdf(pdf_path, args)

    if args.raw:
        for i, page_text in enumerate(ocr_pages, start=1):
            print(f"\n--- Page {i} raw OCR ---\n{page_text}\n")

    report = extractor.extract(ocr_pages, source_file=str(pdf_path))

    # Remove raw pages from output unless --raw requested
    result = report.to_dict()
    if not args.raw:
        result.pop("raw_ocr_pages", None)

    return result


def main():
    args = parse_args()

    if args.output and len(args.pdfs) > 1:
        print("Use --output-dir when processing multiple PDFs.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for pdf_str in args.pdfs:
        pdf_path = Path(pdf_str)
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}", file=sys.stderr)
            continue

        result = process_pdf(pdf_path, args)
        all_results.append(result)

        # Per-file output
        if output_dir:
            out_path = output_dir / (pdf_path.stem + ".json")
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"Saved: {out_path}")

    # Single-file output
    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(all_results[0], ensure_ascii=False, indent=2))
        print(f"\nSaved: {out}")
    elif not output_dir:
        # Print to stdout
        payload = all_results[0] if len(all_results) == 1 else all_results
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
