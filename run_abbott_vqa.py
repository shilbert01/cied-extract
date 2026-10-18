#!/usr/bin/env python3
"""
Run VQA extraction for all Abbott Merlin image-based samples.
Requires CUDA — run on bigboy.

Usage:
    python3 run_abbott_vqa.py [--dpi 300] [--output-dir results/]
"""
import argparse
import json
import sys
from pathlib import Path

# ── model load (once) ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--output-dir", default="results/abbott_vqa")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from vqa import load_model
    from extractors.abbott_merlin import extract, detect_report_type
    from validate import score

    model, processor = load_model("cuda")

    samples = [
        ("samples/abbott/aveir_dr_lp/FreezeReport/001/report.pdf", "freeze"),
        ("samples/abbott/aveir_dr_lp/SummaryReport/001/report.pdf", "summary"),
        ("samples/abbott/entrant_dr/FastPathSummary/001/report.pdf", "fastpath_icd"),
        ("samples/abbott/endurity_mri_2172/FastPathSummary/001/report.pdf", "fastpath_pm"),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path, expected_type in samples:
        pdf = Path(pdf_path)
        print(f"\n{'='*60}")
        print(f"  {pdf_path}")
        detected = detect_report_type(pdf_path)
        print(f"  report_type: {detected}")

        result = extract(pdf, model, processor, dpi=args.dpi, report_type=detected)

        out_file = out_dir / f"{pdf.parent.parent.parent.name}_{pdf.parent.parent.name}_{detected}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"  → saved {out_file}")

        if args.verbose:
            for k, v in sorted(result.items()):
                print(f"    {k}: {v!r}")

        # Score against ground truth if available
        gt_path = pdf.parent / "ground_truth.json"
        if gt_path.exists():
            truth = json.loads(gt_path.read_text())
            from validate import score
            match, total, misses = score(result, truth)
            pct = 100 * match // total if total else 0
            print(f"  score: {match}/{total} ({pct}%)")
            for field, got, expected in misses:
                print(f"    ✗ {field:<35} got={got!r}  expected={expected!r}")

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
