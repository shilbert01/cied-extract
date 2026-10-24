#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate all annotated samples against their respective extractors.

Usage:
    python validate.py                          # all samples
    python validate.py --vendor biotronik       # one manufacturer
    python validate.py samples/abbott/.../001   # single sample dir
    python validate.py --device cpu             # force CPU (no GPU)
    python validate.py --backend ocr            # use EasyOCR instead of VQA
    python validate.py --backend auto           # VQA if CUDA available, else OCR
"""
import argparse
import json
import sys
from pathlib import Path


# ── scoring ───────────────────────────────────────────────────────────────────

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def score(extracted: dict, truth: dict) -> tuple[int, int, list]:
    ef = _flatten(extracted)
    tf = _flatten(truth)
    keys = sorted(set(tf))
    match = total = 0
    misses = []
    for k in keys:
        tv = str(tf[k]) if tf.get(k) is not None else None
        ev = str(ef.get(k)) if ef.get(k) is not None else None
        if tv is None:
            continue
        total += 1
        if ev == tv:
            match += 1
        else:
            misses.append((k, ev, tv))
    return match, total, misses


# ── dispatch ──────────────────────────────────────────────────────────────────

_vqa_model = _vqa_processor = None
_vqa27b_model = _vqa27b_processor = None


def _load_vqa(device: str):
    global _vqa_model, _vqa_processor
    if _vqa_model is None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError(
                "torch not available — VQA extraction requires a CUDA GPU"
            )
        from vqa import load_model
        _vqa_model, _vqa_processor = load_model(device)
    return _vqa_model, _vqa_processor


def _load_vqa27b(device: str):
    global _vqa27b_model, _vqa27b_processor
    if _vqa27b_model is None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError(
                "torch not available — VQA27B extraction requires a CUDA GPU"
            )
        from vqa import load_model_27b
        _vqa27b_model, _vqa27b_processor = load_model_27b(device)
    return _vqa27b_model, _vqa27b_processor


def run_sample(sample_dir: Path, device: str, meta: dict, backend: str = "vqa") -> dict:
    pdf_path = sample_dir / "report.pdf"
    manufacturer = meta.get("manufacturer", "")
    device_type = meta.get("device_type", "")
    extraction = meta.get("extraction_method", "")

    if manufacturer == "Biotronik":
        from extractors.biotronik import extract
        return extract(pdf_path)

    if manufacturer == "ELA Medical / Sorin":
        from extractors.ela_sorin import extract
        return extract(pdf_path)

    if manufacturer == "Boston Scientific":
        if "S-ICD" in device_type or "sicd" in extraction or "hl7" in extraction:
            hl7_path = sample_dir / "report.hl7"
            from extractors.bsc_sicd import extract
            return extract(pdf_path, hl7_path if hl7_path.exists() else None)
        else:
            from extractors.bsc_crtd import extract
            return extract(pdf_path)

    if manufacturer == "Medtronic":
        # .26X and .V26 are text-format Device Data Archives from the 2090 programmer
        for suffix in (".26X", ".V26", ".26x", ".v26"):
            p = sample_dir / f"report{suffix}"
            if p.exists():
                from extractors.medtronic_2090 import extract
                return extract(p)
        # CareLink Quick Look II PDF
        if pdf_path.exists():
            from extractors.medtronic_carelink import extract
            return extract(pdf_path)
        raise FileNotFoundError(f"No .26X/.V26 or PDF report found in {sample_dir}")

    if manufacturer == "Abbott":
        log_path = sample_dir / "device_export.log"
        if log_path.exists():
            from extractors.abbott_log import extract as log_extract
            return log_extract(log_path)
        if backend == "ocr":
            from extractors.abbott_merlin_ocr import extract as ocr_extract
            return ocr_extract(pdf_path)
        from extractors.abbott_merlin import extract, detect_report_type
        rt = detect_report_type(pdf_path)
        if backend == "vqa27b":
            model, processor = _load_vqa27b(device)
        else:
            model, processor = _load_vqa(device)
        return extract(pdf_path, model, processor, report_type=rt)

    raise ValueError(f"No extractor registered for manufacturer '{manufacturer}'")


# ── main ──────────────────────────────────────────────────────────────────────

def find_samples(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("ground_truth.json"))


def _auto_select_gpu() -> None:
    """Pick the first GPU with >6 GB free via nvidia-smi and set CUDA_VISIBLE_DEVICES.

    Must be called before any CUDA initialization (i.e. before torch is imported).
    Skipped when CUDA_VISIBLE_DEVICES is already set externally.
    Uses subprocess.run(check=False) so a bad GPU (nvidia-smi non-zero exit) doesn't
    abort the whole selection — we still parse stdout for healthy GPUs.
    """
    import os, subprocess
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
            text=True, timeout=5, capture_output=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            idx = parts[0].strip()
            if not idx.isdigit():
                continue
            try:
                free_mb = int(parts[1].strip().replace(" MiB", ""))
            except ValueError:
                continue
            if free_mb > 6000:
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
                return
    except Exception:
        pass  # nvidia-smi not available; let torch pick


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="Sample dirs (default: all under samples/)")
    p.add_argument("--vendor", help="Filter by manufacturer substring (case-insensitive)")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument(
        "--backend", choices=["vqa", "ocr", "vqa27b", "auto"], default="vqa",
        help="Abbott PDF backend: vqa=Qwen3-VL-8B (GPU), ocr=PaddleOCR+regex (CPU), "
             "vqa27b=Qwen3-VL-27B (multi-GPU, HF transformers), auto=vqa if CUDA else ocr",
    )
    args = p.parse_args()

    if args.device:
        device = args.device
    else:
        _auto_select_gpu()   # must run before torch.cuda.is_available()
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    backend = args.backend
    if backend == "auto":
        try:
            import torch
            backend = "vqa" if torch.cuda.is_available() else "ocr"
        except ImportError:
            backend = "ocr"
        print(f"  [auto] selected backend: {backend}")

    if args.paths:
        samples = [Path(s) for s in args.paths]
    else:
        samples = find_samples(Path("samples"))

    if args.vendor:
        def _keep(d: Path) -> bool:
            m = (d / "meta.json")
            if not m.exists():
                return False
            return args.vendor.lower() in json.loads(m.read_text()).get("manufacturer", "").lower()
        samples = [s for s in samples if _keep(s)]

    if not samples:
        print("No samples found.")
        sys.exit(1)

    total_match = total_fields = 0

    for sample_dir in samples:
        gt_path = sample_dir / "ground_truth.json"
        pdf_path = sample_dir / "report.pdf"
        meta_path = sample_dir / "meta.json"
        log_path_check = sample_dir / "device_export.log"
        if not gt_path.exists() or (not pdf_path.exists() and not log_path_check.exists()):
            continue

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        truth = json.loads(gt_path.read_text())
        label = (
            f"{meta.get('manufacturer','?')} / "
            f"{meta.get('device_family','?')} / "
            f"{meta.get('report_type','?')} [{sample_dir.name}]"
        )
        print(f"\n{'═'*72}")
        print(f"  {label}")
        print(f"{'═'*72}")

        try:
            result = run_sample(sample_dir, device, meta, backend=backend)
        except NotImplementedError as e:
            print(f"  SKIP (not implemented): {e}")
            continue
        except ModuleNotFoundError as e:
            print(f"  SKIP (missing dependency): {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        match, total, misses = score(result, truth)
        total_match += match
        total_fields += total
        pct = 100 * match // total if total else 0
        print(f"  {match}/{total} fields correct ({pct}%)")
        for field, got, expected in misses:
            print(f"    ✗ {field:<40} got={got!r:25}  expected={expected!r}")

    print(f"\n{'─'*72}")
    overall = 100 * total_match // total_fields if total_fields else 0
    print(f"OVERALL: {total_match}/{total_fields} ({overall}%) across {len(samples)} sample(s)")


if __name__ == "__main__":
    main()
