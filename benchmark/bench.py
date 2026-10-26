#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-backend benchmark for Abbott image-based PDF extraction.

Contenders:
  1. qwen-vl        — Qwen3-VL-8B-Instruct (4-bit), native VLM (current production)
  2. paddleocr      — PaddleOCR text extraction only (no LLM reasoning)
  3. oxr            — OXR-1.0 document parser (markdown/JSON output)
  4. paddleocr+llm  — PaddleOCR text → light LLM for structured extraction
  5. oxr+llm        — OXR markdown → light LLM for structured extraction

Metrics per contender per sample:
  - accuracy: field match count / total ground truth fields
  - latency_s: wall-clock seconds
  - per-field misses

Usage:
    python bench.py                              # all backends, all samples
    python bench.py --backends qwen-vl oxr+llm   # specific backends
    python bench.py --samples 001 002            # specific sample dirs
    python bench.py --llm-url http://bigboy:11434 # Ollama endpoint for +llm backends
    python bench.py --oxr-url http://localhost:8000 # OXR API endpoint
    python bench.py --report results.json        # save results

Run on bigboy:
    python benchmark/bench.py --llm-url http://localhost:11434


Phased execution (when OXR and Ollama can't share a GPU):
    # Phase 1 — Ollama up: run qwen-vl and paddleocr+llm
    python bench.py --backends qwen-vl paddleocr paddleocr+llm

    # Phase 2 — stop Ollama, start OXR: collect OXR markdown (cached to disk)
    python bench.py --backends oxr --cache-dir ./cache

    # Phase 3 — stop OXR, restart Ollama: run LLM on cached OXR output
    python bench.py --backends oxr+llm --cache-dir ./cache
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path so we can import extractors
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── scoring (same as validate.py) ────────────────────────────────────────────

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


# ── sample discovery ─────────────────────────────────────────────────────────

def find_abbott_pdf_samples(samples_dir: Path) -> list[dict]:
    """Find all Abbott samples that have both a PDF and ground truth."""
    results = []
    for gt in sorted(samples_dir.rglob("ground_truth.json")):
        sample_dir = gt.parent
        pdf = sample_dir / "report.pdf"
        meta_path = sample_dir / "meta.json"
        if not pdf.exists():
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("manufacturer", "Abbott") != "Abbott":
            continue
        results.append({
            "dir": sample_dir,
            "pdf": pdf,
            "ground_truth": json.loads(gt.read_text()),
            "meta": meta,
            "label": (
                f"{meta.get('device_family', '?')}/"
                f"{meta.get('report_type', '?')}/"
                f"{sample_dir.name}"
            ),
        })
    return results


# ── backend base class ───────────────────────────────────────────────────────

class Backend:
    name: str = "base"
    cache_dir: Path | None = None

    def setup(self, **kwargs):
        """One-time setup (load models, connect to APIs)."""
        if kwargs.get("cache_dir"):
            self.cache_dir = Path(kwargs["cache_dir"])
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        """Extract structured data from a PDF. Returns a flat dict of fields."""
        raise NotImplementedError

    def _cache_key(self, pdf_path: Path) -> str:
        return pdf_path.stem + "__" + pdf_path.parent.name

    def _save_cache(self, pdf_path: Path, prefix: str, text: str):
        if self.cache_dir:
            p = self.cache_dir / f"{prefix}__{self._cache_key(pdf_path)}.txt"
            p.write_text(text, encoding="utf-8")

    def _load_cache(self, pdf_path: Path, prefix: str) -> str | None:
        if self.cache_dir:
            p = self.cache_dir / f"{prefix}__{self._cache_key(pdf_path)}.txt"
            if p.exists():
                return p.read_text(encoding="utf-8")
        return None


# ── backend: Qwen3-VL (native VLM, current production) ──────────────────────

class QwenVLBackend(Backend):
    name = "qwen-vl"

    def setup(self, **kwargs):
        from vqa import load_model
        device = kwargs.get("device", "cuda")
        self.model, self.processor = load_model(device)

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        from extractors.abbott_merlin import extract, detect_report_type
        rt = detect_report_type(pdf_path)
        return extract(pdf_path, self.model, self.processor, report_type=rt)


# ── backend: OXR (document parser, REST API) ────────────────────────────────

class OXRBackend(Backend):
    name = "oxr"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.oxr_url = kwargs.get("oxr_url", "http://localhost:8000")

    def _call_oxr(self, pdf_path: Path) -> str:
        import urllib.request
        url = f"{self.oxr_url}/v1/parse/sync"
        boundary = "----BenchBoundary"
        pdf_bytes = pdf_path.read_bytes()
        filename = pdf_path.name

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode() + pdf_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="api_key"\r\n\r\n'
            f"dummy\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="output_formats"\r\n\r\n'
            f"markdown\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
        return result.get("markdown", "") or ""

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        cached = self._load_cache(pdf_path, "oxr")
        if cached is not None:
            markdown = cached
        else:
            markdown = self._call_oxr(pdf_path)
            self._save_cache(pdf_path, "oxr", markdown)
        return {"_raw_markdown": markdown, "report_type": "unknown"}


# ── backend: OXR + LLM (document parser → light LLM for extraction) ─────────

EXTRACTION_PROMPT = """\
You are a medical data extraction assistant. Below is the OCR text extracted \
from an Abbott pacemaker/ICD interrogation report. Extract all clinical \
parameters into a flat JSON object.

RULES:
- Copy values EXACTLY as printed (German decimal comma, German units).
- Use null for fields not present in the text.
- Return ONLY the JSON object — no markdown fences, no explanation.

FIELDS TO EXTRACT (use these exact keys):
- report_type, patient_name, followup_date, device_model, device_serial
- battery_voltage_v, battery_longevity, battery_current_ua, battery_remaining_pct
- threshold_a_v, threshold_a_pw_ms, sensing_a_mv, impedance_a_ohm
- threshold_rv_v, threshold_rv_pw_ms, sensing_rv_mv, impedance_rv_ohm
- threshold_lv_v, threshold_lv_pw_ms, sensing_lv_mv, impedance_lv_ohm
- impedance_hv_ohm
- mode, base_rate_bpm, upper_rate_bpm
- stim_a_amplitude_v, stim_a_pw_ms, sense_ability_a, sensitivity_a_mv
- stim_v_amplitude_v, stim_v_pw_ms, sense_ability_v, sensitivity_v_mv
- ap_paced_pct, vp_paced_pct, mode_switch_pct, ams_episodes
- lead_a_model, lead_a_serial, lead_rv_model, lead_rv_serial
- lead_lv_model, lead_lv_serial, patient_dob

OCR TEXT:
{ocr_text}
"""


def _query_ollama(url: str, model: str, prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 2048},
    })
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()).get("response", "")


def _parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM response."""
    import re
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


class OXRLLMBackend(Backend):
    name = "oxr+llm"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.llm_url = kwargs.get("llm_url", "http://localhost:11434")
        self.llm_model = kwargs.get("llm_model", "qwen3:8b")
        self._oxr = OXRBackend()
        self._oxr.setup(**kwargs)

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        cached = self._load_cache(pdf_path, "oxr")
        if cached is not None:
            markdown = cached
        else:
            oxr_result = self._oxr.extract(pdf_path, meta)
            markdown = oxr_result.get("_raw_markdown", "")
        if not markdown:
            return {}
        prompt = EXTRACTION_PROMPT.format(ocr_text=markdown[:8000])
        raw = _query_ollama(self.llm_url, self.llm_model, prompt)
        return _parse_llm_json(raw)


# ── backend: PaddleOCR (text extraction only) ───────────────────────────────

class PaddleOCRBackend(Backend):
    name = "paddleocr"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self._dpi = kwargs.get("dpi", 200)

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        cached = self._load_cache(pdf_path, "paddle")
        if cached is not None:
            return {"_raw_text": cached, "report_type": "unknown"}

        import pdf2image
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return {"_error": "paddleocr not installed"}

        images = pdf2image.convert_from_path(str(pdf_path), dpi=self._dpi)
        ocr = PaddleOCR(use_angle_cls=True, lang="de", show_log=False)

        full_text = []
        for img in images:
            import numpy as np
            result = ocr.ocr(np.array(img), cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0]]
                full_text.extend(lines)

        text = "\n".join(full_text)
        self._save_cache(pdf_path, "paddle", text)
        return {"_raw_text": text, "report_type": "unknown"}


# ── backend: PaddleOCR + LLM ────────────────────────────────────────────────

class PaddleOCRLLMBackend(Backend):
    name = "paddleocr+llm"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.llm_url = kwargs.get("llm_url", "http://localhost:11434")
        self.llm_model = kwargs.get("llm_model", "qwen3:8b")
        self._paddle = PaddleOCRBackend()
        self._paddle.setup(**kwargs)

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        cached = self._load_cache(pdf_path, "paddle")
        if cached is not None:
            text = cached
        else:
            paddle_result = self._paddle.extract(pdf_path, meta)
            text = paddle_result.get("_raw_text", "")
        if not text:
            return {}
        prompt = EXTRACTION_PROMPT.format(ocr_text=text[:8000])
        raw = _query_ollama(self.llm_url, self.llm_model, prompt)
        return _parse_llm_json(raw)


# ── backend: deepseek-ocr via Ollama (vision model, direct extraction) ──────

class DeepseekOCRBackend(Backend):
    name = "deepseek-ocr"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.llm_url = kwargs.get("llm_url", "http://localhost:11434")
        self.llm_model = "deepseek-ocr:3b"
        self._dpi = kwargs.get("dpi", 200)

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        import base64
        import pdf2image
        import urllib.request

        images = pdf2image.convert_from_path(str(pdf_path), dpi=self._dpi,
                                              first_page=1, last_page=1)
        import io
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = json.dumps({
            "model": self.llm_model,
            "prompt": EXTRACTION_PROMPT.format(
                ocr_text="[See the attached image of an Abbott pacemaker report]"),
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 2048},
        })
        req = urllib.request.Request(
            f"{self.llm_url}/api/generate",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = json.loads(resp.read()).get("response", "")
        return _parse_llm_json(raw)


# ── registry ─────────────────────────────────────────────────────────────────

BACKENDS = {
    "qwen-vl":       QwenVLBackend,
    "oxr":           OXRBackend,
    "oxr+llm":       OXRLLMBackend,
    "paddleocr":     PaddleOCRBackend,
    "paddleocr+llm": PaddleOCRLLMBackend,
    "deepseek-ocr":  DeepseekOCRBackend,
}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Multi-backend Abbott extraction benchmark")
    p.add_argument("--backends", nargs="+", default=list(BACKENDS.keys()),
                   choices=list(BACKENDS.keys()), help="Backends to benchmark")
    p.add_argument("--samples", nargs="+", default=None,
                   help="Filter by sample dir name (e.g. 001 002)")
    p.add_argument("--samples-dir", default=str(PROJECT_ROOT / "samples" / "abbott"),
                   help="Root of Abbott samples")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--llm-url", default="http://localhost:11434",
                   help="Ollama API URL for +llm backends")
    p.add_argument("--llm-model", default="qwen3:8b",
                   help="Ollama model for +llm backends")
    p.add_argument("--oxr-url", default="http://localhost:8000",
                   help="OXR API endpoint")
    p.add_argument("--cache-dir", default=None,
                   help="Cache OCR text to disk (enables phased runs)")
    p.add_argument("--report", default=None, help="Save JSON results to file")
    p.add_argument("--verbose", action="store_true", help="Print per-field misses")
    args = p.parse_args()

    samples = find_abbott_pdf_samples(Path(args.samples_dir))
    if args.samples:
        samples = [s for s in samples if s["dir"].name in args.samples]

    if not samples:
        print("No samples found.")
        sys.exit(1)

    print(f"Found {len(samples)} Abbott PDF sample(s)")
    print(f"Backends: {', '.join(args.backends)}\n")

    all_results = {}

    for backend_name in args.backends:
        cls = BACKENDS[backend_name]
        backend = cls()
        print(f"{'═' * 72}")
        print(f"  BACKEND: {backend_name}")
        print(f"{'═' * 72}")

        try:
            backend.setup(
                device=args.device,
                llm_url=args.llm_url,
                llm_model=args.llm_model,
                oxr_url=args.oxr_url,
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            print(f"  SETUP FAILED: {e}\n")
            continue

        backend_results = []

        for sample in samples:
            label = sample["label"]
            print(f"\n  {label} … ", end="", flush=True)

            t0 = time.monotonic()
            try:
                extracted = backend.extract(sample["pdf"], sample["meta"])
            except Exception as e:
                print(f"ERROR: {e}")
                backend_results.append({
                    "label": label, "error": str(e),
                    "match": 0, "total": 0, "pct": 0, "latency_s": 0,
                })
                continue
            latency = time.monotonic() - t0

            # Strip internal fields before scoring
            extracted = {k: v for k, v in extracted.items()
                         if not k.startswith("_")}

            match, total, misses = score(extracted, sample["ground_truth"])
            pct = round(100 * match / total, 1) if total else 0

            print(f"{match}/{total} ({pct}%) in {latency:.1f}s")

            if args.verbose and misses:
                for field, got, expected in misses:
                    print(f"    ✗ {field:<40} got={got!r:25} expected={expected!r}")

            backend_results.append({
                "label": label,
                "match": match,
                "total": total,
                "pct": pct,
                "latency_s": round(latency, 2),
                "misses": [(f, g, e) for f, g, e in misses],
            })

        # Summary for this backend
        total_match = sum(r["match"] for r in backend_results)
        total_fields = sum(r["total"] for r in backend_results)
        total_time = sum(r["latency_s"] for r in backend_results)
        overall_pct = round(100 * total_match / total_fields, 1) if total_fields else 0
        avg_time = round(total_time / len(backend_results), 1) if backend_results else 0

        print(f"\n  ── {backend_name} summary ──")
        print(f"  Accuracy: {total_match}/{total_fields} ({overall_pct}%)")
        print(f"  Total time: {total_time:.1f}s | Avg per sample: {avg_time:.1f}s")

        all_results[backend_name] = {
            "accuracy_pct": overall_pct,
            "total_match": total_match,
            "total_fields": total_fields,
            "total_time_s": round(total_time, 2),
            "avg_time_s": avg_time,
            "samples": backend_results,
        }

    # ── comparison table ─────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  COMPARISON")
    print(f"{'═' * 72}")
    print(f"  {'Backend':<20} {'Accuracy':>10} {'Avg time':>10} {'Fields':>10}")
    print(f"  {'─' * 52}")
    for name, res in all_results.items():
        print(f"  {name:<20} {res['accuracy_pct']:>9.1f}% {res['avg_time_s']:>9.1f}s "
              f"{res['total_match']}/{res['total_fields']:>4}")

    if args.report:
        Path(args.report).write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print(f"\nResults saved to {args.report}")


if __name__ == "__main__":
    main()
