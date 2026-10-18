#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-way VQA benchmark on Abbott Merlin reports:
  1. PaddleOCR-VL-1.6  — OCR mode  (raw text → regex parse, no GPU prompt)
  2. PaddleOCR-VL-1.6  — VQA mode  (same JSON prompt as Qwen, test if it works)
  3. Qwen2-VL-7B-Instruct (4-bit)  — current production model
  4. Qwen3-VL-8B-Instruct (4-bit)  — candidate upgrade

Usage (run on bigboy):
    .venv/bin/python3 benchmark_vqa.py [--models all] [--report all]

Output: benchmark_results.json + printed table
"""
import argparse
import json
import re
import time
from pathlib import Path

import pdf2image

from extractors.abbott_merlin import (
    PROMPT_FREEZE, PROMPT_SUMMARY, PROMPT_FASTPATH_ICD, PROMPT_FASTPATH_PM,
    _parse_json, _postprocess, detect_report_type,
)
from validate import score

# ── sample registry ───────────────────────────────────────────────────────────

SAMPLES = {
    "freeze":       "samples/abbott/aveir_dr_lp/FreezeReport/001",
    "summary":      "samples/abbott/aveir_dr_lp/SummaryReport/001",
    "fastpath_icd": "samples/abbott/entrant_dr/FastPathSummary/001",
    "fastpath_pm":  "samples/abbott/endurity_mri_2172/FastPathSummary/001",
}

PROMPTS = {
    "freeze":       PROMPT_FREEZE,
    "summary":      PROMPT_SUMMARY,
    "fastpath_icd": PROMPT_FASTPATH_ICD,
    "fastpath_pm":  PROMPT_FASTPATH_PM,
}


# ── backend: PaddleOCR-VL-1.6 ─────────────────────────────────────────────────

class PaddleOCRBackend:
    MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

    def __init__(self, mode: str = "ocr"):
        assert mode in ("ocr", "vqa")
        self.mode = mode
        self.name = f"PaddleOCR-VL-1.6 [{mode}]"
        self.model = None
        self.processor = None

    def load(self, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch
        print(f"Loading {self.MODEL_ID} …")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print("  ready.")

    def _run_model(self, image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        import torch
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def extract(self, image, report_type: str) -> dict:
        if self.mode == "vqa":
            # Use the same structured JSON prompt as Qwen — tests if PaddleOCR-VL
            # can do field extraction, not just OCR.
            raw = self._run_model(image, PROMPTS[report_type])
            try:
                result = _parse_json(raw)
                if report_type != "summary":
                    result = _postprocess(result)
                return result
            except Exception:
                return {"_raw": raw, "_parse_error": True}

        else:
            # OCR mode: extract all text, then regex-parse key fields.
            ocr_prompt = (
                "Please perform OCR on this document image. "
                "Extract ALL visible text exactly as printed, preserving layout."
            )
            ocr_text = self._run_model(image, ocr_prompt)
            return _paddle_ocr_regex(ocr_text, report_type)


def _paddle_ocr_regex(text: str, report_type: str) -> dict:
    """Best-effort regex extraction from PaddleOCR raw text output."""

    def find(pattern, default=None, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else default

    if report_type == "freeze":
        return {
            "report_type": "FreezeReport",
            "followup_date": find(r"(\d{1,2}\s+\w+\s+\d{4})\s*$", flags=re.MULTILINE),
            "device":        find(r"(Aveir[™\w\s/()]+(?:RA|RV\)))", flags=re.IGNORECASE),
            "facility":      find(r"(HZ\s+\S+)"),
            "mode":          find(r"Betriebsart\s+(\w+)"),
            "base_rate_bpm": find(r"Grundfrq\.\s+(\d+)"),
            "av_paced_ms":   find(r"AV\s+stimuliert\s+(\d+)"),
            "av_sensed_ms":  find(r"AV\s+wahrgen\.\s+(\d+)"),
            "upper_rate_bpm":     find(r"Obere\s+Grenzfrq\.\s+(\d+)"),
            "max_sensor_rate_bpm": find(r"Max\.\s+Sensorfrq\.\s+(\d+)"),
            "rate_hysteresis":    find(r"Frequenzhysteresedif\.\s+(\S+)"),
            "vip":                find(r"VIP[™\s]+(\w+)"),
            "rate_adaptive_av":   find(r"Frequenzabh\.\s+AV-Intervall\s+(\w+)"),
            "rate_adaptive_pvarp": find(r"Frequenzabh\.\s+PVARP/VREF\s+(\w+)"),
            "vent_safety_pacing": find(r"Vent\.\s+Sicherheits-Stim\.\s+(\w+)"),
        }

    if report_type == "fastpath_icd":
        return {
            "report_type":    "FastPathSummary",
            "followup_date":  find(r"(\d{1,2}\s+\w+\s+\d{4})"),
            "device_model":   find(r"Entrant[™\s]+DR\s+(\w+)"),
            "device_serial":  find(r"(\d{9})"),
            "battery_eri":    find(r"ERI\s+(>?\s*\d+\s*Jhr\.)"),
            "battery_current_ua":   find(r"Batteriestrom\s+(\d+)"),
            "battery_remaining_pct": find(r"Kapazit[äa]t.*?(\>?\d+)"),
            "sensing_v":      find(r"Signalamplitude.*?V.*?(\S+)"),
            "impedance_v":    find(r"Elektrodenimpedanz.*?V.*?(\S+)"),
            "impedance_hv":   find(r"Elektrodenimpedanz.*?HV.*?(\S+)"),
            "mode":           find(r"Betriebsart\s+(\w+)"),
            "vt_zone":        find(r"VT.*?(\w+)"),
            "vf_zone":        find(r"VF.*?(\w+)"),
            "detection_params": find(r"Detektionsparameter\s+(\w+)"),
            "therapy":        find(r"Therapie.*?(\w+)"),
            "sense_ability_v": find(r"SenseAbility[™\s]+(\w+)"),
            "sensing_config_v": find(r"Empfindlichkeit\s+(\w+)"),
            "vt_vf_episodes": find(r"VT/VF-Episoden.*?(\d+)"),
        }

    if report_type == "fastpath_pm":
        return {
            "report_type":    "FastPathSummary",
            "followup_date":  find(r"(\d{1,2}\s+\w+\s+\d{4})"),
            "device_model":   find(r"Endurity\s+MRI[™\s]+(\d+)"),
            "device_serial":  find(r"(\d{7})"),
            "battery_voltage_v":    find(r"Spannung\s+([\d,]+)"),
            "battery_mag_freq_bpm": find(r"Magnetfrequenz\s+([\d,]+)"),
            "battery_remaining_pct": find(r"Kapazit[äa]t.*?(\>?\d+)"),
            "mode":           find(r"Betriebsart\s+(\w+)"),
            "base_rate_bpm":  find(r"Grundfrequenz.*?(\d+)"),
            "upper_rate_bpm": find(r"Grenzfr\w*\s+(\d+)"),
            "av_paced_ms":    find(r"Stimuliertes.*?(\d+)"),
            "av_sensed_ms":   find(r"Wahrg\.\s+AV.*?(\d+)"),
            "stim_a_pct":     find(r"AP.*?(>?\d+)"),
            "stim_v_pct":     find(r"VP.*?(>?\d+)"),
            "ams_episodes":   find(r"AMS-Episoden\s+(\d+)"),
            "mode_switch_pct": find(r"Mode-Switch\s+([\d<>]+)"),
            "af_burden_pct":  find(r"AT/AF-Burden\s+([\d<>]+)"),
        }

    # summary — most complex, nested
    return {"_ocr_raw": text, "_note": "regex not implemented for summary in OCR mode"}


# ── backend: Qwen2-VL-7B ──────────────────────────────────────────────────────

class Qwen2VLBackend:
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    name = "Qwen2-VL-7B-Instruct (4-bit)"

    def __init__(self):
        self.model = self.processor = None

    def load(self, device: str = "cuda"):
        from vqa import load_model
        print(f"Loading {self.MODEL_ID} (4-bit) …")
        self.model, self.processor = load_model(device)
        print("  ready.")

    def extract(self, image, report_type: str) -> dict:
        from qwen_vl_utils import process_vision_info
        import torch
        prompt = PROMPTS[report_type]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        result = _parse_json(raw)
        if report_type != "summary":
            result = _postprocess(result)
        return result


# ── backend: Qwen3-VL-8B ──────────────────────────────────────────────────────

class Qwen3VLBackend:
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    name = "Qwen3-VL-8B-Instruct (4-bit)"

    def __init__(self):
        self.model = self.processor = None

    def load(self, device: str = "cuda"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        import torch
        print(f"Loading {self.MODEL_ID} (4-bit) …")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            quantization_config=bnb,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        print("  ready.")

    def extract(self, image, report_type: str) -> dict:
        import torch
        prompt = PROMPTS[report_type]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        result = _parse_json(raw)
        if report_type != "summary":
            result = _postprocess(result)
        return result


# ── main ──────────────────────────────────────────────────────────────────────

ALL_MODELS = {
    "paddle-ocr": lambda: PaddleOCRBackend("ocr"),
    "paddle-vqa": lambda: PaddleOCRBackend("vqa"),
    "qwen2":      lambda: Qwen2VLBackend(),
    "qwen3":      lambda: Qwen3VLBackend(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",  default="paddle-ocr,paddle-vqa,qwen2,qwen3",
                    help="Comma-separated model keys (default: all)")
    ap.add_argument("--report",  default="all",
                    help="Report type key or 'all' (freeze|summary|fastpath_icd|fastpath_pm)")
    ap.add_argument("--dpi",     type=int, default=300)
    ap.add_argument("--device",  default="cuda")
    args = ap.parse_args()

    model_keys = args.models.split(",")
    report_keys = list(SAMPLES.keys()) if args.report == "all" else [args.report]

    # Load all images upfront
    images = {}
    report_types = {}
    ground_truths = {}
    for rk in report_keys:
        sample_dir = Path(SAMPLES[rk])
        images[rk] = pdf2image.convert_from_path(
            str(sample_dir / "report.pdf"), dpi=args.dpi
        )[0]
        report_types[rk] = rk
        gt_path = sample_dir / "ground_truth.json"
        if gt_path.exists():
            ground_truths[rk] = json.loads(gt_path.read_text())

    results = {}   # {model_key: {report_key: {match, total, pct, latency_s, misses}}}

    for mk in model_keys:
        print(f"\n{'═'*64}")
        print(f"  MODEL: {mk}")
        print(f"{'═'*64}")
        backend = ALL_MODELS[mk]()
        t0 = time.time()
        backend.load(args.device)
        load_time = time.time() - t0
        print(f"  Load time: {load_time:.1f}s")

        results[mk] = {}
        for rk in report_keys:
            print(f"\n  [{rk}]")
            t0 = time.time()
            try:
                extracted = backend.extract(images[rk], rk)
            except Exception as e:
                print(f"    ERROR: {e}")
                results[mk][rk] = {"error": str(e)}
                continue
            elapsed = time.time() - t0

            if rk in ground_truths:
                match, total, misses = score(extracted, ground_truths[rk])
                pct = 100 * match // total if total else 0
                results[mk][rk] = {
                    "match": match, "total": total, "pct": pct,
                    "latency_s": round(elapsed, 1),
                    "misses": [(f, g, e) for f, g, e in misses],
                }
                print(f"    {match}/{total} ({pct}%)  [{elapsed:.1f}s]")
                for field, got, expected in misses:
                    print(f"      ✗ {field:<40} got={got!r}  expected={expected!r}")
            else:
                results[mk][rk] = {"latency_s": round(elapsed, 1), "no_gt": True}
                print(f"    no ground truth  [{elapsed:.1f}s]")

        # Free GPU memory before next model
        del backend
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*80}")
    print("SUMMARY TABLE")
    print(f"{'═'*80}")

    col_w = 12
    header = f"{'Model':<28}" + "".join(f"{rk[:col_w]:>{col_w}}" for rk in report_keys) + f"{'TOTAL':>{col_w}}"
    print(header)
    print("─" * len(header))

    for mk in model_keys:
        row = f"{mk:<28}"
        total_match = total_fields = 0
        for rk in report_keys:
            r = results.get(mk, {}).get(rk, {})
            if "error" in r:
                cell = "ERR"
            elif "no_gt" in r:
                cell = "─"
            else:
                cell = f"{r['pct']}%"
                total_match += r.get("match", 0)
                total_fields += r.get("total", 0)
        row += "".join(
            f"{('ERR' if 'error' in results.get(mk, {}).get(rk, {}) else '─' if 'no_gt' in results.get(mk, {}).get(rk, {}) else str(results[mk][rk]['pct'])+'%'):>{col_w}}"
            for rk in report_keys
        )
        overall = f"{100*total_match//total_fields}%" if total_fields else "─"
        row += f"{overall:>{col_w}}"
        print(row)

    print(f"\nLatency (seconds per report):")
    for mk in model_keys:
        lats = [
            f"{rk}={results.get(mk, {}).get(rk, {}).get('latency_s', '?')}s"
            for rk in report_keys
        ]
        print(f"  {mk:<28} " + "  ".join(lats))

    out = Path("benchmark_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
