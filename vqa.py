#!/usr/bin/env python3
"""
Direct VQA extraction using Qwen3-VL-8B-Instruct (4-bit quantized).
Sends the report image + a structured prompt in a single pass — no OCR, no regex.

Usage:
    python vqa.py report.pdf [--output result.json] [--dpi 300]
    python vqa.py report.pdf --compare-json results/known_good.json
    python vqa.py report.pdf --verbose   # show raw model response
"""
import argparse
import json
import os
import re
from pathlib import Path

import pdf2image
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

MODEL_ID     = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_ID_27B = os.environ.get("VQA_MODEL_27B", "Qwen/Qwen3.8-27B")

# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA = {
    "report_type": "SummaryReport | FreezeReport | unknown",
    "patient": None,
    "device": None,
    "facility": None,
    "interrogation_date": None,
    "visit_type": None,
    "battery_ra": {
        "implant_date": None,
        "voltage": None,
        "current_ua": None,
        "remaining_capacity_pct": None,
        "estimated_longevity": None,
        "rrt": None,
    },
    "battery_rv": {
        "implant_date": None,
        "voltage": None,
        "current_ua": None,
        "remaining_capacity_pct": None,
        "estimated_longevity": None,
        "rrt": None,
    },
    "test_ra": {
        "pacing_threshold": None,
        "pacing_threshold_date": None,
        "sensing_mv": None,
        "sensing_date": None,
        "impedance_ohm": None,
        "impedance_date": None,
    },
    "test_rv": {
        "pacing_threshold": None,
        "pacing_threshold_date": None,
        "sensing_mv": None,
        "sensing_date": None,
        "impedance_ohm": None,
        "impedance_date": None,
    },
    "parameters": {
        "mode": None,
        "base_rate": None,
        "upper_rate_limit": None,
        "max_sensor_rate": None,
        "av_paced_ms": None,
        "av_sensed_ms": None,
        "rate_hysteresis": None,
        "vip_enabled": None,
        "rate_adaptive_av": None,
        "rate_adaptive_pvarp": None,
        "ventricular_safety_pacing": None,
    },
    "stimulation_ra": {
        "configuration": None,
        "amplitude_v": None,
        "pulse_width_ms": None,
        "senseability": None,
        "sensitivity_mv": None,
    },
    "stimulation_rv": {
        "configuration": None,
        "amplitude_v": None,
        "pulse_width_ms": None,
        "senseability": None,
        "sensitivity_mv": None,
    },
    "diagnostics": {
        "since": None,
        "ap_paced_pct": None,
        "vp_paced_pct": None,
        "mode_switch_pct": None,
        "ams_episodes": None,
    },
}

PROMPT = """\
You are a medical data extraction assistant. Extract pacemaker interrogation \
parameters from this Abbott Aveir pacemaker report image and fill the JSON \
schema below.

IMPORTANT RULES:
- Copy values EXACTLY as printed: use German decimal comma (e.g. "12,0 uA", \
"3,0 V", "11,3 Jahre"), keep German units (Jahre, min⁻¹, ms, mV, V, Ω, %).
- Set a field to null only if it is genuinely not present in this report.
- Return ONLY the JSON object — no explanation, no markdown fences.

WHERE TO FIND EACH GROUP:
1. HEADER (top of page): the first few lines contain Abbott / patient name / \
device / facility / date / visit type. Patient name is the line immediately \
after "Abbott" (may be a name or "DEMO"). Device is the line starting with \
"Aveir" — copy it exactly including "™" and model numbers. Facility is the \
line starting with "HZ". Date is in "dd Mon yyyy" format. Visit type is \
"Bei der Nachsorge", "Erstprogrammierung", or similar.
   report_type = "SummaryReport" if the page title contains "FastPath", \
"FreezeReport" if it contains "Schlüsselparameter".

2. BATTERY ("Batterieinformationen" section): two blocks, first = RA, second = RV.
   Each block has: "Laufzeit: X Jahre" (estimated_longevity), then an "RRT" \
label followed by the RRT value on the next line (e.g. ">5 Jahre") — put that \
value in rrt, NOT the word "RRT". Then a column table: Datum der Implantation / \
Spannung / Batteriestrom / Verbleibende Kapazität bis RRT (preserve space \
before % in percentages, e.g. "99 %").

3. TEST RESULTS ("Testergeb." section): two lead blocks labelled RA and RV.
   Under each lead find values printed as "X,XX V @ X,X ms (dd Mon yyyy)".
   IMPORTANT — split value and date into SEPARATE fields:
   - "Reizschwelle" → pacing_threshold = "X,XX V @ X,X ms" (NO date),
     pacing_threshold_date = "dd Mon yyyy"
   - "Wahrnehmung"  → sensing_mv = "X,X mV" (NO date),
     sensing_date = "dd Mon yyyy"
   - "Impedanz"     → impedance_ohm = "XXX Ω" (NO date),
     impedance_date = "dd Mon yyyy"

4. PARAMETERS ("Parameter" section): column layout — key column then value column.

5. STIMULATION ("Stimulation & Wahr." section): RA column then RV column.
   "SenseAbility™" row → senseability field. Read the value carefully: it is \
either "Ein" (on) or "Aus" (off). Do not guess — read what is printed.

6. DIAGNOSTICS (last section, "Seit:" line): percentages AP/VP/Mode-Switch \
in order, integer = AMS episodes.

""" + json.dumps(SCHEMA, ensure_ascii=False, indent=2)


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading {MODEL_ID} (4-bit) on {device} …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},  # index 0 of CUDA_VISIBLE_DEVICES (auto-selected by validate.py)
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model ready.")
    return model, processor


def load_model_27b(device: str):
    print(f"Loading {MODEL_ID_27B} (4-bit) on {device} …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID_27B,
        quantization_config=bnb,
        device_map="auto",  # spread across all available GPUs
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID_27B)
    print("27B model ready.")
    return model, processor


def vqa(model, processor, image: Image.Image, prompt: str = None) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt or PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    think_token_id = processor.tokenizer.convert_tokens_to_ids("<think>")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1280,
            suppress_tokens=[think_token_id] if isinstance(think_token_id, int) else None,
        )

    return processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]


# ── JSON parsing & post-processing ───────────────────────────────────────────

_DATE_SUFFIX = re.compile(r'\s*\(\d{1,2}\s+\w+\s+\d{4}\)\s*$')
_TEST_VALUE_FIELDS = ("pacing_threshold", "sensing_mv", "impedance_ohm")


def parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "could not parse JSON", "raw": raw[:500]}


def postprocess(result: dict) -> dict:
    """Strip trailing date parentheticals from test measurement values.

    The model reliably fills the separate *_date fields but sometimes also
    appends "(dd Mon yyyy)" to the value itself. Remove the duplicate.
    """
    for lead in ("test_ra", "test_rv"):
        block = result.get(lead)
        if not isinstance(block, dict):
            continue
        for field in _TEST_VALUE_FIELDS:
            val = block.get(field)
            if isinstance(val, str):
                block[field] = _DATE_SUFFIX.sub("", val).strip()
    return result


# ── comparison ────────────────────────────────────────────────────────────────

def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def compare(vqa_result: dict, ref: dict) -> None:
    vf = flatten(vqa_result)
    rf = flatten(ref)
    all_keys = sorted(set(vf) | set(rf))

    same = differ = one_sided = 0
    print("\n── Field comparison (Qwen3-VL VQA  vs  reference) ──")
    print(f"{'Field':<45} {'Qwen3-VL':<30} {'Reference':<30}")
    print("─" * 107)
    for k in all_keys:
        if k in ("raw_ocr_pages", "source_file"):
            continue
        v = str(vf.get(k)) if vf.get(k) is not None else "null"
        r = str(rf.get(k)) if rf.get(k) is not None else "null"
        if v == r:
            mark = "✓"; same += 1
        elif v != "null" and r != "null":
            mark = "△"; differ += 1
        else:
            mark = " "; one_sided += 1
        print(f"{mark} {k:<44} {v:<30} {r:<30}")
    print(f"\n✓ {same} match  |  △ {differ} differ  |  {one_sided} one-sided")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="Input PDF")
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output", metavar="FILE", help="Save result JSON to this file")
    p.add_argument("--verbose", action="store_true", help="Print raw model response")
    p.add_argument("--compare-json", metavar="FILE",
                   help="Compare against an existing extractor JSON file")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device)

    images = pdf2image.convert_from_path(args.pdf, dpi=args.dpi)
    print(f"PDF: {len(images)} page(s) at {args.dpi} DPI — using page 1")
    image = images[0]

    print("Running VQA …")
    raw = vqa(model, processor, image)

    if args.verbose:
        print("\n── Raw model response ──\n" + raw)

    result = postprocess(parse_json(raw))
    print("\n── Extracted JSON ──")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved: {args.output}")

    if args.compare_json:
        ref = json.loads(Path(args.compare_json).read_text())
        ref.pop("raw_ocr_pages", None)
        compare(result, ref)


if __name__ == "__main__":
    main()
