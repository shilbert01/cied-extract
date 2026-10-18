"""
Abbott Merlin PCS report extractor — VQA-based (Qwen2-VL-7B-Instruct).

Handles three report types emitted by Merlin PCS for Abbott devices:
  - FreezeReport        : single-page ECG + Schlüsselparameter table
  - FastPathSummary ICD : FastPath™ Übersicht for ICD devices (e.g. Entrant DR)
  - FastPathSummary PM  : FastPath™ Übersicht for pacemakers (e.g. Endurity MRI)

The report type is detected from the rendered page image title.
Model must be loaded externally (loaded once, reused across calls).

Usage:
    from vqa import load_model
    from extractors.abbott_merlin import extract, detect_report_type

    model, processor = load_model("cuda")
    result = extract("report.pdf", model, processor)
"""
import json
import re
from pathlib import Path
from typing import Optional

import pdf2image

# ── schemas ───────────────────────────────────────────────────────────────────

SCHEMA_FREEZE = {
    "report_type": "FreezeReport",
    "followup_date": None,
    "device": None,
    "facility": None,
    "mode": None,
    "base_rate_bpm": None,
    "av_paced_ms": None,
    "av_sensed_ms": None,
    "upper_rate_bpm": None,
    "max_sensor_rate_bpm": None,
    "rate_hysteresis": None,
    "vip": None,
    "rate_adaptive_av": None,
    "rate_adaptive_pvarp": None,
    "vent_safety_pacing": None,
}

SCHEMA_FASTPATH_ICD = {
    "report_type": "FastPathSummary",
    "followup_date": None,
    "device_model": None,
    "device_serial": None,
    "battery_eri": None,
    "battery_current_ua": None,
    "battery_remaining_pct": None,
    "sensing_v": None,
    "impedance_v": None,
    "impedance_hv": None,
    "mode": None,
    "vt_zone": None,
    "vf_zone": None,
    "detection_params": None,
    "therapy": None,
    "sense_ability_v": None,
    "sensing_config_v": None,
    "vt_vf_episodes": None,
}

SCHEMA_FASTPATH_PM = {
    "report_type": "FastPathSummary",
    "followup_date": None,
    "device_model": None,
    "device_serial": None,
    "battery_voltage_v": None,
    "battery_mag_freq_bpm": None,
    "battery_remaining_pct": None,
    "mode": None,
    "base_rate_bpm": None,
    "upper_rate_bpm": None,
    "av_paced_ms": None,
    "av_sensed_ms": None,
    "autocapture_a": None,
    "autocapture_v": None,
    "amplitude_a_v": None,
    "amplitude_v_v": None,
    "pulse_width_a_ms": None,
    "pulse_width_v_ms": None,
    "sense_ability_a": None,
    "sense_ability_v": None,
    "sensitivity_a_mv": None,
    "sensitivity_v_mv": None,
    "stim_a_pct": None,
    "stim_v_pct": None,
    "ams_episodes": None,
    "mode_switch_pct": None,
    "af_burden_pct": None,
    "diagnostics_since": None,
}

_BATTERY_BLOCK = {
    "implant_date": None,
    "voltage": None,
    "current_ua": None,
    "remaining_capacity_pct": None,
    "estimated_longevity": None,
    "rrt": None,
}
_TEST_BLOCK = {
    "pacing_threshold": None,
    "pacing_threshold_date": None,
    "sensing_mv": None,
    "sensing_date": None,
    "impedance_ohm": None,
    "impedance_date": None,
}
_STIM_BLOCK = {
    "configuration": None,
    "amplitude_v": None,
    "pulse_width_ms": None,
    "senseability": None,
    "sensitivity_mv": None,
}

SCHEMA_SUMMARY = {
    "report_type": "SummaryReport",
    "patient": None,
    "device": None,
    "facility": None,
    "interrogation_date": None,
    "visit_type": None,
    "battery_ra": dict(_BATTERY_BLOCK),
    "battery_rv": dict(_BATTERY_BLOCK),
    "test_ra": dict(_TEST_BLOCK),
    "test_rv": dict(_TEST_BLOCK),
    "parameters": {
        "mode": None,
        "base_rate": None,
        "av_paced_ms": None,
        "av_sensed_ms": None,
        "upper_rate_limit": None,
        "max_sensor_rate": None,
        "rate_hysteresis": None,
        "vip_enabled": None,
        "rate_adaptive_av": None,
        "rate_adaptive_pvarp": None,
        "ventricular_safety_pacing": None,
    },
    "stimulation_ra": dict(_STIM_BLOCK),
    "stimulation_rv": dict(_STIM_BLOCK),
    "diagnostics": {
        "since": None,
        "ap_paced_pct": None,
        "vp_paced_pct": None,
        "mode_switch_pct": None,
        "ams_episodes": None,
    },
}

# ── prompts ───────────────────────────────────────────────────────────────────

_COMMON_RULES = """\
IMPORTANT RULES:
- For NUMERIC fields: extract only the number, WITHOUT the unit suffix.
  "60 min⁻¹" → "60",  "2,5 V" → "2,5",  "0,4 ms" → "0,4",  "12 uA" → "12".
  Use German decimal comma (e.g. "3,08", "0,5").
- Preserve ">" or "<" prefixes (e.g. ">95", ">99", "> 5 Jhr.") and do NOT strip those.
- For TEXT fields (Aus, Ein, Auto, DDD, DDDR, Stimulation …): copy exactly as printed.
- For DATE fields: extract only the date, strip any leading word like "Seit".
  "Seit 3 Mär 2025" → "3 Mär 2025".
- IMPORTANT: If a test table cell says "Nicht durchgeführt", return that exact string — \
do NOT return null. Only return null if the label itself is missing from the image.
- Return ONLY the JSON object — no markdown fences, no explanation.
"""

PROMPT_FREEZE = """\
You are a medical data extraction assistant. Extract pacemaker parameters from \
this Abbott Merlin PCS FreezeReport (Ausschnitt) image.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- followup_date: top-right corner (format "dd Mon yyyy")
- device: centre top, line starting with "Aveir" — copy exactly
- facility: centre top, line starting with "HZ"
- Schlüsselparameter table (left column = label, right column = value):
  - mode ← "Betriebsart" (text, e.g. "DDD")
  - base_rate_bpm ← "Grundfrq." — extract ONLY the number (e.g. "50"), NOT "50 min⁻¹"
  - av_paced_ms ← "AV stimuliert" — number only (e.g. "200"), NOT "200 ms"
  - av_sensed_ms ← "AV wahrgen." — number only (e.g. "150")
  - upper_rate_bpm ← "Obere Grenzfrq." — number only (e.g. "130")
  - max_sensor_rate_bpm ← "Max. Sensorfrq." — number only (e.g. "130")
  - rate_hysteresis ← "Frequenzhysteresedif." (text, e.g. "Aus")
  - vip ← "Ventricular Intrinsic Preference (VIP™)" (text, e.g. "Aus")
  - rate_adaptive_av ← "Frequenzabh. AV-Intervall" (text, e.g. "Mittel")
  - rate_adaptive_pvarp ← "Frequenzabh. PVARP/VREF" (text, e.g. "Hoch")
  - vent_safety_pacing ← "Vent. Sicherheits-Stim." (text, e.g. "Ein")

""" + json.dumps(SCHEMA_FREEZE, ensure_ascii=False, indent=2)

PROMPT_FASTPATH_ICD = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS FastPath™ Übersicht report for an ICD device.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- followup_date: top-right corner (format "d Mon yyyy" e.g. "4 Apr 2025")
- device_model: top centre, the model code after "Entrant™ DR " (e.g. "CDDRA300Q")
- device_serial: top centre, the number after the model code (e.g. "810131959")
- Batterie section:
  - battery_eri ← value after "ERI" label (e.g. "> 5 Jhr.")
  - battery_current_ua ← "Batteriestrom" value (number, e.g. "12")
  - battery_remaining_pct ← "Verbleib. Kapazität bis ERI" value (e.g. ">95")
- Testergebnisse section (tests performed during follow-up — if a row shows
  "Nicht durchgeführt", return that exact string, NOT null):
  - sensing_v ← "Signalamplitude" value for V row (e.g. "Nicht durchgeführt" or a number)
  - impedance_v ← "Elektrodenimpedanz" value for V row
  - impedance_hv ← "Elektrodenimpedanz" value for HV coil row
- Parameter section (a table with label on the left, value on the right):
  - mode ← "Betriebsart" row — value may be "Stimulation", "VVI", "DDI" etc.
  - vt_zone ← "Zonenkonfiguration" row, VT column (often "Aus" = disabled)
  - vf_zone ← "Zonenkonfiguration" row, VF column (often "Aus")
  - detection_params ← "Detektionsparameter" row (often "Aus")
  - therapy ← row labelled "Therapie" or "Therapie (AUS)" (often "Aus")
- Stimulation und Wahrnehmung section — find the V (ventricular) column:
  - sense_ability_v ← "SenseAbility™" or "SenseAbility" row, V column (e.g. "Ein")
  - sensing_config_v ← "Empfindlichkeit" row, V column (e.g. "Auto" or a number)
- Diagnostikübersicht section:
  - vt_vf_episodes ← "VT/VF-Episoden:" number (return as string, e.g. "0")

""" + json.dumps(SCHEMA_FASTPATH_ICD, ensure_ascii=False, indent=2)

PROMPT_FASTPATH_PM = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS FastPath™ Übersicht report for a pacemaker device.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- followup_date: the report generation date shown in the TOP-RIGHT corner of the
  page, printed as part of the patient/device header (format "d Mon yyyy",
  e.g. "4 Apr 2025"). IGNORE any date that follows the word "Seit" — that is
  the diagnostics_since date, not the followup_date.
- device_model: top centre, model number after "Endurity MRI™ " (e.g. "2172")
- device_serial: top centre, number after the model (e.g. "5153558")
- Batterie section:
  - battery_voltage_v ← "Spannung" value (e.g. "3,08")
  - battery_mag_freq_bpm ← "Magnetfrequenz" value (number before "min⁻¹")
  - battery_remaining_pct ← "Verbleib. Kapazität bis ERI" value
- Testergebnisse section: fields are "Nicht durchgeführt" if no test was done
  - sensing_a / sensing_v ← "Signalamplitude" for A and V rows
  - impedance_a / impedance_v ← "Elektrodenimpedanz" for A and V rows
- Parameter section:
  - mode ← "Betriebsart"
  - base_rate_bpm ← "Grundfrequenz (GF)" value (number before "min-1")
  - upper_rate_bpm ← "Ob Grenzfr MTR" value
  - av_paced_ms ← "Stimuliertes AV-Intervall" value
  - av_sensed_ms ← "Wahrg. AV-Int." value
- Stimulation und Wahrnehmung section — A column then V column:
  - autocapture_a / autocapture_v ← "ACap™ Confirm/V. AutoCapture" row
  - amplitude_a_v / amplitude_v_v ← "Impulsamplitude" row (number before "V")
  - pulse_width_a_ms / pulse_width_v_ms ← "Impulsdauer" row (number before "ms")
  - sense_ability_a / sense_ability_v ← "SenseAbility™" row
  - sensitivity_a_mv / sensitivity_v_mv ← "Empfindlichkeit" row (number before "mV")
- Diagnostikübersicht section (statistics since last follow-up):
  - diagnostics_since ← date shown after the word "Seit" — strip the "Seit" prefix
    (e.g. "Seit 3 Mär 2025" → "3 Mär 2025")
  - stim_a_pct ← stimulation percentage for atrium: look for "AP" or "A Stimulation"
    percentage — may show ">99" if very high (return the printed value, no % sign)
  - stim_v_pct ← stimulation percentage for ventricle: look for "VP" or "V Stimulation"
    percentage (same format)
  - ams_episodes ← "AMS-Episoden" count (number as string, e.g. "0")
  - mode_switch_pct ← "Mode-Switch" percentage value (no % sign)
  - af_burden_pct ← "AT/AF-Burden" percentage value (no % sign)

""" + json.dumps(SCHEMA_FASTPATH_PM, ensure_ascii=False, indent=2)

PROMPT_SUMMARY = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS FastPath™ Übersicht / SummaryReport for an Aveir™ DR LP device.

IMPORTANT RULES:
- Copy values EXACTLY as printed, INCLUDING units (e.g. "2,5 V", "0,4 ms",
  "330 Ω", "12,0 uA", "99 %", "50 min⁻¹", "11,3 Jahre").
- Use German decimal comma (e.g. "3,08", "0,5").
- Preserve ">" and "<" prefixes (e.g. ">3,3 V", "<1%", ">5 Jahre").
- For test measurements: each row shows "Nicht durchgeführt" (today's test, not
  done) THEN the last measured value with its date in parentheses on the next line.
  Extract ONLY the last measured value and its date — ignore "Nicht durchgeführt".
  E.g. "2,75 V @ 0,4 ms (22 Jun 2026)" → pacing_threshold="2,75 V @ 0,4 ms",
  pacing_threshold_date="22 Jun 2026".
- For null/unavailable fields: use null (not a string).
- Return ONLY the JSON object — no markdown fences, no explanation.

WHERE TO FIND EACH FIELD:
- patient: top centre header (e.g. "DEMO")
- device: top centre, line starting "Aveir™ DR LP" (copy exactly as printed)
- facility: top centre, line starting "HZ" (e.g. "HZ Leipzig")
- interrogation_date: top-right corner, first date shown (e.g. "24 Jun 2026")
- visit_type: top-right corner, below the date (e.g. "Bei der Nachsorge")

Batterieinformationen section — RA subsection then RV subsection:
  For each (battery_ra / battery_rv):
  - estimated_longevity ← "Laufzeit: X Jahre" value (e.g. "11,3 Jahre")
  - rrt ← value shown next to "RRT" (e.g. ">5 Jahre")
  - implant_date ← "Datum der Implantation" value (e.g. "21 Jun 2026")
  - voltage ← "Spannung" value (e.g. ">3,3 V")
  - current_ua ← "Batteriestrom" value (e.g. "12,0 uA")
  - remaining_capacity_pct ← "Verbleibende Kapazität bis RRT" value (e.g. "99 %")

Testergeb. section — RA row then RV row, three columns: Reizschwelle / Wahrnehmung / Impedanz:
  For each lead (test_ra / test_rv), ignore "Nicht durchgeführt" lines and use only
  the line with the actual measured value and date in parentheses:
  - pacing_threshold ← value before the date in Reizschwelle (e.g. "2,75 V @ 0,4 ms")
  - pacing_threshold_date ← date in parentheses in Reizschwelle (e.g. "22 Jun 2026")
  - sensing_mv ← value before the date in Wahrnehmung (e.g. "7,5 mV")
  - sensing_date ← date in parentheses in Wahrnehmung (e.g. "19 Jun 2026")
  - impedance_ohm ← value before the date in Impedanz (e.g. "330 Ω")
  - impedance_date ← date in parentheses in Impedanz (e.g. "19 Jun 2026")

Parameter section (left side):
  - mode ← "Betriebsart" (e.g. "DDD")
  - base_rate ← "Grundfrq." value WITH unit (e.g. "50 min⁻¹")
  - max_sensor_rate ← "Max. Sensorfrq." value WITH unit (e.g. "130 min⁻¹")
  - upper_rate_limit ← "Obere Grenzfrq." value WITH unit (e.g. "130 min⁻¹")
  - av_paced_ms ← "AV stimuliert" value WITH unit (e.g. "200 ms")
  - av_sensed_ms ← "AV wahrgen." value WITH unit (e.g. "150 ms")
  - rate_hysteresis, vip_enabled, rate_adaptive_av, rate_adaptive_pvarp,
    ventricular_safety_pacing ← set null if not visible in this section

Stimulation & Wahr. section — RA column then RV column:
  For each (stimulation_ra / stimulation_rv):
  - configuration ← "Elektrodenkonfig." row (e.g. "Bipolar")
  - amplitude_v ← "Impulsampl." row WITH unit (e.g. "2,5 V")
  - pulse_width_ms ← "Impulsdauer" row WITH unit (e.g. "0,4 ms")
  - senseability ← "SenseAbility™" row (e.g. "Aus")
  - sensitivity_mv ← "Sensitivität" row WITH unit (e.g. "0,5 mV" for RA, "2,0 mV" for RV)

Diagnostikübersicht section (left block):
  - since ← "Seit:" value INCLUDING time (e.g. "22 Jun 2026 20:11")
  - ap_paced_pct ← "Stimuliert (AP)" value in the FIRST (Abgelastet) column (e.g. "27%")
  - vp_paced_pct ← "Stimuliert (VP)" value in the FIRST (Abgelastet) column (e.g. "16%")
  - mode_switch_pct ← "Mode-Switch" value (e.g. "<1%")
  - ams_episodes ← "AMS-Episoden" value as string (e.g. "1")

""" + json.dumps(SCHEMA_SUMMARY, ensure_ascii=False, indent=2)

# ── report type detection ─────────────────────────────────────────────────────

def detect_report_type(pdf_path: str | Path) -> str:
    """
    Detect report type from meta.json, PDF footer text, or path hints.
    Returns: 'freeze' | 'fastpath_icd' | 'fastpath_pm' | 'summary' | None
    """
    import json as _json
    import pdfplumber

    pdf_path = Path(pdf_path)

    # 1. Check meta.json in sample directory
    meta_path = pdf_path.parent / "meta.json"
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        rt = meta.get("report_type", "")
        dt = meta.get("device_type", "")
        if rt == "FreezeReport":
            return "freeze"
        if rt == "SummaryReport":
            return "summary"
        if "ICD" in dt or "S-ICD" in dt:
            return "fastpath_icd"
        if "Pacemaker" in dt or "PM" in dt:
            return "fastpath_pm"

    # 2. Try PDF text (works only if footer is text-based)
    with pdfplumber.open(str(pdf_path)) as pdf:
        text = (pdf.pages[0].extract_text() or "").lower()

    if "ausschnitt" in text or "freeze" in str(pdf_path).lower():
        return "freeze"
    if "icd" in text:
        return "fastpath_icd"
    if "schrittmacher" in text:
        return "fastpath_pm"

    return None  # unknown / unsupported report type

# ── VQA inference ─────────────────────────────────────────────────────────────

def _vqa(model, processor, image, prompt: str) -> str:
    from qwen_vl_utils import process_vision_info
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(raw)


_UNIT_SUFFIX = re.compile(
    r"\s+(min[⁻\-]1|min⁻¹|mV|Ohm|uA|UA)\s*$"
    r"|(?<=[0-9])\s+ms$"    # "0,4 ms" but not standalone text "ms"
    r"|(?<=[0-9])\s+V$"     # "2,5 V" but not "Ein" or "Aus"
    r"|\s*%$"               # ">95 %" or ">95%" → ">95"
)


def _postprocess(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, str):
            v = _UNIT_SUFFIX.sub("", v).strip()
            # Strip "Seit " prefix from date fields
            if k.endswith("_since") or k == "diagnostics_since":
                v = re.sub(r"^Seit\s+", "", v)
        cleaned[k] = v
    return cleaned


# ── entry point ───────────────────────────────────────────────────────────────

def extract(
    pdf_path: str | Path,
    model=None,
    processor=None,
    dpi: int = 300,
    report_type: Optional[str] = None,
) -> dict:
    """
    Extract from an Abbott Merlin PCS report PDF using VQA.

    model / processor must be pre-loaded Qwen2-VL objects (call vqa.load_model()).
    If report_type is None, it is auto-detected.
    """
    pdf_path = Path(pdf_path)
    if model is None or processor is None:
        raise ValueError("model and processor must be provided (load via vqa.load_model())")

    if report_type is None:
        report_type = detect_report_type(pdf_path)

    if report_type is None:
        raise NotImplementedError(
            f"Unsupported Abbott report type in {pdf_path}. "
            "Supported: 'freeze', 'fastpath_icd', 'fastpath_pm', 'summary'."
        )

    images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
    image = images[0]

    if report_type == "freeze":
        prompt = PROMPT_FREEZE
    elif report_type == "fastpath_icd":
        prompt = PROMPT_FASTPATH_ICD
    elif report_type == "fastpath_pm":
        prompt = PROMPT_FASTPATH_PM
    else:
        prompt = PROMPT_SUMMARY

    raw = _vqa(model, processor, image, prompt)
    result = _parse_json(raw)
    # SummaryReport keeps units in values; only strip for other report types
    if report_type != "summary":
        result = _postprocess(result)

    result["source_file"] = str(pdf_path)
    result["manufacturer"] = "Abbott"
    result["vqa_model"] = "Qwen2-VL-7B-Instruct"
    return result
