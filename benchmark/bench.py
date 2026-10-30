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


_V_RV_ALIASES = {
    "amplitude_v_v": "amplitude_rv_v",
    "pulse_width_v_ms": "pulse_width_rv_ms",
    "sensitivity_v_mv": "sensitivity_rv_mv",
    "autocapture_v": "autocapture_rv",
    "sense_ability_v": "sense_ability_rv",
    "sensing_config_v": "sensing_config_rv",
    "stim_v_pct": "stim_rv_pct",
    "impedance_v": "impedance_rv_ohm",
}
_RV_V_ALIASES = {v: k for k, v in _V_RV_ALIASES.items()}


def score(extracted: dict, truth: dict) -> tuple[int, int, list]:
    ef = _flatten(extracted)
    tf = _flatten(truth)
    # Also store _v_ ↔ _rv_ aliases in extracted so either convention matches
    for alias_map in (_V_RV_ALIASES, _RV_V_ALIASES):
        for src, dst in alias_map.items():
            if src in ef and dst not in ef:
                ef[dst] = ef[src]
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
        parts = pdf_path.parts
        try:
            idx = parts.index("abbott")
            return "__".join(parts[idx + 1:]).replace(".pdf", "")
        except ValueError:
            return "__".join(parts[-4:]).replace(".pdf", "")

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
        return result.get("data", {}).get("result", {}).get("outputs", {}).get("markdown", "") or ""

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        cached = self._load_cache(pdf_path, "oxr")
        if cached is not None:
            markdown = cached
        else:
            markdown = self._call_oxr(pdf_path)
            self._save_cache(pdf_path, "oxr", markdown)
        return {"_raw_markdown": markdown, "report_type": "unknown"}


# ── backend: OXR + LLM (document parser → light LLM for extraction) ─────────
# Import production schemas lazily to avoid pulling in pdf2image at module level.
import importlib as _il
import re as _re

def _load_abbott():
    """Import abbott_merlin without triggering pdf2image (not installed on bigboy)."""
    import types, importlib.util
    spec = importlib.util.spec_from_file_location(
        "abbott_merlin", PROJECT_ROOT / "extractors" / "abbott_merlin.py",
        submodule_search_locations=[],
    )
    # Stub pdf2image so import doesn't fail
    import sys
    _orig = sys.modules.get("pdf2image")
    sys.modules["pdf2image"] = types.ModuleType("pdf2image")
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if _orig is not None:
            sys.modules["pdf2image"] = _orig
        else:
            sys.modules.pop("pdf2image", None)
    return mod

_abbott = _load_abbott()
_COMMON_RULES = _abbott._COMMON_RULES
SCHEMA_FREEZE = _abbott.SCHEMA_FREEZE
SCHEMA_FASTPATH_ICD = _abbott.SCHEMA_FASTPATH_ICD
SCHEMA_FASTPATH_PM = _abbott.SCHEMA_FASTPATH_PM
SCHEMA_SUMMARY = _abbott.SCHEMA_SUMMARY
SCHEMA_WRAPUP_ICD = _abbott.SCHEMA_WRAPUP_ICD
SCHEMA_PSA = _abbott.SCHEMA_PSA
SCHEMA_PARAMETER_ICD = _abbott.SCHEMA_PARAMETER_ICD
SCHEMA_UNIFIED = _abbott.SCHEMA_UNIFIED
_KEY_RENAME = _abbott._KEY_RENAME
_UNIT_SUFFIX = _abbott._UNIT_SUFFIX


def _postprocess(result: dict, report_type: str = "") -> dict:
    """Strip units and normalize dates — mirrors abbott_merlin._postprocess
    but skips _KEY_RENAME since ground truths use _v_ convention.
    Summary reports keep units since their ground truths include them."""
    cleaned = {}
    skip_unit_strip = report_type == "summary"
    for k, v in result.items():
        if isinstance(v, str):
            v = _re.sub(r'^(?:rightarrow|[▸▶►‣•→])\s*', '', v).strip()
            if not skip_unit_strip:
                v = _UNIT_SUFFIX.sub("", v).strip()
            if k.endswith("_since") or k == "diagnostics_since" or k == "diagnostics.since":
                v = _re.sub(r"^Seit\s+", "", v)
        cleaned[k] = v
    return cleaned


def _simplify_markdown(md: str) -> str:
    """Convert OXR HTML-table markdown to pipe-delimited plain text."""
    out = _re.sub(r'!\[.*?\]\(.*?\)\s*', '', md)
    out = _re.sub(r'\$?\^?\{-1\}\$?', '⁻¹', out)
    out = _re.sub(r'\$\^?\{[^}]*\}\$?', '', out)
    out = _re.sub(r'</?table>', '', out)
    out = _re.sub(r'<tr>', '\n', out)
    out = _re.sub(r'</tr>', '', out)
    out = _re.sub(r'<td[^>]*>', ' | ', out)
    out = _re.sub(r'</td>', '', out)
    out = _re.sub(r'</?(?:th|thead|tbody)[^>]*>', '', out)
    out = _re.sub(r'&lt;', '<', out)
    out = _re.sub(r'&gt;', '>', out)
    out = _re.sub(r'&amp;', '&', out)
    out = _re.sub(r'(?<=[a-zA-Z])TM(?=[\s,.);\n]|$)', '™', out)
    out = out.replace('rightarrow', '→')
    out = _re.sub(r'\n{3,}', '\n\n', out)
    return out.strip()


def _classify_markdown(text: str) -> str:
    """Classify Abbott report type from OXR markdown text."""
    low = text[:3000].lower()
    if "psa-bericht" in low:
        return "psa"
    if "ausschnitt" in low or "schlüsselparameter" in low or "freeze" in low:
        return "freeze"
    if "wrap-up" in low or "neue programmierung" in low:
        return "wrapup"
    # Aveir summary: has RA/RV battery subsections and Testergeb. with RA/RV rows
    if ("batterieinformation" in low or ("laufzeit" in low and "\nra\n" in low.replace("\r", ""))) \
            and ("testergeb" in low or "stimulation & wahr" in low):
        return "summary"
    if "modus & sensor" in low or "refraktärzeiten" in low:
        return "parameters_icd"
    if "fastpath" in low or "fastpath™" in low:
        if "vt/vf" in low or "zonenkonfiguration" in low or "therapie" in low:
            return "fastpath_icd"
        return "fastpath_pm"
    if "icd" in low or "defibrill" in low or "entrant" in low:
        return "fastpath_icd"
    if "schrittmacher" in low:
        return "fastpath_pm"
    return "unified"


def _build_text_prompt(report_type: str, markdown: str) -> str:
    """Build a per-report-type extraction prompt adapted for OCR text input."""

    # Reuse _COMMON_RULES from production (unit stripping, etc.)
    intro = "You are a medical data extraction assistant. Below is structured " \
            "text (markdown with tables) extracted from an Abbott Merlin PCS " \
            "report page. Extract the requested fields.\n\n"

    schema_map = {
        "freeze":         SCHEMA_FREEZE,
        "fastpath_icd":   SCHEMA_FASTPATH_ICD,
        "fastpath_pm":    SCHEMA_FASTPATH_PM,
        "summary":        SCHEMA_SUMMARY,
        "wrapup":         SCHEMA_WRAPUP_ICD,
        "psa":            SCHEMA_PSA,
        "parameters_icd": SCHEMA_PARAMETER_ICD,
        "unified":        SCHEMA_UNIFIED,
    }
    schema = schema_map.get(report_type, SCHEMA_UNIFIED)

    rules = _COMMON_RULES

    # Per-type field-location hints adapted from the production VQA prompts
    hints = {
        "freeze": """\
WHERE TO FIND EACH FIELD:
- followup_date: look for a date like "24 Jun 2026" near the header
- device: line mentioning "Aveir" or device name — copy exactly
- facility: line starting with "HZ"
- Schlüsselparameter table (left column = label, right column = value):
  mode ← "Betriebsart", base_rate_bpm ← "Grundfrq." (number only),
  av_paced_ms ← "AV stimuliert" (number only), av_sensed_ms ← "AV wahrgen." (number only),
  upper_rate_bpm ← "Obere Grenzfrq." (number only), max_sensor_rate_bpm ← "Max. Sensorfrq." (number only),
  rate_hysteresis ← "Frequenzhysteresedif.", vip ← "Ventricular Intrinsic Preference (VIP™)",
  rate_adaptive_av ← "Frequenzabh. AV-Intervall", rate_adaptive_pvarp ← "Frequenzabh. PVARP/VREF",
  vent_safety_pacing ← "Vent. Sicherheits-Stim."
""",
        "fastpath_pm": """\
WHERE TO FIND EACH FIELD:
- followup_date: date near header (e.g. "4 Apr 2025"). If time on next line, combine as "d Mon yyyy HH:MM".
- device_model: model NUMBER after device name (e.g. "2172" from "Endurity MRI™ 2172")
- device_serial: number after model (e.g. "5153558")
- Batterie: battery_voltage_v ← "Spannung", battery_mag_freq_bpm ← "Magnetfrequenz",
  battery_remaining_pct ← "Verbleib. Kapazität bis ERI"
- Testergebnisse: For each lead (A, V/RV, LV if CRT):
  threshold_*_v ← "Reizschwelle" amplitude (number before "V"),
  threshold_*_pw_ms ← pulse width (number after "@" before "ms"),
  sensing_*_mv ← "Signalamplitude", impedance_*_ohm ← "Elektrodenimpedanz"
  If "Nicht durchgeführt" return that exact string, NOT null.
- Parameter: mode ← "Betriebsart", base_rate_bpm ← "Grundfrequenz (GF)" (number only),
  upper_rate_bpm ← "Ob Grenzfr MTR", av_paced_ms ← "Stimuliertes AV-Intervall",
  av_sensed_ms ← "Wahrg. AV-Int."
- Stimulation und Wahrnehmung: autocapture_* ← "ACap™ Confirm/V. AutoCapture",
  amplitude_*_v ← "Impulsamplitude" (number before "V"),
  pulse_width_*_ms ← "Impulsdauer" (number before "ms"),
  sense_ability_* ← "SenseAbility™", sensitivity_*_mv ← "Empfindlichkeit" (number before "mV")
- Diagnostik: diagnostics_since ← date after "Seit" (strip "Seit" prefix),
  stim_a_pct ← "AP" %, stim_rv_pct ← "VP" %, ams_episodes ← "AMS-Episoden",
  mode_switch_pct ← "Mode-Switch", af_burden_pct ← "AT/AF-Burden"
""",
        "fastpath_icd": """\
WHERE TO FIND EACH FIELD:
- followup_date: date near header. If time on next line, combine as "d Mon yyyy HH:MM".
- device_model: model CODE after device name (e.g. "CDDRA300Q" from "Entrant™ DR CDDRA300Q")
- device_serial: number after model code
- Batterie: battery_eri ← "ERI", battery_current_ua ← "Batteriestrom",
  battery_remaining_pct ← "Verbleib. Kapazität bis ERI"
- Testergebnisse: threshold_rv_v, threshold_rv_pw_ms, sensing_rv_mv, impedance_rv_ohm,
  threshold_lv_v, threshold_lv_pw_ms, sensing_lv_mv, impedance_lv_ohm,
  shock_impedance_ohm ← HV coil impedance
- Parameter: mode ← "Betriebsart" (word like "Stimulation", "VVI", "DDI"),
  vt_zone ← "Zonenkonfiguration" VT column, vf_zone ← VF column,
  detection_params ← "Detektionsparameter", therapy ← "Therapie"
- Stimulation: sense_ability_rv ← "SenseAbility™" RV/V column,
  sensing_config_rv ← "Empfindlichkeit" RV/V column,
  sense_ability_lv, sensing_config_lv ← LV column (null if no LV)
- Diagnostik: vt_vf_episodes ← "VT/VF-Episoden:" number
""",
        "summary": """\
WHERE TO FIND EACH FIELD (SummaryReport for Aveir™ DR LP):
IMPORTANT: This report uses units in values — copy EXACTLY as printed INCLUDING units.
- patient: header (e.g. "DEMO"), device: "Aveir™ DR LP..." line, facility: "HZ..." line
- interrogation_date: top-right date, visit_type: below date
- Batterieinformationen — RA then RV subsections:
  For each (battery_ra / battery_rv): estimated_longevity ← "Laufzeit",
  rrt ← "RRT", implant_date ← "Datum der Implantation", voltage ← "Spannung",
  current_ua ← "Batteriestrom", remaining_capacity_pct ← "Verbleibende Kapazität bis RRT"
- Testergeb. — RA then RV rows: For each (test_ra / test_rv):
  pacing_threshold ← value in Reizschwelle (e.g. "2,75 V @ 0,4 ms"),
  pacing_threshold_date ← date in parentheses, sensing_mv ← Wahrnehmung value,
  sensing_date, impedance_ohm ← Impedanz value, impedance_date
- Parameter: mode ← "Betriebsart", base_rate ← "Grundfrq." WITH unit,
  max_sensor_rate, upper_rate_limit, av_paced_ms, av_sensed_ms (all WITH units)
- Stimulation: For each (stimulation_ra / stimulation_rv):
  configuration ← "Elektrodenkonfig.", amplitude_v ← "Impulsampl." WITH unit,
  pulse_width_ms ← "Impulsdauer" WITH unit, senseability ← "SenseAbility™",
  sensitivity_mv ← "Sensitivität" WITH unit
- Diagnostik: since ← "Seit:" value WITH time, ap_paced_pct, vp_paced_pct,
  mode_switch_pct, ams_episodes
""",
        "wrapup": """\
WHERE TO FIND EACH FIELD:
- followup_date: date near header. If time, combine as "d Mon yyyy HH:MM".
- device_model: model code (e.g. "3367-40QC"), device_serial: number after model
- Patient: patient_dob ← "Geburtsdatum"
- Aggregat table (rows for device, A Elektrode, RV/V Elektrode, LV Elektrode):
  implant_date ← generator row, lead_a_model/serial/implant_date ← A row,
  lead_rv_model/serial/implant_date ← RV/V row, lead_lv_* ← LV row (null if absent)
- Batterie: battery_voltage_v ← "Spannung", battery_longevity ← "Laufzeit" WITH unit,
  battery_charging_time_s ← "Letzte max. Aufladung", battery_current_ua ← "Batteriestrom",
  battery_remaining_pct ← "Verbleib. Kapazität bis ERI", battery_eri ← "ERI"
- Testergebnisse: threshold_*_v, threshold_*_pw_ms, sensing_*_mv, impedance_*_ohm
  for A, RV, LV rows; impedance_hv_ohm ← HV coil
- Parameter: mode, base_rate_bpm, upper_rate_bpm, av_paced_ms, av_sensed_ms,
  vt_zone, vf_zone, therapy
""",
        "psa": """\
WHERE TO FIND EACH FIELD:
- implant_date: date in header, device_model: model number (e.g. "2172"), device_serial
- Lead table (A, RV rows): lead_a_model, lead_a_serial, lead_a_implant_date,
  lead_rv_model, lead_rv_serial, lead_rv_implant_date
- Testergebnisse (Reizschw. / Signalamplitude / Impedanz):
  For A: threshold_a_v ← amplitude before "V", threshold_a_pw_ms ← after "bei" before "ms",
  threshold_a_ma ← current before "mA", sensing_a_mv ← before "mV",
  sensing_a_slew_vs ← before "V/s", impedance_a_ohm ← before "Ω"
  For RV: same pattern with _rv_ prefix.
""",
        "parameters_icd": """\
WHERE TO FIND EACH FIELD:
- followup_date: header date. device_model: code after device name, device_serial: number after model
- Aggregat table: implant_date ← ICD row, lead_a_model/serial/implant_date ← A row,
  lead_rv_model/serial/implant_date ← V row
- Modus & Sensor: mode ← "Betriebsart"
- Frequenzen: base_rate_bpm ← "Grundfrequenz", upper_rate_bpm ← "Obere Grenzfrequenz"
- Intervalle: av_paced_ms ← "Stimuliertes AV-Intervall", av_sensed_ms ← "Wahrg. AV-Int."
- Stimulation: amplitude_a_v, pulse_width_a_ms (A column), amplitude_rv_v, pulse_width_rv_ms (V column)
""",
    }

    hint_block = hints.get(report_type, "Extract all visible fields.\n")

    markdown = _simplify_markdown(markdown)

    flat = _flatten(schema, "")
    schema_json = json.dumps(flat, ensure_ascii=False, indent=2)

    if report_type == "summary":
        rules = (
            "IMPORTANT RULES:\n"
            "- Copy values EXACTLY as printed INCLUDING units (e.g. '12,0 uA', '99 %', '2,5 V', '0,4 ms').\n"
            "- Use German decimal comma (e.g. '3,08', '0,5').\n"
            "- Preserve '>' or '<' prefixes (e.g. '>3,3 V', '<1%').\n"
            "- 'Nicht durchgeführt' means 'not performed' — if a value follows it, extract the value, not 'Nicht durchgeführt'.\n"
            "- null for any field not visible in the text.\n"
        )

    return (
        intro + rules + "\n" + hint_block + "\n"
        "Return a FLAT JSON object matching these keys (null for missing fields):\n"
        + schema_json + "\n\nOCR TEXT:\n" + markdown
    )


def _query_ollama(url: str, model: str, prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    import urllib.request
    prompt = "/no_think\n" + prompt
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 4096, "num_ctx": 16384},
    })
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
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
        report_type = _classify_markdown(markdown)
        prompt = _build_text_prompt(report_type, markdown[:8000])
        raw = _query_ollama(self.llm_url, self.llm_model, prompt)
        result = _parse_llm_json(raw)
        result = _postprocess(result, report_type)
        return result


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
        report_type = _classify_markdown(text)
        prompt = _build_text_prompt(report_type, text[:8000])
        raw = _query_ollama(self.llm_url, self.llm_model, prompt)
        result = _parse_llm_json(raw)
        result = _postprocess(result, report_type)
        return result


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

        prompt = _build_text_prompt("unified",
            "[See the attached image of an Abbott pacemaker report]")
        payload = json.dumps({
            "model": self.llm_model,
            "prompt": prompt,
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
        result = _parse_llm_json(raw)
        result = _postprocess(result, "unified")
        return result


# ── backend: InternVL3.5-8B (VLM contender) ──────────────────────────────────

def _internvl_build_transform(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448,
                                  use_thumbnail=True):
    import math
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    ), key=lambda x: x[0] * x[1])
    best_ratio = min(target_ratios,
                     key=lambda r: abs(aspect_ratio - r[0] / r[1]))
    tw = image_size * best_ratio[0]
    th = image_size * best_ratio[1]
    resized = image.resize((tw, th))
    blocks = best_ratio[0] * best_ratio[1]
    cols = tw // image_size
    tiles = [resized.crop(((i % cols) * image_size, (i // cols) * image_size,
                           ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size))
             for i in range(blocks)]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_load_image(pil_img, max_num=12):
    import torch
    transform = _internvl_build_transform()
    tiles = _internvl_dynamic_preprocess(pil_img, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values.to(torch.bfloat16)


class InternVLBackend(Backend):
    name = "internvl"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        import transformers.modeling_utils as _mu

        _orig_init_subclass = _mu.PreTrainedModel.__init_subclass__
        @classmethod
        def _patched_init_subclass(cls, **kw):
            _orig_init_subclass.__func__(cls, **kw)
            if not hasattr(cls, "all_tied_weights_keys"):
                cls.all_tied_weights_keys = {}
        _mu.PreTrainedModel.__init_subclass__ = _patched_init_subclass

        model_id = "OpenGVLab/InternVL3_5-8B"
        device = kwargs.get("device", "cuda")
        print(f"Loading {model_id} (BNB 4-bit) on {device} …")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False,
        )
        _mu.PreTrainedModel.__init_subclass__ = _orig_init_subclass
        print("InternVL3.5-8B ready.")

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        import torch
        import pdf2image
        from extractors.abbott_merlin import (
            detect_report_type, _parse_json, _postprocess,
            PROMPT_SUMMARY, PROMPT_FASTPATH_ICD, PROMPT_FASTPATH_PM,
            PROMPT_FREEZE, PROMPT_WRAPUP_ICD, PROMPT_PSA,
            PROMPT_PARAMETER_ICD, PROMPT_UNIFIED,
        )
        rt = detect_report_type(pdf_path)
        prompts = {
            "freeze": PROMPT_FREEZE, "fastpath_icd": PROMPT_FASTPATH_ICD,
            "fastpath_pm": PROMPT_FASTPATH_PM, "summary": PROMPT_SUMMARY,
            "wrapup": PROMPT_WRAPUP_ICD, "psa": PROMPT_PSA,
            "parameters_icd": PROMPT_PARAMETER_ICD,
        }
        prompt = prompts.get(rt, PROMPT_UNIFIED)

        images = pdf2image.convert_from_path(str(pdf_path), dpi=200,
                                              first_page=1, last_page=1)
        pixel_values = _internvl_load_image(images[0], max_num=6).to(self.model.device)

        question = "<image>\n" + prompt
        generation_config = {"max_new_tokens": 2048, "do_sample": False}
        response = self.model.chat(
            self.tokenizer, pixel_values, question,
            generation_config, history=None, return_history=False,
        )
        result = _parse_json(response)
        if rt != "summary":
            result = _postprocess(result)
        return result


# ── backend: Qwen3-VL-32B (larger Qwen VLM) ─────────────────────────────────

class Qwen32BBackend(Backend):
    name = "qwen-vl-32b"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        model_id = "Qwen/Qwen3-VL-32B-Instruct"
        device = kwargs.get("device", "cuda")
        print(f"Loading {model_id} (4-bit) on {device} …")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Qwen3-VL-32B ready.")

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        from extractors.abbott_merlin import extract, detect_report_type
        rt = detect_report_type(pdf_path)
        return extract(pdf_path, self.model, self.processor, report_type=rt)


# ── backend: Qwen3.8-27B via Ollama (GGUF Q4, vision) ──────────────────────

class Qwen38Backend(Backend):
    name = "qwen3.8-27b"

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.llm_url = kwargs.get("llm_url", "http://localhost:11434")
        self.llm_model = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"
        self._dpi = kwargs.get("dpi", 200)
        print(f"Using {self.llm_model} via Ollama at {self.llm_url}")

    def _ollama_vqa(self, b64_image: str, prompt: str, num_predict: int = 4096) -> str:
        return self._ollama_vqa_multi([b64_image], prompt, num_predict)

    def _ollama_vqa_multi(self, b64_images: list[str], prompt: str, num_predict: int = 4096) -> str:
        import urllib.request
        num_ctx = 16384 + 4096 * max(0, len(b64_images) - 1)
        payload = json.dumps({
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt, "images": b64_images}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx},
        })
        req = urllib.request.Request(
            f"{self.llm_url}/api/chat",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            return json.loads(resp.read()).get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            raise RuntimeError(f"Ollama HTTP {e.code}: {body}")

    def _image_to_b64(self, image) -> str:
        import base64, io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _classify_report(self, image) -> str | None:
        from extractors.abbott_merlin import CLASSIFY_PROMPT_FAST
        w, h = image.size
        crop = image.crop((0, 0, w, int(h * 0.30)))
        b64 = self._image_to_b64(crop)
        raw = self._ollama_vqa(b64, CLASSIFY_PROMPT_FAST, num_predict=20).strip().lower()
        for rt in ("testergebnisse_icd", "parameters_icd", "fastpath_icd",
                    "fastpath_pm", "summary", "freeze", "wrapup", "psa"):
            if rt in raw:
                return rt
        return None

    def extract(self, pdf_path: Path, meta: dict) -> dict:
        import pdf2image
        from extractors.abbott_merlin import (
            detect_report_type, _parse_json, _postprocess, _page_count,
            _KEY_RENAME, _backfill_schema,
            PROMPT_SUMMARY, PROMPT_FASTPATH_ICD, PROMPT_FASTPATH_PM,
            PROMPT_FREEZE, PROMPT_WRAPUP_ICD, PROMPT_PSA,
            PROMPT_PARAMETER_ICD, PROMPT_TESTERGEBNISSE_RV,
            SCHEMA_SUMMARY,
        )

        images = pdf2image.convert_from_path(str(pdf_path), dpi=self._dpi,
                                              first_page=1, last_page=1)
        image = images[0]
        threshold_result: dict = {}

        rt = detect_report_type(pdf_path)
        if rt is None:
            rt = self._classify_report(image)
        if rt is None:
            rt = "summary"

        if rt == "parameters_icd":
            n = _page_count(pdf_path)
            if n > 1:
                last_imgs = pdf2image.convert_from_path(
                    str(pdf_path), dpi=self._dpi, first_page=n, last_page=n)
                last_type = self._classify_report(last_imgs[0])
                if last_type == "wrapup":
                    image = last_imgs[0]
                    rt = "wrapup"

        if rt == "testergebnisse_icd":
            n = _page_count(pdf_path)
            if n > 1:
                p2_imgs = pdf2image.convert_from_path(
                    str(pdf_path), dpi=self._dpi, first_page=2, last_page=2)
                p2_type = self._classify_report(p2_imgs[0])
                if p2_type in ("parameters_icd", "testergebnisse_icd", None):
                    b64_p1 = self._image_to_b64(image)
                    raw_thr = self._ollama_vqa(b64_p1, PROMPT_TESTERGEBNISSE_RV)
                    try:
                        threshold_result = _parse_json(raw_thr)
                        threshold_result = _postprocess(threshold_result)
                    except Exception:
                        threshold_result = {}
                    image = p2_imgs[0]
                    rt = "parameters_icd"

        prompts = {
            "psa": PROMPT_PSA, "freeze": PROMPT_FREEZE,
            "fastpath_icd": PROMPT_FASTPATH_ICD,
            "fastpath_pm": PROMPT_FASTPATH_PM,
            "wrapup": PROMPT_WRAPUP_ICD,
            "summary": PROMPT_SUMMARY,
        }

        if rt == "parameters_icd":
            n = _page_count(pdf_path)
            if n > 1:
                all_imgs = pdf2image.convert_from_path(str(pdf_path), dpi=self._dpi)
                b64s = [self._image_to_b64(img) for img in all_imgs]
            else:
                b64s = [self._image_to_b64(image)]
            raw = self._ollama_vqa_multi(b64s, PROMPT_PARAMETER_ICD)
            result = _parse_json(raw)
            result = _postprocess(result)
        else:
            prompt = prompts.get(rt, PROMPT_SUMMARY)
            b64 = self._image_to_b64(image)
            raw = self._ollama_vqa(b64, prompt)
            result = _parse_json(raw)
            if rt != "summary":
                result = _postprocess(result)

        if rt == "summary":
            _backfill_schema(result, SCHEMA_SUMMARY)

        for k, v in threshold_result.items():
            if k not in ("report_type", "source_file", "manufacturer", "vqa_model") and v is not None:
                result.setdefault(k, v)
                if result.get(k) is None:
                    result[k] = v

        _SKIP_MERGE = {"report_type", "source_file", "manufacturer", "vqa_model",
                       "patient_name", "followup_date", "device_model", "device_serial"}
        if rt != "wrapup":
            n = _page_count(pdf_path)
            for pg in range(2, n + 1):
                pg_imgs = pdf2image.convert_from_path(
                    str(pdf_path), dpi=self._dpi, first_page=pg, last_page=pg)
                pg_type = self._classify_report(pg_imgs[0])
                if pg_type == "wrapup":
                    b64_wu = self._image_to_b64(pg_imgs[0])
                    raw_wu = self._ollama_vqa(b64_wu, PROMPT_WRAPUP_ICD)
                    try:
                        wu = _parse_json(raw_wu)
                        wu = _postprocess(wu)
                        wu = {_KEY_RENAME.get(k, k): v for k, v in wu.items()}
                        for k, v in wu.items():
                            if k in _SKIP_MERGE or v is None:
                                continue
                            if result.get(k) is None:
                                result[k] = v
                    except Exception:
                        pass
                    break

        result = {_KEY_RENAME.get(k, k): v for k, v in result.items()}
        return result


# ── registry ─────────────────────────────────────────────────────────────────

BACKENDS = {
    "qwen-vl":       QwenVLBackend,
    "internvl":      InternVLBackend,
    "qwen-vl-32b":   Qwen32BBackend,
    "qwen3.8-27b":   Qwen38Backend,
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
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
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
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

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
