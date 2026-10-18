"""
Boston Scientific EMBLEM S-ICD extractor.

Primary source: HL7 v2.6 sidecar file (IHE PCD-09, MDC-coded OBX segments).
Supplement: pdfplumber PDF for system_impedance_ohm (absent from HL7).

The HL7 file is a single continuous string (no newline segment separators).
"""
import re
from pathlib import Path
from typing import Optional

import pdfplumber


# ── HL7 helpers ───────────────────────────────────────────────────────────────

def _obx_map(hl7: str) -> list[dict]:
    """Return list of {mdc, group, value, unit} dicts for all OBX segments."""
    results = []
    # OBX|seq|type|code^mdc_name^MDC|group|value|unit|...|F
    for m in re.finditer(
        r"OBX\|(\d+)\|\w+\|(?:\d+)\^([^|^]+)\^MDC\|([^|]*)\|([^|]*)\|([^|]*)\|",
        hl7,
    ):
        results.append({
            "mdc": m.group(2),
            "group": m.group(3),
            "value": m.group(4),
            "unit": m.group(5),
        })
    return results


def _obx_val(obs: list[dict], mdc: str, group: str = "") -> Optional[str]:
    """Return first matching OBX value for given MDC code and optional group."""
    for o in obs:
        if o["mdc"] == mdc and (not group or o["group"] == group):
            v = o["value"]
            if not v:
                return None
            # CWE values: "code^display^system" — return the display name
            if "^" in v:
                parts = v.split("^")
                return parts[1] if len(parts) > 1 else parts[0]
            return v
    return None


def _norm_hl7_date(raw: Optional[str]) -> Optional[str]:
    """Convert HL7 datetime YYYYMMDD[HHmmss][+TZ] → DD.MM.YYYY."""
    if not raw:
        return None
    m = re.match(r"(\d{4})(\d{2})(\d{2})", raw)
    return f"{m.group(3)}.{m.group(2)}.{m.group(1)}" if m else raw


def _ms_to_bpm(ms_str: Optional[str]) -> Optional[str]:
    """Convert detection interval in ms to bpm (rounded)."""
    if not ms_str:
        return None
    try:
        return str(60000 // int(ms_str))
    except (ValueError, ZeroDivisionError):
        return None


# ── PDF supplement ─────────────────────────────────────────────────────────────

def _parse_pdf(pdf_path: str | Path) -> dict:
    """Extract fields available only in the PDF (primarily system impedance)."""
    out: dict = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        text = pdf.pages[0].extract_text() or ""

    m = re.search(r"Systemimpedanz\s*=\s*([\d.]+)\s*Ohm", text, re.IGNORECASE)
    if m:
        out["system_impedance_ohm"] = m.group(1)

    # Print date as fallback followup_date
    m = re.search(r"Datum der Nachsorgeuntersuchung:\s*(\d{2}/\d{2}/\d{4})", text)
    if m:
        parts = m.group(1).split("/")
        out["followup_date_pdf"] = f"{parts[1]}.{parts[0]}.{parts[2]}"

    # Last follow-up date
    m = re.search(r"Datum der letzten Nachsorgeuntersuchung:\s*(\d{2}/\d{2}/\d{4})", text)
    if m:
        parts = m.group(1).split("/")
        out["last_followup"] = f"{parts[1]}.{parts[0]}.{parts[2]}"

    m = re.search(r"Softwareversion des Programmiergeräts:\s*([\d.]+)", text)
    if m:
        out["programmer_software_version"] = m.group(1)

    m = re.search(r"Softwareversion des Geräts:\s*([\d.]+)", text)
    if m:
        out["device_software_version"] = m.group(1)

    m = re.search(r"SMART\s*Pass:\s*(\w+)", text)
    if m:
        out["smart_pass"] = m.group(1)

    m = re.search(r"Post\s*Shock\s*Pacing:\s*(\w+)", text)
    if m:
        out["post_shock_pacing"] = m.group(1)

    m = re.search(r"Verstärkungseinstellung:\s*([\w.]+)", text)
    if m:
        out["gain_setting"] = m.group(1)

    m = re.search(r"Wahrnehmungskonfiguration:\s*(\w+)", text)
    if m:
        out["sensing_configuration"] = m.group(1)

    m = re.search(r"Schockpolarität:\s*(\w+)", text)
    if m:
        out["shock_polarity"] = m.group(1)

    return out


# ── entry point ────────────────────────────────────────────────────────────────

def extract(pdf_path: str | Path, hl7_path: Optional[str | Path] = None) -> dict:
    """Extract from BSC EMBLEM S-ICD GESAMTBERICHT (HL7 primary + PDF supplement)."""
    pdf_path = Path(pdf_path)
    if hl7_path is None:
        hl7_path = pdf_path.parent / "report.hl7"

    result: dict = {
        "source_file": str(pdf_path),
        "manufacturer": "Boston Scientific",
        "report_type": "GESAMTBERICHT",
    }

    # ── HL7 ──────────────────────────────────────────────────────────────────
    hl7 = Path(hl7_path).read_text(encoding="utf-8", errors="replace")

    # Patient name from PID field 5: "Familienname^Vorname"
    pid_m = re.search(r"PID\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|([^|]+)\|", hl7)
    if pid_m:
        parts = pid_m.group(1).split("^")
        if len(parts) >= 2:
            result["patient_name"] = f"{parts[0]}, {parts[1]}"
        else:
            result["patient_name"] = parts[0]

    obs = _obx_map(hl7)

    result["followup_date"] = _norm_hl7_date(_obx_val(obs, "MDC_IDC_SESS_DTM"))
    result["device_model"]  = _obx_val(obs, "MDC_IDC_DEV_MODEL")
    result["device_serial"] = _obx_val(obs, "MDC_IDC_DEV_SERIAL")
    result["implant_date"]  = _norm_hl7_date(_obx_val(obs, "MDC_IDC_DEV_IMPLANT_DT"))
    result["lead_model"]    = _obx_val(obs, "MDC_IDC_LEAD_MODEL")
    result["lead_serial"]   = _obx_val(obs, "MDC_IDC_LEAD_SERIAL")

    result["battery_remaining_pct"] = _obx_val(obs, "MDC_IDC_MSMT_BATTERY_REMAINING_PERCENTAGE")

    result["shocks_delivered_recent"] = _obx_val(obs, "MDC_IDC_STAT_TACHYTHERAPY_SHOCKS_DELIVERED_RECENT")
    result["shocks_delivered_total"]  = _obx_val(obs, "MDC_IDC_STAT_TACHYTHERAPY_SHOCKS_DELIVERED_TOTAL")

    # Detection zones — group "1" = VF, "2" = VT (conditional)
    vf_ms = _obx_val(obs, "MDC_IDC_SET_ZONE_DETECTION_INTERVAL", group="1")
    vt_ms = _obx_val(obs, "MDC_IDC_SET_ZONE_DETECTION_INTERVAL", group="2")
    result["vf_zone_bpm"] = _ms_to_bpm(vf_ms)
    result["vt_zone_bpm"] = _ms_to_bpm(vt_ms)
    result["vf_shock_energy_j"] = _obx_val(obs, "MDC_IDC_SET_ZONE_SHOCK_ENERGY_1", group="1")
    result["vt_shock_energy_j"] = _obx_val(obs, "MDC_IDC_SET_ZONE_SHOCK_ENERGY_1", group="2")

    # Therapy status from HL7
    therapy_raw = _obx_val(obs, "MDC_IDC_SET_TACHYTHERAPY_VSTAT")
    if therapy_raw:
        result["therapy"] = "ON" if "On" in therapy_raw else "OFF"

    # Untreated episodes from HL7 (group 1 = "Other" = untreated)
    result["untreated_episodes_recent"] = _obx_val(obs, "MDC_IDC_STAT_EPISODE_RECENT_COUNT", group="1")
    result["untreated_episodes_total"] = _obx_val(obs, "MDC_IDC_STAT_EPISODE_TOTAL_COUNT", group="1")

    # ── PDF supplement ────────────────────────────────────────────────────────
    pdf_data = _parse_pdf(pdf_path)
    result["system_impedance_ohm"] = pdf_data.get("system_impedance_ohm")
    result["last_followup"] = pdf_data.get("last_followup")
    if not result.get("followup_date"):
        result["followup_date"] = pdf_data.get("followup_date_pdf")

    result["programmer_software_version"] = pdf_data.get("programmer_software_version")
    result["device_software_version"] = pdf_data.get("device_software_version")
    result["smart_pass"] = pdf_data.get("smart_pass")
    result["post_shock_pacing"] = pdf_data.get("post_shock_pacing")
    result["gain_setting"] = pdf_data.get("gain_setting")
    result["sensing_configuration"] = pdf_data.get("sensing_configuration")
    result["shock_polarity"] = pdf_data.get("shock_polarity")

    return {k: v for k, v in result.items() if v is not None}
