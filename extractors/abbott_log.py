"""
Abbott Merlin PCS .log file extractor.

The Merlin programmer exports a companion .log file alongside the session PDF.
Format: each line is four FS (\x1c) separated fields:
    code \x1c field_name \x1c value \x1c unit \n

Example:
    306\x1cRV Pulse Amplitude\x1c2.5\x1cV\n
    301\x1cMode\x1cDDD\x1c\n

This extractor produces the same FastPath PM schema as the VQA PDF extractor
(extractors/abbott_merlin.py) for fields present in the log, plus additional
fields not visible on the FastPath summary page.

Usage:
    from extractors.abbott_log import extract
    result = extract("4177548.log")
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

SEP = '\x1c'

# English→German translations for values that appear on the German FastPath display
_EN_DE = {
    "Off": "Aus",
    "On":  "Ein",
}

# Atrial AutoSense "Fixed" = SenseAbility disabled (Aus); "Auto"/"Automatic" = enabled (Ein)
_AUTOSENSE = {
    "Fixed":     "Aus",
    "Auto":      "Ein",
    "Automatic": "Ein",
    "Off":       "Aus",
    "On":        "Ein",
}


def _detect_device_class(rec: dict) -> str:
    """Return 'CRT-D', 'ICD', or 'PM' from lead configuration in log."""
    if "Model Number: SJM LV Lead" in rec or "LV Lead Serial Number" in rec:
        return "CRT-D"
    if "Model Number: SJM RV Defib Lead" in rec or "VF Detection Interval" in rec:
        return "ICD"
    return "PM"


def _parse(log_path: str | Path) -> dict[str, tuple[str, str]]:
    """Return {field_name: (value, unit)} for every record in the log."""
    records: dict[str, tuple[str, str]] = {}
    with open(log_path, encoding='latin-1') as f:
        for line in f:
            parts = line.rstrip('\n').split(SEP)
            if len(parts) < 3:
                continue
            _code, name, value = parts[0], parts[1], parts[2]
            unit = parts[3] if len(parts) > 3 else ''
            records[name] = (value, unit)
    return records


def _get(rec: dict, name: str) -> Optional[str]:
    """Return value+unit concatenated, or None if field absent."""
    if name not in rec:
        return None
    v, u = rec[name]
    return (v + u).strip() or None


def _num(rec: dict, name: str, decimals: int = None) -> Optional[str]:
    """
    Return numeric value with German decimal comma, unit stripped.
    Optionally round to `decimals` places.
    """
    if name not in rec:
        return None
    v, _u = rec[name]
    v = v.strip()
    if not v:
        return None
    try:
        f = float(v)
        if decimals is not None:
            f = round(f, decimals)
            v = f"{f:.{decimals}f}"
        else:
            # Strip trailing .0 only if it was an integer
            v = str(int(f)) if f == int(f) else v
    except ValueError:
        pass
    return v.replace('.', ',')


def _translate(rec: dict, name: str, table: dict = None) -> Optional[str]:
    """Return value translated through table (default: English→German)."""
    if name not in rec:
        return None
    v = rec[name][0].strip()
    return (table or _EN_DE).get(v, v) or None


def _date_us_to_de(date_str: str) -> Optional[str]:
    """Convert 'MM/DD/YYYY ...' → 'D Mon YYYY' German abbreviated month."""
    _DE_MONTHS = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str.strip())
    if not m:
        return None
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{day} {_DE_MONTHS[month - 1]} {year}"


def extract(log_path: str | Path) -> dict:
    """
    Extract structured parameters from an Abbott Merlin .log file.

    Returns a dict compatible with the FastPath PM/ICD/CRT-D VQA schema, plus
    additional fields available in the log but not on the FastPath page.
    """
    rec = _parse(log_path)
    device_class = _detect_device_class(rec)

    # ── session / device identity ─────────────────────────────────────────────
    session_ts = rec.get("Session Timestamp", ("", ""))[0]
    result: dict = {
        "source_file":    str(log_path),
        "manufacturer":   "Abbott",
        "device_class":   device_class,
        "report_type":    "FastPathSummary",
        "extraction_method": "log",
        "followup_date":  _date_us_to_de(session_ts) if session_ts else None,
        "device_model":   _get(rec, "Device Model Number"),
        "device_serial":  _get(rec, "Device Serial Number"),

        # ── battery ───────────────────────────────────────────────────────────
        # Raw voltage rounded to 2 dp; ICD/CRT-D logs omit this field
        "battery_voltage_v":      _num(rec, "Unloaded Battery Voltage", decimals=2),
        # Magnet rate in ppm; ICDs use "Normal"/"Battery Test" text, not a numeric rate
        "battery_mag_freq_bpm":   _num(rec, "Magnet Rate", decimals=1),
        # Remaining capacity is a computed display value — not stored in log
        "battery_remaining_pct":  None,

        # ── parameters ────────────────────────────────────────────────────────
        "mode":          _get(rec, "Mode"),
        "base_rate_bpm": _num(rec, "Base Rate"),
        # PM: "Maximum Tracking Rate"; CRT-D/ICD: "Maximum Pacing Rate"
        "upper_rate_bpm": (_num(rec, "Maximum Tracking Rate")
                           or _num(rec, "Maximum Pacing Rate")),
        "av_paced_ms":   _num(rec, "Paced AV Delay"),
        "av_sensed_ms":  _num(rec, "Sensed AV Delay"),

        # ── stimulation & sensing ─────────────────────────────────────────────
        "autocapture_a": _translate(rec, "Atrial AutoThreshold"),   # Monitor/Aus/Ein
        "autocapture_v": _translate(rec, "RV AutoCapture"),
        "amplitude_a_v":      _num(rec, "Atrial Pulse Amplitude"),
        "amplitude_rv_v":     _num(rec, "RV Pulse Amplitude"),
        "pulse_width_a_ms":   _num(rec, "Atrial Pulse Width"),
        "pulse_width_rv_ms":  _num(rec, "RV Pulse Width"),
        # SenseAbility: "Fixed"/"Automatic" AutoSense = Aus; "Auto"/"On" = Ein
        "sense_ability_a": _translate(rec, "Atrial AutoSense",    _AUTOSENSE),
        "sense_ability_v": _translate(rec, "Ventricular AutoSense", _AUTOSENSE),
        # PM uses "Ventricular Sensitivity"; ICD/CRT-D uses "Ventricular Pacemaker Max Sensitivity"
        "sensitivity_a_mv": (_num(rec, "Atrial Sensitivity", decimals=1)
                             or _num(rec, "Atrial Max Sensitivity", decimals=1)),
        "sensitivity_rv_mv": (_num(rec, "Ventricular Sensitivity", decimals=1)
                              or _num(rec, "Ventricular Pacemaker Max Sensitivity", decimals=1)),

        # ── diagnostics ───────────────────────────────────────────────────────
        "stim_a_pct": _num(rec, "Event Histogram Percent Paced In Atrium"),
        "stim_v_pct": _num(rec, "Event Histogram Percent Paced In Ventricle"),
        "ams_episodes": _num(rec, "Event Histogram Auto Mode Switch Count"),
        # mode_switch_pct / af_burden_pct: stored as episode counts/time, not %.
        # If total AT/AF time in recent week = 0, burden = 0.
        "mode_switch_pct": "0" if rec.get("Total Number of AT/AF Episodes Since Last Cleared", ("1",))[0].strip() == "0" else None,
        "af_burden_pct":   "0" if rec.get("Total Time in AT/AF Recent Week", ("1",))[0].strip() == "0" else None,
        "diagnostics_since": _date_us_to_de(
            rec.get("AT/AF Last Cleared Date", ("",))[0]
        ),

        # ── additional fields not on FastPath page ────────────────────────────
        "patient_name":   _get(rec, "Patient Name"),
        "patient_dob":    _date_us_to_de(
            rec.get("Patient Date of Birth", ("",))[0]
        ),
        "implant_date_device": _date_us_to_de(
            rec.get("Implant Date: Device", ("",))[0]
        ),
        "atrial_lead_model":  _get(rec, "Model Number: SJM Atrial Lead"),
        "atrial_lead_serial": _get(rec, "Atrial Lead Serial Number"),
        # PM has a pace/sense RV lead; ICD/CRT-D has a defibrillation lead
        "rv_lead_model":  (_get(rec, "Model Number: SJM RV Pace/Sense Lead")
                           or _get(rec, "Model Number: SJM RV Defib Lead")),
        "rv_lead_serial": _get(rec, "RV Lead Serial Number"),
        # Raw measured impedances (more precise than FastPath rounded display)
        "impedance_a_ohm":  _num(rec, "Atrial Pacing Lead Impedance"),
        "impedance_rv_ohm": _num(rec, "RV Pacing Lead Impedance"),
        # Capture test results — use adapter-compatible threshold_* naming
        "threshold_a_v":       _num(rec, "A. Capture Test Threshold Amplitude"),
        "threshold_rv_v":      _num(rec, "RV. Capture Test Threshold Amplitude"),
        "threshold_a_pw_ms":   _num(rec, "A. Capture Test Pulse Width"),
        "threshold_rv_pw_ms":  _num(rec, "RV. Capture Test Pulse Width"),
        # Measured sensing amplitudes
        "sensing_a_mv":        _num(rec, "Atrial Signal Amplitude", decimals=1),
        "sensing_rv_mv":       _num(rec, "Ventricular Signal Amplitude", decimals=1),
        "longevity_estimate":     _get(rec, "Longevity Estimate"),
        "battery_current_ua":     _num(rec, "Battery Current"),
    }

    # ── ICD / CRT-D specific ──────────────────────────────────────────────────
    if device_class in ("ICD", "CRT-D"):
        result.update({
            # Detection zone intervals (ms); bpm = 60000 / interval
            "zone_vf_interval_ms":  _num(rec, "VF Detection Interval"),
            "zone_vt2_interval_ms": _num(rec, "VT-2 Detection Interval"),
            "zone_vt1_interval_ms": _num(rec, "VT-1 Detection Interval"),
            # Zone configuration string
            "zone_config": _get(rec, "Zone Configuration"),
            # First VF shock energy (escalation is VF Energy 1 → 2 → 3)
            "vf_energy_1_j": _num(rec, "VF Energy 1"),
            # ATP pacing output (used inside all tach zones)
            "atp_pulse_amplitude_v": _num(rec, "ATP Pulse Amplitude"),
            "atp_pulse_width_ms":    _num(rec, "ATP Pulse Width", decimals=1),
            # Shock vector
            "shock_config": _get(rec, "Shock Configuration"),
        })

    if device_class in ("ICD", "CRT-D"):
        result["shock_impedance_ohm"] = _num(rec, "HV Lead Impedance")

    if device_class == "CRT-D":
        result.update({
            "lv_lead_model":       _get(rec, "Model Number: SJM LV Lead"),
            "lv_lead_serial":      _get(rec, "LV Lead Serial Number"),
            "impedance_lv_ohm":    _num(rec, "LV Pacing Lead Impedance"),
            "threshold_lv_v":      _num(rec, "LV. Capture Test Threshold Amplitude"),
            "threshold_lv_pw_ms":  _num(rec, "LV. Capture Test Pulse Width"),
            "amplitude_lv_v":      _num(rec, "LV Pulse Amplitude"),
            "pulse_width_lv_ms":   _num(rec, "LV Pulse Width"),
        })

    return result
