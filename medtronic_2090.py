"""
Medtronic 2090 programmer Device Data Archive extractor.

Handles .26X and .V26 files exported by the Medtronic 2090 / Empowered programmer.
These files are ASCII text combining raw hex parameter memory dumps (skipped) with
structured "key","value" context blocks that contain all clinically relevant data.

Covers device families: Attesta, Sphera, Adapta, Sensia, Versa, Relia, EnPulse,
Kappa, InSync III, and Vitatron devices programmed with the 2090 programmer.

Usage:
    from extractors.medtronic_2090 import extract
    result = extract("session.26X")
"""
import re
from pathlib import Path
from typing import Optional


# Matches: "Some Key", "some value"
_KV = re.compile(r'^"([^"]+)",\s*"([^"]*)"$')
# Matches: // Family Id 0x18  Model Id 0x1
_FAMILY = re.compile(r'//\s*Family Id\s+(0x[0-9a-fA-F]+)\s+Model Id\s+(0x[0-9a-fA-F]+)')


def _parse_kvs(text: str) -> dict[str, str]:
    """
    Extract all "key","value" pairs from a .26X/.V26 file.
    Duplicate keys keep the FIRST occurrence (header context values are most reliable).
    """
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        m = _KV.match(line)
        if m:
            k, v = m.group(1), m.group(2)
            if k not in result:
                result[k] = v
    return result


def _first(kv: dict, *keys: str) -> Optional[str]:
    """Return first non-empty value found among the given keys."""
    for k in keys:
        v = kv.get(k, "").strip()
        if v:
            return v
    return None


def _num(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    try:
        return float(val.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _parse_date_key_time(val: str) -> str:
    """
    Normalise KEY_Time format to ISO-ish.
    Formats seen: "07/09/2026 20:18:21"  → "2026-07-09"
    """
    # MM/DD/YYYY HH:MM:SS
    m = re.match(r'(\d{2})/(\d{2})/(\d{4})', val)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return val.split()[0]


def _parse_date_german(val: str) -> str:
    """
    German date format: "09.07.26 20:18:23" or "20.07.25 14:48:22"
    Two-digit year → assume 2000+.
    """
    m = re.match(r'(\d{2})\.(\d{2})\.(\d{2,4})', val)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        if len(year) == 2:
            year = "20" + year
        return f"{year}-{month}-{day}"
    return val.split()[0]


def extract(file_path: str | Path) -> dict:
    """
    Parse a Medtronic 2090 .26X or .V26 file and return a structured dict.

    Returned keys:
        report_type, manufacturer, followup_date, implant_date
        device_model, device_serial, device_family_id, device_model_id
        mode, base_rate_bpm, upper_rate_bpm, av_paced_ms, av_sensed_ms
        amplitude_a_v, pulse_width_a_ms, sensitivity_a_mv
        amplitude_v_v, pulse_width_v_ms, sensitivity_v_mv
        amplitude_lv_v, pulse_width_lv_ms         (CRT only)
        impedance_a_ohm, impedance_rv_ohm, impedance_lv_ohm
        threshold_v_v, threshold_v_pw_ms, threshold_date
        battery_voltage_v, battery_longevity_months, battery_current_ua
        eri_active
        source_file
    """
    file_path = Path(file_path)
    text = file_path.read_text(encoding="latin-1", errors="replace")

    # Extract Family/Model Id from first occurrence
    family_id = model_id = None
    m = _FAMILY.search(text)
    if m:
        family_id = m.group(1)
        model_id = m.group(2)

    kv = _parse_kvs(text)

    # ── device identity ────────────────────────────────────────────────────────
    raw_model = _first(kv, "KEY_Model") or ""
    device_model = raw_model.strip()
    device_serial = _first(kv, "KEY_SerialNum", "Seriennummer")

    # ── dates ──────────────────────────────────────────────────────────────────
    raw_followup = _first(kv, "KEY_Time")
    followup_date = _parse_date_key_time(raw_followup) if raw_followup else None

    raw_implant = _first(kv, "Implantationszeit")
    implant_date = _parse_date_german(raw_implant) if raw_implant else None

    # ── pacing parameters ─────────────────────────────────────────────────────
    mode = _first(kv, "Betriebsart")

    # Base rate: "Interv. freq." (space variant) or "Interv.freq." (no-space, InSync III)
    base_rate = _first(kv, "Interv. freq.", "Interv.freq.")

    upper_rate = _first(kv,
        "Maximale Synchronfrequenz", "Max. Sensorfreq.",
        "Maximale Sensorfrequenz", "Max. Synch.freq.",
    )

    av_paced = _first(kv, "Stimuliertes AV")
    av_sensed = _first(kv, "Wahrgenommenes AV")

    # Atrial stimulation
    amp_a = _first(kv, "A. Amplitude")
    pw_a  = _first(kv, "A. Impulsdauer")
    sens_a = _first(kv, "A. Empfindlichkeit")

    # Ventricular stimulation — dual-chamber uses "V." prefix, CRT-P uses "RV"
    amp_v  = _first(kv, "V. Amplitude", "RV Amplitude")
    pw_v   = _first(kv, "V. Impulsdauer", "RV Impulsdauer")
    sens_v = _first(kv, "V. Empfindlichkeit", "RV Empfindlichkeit")

    # LV stimulation (CRT-P/CRT-D only)
    amp_lv = _first(kv, "LV Amplitude")
    pw_lv  = _first(kv, "LV Impulsdauer")

    # ── lead impedances ────────────────────────────────────────────────────────
    imp_a  = _first(kv, "AleadImpedance")
    imp_rv = _first(kv, "VleadImpedance")
    imp_lv = _first(kv, "LVleadImpedance")

    # ── pacing threshold (last auto-measured) ──────────────────────────────────
    thresh_v  = _first(kv, "Zuletzt gemessene Rheobase")
    thresh_pw = _first(kv, "Zuletzt gemessene Chronaxie")
    raw_thresh_date = _first(kv, "Zuletzt gemessene PTS-Zeit")
    thresh_date = _parse_date_german(raw_thresh_date) if raw_thresh_date else None

    # ── battery ───────────────────────────────────────────────────────────────
    batt_v   = _first(kv, "batteryVoltage")
    batt_lon = _first(kv, "batteryLongevityAvg")       # months
    batt_cur = _first(kv, "instantBatteryCurrent")     # µA
    eri_raw  = _first(kv, "eriStatus", "eriOccurred")
    eri_active = eri_raw not in (None, "0", "") if eri_raw is not None else None

    result = {
        "report_type": "DeviceDataArchive",
        "manufacturer": "Medtronic",
        "source_file": str(file_path),
        "device_family_id": family_id,
        "device_model_id": model_id,
    }

    if device_model:    result["device_model"]   = device_model
    if device_serial:   result["device_serial"]  = device_serial
    if followup_date:   result["followup_date"]  = followup_date
    if implant_date:    result["implant_date"]   = implant_date
    if mode:            result["mode"]           = mode
    if base_rate:       result["base_rate_bpm"]  = base_rate
    if upper_rate:      result["upper_rate_bpm"] = upper_rate
    if av_paced:        result["av_paced_ms"]    = av_paced
    if av_sensed:       result["av_sensed_ms"]   = av_sensed
    if amp_a:           result["amplitude_a_v"]  = amp_a
    if pw_a:            result["pulse_width_a_ms"] = pw_a
    if sens_a:          result["sensitivity_a_mv"] = sens_a
    if amp_v:           result["amplitude_v_v"]  = amp_v
    if pw_v:            result["pulse_width_v_ms"] = pw_v
    if sens_v:          result["sensitivity_v_mv"] = sens_v
    if amp_lv:          result["amplitude_lv_v"] = amp_lv
    if pw_lv:           result["pulse_width_lv_ms"] = pw_lv
    if imp_a:           result["impedance_a_ohm"]  = _num(imp_a)
    if imp_rv:          result["impedance_rv_ohm"] = _num(imp_rv)
    if imp_lv:          result["impedance_lv_ohm"] = _num(imp_lv)
    if thresh_v:        result["threshold_v_v"]    = thresh_v
    if thresh_pw:       result["threshold_v_pw_ms"] = thresh_pw
    if thresh_date:     result["threshold_date"]   = thresh_date
    if batt_v:          result["battery_voltage_v"] = _num(batt_v)
    if batt_lon:        result["battery_longevity_months"] = _num(batt_lon)
    if batt_cur:        result["battery_current_ua"] = _num(batt_cur)
    if eri_active is not None:
        result["eri_active"] = eri_active

    return result
