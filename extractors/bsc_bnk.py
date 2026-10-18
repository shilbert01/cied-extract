"""
Boston Scientific .bnk (PACEART key/value dump) extractor.

The .bnk format is a plain-text export from Boston Scientific's PACEART system.
Lines starting with '#' are header comments containing the device model/serial
and save date. Data lines are comma-separated key,value pairs. The sentinel
'K.A' (keine Angabe) means "no data".

Ported from KardiSynch's boston-scientific-parser.ts (parseBostonScientificBnk).
"""
import re
from pathlib import Path
from typing import Optional


NO_DATA = "K.A"

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mär": "03", "mar": "03", "m?r": "03",
    "apr": "04", "mai": "05", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "okt": "10", "oct": "10",
    "nov": "11", "dez": "12", "dec": "12",
}


def _has_value(v: Optional[str]) -> bool:
    return bool(v) and v != NO_DATA


def _extract_numeric(s: Optional[str]) -> Optional[str]:
    """Extract the first number from a string like '500.0 Ω', '0.4 ms', '>132 months'."""
    if not _has_value(s):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return m.group(1) if m else None


def _normalize_month(token: str) -> Optional[str]:
    return _MONTH_MAP.get(token.lower())


def _build_parts_date(data: dict, day_key: Optional[str], month_key: str, year_key: str) -> Optional[str]:
    """Build DD.MM.YYYY from separate day/month/year keys. Day is optional (defaults to 01)."""
    month_raw = data.get(month_key)
    year = data.get(year_key)
    if not _has_value(month_raw) or not _has_value(year):
        return None
    month = _normalize_month(month_raw) or month_raw
    day = "01"
    if day_key:
        day_raw = data.get(day_key)
        if _has_value(day_raw):
            day = day_raw.zfill(2)
    return f"{day}.{month}.{year}"


def _parse_header(text: str) -> dict:
    """Extract interrogation date, device model and serial from # comment header."""
    out: dict = {}

    m = re.search(
        r"^#\s*TYPE:.*?SAVE DATE:\s*(\d{1,2})\s+([A-Za-zÄäÖöÜü?]+)\s+(\d{4})",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        month_tok = m.group(2)
        if month_tok == "M?r":
            month_tok = "Mär"
        month = _normalize_month(month_tok)
        if month:
            out["interrogation_date"] = f"{m.group(1).zfill(2)}.{month}.{m.group(3)}"

    m = re.search(
        r"^#\s*DEVICE\s+MODEL:\s*(\S+)\s+SERIAL:\s*(\S+)",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        out["device_model"] = m.group(1)
        out["device_serial"] = m.group(2)

    return out


def _chamber_from_position(position: Optional[str]) -> Optional[str]:
    """Infer clinical chamber from the Position text, not from the slot key."""
    if not position:
        return None
    if re.search(r"\bLV\b", position, re.IGNORECASE):
        return "LV"
    if re.search(r"vorhof|atrial|atrium", position, re.IGNORECASE):
        return "RA"
    if re.search(r"ventrikel|ventric", position, re.IGNORECASE):
        return "RV"
    return None


def _infer_device_type(model: str, data: dict, leads: list) -> str:
    model_up = model.upper()
    if "CRT-D" in model_up:
        return "CRT-D"
    if "CRT-P" in model_up:
        return "CRT-P"
    if "S-ICD" in model_up or "EMBLEM" in model_up or "SQ-RX" in model_up:
        return "S-ICD"
    if "ICD" in model_up or "DYNAGEN" in model_up or "ORIGEN" in model_up or "AUTOGEN" in model_up:
        return "ICD"
    if "ACCOLADE" in model_up or "FORMIO" in model_up or "PROPONENT" in model_up:
        return "Pacemaker"

    has_icd = _has_value(data.get("PatientDFT")) or _has_value(data.get("PatientShockImped"))
    has_lv = any(l.get("chamber") == "LV" for l in leads)
    if has_lv:
        return "CRT-D" if has_icd else "CRT-P"
    if has_icd:
        return "ICD"
    return "Pacemaker"


def extract(file_path: str | Path) -> dict:
    """Extract from a Boston Scientific .bnk (PACEART) file."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="replace")

    result: dict = {
        "source_file": str(file_path),
        "manufacturer": "Boston Scientific",
        "report_type": "PACEART",
    }

    # Parse key,value lines into a map
    data: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        idx = line.find(",")
        if idx == -1:
            continue
        key = line[:idx].strip()
        value = line[idx + 1:].strip()
        data[key] = value

    # Header (device identity + interrogation date)
    header = _parse_header(text)
    result["interrogation_date"] = header.get("interrogation_date")
    result["device_model"] = header.get("device_model")
    result["device_serial"] = header.get("device_serial")

    # Patient
    last_name = data.get("PatientLastName", "")
    first_name = data.get("PatientFirstName", "")
    if _has_value(last_name):
        if _has_value(first_name):
            result["patient_name"] = f"{last_name}, {first_name}"
        else:
            result["patient_name"] = last_name

    result["patient_dob"] = _build_parts_date(
        data, "PatientBirthDay", "PatientBirthMonth", "PatientBirthYear"
    )

    # Device implant date
    result["implant_date"] = _build_parts_date(
        data, "PatientData.ImplantDay", "PatientData.ImplantMonth", "PatientData.ImplantYear"
    )

    # Battery
    time_to_eri = _extract_numeric(data.get("BatteryLongevityParams.TimeToERI"))
    if time_to_eri:
        result["battery_longevity_months"] = time_to_eri
    battery_pct = _extract_numeric(data.get("BatteryLongevityParams.RemainingBatteryPercent"))
    if battery_pct:
        result["battery_remaining_pct"] = battery_pct
    battery_phase = data.get("BatteryStatus.BatteryPhase")
    if _has_value(battery_phase):
        result["battery_status"] = battery_phase

    # Leads
    lead_info: list[dict] = []

    # Atrial lead (slot A)
    def _read_lead_slot(slot_key: str, default_chamber: str) -> Optional[dict]:
        manufacturer = data.get(f"PatientLead{slot_key}Manufacturer")
        model = data.get(f"PatientLead{slot_key}ModelNum")
        serial = data.get(f"PatientLead{slot_key}SerialNum")
        position = data.get(f"PatientLead{slot_key}Position")
        if not _has_value(model) and not _has_value(serial) and not _has_value(position):
            return None
        chamber = _chamber_from_position(position) or default_chamber
        return {
            "chamber": chamber,
            "manufacturer": manufacturer if _has_value(manufacturer) else None,
            "model": model if _has_value(model) else None,
            "serial": serial if _has_value(serial) else None,
            "position": position if _has_value(position) else None,
        }

    atrial = _read_lead_slot("A", "RA")
    if atrial:
        atrial["impedance_ohm"] = _extract_numeric(data.get("PatientAtrialImped"))
        thresh_mv = _extract_numeric(data.get("PatientAtrialThreshAmpl"))
        if thresh_mv:
            atrial["threshold_v"] = str(float(thresh_mv) / 1000)
        atrial["threshold_pw_ms"] = _extract_numeric(data.get("PatientAtrialThreshPW"))
        atrial["implant_date"] = _build_parts_date(
            data, None, "PatientData.LeadA.ImplantMonth", "PatientData.LeadA.ImplantYear"
        )
        lead_info.append(atrial)

    # Ventricular leads (slots V1-V5)
    generic_v_attached = False
    for i in range(1, 6):
        slot = _read_lead_slot(f"V{i}", "RV")
        if not slot:
            continue
        is_lv = slot["chamber"] == "LV"

        if is_lv:
            slot["impedance_ohm"] = _extract_numeric(data.get("PatientData.LVMsmts.LeadImped"))
            thresh_mv = _extract_numeric(data.get("PatientData.LVMsmts.PaceThreshAmpl"))
            if thresh_mv:
                slot["threshold_v"] = str(float(thresh_mv) / 1000)
            slot["threshold_pw_ms"] = _extract_numeric(data.get("PatientData.LVMsmts.PaceThreshPW"))
        elif not generic_v_attached:
            generic_v_attached = True
            slot["impedance_ohm"] = _extract_numeric(data.get("PatientVImped"))
            thresh_mv = _extract_numeric(data.get("PatientVThreshAmpl"))
            if thresh_mv:
                slot["threshold_v"] = str(float(thresh_mv) / 1000)
            slot["threshold_pw_ms"] = _extract_numeric(data.get("PatientVThreshPW"))
            shock_imp = _extract_numeric(data.get("PatientShockImped"))
            if shock_imp:
                slot["shock_impedance_ohm"] = shock_imp

        slot["implant_date"] = _build_parts_date(
            data, None, f"PatientData.Lead{i}.ImplantMonth", f"PatientData.Lead{i}.ImplantYear"
        )
        lead_info.append(slot)

    # Device type (inferred)
    result["device_type"] = _infer_device_type(
        result.get("device_model") or "", data, lead_info
    )

    # Flatten lead data into result dict
    for lead in lead_info:
        ch = lead["chamber"].lower()
        prefix = f"lead_{ch}"
        if lead.get("model"):
            result[f"{prefix}_model"] = lead["model"]
        if lead.get("serial"):
            result[f"{prefix}_serial"] = lead["serial"]
        if lead.get("position"):
            result[f"{prefix}_position"] = lead["position"]
        if lead.get("impedance_ohm"):
            result[f"impedance_{ch}_ohm"] = lead["impedance_ohm"]
        if lead.get("threshold_v"):
            result[f"threshold_{ch}_v"] = lead["threshold_v"]
        if lead.get("threshold_pw_ms"):
            result[f"threshold_{ch}_pw_ms"] = lead["threshold_pw_ms"]
        if lead.get("shock_impedance_ohm"):
            result[f"shock_impedance_{ch}_ohm"] = lead["shock_impedance_ohm"]
        if lead.get("implant_date"):
            result[f"{prefix}_implant_date"] = lead["implant_date"]

    # Additional fields
    ef = data.get("PatientLeftVentEjectFraction")
    if _has_value(ef):
        result["ejection_fraction"] = ef
    nyha = data.get("PatientFuncHeartClass")
    if _has_value(nyha):
        result["nyha_class"] = nyha

    return {k: v for k, v in result.items() if v is not None}
