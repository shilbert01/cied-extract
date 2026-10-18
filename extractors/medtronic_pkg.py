"""
Medtronic .pkg / PublicDiscreteData.xml extractor.

Handles modern Medtronic exports (.pkg ZIP archives containing
PublicDiscreteData.xml) and standalone PublicDiscreteData.xml files.

The .pkg archive is a ZIP with:
  - Public/PublicDiscreteData.xml  — structured device/patient/measurement data
  - Reports/*.pdf                  — optional PDF report (ignored here)

The XML uses a nested Composite/Field/Array structure with named parameters.

Usage:
    from extractors.medtronic_pkg import extract, extract_xml
    result = extract("session.pkg")          # .pkg ZIP
    result = extract_xml(xml_string)         # standalone XML
"""
import re
import zipfile
import tempfile
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET


def _norm_date(raw: Optional[str]) -> Optional[str]:
    """Normalize various date formats to DD.MM.YYYY."""
    if not raw:
        return None
    # ISO: 2026-07-09 or 2026-07-09T14:30:00
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', raw)
    if m:
        return f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
    # YYYYMMDD or YYYYMMDDHHMMSS
    m = re.match(r'(\d{4})(\d{2})(\d{2})', raw)
    if m:
        return f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
    return raw


# ── XML traversal helpers ────────────────────────────────────────────────────

def _find_field(parent: ET.Element, name: str) -> Optional[ET.Element]:
    """Find a <Field name="..."> child with the given name."""
    for f in parent.findall("Field"):
        if f.get("name") == name:
            return f
    return None


def _get_text(el: Optional[ET.Element]) -> Optional[str]:
    """Get text content from a value element (String, Integer, Real, etc.)."""
    if el is None:
        return None
    # Direct text content
    if el.text and el.text.strip():
        return el.text.strip()
    return None


def _field_value(field: ET.Element) -> Optional[str]:
    """Extract the scalar value from a Field element."""
    for tag in ("String", "Integer", "Real", "Discrete", "Date", "DateTime"):
        child = field.find(tag)
        if child is not None:
            t = _get_text(child)
            if t:
                return t
    return None


def _find_composite(field: ET.Element) -> Optional[ET.Element]:
    """Return the <Composite> child of a Field, if any."""
    return field.find("Composite")


def _find_value_in_composite(composite: ET.Element, target_name: str) -> Optional[str]:
    """Find a named Field inside a Composite and return its scalar value or Composite."""
    field = _find_field(composite, target_name)
    if field is None:
        return None
    # Try scalar value first
    val = _field_value(field)
    if val:
        return val
    # If it has a Composite child, try to get a value from within
    comp = _find_composite(field)
    if comp is not None:
        return comp  # Return the element for further traversal
    return None


def _find_composite_in_composite(composite: ET.Element, target_name: str) -> Optional[ET.Element]:
    """Find a named Field inside a Composite and return its Composite child."""
    field = _find_field(composite, target_name)
    if field is None:
        return None
    return _find_composite(field)


def _get_composites_from_array(field: ET.Element) -> list[ET.Element]:
    """Get all Composite elements from a Field's Array child."""
    arr = field.find("Array") if field.tag == "Field" else field.find("Array")
    if arr is None:
        # The field itself might be a Composite containing an Array
        comp = field.find("Composite")
        if comp is not None:
            arr = comp.find("Array")
    if arr is None:
        return []
    return arr.findall("Composite")


# ── Device type inference ────────────────────────────────────────────────────

_DEVICE_TYPE_MAP = {
    "CRT_D": "CRT-D",
    "CRT_P": "CRT-P",
    "IPG": "Pacemaker",
}

_FAMILY_TYPE = [
    ("REVEAL", "ICM"), ("LINQ", "ICM"),
    ("MICRA", "Leadless Pacemaker"),
    ("AMPLIA", "CRT-D"),
    ("PROTECTA", "ICD"), ("VISIA", "ICD"), ("EVERA", "ICD"),
]


def _infer_device_type(model: str) -> str:
    upper = model.upper()
    for keyword, dtype in _FAMILY_TYPE:
        if keyword in upper:
            return dtype
    if "CRT-D" in upper:
        return "CRT-D"
    if "CRT-P" in upper:
        return "CRT-P"
    if "CRT" in upper:
        return "CRT"
    if "ICD" in upper:
        return "ICD"
    return "Pacemaker"


# ── Parameter extraction ─────────────────────────────────────────────────────

def _extract_params(root: ET.Element) -> list[ET.Element]:
    """Navigate to the NormalizedParameterCollection composites."""
    value_field = _find_field(root, "Value")
    if value_field is None:
        return []

    # Value → Composite → ContextCollection
    value_comp = _find_composite(value_field)
    if value_comp is None:
        return []

    ctx_field = _find_field(value_comp, "ContextCollection")
    if ctx_field is None:
        return []

    contexts = _get_composites_from_array(ctx_field)

    def params_from_context(ctx: ET.Element) -> list[ET.Element]:
        param_field = _find_field(ctx, "NormalizedParameterCollection")
        if param_field is None:
            return []
        return _get_composites_from_array(param_field)

    # Prefer "NoPendingSettings" context
    for ctx in contexts:
        name_field = _find_field(ctx, "Name")
        if name_field is not None:
            name_str = name_field.find("String")
            if name_str is not None and _get_text(name_str) == "NoPendingSettings":
                params = params_from_context(ctx)
                if params:
                    return params

    # Fallback: first context with parameters
    for ctx in contexts:
        params = params_from_context(ctx)
        if params:
            return params

    return []


def _find_param(params: list[ET.Element], name: str) -> Optional[ET.Element]:
    """Find a parameter composite by its Name field."""
    for p in params:
        name_field = _find_field(p, "Name")
        if name_field is not None:
            name_str = name_field.find("String")
            if name_str is not None and _get_text(name_str) == name:
                return p
    return None


def _param_current_value(params: list[ET.Element], name: str) -> Optional[str]:
    """Get the scalar Current value for a named parameter."""
    p = _find_param(params, name)
    if p is None:
        return None
    val = _find_value_in_composite(p, "Current")
    if isinstance(val, str):
        return val
    if isinstance(val, ET.Element):
        # Try Name sub-field (used for DeviceModelName)
        name_sub = _find_field(val, "Name")
        if name_sub is not None:
            s = name_sub.find("String")
            if s is not None:
                return _get_text(s)
    return None


def _param_current_composite(params: list[ET.Element], name: str) -> Optional[ET.Element]:
    """Get the Current value as a Composite element for nested parameters."""
    p = _find_param(params, name)
    if p is None:
        return None
    current_field = _find_field(p, "Current")
    if current_field is None:
        return None
    # Current might contain a scalar or a Composite
    comp = _find_composite(current_field)
    if comp is not None:
        return comp
    return None


def _get_numeric(params: list[ET.Element], param_name: str) -> Optional[str]:
    """Get a numeric parameter value as string."""
    val = _param_current_value(params, param_name)
    if val is None:
        return None
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return None


def _get_status_subfield(params: list[ET.Element], param_name: str,
                          sub_field: str) -> Optional[str]:
    """Get a numeric sub-field from a Status-type parameter's Current composite."""
    comp = _param_current_composite(params, param_name)
    if comp is None:
        return None
    val = _find_value_in_composite(comp, sub_field)
    if isinstance(val, str):
        try:
            f = float(val)
            if f == int(f):
                return str(int(f))
            return str(f)
        except (ValueError, TypeError):
            return None
    return None


# ── Main XML parser ──────────────────────────────────────────────────────────

def extract_xml(xml_content: str, source_file: str = "") -> dict:
    """Parse PublicDiscreteData.xml content and return a flat extraction dict."""
    root = ET.fromstring(xml_content)
    params = _extract_params(root)

    result: dict = {
        "source_file": source_file,
        "manufacturer": "Medtronic",
        "report_type": "DeviceFollowUpReport",
    }

    # Interrogation date (SavedDateTime at root level)
    saved_field = _find_field(root, "SavedDateTime")
    if saved_field is not None:
        dt = saved_field.find("DateTime")
        if dt is not None:
            result["followup_date"] = _norm_date(_get_text(dt))

    # Device
    device_model = _param_current_value(params, "DeviceModelName") or ""
    device_serial = _param_current_value(params, "DeviceSerialNumber") or ""
    raw_device_type = _param_current_value(params, "DeviceType") or ""

    device_type = _DEVICE_TYPE_MAP.get(raw_device_type, raw_device_type)
    if device_type in ("Unknown", "") and device_model:
        device_type = _infer_device_type(device_model)
    if re.search(r"MICRA", device_model, re.IGNORECASE):
        device_type = "Leadless Pacemaker"

    if device_model:
        result["device_model"] = device_model
    if device_serial:
        result["device_serial"] = device_serial
    if device_type:
        result["device_type"] = device_type

    # Implant date
    status_comp = _param_current_composite(params, "DeviceStatus")
    if status_comp is not None:
        imp_date = _find_value_in_composite(status_comp, "ImplantDateTime")
        if isinstance(imp_date, str):
            result["implant_date"] = _norm_date(imp_date)

    # Patient
    patient_name = _param_current_value(params, "PatientName")
    if patient_name:
        if "," in patient_name:
            parts = patient_name.split(",", 1)
            result["patient_name"] = f"{parts[0].strip()}, {parts[1].strip()}"
        elif " " in patient_name:
            parts = patient_name.split(None, 1)
            result["patient_name"] = f"{parts[0]}, {parts[1]}" if len(parts) > 1 else parts[0]
        else:
            result["patient_name"] = patient_name

    patient_dob = _param_current_value(params, "PatientBirthDate")
    if patient_dob:
        result["patient_dob"] = _norm_date(patient_dob)

    # Battery
    battery_comp = _param_current_composite(params, "BatteryStatus")
    if battery_comp is not None:
        voltage_status = _find_composite_in_composite(battery_comp, "VoltageStatus")
        if voltage_status is not None:
            voltage_field = _find_field(voltage_status, "Voltage")
            if voltage_field is not None:
                real = voltage_field.find("Real")
                if real is not None:
                    txt = _get_text(real)
                    if txt:
                        try:
                            result["battery_voltage_v"] = str(float(txt))
                        except ValueError:
                            pass

    longevity_comp = _param_current_composite(params, "DeviceLongevityStatus")
    if longevity_comp is not None:
        years = _find_value_in_composite(longevity_comp, "UserRepresentationAverageRemainingDuration")
        if isinstance(years, str):
            try:
                result["battery_longevity_years"] = str(float(years))
            except ValueError:
                pass
        else:
            months = _find_value_in_composite(longevity_comp, "IDCORepresentationAverageRemainingDuration")
            if isinstance(months, str):
                try:
                    result["battery_longevity_months"] = str(float(months))
                except ValueError:
                    pass

    # Leads
    for i in range(1, 5):
        loc = _param_current_value(params, f"Lead{i}Location")
        model = _param_current_value(params, f"Lead{i}Model")
        serial = _param_current_value(params, f"Lead{i}SerialNumber")
        mfg = _param_current_value(params, f"Lead{i}Manufacturer")
        imp_date = _param_current_value(params, f"ImplantLead{i}Date")

        if not loc and not model:
            continue

        loc_upper = (loc or "").upper()
        if "RV" in loc_upper or "RIGHT VENTRICLE" in loc_upper:
            chamber = "rv"
        elif loc_upper in ("RA", "A") or "ATRIUM" in loc_upper:
            chamber = "a"
        elif "LV" in loc_upper or "LEFT VENTRICLE" in loc_upper:
            chamber = "lv"
        else:
            chamber = loc_upper.lower() if loc else f"lead{i}"

        prefix = f"lead_{chamber}"
        if loc:
            result[f"{prefix}_location"] = loc
        if model:
            result[f"{prefix}_model"] = model
        if serial:
            result[f"{prefix}_serial"] = serial
        if mfg:
            result[f"{prefix}_manufacturer"] = mfg
        if imp_date:
            result[f"{prefix}_implant_date"] = _norm_date(imp_date)

        # Electrical measurements by chamber
        if chamber == "rv":
            sense = _get_numeric(params, "VSEventDetectionRVSensingThreshold")
            if sense:
                result["sensing_rv_mv"] = sense
            amp = _get_numeric(params, "VPacingTherapyRVPacingAmplitude")
            if amp:
                result["amplitude_rv_v"] = amp
            pw = _get_numeric(params, "VPacingTherapyRVPacingPulseWidth")
            if pw:
                result["pulse_width_rv_ms"] = pw
            thresh = _get_status_subfield(params, "VPacingTherapyAdaptRVPacingAmplitudeStatus", "PacingThreshold")
            if thresh:
                result["threshold_rv_v"] = thresh
        elif chamber == "a":
            sense = _get_numeric(params, "ASEventDetectionRASensingThreshold")
            if sense:
                result["sensing_a_mv"] = sense
            amp = _get_numeric(params, "APacingTherapyRAPacingAmplitude")
            if amp:
                result["amplitude_a_v"] = amp
            thresh = _get_status_subfield(params, "APacingTherapyAdaptRAPacingAmplitudeStatus", "PacingThreshold")
            if thresh:
                result["threshold_a_v"] = thresh
        elif chamber == "lv":
            amp = _get_numeric(params, "VPacingTherapyLVPacingPathwayAAmplitude")
            if amp:
                result["amplitude_lv_v"] = amp
            thresh = _get_status_subfield(params, "VPacingTherapyAdaptLVPacingAmplitudeStatus", "PacingThreshold")
            if thresh:
                result["threshold_lv_v"] = thresh

    # Leadless pacemaker: no Lead1-4 but still has RV parameters
    has_leads = any(k.startswith("lead_") for k in result)
    if not has_leads and (device_type == "Leadless Pacemaker" or re.search(r"MICRA", device_model, re.IGNORECASE)):
        sense = _get_numeric(params, "VSEventDetectionRVSensingThreshold")
        if sense:
            result["sensing_rv_mv"] = sense
        amp = _get_numeric(params, "VPacingTherapyRVPacingAmplitude")
        if amp:
            result["amplitude_rv_v"] = amp
        thresh = _get_status_subfield(params, "VPacingTherapyAdaptRVPacingAmplitudeStatus", "PacingThreshold")
        if thresh:
            result["threshold_rv_v"] = thresh

    return {k: v for k, v in result.items() if v is not None}


# ── .pkg ZIP handler ─────────────────────────────────────────────────────────

def extract(file_path: str | Path) -> dict:
    """Extract from a Medtronic .pkg archive or standalone XML."""
    file_path = Path(file_path)

    # If it's an XML file directly, parse it
    if file_path.suffix.lower() == ".xml":
        xml_content = file_path.read_text(encoding="utf-8", errors="replace")
        return extract_xml(xml_content, str(file_path))

    # .pkg is a ZIP archive
    with zipfile.ZipFile(str(file_path), "r") as zf:
        # Find PublicDiscreteData.xml
        xml_name = None
        for name in zf.namelist():
            if name.endswith("PublicDiscreteData.xml"):
                xml_name = name
                break

        if xml_name is None:
            return {
                "source_file": str(file_path),
                "manufacturer": "Medtronic",
                "error": "No PublicDiscreteData.xml found in .pkg archive",
            }

        xml_content = zf.read(xml_name).decode("utf-8", errors="replace")
        return extract_xml(xml_content, str(file_path))
