"""
Microport / Paceart XML extractor.

Parses the multi-vendor Paceart remote-monitoring XML export format.
Root element is <Paceart> containing <PatientRecords> and <LookupTables>.
Device and lead details are resolved via GUID references into the lookup tables.

Paceart XML stores most values as XML attributes, not child-element text.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def _val(el: Optional[ET.Element], name: str) -> Optional[str]:
    """Return a named value from an element: try attribute first, then child text."""
    if el is None:
        return None
    # Attribute (most common in Paceart)
    v = el.get(name)
    if v is not None:
        return v.strip() if isinstance(v, str) else str(v)
    # Child element text
    child = el.find(name)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _as_list(el: Optional[ET.Element], tag: str) -> list[ET.Element]:
    if el is None:
        return []
    return el.findall(tag)


def _find_by_chamber(items: list[ET.Element], chamber: str) -> Optional[ET.Element]:
    for item in items:
        ch = _val(item, "Chamber")
        if ch == chamber:
            return item
    return None


def _lookup_by_guid(details: list[ET.Element], guid: Optional[str]) -> Optional[ET.Element]:
    if not guid:
        return None
    for d in details:
        if _val(d, "GUID") == guid:
            return d
    return None


def _infer_device_type(model: str) -> str:
    upper = model.upper()
    if "CRT-D" in upper:
        return "CRT-D"
    if "CRT-P" in upper or "CRT" in upper:
        return "CRT-P"
    if "ICD" in upper:
        return "ICD"
    return "Pacemaker"


def _parse_xml(xml_content: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(xml_content)
    except ET.ParseError:
        pass
    stripped = re.sub(r'^<\?xml[^?]*\?>', '', xml_content, count=1).lstrip()
    try:
        return ET.fromstring(stripped)
    except ET.ParseError:
        return None


def extract(source: str | Path, is_content: bool = False) -> dict:
    """Extract from a Microport/Paceart XML export.

    Args:
        source: file path or raw XML string (when is_content=True)
        is_content: if True, treat source as XML string content
    """
    if is_content:
        xml_str = source if isinstance(source, str) else str(source)
        source_file = "<content>"
    else:
        p = Path(source)
        raw = p.read_bytes()
        if raw[:2] == b'\xff\xfe':
            xml_str = raw[2:].decode('utf-16-le')
        elif raw[:2] == b'\xfe\xff':
            xml_str = raw[2:].decode('utf-16-be')
        elif raw[:3] == b'\xef\xbb\xbf':
            xml_str = raw[3:].decode('utf-8')
        else:
            prolog = raw[:200].decode('latin-1')
            m = re.search(r'encoding=["\']([^"\']+)["\']', prolog, re.IGNORECASE)
            enc = 'utf-8'
            if m:
                declared = m.group(1).lower()
                if declared in ('iso-8859-1', 'latin1', 'windows-1252', 'cp1252'):
                    enc = 'latin-1'
                elif 'utf-16' in declared:
                    enc = 'utf-16'
            xml_str = raw.decode(enc, errors='replace')
        source_file = str(p)

    root = _parse_xml(xml_str)
    if root is None or root.tag != "Paceart":
        return {"source_file": source_file, "manufacturer": "Microport", "error": "not a Paceart XML"}

    result: dict = {"source_file": source_file}

    # Lookup tables
    lt = root.find("LookupTables")
    pacemaker_details = _as_list(lt.find("Devices/Pacemakers") if lt is not None else None, "PacemakerDetail")
    lead_details = _as_list(lt.find("Devices/Leads") if lt is not None else None, "LeadDetail")

    # Patient record (take first)
    pr_container = root.find("PatientRecords")
    patient_records = _as_list(pr_container, "PatientRecord") if pr_container is not None else []
    if not patient_records:
        result["manufacturer"] = "Microport"
        return {k: v for k, v in result.items() if v is not None}
    pr = patient_records[0]

    # Demographics
    demo = pr.find("Demographics")
    name_first = _val(demo, "nameFirst")
    name_last = _val(demo, "nameLast")
    if name_first:
        result["patient_name"] = f"{name_last}, {name_first}" if name_last else name_first
    elif name_last:
        result["patient_name"] = name_last
    dob = _val(demo, "BirthDate")
    if dob:
        result["patient_dob"] = dob

    # Device
    devices = pr.find("Devices")
    pacemaker = devices.find("Pacemaker") if devices is not None else None
    serial = _val(pacemaker, "SerialNumber")
    if serial:
        result["device_serial"] = serial

    # Resolve device model/manufacturer via GUID lookup
    guid = None
    if pacemaker is not None:
        lookup = pacemaker.find("PacemakerLookup")
        if lookup is not None:
            ref = lookup.find("PacemakerReference")
            if ref is not None:
                guid = _val(ref, "GUID")

    device_detail = _lookup_by_guid(pacemaker_details, guid)
    device_model = _val(device_detail, "Model") or "Unknown"
    manufacturer = _val(device_detail, "Manufacturer") or "Microport"
    result["manufacturer"] = manufacturer
    result["device_model"] = device_model
    result["device_type"] = _infer_device_type(device_model)

    # Interrogation — take latest PacemakerClinic
    tests = pr.find("Tests")
    clinics = _as_list(tests, "PacemakerClinic") if tests is not None else []
    clinic = clinics[-1] if clinics else None
    clinic_date = _val(clinic, "Date")
    if clinic_date:
        result["interrogation_date"] = clinic_date

    evaluation = clinic.find("Evaluation") if clinic is not None else None
    telemetry = evaluation.find("PacemakerTelemetry") if evaluation is not None else None
    thresholds = evaluation.find("Thresholds") if evaluation is not None else None

    # Battery
    bv = _val(telemetry, "BatteryVoltage")
    if bv:
        result["battery_voltage"] = bv
    bi = _val(telemetry, "BatteryImpedance_ohms")
    if bi:
        result["battery_impedance_ohm"] = bi

    # Leads
    lead_els = _as_list(devices, "Lead") if devices is not None else []
    sensing_list = _as_list(thresholds, "Sensing") if thresholds is not None else []
    capture_list = _as_list(thresholds, "Capture") if thresholds is not None else []
    telemetry_leads = _as_list(telemetry, "Lead") if telemetry is not None else []

    for lead_el in lead_els:
        implant_info = lead_el.find("ImplantInformation")
        chamber_raw = _val(implant_info, "Chamber") or ""
        chamber = "RA" if chamber_raw == "Atrium" else ("RV" if chamber_raw == "Ventricle" else chamber_raw)
        measure_chamber = chamber if chamber in ("RA", "RV", "LV") else chamber_raw

        # Resolve lead model via GUID
        lead_lookup = lead_el.find("LeadLookup")
        lead_ref = lead_lookup.find("LeadReference") if lead_lookup is not None else None
        lead_guid = _val(lead_ref, "GUID") if lead_ref is not None else None
        ld = _lookup_by_guid(lead_details, lead_guid)

        prefix = chamber.lower() if chamber else "unknown"

        lead_model = _val(ld, "Model")
        if lead_model:
            result[f"lead_{prefix}_model"] = lead_model
        lead_serial = _val(lead_el, "SerialNumber")
        if lead_serial:
            result[f"lead_{prefix}_serial"] = lead_serial
        implant_date = _val(implant_info, "Date")
        if implant_date:
            result[f"lead_{prefix}_implant_date"] = implant_date

        # Sensing
        sensing_entry = _find_by_chamber(sensing_list, measure_chamber)
        if sensing_entry is not None:
            amp_mv = _val(sensing_entry, "Amplitude_millivolts")
            if amp_mv:
                result[f"sensing_{prefix}_mv"] = amp_mv

        # Capture threshold
        capture_entry = _find_by_chamber(capture_list, measure_chamber)
        if capture_entry is not None:
            cap_v = _val(capture_entry, "Amplitude_volts")
            if cap_v:
                result[f"pacing_threshold_{prefix}_v"] = cap_v
            cap_pw = _val(capture_entry, "Duration_ms")
            if cap_pw:
                result[f"pacing_threshold_{prefix}_pw_ms"] = cap_pw

        # Impedance
        imp_entry = _find_by_chamber(telemetry_leads, measure_chamber)
        if imp_entry is not None:
            imp_val = _val(imp_entry, "BipolarImpedance_ohms")
            if imp_val:
                result[f"impedance_{prefix}_ohm"] = imp_val

    return {k: v for k, v in result.items() if v is not None}
