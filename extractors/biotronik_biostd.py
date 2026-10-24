"""
Biotronik BIOSTD (CardDAS) XML extractor.

Parses the Biotronik PSW/ICS programmer XML export in GE Centricity CardDAS
format (InterfaceData namespace). These files use AttributeID-based entries
grouped in numbered tables.

Key tables:
  9110 — test results (impedance, thresholds, sensing, battery)
  9002 — programmed parameters (mode, rates, amplitudes, zones)

Usage:
    from extractors.biotronik_biostd import extract
    result = extract("BIOSTD_2022-02-01_Weinrich_M_60860817.xml")
"""
import re
from pathlib import Path
from typing import Optional

import xml.etree.ElementTree as ET

NS = 'http://med.ge.com//carddas'


def _parse_tables(root) -> dict[str, list[dict]]:
    """Return {table_name: [{id, name, value}, ...]}."""
    tables: dict[str, list[dict]] = {}
    for table in root.iter(f'{{{NS}}}Table'):
        tname_el = table.find(f'{{{NS}}}TableName')
        tname = tname_el.text if tname_el is not None else ''
        entries = []
        for entry in table.findall(f'{{{NS}}}TableEntry'):
            aid = entry.findtext(f'{{{NS}}}AttributeID', '').strip()
            aname = entry.findtext(f'{{{NS}}}AttributeName', '').strip()
            aval = (entry.findtext(f'{{{NS}}}CharValue', '')
                    or entry.findtext(f'{{{NS}}}NumericValue', '')).strip()
            if aid or aname:
                entries.append({'id': aid, 'name': aname, 'value': aval})
        tables.setdefault(tname, []).extend(entries)
    return tables


def _by_id(entries: list[dict], aid: str) -> Optional[str]:
    """Find first entry by AttributeID, return value."""
    for e in entries:
        if e['id'] == aid and e['value'] and e['value'] != '.':
            return None if e['value'].startswith('----') else e['value']
    return None


def _by_name(entries: list[dict], name: str) -> Optional[str]:
    """Find first entry by AttributeName, return value."""
    for e in entries:
        if e['name'] == name and e['value'] and e['value'] != '.':
            return None if e['value'].startswith('----') else e['value']
    return None


def extract(xml_path: str | Path) -> dict:
    """Extract clinical parameters from a Biotronik BIOSTD CardDAS XML file."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    result: dict = {
        "source_file": str(xml_path),
        "manufacturer": "Biotronik",
        "report_type": "FollowUpReport",
    }

    # ── Patient ─────────────────────────────────────────────────────────
    pd = root.find(f'.//{{{NS}}}PersonalData')
    if pd is not None:
        first = (pd.findtext(f'{{{NS}}}FirstName') or '').strip()
        last = (pd.findtext(f'{{{NS}}}LastName') or '').strip()
        if last:
            result["patient_name"] = f"{last}, {first}" if first else last
        dob = (pd.findtext(f'{{{NS}}}DateOfBirth') or '').strip()
        if dob:
            m = re.match(r'(\d{4})-(\d{2})-(\d{2})', dob)
            if m:
                result["patient_dob"] = f"{m.group(3)}.{m.group(2)}.{m.group(1)}"

    # ── Dates ───────────────────────────────────────────────────────────
    exam_date = root.find(f'.//{{{NS}}}ExaminationDate')
    if exam_date is not None and exam_date.text:
        m = re.match(r'(\d{4})-(\d{2})-(\d{2})', exam_date.text.strip())
        if m:
            result["followup_date"] = f"{m.group(3)}.{m.group(2)}.{m.group(1)}"

    # ── Device (from header comment or filename) ────────────────────────
    impl = root.find(f'.//{{{NS}}}Implantation')
    if impl is not None:
        for child in impl:
            tag = child.tag.split('}')[-1]
            if tag == 'DeviceModelName':
                result["device_model"] = (child.text or '').strip()
            elif tag == 'DeviceSerialNumber':
                result["device_serial"] = (child.text or '').strip()
            elif tag == 'ImplantDate':
                dt = (child.text or '').strip()
                m2 = re.match(r'(\d{4})-(\d{2})-(\d{2})', dt)
                if m2:
                    result["implant_date"] = f"{m2.group(3)}.{m2.group(2)}.{m2.group(1)}"

    tables = _parse_tables(root)

    # ── Test results (table 9110) ───────────────────────────────────────
    t9110 = tables.get('9110', [])

    # AttributeIDs: 72=imp_RA, 73=imp_RV, 524=imp_LV, 713=shock_imp
    #               831=sens_RA, 833=sens_RV, 521=sens_LV
    #               832=thresh_RA, 834=thresh_RV, 522=thresh_LV
    #               119=battery_V, 880=battery_pct
    result["impedance_a_ohm"] = _by_id(t9110, '72')
    result["impedance_rv_ohm"] = _by_id(t9110, '73')
    result["impedance_lv_ohm"] = _by_id(t9110, '524')
    result["shock_impedance_ohm"] = _by_id(t9110, '713')
    result["sensing_a_mv"] = _by_id(t9110, '831')
    result["sensing_rv_mv"] = _by_id(t9110, '833')
    result["sensing_lv_mv"] = _by_id(t9110, '521')
    result["threshold_a_v"] = _by_id(t9110, '832')
    result["threshold_rv_v"] = _by_id(t9110, '834')
    result["threshold_lv_v"] = _by_id(t9110, '522')
    result["battery_voltage_v"] = _by_id(t9110, '119')
    result["battery_remaining_pct"] = _by_id(t9110, '880')

    # ── Programmed parameters (table 9002) ──────────────────────────────
    t9002 = tables.get('9002', [])

    result["mode"] = _by_id(t9002, '99')
    result["base_rate_bpm"] = _by_id(t9002, '69')
    result["upper_rate_bpm"] = _by_id(t9002, '387')
    result["amplitude_rv_v"] = _by_id(t9002, '77')
    result["amplitude_lv_v"] = _by_id(t9002, '515')
    result["vt1_bpm"] = _by_id(t9002, '639')
    result["vt2_bpm"] = _by_id(t9002, '643')
    result["vf_bpm"] = _by_id(t9002, '648')

    # ── Threshold details from capture test tables ──────────────────────
    # Table 9019 = RV capture test, 9131 = LV capture test
    t9019 = tables.get('9019', [])
    t9131 = tables.get('9131', [])
    if not result.get("threshold_rv_v"):
        result["threshold_rv_v"] = _by_name(t9019, 'Schwelle')
    if not result.get("threshold_lv_v"):
        result["threshold_lv_v"] = _by_name(t9131, 'Schwelle')

    return result
