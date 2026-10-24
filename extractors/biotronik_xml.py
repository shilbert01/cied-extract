"""
Biotronik ICS3000 XML extractor.

Parses the proprietary Biotronik IEEE-11073 XML export from ICS3000
programmers. These files contain structured device/patient/measurement
data in a section/value tree.

Sections:
  DEV  — device model, serial, implant date
  SESS — session (followup) date
  PT   — patient name, DOB
  MSMT — measured values: battery, lead impedances, thresholds, sensing
  SET  — programmed parameters: mode, rates, amplitudes, pulse widths, zones

Usage:
    from extractors.biotronik_xml import extract
    result = extract("BIOIEEE_60860817_Weinrich_M_2022-02-01.xml")
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import xml.etree.ElementTree as ET


def _val(section, name: str) -> Optional[str]:
    """Get a <value name='X'> text from a section element."""
    for v in section.iter('value'):
        if v.get('name') == name:
            t = (v.text or '').strip()
            return t if t else None
    return None


def _section(parent, name: str):
    """Find a direct child <section name='X'>."""
    for s in parent:
        if s.tag == 'section' and s.get('name') == name:
            return s
    return None


def _sections(parent, name: str) -> list:
    """Find all direct child <section name='X'>."""
    return [s for s in parent if s.tag == 'section' and s.get('name') == name]


def _find_idc(root) -> Optional[ET.Element]:
    """Navigate to the IDC section: dataset > section[MDC] > section[IDC]."""
    for dataset in root.iter('dataset'):
        mdc = _section(dataset, 'MDC')
        if mdc is not None:
            return _section(mdc, 'IDC')
    return None


def _parse_datetime(raw: Optional[str]) -> Optional[str]:
    """Convert '20220201T131253+0200' or '20170712T000000' → 'DD.MM.YYYY'."""
    if not raw:
        return None
    m = re.match(r'(\d{4})(\d{2})(\d{2})T', raw)
    if m:
        return f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
    return None


def _parse_dob(raw: Optional[str]) -> Optional[str]:
    """Convert '19431119' → '19.11.1943'."""
    if not raw or len(raw) < 8:
        return None
    return f"{raw[6:8]}.{raw[4:6]}.{raw[0:4]}"


def _lead_msmt(msmt, channel: str) -> dict:
    """Extract measurements for a lead channel (LEADCHNL_RA/RV/LV)."""
    out = {}
    ch = _section(msmt, channel)
    if ch is None:
        return out

    # Pacing threshold
    pt = _section(ch, 'PACING_THRESHOLD')
    if pt is not None:
        out['threshold_v'] = _val(pt, 'AMPLITUDE')
        out['threshold_pw_ms'] = _val(pt, 'PULSEWIDTH')

    # Impedance
    imp = _section(ch, 'IMPEDANCE')
    if imp is not None:
        out['impedance_ohm'] = _val(imp, 'VALUE')

    # Sensing
    sens = _section(ch, 'SENSING')
    if sens is not None:
        out['sensing_mv'] = _val(sens, 'AMPLITUDE')

    return out


def _lead_settings(settings, channel: str) -> dict:
    """Extract programmed parameters for a lead channel."""
    out = {}
    ch = _section(settings, channel)
    if ch is None:
        return out

    pacing = _section(ch, 'PACING')
    if pacing is not None:
        out['amplitude_v'] = _val(pacing, 'AMPLITUDE')
        out['pulse_width_ms'] = _val(pacing, 'PULSEWIDTH')

    sensing = _section(ch, 'SENSING')
    if sensing is not None:
        out['sensitivity_mv'] = _val(sensing, 'SENSITIVITY')

    return out


def extract(xml_path: str | Path) -> dict:
    """Extract clinical parameters from a Biotronik ICS3000 XML export."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    result: dict = {
        "source_file": str(xml_path),
        "manufacturer": "Biotronik",
        "report_type": "FollowUpReport",
    }

    idc = _find_idc(root)
    if idc is None:
        return result

    # ── Device ──────────────────────────────────────────────────────────
    dev = _section(idc, 'DEV')
    if dev is not None:
        result["device_model"] = _val(dev, 'MODEL')
        result["device_serial"] = _val(dev, 'SERIAL')
        result["device_type"] = _val(dev, 'TYPE')
        result["implant_date"] = _parse_datetime(_val(dev, 'IMPLANT_DT'))

    # ── Session ─────────────────────────────────────────────────────────
    sess = _section(idc, 'SESS')
    if sess is not None:
        result["followup_date"] = _parse_datetime(_val(sess, 'DTM'))
        result["last_followup"] = _parse_datetime(_val(sess, 'DTM_PREVIOUS'))

    # ── Patient ─────────────────────────────────────────────────────────
    attr = _section(_section(root.find('dataset'), 'MDC'), 'ATTR') if root.find('dataset') is not None else None
    pt = _section(attr, 'PT') if attr is not None else None
    if pt is not None:
        given = _val(pt, 'NAME_GIVEN') or ''
        family = _val(pt, 'NAME_FAMILY') or ''
        if family:
            result["patient_name"] = f"{family}, {given}" if given else family
        result["patient_dob"] = _parse_dob(_val(pt, 'DOB'))

    # ── Measurements ────────────────────────────────────────────────────
    msmts = _sections(idc, 'MSMT')
    msmt = msmts[0] if msmts else None
    if msmt is not None:
        # Battery
        bat = _section(msmt, 'BATTERY')
        if bat is not None:
            result["battery_voltage_v"] = _val(bat, 'VOLTAGE')
            result["battery_status"] = _val(bat, 'STATUS')
            result["battery_remaining_pct"] = _val(bat, 'REMAINING_PERCENTAGE')

        # Charge time
        cap = _section(msmt, 'CAP')
        if cap is not None:
            result["charge_time_s"] = _val(cap, 'CHARGE_TIME')
            result["charge_energy_j"] = _val(cap, 'CHARGE_ENERGY')

        # Lead measurements
        for prefix, channel in [('a', 'LEADCHNL_RA'), ('rv', 'LEADCHNL_RV'), ('lv', 'LEADCHNL_LV')]:
            lm = _lead_msmt(msmt, channel)
            if lm.get('threshold_v'):
                result[f"threshold_{prefix}_v"] = lm['threshold_v']
            if lm.get('threshold_pw_ms'):
                result[f"threshold_{prefix}_pw_ms"] = lm['threshold_pw_ms']
            if lm.get('impedance_ohm'):
                result[f"impedance_{prefix}_ohm"] = lm['impedance_ohm']
            if lm.get('sensing_mv'):
                result[f"sensing_{prefix}_mv"] = lm['sensing_mv']

        # Shock impedance
        hv = _section(msmt, 'LEADHVCHNL')
        if hv is not None:
            result["shock_impedance_ohm"] = _val(hv, 'IMPEDANCE')

    # ── Settings ────────────────────────────────────────────────────────
    settings_list = _sections(idc, 'SET')
    settings = settings_list[0] if settings_list else None
    if settings is not None:
        # Mode and rates
        brady = _section(settings, 'BRADY')
        if brady is not None:
            result["mode"] = _val(brady, 'MODE')
            result["base_rate_bpm"] = _val(brady, 'LOWRATE')
            result["upper_rate_bpm"] = _val(brady, 'MAX_SENSOR_RATE')

        # Lead programmed parameters
        for prefix, channel in [('a', 'LEADCHNL_RA'), ('rv', 'LEADCHNL_RV'), ('lv', 'LEADCHNL_LV')]:
            ls = _lead_settings(settings, channel)
            if ls.get('amplitude_v'):
                result[f"amplitude_{prefix}_v"] = ls['amplitude_v']
            if ls.get('pulse_width_ms'):
                result[f"pulse_width_{prefix}_ms"] = ls['pulse_width_ms']

        # VT/VF zones
        zones = _sections(settings, 'ZONE')
        for zone in zones:
            zone_type = _val(zone, 'TYPE') or ''
            det_interval = _val(zone, 'DETECTION_INTERVAL')
            if det_interval and det_interval.isdigit():
                bpm = str(round(60000 / int(det_interval)))
                if 'VF' in zone_type:
                    result["vf_bpm"] = bpm
                elif 'VT' in zone_type:
                    vendor = _val(zone, 'VENDOR_TYPE') or ''
                    if 'VT2' in vendor:
                        result["vt2_bpm"] = bpm
                    else:
                        result["vt1_bpm"] = bpm

    return result
