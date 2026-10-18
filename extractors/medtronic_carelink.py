"""
Medtronic CareLink Network — Quick Look II PDF extractor.

Text-based PDF (pdfplumber); German-language Carelink report.
Covers CRT-D and ICD devices (e.g. Compia MRI Quad, Evita, Cobalt, etc.).

Usage:
    from extractors.medtronic_carelink import extract
    result = extract("quicklook.pdf")
"""
import re
from pathlib import Path
from typing import Optional

import pdfplumber


def _txt(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def _find(pattern: str, text: str, group: int = 1, flags: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _date_carelink(s: Optional[str]) -> Optional[str]:
    """Convert '20-Jul-2026' or '02-Sep-2022' → 'YYYY-MM-DD'."""
    if not s:
        return None
    _months = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
               "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
               "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    m = re.match(r'(\d{2})-([A-Za-z]{3})-(\d{4})', s.strip())
    if m:
        return f"{m.group(3)}-{_months.get(m.group(2), m.group(2))}-{m.group(1)}"
    return s


def extract(pdf_path: str | Path) -> dict:
    """
    Extract structured parameters from a Medtronic CareLink Quick Look II PDF.

    Returns keys:
        report_type, manufacturer, source_file
        followup_date, implant_date, longevity_years
        device_model, device_serial
        mode, base_rate_bpm, upper_rate_bpm, av_paced_ms, av_sensed_ms
        v_stimulation_sequence
        impedance_a_ohm, impedance_rv_ohm, impedance_lv_ohm, impedance_rv_defib_ohm
        threshold_a_v, threshold_a_pw_ms, threshold_rv_v, threshold_rv_pw_ms
        threshold_lv_v, threshold_lv_pw_ms
        amplitude_a_v, pulse_width_a_ms, amplitude_rv_v, pulse_width_rv_ms
        amplitude_lv_v, pulse_width_lv_ms
        signal_a_mv, signal_rv_mv, sensitivity_a_mv, sensitivity_rv_mv
        zone_ataf_rate, zone_vf_rate, zone_vt_range
        vf_episodes, vt_episodes, ataf_episodes
        ataf_duration_h_per_day, ataf_longest_s
        vp_pct, as_vs_pct, as_vp_pct, ap_vs_pct, ap_vp_pct
        patient_activity_h_per_day
        optivol_alert
    """
    pdf_path = Path(pdf_path)
    text = _txt(pdf_path)

    # ── device identity ────────────────────────────────────────────────────────
    # "Gerät: Compia MRI™ Quad CRTD DTMC2QQ Seriennummer: RPF632493S Abfragedatum: 20-Jul-2026"
    device_model  = _find(r'Ger[äa]t:\s+(.+?)\s+Seriennummer:', text)
    device_serial = _find(r'Seriennummer:\s+(\S+)', text)
    followup_raw  = _find(r'Abfragedatum:\s+(\d{2}-[A-Za-z]{3}-\d{4})', text)
    implant_raw   = _find(r'Implantiert:\s+(\d{2}-[A-Za-z]{3}-\d{4})', text)
    longevity     = _find(r'Verbleibende Laufzeit\s+([\d.,]+)\s+Jahr', text)

    # ── pacing parameters ─────────────────────────────────────────────────────
    mode       = _find(r'Betriebsart\s+(\S+)', text)
    base_rate  = _find(r'Interv\.?freq\.?\s+([\d,]+)\s+min', text)
    max_sync   = _find(r'Max\.\s*Sync\.\s+([\d,]+)\s+min', text)
    max_sensor = _find(r'Max\.\s*Sensor\s+([\d,]+)\s+min', text)
    av_paced   = _find(r'Stim\.\s*AV-Int\.\s+([\d,]+)\s*ms', text)
    av_sensed  = _find(r'Wahrgen\.\s*AV-Int\.?\s*([\d,]+)\s*ms', text)
    v_stim_seq = _find(r'V\.\s*Stimulation\s+(\S+)', text)

    # ── lead impedances ────────────────────────────────────────────────────────
    # "Stimulationsimpedanz 304 Ohm 399 Ohm 779 Ohm"  (A, RV, LV)
    m_imp = re.search(r'Stimulationsimpedanz\s+([\d,]+)\s*Ohm\s+([\d,]+)\s*Ohm(?:\s+([\d,]+)\s*Ohm)?', text)
    imp_a  = m_imp.group(1) if m_imp else None
    imp_rv = m_imp.group(2) if m_imp else None
    imp_lv = m_imp.group(3) if m_imp else None
    imp_rv_defib = _find(r'Defibrillationsimpedanz\s+RV=([\d,]+)\s*Ohm', text)

    # ── thresholds ─────────────────────────────────────────────────────────────
    # "Stimulationsreizschwelle 0.625 V @ 0.4 ms  0.625 V @ 0.4 ms  0.875 V @ 1.00 ms"
    m_thr = re.search(
        r'Stimulationsreizschwelle\s+([\d.]+)\s*V\s*@\s*([\d.]+)\s*ms\s+'
        r'([\d.]+)\s*V\s*@\s*([\d.]+)\s*ms(?:\s+([\d.]+)\s*V\s*@\s*([\d.]+)\s*ms)?',
        text)
    thr_a_v   = m_thr.group(1) if m_thr else None
    thr_a_pw  = m_thr.group(2) if m_thr else None
    thr_rv_v  = m_thr.group(3) if m_thr else None
    thr_rv_pw = m_thr.group(4) if m_thr else None
    thr_lv_v  = m_thr.group(5) if m_thr else None
    thr_lv_pw = m_thr.group(6) if m_thr else None

    # ── programmed amplitude / pulse width ────────────────────────────────────
    # "Programmierte Amplitude/Impulsdauer 1.50 V / 0.40 ms  2.00 V / 0.40 ms  2.00 V / 1.00 ms"
    m_amp = re.search(
        r'Programmierte Amplitude/Impulsdauer\s+([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms\s+'
        r'([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms(?:\s+([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms)?',
        text)
    amp_a_v   = m_amp.group(1) if m_amp else None
    pw_a_ms   = m_amp.group(2) if m_amp else None
    amp_rv_v  = m_amp.group(3) if m_amp else None
    pw_rv_ms  = m_amp.group(4) if m_amp else None
    amp_lv_v  = m_amp.group(5) if m_amp else None
    pw_lv_ms  = m_amp.group(6) if m_amp else None

    # ── signal / sensing ──────────────────────────────────────────────────────
    # "Gemessene P-Welle/R-Zacke 1.1 mV  1.0 mV"
    m_sig = re.search(r'Gemessene P-Welle/R-Zacke\s+([\d.]+)\s*mV\s+([\d.]+)\s*mV', text)
    sig_a_mv  = m_sig.group(1) if m_sig else None
    sig_rv_mv = m_sig.group(2) if m_sig else None
    # "Programmierte Empfindlichkeit 0.90 mV  0.30 mV"
    m_sens = re.search(r'Programmierte Empfindlichkeit\s+([\d.]+)\s*mV\s+([\d.]+)\s*mV', text)
    sens_a_mv  = m_sens.group(1) if m_sens else None
    sens_rv_mv = m_sens.group(2) if m_sens else None

    # ── detection zones ───────────────────────────────────────────────────────
    zone_ataf = _find(r'AT/AF\s+Monitor\s+>([\d]+)\s*min', text)
    zone_vf   = _find(r'VF\s+Ein\s+>([\d]+)\s*min', text)
    m_vt = re.search(r'VT\s+Ein\s+([\d]+)-([\d]+)\s*min', text)
    zone_vt   = f"{m_vt.group(1)}-{m_vt.group(2)}" if m_vt else None

    # ── episode counts ────────────────────────────────────────────────────────
    # Under "Behandelt" section
    vf_ep   = _find(r'VF\s+(\d+)\s*\n', text)
    vt_ep   = _find(r'VT\s+(\d+)\s*\n', text)
    ataf_ep = _find(r'AT/AF\s+(\d+)\s*\n', text)

    # ── AT/AF burden ──────────────────────────────────────────────────────────
    ataf_dur     = _find(r'Dauer AT/AF\s+([\d.<]+)\s+Stunden', text)
    ataf_longest = _find(r'Längste AT/AF\s+([\d]+)\s+Sekunden', text)

    # ── % stimulation ─────────────────────────────────────────────────────────
    vp_pct    = _find(r'VP gesamt\s+([\d.]+)%', text)
    as_vs_pct = _find(r'AS-VS\s+(<?\s*[\d.]+)\s*%', text)
    as_vp_pct = _find(r'AS-VP\s+(<?\s*[\d.]+)\s*%', text)
    ap_vs_pct = _find(r'AP-VS\s+(<?\s*[\d.]+)\s*%', text)
    ap_vp_pct = _find(r'AP-VP\s+(<?\s*[\d.]+)\s*%', text)

    # ── functional status ─────────────────────────────────────────────────────
    activity = _find(r'Patientenaktivit[äa]t\s+([\d.]+)\s+Stunden', text)

    # ── alerts ────────────────────────────────────────────────────────────────
    optivol = bool(re.search(r'OptiVol Fl[üu]ssigkeitsansammlung', text))

    result: dict = {
        "report_type":       "QuickLookII",
        "manufacturer":      "Medtronic",
        "source_file":       str(pdf_path),
        "followup_date":     _date_carelink(followup_raw),
        "implant_date":      _date_carelink(implant_raw),
        "longevity_years":   _num(longevity),
        "device_model":      device_model,
        "device_serial":     device_serial,
        "mode":              mode,
        "base_rate_bpm":     base_rate,
        "upper_rate_bpm":    max_sync,
        "max_sensor_bpm":    max_sensor,
        "av_paced_ms":       av_paced,
        "av_sensed_ms":      av_sensed,
        "v_stimulation_sequence": v_stim_seq,
        "impedance_a_ohm":        _num(imp_a),
        "impedance_rv_ohm":       _num(imp_rv),
        "impedance_lv_ohm":       _num(imp_lv),
        "impedance_rv_defib_ohm": _num(imp_rv_defib),
        "threshold_a_v":     thr_a_v,
        "threshold_a_pw_ms": thr_a_pw,
        "threshold_rv_v":    thr_rv_v,
        "threshold_rv_pw_ms":thr_rv_pw,
        "threshold_lv_v":    thr_lv_v,
        "threshold_lv_pw_ms":thr_lv_pw,
        "amplitude_a_v":     amp_a_v,
        "pulse_width_a_ms":  pw_a_ms,
        "amplitude_rv_v":    amp_rv_v,
        "pulse_width_rv_ms": pw_rv_ms,
        "amplitude_lv_v":    amp_lv_v,
        "pulse_width_lv_ms": pw_lv_ms,
        "signal_a_mv":       sig_a_mv,
        "signal_rv_mv":      sig_rv_mv,
        "sensitivity_a_mv":  sens_a_mv,
        "sensitivity_rv_mv": sens_rv_mv,
        "zone_ataf_rate_min": zone_ataf,
        "zone_vf_rate_min":   zone_vf,
        "zone_vt_range_min":  zone_vt,
        "vf_episodes":        vf_ep,
        "vt_episodes":        vt_ep,
        "ataf_episodes":      ataf_ep,
        "ataf_duration_h_per_day": ataf_dur,
        "ataf_longest_s":     ataf_longest,
        "vp_pct":     vp_pct,
        "as_vs_pct":  as_vs_pct,
        "as_vp_pct":  as_vp_pct,
        "ap_vs_pct":  ap_vs_pct,
        "ap_vp_pct":  ap_vp_pct,
        "patient_activity_h_per_day": activity,
        "optivol_alert": optivol,
    }
    return result
