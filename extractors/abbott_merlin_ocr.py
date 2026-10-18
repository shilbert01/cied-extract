"""
Abbott Merlin PCS — CPU-only OCR extractor (PaddleOCR backend).

Uses PaddleOCR 3.x (PP-OCRv6) with enable_mkldnn=False for CPU inference.
For image-based Abbott PDFs on systems without a GPU.
Handles FastPathSummary reports; other report types return partial data.

Usage:
    from extractors.abbott_merlin_ocr import extract
    result = extract("report.pdf")
"""
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from pdf2image import convert_from_path


def _ocr_pages(pdf_path: Path) -> list[str]:
    """Convert PDF to images, run PaddleOCR on each page, return flat line list."""
    from paddleocr import PaddleOCR  # lazy import — heavy init

    ocr = PaddleOCR(lang="de", device="cpu", enable_mkldnn=False)
    pages = convert_from_path(pdf_path, dpi=200)

    all_lines: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, page_img in enumerate(pages):
            img_path = os.path.join(tmp_dir, f"page_{i}.png")
            page_img.save(img_path)
            results = ocr.predict(img_path)
            for res in results:
                all_lines.extend(res["rec_texts"])

    return all_lines


def _after(lines: list[str], label_re: str, value_re: str,
           skip: int = 0, flags: int = re.IGNORECASE) -> Optional[str]:
    """Return the (skip+1)-th match of value_re in lines following label_re."""
    found_label = False
    hits = 0
    for line in lines:
        s = line.strip()
        if not found_label:
            if re.search(label_re, s, flags):
                found_label = True
            continue
        # past the label: look for value match; stop at next label-looking line
        if re.search(label_re, s, flags):
            break
        if re.search(value_re, s, flags):
            if hits == skip:
                m = re.search(value_re, s, flags)
                return m.group(1).strip() if m.lastindex else s
            hits += 1
    return None


def _first_match(lines: list[str], value_re: str, group: int = 1,
                 flags: int = re.IGNORECASE) -> Optional[str]:
    for line in lines:
        m = re.search(value_re, line.strip(), flags)
        if m:
            return m.group(group).strip()
    return None


def _section(lines: list[str], start_re: str, end_re: str,
             flags: int = re.IGNORECASE) -> list[str]:
    """Return lines from (but not including) start_re up to end_re."""
    result: list[str] = []
    inside = False
    for line in lines:
        s = line.strip()
        if not inside:
            if re.search(start_re, s, flags):
                inside = True
            continue
        if re.search(end_re, s, flags):
            break
        result.append(s)
    return result


def _extract_icd(lines: list[str], pdf_path: Path) -> dict:
    """ICD FastPathSummary extraction — Entrant DR/VR, Quadra Assura ICD variants."""
    # identity — "Entrant™ DR CDDRA300Q ICD 810131959"
    device_serial = _first_match(lines, r'\bICD\s+(\d{5,10})\b')
    device_model  = _first_match(lines, r'\b([A-Z]{2,4}[A-Z0-9]+Q)\s+ICD\b')

    # followup date
    followup_date = _first_match(
        lines, r'FastPath\S*\s+[ÜU]bersicht.*?(\d+[-\s][A-Za-z]{3}[\s-]?\d{4})',
        flags=re.IGNORECASE | re.DOTALL)
    if not followup_date:
        followup_date = _first_match(
            lines, r'^(\d+[-\s][A-Za-z]{3}[\s-]?\d{4})\s*$', flags=re.IGNORECASE)
    diagnostics_since = _first_match(lines, r'Seit\s+(\d+\s+\w{3}\s+\d{4})')

    # battery (between "Batterie" header and "Testergebnisse"/"Parameter")
    batt_section = _section(lines, start_re=r'^Batterie$', end_re=r'Testergebnisse|^Parameter$')
    battery_current_ua = _first_match(batt_section, r'^(\d+)\s*[uµ][Aa]')
    # ERI value line comes immediately after the "ERI" label line
    battery_eri       = _after(batt_section, r'^ERI$', r'^(.+)$')
    battery_remaining = _first_match(batt_section, r'^(>?[\d,]+)%')

    # test results (sensing/impedance, between "Testergebnisse" and "Parameter")
    test_section = _section(lines, start_re=r'Testergebnisse', end_re=r'^Parameter$')
    _test_val_re = r'^(Nicht durchgeführt|\d[\d,]*)$'
    sensing_v    = _after(test_section, r'^V$',  _test_val_re, skip=0)
    impedance_v  = _after(test_section, r'^V$',  _test_val_re, skip=1)
    impedance_hv = _after(test_section, r'^HV$', _test_val_re, skip=0)

    # operating mode — "Stimulation" (longer than the {2,5} PM pattern)
    mode = _after(lines, r'^Betriebsart$', r'^(\S.{0,25})$')

    # zone table: "Detektionsparameter | Aus | Aus" (VT col, VF col)
    vt_zone       = _after(lines, r'^Detektionsparameter$', r'^(Aus|Ein)$', skip=0)
    vf_zone       = _after(lines, r'^Detektionsparameter$', r'^(Aus|Ein)$', skip=1)
    detection_params = vt_zone
    therapy       = _after(lines, r'^Therapie', r'^(Aus|Ein)$')

    # sensing section (ICD V-channel only)
    stim_section    = _section(lines, start_re=r'Stimulation und Wahrnehmung', end_re=r'Diagnostik')
    sense_ability_v = _after(stim_section, r'SenseAbility', r'^(Aus|Ein|Monitor)$')
    sensing_config_v = _after(stim_section, r'Empfindlichkeit', r'^(\S+)$')

    # episodes
    vt_vf_episodes = _first_match(lines, r'VT/VF-Episoden:\s*(\d+)')

    return {
        "report_type":           "FastPathSummary",
        "source_file":           str(pdf_path),
        "followup_date":         followup_date,
        "diagnostics_since":     diagnostics_since,
        "device_model":          device_model,
        "device_serial":         device_serial,
        "battery_current_ua":    battery_current_ua,
        "battery_eri":           battery_eri,
        "battery_remaining_pct": battery_remaining,
        "sensing_v":             sensing_v,
        "impedance_v":           impedance_v,
        "impedance_hv":          impedance_hv,
        "mode":                  mode,
        "vt_zone":               vt_zone,
        "vf_zone":               vf_zone,
        "detection_params":      detection_params,
        "therapy":               therapy,
        "sense_ability_v":       sense_ability_v,
        "sensing_config_v":      sensing_config_v,
        "vt_vf_episodes":        vt_vf_episodes,
    }


def extract(pdf_path: str | Path) -> dict:
    """
    Extract structured parameters from an Abbott Merlin FastPathSummary PDF
    using PaddleOCR PP-OCRv6 (CPU-only, no GPU, enable_mkldnn=False).

    Returns the same field set as abbott_merlin.py for FastPathSummary reports.
    """
    pdf_path = Path(pdf_path)
    lines = _ocr_pages(pdf_path)
    text = "\n".join(lines)

    # ── ICD branch ─────────────────────────────────────────────────────────────
    if any(re.search(r'\bICD\b', line) for line in lines):
        return _extract_icd(lines, pdf_path)

    # ── device identity ────────────────────────────────────────────────────────
    # "Endurity MRI™ 2172 Schrittmacher 4177548"
    device_serial = _first_match(
        lines, r'(?:Schrittmacher|Herzschrittmacher|CRT-?[DP])\s+(\d{5,10})')
    device_model = _first_match(
        lines, r'(?:Endurity|Aveir|Entrant|Quadra|Assura)\S*\s+\S*\s*(?:MRI\S*|LP|DR|VR|HF)?\s*(\d{3,4})',
        flags=re.IGNORECASE)

    # ── followup date ──────────────────────────────────────────────────────────
    # Footer: "FastPath™ Übersicht Seite 1 von 1  6-Jul2026 17:11"
    # or a standalone date line like "6 Jul 2026"
    followup_date = _first_match(
        lines, r'FastPath\S*\s+[ÜU]bersicht.*?(\d+[-\s][A-Za-z]{3}[\s-]?\d{4})',
        flags=re.IGNORECASE | re.DOTALL)
    if not followup_date:
        followup_date = _first_match(
            lines, r'^(\d+[-\s][A-Za-z]{3}[\s-]?\d{4})\s*$', flags=re.IGNORECASE)
    diagnostics_since = _first_match(lines, r'Seit\s+(\d+\s+\w{3}\s+\d{4})')

    # ── battery ────────────────────────────────────────────────────────────────
    # PaddleOCR reads values before labels in the battery table:
    #   "3,10 V" / "Spannung"  |  "100,0 min-1" / "Magnetfrequenz"  |  ">95%"
    # Extract within the Batterie section (between "Batterie" header and "Testergebnisse").
    batt_section = _section(lines, start_re=r'^Batterie$', end_re=r'Testergebnisse|Parameter')
    battery_voltage_v = _first_match(batt_section, r'^([\d,]+)\s*V$')
    battery_mag_freq  = _first_match(batt_section, r'^([\d,]+)\s*min-1$')
    battery_remaining = _first_match(batt_section, r'^(>?[\d,]+)%$')

    # ── pacing parameters — use label → next-value approach ──────────────────
    mode       = _after(lines, r'^Betriebsart$', r'^([A-Z]{2,5}(?:\s+Rate\s+Resp\.?)?)$')
    base_rate  = _after(lines, r'Grundfrequenz', r'^(\d+)\s*min-1$')
    upper_rate = _after(lines, r'(?:MTR|Ob\s*Grenzfr)', r'^(\d+)\s*min-1$')
    av_paced   = _after(lines, r'Stimuliertes\s+AV-Intervall', r'^(\d+)\s*ms$')
    av_sensed  = _after(lines, r'Wahrg\.?\s+AV-Int', r'^(\d+)\s*ms$')

    # ── stimulation table — anchor on "ACap" / "AutoCapture" header ───────────
    # The stimulation table starts with the header row "ACap Confirm/V. AutoCapture"
    stim_section = _section(
        lines,
        start_re=r'ACap|AutoCapture|Stimulation und Wahrnehmung',
        end_re=r'Diagnostik[üu]bersicht|Episoden[üu]bersicht',
    )
    stim_text = "\n".join(stim_section)

    def _two_values(section_lines: list[str], label_re: str, value_re: str) -> tuple[Optional[str], Optional[str]]:
        """Find A-column and V-column values after label in stim section."""
        found = False
        hits: list[str] = []
        for line in section_lines:
            s = line.strip()
            if not found:
                if re.search(label_re, s, re.IGNORECASE):
                    found = True
                continue
            if re.search(label_re, s, re.IGNORECASE):
                break
            m = re.search(value_re, s, re.IGNORECASE)
            if m:
                hits.append(m.group(1))
                if len(hits) == 2:
                    break
        return (hits[0] if hits else None, hits[1] if len(hits) > 1 else None)

    # autocapture: Monitor/Aus/Ein
    _ac_re = r'^(Monitor|Aus|Ein|Best[aä]tigen)$'
    autocapture_a, autocapture_v = _two_values(stim_section, r'ACap|AutoCapture', _ac_re)

    # amplitude: "2,5V (3.3:1)" → extract leading number
    _amp_extract = r'^([\d,]+)V'
    amplitude_a_raw, amplitude_v_raw = _two_values(stim_section, r'Impulsamplitude|Amplitude', _amp_extract)
    amplitude_a = amplitude_a_raw
    amplitude_v = amplitude_v_raw

    # pulse width: "0,4 ms" or "0,4ms"
    _pw_extract = r'^([\d,]+)\s*ms$'
    pw_a, pw_v = _two_values(stim_section, r'Impulsdauer', _pw_extract)

    # SenseAbility
    _sa_re = r'^(Aus|Ein|Monitor)$'
    sense_a, sense_v = _two_values(stim_section, r'SenseAbility', _sa_re)

    # sensitivity: "0,5 mV (7,6:1)" or "2,0mV"
    _sens_extract = r'^([\d,]+)\s*m[Vv]'
    sensitivity_a, sensitivity_v = _two_values(stim_section, r'Empfindlichkeit', _sens_extract)

    # ── stimulation percentages ────────────────────────────────────────────────
    # In Diagnostikübersicht: "AP 100 % ... VP 100 %"
    # PaddleOCR separates them; "AP" appears then "100 %" then episode table headers
    diag_section = _section(
        lines,
        start_re=r'Diagnostik[üu]bersicht',
        end_re=r'Keine Warnhinweise|Manuell programmiert|FastPath',
    )
    # AP%/VP%: search full lines list after "Diagnostikübersicht" — don't use the
    # bounded diag_section because PDF column layout can push "VP" past the section end.
    def _parse_pct_line(raw: str) -> str | None:
        """Extract '>99' / '100' etc. from a percentage OCR line.

        PaddleOCR occasionally reads '>99 %' as '% 66<' (mirrored + 9→6, >→<).
        Reverse the string and apply mirror substitutions as a fallback.
        """
        s = raw.strip()
        m = re.match(r'^(>?[\d,]+)\s*%$', s)
        if m:
            return m.group(1)
        # mirrored: '% 66<' → reverse → '<66 %' → fix → '>99 %'
        if s.startswith('%'):
            rev = s[::-1].translate(str.maketrans('69<>', '96><')).strip()
            m2 = re.match(r'^(>?[\d,]+)\s*%$', rev)
            if m2:
                return m2.group(1)
        return None

    stim_a_pct = stim_v_pct = None
    past_diag = False
    for i, line in enumerate(lines):
        s = line.strip()
        if re.search(r'Diagnostik[üu]bersicht', s, re.IGNORECASE):
            past_diag = True
        if not past_diag:
            continue
        if s == "AP" and stim_a_pct is None:
            for j in range(i + 1, min(i + 5, len(lines))):
                v = _parse_pct_line(lines[j])
                if v is not None:
                    stim_a_pct = v
                    break
        elif s == "VP" and stim_v_pct is None:
            for j in range(i + 1, min(i + 5, len(lines))):
                v = _parse_pct_line(lines[j])
                if v is not None:
                    stim_v_pct = v
                    break
        if stim_a_pct and stim_v_pct:
            break

    # ── AT/AF ─────────────────────────────────────────────────────────────────
    # PaddleOCR splits "Mode-Switch" column header into fragments ("Mde","otch")
    # and "AT/AF-Burden" into "TAen" or similar. The percentage values follow each.
    # AMS episodes: standalone count after "AMS-Anfang" label row.
    ams_episodes = mode_switch_pct = af_burden_pct = None

    # AMS episode count: first standalone digit found after "AMS" label line
    for i, line in enumerate(diag_section):
        if re.search(r'AMS-Anfang', line.strip(), re.IGNORECASE):
            for j in range(i + 1, min(i + 4, len(diag_section))):
                if re.match(r'^\d+$', diag_section[j].strip()):
                    ams_episodes = diag_section[j].strip()
                    break
            break
    if ams_episodes is None:
        ams_episodes = "0"

    # Mode-Switch %: pct value near "Mode"/"Mde"/"otch" column header fragments
    for i, line in enumerate(diag_section):
        if re.search(r'^Mde$|^otch$|Mode.Switch', line.strip(), re.IGNORECASE):
            for j in range(i + 1, min(i + 5, len(diag_section))):
                s = diag_section[j].strip()
                m = re.match(r'^%?(\d+)%?$', s)
                if m and ('%' in s or s.isdigit()):
                    mode_switch_pct = m.group(1)
                    break
            if mode_switch_pct:
                break
    if mode_switch_pct is None:
        mode_switch_pct = "0"

    # AT/AF burden %: pct value near "TAen"/"AT/AF"/"ATIAF" fragments
    for i, line in enumerate(diag_section):
        if re.search(r'^TAen$|AT.AF|ATIAF', line.strip(), re.IGNORECASE):
            for j in range(i + 1, min(i + 5, len(diag_section))):
                s = diag_section[j].strip()
                m = re.match(r'^%?(\d+)%?$', s)
                if m and ('%' in s or s.isdigit()):
                    af_burden_pct = m.group(1)
                    break
            if af_burden_pct:
                break
    if af_burden_pct is None:
        af_burden_pct = "0"

    return {
        "report_type":           "FastPathSummary",
        "source_file":           str(pdf_path),
        "followup_date":         followup_date,
        "diagnostics_since":     diagnostics_since,
        "device_model":          device_model,
        "device_serial":         device_serial,
        "battery_voltage_v":     battery_voltage_v,
        "battery_mag_freq_bpm":  battery_mag_freq,
        "battery_remaining_pct": battery_remaining,
        "mode":                  mode,
        "base_rate_bpm":         base_rate,
        "upper_rate_bpm":        upper_rate,
        "av_paced_ms":           av_paced,
        "av_sensed_ms":          av_sensed,
        "autocapture_a":         autocapture_a,
        "autocapture_v":         autocapture_v,
        "amplitude_a_v":         amplitude_a,
        "amplitude_v_v":         amplitude_v,
        "pulse_width_a_ms":      pw_a,
        "pulse_width_v_ms":      pw_v,
        "sense_ability_a":       sense_a,
        "sense_ability_v":       sense_v,
        "sensitivity_a_mv":      sensitivity_a,
        "sensitivity_v_mv":      sensitivity_v,
        "stim_a_pct":            stim_a_pct,
        "stim_v_pct":            stim_v_pct,
        "ams_episodes":          ams_episodes,
        "mode_switch_pct":       mode_switch_pct,
        "af_burden_pct":         af_burden_pct,
    }
