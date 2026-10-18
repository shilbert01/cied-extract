"""
Parameter extractor for Abbott Aveir pacemaker reports.

Two OCR layouts handled:
  FreezeReport / Schlüsselparameter  →  inline "Key Value" on the same line
  SummaryReport sections             →  column: key-block then value-block
"""
import re
from typing import Optional
from models import (
    PacemakerReport, BatteryInfo, LeadTestResult,
    ProgrammedParameters, StimulationSettings, Diagnostics,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_section(lines: list[str], start_re: str,
                  end_re: str | None = None, max_lines: int = 60) -> list[str]:
    """Non-blank lines between the first start_re match and the first end_re match."""
    start = None
    for i, line in enumerate(lines):
        if re.search(start_re, line, re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return []
    result = []
    for line in lines[start: start + max_lines]:
        if end_re and re.search(end_re, line, re.IGNORECASE):
            break
        if line:
            result.append(line)
    return result


def _inline_value(line: str, key_pattern: str) -> Optional[str]:
    """Extract everything after key_pattern on the same line, stripped of punctuation."""
    m = re.match(key_pattern, line, re.IGNORECASE)
    if not m:
        return None
    rest = re.sub(r'^[\s.:,\-]+', '', line[m.end():]).strip()
    # Discard if rest is empty or only non-alphanumeric (e.g. "™)")
    return rest if rest and re.search(r'[\w\d]', rest) else None


# ── header ────────────────────────────────────────────────────────────────────

def _parse_header(report: PacemakerReport, lines: list[str]) -> None:
    # Report type — check before anything else to avoid false matches
    for line in lines:
        if 'FastPath' in line:
            report.report_type = 'SummaryReport'
            break
        if line == 'Ausschnitt' or 'Schlüsselparameter' in line:
            report.report_type = 'FreezeReport'
            break

    # Patient: line immediately after "Abbott"
    for i, line in enumerate(lines):
        if line == 'Abbott' and i + 1 < len(lines):
            candidate = lines[i + 1]
            if candidate and not re.match(r'^Aveir|^HZ\s|^\d|^FastPath|^Ausschnitt', candidate):
                report.patient = candidate
            break

    # Device: first line starting with "Aveir"
    for line in lines:
        if re.match(r'^Aveir', line):
            report.device = re.sub(r'\s{2,}.*', '', line).strip()
            break

    # Facility
    for line in lines:
        if re.match(r'^HZ\s+\w+', line):
            report.facility = line
            break

    # Date: standalone "dd Mon yyyy"
    for line in lines:
        if re.match(r'^\d{1,2}\s+\w+\s+\d{4}$', line):
            report.interrogation_date = line
            break

    # Visit type
    for line in lines:
        if 'Nachsorge' in line or 'Erstprogrammierung' in line:
            report.visit_type = line
            break


# ── battery (SummaryReport) ───────────────────────────────────────────────────

def _parse_battery(lines: list[str]) -> tuple[Optional[BatteryInfo], Optional[BatteryInfo]]:
    sect = _find_section(lines, r'Batterieinformation', r'Testergeb', max_lines=40)
    if not sect:
        return None, None

    laufzeiten = [m.group(1) for line in sect
                  for m in [re.search(r'Laufzeit:\s*([\d,\.]+\s*Jahre)', line)] if m]

    # RRT value is the line after the "RRT" label
    rrts = []
    for i, line in enumerate(sect):
        if line == 'RRT' and i + 1 < len(sect):
            rrts.append(sect[i + 1])

    # "Datum der Implantation" blocks: keys listed first, then the 4 values follow
    SKIP = re.compile(r'^(Spannung$|Batteriestrom$|Verbleibende|Laufzeit)')
    implant_blocks: list[list[str]] = []
    i = 0
    while i < len(sect):
        if sect[i] == 'Datum der Implantation':
            j, vals = i + 1, []
            while j < len(sect) and sect[j] != 'Datum der Implantation':
                if not SKIP.search(sect[j]):
                    vals.append(sect[j])
                j += 1
            implant_blocks.append(vals)
            i = j
        else:
            i += 1

    def make(longevity, rrt, vals) -> BatteryInfo:
        b = BatteryInfo()
        b.estimated_longevity = longevity
        b.rrt = rrt
        if vals:
            b.implant_date             = vals[0] if len(vals) > 0 else None
            b.voltage                  = vals[1] if len(vals) > 1 else None
            b.current_ua               = vals[2] if len(vals) > 2 else None
            b.remaining_capacity_pct   = vals[3] if len(vals) > 3 else None
        return b

    ra = make(laufzeiten[0] if laufzeiten         else None,
              rrts[0]       if rrts               else None,
              implant_blocks[0] if implant_blocks  else [])
    rv = make(laufzeiten[1] if len(laufzeiten) > 1 else None,
              rrts[1]       if len(rrts) > 1       else None,
              implant_blocks[1] if len(implant_blocks) > 1 else [])

    return (ra if ra.implant_date else None,
            rv if rv.implant_date else None)


# ── test results (SummaryReport) ──────────────────────────────────────────────

def _parse_test_results(lines: list[str]) -> tuple[Optional[LeadTestResult], Optional[LeadTestResult]]:
    sect = _find_section(lines, r'Testergeb', r'^Parameter', max_lines=50)
    if not sect:
        return None, None

    ra, rv = LeadTestResult(), LeadTestResult()

    # Threshold: lead label ("RA"/"RV") → Reizschwelle block → value on nearby line
    for lead, result in [('RA', ra), ('RV', rv)]:
        for i, line in enumerate(sect):
            if line == lead:
                for j in range(i + 1, min(i + 6, len(sect))):
                    m = re.search(r'([\d,\.]+\s*V\s*@\s*[\d,\.]+\s*ms)', sect[j])
                    if m:
                        result.pacing_threshold = m.group(1)
                        dm = re.search(r'\((\d{1,2}\s+\w+\s+\d{4})\)', sect[j])
                        result.pacing_threshold_date = dm.group(1) if dm else None
                        break
                break

    # Wahrnehmung and Impedanz: first occurrence = RA, second = RV
    for label, attr, val_re in [
        ('Wahrnehmung', 'sensing_mv',    r'([\d,\.]+\s*mV)'),
        ('Impedanz',    'impedance_ohm', r'([\d]+\s*Ω)'),
    ]:
        hits = []
        for i, line in enumerate(sect):
            if line == label:
                for j in range(i + 1, min(i + 4, len(sect))):
                    m = re.search(val_re, sect[j])
                    if m:
                        dm = re.search(r'\((\d{1,2}\s+\w+\s+\d{4})\)', sect[j])
                        hits.append((m.group(1), dm.group(1) if dm else None))
                        break
        date_attr = re.sub(r'_(mv|ohm)$', '_date', attr)
        if hits:
            setattr(ra, attr, hits[0][0]); setattr(ra, date_attr, hits[0][1])
        if len(hits) > 1:
            setattr(rv, attr, hits[1][0]); setattr(rv, date_attr, hits[1][1])

    has_ra = any([ra.pacing_threshold, ra.sensing_mv, ra.impedance_ohm])
    has_rv = any([rv.pacing_threshold, rv.sensing_mv, rv.impedance_ohm])
    return ra if has_ra else None, rv if has_rv else None


# ── parameters ────────────────────────────────────────────────────────────────

PARAM_KEY_DEFS = [
    ('mode',                     r'^Betriebsart'),
    ('base_rate',                r'^Grundfrq'),
    ('upper_rate_limit',         r'^Obere\s+Grenzfrq'),
    ('max_sensor_rate',          r'^Max\.?\s*Sensorfrq'),
    ('av_paced_ms',              r'^AV\s+stimuliert'),
    ('av_sensed_ms',             r'^AV\s+wahrgen'),
    ('rate_hysteresis',          r'^Frequenzhysteresedif'),
    # Patterns must consume the full key so _inline_value sees only the value
    ('vip_enabled',              r'^Ventricular Intrinsic Preference[^)]*\)'),
    ('rate_adaptive_av',         r'^Frequenzabh\.\s*AV-Intervall'),
    ('rate_adaptive_pvarp',      r'^Frequenzabh\.\s*PVARP\S*'),           # handles PVARP/VREF and OCR typo PVARP/REF
    ('ventricular_safety_pacing',r'^Vent\.\s+Sicherheits-Stim\.'),
]

_PARAM_SKIP = re.compile(
    r'^(i2i-|Parameter$|Schlüssel|FastPath|Batterie|Testergeb|Stimulation'
    r'|Diagnostik|Seit:|S\.\s+\d|Ausschnitt)',
    re.IGNORECASE
)


def _parse_parameters(lines: list[str], report_type: Optional[str]) -> Optional[ProgrammedParameters]:
    sect = _find_section(
        lines,
        r'^(Parameter|Schlüsselparameter)',
        r'^(Stimulation|i2i-Diagnostik|Diagnostik|Ausschnitt\s+\d)',
        max_lines=40,
    )
    if not sect:
        return None

    params = ProgrammedParameters()

    if report_type == 'FreezeReport':
        # Inline: "Betriebsart DDD", "Grundfrq. 50 min¹"
        for line in sect:
            for field, pattern in PARAM_KEY_DEFS:
                val = _inline_value(line, pattern)
                if val is not None:
                    setattr(params, field, val)
                    break
    else:
        # Column: key block then value block
        found_keys, value_lines = [], []
        for line in sect:
            matched = False
            for field, pattern in PARAM_KEY_DEFS:
                if re.search(pattern, line, re.IGNORECASE):
                    found_keys.append(field)
                    matched = True
                    break
            if not matched and not _PARAM_SKIP.search(line):
                value_lines.append(line)
        for i, field in enumerate(found_keys):
            if i < len(value_lines):
                setattr(params, field, value_lines[i])

    return params if params.mode else None


# ── stimulation (SummaryReport) ───────────────────────────────────────────────

STIM_KEY_DEFS = [
    ('configuration',  r'^Elektrodenkonfig'),
    ('amplitude_v',    r'^Impulsampl'),
    ('pulse_width_ms', r'^Impuls(dauer|duer)'),
    ('senseability',   r'^SenseAbility'),
    ('sensitivity_mv', r'^Sensitiv'),
]

_STIM_SKIP = re.compile(r'^(RA$|RV$|Stimulation)', re.IGNORECASE)


def _parse_stimulation(lines: list[str]) -> tuple[Optional[StimulationSettings], Optional[StimulationSettings]]:
    sect = _find_section(
        lines,
        r'^Stimulation\s*&',
        r'^(Diagnostik|Stimuliert\s*&|i2i-Diagnostik)',
        max_lines=25,
    )
    if not sect:
        return None, None

    found_keys, value_lines = [], []
    for line in sect:
        matched = False
        for field, pattern in STIM_KEY_DEFS:
            if re.search(pattern, line, re.IGNORECASE):
                found_keys.append(field)
                matched = True
                break
        if not matched and not _STIM_SKIP.search(line):
            value_lines.append(line)

    n = len(found_keys)
    ra_vals = value_lines[:n]
    rv_vals = value_lines[n:]

    # OCR sometimes merges "Bipolar Bipolar" → "Bipolar" → RV is short by 1
    if ra_vals and len(rv_vals) == n - 1:
        rv_vals.insert(0, ra_vals[0])

    ra, rv = StimulationSettings(), StimulationSettings()
    for i, field in enumerate(found_keys):
        if i < len(ra_vals):
            setattr(ra, field, ra_vals[i])
        if i < len(rv_vals):
            setattr(rv, field, rv_vals[i])

    return (ra if ra.configuration else None,
            rv if rv.amplitude_v   else None)


# ── diagnostics (SummaryReport) ───────────────────────────────────────────────

def _parse_diagnostics(lines: list[str]) -> Optional[Diagnostics]:
    sect = _find_section(
        lines,
        r'^(Diagnostik|Stimuliert\s*&\s*Wahr)',
        max_lines=25,
    )
    if not sect:
        return None

    diag = Diagnostics()

    for line in sect:
        m = re.match(r'Seit:\s*(.+)', line)
        if m:
            diag.since = m.group(1).strip()
            break

    pcts = re.findall(r'[<>]?\d+\s*%', '\n'.join(sect))
    if len(pcts) > 0: diag.ap_paced_pct    = pcts[0]
    if len(pcts) > 1: diag.vp_paced_pct    = pcts[1]
    if len(pcts) > 2: diag.mode_switch_pct = pcts[2]

    for line in sect:
        if re.match(r'^\d+$', line):
            diag.ams_episodes = line
            break

    return diag if diag.ap_paced_pct else None


# ── entry point ───────────────────────────────────────────────────────────────

def extract(ocr_pages: list[str], source_file: str = "") -> PacemakerReport:
    full_text = "\n".join(ocr_pages)
    lines = [l.strip() for l in full_text.split('\n')]

    report = PacemakerReport(source_file=source_file, raw_ocr_pages=ocr_pages)

    _parse_header(report, lines)
    report.battery_ra, report.battery_rv   = _parse_battery(lines)
    report.test_ra,    report.test_rv      = _parse_test_results(lines)
    report.parameters                      = _parse_parameters(lines, report.report_type)
    report.stimulation_ra, report.stimulation_rv = _parse_stimulation(lines)
    report.diagnostics                     = _parse_diagnostics(lines)

    return report
