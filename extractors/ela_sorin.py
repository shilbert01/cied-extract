"""
ELA Medical / Sorin / MicroPort CRM SYNERGY report extractor.

Uses pdfplumber text extraction (no OCR needed — text-based PDFs).
Two supported device classes share the same SYNERGY report format:
  - CRT-D (e.g., ENERGY A): patient+device on page 2, tests on page 5, params on page 6
  - Pacemaker (e.g., TEO DR): patient+device on page 1, params on page 2, tests on page 3

Device class is auto-detected from report content.
"""
import re
from pathlib import Path
from typing import Optional

import pdfplumber


# ── helpers ───────────────────────────────────────────────────────────────────

def _find(pattern: str, text: str, group: int = 1, flags: int = re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _month_de(s: str) -> str:
    """Normalise German/abbreviated month names to two-digit month number."""
    months = {
        "jan": "01", "januar": "01",
        "feb": "02", "februar": "02",
        "mär": "03", "mar": "03", "märz": "03",
        "apr": "04", "april": "04",
        "mai": "05",
        "jun": "06", "juni": "06",
        "jul": "07", "juli": "07",
        "aug": "08", "august": "08",
        "sep": "09", "sept": "09", "september": "09",
        "okt": "10", "oktober": "10",
        "nov": "11", "november": "11",
        "dez": "12", "dezember": "12",
    }
    return months.get(s.lower(), s)


def _norm_date(raw: Optional[str]) -> Optional[str]:
    """Convert 'D/Mon/YYYY' or 'D/Monat/YYYY' → 'DD.MM.YYYY'."""
    if not raw:
        return None
    m = re.match(r"(\d{1,2})/(\w+)/(\d{4})", raw.strip())
    if not m:
        return raw.strip()
    day = m.group(1).zfill(2)
    mon = _month_de(m.group(2))
    return f"{day}.{mon}.{m.group(3)}"


# ── patient / device header ───────────────────────────────────────────────────

def _parse_patient_crtd(text: str) -> dict:
    """Energy A CRT-D: page 2 has 'Nachname X' and 'Vorname Y' separately."""
    out: dict = {}
    last = _find(r"Nachname\s+(.+)", text)
    first = _find(r"Vorname\s+(.+)", text)
    if last and first:
        out["patient_name"] = f"{last}, {first}"
    elif last:
        out["patient_name"] = last
    dob = _find(r"Geburtsdatum\s+(\d{1,2}/\w+/\d{4})", text)
    out["patient_dob"] = _norm_date(dob)
    out["device_model"] = _find(r"^Modell\s+(.+)", text, flags=re.MULTILINE | re.IGNORECASE)
    out["device_serial"] = _find(r"^Seriennummer\s+(\S+)", text, flags=re.MULTILINE | re.IGNORECASE)
    impl = _find(r"^Implantation\s+(\d{1,2}/\w+/\d{4})", text, flags=re.MULTILINE | re.IGNORECASE)
    out["implant_date"] = _norm_date(impl)
    # Battery status
    out["battery_voltage_v"] = _find(r"STATUS\s+Spannung\s+([\d.]+)\s*V", text)
    out["battery_life"] = _find(r"Zeit bis RRT\s+(.+?)$", text, flags=re.MULTILINE)
    out["last_shock_impedance_ohm"] = _find(r"Letzte Schockimp\.\s+([\d.]+)\s*Ohm", text)
    shock_date = _find(r"Letzter Schock.*?Datum\s+(\d{1,2}/\w+/\d{4})", text, flags=re.DOTALL)
    out["last_shock_date"] = _norm_date(shock_date)
    return {k: v for k, v in out.items() if v is not None}


def _parse_patient_pm(text: str) -> dict:
    """TEO DR pacemaker: page 1 has 'Name X, Y' in two-column layout."""
    out: dict = {}
    out["patient_name"] = _find(r"^Name\s+(.+)", text, flags=re.MULTILINE | re.IGNORECASE)
    dob = _find(r"Geburtsdatum\s+(\d{1,2}/\w+/\d{4})", text)
    out["patient_dob"] = _norm_date(dob)

    # Device section follows "Schrittmacher" heading
    dev_section = re.split(r"Schrittmacher", text, maxsplit=1, flags=re.IGNORECASE)
    src = dev_section[1] if len(dev_section) > 1 else text
    model_raw = _find(r"^Modell\s+(.+)", src, flags=re.MULTILINE | re.IGNORECASE)
    # Right-column observations (e.g. "stillgelegte RV Elektrode") start with a lowercase word
    if model_raw:
        model_raw = re.sub(r"\s+[a-z]\S.*$", "", model_raw).strip()
    out["device_model"] = model_raw
    out["device_serial"] = _find(r"^Seriennummer\s+(\S+)", src, flags=re.MULTILINE | re.IGNORECASE)
    impl = _find(r"^Implantation\s+(\d{1,2}/\w+/\d{4})", src, flags=re.MULTILINE | re.IGNORECASE)
    out["implant_date"] = _norm_date(impl)
    return {k: v for k, v in out.items() if v is not None}


# ── parameters (mode, rates, stimulation) ─────────────────────────────────────

def _parse_params(text: str) -> dict:
    """Extract from the Basisparameter section (shared format, both device types)."""
    out: dict = {}
    # "Modus DDDR" may appear mid-line in CRT-D due to two-column layout
    out["mode"] = _find(r"\bModus\s+([A-Z]{2,6})", text, flags=re.IGNORECASE)
    # Handles both "Basisfrequenz 60 min-1" (PM) and "Basisfreq. 90 min-1" (CRT-D)
    out["base_rate_bpm"] = _find(r"Basisfreq(?:uenz)?\s*\.?\s+(\d+)\s+min", text)
    out["max_rate_bpm"] = _find(r"Max\.\s+Frequenz\s+(\d+)\s+min", text)

    # PM: "A Stimulation 2.5 V / 0.35 ms / bipolar"
    m = re.search(r"^A\s+Stimulation\s+([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms", text,
                  re.MULTILINE | re.IGNORECASE)
    if m:
        out["amplitude_a_v"] = m.group(1)
        out["pulse_width_a_ms"] = m.group(2)

    # CRT-D: "A Ampl. 2.5 V" and "A-Impulssbreite 0.35 ms" (separate lines, right column)
    if not out.get("amplitude_a_v"):
        out["amplitude_a_v"] = _find(r"\bA\s+Ampl\.\s+([\d.]+)\s*V", text)
    if not out.get("pulse_width_a_ms"):
        out["pulse_width_a_ms"] = _find(r"A-Impulss?breite\s+([\d.]+)\s*ms", text)

    # PM: "V Stimulation 3.5 V / 0.35 ms" — use rv_ prefix for adapter compatibility
    m = re.search(r"^V\s+Stimulation\s+([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms", text,
                  re.MULTILINE | re.IGNORECASE)
    if m:
        out["amplitude_rv_v"] = m.group(1)
        out["pulse_width_rv_ms"] = m.group(2)

    # CRT-D: RV "RV-Stimulationsamplitude X V" + "RV-Impulsbreite Y ms"
    if not out.get("amplitude_rv_v"):
        out["amplitude_rv_v"] = _find(r"RV-Stimulationsamplitude\s+([\d.]+)\s*V", text)
    if not out.get("pulse_width_rv_ms"):
        out["pulse_width_rv_ms"] = _find(r"RV-Impulsbreite\s+([\d.]+)\s*ms", text)

    # CRT-D: LV "LV Ampl. / Dauer 2.5 V / 1.50 ms"
    m = re.search(r"LV\s+Ampl\.\s*/\s*Dauer\s+([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms", text, re.IGNORECASE)
    if m:
        out["amplitude_lv_v"] = m.group(1)
        out["pulse_width_lv_ms"] = m.group(2)

    # Sensing (programmed, from parameter section)
    out["sensing_prog_a_mv"] = _find(r"A\s+Empfindlichkeit\s+([\d.]+)\s*mV", text)
    out["sensing_prog_rv_mv"] = _find(r"^V\s+Empfindlichkeit\s+([\d.]+)\s*mV", text,
                                       flags=re.MULTILINE | re.IGNORECASE)
    if not out.get("sensing_prog_rv_mv"):
        out["sensing_prog_rv_mv"] = _find(r"RV\s+Empfindlichkeit\s+([\d.]+)\s*mV", text)

    # Battery (pacemaker page 2)
    m = re.search(r"Impedanz\s+([\d.]+)\s*kOhm", text, re.IGNORECASE)
    if m:
        out["battery_impedance_kohm"] = m.group(1)
    # Capture only "X Jahre" or "> X Jahre", not trailing right-column text
    out["battery_life"] = _find(r"Restlaufzeit\s+(.+?Jahre)", text)

    return {k: v for k, v in out.items() if v is not None}


# ── test results (measured impedances, thresholds, sensing) ──────────────────

def _parse_tests_crtd(text: str) -> dict:
    """Energy A CRT-D page 5: two-column layout causes mixed line content."""
    out: dict = {}

    # A and RV: "372 Ohm [right-col junk]\nImpedanz A-Elektrode"
    out["impedance_a_ohm"] = _find(
        r"([\d]+)\s+Ohm\b[^\n]*\nImpedanz\s+A-Elektrode", text)
    out["impedance_rv_ohm"] = _find(
        r"([\d]+)\s+Ohm\b[^\n]*\nImpedanz\s+RV-Elektrode", text)
    # LV: "Impedanz LV-Elektrode 626 Ohm [right-col junk]"
    out["impedance_lv_ohm"] = _find(
        r"Impedanz\s+LV-Elektrode\s+([\d]+)\s+Ohm", text)

    # A threshold: "0.75 V / 0.35 ms\nA-Elektrode date"
    m = re.search(r"([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms\s+A-Elektrode", text)
    if m:
        out["threshold_a_v"] = m.group(1)
        out["threshold_a_pulse_ms"] = m.group(2)
    # RV: "1.00 V / 1.00 ms [right-col junk]\nRV-Elektrode"
    m = re.search(r"([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms[^\n]*\nRV-Elektrode", text)
    if m:
        out["threshold_rv_v"] = m.group(1)
        out["threshold_rv_pulse_ms"] = m.group(2)
    # LV: "LV-Elektrode 1.50 V / 1.50 ms [right-col junk]"
    m = re.search(r"LV-Elektrode\s+([\d.]+)\s*V\s*/\s*([\d.]+)\s*ms", text)
    if m:
        out["threshold_lv_v"] = m.group(1)
        out["threshold_lv_pulse_ms"] = m.group(2)

    # Sensing — value appears before label due to two-column layout
    # "0.4 mV [right-col junk]\nWahrnehmung A"
    m = re.search(r"([\d.]+)\s*mV\b[^\n]*\nWahrnehmung\s+A", text)
    if m:
        out["sensing_a_mv"] = m.group(1)
    # "8.5 mV\nWahrnehmung V ..."
    m = re.search(r"([\d.]+)\s*mV\s*\nWahrnehmung\s+V", text)
    if m:
        out["sensing_rv_mv"] = m.group(1)

    # Followup date = latest measurement date on this page
    dates = re.findall(r"(\d{1,2}/\w+/\d{4})", text)
    if dates:
        out["followup_date"] = _norm_date(sorted(dates)[-1])

    return {k: v for k, v in out.items() if v is not None}


def _parse_tests_pm(text: str) -> dict:
    """TEO DR pacemaker page 3: two-column layout, values on separate lines."""
    out: dict = {}

    # Layout: "Impedanz\nA V\n387 Ohm 567 Ohm\nImpedanz\nDATE DATE"
    # pdfplumber merges both impedance values onto one line: "387 Ohm 567 Ohm"
    m = re.search(r"Impedanz\s+A\s+V\s+([\d.]+)\s+Ohm\s+([\d.]+)\s+Ohm", text, re.IGNORECASE)
    if m:
        out["impedance_a_ohm"] = m.group(1)
        out["impedance_rv_ohm"] = m.group(2)

    # "Reizschwelle\nA V\n0.25 V 0.25 V"
    m = re.search(r"Reizschwelle\s+A\s+V\s+([\d.]+)\s+V\s+([\d.]+)\s+V", text, re.IGNORECASE)
    if m:
        out["threshold_a_v"] = m.group(1)
        out["threshold_rv_v"] = m.group(2)

    # "Wahrnehm.\nA V\n1.30 mV -"
    m = re.search(r"Wahrnehm\.\s+A\s+V\s+([\d.]+)\s+mV\s+(-|\S+)", text, re.IGNORECASE)
    if m:
        out["sensing_a_mv"] = m.group(1)
        v_sense = m.group(2)
        out["sensing_rv_mv"] = None if v_sense == "-" else v_sense

    # Test date (e.g., "28/Juni/2026")
    dates = re.findall(r"\d{1,2}/\w+/\d{4}", text)
    non_stat = [d for d in dates if "Statistiken" not in text[max(0, text.index(d)-20):text.index(d)]]
    if dates:
        # Use last unique date (test date, not stats period start)
        out["followup_date"] = _norm_date(dates[-1])

    return out


# ── entry point ───────────────────────────────────────────────────────────────

def extract(pdf_path: str | Path) -> dict:
    """Extract clinical parameters from an ELA/Sorin SYNERGY report PDF."""
    result: dict = {
        "source_file": str(pdf_path),
        "manufacturer": "ELA Medical / Sorin",
        "report_type": "FollowUpReport",
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]

    # Detect CRT-D vs pacemaker
    full_text = "\n".join(pages)
    is_crtd = bool(re.search(r"CRT-D|Defibrillator|Schock|ICD", full_text, re.IGNORECASE))

    if is_crtd:
        # Page 2 (index 1): patient + device + battery
        result.update(_parse_patient_crtd(pages[1]))
        # Page 5 (index 4): test results
        if len(pages) > 4:
            result.update(_parse_tests_crtd(pages[4]))
        # Page 6 (index 5): mode, rates, stimulation
        if len(pages) > 5:
            result.update(_parse_params(pages[5]))
    else:
        # Page 1 (index 0): patient + device
        result.update(_parse_patient_pm(pages[0]))
        # Page 2 (index 1): parameters + battery
        if len(pages) > 1:
            result.update(_parse_params(pages[1]))
        # Page 3 (index 2): test results + followup date
        if len(pages) > 2:
            result.update(_parse_tests_pm(pages[2]))

    return result
