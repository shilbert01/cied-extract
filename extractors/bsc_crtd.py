"""
Boston Scientific INOGEN CRT-D extractor.

Uses pdfplumber spatial word extraction to handle the three-column layout:
  Implantation | Vorh. Sitzung | Letzte

The "Letzte" (most recent) column is at x ≥ 475 and contains the current values.
Settings (mode, rates, stimulation energy) are extracted via text regex.
"""
import re
from pathlib import Path
from typing import Optional

import pdfplumber


# ── column x-position ranges ──────────────────────────────────────────────────
_COL_LABEL = (0,    230)
_COL_IMPL  = (230,  350)
_COL_PREV  = (350,  475)
_COL_LAST  = (475,  600)


def _rows(page) -> list[list]:
    words = page.extract_words(keep_blank_chars=False)
    buckets: dict[int, list] = {}
    for w in words:
        bucket = (round(w["top"]) // 2) * 2
        buckets.setdefault(bucket, []).append(w)
    return [sorted(ws, key=lambda w: w["x0"]) for ws in sorted(buckets.values(), key=lambda ws: ws[0]["top"])]


def _label(row: list) -> str:
    return " ".join(w["text"] for w in row if w["x0"] < _COL_LABEL[1])


def _last_col_text(row: list) -> str:
    """Concatenate all words in the Letzte column."""
    return " ".join(w["text"] for w in row if w["x0"] >= _COL_LAST[0])


def _find_row(rows: list, pattern: str) -> Optional[list]:
    for row in rows:
        if re.search(pattern, _label(row), re.IGNORECASE):
            return row
    return None


def _de_num(s: str) -> str:
    """Normalize German decimal comma to dot: '1,9' → '1.9'."""
    return s.replace(",", ".")


def _parse_impedance(row: list) -> Optional[str]:
    """Extract 'Letzte' column impedance value (digits before any unit suffix)."""
    txt = _last_col_text(row).strip()
    m = re.match(r"([\d,]+)", txt)
    return _de_num(m.group(1)) if m else None


def _parse_threshold(row: list) -> tuple[Optional[str], Optional[str]]:
    """Extract 'Letzte' column threshold as (voltage, pulse_width_ms)."""
    words = [w for w in row if w["x0"] >= _COL_LAST[0]]
    tokens = [w["text"] for w in words]
    joined = " ".join(tokens)
    # Format: "0,9 V @ 0,5 ms" or "1,1V@0,5ms"
    m = re.search(r"([\d,]+)\s*V\s*@\s*([\d,]+)\s*ms", joined)
    if m:
        return _de_num(m.group(1)), _de_num(m.group(2))
    return None, None


def _parse_amplitude(row: list) -> Optional[str]:
    """Extract 'Letzte' column amplitude (mV), stripping unit."""
    txt = _last_col_text(row)
    m = re.search(r"([\d,]+)\s*mV", txt)
    return _de_num(m.group(1)) if m else None


# ── text-based extraction for settings and stats ──────────────────────────────

def _find(pattern: str, text: str, group: int = 1, flags: int = re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _parse_settings(text: str) -> dict:
    out: dict = {}
    # "Modus DDDR" is in Brady/CRT section; "Tachy-Modus" is separate — anchor to avoid
    out["mode"] = _find(r"(?<!Tachy-)Modus\s+([A-Z]{2,6})", text)
    out["base_rate_bpm"] = _find(r"Untere\s+Grenzfrequenz\s+([\d,]+)min", text)
    if out.get("base_rate_bpm"):
        out["base_rate_bpm"] = _de_num(out["base_rate_bpm"])
    out["max_rate_bpm"] = _find(r"Max\.\s+Trackingfrequenz\s+([\d,]+)min", text)
    if out.get("max_rate_bpm"):
        out["max_rate_bpm"] = _de_num(out["max_rate_bpm"])

    # Stimulation energy: "Atrial 2,2V@0,5ms" and "Rechtsventrikulär 2,0V@0,5ms"
    m = re.search(r"Atrial\s+([\d,]+)V@([\d,]+)ms", text)
    if m:
        out["amplitude_a_v"] = _de_num(m.group(1))
        out["pulse_width_a_ms"] = _de_num(m.group(2))
    m = re.search(r"Rechtsventrikulär\s+([\d,]+)V@([\d,]+)ms", text)
    if m:
        out["amplitude_rv_v"] = _de_num(m.group(1))
        out["pulse_width_rv_ms"] = _de_num(m.group(2))

    # Tachy zones
    out["vf_bpm"] = _find(r"\bVF\s+([\d,]+)min", text)
    out["vt_bpm"] = _find(r"\bVT\s+([\d,]+)min", text)
    out["vt1_bpm"] = _find(r"\bVT-1\s+([\d,]+)min", text)
    out["atr_switch_bpm"] = _find(r"ATR\s+Mode\s+Switch\s+([\d,]+)min", text)

    return {k: v for k, v in out.items() if v is not None}


def _parse_stats(text: str) -> dict:
    out: dict = {}
    # "% A stimuliert 91 100" — take the "Seit ltzt. Reset" (second) value
    m = re.search(r"%\s+A\s+stimuliert\s+(\S+)\s+(\S+)", text)
    if m:
        out["stim_a_pct"] = m.group(2)
    m = re.search(r"%\s+RV\s+stimuliert\s+(\S+)\s+(\S+)", text)
    if m:
        out["stim_rv_pct"] = m.group(2)
    m = re.search(r"%\s+LV\s+stimuliert\s+(\S+)\s+(\S+)", text)
    if m:
        out["stim_lv_pct"] = m.group(2)

    # AT/AF burden
    out["af_burden_pct"] = _find(r"%\s+AT/AF\s+([<\d,.]+)", text)

    return {k: v for k, v in out.items() if v is not None}


# ── entry point ────────────────────────────────────────────────────────────────

def extract(pdf_path: str | Path) -> dict:
    """Extract clinical parameters from a BSC INOGEN CRT-D DeviceFollowUpReport."""
    result: dict = {
        "source_file": str(pdf_path),
        "manufacturer": "Boston Scientific",
        "report_type": "DeviceFollowUpReport",
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
        page1_rows = _rows(pdf.pages[0])

    p1 = pages[0]

    # Patient + device header (text-based, page 1)
    # "Rudolph, Rüdiger" — name is first non-header line
    m = re.search(r"^([A-ZÄÖÜa-zäöüß]+,\s+[A-ZÄÖÜa-zäöüß]+)\s+Letzte", p1, re.MULTILINE)
    if m:
        result["patient_name"] = m.group(1)

    m = re.search(r"Geburtsdatum\s+(\d{1,2})(\w{3})(\d{4})", p1)
    if m:
        months = {"Jan":"01","Feb":"02","Mär":"03","Mar":"03","Apr":"04","Mai":"05","Jun":"06",
                  "Jul":"07","Aug":"08","Sep":"09","Okt":"10","Nov":"11","Dez":"12"}
        mon = months.get(m.group(2), m.group(2))
        result["patient_dob"] = f"{m.group(1).zfill(2)}.{mon}.{m.group(3)}"

    # Device model + serial — two-column: "INOGEN CRT-DG141/ 140837 Dat. Implant."
    m = re.search(r"Aggregat\s+(INOGEN\s+CRT-D\w+)\s*/\s*(\d+)", p1)
    if m:
        result["device_model"] = m.group(1)   # keep original spacing
        result["device_serial"] = m.group(2)

    months_de = {"Jan":"01","Feb":"02","Mär":"03","Mar":"03","Apr":"04","Mai":"05","Jun":"06",
                 "Jul":"07","Aug":"08","Sep":"09","Okt":"10","Nov":"11","Dez":"12"}

    def _parse_date(raw: str) -> str:
        parts = raw.strip().split()
        mon = months_de.get(parts[1], parts[1])
        return f"{parts[0].zfill(2)}.{mon}.{parts[2]}"

    # Implant date: right-column value on the Tachy-Modus row
    # Text: "Tachy-Modus Überw.+Therapie 11 Feb 2021"
    m = re.search(r"Tachy-Modus\s+\S+\s+(\d{1,2}\s+\w+\s+\d{4})", p1)
    if m:
        result["implant_date"] = _parse_date(m.group(1))

    # Follow-up date: right-column value on the Geburtsdatum row
    # Text: "Geburtsdatum 20Mai1955 28 Mai 2026"
    m = re.search(r"Geburtsdatum\s+\S+\s+(\d{1,2}\s+\w+\s+\d{4})", p1)
    if m:
        result["followup_date"] = _parse_date(m.group(1))

    # Battery
    result["battery_life"] = _find(r"Ungefähre Zeit bis zur Explantation:\s+(.+?)$", p1,
                                    flags=re.MULTILINE)
    result["charge_time_s"] = _find(r"Ladezeit\s+([\d,]+)s", p1)
    if result.get("charge_time_s"):
        result["charge_time_s"] = _de_num(result["charge_time_s"])

    # Electrode data from spatial three-column table (page 1)
    def _lead_section(rows, heading_pattern):
        """Find rows belonging to a lead section (between two headings)."""
        in_section = False
        section_rows = []
        for row in rows:
            lbl = _label(row)
            if re.search(heading_pattern, lbl, re.IGNORECASE):
                in_section = True
                continue
            if in_section:
                if re.search(r"^(Rechtsventrikulär|Linksventrikulär|Schockvektor|Einstellungen)", lbl, re.IGNORECASE):
                    break
                section_rows.append(row)
        return section_rows

    a_rows  = _lead_section(page1_rows, r"^Atrial$")
    rv_rows = _lead_section(page1_rows, r"^Rechtsventrikulär$")
    lv_rows = _lead_section(page1_rows, r"^Linksventrikulär$")

    def _extract_lead(rows, prefix):
        row = _find_row(rows, r"Stimulation-Impedanz")
        if row:
            result[f"impedance_{prefix}_ohm"] = _parse_impedance(row)
        row = _find_row(rows, r"Stim\.-Reizschwelle")
        if row:
            v, pw = _parse_threshold(row)
            result[f"threshold_{prefix}_v"] = v
            result[f"threshold_{prefix}_pulse_ms"] = pw
        row = _find_row(rows, r"Intrins\.Ampl\.")
        if row:
            result[f"sensing_{prefix}_mv"] = _parse_amplitude(row)

    _extract_lead(a_rows, "a")
    _extract_lead(rv_rows, "rv")
    _extract_lead(lv_rows, "lv")

    # Shock impedance
    shock_row = _find_row(page1_rows, r"Schock-Impedanz")
    if shock_row:
        result["shock_impedance_ohm"] = _parse_impedance(shock_row)

    # Settings (text-based, page 1)
    result.update(_parse_settings(p1))

    # Statistics (page 3, index 2)
    if len(pages) > 2:
        result.update(_parse_stats(pages[2]))

    return {k: v for k, v in result.items() if v is not None}
