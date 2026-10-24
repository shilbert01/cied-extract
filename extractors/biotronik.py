"""
Biotronik BIOSTD report extractor.

Uses pdfplumber text + spatial (x-position) extraction to handle
multi-column A/RV/LV layouts without needing OCR.

Target: page 1 (Implantatstatus summary) and page 3 (Nachsorge 1. Abfrage).
Page 3 is primary — it has battery voltage, full test results, and
stimulation percentages. Page 1 supplements where page 3 is missing data.
"""
import re
from pathlib import Path
from typing import Optional

import pdfplumber


# ── column x-position ranges (derived from spatial analysis of BIOSTD layout) ─
# Values in each row are placed at fixed x positions for A / RV / LV columns.
# Labels always start at x ≈ 65.
_COL_LABEL = (0,   310)
_COL_A     = (310, 375)
_COL_RV    = (375, 455)
_COL_LV    = (455, 560)


def _col(x: float) -> Optional[str]:
    if _COL_A[0] <= x < _COL_A[1]:
        return "a"
    if _COL_RV[0] <= x < _COL_RV[1]:
        return "rv"
    if _COL_LV[0] <= x < _COL_LV[1]:
        return "lv"
    return None


def _rows(page) -> list[dict]:
    """Group words into rows by y position (±2 pt tolerance)."""
    words = page.extract_words(keep_blank_chars=False)
    rows: dict[int, list] = {}
    for w in words:
        y = round(w["top"])
        # Snap to nearest 2-pt bucket
        bucket = (y // 2) * 2
        rows.setdefault(bucket, []).append(w)
    return [sorted(ws, key=lambda w: w["x0"]) for ws in sorted(rows.values(), key=lambda ws: ws[0]["top"])]


def _label(row: list) -> str:
    """Concatenate all words in the label column of a row."""
    return " ".join(w["text"] for w in row if w["x0"] < _COL_LABEL[1])


def _values(row: list) -> dict[str, str]:
    """Return {column_key: value} for value-column words, ignoring '-----'."""
    out: dict[str, str] = {}
    pending: dict[str, list] = {}  # multi-word values like "LV1 → LV2"
    for w in row:
        if w["x0"] < _COL_LABEL[1]:
            continue
        col = _col(w["x0"])
        if col is None:
            continue
        if w["text"] == "-----":
            continue
        pending.setdefault(col, []).append(w["text"])
    for col, tokens in pending.items():
        out[col] = " ".join(tokens)
    return out


def _find_row(rows: list, pattern: str) -> Optional[list]:
    """Return first row whose label matches pattern (case-insensitive)."""
    for row in rows:
        if re.search(pattern, _label(row), re.IGNORECASE):
            return row
    return None


def _scalar(rows: list, pattern: str) -> Optional[str]:
    """Return the first value from the first matching row (any column)."""
    row = _find_row(rows, pattern)
    if row is None:
        return None
    vals = _values(row)
    for col in ("a", "rv", "lv"):
        if col in vals:
            return vals[col]
    # Fall back: any word right of the label column
    for w in row:
        if w["x0"] >= _COL_LABEL[1] and w["text"] not in ("-----", "/"):
            return w["text"]
    return None


def _3col(rows: list, pattern: str) -> dict[str, Optional[str]]:
    """Return {'a': v, 'rv': v, 'lv': v} for a multi-column row."""
    row = _find_row(rows, pattern)
    if row is None:
        return {"a": None, "rv": None, "lv": None}
    vals = _values(row)
    return {"a": vals.get("a"), "rv": vals.get("rv"), "lv": vals.get("lv")}


def _slash_values(rows: list, pattern: str) -> list[Optional[str]]:
    """Extract slash-separated values like '150 / 182 / 231' or '----- / 10 / 89 / 99'."""
    row = _find_row(rows, pattern)
    if row is None:
        return []
    # Collect all tokens right of label, reconstruct as string, split on /
    tokens = [w["text"] for w in row if w["x0"] >= _COL_LABEL[1]]
    joined = " ".join(tokens)
    parts = [p.strip() for p in joined.split("/")]
    return [None if p == "-----" else p for p in parts]


# ── header parser (shared across all pages) ──────────────────────────────────

def _parse_header(text: str) -> dict:
    """Extract fields from the repeating page header block."""
    out: dict = {}

    # Patient name (first line: "Lastname, Firstname")
    first_line = text.split("\n")[0].strip()
    if "," in first_line:
        out["patient_name"] = first_line

    # DOB
    m = re.search(r"Geburtsdatum:\s*(\d{2}\.\d{2}\.\d{4})", text)
    if m:
        out["patient_dob"] = m.group(1)

    # Current follow-up date, device, serial, programmer from "Aktuelle Nachsorge:" line
    m = re.search(
        r"Aktuelle Nachsorge:\s*([\d.]+)\s+[\d:]+\s+(.*?)\s*\(\s*SN:\s*(\w+)[^)]*\)\s+(.+?)(?:\s+Seite|\s*$)",
        text,
        re.MULTILINE,
    )
    if m:
        out["followup_date"]  = m.group(1)
        out["device_model"]   = m.group(2).strip()
        out["device_serial"]  = m.group(3)
        out["programmer"]     = m.group(4)

    # Last follow-up and implant date from header line
    m = re.search(r"Letzte Nachsorge:\s*([\d.]+)", text)
    if m:
        out["last_followup"] = m.group(1)
    m = re.search(r"Implantation:\s*([\d.]+)", text)
    if m:
        out["implant_date"] = m.group(1)

    return out


# ── page 3 extractor ("Nachsorge - 1. Abfrage") ──────────────────────────────

def _parse_nachsorge(page) -> dict:
    rows = _rows(page)
    out: dict = {}

    # Implantatstatus
    out["mode"]           = _scalar(rows, r"Modus/Ven\.\s+Stimulation")
    base_upper            = _slash_values(rows, r"Grund-/Ob\.\s+Grenzfr\.")
    out["base_rate_bpm"]  = base_upper[0] if len(base_upper) > 0 else None
    out["upper_rate_bpm"] = base_upper[1] if len(base_upper) > 1 else None

    amp = _3col(rows, r"Impulsamplitude\s+\[V\]")
    out["amplitude_a_v"]  = amp["a"]
    out["amplitude_rv_v"] = amp["rv"]
    out["amplitude_lv_v"] = amp["lv"]

    pw = _3col(rows, r"Impulsdauer\s+\[ms\]")
    out["pulse_width_a_ms"]  = pw["a"]
    out["pulse_width_rv_ms"] = pw["rv"]
    out["pulse_width_lv_ms"] = pw["lv"]

    vt_vf = _slash_values(rows, r"VT1/VT2/VF")
    out["vt1_bpm"] = vt_vf[0] if len(vt_vf) > 0 else None
    out["vt2_bpm"] = vt_vf[1] if len(vt_vf) > 1 else None
    out["vf_bpm"]  = vt_vf[2] if len(vt_vf) > 2 else None

    # Battery
    out["battery_voltage_v"]       = _scalar(rows, r"Batteriespannung")
    out["battery_remaining_pct"]   = _scalar(rows, r"Batterie-Restkapazität")
    out["battery_status"]          = _scalar(rows, r"Ladezustand")

    # Charge time: "9.1 s (40 J)" — parse from full row text
    row = _find_row(rows, r"Dauer letzter Ladevorgang")
    if row:
        full = " ".join(w["text"] for w in row)
        m = re.search(r"([\d.]+)\s*s\s*\((\d+)\s*J\)", full)
        if m:
            out["charge_time_s"]     = m.group(1)
            out["charge_energy_j"]   = m.group(2)

    out["home_monitoring"] = _scalar(rows, r"Home Monitoring")
    out["mrt_program"]     = _scalar(rows, r"MRT-Programm")

    # Test results
    sensing = _3col(rows, r"Wahrnehmungsamplitude")
    out["sensing_a_mv"]  = sensing["a"]
    out["sensing_rv_mv"] = sensing["rv"]
    out["sensing_lv_mv"] = sensing["lv"]

    thresh = _3col(rows, r"Reizschwelle\s+\[V\]")
    out["threshold_a_v"]  = thresh["a"]
    out["threshold_rv_v"] = thresh["rv"]
    out["threshold_lv_v"] = thresh["lv"]

    # Threshold pulse width: "Impulsdauer [ms]" row immediately after "Reizschwelle [V]"
    reiz_idx = next(
        (i for i, r in enumerate(rows)
         if re.search(r"Reizschwelle\s+\[V\]", _label(r), re.IGNORECASE)),
        None,
    )
    if reiz_idx is not None:
        for r in rows[reiz_idx + 1 : reiz_idx + 3]:
            if re.search(r"Impulsdauer\s+\[ms\]", _label(r), re.IGNORECASE):
                tpw = _values(r)
                out["threshold_a_pw_ms"]  = tpw.get("a")
                out["threshold_rv_pw_ms"] = tpw.get("rv")
                out["threshold_lv_pw_ms"] = tpw.get("lv")
                break

    imp = _3col(rows, r"Stimulationsimpedanz")
    out["impedance_a_ohm"]  = imp["a"]
    out["impedance_rv_ohm"] = imp["rv"]
    out["impedance_lv_ohm"] = imp["lv"]

    out["shock_impedance_ohm"] = _scalar(rows, r"Schockimpedanz")

    # Diagnostics — "Stim. A/LV/BiV/CRT [%] ----- / 10 / 89 / 99"
    stim = _slash_values(rows, r"Stim\.\s+A/LV/BiV/CRT")
    out["stim_a_pct"]   = stim[0] if len(stim) > 0 else None
    out["stim_lv_pct"]  = stim[1] if len(stim) > 1 else None
    out["stim_biv_pct"] = stim[2] if len(stim) > 2 else None
    out["stim_crt_pct"] = stim[3] if len(stim) > 3 else None

    out["af_burden_pct"] = _scalar(rows, r"Atriale Arrhythmielast")

    # Episodes — "Neue Episoden VF/VT/andere ----- / ----- / 1"
    ep = _slash_values(rows, r"Neue Episoden")
    out["episodes_vf"]    = ep[0] if len(ep) > 0 else None
    out["episodes_vt"]    = ep[1] if len(ep) > 1 else None
    out["episodes_other"] = ep[2] if len(ep) > 2 else None

    return out


# ── entry point ───────────────────────────────────────────────────────────────

def extract(pdf_path: str | Path) -> dict:
    """Extract clinical parameters from a Biotronik BIOSTD report PDF."""
    result: dict = {
        "source_file": str(pdf_path),
        "manufacturer": "Biotronik",
        "report_type": "FollowUpReport",
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        if len(pdf.pages) < 3:
            raise ValueError(f"Expected ≥3 pages, got {len(pdf.pages)}")

        # Header fields from page 1 text
        p1_text = pdf.pages[0].extract_text() or ""
        result.update(_parse_header(p1_text))

        # Primary clinical data from page 3 (Nachsorge - 1. Abfrage)
        result.update(_parse_nachsorge(pdf.pages[2]))

        # Fall back to page 1 spatial for test fields absent on page 3
        p1_rows = _rows(pdf.pages[0])
        _FALLBACK_FIELDS = [
            ("threshold_a_v",  lambda: _3col(p1_rows, r"Reizschwelle\s+\[V\]")["a"]),
            ("threshold_lv_v", lambda: _3col(p1_rows, r"Reizschwelle\s+\[V\]")["lv"]),
            ("threshold_rv_v", lambda: _3col(p1_rows, r"Reizschwelle\s+\[V\]")["rv"]),
            ("sensing_lv_mv",  lambda: _3col(p1_rows, r"Wahrnehmungsamplitude")["lv"]),
            ("sensing_rv_mv",  lambda: _3col(p1_rows, r"Wahrnehmungsamplitude")["rv"]),
            ("sensing_a_mv",   lambda: _3col(p1_rows, r"Wahrnehmungsamplitude")["a"]),
        ]
        for field, getter in _FALLBACK_FIELDS:
            if result.get(field) is None:
                result[field] = getter()

        # Threshold pulse width fallback from page 1
        pw_fields = ("threshold_a_pw_ms", "threshold_rv_pw_ms", "threshold_lv_pw_ms")
        if any(result.get(f) is None for f in pw_fields):
            p1_reiz = next(
                (i for i, r in enumerate(p1_rows)
                 if re.search(r"Reizschwelle\s+\[V\]", _label(r), re.IGNORECASE)),
                None,
            )
            if p1_reiz is not None:
                for r in p1_rows[p1_reiz + 1 : p1_reiz + 3]:
                    if re.search(r"Impulsdauer\s+\[ms\]", _label(r), re.IGNORECASE):
                        tpw = _values(r)
                        for ch in ("a", "rv", "lv"):
                            key = f"threshold_{ch}_pw_ms"
                            if result.get(key) is None:
                                result[key] = tpw.get(ch)
                        break

    return result
