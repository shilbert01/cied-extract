#!/usr/bin/env python3
"""
Generate a comparison PDF showing the original report page alongside
extraction results from both the PaddleOCR+regex pipeline and the
Qwen2-VL VQA pipeline.
"""
import json
from pathlib import Path

import pdf2image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

# ── load data ─────────────────────────────────────────────────────────────────

PDF_PATH  = Path("Aveir(TM)-LP_LSP201A-LSP202V_40.41_SummaryReport.pdf")
REGEX_JSON = Path("results/Aveir(TM)-LP_LSP201A-LSP202V_40.41_SummaryReport.json")
VQA_JSON   = Path("results/SummaryReport_vqa_final.json")
OUT_PDF    = Path("results/CiedOCR_comparison.pdf")

regex = json.loads(REGEX_JSON.read_text())
vqa   = json.loads(VQA_JSON.read_text())

page_img = pdf2image.convert_from_path(str(PDF_PATH), dpi=150)[0]

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(val):
    if val is None:
        return "—"
    if isinstance(val, dict):
        return ""
    return str(val)

def section_rows(title, regex_block, vqa_block, fields):
    """Return list of (label, regex_val, vqa_val, match) for a section."""
    rows = [("", title, "", None)]  # section header row
    rb = regex_block or {}
    vb = vqa_block  or {}
    for label, key in fields:
        rv = fmt(rb.get(key))
        vv = fmt(vb.get(key))
        match = (rv == vv)
        rows.append((label, rv, vv, match))
    return rows

# ── build table data ──────────────────────────────────────────────────────────

SECTIONS = []

# Header
SECTIONS += section_rows("HEADER", regex, vqa, [
    ("Report type",       "report_type"),
    ("Patient",           "patient"),
    ("Device",            "device"),
    ("Facility",          "facility"),
    ("Date",              "interrogation_date"),
    ("Visit type",        "visit_type"),
])

# Battery RA
SECTIONS += section_rows("BATTERY — RA", regex.get("battery_ra"), vqa.get("battery_ra"), [
    ("Implant date",      "implant_date"),
    ("Voltage",           "voltage"),
    ("Current",           "current_ua"),
    ("Remaining cap.",    "remaining_capacity_pct"),
    ("Est. longevity",    "estimated_longevity"),
    ("RRT",               "rrt"),
])

# Battery RV
SECTIONS += section_rows("BATTERY — RV", regex.get("battery_rv"), vqa.get("battery_rv"), [
    ("Implant date",      "implant_date"),
    ("Voltage",           "voltage"),
    ("Current",           "current_ua"),
    ("Remaining cap.",    "remaining_capacity_pct"),
    ("Est. longevity",    "estimated_longevity"),
    ("RRT",               "rrt"),
])

# Test RA
SECTIONS += section_rows("TEST RESULTS — RA", regex.get("test_ra"), vqa.get("test_ra"), [
    ("Pacing threshold",  "pacing_threshold"),
    ("  date",            "pacing_threshold_date"),
    ("Sensing",           "sensing_mv"),
    ("  date",            "sensing_date"),
    ("Impedance",         "impedance_ohm"),
    ("  date",            "impedance_date"),
])

# Test RV
SECTIONS += section_rows("TEST RESULTS — RV", regex.get("test_rv"), vqa.get("test_rv"), [
    ("Pacing threshold",  "pacing_threshold"),
    ("  date",            "pacing_threshold_date"),
    ("Sensing",           "sensing_mv"),
    ("  date",            "sensing_date"),
    ("Impedance",         "impedance_ohm"),
    ("  date",            "impedance_date"),
])

# Parameters
SECTIONS += section_rows("PARAMETERS", regex.get("parameters"), vqa.get("parameters"), [
    ("Mode",              "mode"),
    ("Base rate",         "base_rate"),
    ("Upper rate limit",  "upper_rate_limit"),
    ("Max sensor rate",   "max_sensor_rate"),
    ("AV paced",          "av_paced_ms"),
    ("AV sensed",         "av_sensed_ms"),
    ("Rate hysteresis",   "rate_hysteresis"),
    ("VIP",               "vip_enabled"),
    ("Rate-adapt. AV",    "rate_adaptive_av"),
    ("Rate-adapt. PVARP", "rate_adaptive_pvarp"),
    ("Vent. safety pac.", "ventricular_safety_pacing"),
])

# Stimulation RA
SECTIONS += section_rows("STIMULATION — RA", regex.get("stimulation_ra"), vqa.get("stimulation_ra"), [
    ("Configuration",     "configuration"),
    ("Amplitude",         "amplitude_v"),
    ("Pulse width",       "pulse_width_ms"),
    ("SenseAbility",      "senseability"),
    ("Sensitivity",       "sensitivity_mv"),
])

# Stimulation RV
SECTIONS += section_rows("STIMULATION — RV", regex.get("stimulation_rv"), vqa.get("stimulation_rv"), [
    ("Configuration",     "configuration"),
    ("Amplitude",         "amplitude_v"),
    ("Pulse width",       "pulse_width_ms"),
    ("SenseAbility",      "senseability"),
    ("Sensitivity",       "sensitivity_mv"),
])

# Diagnostics
SECTIONS += section_rows("DIAGNOSTICS", regex.get("diagnostics"), vqa.get("diagnostics"), [
    ("Since",             "since"),
    ("AP paced %",        "ap_paced_pct"),
    ("VP paced %",        "vp_paced_pct"),
    ("Mode switch %",     "mode_switch_pct"),
    ("AMS episodes",      "ams_episodes"),
])

# ── count stats ───────────────────────────────────────────────────────────────

data_rows = [r for r in SECTIONS if r[3] is not None]
n_match  = sum(1 for r in data_rows if r[3])
n_differ = sum(1 for r in data_rows if not r[3] and r[1] != "—" and r[2] != "—")
n_only_r = sum(1 for r in data_rows if not r[3] and r[1] != "—" and r[2] == "—")
n_only_v = sum(1 for r in data_rows if not r[3] and r[1] == "—" and r[2] != "—")
n_total  = len(data_rows)

# ── colours ───────────────────────────────────────────────────────────────────

C_HEADER  = "#2c3e50"   # dark blue-grey for section headers
C_MATCH   = "#d5f5e3"   # green tint
C_DIFFER  = "#fde8d8"   # orange tint
C_ONE_R   = "#eaf2ff"   # blue tint  (regex only)
C_ONE_V   = "#fff3cd"   # yellow (vqa only)
C_NULL    = "#f8f9fa"   # near-white for both-null rows
C_HEAD_BG = "#ecf0f1"   # column header background

# ── PDF ───────────────────────────────────────────────────────────────────────

with PdfPages(str(OUT_PDF)) as pdf:

    # ── PAGE 1: original report image ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    ax.imshow(page_img)
    ax.axis("off")
    ax.set_title("Original Report — Abbott Aveir DR LP SummaryReport",
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout(pad=0.5)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # ── PAGE 2+: comparison table ─────────────────────────────────────────────
    ROWS_PER_PAGE = 45
    chunks = [SECTIONS[i:i+ROWS_PER_PAGE]
              for i in range(0, len(SECTIONS), ROWS_PER_PAGE)]

    for page_num, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax.axis("off")

        # Title
        title = "Extraction Comparison: PaddleOCR+Regex  vs  Qwen2-VL VQA"
        if len(chunks) > 1:
            title += f"  (page {page_num+1}/{len(chunks)})"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        col_labels = ["Field", "PaddleOCR + Regex", "Qwen2-VL-7B VQA"]
        col_w = [0.28, 0.36, 0.36]
        n_rows = len(chunk) + 1  # +1 for column header
        row_h = 0.88 / n_rows
        x_starts = [0.01, 0.30, 0.67]
        y_top = 0.93

        # Column header row
        for ci, (label, xs, cw) in enumerate(zip(col_labels, x_starts, col_w)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (xs, y_top - row_h), cw - 0.005, row_h * 0.92,
                boxstyle="round,pad=0.005", linewidth=0,
                facecolor=C_HEAD_BG, transform=ax.transAxes, clip_on=False
            ))
            ax.text(xs + cw/2, y_top - row_h/2, label,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    transform=ax.transAxes)

        for ri, row in enumerate(chunk):
            label, rv, vv, match = row
            y = y_top - (ri + 2) * row_h

            is_section = match is None
            if is_section:
                bg = C_HEADER
                fc = "white"
                fw = "bold"
                fs = 7.5
                display = [rv, "", ""]  # section title spans field col
            else:
                fc = "#1a1a1a"
                fw = "normal"
                fs = 7
                display = [label, rv, vv]
                both_null = rv == "—" and vv == "—"
                if both_null:
                    bg = C_NULL
                elif match:
                    bg = C_MATCH
                elif rv != "—" and vv != "—":
                    bg = C_DIFFER
                elif rv != "—":
                    bg = C_ONE_R
                else:
                    bg = C_ONE_V

            for ci, (text, xs, cw) in enumerate(zip(display, x_starts, col_w)):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (xs, y), cw - 0.005, row_h * 0.92,
                    boxstyle="round,pad=0.003", linewidth=0,
                    facecolor=bg, transform=ax.transAxes, clip_on=False
                ))
                if is_section and ci == 0:
                    # section title across full width
                    ax.text(0.5, y + row_h * 0.46, text,
                            ha="center", va="center", fontsize=fs,
                            fontweight=fw, color=fc,
                            transform=ax.transAxes)
                    break
                elif not is_section:
                    ha = "left" if ci == 0 else "left"
                    ax.text(xs + 0.008, y + row_h * 0.46, text,
                            ha="left", va="center", fontsize=fs,
                            fontweight=fw, color=fc,
                            transform=ax.transAxes, clip_on=False)

        # Legend + stats (last page only)
        if page_num == len(chunks) - 1:
            ly = 0.025
            legend_items = [
                (C_MATCH,  f"Match ({n_match})"),
                (C_DIFFER, f"Both extracted, differ ({n_differ})"),
                (C_ONE_R,  f"Regex only ({n_only_r})"),
                (C_ONE_V,  f"VQA only ({n_only_v})"),
                (C_NULL,   f"Both null"),
            ]
            lx = 0.01
            for color, label in legend_items:
                ax.add_patch(mpatches.Rectangle(
                    (lx, ly), 0.018, 0.018,
                    facecolor=color, edgecolor="#aaa", linewidth=0.5,
                    transform=ax.transAxes
                ))
                ax.text(lx + 0.022, ly + 0.009, label,
                        va="center", fontsize=6.5, transform=ax.transAxes)
                lx += 0.19
            ax.text(0.99, ly + 0.009,
                    f"Total data fields: {n_total}  |  VQA accuracy: {n_match}/{n_total} ({100*n_match//n_total}%)",
                    ha="right", va="center", fontsize=7, transform=ax.transAxes,
                    color="#555")

        fig.tight_layout(pad=0.3)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

print(f"Written: {OUT_PDF}")
