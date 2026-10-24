"""
Abbott Merlin PCS report extractor — VQA-based (Qwen2-VL-7B-Instruct).

Handles report types emitted by Merlin PCS for Abbott devices:
  - FreezeReport        : single-page ECG + Schlüsselparameter table
  - FastPathSummary ICD : FastPath™ Übersicht for ICD devices (e.g. Entrant DR)
  - FastPathSummary PM  : FastPath™ Übersicht for pacemakers (e.g. Endurity MRI)
  - WrapupReport        : Wrap-up™ Report (last page of ICD Complete exports)
  - parameters_icd      : ICD Complete report — page 1 is Parameters (no FastPath)
  - PSAReport           : PSA-Bericht — implant lead test (threshold/sensing/impedance)

The report type is detected from the rendered page image title.
Model must be loaded externally (loaded once, reused across calls).

Usage:
    from vqa import load_model
    from extractors.abbott_merlin import extract, detect_report_type

    model, processor = load_model("cuda")
    result = extract("report.pdf", model, processor)
"""
import json
import re
from pathlib import Path
from typing import Optional

import pdf2image

# ── schemas ───────────────────────────────────────────────────────────────────

SCHEMA_FREEZE = {
    "report_type": "FreezeReport",
    "patient_name": None,
    "followup_date": None,
    "device": None,
    "facility": None,
    "mode": None,
    "base_rate_bpm": None,
    "av_paced_ms": None,
    "av_sensed_ms": None,
    "upper_rate_bpm": None,
    "max_sensor_rate_bpm": None,
    "rate_hysteresis": None,
    "vip": None,
    "rate_adaptive_av": None,
    "rate_adaptive_pvarp": None,
    "vent_safety_pacing": None,
}

SCHEMA_FASTPATH_ICD = {
    "report_type": "FastPathSummary",
    "patient_name": None,
    "followup_date": None,
    "device_model": None,
    "device_serial": None,
    "battery_eri": None,
    "battery_current_ua": None,
    "battery_remaining_pct": None,
    "threshold_rv_v": None,
    "threshold_rv_pw_ms": None,
    "sensing_rv_mv": None,
    "impedance_rv_ohm": None,
    "threshold_lv_v": None,
    "threshold_lv_pw_ms": None,
    "sensing_lv_mv": None,
    "impedance_lv_ohm": None,
    "shock_impedance_ohm": None,
    "mode": None,
    "vt_zone": None,
    "vf_zone": None,
    "detection_params": None,
    "therapy": None,
    "sense_ability_rv": None,
    "sensing_config_rv": None,
    "sense_ability_lv": None,
    "sensing_config_lv": None,
    "vt_vf_episodes": None,
}

SCHEMA_FASTPATH_PM = {
    "report_type": "FastPathSummary",
    "patient_name": None,
    "followup_date": None,
    "device_model": None,
    "device_serial": None,
    "battery_voltage_v": None,
    "battery_mag_freq_bpm": None,
    "battery_remaining_pct": None,
    "threshold_a_v": None,
    "threshold_a_pw_ms": None,
    "sensing_a_mv": None,
    "impedance_a_ohm": None,
    "threshold_rv_v": None,
    "threshold_rv_pw_ms": None,
    "sensing_rv_mv": None,
    "impedance_rv_ohm": None,
    "threshold_lv_v": None,
    "threshold_lv_pw_ms": None,
    "sensing_lv_mv": None,
    "impedance_lv_ohm": None,
    "mode": None,
    "base_rate_bpm": None,
    "upper_rate_bpm": None,
    "av_paced_ms": None,
    "av_sensed_ms": None,
    "autocapture_a": None,
    "autocapture_rv": None,
    "autocapture_lv": None,
    "amplitude_a_v": None,
    "amplitude_rv_v": None,
    "amplitude_lv_v": None,
    "pulse_width_a_ms": None,
    "pulse_width_rv_ms": None,
    "pulse_width_lv_ms": None,
    "sense_ability_a": None,
    "sense_ability_rv": None,
    "sense_ability_lv": None,
    "sensitivity_a_mv": None,
    "sensitivity_rv_mv": None,
    "sensitivity_lv_mv": None,
    "stim_a_pct": None,
    "stim_rv_pct": None,
    "stim_lv_pct": None,
    "ams_episodes": None,
    "mode_switch_pct": None,
    "af_burden_pct": None,
    "diagnostics_since": None,
}

_BATTERY_BLOCK = {
    "implant_date": None,
    "voltage": None,
    "current_ua": None,
    "remaining_capacity_pct": None,
    "estimated_longevity": None,
    "rrt": None,
}
_TEST_BLOCK = {
    "pacing_threshold": None,
    "pacing_threshold_date": None,
    "sensing_mv": None,
    "sensing_date": None,
    "impedance_ohm": None,
    "impedance_date": None,
}
_STIM_BLOCK = {
    "configuration": None,
    "amplitude_v": None,
    "pulse_width_ms": None,
    "senseability": None,
    "sensitivity_mv": None,
}

SCHEMA_SUMMARY = {
    "report_type": "SummaryReport",
    "patient": None,
    "device": None,
    "facility": None,
    "interrogation_date": None,
    "visit_type": None,
    "battery_ra": dict(_BATTERY_BLOCK),
    "battery_rv": dict(_BATTERY_BLOCK),
    "test_ra": dict(_TEST_BLOCK),
    "test_rv": dict(_TEST_BLOCK),
    "parameters": {
        "mode": None,
        "base_rate": None,
        "av_paced_ms": None,
        "av_sensed_ms": None,
        "upper_rate_limit": None,
        "max_sensor_rate": None,
        "rate_hysteresis": None,
        "vip_enabled": None,
        "rate_adaptive_av": None,
        "rate_adaptive_pvarp": None,
        "ventricular_safety_pacing": None,
    },
    "stimulation_ra": dict(_STIM_BLOCK),
    "stimulation_rv": dict(_STIM_BLOCK),
    "diagnostics": {
        "since": None,
        "ap_paced_pct": None,
        "vp_paced_pct": None,
        "mode_switch_pct": None,
        "ams_episodes": None,
    },
}

SCHEMA_WRAPUP_ICD = {
    "report_type": "WrapupReport",
    "patient_name": None,
    "patient_dob": None,             # Geburtsdatum e.g. "30 Sep 1939"
    "followup_date": None,
    "device_model": None,
    "device_serial": None,
    "implant_date": None,            # Impl.datum for generator row
    # Aggregat table — lead details
    "lead_a_model": None,            # A Elektrode Modell e.g. "Tendril™ STS 2088TC / 52 cm"
    "lead_a_serial": None,           # A Elektrode S/N e.g. "EEL580765"
    "lead_a_implant_date": None,     # A Elektrode Impl.datum
    "lead_rv_model": None,           # RV/V Elektrode Modell
    "lead_rv_serial": None,          # RV/V Elektrode S/N
    "lead_rv_implant_date": None,    # RV/V Elektrode Impl.datum
    "lead_lv_model": None,           # LV Elektrode Modell (CRT only)
    "lead_lv_serial": None,          # LV Elektrode S/N (CRT only)
    "lead_lv_implant_date": None,    # LV Elektrode Impl.datum (CRT only)
    # Batterie section
    "battery_voltage_v": None,       # Spannung e.g. "2,96"
    "battery_longevity": None,       # Laufzeit e.g. "1,5 Jahre"
    "battery_charging_time_s": None, # Letzte max. Aufladung e.g. "9,4"
    "battery_current_ua": None,      # Batteriestrom e.g. "25"
    "battery_remaining_pct": None,   # Verbleib. Kapazität bis ERI e.g. "29"
    "battery_eri": None,             # ERI e.g. "> 5 Jhr."
    # Testergebnisse section
    "threshold_a_v": None,           # Reizschwelle A amplitude e.g. "1,0"
    "threshold_a_pw_ms": None,       # Reizschwelle A pulse width e.g. "0,4"
    "sensing_a_mv": None,            # Signalamplitude A e.g. "2,1"
    "impedance_a_ohm": None,         # Elektrodenimpedanz A e.g. "600"
    "threshold_rv_v": None,          # Reizschwelle RV amplitude e.g. "0,5"
    "threshold_rv_pw_ms": None,      # Reizschwelle RV pulse width e.g. "0,4"
    "sensing_rv_mv": None,           # Signalamplitude RV e.g. "11,8"
    "impedance_rv_ohm": None,        # Elektrodenimpedanz RV e.g. "1175"
    "threshold_lv_v": None,          # Reizschwelle LV amplitude e.g. "1,25"
    "threshold_lv_pw_ms": None,      # Reizschwelle LV pulse width e.g. "0,4"
    "sensing_lv_mv": None,           # Signalamplitude LV (CRT devices) e.g. "12,0"
    "impedance_lv_ohm": None,        # Elektrodenimpedanz LV e.g. "940"
    "impedance_hv_ohm": None,        # HV coil impedance e.g. "74"
    # Parameter section
    "mode": None,                    # Betriebsart e.g. "DDD"
    "base_rate_bpm": None,           # GF Grundfrequenz e.g. "60"
    "upper_rate_bpm": None,          # MTR e.g. "130"
    "av_paced_ms": None,             # Stimuliertes AV-Intervall e.g. "130"
    "av_sensed_ms": None,            # Wahrg. AV-Int. e.g. "100"
    "vt_zone": None,                 # lowest VT detection rate bpm e.g. "160"
    "vf_zone": None,                 # VF detection rate bpm e.g. "230"
    "therapy": None,                 # Therapie status e.g. "Aus"
    # Changes made at this visit
    "new_programming": None,
}

SCHEMA_PSA = {
    "report_type": "PSAReport",
    "patient_name": None,
    # Device header (top-right / top-centre of page)
    "implant_date": None,       # date shown in top-right, e.g. "8 Jul 2026"
    "device_model": None,       # e.g. "2172"
    "device_serial": None,      # e.g. "4130170"
    # Lead table (Elektrode / Hersteller / Modell / Serien-Nr. / Impl.datum)
    "lead_a_model": None,       # Modell for A electrode, e.g. "Tendril™ STS 2088TC"
    "lead_a_serial": None,      # Serien-Nr. for A electrode
    "lead_a_implant_date": None,
    "lead_rv_model": None,      # Modell for RV electrode
    "lead_rv_serial": None,
    "lead_rv_implant_date": None,
    # Testergebnisse — A lead
    "threshold_a_v": None,      # Reizschw. amplitude, number only, e.g. "0,9"
    "threshold_a_pw_ms": None,  # Reizschw. pulse width, e.g. "0,4"
    "threshold_a_ma": None,     # Reizschw. current, e.g. "1,7"
    "sensing_a_mv": None,       # Signalamplitude mV value, e.g. "5,3"
    "sensing_a_slew_vs": None,  # Signalamplitude slew rate V/s value, e.g. "1,7"
    "impedance_a_ohm": None,    # Impedanz value, e.g. "477"
    # Testergebnisse — RV lead
    "threshold_rv_v": None,
    "threshold_rv_pw_ms": None,
    "threshold_rv_ma": None,
    "sensing_rv_mv": None,
    "sensing_rv_slew_vs": None,
    "impedance_rv_ohm": None,
}

SCHEMA_PARAMETER_ICD = {
    "report_type": "ParameterReport",
    "patient_name": None,
    "followup_date": None,
    "device_model": None,        # e.g. "CDDRA300Q"
    "device_serial": None,       # e.g. "810187613"
    "implant_date": None,        # device Impl.datum e.g. "27 Mär 2026"
    "lead_a_model": None,        # A Elektrode Modell
    "lead_a_serial": None,       # A Elektrode S/N
    "lead_a_implant_date": None,
    "lead_rv_model": None,       # V Elektrode Modell
    "lead_rv_serial": None,
    "lead_rv_implant_date": None,
    "mode": None,                # Betriebsart e.g. "DDD"
    "base_rate_bpm": None,       # Grundfrequenz e.g. "60"
    "upper_rate_bpm": None,      # MTR e.g. "100"
    "av_paced_ms": None,         # Stimuliertes AV-Intervall e.g. "180"
    "av_sensed_ms": None,        # Wahrg. AV-Int. e.g. "150"
    "amplitude_a_v": None,       # Impulsamplitude A e.g. "3,125"
    "pulse_width_a_ms": None,    # Impulsdauer A e.g. "0,5"
    "amplitude_rv_v": None,      # Impulsamplitude V e.g. "3,0"
    "pulse_width_rv_ms": None,   # Impulsdauer V e.g. "1,0"
}

SCHEMA_TESTERGEBNISSE_RV = {
    "report_type": "TestergebnisseReport",
    "followup_date": None,
    "threshold_rv_v": None,      # most recent V threshold amplitude e.g. "1,75"
    "threshold_rv_pw_ms": None,  # most recent V threshold pulse width e.g. "0,5"
}


# ── prompts ───────────────────────────────────────────────────────────────────

_COMMON_RULES = """\
IMPORTANT RULES:
- For NUMERIC fields: extract only the number, WITHOUT the unit suffix.
  "60 min⁻¹" → "60",  "2,5 V" → "2,5",  "0,4 ms" → "0,4",  "12 uA" → "12".
  Use German decimal comma (e.g. "3,08", "0,5").
- Preserve ">" or "<" prefixes (e.g. ">95", ">99", "> 5 Jhr.") and do NOT strip those.
- For TEXT fields (Aus, Ein, Auto, DDD, DDDR, Stimulation …): copy exactly as printed.
- For DATE fields: extract only the date, strip any leading word like "Seit".
  "Seit 3 Mär 2025" → "3 Mär 2025".
- IMPORTANT: If a test table cell says "Nicht durchgeführt", return that exact string — \
do NOT return null. Only return null if the label itself is missing from the image.
- Return ONLY the JSON object — no markdown fences, no explanation.
"""

PROMPT_FREEZE = """\
You are a medical data extraction assistant. Extract pacemaker parameters from \
this Abbott Merlin PCS FreezeReport (Ausschnitt) image.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- patient_name: top centre header, the patient's name (format "Lastname, Firstname")
- followup_date: top-right corner (format "dd Mon yyyy")
- device: centre top, line starting with "Aveir" — copy exactly
- facility: centre top, line starting with "HZ"
- Schlüsselparameter table (left column = label, right column = value):
  - mode ← "Betriebsart" (text, e.g. "DDD")
  - base_rate_bpm ← "Grundfrq." — extract ONLY the number (e.g. "50"), NOT "50 min⁻¹"
  - av_paced_ms ← "AV stimuliert" — number only (e.g. "200"), NOT "200 ms"
  - av_sensed_ms ← "AV wahrgen." — number only (e.g. "150")
  - upper_rate_bpm ← "Obere Grenzfrq." — number only (e.g. "130")
  - max_sensor_rate_bpm ← "Max. Sensorfrq." — number only (e.g. "130")
  - rate_hysteresis ← "Frequenzhysteresedif." (text, e.g. "Aus")
  - vip ← "Ventricular Intrinsic Preference (VIP™)" (text, e.g. "Aus")
  - rate_adaptive_av ← "Frequenzabh. AV-Intervall" (text, e.g. "Mittel")
  - rate_adaptive_pvarp ← "Frequenzabh. PVARP/VREF" (text, e.g. "Hoch")
  - vent_safety_pacing ← "Vent. Sicherheits-Stim." (text, e.g. "Ein")

""" + json.dumps(SCHEMA_FREEZE, ensure_ascii=False, indent=2)

PROMPT_FASTPATH_ICD = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS FastPath™ Übersicht report for an ICD device.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- patient_name: top centre header, the patient's name (format "Lastname, Firstname")
- followup_date: top-right corner (format "d Mon yyyy" e.g. "4 Apr 2025")
- device_model: top centre, the model code after "Entrant™ DR " (e.g. "CDDRA300Q")
- device_serial: top centre, the number after the model code (e.g. "810131959")
- Batterie section:
  - battery_eri ← value after "ERI" label (e.g. "> 5 Jhr.")
  - battery_current_ua ← "Batteriestrom" value (number, e.g. "12")
  - battery_remaining_pct ← "Verbleib. Kapazität bis ERI" value (e.g. ">95")
- Testergebnisse section (tests performed during follow-up — if a row shows
  "Nicht durchgeführt", return that exact string, NOT null):
  For each lead row (RV, and LV if CRT-D), extract three measurements:
  - threshold_rv_v ← "Reizschwelle" amplitude for RV row (number before "V", e.g. "1,75")
  - threshold_rv_pw_ms ← "Reizschwelle" pulse width for RV row (number after "@" before "ms", e.g. "0,5")
  - sensing_rv_mv ← "Signalamplitude" value for RV row (e.g. "Nicht durchgeführt" or a number)
  - impedance_rv_ohm ← "Elektrodenimpedanz" value for RV row
  - threshold_lv_v ← "Reizschwelle" amplitude for LV row (CRT-D; null if no LV lead)
  - threshold_lv_pw_ms ← "Reizschwelle" pulse width for LV row (null if no LV lead)
  - sensing_lv_mv ← "Signalamplitude" value for LV row (CRT-D; null if no LV lead)
  - impedance_lv_ohm ← "Elektrodenimpedanz" value for LV row (null if no LV lead)
  - shock_impedance_ohm ← "Elektrodenimpedanz" value for HV coil row
- Parameter section (a table with label on the left, value on the right):
  - mode ← "Betriebsart" row — the value is a WORD like "Stimulation", "VVI",
    "DDI", never a single letter. Do not confuse it with the "A" or "V" column
    headers in the Stimulation section.
  - vt_zone ← "Zonenkonfiguration" row, VT column (often "Aus" = disabled)
  - vf_zone ← "Zonenkonfiguration" row, VF column (often "Aus")
  - detection_params ← "Detektionsparameter" row (often "Aus")
  - therapy ← row labelled "Therapie" or "Therapie (AUS)" (often "Aus")
- Stimulation und Wahrnehmung section — find the RV (ventricular) column,
  and the LV column if present (CRT-D devices have A, RV, LV):
  - sense_ability_rv ← "SenseAbility™" or "SenseAbility" row, RV/V column (e.g. "Ein")
  - sensing_config_rv ← "Empfindlichkeit" row, RV/V column (e.g. "Auto" or a number)
  - sense_ability_lv ← "SenseAbility™" row, LV column (null if no LV column)
  - sensing_config_lv ← "Empfindlichkeit" row, LV column (null if no LV column)
- Diagnostikübersicht section:
  - vt_vf_episodes ← "VT/VF-Episoden:" number (return as string, e.g. "0")

""" + json.dumps(SCHEMA_FASTPATH_ICD, ensure_ascii=False, indent=2)

PROMPT_FASTPATH_PM = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS FastPath™ Übersicht report for a pacemaker device.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- patient_name: top centre header, the patient's name (format "Lastname, Firstname")
- followup_date: the report generation date shown in the TOP-RIGHT corner of the
  page, printed as part of the patient/device header (format "d Mon yyyy",
  e.g. "4 Apr 2025"). IGNORE any date that follows the word "Seit" — that is
  the diagnostics_since date, not the followup_date.
- device_model: top centre, model number after "Endurity MRI™ " (e.g. "2172")
- device_serial: top centre, number after the model (e.g. "5153558")
- Batterie section:
  - battery_voltage_v ← "Spannung" value (e.g. "3,08")
  - battery_mag_freq_bpm ← "Magnetfrequenz" value (number before "min⁻¹")
  - battery_remaining_pct ← "Verbleib. Kapazität bis ERI" value
- Testergebnisse section: fields are "Nicht durchgeführt" if no test was done.
  Dual-chamber devices (DR) have A and V rows. CRT devices (CRT-P) have A, RV,
  and LV rows — map them to the _a_, _rv_, _lv_ fields respectively.
  For each lead, extract three measurements:
  - threshold_a_v / threshold_rv_v / threshold_lv_v ← "Reizschwelle" amplitude
    (number before "V", e.g. "0,75"; or "Nicht durchgeführt")
  - threshold_a_pw_ms / threshold_rv_pw_ms / threshold_lv_pw_ms ← "Reizschwelle"
    pulse width (number after "@" before "ms", e.g. "0,4"; null if threshold not performed)
  - sensing_a_mv / sensing_rv_mv / sensing_lv_mv ← "Signalamplitude" for A, RV, LV rows
  - impedance_a_ohm / impedance_rv_ohm / impedance_lv_ohm ← "Elektrodenimpedanz" for A, RV, LV rows
  IMPORTANT: if the report shows only RV and LV columns (no A column), use
  the _rv_ and _lv_ fields. Do NOT put LV values into the _a_ fields.
- Parameter section:
  - mode ← "Betriebsart"
  - base_rate_bpm ← "Grundfrequenz (GF)" value (number before "min-1")
  - upper_rate_bpm ← "Ob Grenzfr MTR" value
  - av_paced_ms ← "Stimuliertes AV-Intervall" value
  - av_sensed_ms ← "Wahrg. AV-Int." value
- Stimulation und Wahrnehmung section — columns vary by device type:
  Dual-chamber (DR): A column then V column.
  CRT (CRT-P): A column, RV column, LV column.
  If only two columns are shown and the device is a CRT, they are RV and LV
  (not A and RV). Map each column to its correct _a_, _rv_, or _lv_ field.
  - autocapture_a / autocapture_rv / autocapture_lv ← "ACap™ Confirm/V. AutoCapture" row
  - amplitude_a_v / amplitude_rv_v / amplitude_lv_v ← "Impulsamplitude" row (number before "V")
  - pulse_width_a_ms / pulse_width_rv_ms / pulse_width_lv_ms ← "Impulsdauer" row (number before "ms")
  - sense_ability_a / sense_ability_rv / sense_ability_lv ← "SenseAbility™" row
  - sensitivity_a_mv / sensitivity_rv_mv / sensitivity_lv_mv ← "Empfindlichkeit" row (number before "mV")
- Diagnostikübersicht section (statistics since last follow-up):
  - diagnostics_since ← date shown after the word "Seit" — strip the "Seit" prefix
    (e.g. "Seit 3 Mär 2025" → "3 Mär 2025")
  - stim_a_pct ← stimulation percentage for atrium: look for "AP" or "A Stimulation"
    percentage — may show ">99" if very high (return the printed value, no % sign)
  - stim_rv_pct ← stimulation percentage for RV: look for "VP" or "V Stimulation"
    percentage (same format)
  - stim_lv_pct ← stimulation percentage for LV (CRT devices; null if not present)
  - ams_episodes ← "AMS-Episoden" count (number as string, e.g. "0")
  - mode_switch_pct ← "Mode-Switch" percentage value (no % sign)
  - af_burden_pct ← "AT/AF-Burden" percentage value (no % sign)

""" + json.dumps(SCHEMA_FASTPATH_PM, ensure_ascii=False, indent=2)

PROMPT_SUMMARY = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS FastPath™ Übersicht / SummaryReport for an Aveir™ DR LP device.

IMPORTANT RULES:
- Copy values EXACTLY as printed, INCLUDING units (e.g. "2,5 V", "0,4 ms",
  "330 Ω", "12,0 uA", "99 %", "50 min⁻¹", "11,3 Jahre").
- Use German decimal comma (e.g. "3,08", "0,5").
- Preserve ">" and "<" prefixes (e.g. ">3,3 V", "<1%", ">5 Jahre").
- For test measurements: each row shows "Nicht durchgeführt" (today's test, not
  done) THEN the last measured value with its date in parentheses on the next line.
  Extract ONLY the last measured value and its date — ignore "Nicht durchgeführt".
  E.g. "2,75 V @ 0,4 ms (22 Jun 2026)" → pacing_threshold="2,75 V @ 0,4 ms",
  pacing_threshold_date="22 Jun 2026".
- For null/unavailable fields: use null (not a string).
- Return ONLY the JSON object — no markdown fences, no explanation.

WHERE TO FIND EACH FIELD:
- patient: top centre header (e.g. "DEMO")
- device: top centre, line starting "Aveir™ DR LP" (copy exactly as printed)
- facility: top centre, line starting "HZ" (e.g. "HZ Leipzig")
- interrogation_date: top-right corner, first date shown (e.g. "24 Jun 2026")
- visit_type: top-right corner, below the date (e.g. "Bei der Nachsorge")

Batterieinformationen section — RA subsection then RV subsection:
  For each (battery_ra / battery_rv):
  - estimated_longevity ← "Laufzeit: X Jahre" value (e.g. "11,3 Jahre")
  - rrt ← value shown next to "RRT" (e.g. ">5 Jahre")
  - implant_date ← "Datum der Implantation" value (e.g. "21 Jun 2026")
  - voltage ← "Spannung" value (e.g. ">3,3 V")
  - current_ua ← "Batteriestrom" value (e.g. "12,0 uA")
  - remaining_capacity_pct ← "Verbleibende Kapazität bis RRT" value (e.g. "99 %")

Testergeb. section — RA row then RV row, three columns: Reizschwelle / Wahrnehmung / Impedanz:
  For each lead (test_ra / test_rv), ignore "Nicht durchgeführt" lines and use only
  the line with the actual measured value and date in parentheses:
  - pacing_threshold ← value before the date in Reizschwelle (e.g. "2,75 V @ 0,4 ms")
  - pacing_threshold_date ← date in parentheses in Reizschwelle (e.g. "22 Jun 2026")
  - sensing_mv ← value before the date in Wahrnehmung (e.g. "7,5 mV")
  - sensing_date ← date in parentheses in Wahrnehmung (e.g. "19 Jun 2026")
  - impedance_ohm ← value before the date in Impedanz (e.g. "330 Ω")
  - impedance_date ← date in parentheses in Impedanz (e.g. "19 Jun 2026")

Parameter section (left side):
  - mode ← "Betriebsart" (e.g. "DDD")
  - base_rate ← "Grundfrq." value WITH unit (e.g. "50 min⁻¹")
  - max_sensor_rate ← "Max. Sensorfrq." value WITH unit (e.g. "130 min⁻¹")
  - upper_rate_limit ← "Obere Grenzfrq." value WITH unit (e.g. "130 min⁻¹")
  - av_paced_ms ← "AV stimuliert" value WITH unit (e.g. "200 ms")
  - av_sensed_ms ← "AV wahrgen." value WITH unit (e.g. "150 ms")
  - rate_hysteresis, vip_enabled, rate_adaptive_av, rate_adaptive_pvarp,
    ventricular_safety_pacing ← set null if not visible in this section

Stimulation & Wahr. section — RA column then RV column:
  For each (stimulation_ra / stimulation_rv):
  - configuration ← "Elektrodenkonfig." row (e.g. "Bipolar")
  - amplitude_v ← "Impulsampl." row WITH unit (e.g. "2,5 V")
  - pulse_width_ms ← "Impulsdauer" row WITH unit (e.g. "0,4 ms")
  - senseability ← "SenseAbility™" row (e.g. "Aus")
  - sensitivity_mv ← "Sensitivität" row WITH unit (e.g. "0,5 mV" for RA, "2,0 mV" for RV)

Diagnostikübersicht section (left block):
  - since ← "Seit:" value INCLUDING time (e.g. "22 Jun 2026 20:11")
  - ap_paced_pct ← "Stimuliert (AP)" value in the FIRST (Abgelastet) column (e.g. "27%")
  - vp_paced_pct ← "Stimuliert (VP)" value in the FIRST (Abgelastet) column (e.g. "16%")
  - mode_switch_pct ← "Mode-Switch" value (e.g. "<1%")
  - ams_episodes ← "AMS-Episoden" value as string (e.g. "1")

""" + json.dumps(SCHEMA_SUMMARY, ensure_ascii=False, indent=2)

PROMPT_WRAPUP_ICD = """\
You are a medical data extraction assistant. Extract fields from this Abbott \
Merlin PCS Wrap-up™ Report page.

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- patient_name: top centre header, the patient's name (format "Lastname, Firstname")
- followup_date: top-right corner (format "d Mon yyyy" e.g. "4 Apr 2025")
- device_model: top centre, model code (e.g. "3367-40QC" or "3562")
- device_serial: top centre, number after the model code

Patient section:
  - patient_dob ← "Geburtsdatum" value (e.g. "30 Sep 1939"); null if not shown

Aggregat table (rows: CRT-P/CRT-D/ICD/PM, A Elektrode, RV/V Elektrode, LV Elektrode):
  Columns are: Hersteller | Modell | S/N | Impl.datum
  - implant_date ← Impl.datum for the generator row (first row, e.g. "7 Jul 2026")
  - lead_a_model ← Modell for the "A Elektrode" row (e.g. "Tendril™ STS 2088TC / 52 cm")
  - lead_a_serial ← S/N for "A Elektrode" row (e.g. "EEL580765")
  - lead_a_implant_date ← Impl.datum for "A Elektrode" row
  - lead_rv_model ← Modell for the "RV Elektrode" or "V Elektrode" row
  - lead_rv_serial ← S/N for "RV Elektrode" or "V Elektrode" row
  - lead_rv_implant_date ← Impl.datum for RV/V Elektrode row
  - lead_lv_model ← Modell for the "LV Elektrode" row (CRT devices only; null if absent)
  - lead_lv_serial ← S/N for "LV Elektrode" row (null if absent)
  - lead_lv_implant_date ← Impl.datum for LV Elektrode row (null if absent)

Batterie section:
  - battery_voltage_v ← "Spannung" value (number before "V", e.g. "2,96")
  - battery_longevity ← "Laufzeit" value WITH unit (e.g. "1,5 Jahre")
  - battery_charging_time_s ← "Letzte max. Aufladung" value (number before "s")
  - battery_current_ua ← "Batteriestrom" value (number before "uA")
  - battery_remaining_pct ← "Verbleib. Kapazität bis ERI" value (number before "%")
  - battery_eri ← "ERI" value (preserve "> " prefix e.g. "> 5 Jhr.")

Testergebnisse section — rows for each lead (A, RV, LV) and HV coil:
  Each lead row shows: Reizschw. (threshold) | Signalamplitude (sensing) | Elektrodenimpedanz
  The threshold format is "X,XXV @ Y,Yms (config)" — extract amplitude and pulse width separately.
  - threshold_a_v ← A row Reizschwelle amplitude (number before "V"; "Nicht durchgeführt" if not tested)
  - threshold_a_pw_ms ← A row Reizschwelle pulse width (number after "@" before "ms")
  - sensing_a_mv ← A row Signalamplitude (number; "Nicht durchgeführt" if not tested)
  - impedance_a_ohm ← A row Elektrodenimpedanz (number before "Ω")
  - threshold_rv_v ← RV row Reizschwelle amplitude
  - threshold_rv_pw_ms ← RV row Reizschwelle pulse width
  - sensing_rv_mv ← RV row Signalamplitude
  - impedance_rv_ohm ← RV row Elektrodenimpedanz
  - threshold_lv_v ← LV row Reizschwelle amplitude (null if no LV lead)
  - threshold_lv_pw_ms ← LV row Reizschwelle pulse width (null if no LV lead)
  - sensing_lv_mv ← LV row Signalamplitude (null if no LV lead)
  - impedance_lv_ohm ← LV row Elektrodenimpedanz (null if no LV lead)
  - impedance_hv_ohm ← HV coil impedance (number; null if no HV coil)
  IMPORTANT: If the report shows only RV and LV rows (no A row), do NOT put LV values
  into the _a_ fields — use _rv_ and _lv_ prefixes matching the row labels.

Parameter section:
  - mode ← "Betriebsart" value (e.g. "DDD", "VVI", "VVIR")
  - base_rate_bpm ← "GF" or "Grundfrequenz" value (number only)
  - upper_rate_bpm ← "MTR" value (number only)
  - av_paced_ms ← "Stimuliertes AV-Intervall" or "AV stimuliert" value (number)
  - av_sensed_ms ← "Wahrg. AV-Int." or "AV wahrgenommen" value (number)
  - vt_zone ← lowest VT detection rate (number from "VT-1:" or "VT:" label, bpm)
  - vf_zone ← VF detection rate (number from "VF:" label, bpm)
  - therapy ← "Therapie" status (e.g. "Aus", "Ein")

Neue Programmierung section (changes made at this visit):
  - new_programming ← copy text summarising what was changed today (null if absent)

""" + json.dumps(SCHEMA_WRAPUP_ICD, ensure_ascii=False, indent=2)

PROMPT_PSA = """\
You are a medical data extraction assistant. Extract implant lead test data from \
this Abbott Merlin PCS PSA-Bericht (Pacing System Analyzer report).

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- patient_name: top centre header, the patient's name (format "Lastname, Firstname")
- implant_date: TOP-RIGHT corner of the page (format "d Mon yyyy", e.g. "8 Jul 2026")
- device_model: TOP-CENTRE header, model number after "Endurity MRI™ " or other device name
  (e.g. "2172" from "Endurity MRI™ 2172 Schrittmacher 4130170")
- device_serial: TOP-CENTRE header, the number after the model (e.g. "4130170")

Lead table — rows are labelled "A" (atrial) and "RV" (right ventricular):
  For each row (lead_a / lead_rv):
  - lead_*_model ← "Modell" column value (e.g. "Tendril™ STS 2088TC"); null if row is empty
  - lead_*_serial ← "Serien-Nr." column value (e.g. "EEL586541"); null if row is empty
  - lead_*_implant_date ← "Impl.datum" column value (e.g. "8 Jul 2026")

Testergebnisse section — three columns: "Reizschw." | "Signalamplitude" | "Impedanz":
  For A row (threshold_a / sensing_a / impedance_a):
  - threshold_a_v ← Reizschw. amplitude: the NUMBER before "V bei" (e.g. "0,9")
  - threshold_a_pw_ms ← Reizschw. pulse width: the NUMBER after "bei" and before "ms"
    (e.g. "0,4" from "0,9 V bei 0,4 ms, 1,7 mA")
  - threshold_a_ma ← Reizschw. current: the NUMBER after the comma, before "mA"
    (e.g. "1,7")
  - sensing_a_mv ← Signalamplitude: the NUMBER before "mV" (e.g. "5,3")
  - sensing_a_slew_vs ← Signalamplitude: the NUMBER before "V/s" (e.g. "1,7")
  - impedance_a_ohm ← Impedanz: the NUMBER before "Ω" or "Ohm" (e.g. "477")
  For RV row — same pattern (threshold_rv / sensing_rv / impedance_rv).
  If a row is entirely absent (single-chamber device with no atrial lead), return null for all A fields.

""" + json.dumps(SCHEMA_PSA, ensure_ascii=False, indent=2)

PROMPT_PARAMETER_ICD = """\
You are a medical data extraction assistant. Extract ICD programming parameters \
from this Abbott Merlin PCS Parameter page (Seite 1 von N).

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- patient_name: top centre header, the patient's name (format "Lastname, Firstname")
- followup_date: TOP-RIGHT corner of the header (format "d Mon yyyy", e.g. "8 Jul 2026")
- device_model: top-centre header, the model code AFTER "Entrant™ DR " or similar ICD name
  (e.g. "CDDRA300Q" from "Entrant™ DR CDDRA300Q ICD 810187613")
- device_serial: top-centre header, the NUMBER after the model code (e.g. "810187613")

Aggregat table (rows: ICD, A Elektrode, V Elektrode — columns: Hersteller/Modell/S/N/Impl.datum):
- implant_date ← Impl.datum for the ICD row (e.g. "27 Mär 2026")
- lead_a_model ← Modell for the "A Elektrode" row (e.g. "Tendril™ STS 2088TC / 52 cm");
  strip the length suffix (e.g. "/ 52 cm") and return just the model name
- lead_a_serial ← S/N for "A Elektrode" row (e.g. "EDG165091")
- lead_a_implant_date ← Impl.datum for "A Elektrode" row (e.g. "27 Mär 2026")
- lead_rv_model ← Modell for the "V Elektrode" row (e.g. "Durata™ 7122Q");
  strip the length suffix and return just the model name
- lead_rv_serial ← S/N for "V Elektrode" row (e.g. "BPA182474")
- lead_rv_implant_date ← Impl.datum for "V Elektrode" row

Modus & Sensor section:
- mode ← "Betriebsart" value (e.g. "DDD", "VVI")

Frequenzen section:
- base_rate_bpm ← "Grundfrequenz (GF)" value, number only (e.g. "60")
- upper_rate_bpm ← "Obere Grenzfrequenz (MTR)" value, number only (e.g. "100")

Intervalle section:
- av_paced_ms ← "Stimuliertes AV-Intervall" value, number only (e.g. "180")
- av_sensed_ms ← "Wahrg. AV-Int." value, number only (e.g. "150")

Stimulation und Wahrnehmung section — two columns (A and V):
- amplitude_a_v ← "Impulsamplitude (Marge)" row, A column value, number only (e.g. "3,125")
- pulse_width_a_ms ← "Impulsdauer" row, A column value, number only (e.g. "0,5")
- amplitude_rv_v ← "Impulsamplitude" row, V column value — if formatted as "X,X V (Y:1)"
  extract only the amplitude before " V " (e.g. "3,0" from "3,0 V (2.4:1)")
- pulse_width_rv_ms ← "Impulsdauer" row, V column value, number only (e.g. "1,0")

""" + json.dumps(SCHEMA_PARAMETER_ICD, ensure_ascii=False, indent=2)

PROMPT_TESTERGEBNISSE_RV = """\
You are a medical data extraction assistant. Extract the most recent ventricular \
pacing threshold from this Abbott Merlin ICD threshold test page \
(Testergebnisse: Ventr. Reizschwelle).

""" + _COMMON_RULES + """
WHERE TO FIND EACH FIELD:
- followup_date: TOP-RIGHT corner (format "d Mon yyyy", e.g. "8 Jul 2026")
- threshold_rv_v: the HIGHLIGHTED (bold/boxed) threshold amplitude in V — the most
  recent test result. Format: "X,XX V @ Y,Y ms" → extract only the amplitude before " V"
  (e.g. "1,75" from "1,75 V @ 0,5 ms (Bi)")
- threshold_rv_pw_ms: the pulse width in ms from the same highlighted/most recent line
  (e.g. "0,5" from "1,75 V @ 0,5 ms (Bi)")
  If multiple threshold rows exist, use the one with the LATEST date.

""" + json.dumps(SCHEMA_TESTERGEBNISSE_RV, ensure_ascii=False, indent=2)


# ── report type detection ─────────────────────────────────────────────────────

def detect_report_type(pdf_path: str | Path) -> str:
    """
    Detect report type from meta.json, PDF footer text, or path hints.
    Returns: 'freeze' | 'fastpath_icd' | 'fastpath_pm' | 'summary' | None
    """
    import json as _json
    import pdfplumber

    pdf_path = Path(pdf_path)

    # 1. Check meta.json in sample directory
    meta_path = pdf_path.parent / "meta.json"
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        rt = meta.get("report_type", "")
        dt = meta.get("device_type", "")
        if rt == "FreezeReport":
            return "freeze"
        if rt == "SummaryReport":
            return "summary"
        if rt == "PSAReport":
            return "psa"
        # Only trust device_type when report_type is explicitly annotated as FastPathSummary.
        # ICD/CRT-D Complete exports have Parameters pages (not FastPath) so they must be
        # classified via VQA rather than assumed to be fastpath_icd.
        if rt == "FastPathSummary":
            if "ICD" in dt or "S-ICD" in dt or "CRT-D" in dt:
                return "fastpath_icd"
            if "Pacemaker" in dt or "PM" in dt or "CRT-P" in dt:
                return "fastpath_pm"

    # 2. Try PDF text (works only if footer is text-based)
    with pdfplumber.open(str(pdf_path)) as pdf:
        text = (pdf.pages[0].extract_text() or "").lower()

    if "ausschnitt" in text or "freeze" in str(pdf_path).lower():
        return "freeze"
    if "icd" in text:
        return "fastpath_icd"
    if "schrittmacher" in text:
        return "fastpath_pm"

    return None  # unknown / unsupported report type

# ── VQA inference ─────────────────────────────────────────────────────────────

def _vqa(model, processor, image, prompt: str, max_new_tokens: int = 1024) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    import torch
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    think_token_id = processor.tokenizer.convert_tokens_to_ids("<think>")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            suppress_tokens=[think_token_id] if isinstance(think_token_id, int) else None,
        )
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def _parse_json(raw: str) -> dict:
    # Strip Qwen3 extended-thinking blocks before searching for JSON
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    # Find the full JSON block (greedy: from first { to last })
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return json.loads(raw)
    candidate = m.group()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fix unquoted German decimal numbers: "key": 0,8, → "key": "0,8",
    # The VQA model omits quotes around decimal comma numbers (e.g. "0,8" → 0,8)
    fixed = re.sub(r":\s*([-+]?\d+),(\d+)([,\s}\]])", r': "\1,\2"\3', candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Also fix trailing commas
    fixed2 = re.sub(r",\s*([}\]])", r"\1", fixed)
    return json.loads(fixed2)

_UNIT_SUFFIX = re.compile(
    r"\s+(min[⁻\-]1|min⁻¹|mV|Ohm|uA|UA)\s*$"
    r"|(?<=[0-9])\s+ms$"    # "0,4 ms" but not standalone text "ms"
    r"|(?<=[0-9])\s+V$"     # "2,5 V" but not "Ein" or "Aus"
    r"|\s*%$"               # ">95 %" or ">95%" → ">95"
)

_KEY_RENAME = {
    "amplitude_v_v": "amplitude_rv_v",
    "pulse_width_v_ms": "pulse_width_rv_ms",
    "sensitivity_v_mv": "sensitivity_rv_mv",
    "autocapture_v": "autocapture_rv",
    "sense_ability_v": "sense_ability_rv",
    "sensing_config_v": "sensing_config_rv",
    "stim_v_pct": "stim_rv_pct",
    "sensing_v": "sensing_rv_mv",
    "impedance_v": "impedance_rv_ohm",
    "impedance_hv": "shock_impedance_ohm",
    "lead_v_model": "lead_rv_model",
    "lead_v_serial": "lead_rv_serial",
    "lead_v_implant_date": "lead_rv_implant_date",
}


def _postprocess(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        k = _KEY_RENAME.get(k, k)
        if isinstance(v, str):
            v = _UNIT_SUFFIX.sub("", v).strip()
            # Strip "Seit " prefix from date fields
            if k.endswith("_since") or k == "diagnostics_since":
                v = re.sub(r"^Seit\s+", "", v)
        cleaned[k] = v
    return cleaned


# ── VQA-based report-type classification ──────────────────────────────────────
# Abbott reports are image-based, so detect_report_type()'s pdfplumber text pass
# fails at runtime (meta.json only exists in the validate harness). Ask the model
# to classify the page instead — robust on rasterised reports.

PROMPT_CLASSIFY = """\
Classify this Abbott Merlin PCS report page into exactly ONE category. Reply with ONLY the \
lowercase label, nothing else.

Categories:
1. psa             – titled "PSA-Bericht"; shows a lead table (Elektrode/Hersteller/Modell/
                     Serien-Nr./Impl.datum) and Testergebnisse with Reizschw./Signalamplitude/
                     Impedanz columns; no battery section; generated at implantation
2. freeze          – page shows EGM/ECG traces or is titled "Ausschnitt" / "FreezeReport"
3. summary         – device is an Aveir™ DR LP or leadless pacemaker (LP / LSPxxx codes)
4. fastpath_icd    – compact single-page FastPath™ Uebersicht for an ICD; contains
                     "Batterie", "Testergebnisse", "Parameter" AND "Diagnostikuebersicht"
                     (with VT/VF episode counts) all visible on one page
5. fastpath_pm     – compact single-page FastPath™ Uebersicht for a pacemaker; same
                     structure but Diagnostikuebersicht shows AP%/VP% pacing statistics
6. wrapup          – Wrap-up™ Report: shows "Batterie" + "Testergebnisse" + "Parameter" +
                     "Neue Programmierung"; does NOT have a "Diagnostikuebersicht" section
7. parameters_icd  – detailed ICD programming tables ("Modus & Sensor", "Frequenzen",
                     "Refraktaerzeiten", "Elektroden"); has VT/VF zone config but NO
                     battery summary section; device is an ICD/CRT-D
8. testergebnisse_icd – TITLED "Testergebnisse: Ventr. Reizschwelle" or similar;
                     shows EGM/ECG traces of a ventricular pacing threshold test;
                     top text shows threshold value like "X,XX V @ Y,Y ms"; NO battery
                     section, NO parameter table, NO FastPath Diagnostikübersicht

Label:"""


def classify_report_type(model, processor, image) -> Optional[str]:
    """VQA report-type detection for image-based reports (no text layer)."""
    raw = _vqa(model, processor, image, PROMPT_CLASSIFY).strip().lower()
    # Check most-specific labels first (psa and parameters_icd before generic ones)
    for rt in ("testergebnisse_icd", "parameters_icd", "fastpath_icd", "fastpath_pm", "summary", "freeze", "wrapup", "psa"):
        if rt in raw or rt.replace("_", " ") in raw or rt.replace("_", "") in raw:
            return rt
    # loose keyword fallback if the model answered descriptively
    if "psa" in raw or "pacing system" in raw or "reizschw" in raw:
        return "psa"
    if "ausschnitt" in raw or "freeze" in raw:
        return "freeze"
    if "aveir" in raw or "leadless" in raw or "summary" in raw:
        return "summary"
    if "wrap" in raw or "neue programmierung" in raw:
        return "wrapup"
    if "testergebnisse" in raw and ("reizschwelle" in raw or "ventr" in raw or "threshold" in raw):
        return "testergebnisse_icd"
    if "refraktär" in raw or "refraktion" in raw or "modus & sensor" in raw:
        return "parameters_icd"
    if "icd" in raw or "defibrill" in raw:
        return "fastpath_icd"
    if "schrittmacher" in raw or "pacemaker" in raw:
        return "fastpath_pm"
    return None


# ── fast header-crop classification ──────────────────────────────────────────
# Render only top 30% of the page and classify with max_new_tokens=20.
# ~0.5s vs ~6-8s for the full-page VQA classification above.

CLASSIFY_PROMPT_FAST = """\
Look at this report page. Reply with ONLY one word — the report type:
- fastpath_pm (FastPath pacemaker overview: Batterie/Testergebnisse/Parameter/Diagnostik, device is Schrittmacher)
- fastpath_icd (FastPath ICD overview: same sections but device is ICD/Defibrillator, has VT/VF)
- summary (Aveir DR LP leadless pacemaker — has RA+RV battery blocks, Testergeb., Stimulation & Wahr.)
- freeze (FreezeReport/Ausschnitt — shows ECG traces and Schlüsselparameter table)
- psa (PSA-Bericht — lead table with Elektrode/Hersteller/Modell columns, implant test results)
- wrapup (Wrap-up Report — Batterie + Testergebnisse + Neue Programmierung, no Diagnostik)
- parameters_icd (Parameter page — Modus & Sensor, Frequenzen, Intervalle tables, no battery section)
- testergebnisse_icd (Testergebnisse ventr. Reizschwelle — ECG traces of threshold test)

Type:"""


def classify_report_type_fast(model, processor, image) -> Optional[str]:
    """Classify by cropping top 30% of the page — ~0.5s instead of ~6-8s."""
    w, h = image.size
    crop = image.crop((0, 0, w, int(h * 0.30)))
    raw = _vqa(model, processor, crop, CLASSIFY_PROMPT_FAST, max_new_tokens=20)
    raw = raw.strip().lower()
    for rt in ("testergebnisse_icd", "parameters_icd", "fastpath_icd",
               "fastpath_pm", "summary", "freeze", "wrapup", "psa"):
        if rt in raw:
            return rt
    return None


# ── single-pass unified prompt ───────────────────────────────────────────────

SCHEMA_UNIFIED = {
    "report_type": "FastPathSummary | FreezeReport | SummaryReport | WrapupReport | PSAReport | ParameterReport",
    "patient_name": None,
    "followup_date": None,
    "implant_date": None,
    "device_model": None,
    "device_serial": None,
    "device": None,
    "facility": None,
    # Battery
    "battery_voltage_v": None,
    "battery_mag_freq_bpm": None,
    "battery_remaining_pct": None,
    "battery_eri": None,
    "battery_current_ua": None,
    "battery_longevity": None,
    "battery_charging_time_s": None,
    # Test results / sensing / impedance
    "threshold_a_v": None,
    "threshold_a_pw_ms": None,
    "threshold_rv_v": None,
    "threshold_rv_pw_ms": None,
    "threshold_lv_v": None,
    "threshold_lv_pw_ms": None,
    "sensing_a_mv": None,
    "sensing_rv_mv": None,
    "sensing_lv_mv": None,
    "impedance_a_ohm": None,
    "impedance_rv_ohm": None,
    "impedance_lv_ohm": None,
    "shock_impedance_ohm": None,
    # Leads (PSA / Parameter / Wrap-up reports)
    "lead_a_model": None,
    "lead_a_serial": None,
    "lead_a_implant_date": None,
    "lead_rv_model": None,
    "lead_rv_serial": None,
    "lead_rv_implant_date": None,
    "lead_lv_model": None,
    "lead_lv_serial": None,
    "lead_lv_implant_date": None,
    # Patient
    "patient_dob": None,
    # Programming
    "mode": None,
    "base_rate_bpm": None,
    "upper_rate_bpm": None,
    "av_paced_ms": None,
    "av_sensed_ms": None,
    # Stimulation
    "autocapture_a": None,
    "autocapture_rv": None,
    "autocapture_lv": None,
    "amplitude_a_v": None,
    "amplitude_rv_v": None,
    "amplitude_lv_v": None,
    "pulse_width_a_ms": None,
    "pulse_width_rv_ms": None,
    "pulse_width_lv_ms": None,
    "sense_ability_a": None,
    "sense_ability_rv": None,
    "sense_ability_lv": None,
    "sensitivity_a_mv": None,
    "sensitivity_rv_mv": None,
    "sensitivity_lv_mv": None,
    # ICD zones
    "vt_zone": None,
    "vf_zone": None,
    "therapy": None,
    # Diagnostics
    "stim_a_pct": None,
    "stim_rv_pct": None,
    "stim_lv_pct": None,
    "ams_episodes": None,
    "mode_switch_pct": None,
    "af_burden_pct": None,
    "diagnostics_since": None,
    "vt_vf_episodes": None,
    # Freeze-specific
    "max_sensor_rate_bpm": None,
    "rate_hysteresis": None,
    "vip": None,
    "rate_adaptive_av": None,
    "rate_adaptive_pvarp": None,
    "vent_safety_pacing": None,
    # PSA-specific
    "threshold_a_ma": None,
    "sensing_a_slew_vs": None,
    "threshold_rv_ma": None,
    "sensing_rv_slew_vs": None,
}

PROMPT_UNIFIED = """\
You are a medical data extraction assistant. Extract ALL visible pacemaker/ICD \
interrogation fields from this Abbott Merlin PCS report page.

STEP 1 — Identify the report type from the page layout:
- "FastPathSummary": compact single-page overview with Batterie + Testergebnisse + \
Parameter + Diagnostikübersicht sections
- "FreezeReport": titled "Ausschnitt", shows ECG traces + Schlüsselparameter table
- "SummaryReport": Aveir™ DR LP device, has Batterieinformationen (RA+RV blocks), \
Testergeb., Parameter, Stimulation & Wahr., Diagnostikübersicht
- "WrapupReport": titled "Wrap-up™ Report", has Batterie + Testergebnisse + Parameter + \
Neue Programmierung
- "PSAReport": titled "PSA-Bericht", has lead table + Testergebnisse (Reizschw./Signalamplitude/Impedanz)
- "ParameterReport": detailed programming tables (Modus & Sensor, Frequenzen, Intervalle)

STEP 2 — Extract fields. Rules:
- NUMERIC fields: number only, NO unit suffix. "60 min⁻¹" → "60", "2,5 V" → "2,5", \
"550 Ω" → "550". Use German decimal comma.
- Preserve ">" or "<" prefixes (e.g. ">95", "> 5 Jhr.").
- TEXT fields (Aus, Ein, Auto, DDD, DDDR…): copy exactly as printed.
- DATE fields: strip leading "Seit" — "Seit 3 Mär 2025" → "3 Mär 2025".
- "Nicht durchgeführt" → return that exact string (NOT null).
- null = field label is not present on this page at all.
- Return ONLY the JSON — no markdown fences, no explanation.

FIELD LOCATIONS:
- Header: patient_name (top centre), followup_date (top-right corner, "d Mon yyyy"), \
device_model (model number/code), device_serial (number after model), \
device (full device line for Aveir™), facility (line starting "HZ"), \
implant_date (for PSA: top-right; for others: "Datum der Implantation")
- Batterie: battery_voltage_v ("Spannung"), battery_mag_freq_bpm ("Magnetfrequenz"), \
battery_remaining_pct ("Verbleib. Kapazität"), battery_eri ("ERI"), \
battery_current_ua ("Batteriestrom"), battery_longevity ("Laufzeit"), \
battery_charging_time_s ("Letzte max. Aufladung")
- Testergebnisse: threshold_*_v ("Reizschwelle" amplitude), threshold_*_pw_ms (pulse width), \
sensing_*_mv ("Signalamplitude"), impedance_*_ohm ("Elektrodenimpedanz"), \
shock_impedance_ohm (HV coil), sensing_*_slew_vs (V/s), threshold_*_ma (mA). \
CRT devices have A, RV, and LV rows — use _a_, _rv_, _lv_ suffixes. If a report \
shows only RV and LV (no A), do NOT put LV values into _a_ fields.
- Patient: patient_dob ← "Geburtsdatum" value (e.g. "30 Sep 1939"); null if not shown
- Leads (PSA/Parameter/Wrap-up): lead_a_model, lead_a_serial, lead_a_implant_date (A row), \
lead_rv_* (V/RV row), lead_lv_* (LV row, CRT devices only; null if absent)
- Parameter: mode ("Betriebsart"), base_rate_bpm ("Grundfrequenz/GF"), \
upper_rate_bpm ("Ob Grenzfr/MTR"), av_paced_ms ("AV stimuliert"), \
av_sensed_ms ("Wahrg. AV-Int.")
- Stimulation: amplitude_*_v ("Impulsamplitude"), pulse_width_*_ms ("Impulsdauer"), \
autocapture_* ("ACap/AutoCapture"), sense_ability_* ("SenseAbility™"), \
sensitivity_*_mv ("Empfindlichkeit/Sensitivität"). \
CRT devices: use _a_, _rv_, _lv_ suffixes for A, RV, LV columns.
- ICD zones: vt_zone, vf_zone ("Zonenkonfiguration"), therapy ("Therapie")
- Diagnostik: stim_a_pct (AP%), stim_rv_pct (VP%), stim_lv_pct (LV%, CRT only), \
ams_episodes ("AMS-Episoden"), mode_switch_pct ("Mode-Switch"), \
af_burden_pct ("AT/AF-Burden"), diagnostics_since ("Seit:" date), \
vt_vf_episodes ("VT/VF-Episoden")
- Freeze: max_sensor_rate_bpm, rate_hysteresis, vip ("VIP™"), \
rate_adaptive_av, rate_adaptive_pvarp, vent_safety_pacing

""" + json.dumps(SCHEMA_UNIFIED, ensure_ascii=False, indent=2)


def extract_single_pass(
    pdf_path: str | Path,
    model=None,
    processor=None,
    dpi: int = 200,
) -> dict:
    pdf_path = Path(pdf_path)
    if model is None or processor is None:
        raise ValueError("model and processor must be provided")

    images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    raw = _vqa(model, processor, images[0], PROMPT_UNIFIED)
    result = _parse_json(raw)
    result = _postprocess(result)
    result = {_KEY_RENAME.get(k, k): v for k, v in result.items()}
    result["source_file"] = str(pdf_path)
    result["manufacturer"] = "Abbott"
    result["vqa_model"] = "Qwen3-VL-8B-Instruct"
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def _page_count(pdf_path: Path) -> int:
    import pdfplumber
    with pdfplumber.open(str(pdf_path)) as pdf:
        return len(pdf.pages)


def extract(
    pdf_path: str | Path,
    model=None,
    processor=None,
    dpi: int = 200,
    report_type: Optional[str] = None,
) -> dict:
    """
    Extract from an Abbott Merlin PCS report PDF using VQA.

    model / processor must be pre-loaded Qwen2-VL objects (call vqa.load_model()).
    If report_type is None, it is auto-detected.
    """
    pdf_path = Path(pdf_path)
    if model is None or processor is None:
        raise ValueError("model and processor must be provided (load via vqa.load_model())")

    images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    image = images[0]
    threshold_result: dict = {}  # populated when page 1 is testergebnisse_icd

    # Resolve report type: cheap detectors first (meta.json / PDF text), then VQA
    # classification of page 1, and finally fall back to 'summary' so extraction
    # always proceeds.
    if report_type is None:
        report_type = detect_report_type(pdf_path)
    if report_type is None:
        report_type = classify_report_type_fast(model, processor, image)
    if report_type is None:
        report_type = "summary"

    # ICD Complete exports: page 1 is a Parameters page, not a FastPath summary.
    # Try the last page for a Wrap-up™ Report which has battery + test results.
    if report_type == "parameters_icd":
        n = _page_count(pdf_path)
        if n > 1:
            last_imgs = pdf2image.convert_from_path(
                str(pdf_path), dpi=dpi, first_page=n, last_page=n
            )
            last_type = classify_report_type_fast(model, processor, last_imgs[0])
            if last_type == "wrapup":
                image = last_imgs[0]
                report_type = "wrapup"
            # else: keep parameters_icd — extract what's available from page 1

    # Testergebnisse (ECG threshold test) as page 1: extract threshold from p1,
    # then use page 2 (Parameter page) for device/lead/programming data.
    if report_type == "testergebnisse_icd":
        n = _page_count(pdf_path)
        threshold_result = {}
        if n > 1:
            p2_imgs = pdf2image.convert_from_path(
                str(pdf_path), dpi=dpi, first_page=2, last_page=2
            )
            p2_type = classify_report_type_fast(model, processor, p2_imgs[0])
            if p2_type in ("parameters_icd", "testergebnisse_icd", None):
                # Extract threshold from page 1
                raw_thr = _vqa(model, processor, image, PROMPT_TESTERGEBNISSE_RV)
                try:
                    threshold_result = _parse_json(raw_thr)
                    threshold_result = _postprocess(threshold_result)
                except Exception:
                    threshold_result = {}
                # Extract programming parameters from page 2
                image = p2_imgs[0]
                report_type = "parameters_icd"

    if report_type == "psa":
        prompt = PROMPT_PSA
    elif report_type == "freeze":
        prompt = PROMPT_FREEZE
    elif report_type == "fastpath_icd":
        prompt = PROMPT_FASTPATH_ICD
    elif report_type == "fastpath_pm":
        prompt = PROMPT_FASTPATH_PM
    elif report_type == "wrapup":
        prompt = PROMPT_WRAPUP_ICD
    elif report_type == "parameters_icd":
        # No Wrap-up page found; extract device/leads/programming from Parameters page.
        prompt = PROMPT_PARAMETER_ICD
    else:
        prompt = PROMPT_SUMMARY

    raw = _vqa(model, processor, image, prompt)
    result = _parse_json(raw)
    # SummaryReport keeps units in values; only strip for other report types
    if report_type not in ("summary",):
        result = _postprocess(result)

    # Merge threshold test data from testergebnisse page (if extracted above)
    for k, v in threshold_result.items():
        if k not in ("report_type", "source_file", "manufacturer", "vqa_model") and v is not None:
            result.setdefault(k, v)
            if result.get(k) is None:
                result[k] = v

    # Multi-page FastPath documents: the last page is often a Wrap-up™ Report
    # with supplementary data (lead serials/models, patient DOB, battery details).
    # Extract from it and merge fields that page 1 doesn't have.
    _SKIP_MERGE = {"report_type", "source_file", "manufacturer", "vqa_model",
                   "patient_name", "followup_date", "device_model", "device_serial"}
    if report_type in ("fastpath_pm", "fastpath_icd"):
        n = _page_count(pdf_path)
        if n > 1:
            last_imgs = pdf2image.convert_from_path(
                str(pdf_path), dpi=dpi, first_page=n, last_page=n
            )
            last_type = classify_report_type_fast(model, processor, last_imgs[0])
            if last_type == "wrapup":
                raw_wu = _vqa(model, processor, last_imgs[0], PROMPT_WRAPUP_ICD)
                try:
                    wu = _parse_json(raw_wu)
                    wu = _postprocess(wu)
                    wu = {_KEY_RENAME.get(k, k): v for k, v in wu.items()}
                    for k, v in wu.items():
                        if k in _SKIP_MERGE or v is None:
                            continue
                        if result.get(k) is None:
                            result[k] = v
                except Exception:
                    pass

    result = {_KEY_RENAME.get(k, k): v for k, v in result.items()}
    result["source_file"] = str(pdf_path)
    result["manufacturer"] = "Abbott"
    result["vqa_model"] = "Qwen3-VL-8B-Instruct"
    return result
