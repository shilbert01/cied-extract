# Changelog

All notable changes to cied-extract are documented here.

## [0.3.2] — 2026-10-25

### Improved

- **Abbott VQA prompts**: clarified that the interrogation time is printed on a separate line below the date in the header — all prompts now instruct the model to combine date + time as "d Mon yyyy HH:MM".
- **Abbott multi-page**: Wrapup page scan now runs for all report types (summary, freeze, parameters, etc.), not just FastPath.

## [0.3.1] — 2026-10-24

### Added

- **Abbott Wrapup**: lead detail fields — model, serial, implant date for A/RV/LV electrodes.
- **Abbott Wrapup**: patient date of birth (`patient_dob`), battery voltage, RV and LV threshold fields.
- **Abbott Wrapup**: improved prompt with structured Aggregat table extraction.
- **Abbott Unified**: LV lead fields (`lead_lv_model`, `lead_lv_serial`, `lead_lv_implant_date`) and `patient_dob`.
- **Abbott FastPath**: automatic Wrapup page detection and merge — multi-page FastPath PDFs now extract supplementary data (lead serials, patient DOB, battery details) from the trailing Wrapup page.

## [0.3.0] — 2026-10-24

Initial public release.

### Added

- **Abbott / St. Jude**: VQA extraction from image-based PDFs using Qwen3-VL-8B-Instruct (4-bit quantized). Supports SummaryReport, FreezeReport, FastPathSummary, ParameterReport (ICD), and PSAReport.
- **Abbott / St. Jude**: CPU log parser for Merlin PCS `.log` exports.
- **Abbott / St. Jude**: PaddleOCR fallback extractor for FastPath reports (CPU-only).
- **Biotronik**: PDF extractor (pdfplumber + regex).
- **Biotronik**: IEEE 11073 XML extractor.
- **Biotronik**: BioStd XML extractor.
- **Boston Scientific**: CRT-D PDF extractor.
- **Boston Scientific**: S-ICD extractor (PDF + HL7 v2.6).
- **Boston Scientific**: PACEART `.bnk` binary parser.
- **ELA / Sorin / LivaNova**: PDF extractor.
- **Medtronic**: CareLink PDF extractor.
- **Medtronic**: 2090 programmer `.26X`/`.V26` text format parser.
- **Medtronic**: `.pdd` binary parser.
- **Medtronic**: `.pkg` ZIP archive parser (PublicDiscreteData.xml).
- **Microport / PACEART**: XML parser.
- FastAPI HTTP service with `POST /v1/extract` (sync and async modes).
- Docker deployment with GPU and CPU-only compose profiles.
- Ground-truth validation harness (`validate.py`).
- Annotation web tool for building ground-truth datasets.
- Fast header-crop classification for Abbott reports (~0.5s vs ~6-8s full-page).
- Thinking token suppression for faster VQA inference.
- 200 DPI rendering for Abbott PDFs (down from 300) without accuracy loss.
- VQA model lazy-loaded on first Abbott request.
- Schema version 1.0, engine version 0.3.0.
