# Changelog

All notable changes to cied-extract are documented here.

## [0.4.0] — 2026-10-30

### Added

- **Selectable VQA backend**: new `CIED_EXTRACT_MODEL` environment variable lets operators choose between `qwen-vl-8b` (default, ~6 GB VRAM, fits 3080 Ti) and `qwen3.8-27b` (~18 GB VRAM, needs RTX 4090+). The 27B model scores 97.5% vs 94.2% for the 8B model on Abbott reports and is ~30% faster per page.
- **Ollama VQA backend** (`vqa.OllamaVQA`): drop-in replacement for the transformers model path. Sends images to an external Ollama instance via `/api/chat` with automatic context window scaling for multi-page documents. Configured via `CIED_EXTRACT_OLLAMA_URL` and `CIED_EXTRACT_OLLAMA_MODEL`.
- **Health endpoint**: `/health` now reports which VQA model is active (`vqa_model` field).
- **Benchmark harness**: `Qwen38Backend` in `benchmark/bench.py` replicates the full production pipeline (VQA classification, multi-page extraction, wrapup merge, key renaming) against Ollama for head-to-head evaluation.

### Improved

- **Abbott `_vqa()` dispatch**: transparently routes to transformers or Ollama based on the model type — no changes needed in report-type-specific extraction logic.
- **GPU lock skipped for Ollama**: when using the 27B backend, the in-process GPU serialization lock is bypassed since inference runs in the external Ollama container.

## [0.3.5] — 2026-10-28

### Added

- **Abbott PSA Report**: `followup_date` (date + time), `device_name`, and `device_class` fields added to PSA schema and prompt.
- **Abbott lead serial dedup**: `_dedup_lead_serials()` postprocessing safety net nulls out cross-row serial contamination — handles both "one row has model" and "no row has model" cases.
- **Abbott placeholder handling**: `<not set>` and box-glyph characters in lead table cells are now converted to null via `_PLACEHOLDER` regex.
- **Abbott Summary schema backfill**: `_backfill_schema()` ensures all schema blocks appear in the output even when the VQA model omits a section.

### Improved

- **Abbott PSA lead table prompt**: each row treated as independent — explicit rules prevent cross-row serial/model/date contamination.
- **Abbott Summary token limit**: `max_new_tokens` raised from 1024 to 2048 to accommodate the large nested schema.

## [0.3.4] — 2026-10-27

### Added

- **Abbott device identification**: `device_name` (commercial name, e.g. "Quadra Allure MP™") and `device_class` (e.g. "ICD", "CRT-P", "Schrittmacher") fields added to FastPath ICD, FastPath PM, Summary, Wrapup, and Parameter ICD schemas and prompts. The header is now parsed as `<name> <model> <class> <serial>` consistently across all report types.

### Improved

- **Abbott Testergebnisse stacked values**: FastPath ICD, FastPath PM, and Summary prompts now explicitly instruct the model to extract only the upper (current session) line from stacked test cells, never falling back to the lower (previous session) line.

## [0.3.3] — 2026-10-27

### Added

- **Abbott ICD Parameter Report**: 12 new fields — `patient_dob`, `max_sensor_rate_bpm`, `autocapture_v`, `sense_ability_v`, `sensing_config_v`, `zone_vt1_interval_ms`, `zone_vt2_interval_ms`, `zone_vf_interval_ms`, `atp_pulse_amplitude_v`, `atp_pulse_width_ms`, `shock_config`, `impedance_hv_ohm`.
- **Multi-page VQA**: Parameter ICD reports now render all pages and pass them to Qwen3-VL in a single multi-image call, enabling extraction of zone config (page 2), ATP details, and shock settings (page 3).
- **Multi-backend benchmark harness** (`benchmark/bench.py`): evaluates `qwen-vl`, `oxr+llm` (OXR markdown + Ollama qwen3:8b), and `deepseek-ocr` backends against ground truth with phased caching.

### Improved

- **Abbott Parameter ICD prompt**: lead model now includes length suffix (e.g. "Durata™ 7122Q / 65 cm"); `_vqa()` accepts single image or list of images.
- **GPU pinning**: `docker-compose.api.yml` sets `NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` for explicit GPU selection.

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
