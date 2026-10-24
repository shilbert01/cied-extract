# Changelog

All notable changes to cied-extract are documented here.

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
