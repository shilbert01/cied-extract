# cied-extract

Structured data extraction from cardiac implantable electronic device (CIED) interrogation reports.

Takes PDF reports, programmer log files, and export archives from pacemaker/ICD/CRT-D programmers and returns structured JSON with clinical parameters — battery status, thresholds, sensing, impedance, programming, and diagnostics.

## Supported Manufacturers

| Manufacturer | Input Formats | Method |
|---|---|---|
| **Abbott / St. Jude** | PDF (image-based), `.log` (Merlin PCS) | Qwen3-VL-8B or Qwen3.8-27B VQA (GPU), CPU log parser |
| **Biotronik** | PDF, XML (IEEE 11073, BioStd) | pdfplumber + regex, XML parser |
| **Boston Scientific** | PDF, `.bnk` (PACEART), HL7 v2.6 | pdfplumber + regex, binary parser |
| **ELA / Sorin / LivaNova** | PDF | pdfplumber + regex |
| **Medtronic** | PDF (CareLink), `.pdd`, `.pkg`, `.xml`, `.26X`/`.V26` | pdfplumber + regex, binary parsers |
| **Microport / PACEART** | XML | XML parser |

## Quick Start

### Docker (recommended)

```bash
# Build
docker build -t cied-extract .

# Run with Qwen3-VL-8B (default — fits 3080 Ti / 12 GB VRAM)
docker compose -f docker-compose.api.yml up -d --build

# Run with Qwen3.8-27B via Ollama (needs RTX 4090 / 24 GB VRAM, higher accuracy)
CIED_EXTRACT_MODEL=qwen3.8-27b docker compose -f docker-compose.api.yml up -d --build

# Health check (shows active model)
curl http://localhost:8082/health

# Extract a PDF
curl -s -X POST \
  -F "pdf=@report.pdf" \
  -F "vendor=biotronik" \
  http://localhost:8082/v1/extract | jq .
```

### CPU-only (no GPU, no Abbott VQA)

```bash
docker compose -f docker-compose.api.yml -f docker-compose.cpu.yml up -d --build
```

## API

**`POST /v1/extract`** — extract structured data from a report.

Content types:
- `application/pdf` — raw PDF body, set vendor via `X-Vendor-Hint` header
- `multipart/form-data` — fields: `pdf`, `log` (sidecar/primary), `hl7` (BSC S-ICD), `vendor`

```bash
# PDF with vendor hint header
curl -s -X POST \
  -H "Content-Type: application/pdf" \
  -H "X-Vendor-Hint: abbott" \
  --data-binary @report.pdf \
  http://localhost:8082/v1/extract

# Multipart with log file
curl -s -X POST \
  -F "log=@export.bnk" \
  -F "vendor=boston_scientific" \
  http://localhost:8082/v1/extract

# Async mode
curl -s -X POST \
  -F "pdf=@report.pdf" \
  -F "vendor=biotronik" \
  "http://localhost:8082/v1/extract?mode=async"
# → {"job_id": "...", "poll": "/v1/jobs/..."}
```

Response:

```json
{
  "schema_version": "1.0",
  "request_id": "...",
  "status": "ok",
  "engine": {
    "name": "cied-extract",
    "version": "0.4.0",
    "extractor": "Biotronik",
    "model": "pdfplumber"
  },
  "processing_ms": 420,
  "report": {
    "report_type": "FollowUpReport",
    "patient_name": "...",
    "followup_date": "...",
    "device_model": "...",
    "battery_voltage_v": "3,07",
    "mode": "DDD",
    "..."
  }
}
```

**`GET /health`** — service status and supported vendors.

**`GET /v1/jobs/{id}`** — poll async job status.

See [`docs/extract-api-v1.md`](docs/extract-api-v1.md) for the full API contract.

## Architecture

- **Text-based PDFs** (Biotronik, ELA/Sorin, BSC, Medtronic CareLink): `pdfplumber` text extraction + regex parsing. CPU-only, fast (<1s).
- **Image-based PDFs** (Abbott): visual question answering with selectable backend. Requires NVIDIA GPU. The VQA model is lazy-loaded on the first Abbott request, so the service starts immediately for non-Abbott vendors.
- **Binary/text log files**: direct parsing (no OCR needed). CPU-only.

### Abbott VQA Models

| Model | VRAM | Speed | Accuracy | Backend |
|---|---|---|---|---|
| Qwen3-VL-8B (default) | ~6 GB (4-bit) | ~42s/page | 94.2% | transformers |
| Qwen3.8-27B | ~18 GB (Q4_K_XL) | ~29s/page | 97.5% | Ollama |

Set `CIED_EXTRACT_MODEL=qwen3.8-27b` to use the larger model. This requires an Ollama instance with the model pulled (`ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL`). Configure the Ollama URL with `CIED_EXTRACT_OLLAMA_URL` (default: `http://host.docker.internal:11434`).

## Project Structure

```
server.py                  # FastAPI HTTP service
vqa.py                     # Qwen3-VL model loader + VQA inference
validate.py                # Ground-truth validation harness
extractors/
  abbott_merlin.py          # Abbott VQA extraction (all report types)
  abbott_merlin_ocr.py      # Abbott PaddleOCR fallback (FastPath only, CPU)
  abbott_log.py             # Abbott .log file parser
  biotronik.py              # Biotronik PDF extractor
  biotronik_xml.py          # Biotronik IEEE 11073 XML
  biotronik_biostd.py       # Biotronik BioStd XML
  bsc_crtd.py               # Boston Scientific CRT-D PDF
  bsc_sicd.py               # Boston Scientific S-ICD (PDF + HL7)
  bsc_bnk.py                # Boston Scientific PACEART .bnk
  ela_sorin.py              # ELA/Sorin/LivaNova PDF
  medtronic_2090.py         # Medtronic 2090 programmer (.26X/.V26)
  medtronic_carelink.py     # Medtronic CareLink PDF
  medtronic_pdd.py          # Medtronic .pdd binary
  medtronic_pkg.py          # Medtronic .pkg ZIP archive
  microport.py              # Microport/PACEART XML
annotator/                 # Ground-truth annotation web tool
```

## GPU Requirements

Abbott image-based PDFs require an NVIDIA GPU. The default Qwen3-VL-8B model runs in 4-bit quantization and needs ~6 GB VRAM (fits an RTX 3080 Ti). The optional Qwen3.8-27B model needs ~18 GB VRAM (requires an RTX 4090 or better). All other vendors are CPU-only.

Pin to a specific GPU with `device_ids` in `docker-compose.api.yml`.

## License

MIT — see [LICENSE](LICENSE).
