# CIED-Extract API — v1 (contract)

Status: **DRAFT** · Owner: CIED-Extract service · Consumers: pacemaker central Pi (now), SaaS tenants (future)

The contract both sides build against. CIED-Extract is a **stateless HTTP extraction
service**: PDF in → structured `PacemakerReport` JSON out, retains nothing. The current
CLI (`main.py`) stays as a dev/batch tool; this API wraps the same `ocr_engine` +
`extractor` code with the model **loaded once at startup**, not per request.

## Design principles
- **Stateless.** No input/output is persisted server-side. The system-of-record stays with the caller (the clinic's central Pi). This is also the privacy story.
- **Versioned path + versioned payload.** Path is `/v1/...`; every response carries `schema_version`. Breaking changes → `/v2`.
- **LAN-first, auth-ready.** No auth required now (deployed on trusted LAN). `Authorization: Bearer <token>` is *accepted and ignored* in v1 so clients can start sending it; when non-local/SaaS lands, the token identifies the tenant and becomes mandatory — no client change needed.
- **Self-describing results.** Every response stamps the engine + model + version so the caller can store an audit trail and decide when to reprocess.

## Endpoints

### `GET /health`
Liveness + readiness. Cheap; used by the central Pi to gate "full vs raw-only" mode.
```json
{
  "status": "ok",              // "ok" | "loading" | "degraded"
  "model_loaded": true,
  "engine_version": "0.3.0",
  "model": "PaddlePaddle/PaddleOCR-VL-1.6",
  "device": "cuda",            // "cuda" | "cpu"
  "schema_version": "1.0"
}
```
`200` when ready; `503` with `{"status":"loading"}` while the model warms up.

### `POST /v1/extract`
Extract structured data from one report PDF.

**Request**
- `Content-Type: application/pdf`, body = raw PDF bytes. (Print-route jobs are converted PS→PDF by the caller before sending; export-route jobs are already PDF.)
- Optional headers:
  - `X-Request-Id: <uuid>` — echoed back for tracing/audit; server generates one if absent.
  - `X-Vendor-Hint: medtronic|boston_scientific|abbott|biotronik|sorin|...` — optional hint; extraction is vendor-agnostic, ignored if unknown.
  - `Authorization: Bearer <token>` — ignored in v1, reserved for tenant auth.
- Optional query: `?dpi=200` (PDF→image DPI, default 200), `?raw=true` (include per-page OCR text in response — debugging only).

**Response `200`** — envelope wrapping the `PacemakerReport`:
```json
{
  "schema_version": "1.0",
  "request_id": "b1f2...",
  "status": "ok",                     // see status values below
  "engine": {
    "name": "cied-extract",
    "version": "0.3.0",
    "model": "PaddlePaddle/PaddleOCR-VL-1.6",
    "device": "cuda"
  },
  "source": { "pages": 2, "sha256": "…", "bytes": 62696 },
  "processing_ms": 4210,
  "report": { /* PacemakerReport, see below (raw_ocr_pages omitted unless ?raw) */ },
  "warnings": []
}
```

**`report` object** (mirrors `models.py::PacemakerReport`; all fields nullable):
```
report_type, patient, device, facility, interrogation_date, visit_type
battery_ra, battery_rv        : { implant_date, voltage, current_ua,
                                  remaining_capacity_pct, estimated_longevity, rrt }
test_ra, test_rv              : { pacing_threshold(+_date), sensing_mv(+_date),
                                  impedance_ohm(+_date) }
parameters                    : { mode, base_rate, av_paced_ms, av_sensed_ms,
                                  upper_rate_limit, max_sensor_rate, rate_hysteresis,
                                  vip_enabled, rate_adaptive_av, rate_adaptive_pvarp,
                                  ventricular_safety_pacing }
stimulation_ra, stimulation_rv: { configuration, amplitude_v, pulse_width_ms,
                                  senseability, sensitivity_mv }
diagnostics                   : { since, ap_paced_pct, vp_paced_pct,
                                  mode_switch_pct, ams_episodes }
```

**`status` values** (in the 200 body — the request succeeded, this is the *extraction* outcome):
- `ok` — text extracted and fields populated.
- `partial` — some sections empty/low-confidence (still returned).
- `no_text` — OCR produced nothing usable (blank/corrupt/non-report PDF). `report` may be mostly null.

**Error responses** (consistent body `{"error": <code>, "detail": "...", "request_id": "..."}`):
| HTTP | error | when |
|------|-------|------|
| 400 | `empty_body` / `bad_pdf` | no body / unparseable PDF |
| 415 | `unsupported_media` | not `application/pdf` |
| 422 | `unprocessable` | valid PDF but extraction blew up |
| 401/403 | `unauthorized` | **v2 only** (auth off in v1) |
| 503 | `model_loading` / `unavailable` | model warming or service overloaded → caller retries |
| 500 | `internal` | unexpected |

Callers distinguish **transient** (`503`, timeout, connection refused → keep job re-queueable) from **terminal** (`400/415/422`, or `200` with `no_text` → mark failed, needs human/raw view).

### Async submission (recommended for real reports)
VL-OCR runs ~1 min/page, so a synchronous connection is held open a long time and
multi-page reports risk client/proxy timeouts. Opt into async with **`?mode=async`**
(or header `Prefer: respond-async`). Same request body + validation as sync.

**`202 Accepted`** — job queued:
```json
{ "job_id": "…", "status": "queued", "poll": "/v1/jobs/<job_id>", "schema_version": "1.0" }
```
Jobs run **one at a time, FIFO** (shared with sync inference). Job state — including the
result — lives **only in memory** with a **1 h TTL**; nothing is written to disk (the
stateless/privacy property holds). `X-Request-Id`, if sent, is used as the `job_id`.

### `GET /v1/jobs/{job_id}`
Poll an async job.
- `queued` / `running` → `200 {"job_id","status","schema_version"}`
- `done` → `200` with the **same envelope as a sync `/v1/extract` success** (report + engine + source + processing_ms).
- `error` → `200 {"job_id","status":"error","error","detail",…}` (extraction blew up; distinct from a transient submit-time error).
- unknown/expired id → `404 {"error":"unknown_job",…}`.

Poll interval suggestion: 5 s. Recommended client flow: `POST ?mode=async` → poll until `status ∈ {done,error}` → on `done` map the report; on `error`/`404`(expired) re-queue or fall back to raw view.

## Versioning & compatibility
- Additive fields (new nullable keys) do **not** bump `schema_version` minor→major; consumers must ignore unknown keys.
- Renames/removals/semantic changes → new `/v2/extract` + `schema_version: "2.0"`; `/v1` kept until all callers move.
- `engine.version` changes freely (model/parser improvements). The central Pi stores it per interrogation so historical reports can be **reprocessed** when the engine improves.

## Deferred (design for, don't build yet)
- **Auth / multi-tenancy:** `Authorization: Bearer` → tenant identity, rate limits, per-tenant quotas.
- **TLS + DPA/AVV** for non-local deployment (PHI leaves the building — see central `ALLOW_REMOTE_PHI` policy flag).
- **Push callbacks** (webhook on completion) as an alternative to polling async jobs, if the central Pi would rather not poll.
- **Durable job queue** (survives a service restart) — current async state is in-memory only, so a restart drops in-flight jobs; the central Pi re-queues via its own state, so this is acceptable for now.
