#!/usr/bin/env python3
"""
CIED-Extract API — FastAPI wrapper implementing docs/extract-api-v1.md.

Dispatches each PDF to the correct per-vendor extractor in `extractors/`:
  * Biotronik / ELA-Sorin / Boston Scientific  → pdfplumber text (no model, CPU)
  * Abbott                                      → Qwen VQA (vqa.py), lazy-loaded
PaddleOCR is retired. The Qwen model is loaded once, on the first Abbott request
(so the service is ready immediately for the pdfplumber vendors).

Stateless on disk; async job state is in memory with a TTL. See the contract doc.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
import hashlib
import io
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

SCHEMA_VERSION = "1.0"
ENGINE_VERSION = "0.3.4"          # device_name/device_class across all Abbott schemas
JOB_TTL_SECONDS = 3600
SUPPORTED_VENDORS = ["biotronik", "ela_sorin", "boston_scientific", "abbott", "medtronic", "microport"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("cied-extract.api")

# pdfplumber vendors are CPU-bound and run concurrently across workers; only the
# Abbott/Qwen GPU path is serialized (via _gpu_lock) so it can't be run twice at once
# and a slow/stuck Abbott job never blocks the fast pdfplumber vendors.
N_WORKERS = int(os.environ.get("CIED_EXTRACT_WORKERS", "3"))
_gpu_lock = threading.Lock()

# ── lazy Qwen VQA model (Abbott only) ─────────────────────────────────────────
_vqa = None                       # (model, processor)
_vqa_state = "unloaded"           # unloaded | loading | ready | error
_vqa_error: str | None = None
_vqa_lock = threading.Lock()


class UnsupportedVendor(Exception):
    pass


def _get_vqa():
    global _vqa, _vqa_state, _vqa_error
    with _vqa_lock:
        if _vqa is None:
            _vqa_state = "loading"
            try:
                # torch 2.4 (base image) lacks nn.Module.set_submodule, which
                # transformers 5.x's bitsandbytes 4-bit integration calls. Shim it
                # so we can keep the memory-friendly 4-bit load without a torch bump.
                import torch.nn as _nn
                if not hasattr(_nn.Module, "set_submodule"):
                    def _set_submodule(self, target, module):
                        *parents, name = target.split(".")
                        mod = self
                        for p in parents:
                            mod = getattr(mod, p)
                        setattr(mod, name, module)
                    _nn.Module.set_submodule = _set_submodule
                from vqa import load_model
                device = os.environ.get("FORCE_DEVICE") or "cuda"
                log.info("Loading Qwen VQA model (%s) for Abbott extraction…", device)
                _vqa = load_model(device)
                _vqa_state = "ready"
                log.info("Qwen VQA model ready")
            except Exception as e:
                _vqa_error = str(e)
                _vqa_state = "error"
                raise
        return _vqa


def _log_ext(log_path: str | None) -> str:
    """Return lowercased extension of the log/sidecar file, or ''."""
    if not log_path:
        return ""
    return os.path.splitext(log_path)[1].lower()


def _dispatch(pdf_path: str, vendor_hint: str, log_path: str | None = None,
              hl7_path: str | None = None) -> dict:
    """Route to the correct per-vendor extractor (mirrors validate.py::run_sample)."""
    v = (vendor_hint or "").lower()
    ext = _log_ext(log_path)

    # ── Microport / Paceart XML ──────────────────────────────────────────
    if "paceart" in v or (v == "microport" and ext == ".xml"):
        from extractors.microport import extract
        return extract(log_path or pdf_path)

    # ── Biotronik ────────────────────────────────────────────────────────
    if "biotronik" in v:
        src = log_path if log_path else pdf_path
        if src.lower().endswith(".xml"):
            with open(src, "r", encoding="utf-8") as f:
                head = f.read(500)
            if "biotronik-ieee11073" in head or "BIOIEEE" in os.path.basename(src):
                from extractors.biotronik_xml import extract as bio_xml_extract
                return bio_xml_extract(src)
            else:
                from extractors.biotronik_biostd import extract as bio_biostd_extract
                return bio_biostd_extract(src)
        from extractors.biotronik import extract
        return extract(pdf_path)

    # ── ELA / Sorin / LivaNova / MicroPort CRM (PDF) ────────────────────
    if "sorin" in v or v.startswith("ela") or "livanova" in v or "microport" in v:
        from extractors.ela_sorin import extract
        return extract(pdf_path)

    # ── Boston Scientific ────────────────────────────────────────────────
    if "boston" in v or v == "bsc":
        if ext == ".bnk":
            from extractors.bsc_bnk import extract as bnk_extract
            return bnk_extract(log_path)
        if hl7_path is not None:
            from extractors.bsc_sicd import extract as sicd_extract
            return sicd_extract(pdf_path, hl7_path)
        from extractors.bsc_crtd import extract
        return extract(pdf_path)

    # ── Abbott / St. Jude ────────────────────────────────────────────────
    if "abbott" in v or "sjm" in v or "jude" in v:
        if log_path is not None:
            from extractors.abbott_log import extract as log_extract
            return log_extract(log_path)
        from extractors.abbott_merlin import extract
        model, processor = _get_vqa()
        return extract(pdf_path, model, processor)

    # ── Medtronic / Vitatron ─────────────────────────────────────────────
    if "medtronic" in v or "vitatron" in v:
        if ext == ".pkg":
            from extractors.medtronic_pkg import extract as pkg_extract
            return pkg_extract(log_path)
        if ext == ".pdd":
            from extractors.medtronic_pdd import extract as pdd_extract
            return pdd_extract(log_path)
        if ext == ".xml":
            from extractors.medtronic_pkg import extract_xml
            with open(log_path, "r", encoding="utf-8") as f:
                return extract_xml(f.read(), log_path)
        # CareLink PDF: detect by content markers
        src = log_path if log_path else pdf_path
        if src.lower().endswith(".pdf"):
            from extractors.medtronic_carelink import extract as cl_extract
            return cl_extract(src)
        # .26X/.V26 text files or fallback
        from extractors.medtronic_2090 import extract
        return extract(src)

    raise UnsupportedVendor(vendor_hint or "(none)")


# ── async job registry (in-memory only) ───────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[str]" = queue.Queue()


def _sweep_jobs():
    now = time.time()
    with _jobs_lock:
        for jid in [k for k, v in _jobs.items() if now - v["created_at"] > JOB_TTL_SECONDS]:
            _jobs.pop(jid, None)


def _worker():
    while True:
        job_id = _job_queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                continue
            body, vendor_hint, log_bytes, hl7_bytes, log_fname = job.pop("_payload")
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            envelope = _do_extract(body, vendor_hint, job_id, log_bytes, hl7_bytes, log_fname)
            with _jobs_lock:
                j = _jobs.get(job_id)
                if j:
                    j.update(status="done", result=envelope, updated_at=time.time())
        except UnsupportedVendor as e:
            with _jobs_lock:
                j = _jobs.get(job_id)
                if j:
                    j.update(status="error",
                             error={"error": "unsupported_vendor", "detail": str(e)},
                             updated_at=time.time())
        except Exception as e:
            log.exception("async extract failed [%s]", job_id)
            with _jobs_lock:
                j = _jobs.get(job_id)
                if j:
                    j.update(status="error",
                             error={"error": "unprocessable", "detail": str(e)},
                             updated_at=time.time())
        _sweep_jobs()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for i in range(N_WORKERS):
        threading.Thread(target=_worker, name=f"async-worker-{i}", daemon=True).start()
    log.info("Started %d async workers", N_WORKERS)
    yield


app = FastAPI(title="CIED-Extract API", version=SCHEMA_VERSION, lifespan=lifespan)


def _err(code: int, error: str, detail: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=code,
                        content={"error": error, "detail": detail, "request_id": request_id})


@app.get("/health")
def health():
    # Service is ready as soon as it is up — the pdfplumber vendors need no model.
    body = {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "supported_vendors": SUPPORTED_VENDORS,
        "vqa_state": _vqa_state,           # Abbott (Qwen) readiness
    }
    if _vqa_state == "error":
        body["vqa_error"] = _vqa_error
    return JSONResponse(status_code=200, content=body)


def _page_count(pdf_bytes: bytes) -> int | None:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return len(pdf.pages)
    except Exception:
        return None


def _do_extract(body: bytes, vendor_hint: str, request_id: str,
                log_bytes: bytes | None = None,
                hl7_bytes: bytes | None = None,
                log_filename: str = "") -> dict:
    t0 = time.monotonic()
    sha = hashlib.sha256(body or log_bytes or b"").hexdigest()
    is_abbott = "abbott" in (vendor_hint or "").lower()
    uses_log = is_abbott and log_bytes is not None
    log_ext = os.path.splitext(log_filename)[1].lower() if log_filename else ".log"
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
        if body:
            tf.write(body)
            tf.flush()
        log_tf = None
        hl7_tf = None
        if log_bytes is not None:
            log_tf = tempfile.NamedTemporaryFile(suffix=log_ext or ".log", delete=False)
            log_tf.write(log_bytes)
            log_tf.flush()
            log_tf.close()
        if hl7_bytes is not None:
            hl7_tf = tempfile.NamedTemporaryFile(suffix=".hl7", delete=False)
            hl7_tf.write(hl7_bytes)
            hl7_tf.flush()
            hl7_tf.close()
        try:
            if is_abbott and not uses_log:
                with _gpu_lock:            # serialize the single GPU/Qwen model
                    report = _dispatch(tf.name, vendor_hint)
            else:
                report = _dispatch(tf.name, vendor_hint,
                                   log_tf.name if log_tf else None,
                                   hl7_tf.name if hl7_tf else None)
        finally:
            if log_tf:
                import os as _os
                _os.unlink(log_tf.name)
            if hl7_tf:
                import os as _os
                _os.unlink(hl7_tf.name)

    if isinstance(report, dict):
        report.pop("source_file", None)    # scrub temp path (privacy)

    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "ok",
        "engine": {
            "name": "cied-extract",
            "version": ENGINE_VERSION,
            "extractor": (report.get("manufacturer") if isinstance(report, dict) else None) or vendor_hint,
            "model": ("log" if uses_log else "qwen-vqa") if is_abbott
                     else ("native" if (log_ext not in ("", ".log") and not body) else "pdfplumber"),
        },
        "source": {"pages": _page_count(body) if body else None, "sha256": sha,
                   "bytes": len(body or log_bytes or b"")},
        "processing_ms": int((time.monotonic() - t0) * 1000),
        "report": report,
        "warnings": [],
    }


def _check_pdf(body: bytes, request_id: str) -> JSONResponse | None:
    if not body:
        return _err(400, "empty_body", "PDF part was empty", request_id)
    if body[:5] != b"%PDF-":
        return _err(400, "bad_pdf", "PDF does not start with %PDF- magic", request_id)
    return None


@app.post("/v1/extract")
async def extract_endpoint(
    request: Request,
    x_request_id: str | None = Header(default=None),
    x_vendor_hint: str | None = Header(default=None),   # selects the extractor
    authorization: str | None = Header(default=None),   # accepted-but-ignored in v1
    prefer: str | None = Header(default=None),
    mode: str = Query(default="sync"),
    dpi: int = Query(default=200, ge=72, le=600),        # advisory (Abbott VQA only)
    raw: bool = Query(default=False),                    # accepted; native extractors omit raw pages
):
    request_id = x_request_id or uuid.uuid4().hex
    ctype = (request.headers.get("content-type") or "").lower()
    log_bytes: bytes | None = None
    log_filename: str = ""
    hl7_bytes: bytes | None = None

    if "multipart/form-data" in ctype:
        # Multipart: 'pdf' (required unless 'log' carries a non-PDF primary file),
        # 'log' (optional sidecar or primary for .bnk/.pdd/.pkg/.xml),
        # 'hl7' (optional, BSC S-ICD)
        form = await request.form()
        pdf_part = form.get("pdf")
        log_part = form.get("log")
        hl7_part = form.get("hl7")
        if pdf_part is None and log_part is None:
            return _err(400, "missing_field", "multipart must include a 'pdf' or 'log' field", request_id)
        body = b""
        if pdf_part is not None:
            body = await pdf_part.read() if hasattr(pdf_part, "read") else pdf_part.encode()
        if log_part is not None:
            log_bytes = await log_part.read() if hasattr(log_part, "read") else log_part.encode()
            log_filename = getattr(log_part, "filename", "") or ""
        if hl7_part is not None:
            hl7_bytes = await hl7_part.read() if hasattr(hl7_part, "read") else hl7_part.encode()
        vendor_hint = x_vendor_hint or str(form.get("vendor") or "")
    elif "application/pdf" in ctype:
        body = await request.body()
        vendor_hint = x_vendor_hint
    else:
        return _err(415, "unsupported_media",
                    "send Content-Type: application/pdf or multipart/form-data", request_id)

    if body:
        err = _check_pdf(body, request_id)
        if err is not None:
            return err

    is_async = mode.lower() == "async" or (prefer or "").lower() == "respond-async"

    if is_async:
        with _jobs_lock:
            _jobs[request_id] = {
                "status": "queued",
                "created_at": time.time(),
                "updated_at": time.time(),
                "result": None,
                "error": None,
                "_payload": (body, vendor_hint, log_bytes, hl7_bytes, log_filename),
            }
        _job_queue.put(request_id)
        return JSONResponse(status_code=202, content={
            "job_id": request_id,
            "status": "queued",
            "poll": f"/v1/jobs/{request_id}",
            "schema_version": SCHEMA_VERSION,
        })

    # synchronous
    try:
        envelope = await run_in_threadpool(
            _do_extract, body, vendor_hint, request_id, log_bytes, hl7_bytes, log_filename
        )
    except UnsupportedVendor as e:
        return _err(422, "unsupported_vendor", str(e), request_id)
    except Exception as e:
        log.exception("extract failed [%s]", request_id)
        return _err(422, "unprocessable", str(e), request_id)
    return JSONResponse(status_code=200, content=envelope)


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    _sweep_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return _err(404, "unknown_job", "unknown or expired job id", job_id)
        status = job["status"]
        if status in ("queued", "running"):
            return {"job_id": job_id, "status": status, "schema_version": SCHEMA_VERSION}
        if status == "done":
            return job["result"]
        return JSONResponse(status_code=200, content={
            "job_id": job_id, "status": "error", "schema_version": SCHEMA_VERSION, **job["error"],
        })
