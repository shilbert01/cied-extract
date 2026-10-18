#!/usr/bin/env python3
"""
CIEDOcr Ground-Truth Annotator — a small internal tool to build/curate ground truth.

Flow: upload a report PDF + pick the vendor → the CIEDOcr extraction API runs the
right per-vendor extractor → the PDF is shown next to every extracted field →
the user cross-checks, edits, adds/removes fields → "Submit" writes a ground-truth
bundle (report.pdf + ground_truth.json + meta.json) laid out like samples/, ready
to fold into the validate.py corpus.

Run:
    pip install -r requirements.txt
    CIEDOCR_URL=http://localhost:8082 uvicorn app:app --host 0.0.0.0 --port 8090
"""
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

CIEDOCR_URL = os.environ.get("CIEDOCR_URL", "http://localhost:8082").rstrip("/")
EXTRACT_TIMEOUT = int(os.environ.get("ANNOTATOR_EXTRACT_TIMEOUT", "600"))

BASE = Path(__file__).parent
WORK = Path(os.environ.get("ANNOTATOR_WORK", BASE / "work"))
SUBMISSIONS = Path(os.environ.get("ANNOTATOR_SUBMISSIONS", BASE / "submissions"))
WORK.mkdir(parents=True, exist_ok=True)
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

# vendor key (X-Vendor-Hint) → manufacturer string used by validate.py / meta.json
MANUFACTURERS = {
    "biotronik":         "Biotronik",
    "ela_sorin":         "ELA Medical / Sorin",
    "boston_scientific": "Boston Scientific",
    "abbott":            "Abbott",
}

app = FastAPI(title="CIEDOcr Ground-Truth Annotator")


def _slug(s: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (s or "").strip())
    return s.strip("_") or "x"


def _extract_via_api(pdf_bytes: bytes, vendor: str) -> dict:
    """Submit to the CIEDOcr API (async) and poll to completion."""
    req = urllib.request.Request(
        f"{CIEDOCR_URL}/v1/extract?mode=async",
        data=pdf_bytes,
        headers={"Content-Type": "application/pdf", "X-Vendor-Hint": vendor},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        submit = json.loads(r.read())
    poll_url = CIEDOCR_URL + submit.get("poll", "/v1/jobs/" + str(submit.get("job_id")))

    deadline = time.monotonic() + EXTRACT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(3)
        with urllib.request.urlopen(poll_url, timeout=15) as r:
            res = json.loads(r.read())
        if res.get("status") in ("queued", "running"):
            continue
        return res
    raise TimeoutError("extraction timed out")


@app.get("/api/vendors")
def vendors():
    return {"vendors": [{"key": k, "label": v} for k, v in MANUFACTURERS.items()]}


@app.post("/api/extract")
async def extract(pdf: UploadFile = File(...), vendor: str = Form(...)):
    data = await pdf.read()
    if not data:
        raise HTTPException(400, "empty PDF")
    if vendor not in MANUFACTURERS:
        raise HTTPException(422, f"unknown vendor '{vendor}'")

    doc_id = uuid.uuid4().hex
    (WORK / f"{doc_id}.pdf").write_bytes(data)
    (WORK / f"{doc_id}.name").write_text(pdf.filename or "report.pdf")

    try:
        res = _extract_via_api(data, vendor)
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"extraction service error: HTTP {e.code}")
    except Exception as e:
        raise HTTPException(502, f"extraction failed: {e}")

    report = res.get("report") if isinstance(res.get("report"), dict) else {}
    report.pop("source_file", None)
    return {
        "doc_id": doc_id,
        "vendor": vendor,
        "status": res.get("status"),
        "report_type": report.get("report_type"),
        "engine": res.get("engine"),
        "fields": report,
    }


@app.get("/api/pdf/{doc_id}")
def pdf(doc_id: str):
    p = WORK / f"{_slug(doc_id)}.pdf"
    if not p.exists():
        raise HTTPException(404, "pdf not found")
    return FileResponse(p, media_type="application/pdf")


@app.post("/api/save")
async def save(request: Request):
    body = await request.json()
    doc_id = _slug(body.get("doc_id", ""))
    vendor = body.get("vendor", "")
    device = body.get("device") or "unknown"
    report_type = body.get("report_type") or "unknown"
    device_type = (body.get("device_type") or "").strip()
    fields = body.get("fields")
    if not isinstance(fields, dict):
        raise HTTPException(400, "fields must be an object")

    src = WORK / f"{doc_id}.pdf"
    if not src.exists():
        raise HTTPException(404, "source PDF expired — re-extract before saving")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = SUBMISSIONS / _slug(vendor) / _slug(device) / _slug(report_type) / ts
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "report.pdf").write_bytes(src.read_bytes())
    (dest / "ground_truth.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2))
    orig_name = (WORK / f"{doc_id}.name")
    (dest / "meta.json").write_text(json.dumps({
        "manufacturer": MANUFACTURERS.get(vendor, vendor),
        "device_type":  device_type,
        "report_type":  report_type,
        "device_model": device,
        "annotated_at": ts,
        "source_pdf":   orig_name.read_text() if orig_name.exists() else "report.pdf",
    }, ensure_ascii=False, indent=2))

    return {"saved": str(dest.relative_to(SUBMISSIONS)), "fields": len(fields)}


# static UI (mounted last so /api/* takes priority)
app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")
