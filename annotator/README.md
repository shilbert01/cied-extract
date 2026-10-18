# CIEDOcr Ground-Truth Annotator

A small internal web tool to build/curate ground truth for the CIEDOcr parsers.

**Flow:** upload a report PDF → pick the vendor → the CIEDOcr extraction API runs the
right per-vendor extractor → the PDF is shown next to every extracted field → you
cross-check, edit values, add missing fields, drop wrong ones → **Submit** writes a
ground-truth bundle laid out like `samples/`.

It reuses the deployed extraction service (`../server.py`), so annotations reflect
exactly what the product extracts.

## Run

```bash
pip install -r requirements.txt
# point at wherever the CIEDOcr API runs (default: http://localhost:8082)
CIEDOCR_URL=http://localhost:8082 uvicorn app:app --host 0.0.0.0 --port 8090
# open http://localhost:8090
```

On bigboy (API at :8082 on the same host) the default works. From another machine
set `CIEDOCR_URL=http://bigboy.fritz.box:8082`.

## Output

Each submission lands in `submissions/<vendor>/<device>/<report_type>/<timestamp>/`:

```
report.pdf          # the original PDF
ground_truth.json   # the human-verified fields
meta.json           # manufacturer / device_type / report_type (for validate.py dispatch)
```

This mirrors the `samples/**/ground_truth.json` layout, so a curated submission can be
copied straight into the sample corpus and scored with `python ../validate.py`.

## Notes

- Values are typed on save: numbers / `true` / `false` / `null` are stored as JSON
  types; anything else (e.g. `"2,5 V"`) stays a string. Objects/arrays are edited as JSON.
- `work/` holds uploaded PDFs transiently between extract and submit; `submissions/` is
  the durable output. Both are git-ignored.
- Internal use only for now — no auth. Add auth/TLS before exposing it beyond the LAN.
