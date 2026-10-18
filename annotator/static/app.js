"use strict";

const $ = (id) => document.getElementById(id);
let state = { doc_id: null, vendor: null };

// ── field value <-> string helpers ───────────────────────────────────────────
// Objects/arrays are edited as JSON (flagged), scalars as plain text with type
// coercion on save so ground_truth.json keeps clean JSON types.
function valueToEditor(v) {
  if (v !== null && typeof v === "object") return { text: JSON.stringify(v, null, 2), json: true };
  if (v === null) return { text: "", json: false };
  return { text: String(v), json: false };
}
function editorToValue(text, isJson) {
  if (isJson) {
    try { return JSON.parse(text); } catch { return text; }
  }
  const t = text.trim();
  if (t === "") return null;
  if (t === "null") return null;
  if (t === "true") return true;
  if (t === "false") return false;
  if (/^-?\d+(\.\d+)?$/.test(t)) return Number(t);
  return text;
}

// ── field rows ────────────────────────────────────────────────────────────────
function addRow(key = "", value = "", isJson = false) {
  const row = document.createElement("div");
  row.className = "field-row" + (isJson ? " json" : "");

  const k = document.createElement("input");
  k.className = "f-key"; k.value = key; k.placeholder = "field_name";

  let val;
  if (isJson) {
    val = document.createElement("textarea");
    val.rows = Math.min(10, (value.match(/\n/g) || []).length + 1);
  } else {
    val = document.createElement("input");
  }
  val.className = "f-val"; val.value = value;

  const del = document.createElement("button");
  del.className = "f-del"; del.textContent = "✕"; del.title = "remove field";
  del.onclick = () => row.remove();

  row.append(k, val, del);
  row.dataset.json = isJson ? "1" : "0";
  $("fields").appendChild(row);
}

function renderFields(fields) {
  $("fields").innerHTML = "";
  const keys = Object.keys(fields);
  keys.forEach((key) => {
    const { text, json } = valueToEditor(fields[key]);
    addRow(key, text, json);
  });
  $("fieldCount").textContent = keys.length ? `(${keys.length})` : "";
}

function collectFields() {
  const out = {};
  $("fields").querySelectorAll(".field-row").forEach((row) => {
    const key = row.querySelector(".f-key").value.trim();
    if (!key) return;
    const text = row.querySelector(".f-val").value;
    out[key] = editorToValue(text, row.dataset.json === "1");
  });
  return out;
}

// ── status ────────────────────────────────────────────────────────────────────
function setStatus(msg, kind = "") { const s = $("status"); s.textContent = msg; s.className = "status " + kind; }

// ── actions ─────────────────────────────────────────────────────────────────
async function extract() {
  const file = $("pdfFile").files[0];
  const vendor = $("vendor").value;
  if (!file) { setStatus("choose a PDF first", "err"); return; }

  const fd = new FormData();
  fd.append("pdf", file);
  fd.append("vendor", vendor);

  setStatus("extracting… (Abbott runs the VQA model, ~1 min)", "busy");
  $("extractBtn").disabled = true; $("saveBtn").disabled = true;
  try {
    const res = await fetch("/api/extract", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
    const data = await res.json();
    state.doc_id = data.doc_id; state.vendor = data.vendor;

    $("pdfEmpty").hidden = true;
    const frame = $("pdfFrame");
    frame.src = `/api/pdf/${data.doc_id}#view=FitH`;
    frame.hidden = false;

    renderFields(data.fields || {});
    $("metaDevice").value = (data.fields && (data.fields.device_model || data.fields.device)) || "";
    $("metaReportType").value = data.report_type || "";
    const eng = data.engine || {};
    $("engineInfo").textContent = eng.model ? `via ${eng.model}` : "";
    $("saveBtn").disabled = false;
    setStatus(`extracted ${Object.keys(data.fields || {}).length} fields`, "ok");
  } catch (e) {
    setStatus("extract failed: " + e.message, "err");
  } finally {
    $("extractBtn").disabled = false;
  }
}

async function save() {
  if (!state.doc_id) return;
  $("saveMsg").textContent = "";
  const payload = {
    doc_id: state.doc_id,
    vendor: state.vendor,
    device: $("metaDevice").value.trim() || "unknown",
    device_type: $("metaDeviceType").value,
    report_type: $("metaReportType").value.trim() || "unknown",
    fields: collectFields(),
  };
  $("saveBtn").disabled = true;
  try {
    const res = await fetch("/api/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
    const data = await res.json();
    $("saveMsg").className = "save-msg ok";
    $("saveMsg").textContent = `✓ saved ground truth (${data.fields} fields) → ${data.saved}`;
  } catch (e) {
    $("saveMsg").className = "save-msg err";
    $("saveMsg").textContent = "save failed: " + e.message;
  } finally {
    $("saveBtn").disabled = false;
  }
}

async function init() {
  const { vendors } = await (await fetch("/api/vendors")).json();
  $("vendor").innerHTML = vendors.map((v) => `<option value="${v.key}">${v.label}</option>`).join("");
  $("extractBtn").onclick = extract;
  $("saveBtn").onclick = save;
  $("addFieldBtn").onclick = () => addRow();
}
init();
