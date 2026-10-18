"""
Medtronic legacy .pdd binary file extractor.

Handles the proprietary binary format exported by Medtronic CareLink/Encore
programmers. Data is extracted from fixed-offset length-prefixed strings and
FF-prefixed value/type marker scanning.

Covers device families: Ensura, Astra, Azure, Protecta, Visia, Evera, Amplia,
Consulta, Reveal LINQ, Micra, and other Medtronic devices using .pdd export.

Usage:
    from extractors.medtronic_pdd import extract
    result = extract("session.pdd")
"""
import re
from pathlib import Path
from typing import Optional


_SERIAL_RE = re.compile(rb'([A-Z]{3}\d{6}[A-Z])(\d{14})?')

_KNOWN_FAMILIES = [
    b'Protecta', b'Visia', b'Evera', b'Viva', b'Brava', b'Claria',
    b'Amplia', b'Consulta', b'Secura', b'Maximo', b'Concerto',
    b'Virtuoso', b'Reveal', b'LINQ',
]

_FAMILY_TYPE: list[tuple[str, str]] = [
    ('REVEAL', 'ICM'), ('LINQ', 'ICM'),
    ('MICRA', 'Leadless Pacemaker'),
    ('AMPLIA', 'CRT-D'),
    ('PROTECTA', 'ICD'), ('VISIA', 'ICD'), ('EVERA', 'ICD'),
]


def _infer_device_type(model: str) -> str:
    upper = model.upper()
    for family, dtype in _FAMILY_TYPE:
        if family in upper:
            return dtype
    if 'CRT-D' in upper:
        return 'CRT-D'
    if 'CRT-P' in upper:
        return 'CRT-P'
    if 'CRT' in upper:
        return 'CRT'
    if 'ICD' in upper:
        return 'ICD'
    return 'Pacemaker'


def _read_len_prefixed_ascii(buf: bytes, offset: int) -> str:
    if offset >= len(buf):
        return ''
    length = buf[offset]
    if not length or offset + 1 + length > len(buf):
        return ''
    try:
        return buf[offset + 1:offset + 1 + length].decode('latin-1')
    except Exception:
        return ''


def _read_len_prefixed_utf16be(buf: bytes, offset: int) -> str:
    if offset >= len(buf):
        return ''
    length = buf[offset]
    if not length or length % 2 != 0 or offset + 1 + length > len(buf):
        return ''
    out = []
    for i in range(offset + 1, offset + 1 + length, 2):
        hi = buf[i]
        lo = buf[i + 1]
        if hi != 0x00 or lo < 0x20 or lo > 0x7E:
            return ''
        out.append(chr(lo))
    return ''.join(out)


def _parse_person_name(raw: str) -> Optional[str]:
    trimmed = ' '.join(raw.split()).strip()
    if not trimmed:
        return None
    if ',' in trimmed:
        parts = trimmed.split(',', 1)
        return f"{parts[0].strip()}, {parts[1].strip()}"
    parts = trimmed.split()
    if len(parts) >= 2:
        return f"{parts[0]}, {' '.join(parts[1:])}"
    return trimmed


def _parse_pdd_structure(buf: bytes) -> list[dict]:
    """Scan for FF + digits + LF + FF + digits + LF patterns (value/type pairs)."""
    entries = []
    i = 0
    while i < len(buf):
        if buf[i] == 0xFF:
            j = i + 1
            val_str = bytearray()
            while j < len(buf) and 0x30 <= buf[j] <= 0x39:
                val_str.append(buf[j])
                j += 1
            if val_str and j < len(buf) and buf[j] == 0x0A:
                if j + 1 < len(buf) and buf[j + 1] == 0xFF:
                    k = j + 2
                    type_str = bytearray()
                    while k < len(buf) and 0x30 <= buf[k] <= 0x39:
                        type_str.append(buf[k])
                        k += 1
                    if type_str and k < len(buf) and buf[k] == 0x0A:
                        entries.append({
                            'offset': i,
                            'value': int(val_str),
                            'type': int(type_str),
                        })
                        i = k
                        continue
        i += 1
    return entries


def _parse_raw_values(buf: bytes) -> list[dict]:
    """Scan for FF (or FF FF) + digits + LF patterns."""
    values = []
    i = 0
    while i < len(buf):
        if buf[i] == 0xFF:
            is_double_ff = False
            if i + 1 < len(buf) and buf[i + 1] == 0xFF:
                is_double_ff = True
                i += 1
            j = i + 1
            val_str = bytearray()
            while j < len(buf) and 0x30 <= buf[j] <= 0x39:
                val_str.append(buf[j])
                j += 1
            if val_str and j < len(buf) and buf[j] == 0x0A:
                values.append({
                    'offset': i,
                    'value': int(val_str),
                    'is_double_ff': is_double_ff,
                })
                i = j
                continue
        i += 1
    return values


def _extract_printable_strings(buf: bytes, min_len: int = 4) -> list[str]:
    """Extract all printable ASCII strings of at least min_len characters."""
    strings = []
    run_start = -1
    i = 0
    while i < len(buf):
        b = buf[i]
        seq_len = 0
        if 32 <= b <= 126:
            seq_len = 1
        elif 0xC2 <= b <= 0xDF:
            seq_len = 2
        elif 0xE0 <= b <= 0xEF:
            seq_len = 3
        elif 0xF0 <= b <= 0xF4:
            seq_len = 4
        if seq_len > 1:
            for k in range(1, seq_len):
                if i + k >= len(buf) or not (0x80 <= buf[i + k] <= 0xBF):
                    seq_len = 0
                    break
        if seq_len > 0:
            if run_start == -1:
                run_start = i
            i += seq_len
        else:
            if run_start != -1:
                s = buf[run_start:i].decode('utf-8', errors='replace').strip()
                if len(s) >= min_len:
                    strings.append(s)
                run_start = -1
            i += 1
    if run_start != -1:
        s = buf[run_start:len(buf)].decode('utf-8', errors='replace').strip()
        if len(s) >= min_len:
            strings.append(s)
    return strings


def extract(file_path: str | Path) -> dict:
    """
    Parse a Medtronic .pdd binary file and return a structured dict.

    Returned keys:
        manufacturer, report_type, source_file
        patient_name, device_model, device_serial, device_type
        interrogation_date, battery_voltage_v
        impedance_a_ohm, impedance_rv_ohm
        threshold_a_v, threshold_rv_v
    """
    file_path = Path(file_path)
    buf = file_path.read_bytes()

    result: dict = {
        'source_file': str(file_path),
        'manufacturer': 'Medtronic',
        'report_type': 'PDD',
    }

    # -- Patient name --
    name = None
    # Try fixed-offset ASCII (offset 0x03)
    raw = _read_len_prefixed_ascii(buf, 0x03)
    name = _parse_person_name(raw)
    # Fallback: UTF-16BE at 0x77
    if not name:
        raw = _read_len_prefixed_utf16be(buf, 0x77)
        name = _parse_person_name(raw)
    # Fallback: scan printable strings for "Last, First" pattern
    if not name:
        strings = _extract_printable_strings(buf)
        name_re = re.compile(r'^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\'-]+),\s+([A-Za-zÀ-ÿ].*)')
        for s in strings:
            if name_re.match(s):
                name = s
                break
    if name:
        result['patient_name'] = name

    # -- Device model --
    model = _read_len_prefixed_ascii(buf, 0x22).strip()
    if not model:
        strings = _extract_printable_strings(buf) if 'strings' not in dir() else strings
        for s in (strings if isinstance(strings, list) else _extract_printable_strings(buf)):
            for fam in _KNOWN_FAMILIES:
                if fam.decode().upper() in s.upper():
                    model = s
                    break
            if model:
                break
    if model:
        result['device_model'] = model
        result['device_type'] = _infer_device_type(model)

    # -- Serial number & interrogation date --
    strings = _extract_printable_strings(buf)
    for s in strings:
        m = _SERIAL_RE.search(s.encode('ascii', errors='ignore'))
        if m:
            result['device_serial'] = m.group(1).decode()
            if m.group(2):
                ts = m.group(2).decode()
                result['interrogation_date'] = f"{ts[6:8]}.{ts[4:6]}.{ts[0:4]}"
            break

    # Fallback date: look for standalone 14-digit timestamp
    if 'interrogation_date' not in result:
        date_re = re.compile(r'^(20\d{12})$')
        for s in strings:
            dm = date_re.match(s)
            if dm:
                ts = dm.group(1)
                result['interrogation_date'] = f"{ts[6:8]}.{ts[4:6]}.{ts[0:4]}"
                break

    # -- Binary structure measurements --
    entries = _parse_pdd_structure(buf)

    # Battery voltage: type-4, value 2000-3500 → /1000 for volts
    # Only assert when all candidates agree
    candidates = [e for e in entries if e['type'] == 4 and 2000 <= e['value'] <= 3500]
    distinct = set(c['value'] for c in candidates)
    if len(distinct) == 1:
        result['battery_voltage_v'] = str(candidates[0]['value'] / 1000)

    # Atrial impedance: type-3, value 737000-738000, last 3 digits = ohms
    type3 = [e for e in entries if e['type'] == 3 and 737000 <= e['value'] < 738000]
    for e in type3:
        imp = e['value'] % 1000
        if 200 < imp < 2000:
            result['impedance_a_ohm'] = str(imp)
            break

    # Thresholds: type-2, value 737400-737600, last 2 digits / 100 = volts
    type2 = [e for e in entries if e['type'] == 2 and 737400 <= e['value'] < 737600]
    thresh_values = []
    for e in type2:
        v = (e['value'] % 100) / 100
        if 0 < v < 5:
            thresh_values.append(v)
    if len(thresh_values) >= 1:
        result['threshold_a_v'] = str(thresh_values[0])
    if len(thresh_values) >= 2:
        result['threshold_rv_v'] = str(thresh_values[1])

    # RV impedance: FF FF prefix, value 589200-589900, last 3 digits = ohms
    raw_values = _parse_raw_values(buf)
    for rv in raw_values:
        if rv['is_double_ff'] and 589200 <= rv['value'] < 589900:
            imp = rv['value'] % 1000
            result['impedance_rv_ohm'] = str(imp)
            break

    return {k: v for k, v in result.items() if v is not None}
