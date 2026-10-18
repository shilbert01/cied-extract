"""Extract all Abbott samples missing ground_truth.json and save raw results."""
import json, sys, traceback
from pathlib import Path

sys.path.insert(0, '/home/herzzentrum/CiedOCR')

# Auto-select GPU before torch loads
import subprocess, os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
            text=True, timeout=5, capture_output=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2: continue
            idx = parts[0].strip()
            if not idx.isdigit(): continue
            try:
                free_mb = int(parts[1].strip().replace(" MiB", ""))
            except ValueError:
                continue
            if free_mb > 6000:
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
                print(f"Using GPU {idx}", flush=True)
                break
    except Exception:
        pass

from validate import _load_vqa, run_sample
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

samples = sorted(
    p.parent for p in Path('samples/abbott').rglob('meta.json')
    if not (p.parent / 'ground_truth.json').exists()
)
print(f"Extracting {len(samples)} samples\n", flush=True)

for sample_dir in samples:
    meta = json.loads((sample_dir / 'meta.json').read_text())
    print(f"{'─'*60}", flush=True)
    print(f"  {sample_dir}", flush=True)
    try:
        result = run_sample(sample_dir, device, meta)
        out_path = sample_dir / '_extraction.json'
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"  → saved {out_path}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()

print("\nDone.", flush=True)
