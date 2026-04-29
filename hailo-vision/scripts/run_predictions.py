"""Run the full pipeline over sample_list.jsonl and emit a predictions JSONL.

Used by Phase 1+ to regenerate predictions after each phase so the benchmark
harness can compare them. Mode is forwarded to ocr(mode=...) and (once later
phases add it) to any feature-level toggle; today it flows into the thumbnail
feature extractor's OCR call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dmmdea/openclaw-mcp-servers/_shared")

os.environ.setdefault("HAILO_VISION_ENABLED", "1")

from openclaw_shared.backends.hailo import maybe_hailo_backend
from openclaw_shared.features.thumbnail import extract_thumbnail_features


def _hailo_only(features: dict) -> dict:
    keep = {}
    for k, v in features.items():
        if not k.startswith("hailo_"):
            continue
        if k == "hailo_clip_embedding":
            continue
        keep[k] = list(v) if isinstance(v, (set, tuple)) else v
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--mode",
        default="production",
        choices=("fast", "production", "research"),
        help=(
            "OCR mode forwarded to HailoRuntime via HAILO_OCR_MODE. "
            "fast: single-scale + greedy CTC (high-volume). "
            "production: SR + multi-scale + greedy CTC (FINAL_REPORT recommended). "
            "research: production stack + TTA + beam + LM (offline ablation only)."
        ),
    )
    args = ap.parse_args()

    records = []
    with args.sample.open() as f:
        for line in f:
            records.append(json.loads(line))

    backend = maybe_hailo_backend()
    if backend is None:
        raise RuntimeError("Hailo backend not available")

    # Forward the mode into the feature extractor via env var (read by
    # openclaw_shared.features.thumbnail's backend.ocr call).
    os.environ["HAILO_OCR_MODE"] = args.mode

    preds = []
    t0 = time.perf_counter()
    for i, r in enumerate(records, 1):
        try:
            feats = extract_thumbnail_features(Path(r["path"]), backend=backend)
            preds.append({**r, **_hailo_only(feats)})
        except Exception as e:
            preds.append({**r, "error": f"{type(e).__name__}: {e}"})
        if i % 5 == 0 or i == len(records):
            dt = time.perf_counter() - t0
            print(f"[{args.mode}] {i}/{len(records)}  ({dt:.1f}s elapsed, {i/dt:.2f} thumb/s)")

    backend.close()

    with args.output.open("w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
    errs = sum(1 for p in preds if "error" in p)
    print(f"wrote {len(preds)} rows → {args.output}  ({errs} errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
