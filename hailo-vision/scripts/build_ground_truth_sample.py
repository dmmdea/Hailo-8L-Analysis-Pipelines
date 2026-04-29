"""Phase 0 helper — build the 90-thumbnail sample list and run the current
pipeline over it, so we have:

  (a) `sample_list.jsonl` — targets for manual ground-truth labeling (one line per
      thumbnail, schema: video_id, channel, path).
  (b) `predictions_baseline.jsonl` — current pipeline's output (ocr_text + entity
      extraction) per sample; re-used as the Phase 1 baseline and as a pre-fill
      draft for the human labeling pass.

Sample selection: all 30 Danmar thumbnails + 10 stratified-random from each of
the 6 peer channels = 90 total. Seed 42 so re-runs are deterministic.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dmmdea/openclaw-mcp-servers/_shared")

os.environ.setdefault("HAILO_VISION_ENABLED", "1")

from openclaw_shared.backends.hailo import maybe_hailo_backend
from openclaw_shared.features.thumbnail import extract_thumbnail_features

THUMBS_ROOT = Path("/home/dmmdea/openclaw-output/youtube-analyst/thumbnails")
OUT_DIR = Path("/home/dmmdea/openclaw-output/hailo-ocr-quality-plan")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DANMAR_CHANNEL = "danmar_recent"
PEER_CHANNELS = ("al_vazquez", "auto_sensacion", "car_motor", "carglobe", "diariomotor", "siempre_auto")
PEER_SAMPLES_PER_CHANNEL = 10
RANDOM_SEED = 42


def build_sample_list() -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    records: list[dict] = []

    danmar = sorted((THUMBS_ROOT / DANMAR_CHANNEL).glob("*.jpg"))
    for p in danmar:
        records.append({"video_id": p.stem, "channel": DANMAR_CHANNEL, "path": str(p)})

    for peer in PEER_CHANNELS:
        all_thumbs = sorted((THUMBS_ROOT / peer).glob("*.jpg"))
        if len(all_thumbs) < PEER_SAMPLES_PER_CHANNEL:
            raise RuntimeError(f"{peer} has only {len(all_thumbs)} thumbs, need {PEER_SAMPLES_PER_CHANNEL}")
        sampled = rng.sample(all_thumbs, PEER_SAMPLES_PER_CHANNEL)
        sampled.sort()
        for p in sampled:
            records.append({"video_id": p.stem, "channel": peer, "path": str(p)})

    return records


def _hailo_only(features: dict) -> dict:
    keep = {}
    for k, v in features.items():
        if not k.startswith("hailo_"):
            continue
        if k == "hailo_clip_embedding":
            continue
        keep[k] = list(v) if isinstance(v, (set, tuple)) else v
    return keep


def run_baseline(records: list[dict]) -> list[dict]:
    backend = maybe_hailo_backend()
    if backend is None:
        raise RuntimeError("Hailo backend not available — check HAILO_VISION_ENABLED and /dev/hailo0")
    print(f"[baseline] backend alive: {type(backend).__name__}")

    preds: list[dict] = []
    t0 = time.perf_counter()
    for i, r in enumerate(records, 1):
        path = Path(r["path"])
        try:
            feats = extract_thumbnail_features(path, backend=backend)
            pred = {**r, **_hailo_only(feats)}
        except Exception as e:
            pred = {**r, "error": f"{type(e).__name__}: {e}"}
        preds.append(pred)
        if i % 10 == 0 or i == len(records):
            dt = time.perf_counter() - t0
            print(f"[baseline] {i}/{len(records)}  ({dt:.1f}s elapsed, {i/dt:.1f} thumb/s)")

    backend.close()
    return preds


def main() -> int:
    records = build_sample_list()
    sample_path = OUT_DIR / "sample_list.jsonl"
    with sample_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sample] wrote {len(records)} rows → {sample_path}")

    preds = run_baseline(records)
    pred_path = OUT_DIR / "predictions_baseline.jsonl"
    with pred_path.open("w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
    print(f"[baseline] wrote {len(preds)} rows → {pred_path}")
    errs = sum(1 for p in preds if "error" in p)
    if errs:
        print(f"[baseline] WARNING: {errs} rows had errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
