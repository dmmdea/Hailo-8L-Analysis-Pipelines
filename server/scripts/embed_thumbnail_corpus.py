"""Batch-embed the thumbnail corpus with TinyCLIP on Hailo-8L.

Walks ${HAILO_PIPELINES_DATA}<channel>/<video_id>.jpg
and writes a parquet at ${HAILO_PIPELINES_DATA}
tinyclip_thumbnails.parquet with columns: channel, video_id, path, embedding (list[float]).

Reuses one persistent VDevice + ROUND_ROBIN scheduler across all thumbnails — first
inference is ~1s (device spin-up), subsequent are ~30 ms each at 35 FPS headroom.

Usage:
  HAILO_VISION_ENABLED=1 \\
    ${HAILO_PIPELINES_HOME} \\
    ${HAILO_PIPELINES_HOME}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

THUMB_ROOT = Path("${HAILO_PIPELINES_DATA}")
OUT_DIR = Path("${HAILO_PIPELINES_DATA}")
OUT_PATH = OUT_DIR / "tinyclip_thumbnails.parquet"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    if os.environ.get("HAILO_VISION_ENABLED") != "1":
        print("ERROR: set HAILO_VISION_ENABLED=1 before running", file=sys.stderr)
        return 2

    import pandas as pd

    from hailo_runtime import HailoRuntime

    jpegs = sorted(THUMB_ROOT.glob("*/*.jpg"))
    print(f"Found {len(jpegs)} thumbnails across {len({p.parent.name for p in jpegs})} channels")
    if not jpegs:
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = HailoRuntime()
    runtime.ensure_initialized()
    print(f"Runtime live. Device status: {runtime.status()['loaded_networks']}")

    rows = []
    t_start = time.monotonic()
    for i, jpeg in enumerate(jpegs):
        try:
            vector = runtime.embed(jpeg)
        except Exception as e:
            print(f"  [{i+1}/{len(jpegs)}] FAIL {jpeg.parent.name}/{jpeg.name}: {e}", file=sys.stderr)
            continue
        rows.append({
            "channel": jpeg.parent.name,
            "video_id": jpeg.stem,
            "path": str(jpeg),
            "embedding": vector,
        })
        if (i + 1) % 25 == 0 or i + 1 == len(jpegs):
            elapsed = time.monotonic() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(jpegs)}] {rate:.1f} thumb/s")

    runtime.close()

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} embeddings to {OUT_PATH}")
    print(f"Channels: {df['channel'].value_counts().to_dict()}")
    print(f"Elapsed: {time.monotonic() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
