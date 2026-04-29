"""End-to-end smoke demo: maybe_hailo_backend() + extract_thumbnail_features.

Proves the openclaw_shared wiring works on a real Danmar thumbnail:
  - grabs a backend via the factory
  - runs OpenCV features (always) + Hailo features (if backend alive)
  - prints both blocks so the Hailo-* keys are visibly populated
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/dmmdea/openclaw-mcp-servers/_shared")

from openclaw_shared.backends.hailo import maybe_hailo_backend
from openclaw_shared.features.thumbnail import extract_thumbnail_features


def main() -> int:
    thumbs = sorted(Path("/home/dmmdea/openclaw-output/youtube-analyst/thumbnails/danmar_recent").glob("*.jpg"))
    if not thumbs:
        print("no thumbnails found", file=sys.stderr)
        return 1

    target = thumbs[0]
    print(f"Target: {target}")

    backend = maybe_hailo_backend()
    print(f"Hailo backend: {backend!r}")

    features = extract_thumbnail_features(target, backend=backend)

    opencv_keys = {k: v for k, v in features.items() if not k.startswith("hailo_")}
    hailo_keys = {k: v for k, v in features.items() if k.startswith("hailo_")}

    print("\n=== OpenCV features (first 10 keys) ===")
    for k in list(opencv_keys)[:10]:
        print(f"  {k} = {opencv_keys[k]}")

    print("\n=== Hailo features ===")
    for k, v in hailo_keys.items():
        if k == "hailo_clip_embedding" and isinstance(v, list):
            print(f"  {k} = list(len={len(v)}, first3={v[:3]}, last3={v[-3:]})")
        else:
            print(f"  {k} = {v}")

    if backend is not None:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
