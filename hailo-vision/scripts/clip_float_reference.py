"""Float-precision CLIP reference embedding pass for Hailo calibration.

Uses open_clip's TinyCLIP-ViT-40M-32-Text-19M-LAION400M (same model family the
Hailo HEF was compiled from) at float32 on CPU. Re-embeds the 210 thumbnails
and writes a parquet next to the Hailo-output one for comparison.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

THUMB_ROOT = Path("/home/dmmdea/openclaw-output/youtube-analyst/thumbnails")
OUT_PATH = Path("/home/dmmdea/openclaw-output/youtube-analyst/embeddings/tinyclip_thumbnails_float.parquet")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def main() -> int:
    jpegs = sorted(THUMB_ROOT.glob("*/*.jpg"))
    print(f"{len(jpegs)} thumbnails to embed on CPU with {MODEL_NAME}/{PRETRAINED}")

    print("Loading float model...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.train(False)

    rows = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i, jpeg in enumerate(jpegs):
            img = Image.open(jpeg).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            feats = model.encode_image(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            rows.append({
                "channel": jpeg.parent.name,
                "video_id": jpeg.stem,
                "path": str(jpeg),
                "embedding": feats.tolist(),
            })
            if (i + 1) % 20 == 0 or i + 1 == len(jpegs):
                dt = time.monotonic() - t0
                rate = (i + 1) / dt
                eta = (len(jpegs) - (i + 1)) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(jpegs)}] {rate:.2f} thumb/s, ETA {eta:.0f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"Elapsed: {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
