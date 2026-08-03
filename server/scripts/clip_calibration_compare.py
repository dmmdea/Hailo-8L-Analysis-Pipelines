"""Calibration comparison: Hailo-quantized TinyCLIP vs CPU float32 TinyCLIP.

Loads both parquets (Hailo + float) for the same 210 thumbnails and measures:
  1. Per-thumbnail cosine(hailo_vec, float_vec) — how faithfully does the HEF
     reproduce the float model's direction for each image?
  2. Spearman rank correlation between the Hailo vs float similarity matrices.
     If rankings line up, the Hailo output is trustworthy for analytical use even
     if absolute cosines drift.
  3. Channel-centroid agreement — do Hailo and float agree on which peer channel
     the target channel is closest to?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HAILO_PATH = Path("${HAILO_PIPELINES_DATA}")
FLOAT_PATH = Path("${HAILO_PIPELINES_DATA}")
OUT_PATH = Path("${HAILO_PIPELINES_DATA}")

TARGET_CHANNEL = "target_channel"


def unit_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def main() -> int:
    h_df = pd.read_parquet(HAILO_PATH)
    f_df = pd.read_parquet(FLOAT_PATH)
    print(f"Hailo: {len(h_df)} rows, Float: {len(f_df)} rows")

    key_cols = ["channel", "video_id"]
    h_df = h_df.sort_values(key_cols).reset_index(drop=True)
    f_df = f_df.sort_values(key_cols).reset_index(drop=True)
    assert (h_df[key_cols] == f_df[key_cols]).all().all(), "row alignment mismatch"

    h_emb = np.stack(h_df["embedding"].apply(np.asarray).values).astype(np.float32)
    f_emb = np.stack(f_df["embedding"].apply(np.asarray).values).astype(np.float32)

    h_dim, f_dim = h_emb.shape[1], f_emb.shape[1]
    print(f"Hailo dim: {h_dim}, Float dim: {f_dim}")

    if h_dim != f_dim:
        print(f"⚠  Dim mismatch — HEF compiler output-shape differs from open_clip's image features.")
        print(f"   Per-thumbnail direct cosine not meaningful. Falling back to rank-based comparison only.")
        direct_cos_meaningful = False
    else:
        direct_cos_meaningful = True

    h_n = unit_normalize(h_emb)
    f_n = unit_normalize(f_emb)

    # 1. per-thumbnail agreement (only if dims match)
    per_thumb = None
    if direct_cos_meaningful:
        per_thumb_cos = np.sum(h_n * f_n, axis=1)
        print("\n=== Per-thumbnail cos(hailo, float) ===")
        print(f"  mean: {per_thumb_cos.mean():.4f}")
        print(f"  std:  {per_thumb_cos.std():.4f}")
        print(f"  min:  {per_thumb_cos.min():.4f}")
        print(f"  max:  {per_thumb_cos.max():.4f}")
        per_thumb = h_df[key_cols].assign(hailo_vs_float_cos=per_thumb_cos).sort_values("hailo_vs_float_cos")

    # 2. rank agreement on the pairwise similarity matrix
    h_sim = h_n @ h_n.T
    f_sim = f_n @ f_n.T
    iu = np.triu_indices(len(h_df), k=1)
    h_flat = h_sim[iu]
    f_flat = f_sim[iu]
    rho, pval = spearmanr(h_flat, f_flat)
    print(f"\n=== Rank agreement on pairwise similarities ===")
    print(f"  Spearman rho: {rho:.4f} (p={pval:.2e})")
    print(f"  n_pairs: {len(h_flat)}")
    if rho >= 0.8:
        print(f"  ✓  Strong rank agreement — Hailo rankings are trustworthy.")
    elif rho >= 0.6:
        print(f"  ~  Moderate rank agreement — use with care, verify claims.")
    else:
        print(f"  ⚠  Low rank agreement — quantization drift is significant.")

    # 3. Channel centroid agreement — the target channel's nearest peer in each embedding space
    channels = sorted(h_df["channel"].unique())

    def nearest_peer(emb_n: np.ndarray, df: pd.DataFrame) -> list[tuple[str, float]]:
        centroids = {}
        for ch in channels:
            mask = (df["channel"] == ch).values
            c = emb_n[mask].mean(axis=0)
            c /= np.linalg.norm(c) or 1.0
            centroids[ch] = c
        target_c = centroids[TARGET_CHANNEL]
        scores = {ch: float(target_c @ centroids[ch]) for ch in channels if ch != TARGET_CHANNEL}
        return sorted(scores.items(), key=lambda x: -x[1])

    h_ranking = nearest_peer(h_n, h_df)
    f_ranking = nearest_peer(f_n, f_df)

    print(f"\n=== the target channel's nearest peer by centroid ===")
    print(f"  {'channel':<18} {'Hailo':>8} {'Float':>8}")
    h_map = dict(h_ranking)
    f_map = dict(f_ranking)
    for ch, _ in h_ranking:
        print(f"  {ch:<18} {h_map[ch]:>8.4f} {f_map[ch]:>8.4f}")

    h_order = [c for c, _ in h_ranking]
    f_order = [c for c, _ in f_ranking]
    order_match = h_order == f_order
    print(f"\n  Ordering match: {order_match}")
    if not order_match:
        print(f"  Hailo order: {h_order}")
        print(f"  Float order: {f_order}")

    # Write xlsx
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"metric": "Spearman rho (pairwise sims)", "value": rho},
        {"metric": "Spearman p-value", "value": pval},
        {"metric": "n_thumbnails", "value": len(h_df)},
        {"metric": "the target channel nearest-peer ordering match", "value": order_match},
        {"metric": "direct_per_thumb_cos_computable", "value": direct_cos_meaningful},
    ])
    peer_cmp = pd.DataFrame([
        {"peer_channel": ch, "hailo_centroid_cos": h_map[ch], "float_centroid_cos": f_map[ch]}
        for ch in [c for c, _ in h_ranking]
    ])
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        peer_cmp.to_excel(xw, sheet_name="target_peer_cosines", index=False)
        if per_thumb is not None:
            per_thumb.to_excel(xw, sheet_name="per_thumbnail_agreement", index=False)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
