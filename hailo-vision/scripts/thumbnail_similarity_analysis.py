"""Thumbnail similarity analysis using TinyCLIP embeddings.

Takes the parquet produced by embed_thumbnail_corpus.py and answers four questions:
  1. Per-channel visual cohesion — how tight is each channel's thumbnail cluster?
  2. Channel-to-channel semantic distance — which peers does Danmar most resemble?
  3. Per-Danmar-thumbnail top-K closest peer thumbnails — for each Danmar video, the 5
     peer thumbnails whose packaging is most visually similar (by CLIP cosine).
  4. Danmar's "visual outliers" — thumbnails most and least similar to the Danmar centroid.

Writes a multi-sheet Excel to
  ~/openclaw-output/youtube-analyst/week-3-hailo/similarity_analysis.xlsx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("/home/dmmdea/openclaw-output/youtube-analyst/embeddings/tinyclip_thumbnails.parquet")
OUT_DIR = Path("/home/dmmdea/openclaw-output/youtube-analyst/week-3-hailo")
OUT_XLSX = OUT_DIR / "similarity_analysis.xlsx"

TARGET_CHANNEL = "danmar_recent"
TOP_K_NEIGHBORS = 5


def unit_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms


def main() -> int:
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} embeddings, {df['channel'].nunique()} channels")

    emb = np.stack(df["embedding"].apply(np.asarray).values).astype(np.float32)
    emb_n = unit_normalize(emb)
    dim = emb_n.shape[1]
    print(f"Embedding dim: {dim}")

    channels = sorted(df["channel"].unique())

    # --- Q1: per-channel cohesion ---
    cohesion_rows = []
    channel_centroids: dict[str, np.ndarray] = {}
    for ch in channels:
        mask = (df["channel"] == ch).values
        ch_emb = emb_n[mask]
        centroid = ch_emb.mean(axis=0)
        centroid_n = centroid / (np.linalg.norm(centroid) or 1.0)
        channel_centroids[ch] = centroid_n
        cos_to_centroid = ch_emb @ centroid_n
        cohesion_rows.append({
            "channel": ch,
            "n_thumbnails": int(mask.sum()),
            "mean_cos_to_centroid": float(cos_to_centroid.mean()),
            "std_cos_to_centroid": float(cos_to_centroid.std()),
            "min_cos_to_centroid": float(cos_to_centroid.min()),
        })
    cohesion_df = pd.DataFrame(cohesion_rows).sort_values("mean_cos_to_centroid", ascending=False)
    print("\nCohesion (how tightly a channel's thumbs cluster around its own centroid):")
    print(cohesion_df.to_string(index=False))

    # --- Q2: channel-to-channel semantic distance ---
    cross = np.zeros((len(channels), len(channels)))
    for i, a in enumerate(channels):
        for j, b in enumerate(channels):
            cross[i, j] = float(channel_centroids[a] @ channel_centroids[b])
    cross_df = pd.DataFrame(cross, index=channels, columns=channels).round(4)
    print("\nChannel-to-channel centroid cosine similarity:")
    print(cross_df.to_string())

    # --- Q3: per-Danmar top-K peer-thumbnail matches ---
    danmar_idx = df.index[df["channel"] == TARGET_CHANNEL].tolist()
    peer_mask = df["channel"] != TARGET_CHANNEL
    peer_indices = df.index[peer_mask].tolist()
    peer_emb = emb_n[peer_mask.values]
    peer_meta = df[peer_mask][["channel", "video_id"]].reset_index(drop=True)

    neighbor_rows = []
    for d_idx in danmar_idx:
        d_vec = emb_n[d_idx]
        sims = peer_emb @ d_vec
        top = np.argsort(-sims)[:TOP_K_NEIGHBORS]
        for rank, peer_row_idx in enumerate(top, 1):
            neighbor_rows.append({
                "danmar_video_id": df.at[d_idx, "video_id"],
                "rank": rank,
                "peer_channel": peer_meta.at[peer_row_idx, "channel"],
                "peer_video_id": peer_meta.at[peer_row_idx, "video_id"],
                "cosine_similarity": float(sims[peer_row_idx]),
            })
    neighbors_df = pd.DataFrame(neighbor_rows)

    # --- Q4: Danmar visual outliers ---
    danmar_emb = emb_n[df["channel"] == TARGET_CHANNEL]
    danmar_meta = df[df["channel"] == TARGET_CHANNEL].reset_index(drop=True)
    danmar_centroid = channel_centroids[TARGET_CHANNEL]
    d_to_centroid = danmar_emb @ danmar_centroid
    outlier_df = danmar_meta.assign(cos_to_danmar_centroid=d_to_centroid).sort_values(
        "cos_to_danmar_centroid"
    )[["video_id", "cos_to_danmar_centroid"]].reset_index(drop=True)

    # --- Write xlsx ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        cohesion_df.to_excel(xw, sheet_name="cohesion", index=False)
        cross_df.to_excel(xw, sheet_name="channel_cross_similarity")
        neighbors_df.to_excel(xw, sheet_name="danmar_top5_peer_matches", index=False)
        outlier_df.to_excel(xw, sheet_name="danmar_outliers", index=False)
    print(f"\nWrote {OUT_XLSX}")

    # --- Punch-line prints ---
    print("\n=== Which peer cluster does Danmar most resemble? ===")
    danmar_to_peers = {ch: float(danmar_centroid @ channel_centroids[ch]) for ch in channels if ch != TARGET_CHANNEL}
    for ch, s in sorted(danmar_to_peers.items(), key=lambda x: -x[1]):
        print(f"  {ch:20s} {s:.4f}")

    print("\n=== Danmar's 3 most unusual thumbnails (furthest from own centroid) ===")
    for _, r in outlier_df.head(3).iterrows():
        print(f"  {r['video_id']}   cos={r['cos_to_danmar_centroid']:.4f}")
    print("\n=== Danmar's 3 most typical thumbnails ===")
    for _, r in outlier_df.tail(3).iterrows():
        print(f"  {r['video_id']}   cos={r['cos_to_danmar_centroid']:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
