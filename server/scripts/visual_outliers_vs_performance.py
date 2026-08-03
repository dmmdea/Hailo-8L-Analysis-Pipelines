"""Cross-reference visual outliers (CLIP-derived) with organic performance.

Joins:
  A) per_video_roi.xlsx — 69 the target channel videos with view_count, outlier_score, promo_views
  B) tinyclip_thumbnails.parquet — 30 recent the target channel thumbnails with 512-d embeddings

Computes:
  organic_views          = view_count - promo_views (per cross-reference-ads rule)
  organic_outlier_score  = organic_views / channel organic-median
  cos_to_target_centroid = how typical the thumbnail is for the target channel's own cluster

Then answers: do visually unusual thumbnails also perform unusually — breakthrough,
misfire, or unrelated?

Output: sheet in ${HAILO_PIPELINES_DATA}
visual_outliers_vs_performance.xlsx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROI_PATH = Path("${HAILO_PIPELINES_DATA}")
EMB_PATH = Path("${HAILO_PIPELINES_DATA}")
OUT_PATH = Path("${HAILO_PIPELINES_DATA}")

TARGET_CHANNEL = "target_channel"


def main() -> int:
    roi = pd.read_excel(ROI_PATH, sheet_name="per_video_roi")
    emb_df = pd.read_parquet(EMB_PATH)
    print(f"ROI rows: {len(roi)}, embedding rows (all channels): {len(emb_df)}")

    target_emb = emb_df[emb_df["channel"] == TARGET_CHANNEL].reset_index(drop=True)
    print(f"the target channel embedding rows: {len(target_emb)}")

    emb = np.stack(target_emb["embedding"].apply(np.asarray).values).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / np.where(norms == 0, 1, norms)
    centroid = emb_n.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    cos_to_centroid = emb_n @ centroid
    target_emb = target_emb.assign(cos_to_target_centroid=cos_to_centroid.astype(float))

    # Compute organic views + organic outlier score on the full 69-video ROI
    roi["promo_views_filled"] = roi["promo_views"].fillna(0)
    roi["organic_views"] = (roi["view_count"] - roi["promo_views_filled"]).clip(lower=0)
    channel_organic_median = float(roi["organic_views"].median())
    roi["organic_outlier_score"] = roi["organic_views"] / max(1.0, channel_organic_median)
    print(f"Channel organic-median views: {channel_organic_median:.0f}")

    # Join on video_id
    joined = target_emb[["video_id", "cos_to_target_centroid"]].merge(
        roi[[
            "video_id", "title", "url", "view_count", "organic_views",
            "outlier_score", "organic_outlier_score",
            "promo_cost_cop", "promo_count", "revenue_usd", "net_usd", "promoted",
        ]],
        on="video_id",
        how="inner",
    )
    print(f"Joined rows: {len(joined)}")

    # Correlations — spearman (rank) since both are skewed
    from scipy.stats import spearmanr
    rho_raw, p_raw = spearmanr(joined["cos_to_target_centroid"], joined["outlier_score"])
    rho_org, p_org = spearmanr(joined["cos_to_target_centroid"], joined["organic_outlier_score"])
    print(f"\nSpearman rank correlation between visual typicality (cos_to_centroid) and performance:")
    print(f"  vs raw outlier_score:       rho = {rho_raw:.3f}  (p = {p_raw:.3f})")
    print(f"  vs organic_outlier_score:   rho = {rho_org:.3f}  (p = {p_org:.3f})")
    if abs(rho_org) < 0.2:
        verdict = "≈ uncorrelated: unusual packaging is NOT systematically good or bad"
    elif rho_org > 0:
        verdict = "+: more typical packaging → better organic performance"
    else:
        verdict = "−: more unusual packaging → better organic performance (breakthroughs)"
    print(f"  verdict: {verdict}")

    # Visual outlier tails
    joined_sorted = joined.sort_values("cos_to_target_centroid")
    print("\n=== 5 MOST VISUALLY UNUSUAL thumbnails (lowest cos_to_centroid) ===")
    cols = ["video_id", "cos_to_target_centroid", "view_count", "organic_outlier_score", "promoted", "net_usd", "title"]
    print(joined_sorted.head(5)[cols].to_string(index=False))
    print("\n=== 5 MOST VISUALLY TYPICAL thumbnails (highest cos_to_centroid) ===")
    print(joined_sorted.tail(5)[cols].to_string(index=False))

    # Top organic outliers — do they tend to be visually typical or unusual?
    top_perf = joined.sort_values("organic_outlier_score", ascending=False).head(5)
    print("\n=== 5 TOP ORGANIC OVER-PERFORMERS and their visual typicality ===")
    print(top_perf[cols].to_string(index=False))

    # Write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {"metric": "videos in join", "value": len(joined)},
        {"metric": "channel organic median views", "value": channel_organic_median},
        {"metric": "Spearman rho (visual typicality vs raw outlier_score)", "value": float(rho_raw)},
        {"metric": "Spearman p (raw outlier)", "value": float(p_raw)},
        {"metric": "Spearman rho (visual typicality vs organic_outlier_score)", "value": float(rho_org)},
        {"metric": "Spearman p (organic outlier)", "value": float(p_org)},
        {"metric": "verdict", "value": verdict},
    ]
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="summary", index=False)
        joined_sorted.to_excel(xw, sheet_name="all_30_joined_ascending_visual", index=False)
        joined_sorted.head(5).to_excel(xw, sheet_name="top5_visual_outliers", index=False)
        joined_sorted.tail(5).to_excel(xw, sheet_name="top5_visual_typicals", index=False)
        top_perf.to_excel(xw, sheet_name="top5_organic_performers", index=False)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
