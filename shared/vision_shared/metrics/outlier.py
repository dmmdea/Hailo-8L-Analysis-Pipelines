"""
Outlier-score computations for a channel's videos.

The playbook (§5 Week 1) requires a time-normalized performance metric per video
that accounts for (a) age (newer videos haven't had time to accumulate views) and
(b) channel audience size (a 100k-view video means different things on a 10k-sub
vs 10M-sub channel). Two metrics are computed:

  1. `views_per_day`  = view_count / max(days_since_upload, 1)
                        Time-normalized raw interest. Comparable within a single
                        channel's recent (< ~1 year) videos.

  2. `outlier_score`  = view_count / channel_median_view_count (over the reference
                        window). 1.0 = average-for-channel, 5.0 = 5× the channel's
                        typical video. This is the standard practitioner definition
                        used by VidIQ, Playboard, and most creator analytics.

For matched-pair analysis later (Week 2+), we'll compare top-decile vs bottom-half
cohorts *within* a single channel using outlier_score as the ranking key. This
avoids cross-channel niche-confound effects.

Why not views/subscribers: view-to-sub ratio is popular but breaks when subs are
hidden or when a channel has huge legacy subscribers that don't watch current
content. Median-based outlier is more robust.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd


# ISO-8601 duration parser (YouTube returns e.g. "PT12M34S")
_ISO_DURATION_RE = re.compile(
    r"^P"
    r"(?:(?P<days>\d+)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+)H)?"
    r"(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$"
)


def parse_iso8601_duration(duration: str | None) -> float | None:
    """Parse an ISO-8601 duration like 'PT12M34S' or 'P1DT2H' into total seconds.

    Returns None if the input is None or unparseable. YouTube's Data API returns
    durations in this format in videos.list contentDetails.duration.
    """
    if not duration:
        return None
    m = _ISO_DURATION_RE.match(duration.strip())
    if not m:
        return None
    parts = {k: float(v) for k, v in m.groupdict(default="0").items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def days_since_upload(published_at: str | datetime, now: datetime | None = None) -> float:
    """Days between `published_at` and now (or a provided `now` for testing)."""
    if isinstance(published_at, str):
        # YouTube publishes as RFC 3339 / ISO 8601 UTC. Handle both "Z" and "+00:00".
        pa = published_at.replace("Z", "+00:00")
        published = datetime.fromisoformat(pa)
    else:
        published = published_at
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = ref - published
    return delta.total_seconds() / 86400.0


def views_per_day(view_count: int, published_at: str | datetime, now: datetime | None = None) -> float:
    """Views normalized by days-since-upload (clamped at 1 day minimum to avoid /0)."""
    age = max(1.0, days_since_upload(published_at, now=now))
    return view_count / age


def channel_outlier_scores(
    videos: Iterable[dict[str, Any]],
    min_days_old: int = 7,
    recent_window_days: int | None = 365,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Compute outlier scores for a channel's videos.

    Args:
      videos: iterable of video dicts, each requiring keys {video_id, title,
              published_at, view_count}. Optional: {duration_iso8601, like_count,
              comment_count, thumbnail_url, url}.
      min_days_old: videos younger than this are flagged but included with
                    `outlier_score = NaN` because their views haven't stabilized.
                    Default 7 days — typical YouTube view curve plateaus within a
                    week for < 100k-sub channels.
      recent_window_days: if set, only videos uploaded in the last N days contribute
                          to the MEDIAN (but all videos get scored). Default 365
                          keeps recency focus while letting us score older videos
                          against the recent baseline. Pass None to use all videos.
      now: override "now" for deterministic testing.

    Returns a DataFrame sorted DESC by outlier_score (NaN rows last) with columns:
      video_id, title, url, published_at, days_old, duration_seconds, view_count,
      like_count, comment_count, views_per_day, outlier_score, outlier_rank,
      is_too_young, is_in_reference_window.

    The reference MEDIAN used is printed in the DataFrame's `.attrs` dict under
    key 'reference_median_views' for transparency (matched-pair analyses need it).
    """
    rows: list[dict[str, Any]] = []
    ref_now = now or datetime.now(timezone.utc)

    for v in videos:
        if v.get("view_count") is None or v.get("published_at") is None:
            continue
        age = days_since_upload(v["published_at"], now=ref_now)
        rows.append(
            {
                "video_id": v.get("video_id"),
                "title": v.get("title"),
                "url": v.get("url"),
                "published_at": v["published_at"],
                "days_old": age,
                "duration_seconds": parse_iso8601_duration(v.get("duration_iso8601")),
                "view_count": int(v["view_count"]),
                "like_count": int(v["like_count"]) if v.get("like_count") is not None else None,
                "comment_count": int(v["comment_count"]) if v.get("comment_count") is not None else None,
                "thumbnail_url": v.get("thumbnail_url"),
                "views_per_day": v["view_count"] / max(1.0, age),
                "is_too_young": age < min_days_old,
                "is_in_reference_window": (
                    recent_window_days is None or age <= recent_window_days
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df.attrs["reference_median_views"] = float("nan")
        return df

    reference_pool = df[df["is_in_reference_window"] & ~df["is_too_young"]]
    if reference_pool.empty:
        # Fallback: use all videos that aren't too young; if STILL none, use everything
        reference_pool = df[~df["is_too_young"]]
    if reference_pool.empty:
        reference_pool = df
    median_views = reference_pool["view_count"].median()
    df.attrs["reference_median_views"] = float(median_views)
    df.attrs["reference_pool_size"] = int(len(reference_pool))

    df["outlier_score"] = df["view_count"] / median_views if median_views > 0 else float("nan")
    # NaN-out outlier_score for too-young videos (their views haven't stabilized)
    df.loc[df["is_too_young"], "outlier_score"] = float("nan")

    df = df.sort_values("outlier_score", ascending=False, na_position="last").reset_index(drop=True)
    df["outlier_rank"] = df["outlier_score"].rank(ascending=False, method="min").astype("Int64")
    return df


def channel_outlier_scores_with_ads(
    videos: Iterable[dict[str, Any]],
    ad_views_by_video_id: Mapping[str, int],
    min_days_old: int = 7,
    recent_window_days: int | None = 365,
    now: datetime | None = None,
    organic_threshold_views: int = 0,
) -> pd.DataFrame:
    """Outlier scoring that separates paid from organic pull.

    Extends `channel_outlier_scores` by joining per-video ad-claimed views and computing
    a second outlier score keyed to the channel's ORGANIC median (median over videos
    with ad_views_claimed <= organic_threshold_views). This is the score to use for
    packaging/quality claims on the owner's own channel — raw outlier_score conflates
    content quality with ad-spend boost.

    Args:
      videos: same shape as `channel_outlier_scores`.
      ad_views_by_video_id: mapping of video_id → integer ad-claimed-view count for
        that video. Videos absent from the map are treated as ad_views_claimed=0.
      min_days_old, recent_window_days, now: forwarded to `channel_outlier_scores`.
      organic_threshold_views: a video is considered "organic" (contributes to the
        organic median) iff its ad_views_claimed is ≤ this threshold. Default 0 —
        any promoted video is excluded from the organic baseline.

    Returns the same DataFrame as `channel_outlier_scores` with these added columns:
      ad_views_claimed, organic_view_count, is_ad_promoted,
      organic_outlier_score, organic_outlier_rank.

    Also sets `df.attrs["reference_organic_median_views"]` for downstream reports.

    Peer channels (where ad-spend isn't available) should call `channel_outlier_scores`
    directly — do not call this with an empty map as that would falsely imply "all
    views are organic" and flip the interpretation.
    """
    df = channel_outlier_scores(
        videos,
        min_days_old=min_days_old,
        recent_window_days=recent_window_days,
        now=now,
    )
    if df.empty:
        df.attrs["reference_organic_median_views"] = float("nan")
        return df

    df["ad_views_claimed"] = (
        df["video_id"].map(lambda vid: int(ad_views_by_video_id.get(vid, 0))).fillna(0).astype(int)
    )
    df["organic_view_count"] = (df["view_count"] - df["ad_views_claimed"]).clip(lower=0).astype(int)
    df["is_ad_promoted"] = df["ad_views_claimed"] > organic_threshold_views

    # Organic reference pool: in window, not too young, AND not ad-promoted.
    organic_pool = df[
        df["is_in_reference_window"]
        & ~df["is_too_young"]
        & ~df["is_ad_promoted"]
    ]
    # Fallback ladder — mirrors the raw-score pool logic so we always produce a median.
    if organic_pool.empty:
        organic_pool = df[~df["is_too_young"] & ~df["is_ad_promoted"]]
    if organic_pool.empty:
        organic_pool = df[~df["is_ad_promoted"]]
    if organic_pool.empty:
        # No organic videos at all — fall back to raw median so a score exists,
        # but flag it prominently for the caller.
        organic_pool = df
        df.attrs["organic_median_fallback_to_raw"] = True

    organic_median = organic_pool["organic_view_count"].median()
    df.attrs["reference_organic_median_views"] = float(organic_median)
    df.attrs["reference_organic_pool_size"] = int(len(organic_pool))

    if organic_median > 0:
        df["organic_outlier_score"] = df["organic_view_count"] / organic_median
    else:
        df["organic_outlier_score"] = float("nan")
    df.loc[df["is_too_young"], "organic_outlier_score"] = float("nan")

    df["organic_outlier_rank"] = (
        df["organic_outlier_score"].rank(ascending=False, method="min").astype("Int64")
    )
    return df
