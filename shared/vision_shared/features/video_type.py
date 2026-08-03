"""
Video-type classification helpers — currently focused on collab detection.

Combines the title-level collab heuristic (from `features.title.is_collab`) with
an optional manual whitelist of video IDs. The whitelist exists to catch collabs
the title heuristic misses (e.g. titles that don't mention the guest channel
explicitly) and to let the analyst override edge cases.

Whitelists are channel-specific by convention. The the target channel pipeline reads from
`~/output/youtube-analyst/config/target_collab_video_ids.txt`; future
peer-side filters should use parallel per-channel files. This module deliberately
does not hard-code a path so it can be reused across channels.
"""
from __future__ import annotations

from pathlib import Path

from .title import extract_title_features


def load_collab_whitelist(path: Path) -> set[str]:
    """Read a collab whitelist file. Returns an empty set if the file is absent.

    Format:
      - One video_id per line.
      - Lines starting with `#` are comments (skipped).
      - Blank/whitespace-only lines are skipped.
      - IDs are stripped of surrounding whitespace.

    The "absent file → empty set" contract means callers can use the same code
    path on hosts where no whitelist has been authored yet.
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        vid = line.strip()
        if vid and not vid.startswith("#"):
            ids.add(vid)
    return ids


def is_collab_video(
    title: str,
    video_id: str | None = None,
    whitelist: set[str] | None = None,
) -> bool:
    """Classify a single video as collab or not.

    Returns True if EITHER:
      - the title heuristic fires (`@`-mention of another channel, or a
        collab keyword like "saludo especial", "ft.", "colaboración"), OR
      - the video_id appears in `whitelist`.

    Title-only when `whitelist` is None or `video_id` is None — this is the
    correct default for non-the target channel channels (no per-peer whitelist authored).

    For DataFrame-scale work, prefer applying `extract_title_features` over the
    title column once and OR-ing the result with `df["video_id"].isin(whitelist)`
    — that's faster than calling this per row.
    """
    if whitelist and video_id is not None and video_id in whitelist:
        return True
    return bool(extract_title_features(title)["is_collab"])
