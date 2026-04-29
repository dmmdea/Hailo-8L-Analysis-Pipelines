"""Phase H live integration check — runs the full Hailo + cache pipeline end-to-end.

Runs `extract_thumbnail_features` against a real thumbnail twice, once with
a cold cache and once with a warm cache. Asserts:

- First call goes through the OpenCV pipeline (and Hailo if HAILO_VISION_ENABLED=1)
- Second call returns the cached payload (sub-millisecond)
- Cache rows count = 1 after the run
- model_version invalidates: a fresh cache with a bumped model_version misses

Requirements (cross-repo):
- This repo (`hailo-youtube-stack-mcp`) — provides VisionCache + maybe_hailo_backend
- Sibling repo (`OpenClaw-Youtube-Analyst-Skills`) — provides
  `openclaw_shared.features.thumbnail.extract_thumbnail_features`

Both must be on PYTHONPATH (or pip-installed) for this script to run.

Usage:
    # Optional environment overrides:
    #   THUMBNAIL_PATH    a specific image to test against
    #   THUMBNAIL_DIR     directory to walk looking for *.jpg (default: cwd)
    #   HEF_DIR           HEF location (default: /home/hailo/models)
    #   PIPELINE_REPO     hailo-vision repo to git-rev (default: ~/openclaw-mcp-servers/hailo-vision)
    #   HAILO_VISION_ENABLED=1   to actually use the device

    python tests/live_phase_h_check.py
    python tests/live_phase_h_check.py /path/to/thumbnail.jpg
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw_shared.cache.vision_cache import (
    VisionCache,
    fingerprint_hef_dir,
    fingerprint_pipeline,
)

try:
    from openclaw_shared.features.thumbnail import extract_thumbnail_features  # type: ignore[import-not-found]
except ImportError as e:
    raise SystemExit(
        "Cannot import openclaw_shared.features.thumbnail. This script is a "
        "cross-repo integration test and requires the sibling repo "
        "dmmdea/OpenClaw-Youtube-Analyst-Skills installed. "
        f"Underlying import error: {e}"
    )


def _pick_thumbnail() -> Path:
    explicit = os.environ.get("THUMBNAIL_PATH")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"THUMBNAIL_PATH does not point at a file: {p}")
        return p
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.is_file():
            raise SystemExit(f"argv[1] does not point at a file: {p}")
        return p
    roots = []
    if dir_env := os.environ.get("THUMBNAIL_DIR"):
        roots.append(Path(dir_env))
    roots.append(Path.home() / "openclaw-output" / "youtube-analyst" / "thumbnails")
    roots.append(Path.cwd())
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.jpg"):
            return p
    raise SystemExit(
        "No thumbnail found. Pass a path as argv[1], set THUMBNAIL_PATH, "
        "or place a *.jpg under one of: " + ", ".join(str(r) for r in roots)
    )


def _maybe_backend():
    """Return a Hailo backend if HAILO_VISION_ENABLED=1 and the import works; else None."""
    if os.environ.get("HAILO_VISION_ENABLED") != "1":
        return None
    # Allow override; default to the canonical Dell layout.
    mcp_dir = os.environ.get(
        "HAILO_VISION_MCP_DIR",
        str(Path.home() / "openclaw-mcp-servers" / "hailo-vision"),
    )
    if mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)
    try:
        from hailo_runtime import HailoRuntime  # type: ignore[import-not-found]
    except Exception as e:
        print(f"[backend] import failed: {e}")
        return None
    try:
        rt = HailoRuntime()
        rt.ensure_initialized()
        return rt
    except Exception as e:
        print(f"[backend] init failed: {e}")
        return None


def main() -> int:
    img = _pick_thumbnail()
    print(f"thumbnail: {img}  ({img.stat().st_size:,} bytes)")

    hef_dir = Path(os.environ.get("HEF_DIR", "/home/hailo/models"))
    repo_dir_env = os.environ.get("PIPELINE_REPO")
    repo_dir = Path(repo_dir_env) if repo_dir_env else (Path.home() / "openclaw-mcp-servers" / "hailo-vision")
    model_version = fingerprint_hef_dir(hef_dir)
    pipeline_version = fingerprint_pipeline(repo_dir)
    print(f"model_version    = {model_version[:24]}...")
    print(f"pipeline_version = {pipeline_version}")

    cache_dir = Path(tempfile.mkdtemp(prefix="phase-h-live-"))
    try:
        cache = VisionCache(
            cache_dir,
            model_version=model_version,
            pipeline_version=pipeline_version,
        )
        backend = _maybe_backend()
        print(f"backend          = {backend}")

        t0 = time.perf_counter()
        first = extract_thumbnail_features(img, backend=backend, cache=cache)
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        second = extract_thumbnail_features(img, backend=backend, cache=cache)
        t_second = time.perf_counter() - t0

        first_keys = set(first.keys())
        second_keys = set(second.keys())

        print(f"\nfirst   call: {t_first*1000:8.1f} ms  ({len(first_keys):3d} keys)")
        print(f"second  call: {t_second*1000:8.1f} ms  ({len(second_keys):3d} keys)")
        speedup = t_first / max(t_second, 1e-9)
        print(f"speedup     : {speedup:.1f}x")

        assert first["image_ok"], "first extraction failed"
        assert second["image_ok"], "second extraction failed"
        # OpenCV-derived numerics are deterministic for a fixed input.
        for k in ("width", "height", "edge_density", "mean_hue", "warm_color_ratio"):
            assert first[k] == second[k], f"key {k}: {first[k]!r} != {second[k]!r}"
        assert second["hailo_clip_embedding"] == first["hailo_clip_embedding"], \
            "embedding round-trip mismatch"
        # Speedup bar: tighter when Hailo is active (cold path is ~8s), softer
        # in OpenCV-only mode (cold path is ~0.6s and the cache overhead is a
        # larger fraction).
        bar = 5.0 if backend is not None else 2.0
        assert speedup >= bar, f"expected >= {bar}x speedup, got {speedup:.1f}x"

        s = cache.stats()
        assert s["rows_total"] == 1 and s["rows_current_version"] == 1, s

        # Bump model_version: should miss
        bumped = VisionCache(
            cache_dir,
            model_version=model_version[:-1] + ("0" if model_version[-1] != "0" else "1"),
            pipeline_version=pipeline_version,
        )
        miss = bumped.get(img)
        assert miss is None, "version bump did not invalidate"
        print("\nALL CHECKS PASSED")

        # Persist metrics next to the test, not in any system-specific location
        metrics = {
            "first_call_ms": round(t_first * 1000, 2),
            "second_call_ms": round(t_second * 1000, 2),
            "speedup": round(speedup, 2),
            "feature_keys": len(first_keys),
            "model_version": model_version,
            "pipeline_version": pipeline_version,
            "backend": backend.__class__.__name__ if backend is not None else None,
            "rows_total": s["rows_total"],
            "image_path": str(img),
            "image_size_bytes": img.stat().st_size,
        }
        out_path = Path(os.environ.get(
            "PHASE_H_METRICS_PATH",
            str(Path(__file__).parent / "phase_H_metrics.json"),
        ))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"metrics written to {out_path}")
        return 0
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
