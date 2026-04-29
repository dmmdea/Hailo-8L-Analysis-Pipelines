"""Verifies extract_thumbnail_features routes through VisionCache when supplied.

This is a CROSS-REPO integration test: it exercises VisionCache (this repo)
against extract_thumbnail_features (sibling YouTube repo
`OpenClaw-Youtube-Analyst-Skills`). It auto-skips when the sibling repo's
`openclaw_shared.features.thumbnail` module is not importable, so this repo
remains independently testable.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw_shared.cache.vision_cache import VisionCache

try:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from openclaw_shared.features.thumbnail import extract_thumbnail_features  # type: ignore[import-not-found]
    _SIBLING_AVAILABLE = True
except ImportError:
    _SIBLING_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    extract_thumbnail_features = None  # type: ignore[assignment]


def _write_img(img: np.ndarray) -> Path:
    p = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
    cv2.imwrite(str(p), img)
    return p


class _CountingCache:
    """Wraps a real cache to count get/put for the wiring assertion."""

    def __init__(self, real: VisionCache) -> None:
        self.real = real
        self.gets = 0
        self.puts = 0

    def get(self, p):
        self.gets += 1
        return self.real.get(p)

    def put(self, p, f):
        self.puts += 1
        return self.real.put(p, f)


@unittest.skipUnless(
    _SIBLING_AVAILABLE,
    "Requires the sibling repo dmmdea/OpenClaw-Youtube-Analyst-Skills "
    "(or any provider of openclaw_shared.features.thumbnail) installed.",
)
class TestThumbnailCacheWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = _CountingCache(
            VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
        )
        self.img = _write_img(np.full((200, 300, 3), 80, dtype=np.uint8))

    def tearDown(self):
        try:
            self.img.unlink()
        except FileNotFoundError:
            pass

    def test_default_signature_unchanged(self):
        # Backwards-compatible: callers that don't pass cache see no cache traffic
        out = extract_thumbnail_features(self.img)
        self.assertTrue(out["image_ok"])
        self.assertEqual(self.cache.gets, 0)
        self.assertEqual(self.cache.puts, 0)

    def test_first_call_misses_then_puts(self):
        out = extract_thumbnail_features(self.img, cache=self.cache)
        self.assertTrue(out["image_ok"])
        self.assertEqual(self.cache.gets, 1)
        self.assertEqual(self.cache.puts, 1)

    def test_second_call_hits_and_skips_recompute(self):
        extract_thumbnail_features(self.img, cache=self.cache)
        # Reset put counter; expect a hit on the next call (no put)
        before_puts = self.cache.puts
        out = extract_thumbnail_features(self.img, cache=self.cache)
        self.assertTrue(out["image_ok"])
        self.assertEqual(self.cache.gets, 2)
        self.assertEqual(self.cache.puts, before_puts)  # no new put

    def test_cache_payload_preserves_features(self):
        first = extract_thumbnail_features(self.img, cache=self.cache)
        second = extract_thumbnail_features(self.img, cache=self.cache)
        # Compare the OpenCV-derived numerics that should be deterministic
        for k in ("width", "height", "edge_density", "mean_hue", "mean_sat", "warm_color_ratio"):
            self.assertEqual(first[k], second[k], f"mismatch on {k}")

    def test_missing_file_not_cached(self):
        out = extract_thumbnail_features(self.tmp / "nope.jpg", cache=self.cache)
        self.assertFalse(out["image_ok"])
        self.assertEqual(self.cache.puts, 0)

    def test_model_version_bump_invalidates(self):
        extract_thumbnail_features(self.img, cache=self.cache)
        # New cache instance with different model_version sees a miss for the same image
        new_real = VisionCache(self.tmp, model_version="m2", pipeline_version="p1")
        new_counting = _CountingCache(new_real)
        out = extract_thumbnail_features(self.img, cache=new_counting)
        self.assertEqual(new_counting.gets, 1)
        self.assertEqual(new_counting.puts, 1)
        self.assertTrue(out["image_ok"])

    def test_cache_put_failure_does_not_break_return(self):
        class _BadCache:
            def get(self, _p):
                return None

            def put(self, _p, _f):
                raise RuntimeError("disk full")

        out = extract_thumbnail_features(self.img, cache=_BadCache())
        self.assertTrue(out["image_ok"])  # still returns features


if __name__ == "__main__":
    unittest.main()
