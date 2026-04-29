"""Tests for openclaw_shared.cache.vision_cache (Phase H)."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw_shared.cache.vision_cache import (
    VisionCache,
    fingerprint_hef_dir,
    fingerprint_pipeline,
    sha256_file,
)


def _write_bytes(b: bytes, suffix: str = ".jpg") -> Path:
    p = Path(tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name)
    p.write_bytes(b)
    return p


def _stub_features(extra: dict | None = None) -> dict:
    out = {
        "image_ok": True,
        "width": 1280,
        "height": 720,
        "hailo_ok": True,
        "hailo_face_count": 2,
        "hailo_ocr_text": "TEST DRIVE",
    }
    if extra:
        out.update(extra)
    return out


class TestSha256File(unittest.TestCase):
    def test_stable_for_same_bytes(self):
        a = _write_bytes(b"hello world")
        b = _write_bytes(b"hello world")
        try:
            self.assertEqual(sha256_file(a), sha256_file(b))
            self.assertEqual(
                sha256_file(a),
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
            )
        finally:
            a.unlink(); b.unlink()

    def test_differs_when_one_byte_changes(self):
        a = _write_bytes(b"hello world")
        b = _write_bytes(b"hello worle")
        try:
            self.assertNotEqual(sha256_file(a), sha256_file(b))
        finally:
            a.unlink(); b.unlink()


class TestFingerprintHefDir(unittest.TestCase):
    def test_glob_default(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.hef").write_bytes(b"AAA")
            (Path(d) / "b.hef").write_bytes(b"BBB")
            (Path(d) / "ignored.txt").write_bytes(b"X")
            fp = fingerprint_hef_dir(d)
            self.assertRegex(fp, r"^[0-9a-f]{64}$")

    def test_explicit_order_matters_for_keys_only_when_set(self):
        # Same files, different declared order = same fingerprint when order is fixed
        # by sorted-glob default; differs when caller passes explicit order.
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.hef").write_bytes(b"AAA")
            (Path(d) / "b.hef").write_bytes(b"BBB")
            default = fingerprint_hef_dir(d)
            ab = fingerprint_hef_dir(d, hefs=["a.hef", "b.hef"])
            ba = fingerprint_hef_dir(d, hefs=["b.hef", "a.hef"])
            self.assertEqual(default, ab)
            self.assertNotEqual(ab, ba)

    def test_changes_when_hef_bytes_change(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.hef"
            f.write_bytes(b"AAA")
            v1 = fingerprint_hef_dir(d)
            f.write_bytes(b"AAB")
            v2 = fingerprint_hef_dir(d)
            self.assertNotEqual(v1, v2)

    def test_missing_hef_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.hef").write_bytes(b"AAA")
            fp = fingerprint_hef_dir(d, hefs=["a.hef", "missing.hef"])
            self.assertRegex(fp, r"^[0-9a-f]{64}$")


class TestFingerprintPipeline(unittest.TestCase):
    def test_fallback_when_no_repo(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(fingerprint_pipeline(d, fallback="vX"), "vX")

    def test_none_returns_fallback(self):
        self.assertEqual(fingerprint_pipeline(None, fallback="v9.9.9"), "v9.9.9")


class TestVisionCacheBasic(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = VisionCache(
            self.tmp,
            model_version="m1",
            pipeline_version="p1",
        )
        self.img = _write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes-1")

    def tearDown(self):
        try:
            self.img.unlink()
        except FileNotFoundError:
            pass

    def test_db_initialized(self):
        self.assertTrue(self.cache.db_path.exists())
        with sqlite3.connect(self.cache.db_path) as c:
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        self.assertIn("vision_facts", tables)

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.get(self.img))

    def test_round_trip_no_embedding(self):
        f = _stub_features()
        self.cache.put(self.img, f)
        got = self.cache.get(self.img)
        self.assertIsNotNone(got)
        self.assertEqual(got["hailo_face_count"], 2)
        self.assertEqual(got["hailo_ocr_text"], "TEST DRIVE")
        self.assertNotIn("_has_embedding", got)  # internal flag stripped on read

    def test_overwrites_on_repeat_put(self):
        self.cache.put(self.img, _stub_features({"hailo_face_count": 1}))
        self.cache.put(self.img, _stub_features({"hailo_face_count": 9}))
        got = self.cache.get(self.img)
        self.assertEqual(got["hailo_face_count"], 9)

    def test_missing_file_get_returns_none(self):
        self.assertIsNone(self.cache.get(self.tmp / "nope.jpg"))

    def test_missing_file_put_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.cache.put(self.tmp / "nope.jpg", _stub_features())


class TestVisionCacheVersionInvalidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.img = _write_bytes(b"image-bytes-version-test")

    def tearDown(self):
        try:
            self.img.unlink()
        except FileNotFoundError:
            pass

    def test_model_version_bump_misses(self):
        c1 = VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
        c1.put(self.img, _stub_features({"hailo_face_count": 3}))
        c2 = VisionCache(self.tmp, model_version="m2", pipeline_version="p1")
        self.assertIsNone(c2.get(self.img))
        # Original key still resolves
        self.assertEqual(
            VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
            .get(self.img)["hailo_face_count"],
            3,
        )

    def test_pipeline_version_bump_misses(self):
        c1 = VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
        c1.put(self.img, _stub_features({"hailo_face_count": 7}))
        c2 = VisionCache(self.tmp, model_version="m1", pipeline_version="p2")
        self.assertIsNone(c2.get(self.img))

    def test_image_bytes_change_misses(self):
        c = VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
        c.put(self.img, _stub_features({"hailo_face_count": 5}))
        # Mutate the same path with new bytes
        self.img.write_bytes(b"different-bytes-now")
        self.assertIsNone(c.get(self.img))


class TestVisionCacheGetOrCompute(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
        self.img = _write_bytes(b"got-or-compute-bytes")
        self.calls = 0

    def tearDown(self):
        try:
            self.img.unlink()
        except FileNotFoundError:
            pass

    def _compute(self, p: Path) -> dict:
        self.calls += 1
        return _stub_features({"hailo_face_count": 11})

    def test_first_call_computes_and_caches(self):
        out = self.cache.get_or_compute(self.img, self._compute)
        self.assertEqual(self.calls, 1)
        self.assertEqual(out["hailo_face_count"], 11)

    def test_second_call_hits_cache(self):
        self.cache.get_or_compute(self.img, self._compute)
        self.cache.get_or_compute(self.img, self._compute)
        self.assertEqual(self.calls, 1)

    def test_failed_compute_not_cached(self):
        def bad_compute(p):
            self.calls += 1
            return {"image_ok": False, "reason": "transient device error"}

        self.cache.get_or_compute(self.img, bad_compute)
        self.cache.get_or_compute(self.img, bad_compute)
        self.assertEqual(self.calls, 2)


class TestVisionCacheEmbeddingSidecar(unittest.TestCase):
    """Embeddings round-trip through the Parquet sidecar.

    Skipped automatically when pyarrow is not installed — the cache module
    declares pyarrow as a soft dep that's only required when an embedding
    actually crosses the cache boundary.
    """

    def setUp(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = VisionCache(self.tmp, model_version="m1", pipeline_version="p1")
        self.img = _write_bytes(b"embedding-test-bytes")

    def tearDown(self):
        try:
            self.img.unlink()
        except (AttributeError, FileNotFoundError):
            pass

    def test_round_trip_embedding(self):
        emb = [0.1, -0.2, 0.3, 0.4, 0.5]
        self.cache.put(self.img, _stub_features({"hailo_clip_embedding": emb}))
        # Caller's dict not mutated
        got = self.cache.get(self.img)
        self.assertIsNotNone(got)
        self.assertEqual(got["hailo_clip_embedding"], emb)
        self.assertNotIn("_has_embedding", got)

    def test_payload_does_not_contain_full_vector(self):
        emb = [0.0] * 512
        self.cache.put(self.img, _stub_features({"hailo_clip_embedding": emb}))
        # Inspect raw SQLite row — embedding should be absent / null in JSON
        with sqlite3.connect(self.cache.db_path) as c:
            row = c.execute("SELECT payload FROM vision_facts").fetchone()
        payload = json.loads(row[0])
        self.assertIsNone(payload["hailo_clip_embedding"])
        self.assertTrue(payload["_has_embedding"])

    def test_two_assets_keep_both_embeddings(self):
        img2 = _write_bytes(b"second-image-bytes")
        try:
            self.cache.put(self.img, _stub_features({"hailo_clip_embedding": [1.0, 2.0, 3.0]}))
            self.cache.put(img2, _stub_features({"hailo_clip_embedding": [4.0, 5.0, 6.0]}))
            self.assertEqual(self.cache.get(self.img)["hailo_clip_embedding"], [1.0, 2.0, 3.0])
            self.assertEqual(self.cache.get(img2)["hailo_clip_embedding"], [4.0, 5.0, 6.0])
        finally:
            img2.unlink()


class TestVisionCacheStats(unittest.TestCase):
    def test_stats_basic(self):
        with tempfile.TemporaryDirectory() as d:
            cache = VisionCache(d, model_version="m1", pipeline_version="p1")
            self.assertEqual(cache.stats(), {"rows_total": 0, "rows_current_version": 0})
            img = _write_bytes(b"stats-test")
            try:
                cache.put(img, _stub_features())
                s = cache.stats()
                self.assertEqual(s["rows_total"], 1)
                self.assertEqual(s["rows_current_version"], 1)

                # A different version should see total>0 but current=0
                cache2 = VisionCache(d, model_version="m2", pipeline_version="p1")
                s2 = cache2.stats()
                self.assertEqual(s2["rows_total"], 1)
                self.assertEqual(s2["rows_current_version"], 0)
            finally:
                img.unlink()


if __name__ == "__main__":
    unittest.main()
