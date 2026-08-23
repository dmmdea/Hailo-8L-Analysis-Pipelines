"""Live-device smoke tests for HailoRuntime.

These tests run inference on /dev/hailo0 and ONLY execute when the operator
explicitly opts in by setting HAILO_VISION_ENABLED=1 AND HAILO_LIVE_TEST=1.
Both guards are intentional — the live tests exercise the VDMA buffer-map
path that crashed the kernel before the DKMS patch. Running them on an
unpatched driver would hang the system.

Run:
  HAILO_VISION_ENABLED=1 HAILO_LIVE_TEST=1 \\
    ${HAILO_PIPELINES_HOME} -m unittest \\
    tests.test_hailo_live -v
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from hailo_runtime import HailoRuntime


def _should_run_live() -> bool:
    return (
        os.environ.get("HAILO_VISION_ENABLED") == "1"
        and os.environ.get("HAILO_LIVE_TEST") == "1"
    )


@unittest.skipUnless(_should_run_live(), "set HAILO_VISION_ENABLED=1 and HAILO_LIVE_TEST=1 to run")
class TestEmbedLive(unittest.TestCase):
    """Proves VDMA path is unblocked by running TinyCLIP on a real thumbnail."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = HailoRuntime()
        cls.runtime.ensure_initialized()
        cls.thumbnail = _pick_thumbnail()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.close()

    def test_embed_returns_nonzero_vector(self) -> None:
        v = self.runtime.embed(self.thumbnail)
        self.assertEqual(len(v), 512, f"expected 512-d TinyCLIP vector, got {len(v)}")
        nonzero = sum(1 for x in v if abs(x) > 1e-6)
        self.assertGreater(nonzero, 100, "TinyCLIP vector is suspiciously sparse — model not running?")

    def test_embed_is_deterministic(self) -> None:
        v1 = self.runtime.embed(self.thumbnail)
        v2 = self.runtime.embed(self.thumbnail)
        max_diff = max(abs(a - b) for a, b in zip(v1, v2))
        self.assertLess(max_diff, 1e-3, "embed() not deterministic — quantization noise or device issue")

    def test_status_reports_initialized(self) -> None:
        s = self.runtime.status()
        self.assertTrue(s["enabled"])
        self.assertTrue(s["initialized"])
        self.assertIn("tinyclip_vit_40m_32_text_19m_laion400m_image_encoder.hef", s["loaded_networks"])


@unittest.skipUnless(_should_run_live(), "set HAILO_VISION_ENABLED=1 and HAILO_LIVE_TEST=1 to run")
class TestFrontierLive(unittest.TestCase):
    """Live-silicon smoke for the frontier tools (pose / segment / zero-shot /
    transcribe). Each skips independently when its HEF or assets are absent so
    the suite stays honest per-capability rather than all-or-nothing."""

    @classmethod
    def setUpClass(cls) -> None:
        from hailo_runtime import HailoRuntime
        cls.runtime = HailoRuntime()
        cls.thumbnail = _pick_thumbnail()

    def _skip_unless(self, fn):
        from hailo_runtime import DependencyMissing, HEFMissing
        try:
            return fn()
        except (HEFMissing, DependencyMissing) as e:
            self.skipTest(str(e))

    def test_pose_runs_and_shapes(self) -> None:
        r = self._skip_unless(lambda: self.runtime.pose(self.thumbnail))
        self.assertIn("people", r)
        for p in r["people"]:
            self.assertEqual(len(p["keypoints"]), 17)
        self.assertFalse(r["scores_sigmoid_applied"],
                         "deployed pose HEF should emit post-sigmoid scores — export changed?")

    def test_segment_masks_align(self) -> None:
        instances, masks, (h, w), sig = self._skip_unless(
            lambda: self.runtime.segment(self.thumbnail))
        self.assertEqual(masks.shape[0], len(instances))
        if len(instances):
            self.assertEqual(masks.shape[1:], (h, w))
        self.assertFalse(sig)

    def test_zero_shot_discriminates(self) -> None:
        ranked = self._skip_unless(
            lambda: self.runtime.zero_shot(self.thumbnail, ["a photo", "pure white noise"]))
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all("similarity" in r for r in ranked))

    def test_transcribe_returns_text(self) -> None:
        import wave as wave_mod
        import numpy as np
        wav = Path(self.thumbnail).parent / "_live_silence.wav"
        with wave_mod.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
        try:
            r = self._skip_unless(lambda: self.runtime.transcribe(wav))
            self.assertIn("text", r)
        finally:
            wav.unlink(missing_ok=True)


def _pick_thumbnail() -> str:
    """Prefer a the target channel thumbnail; fall back to any peer thumbnail on disk."""
    base = Path("${HAILO_PIPELINES_DATA}")
    for subdir in ("target_channel", "auto_sensacion", "siempre_auto"):
        d = base / subdir
        if d.exists():
            jpegs = sorted(d.glob("*.jpg"))
            if jpegs:
                return str(jpegs[0])
    raise RuntimeError(f"no test thumbnail found under {base}")


if __name__ == "__main__":
    unittest.main()
