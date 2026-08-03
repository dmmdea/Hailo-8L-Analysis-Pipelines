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
