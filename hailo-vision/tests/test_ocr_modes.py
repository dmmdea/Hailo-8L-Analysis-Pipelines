"""B0 mode-semantics contract tests.

The closed 9-phase OCR plan proved that the old `quality` mode entangled
four independently-tunable knobs (SR / multi-scale / TTA / beam) and only
multi-scale produced a monotonic gain on the deployed rec HEF. B0 split
that into three explicit modes — fast, production, research — with the
cost-vs-quality trade-off on the surface. These tests pin the contract
without touching /dev/hailo0:

  * Mode validation runs BEFORE ensure_initialized() so callers get a
    clean ValueError on bad mode strings even with HAILO_VISION_ENABLED=0.
  * VALID_OCR_MODES is the single source of truth for accepted values;
    consumers (thumbnail.py, run_predictions.py, grid_search_phase6.py)
    must stay in sync with this tuple.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from hailo_runtime import HailoRuntime, HailoRuntimeDisabled, VALID_OCR_MODES


class TestOcrModeContract(unittest.TestCase):
    def test_valid_modes_pinned(self) -> None:
        # If this assertion needs to change, also update:
        # - openclaw_shared.features.thumbnail._VALID_OCR_MODES
        # - hailo-vision/scripts/run_predictions.py argparse choices
        # - FINAL_REPORT.md production stanza
        self.assertEqual(VALID_OCR_MODES, ("fast", "production", "research"))

    def test_invalid_mode_raises_before_init(self) -> None:
        """ValueError must precede HailoRuntimeDisabled — a misconfigured
        mode is a programmer error, not a device problem, and should fail
        loud regardless of whether the device is reachable."""
        runtime = HailoRuntime()
        with mock.patch.dict(os.environ, {"HAILO_VISION_ENABLED": "0"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                runtime.ocr("/tmp/anything.jpg", mode="quality")
            self.assertIn("quality", str(ctx.exception))
            self.assertIn("fast", str(ctx.exception))
            self.assertIn("production", str(ctx.exception))
            self.assertIn("research", str(ctx.exception))

    def test_invalid_mode_with_enabled_env_still_raises_value_error(self) -> None:
        """Even when the runtime would otherwise try to init, mode validation
        runs first."""
        runtime = HailoRuntime()
        with mock.patch.dict(os.environ, {"HAILO_VISION_ENABLED": "1"}, clear=False):
            with self.assertRaises(ValueError):
                runtime.ocr("/tmp/anything.jpg", mode="banana")

    def test_valid_modes_pass_validation_then_hit_init_guard(self) -> None:
        """Each valid mode parses cleanly through validation and only then
        hits the device-init guard. Asserting HailoRuntimeDisabled (not
        ValueError) confirms the mode itself was accepted."""
        runtime = HailoRuntime()
        with mock.patch.dict(os.environ, {"HAILO_VISION_ENABLED": "0"}, clear=False):
            for mode in VALID_OCR_MODES:
                with self.subTest(mode=mode):
                    with self.assertRaises(HailoRuntimeDisabled):
                        runtime.ocr("/tmp/anything.jpg", mode=mode)


if __name__ == "__main__":
    unittest.main()
