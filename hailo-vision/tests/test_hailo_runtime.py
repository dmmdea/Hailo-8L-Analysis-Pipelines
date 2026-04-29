"""Unit tests for HailoRuntime scaffold.

These tests verify the *scaffold* — error shapes, enablement guard, HEF path
discovery, status reporting. They do NOT touch /dev/hailo0. Once the driver
fix lands, live-device tests are added as tests/test_hailo_runtime_live.py.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from hailo_runtime import (
    ALL_HEFS,
    HailoRuntime,
    HailoRuntimeDisabled,
    HEFMissing,
)


class TestScaffoldGuards(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = HailoRuntime()

    def test_disabled_by_default_blocks_init(self) -> None:
        with mock.patch.dict(os.environ, {"HAILO_VISION_ENABLED": "0"}, clear=False):
            with self.assertRaises(HailoRuntimeDisabled):
                self.runtime.ensure_initialized()

    def test_status_safe_when_disabled(self) -> None:
        with mock.patch.dict(os.environ, {"HAILO_VISION_ENABLED": "0"}, clear=False):
            s = self.runtime.status()
            self.assertFalse(s["enabled"])
            self.assertFalse(s["initialized"])
            self.assertEqual(s["loaded_networks"], [])
            self.assertIn("models_dir", s)

    def test_hefs_missing_when_dir_empty(self) -> None:
        with mock.patch.dict(os.environ, {"HAILO_MODELS_DIR": "/tmp/nonexistent-hailo-dir"}, clear=False):
            s = self.runtime.status()
            self.assertEqual(sorted(s["hefs_missing"]), sorted(ALL_HEFS))

    def test_enabled_but_missing_hefs_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HAILO_VISION_ENABLED": "1", "HAILO_MODELS_DIR": "/tmp/nonexistent-hailo-dir"},
            clear=False,
        ):
            with self.assertRaises(HEFMissing):
                self.runtime.ensure_initialized()


class TestHEFPresence(unittest.TestCase):
    """Smoke test confirming Phase 2 downloads landed at the canonical location."""

    def test_all_4_hefs_on_disk(self) -> None:
        models_dir = Path("/mnt/ai/hailo/models")
        if not models_dir.exists():
            self.skipTest(f"{models_dir} not populated on this host")
        for hef in ALL_HEFS:
            self.assertTrue((models_dir / hef).exists(), f"missing HEF: {hef}")


if __name__ == "__main__":
    unittest.main()
