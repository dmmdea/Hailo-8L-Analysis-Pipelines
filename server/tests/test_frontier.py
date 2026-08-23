"""Unit tests for the frontier additions — device-free.

Covers the host-side math that the NPU legs depend on:
  - decoders.py: yolov8 pose/seg decode against SYNTHETIC head tensors whose
    correct answer is computed by hand (a peaked DFL distribution decodes to a
    known box; a keypoint value decodes to a known pixel), plus the seg mask
    path via a constructed prototype.
  - whisper_npu: mel spectrogram shape/dtype, and — when torch is available —
    numerical parity with the reference torch.stft implementation the port
    replaces (the discriminating test for the STFT layout).
  - server/http registry: the five new sidecar tools are exposed.

Live-silicon behavior is covered by tests/test_hailo_live.py extensions and
the deployment validation; nothing here touches /dev/hailo0.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from decoders import REG_MAX, yolov8_pose_decode, yolov8_seg_decode  # noqa: E402


def _peaked_dfl(bin_index: int) -> np.ndarray:
    """A DFL distribution whose softmax·arange ≈ bin_index (one hot at the bin)."""
    d = np.full(REG_MAX + 1, -20.0, dtype=np.float32)
    d[bin_index] = 20.0
    return d


def _pose_outputs(cell_rc=(10, 20), bin_index=4, score=0.9, kpt_raw=0.75):
    """Synthetic 3-scale pose head with ONE confident cell on stride 8."""
    outs = {}
    for stride, hw in ((8, 80), (16, 40), (32, 20)):
        box = np.zeros((1, hw, hw, 64), dtype=np.float32)
        box[..., :] = np.tile(_peaked_dfl(0), 4)  # elsewhere: zero-distance boxes
        sc = np.zeros((1, hw, hw, 1), dtype=np.float32)
        kpt = np.zeros((1, hw, hw, 51), dtype=np.float32)
        if stride == 8:
            r, c = cell_rc
            box[0, r, c] = np.tile(_peaked_dfl(bin_index), 4)
            sc[0, r, c, 0] = score
            kpt[0, r, c] = np.tile([kpt_raw, kpt_raw, 3.0], 17)  # conf logit 3 → ~0.95
        outs[f"model/conv{stride}_box"] = box
        outs[f"model/conv{stride}_score"] = sc
        outs[f"model/conv{stride}_kpt"] = kpt
    return outs


class TestPoseDecode(unittest.TestCase):
    def test_hand_computed_box_and_keypoints(self):
        dec = yolov8_pose_decode(_pose_outputs(), score_thres=0.3)
        self.assertEqual(len(dec["scores"]), 1, "exactly the one confident cell survives")
        # cell (10, 20) stride 8 → center (20.5·8, 10.5·8) = (164, 84); DFL bin 4
        # → each side 4·8 = 32 px → xyxy (132, 52, 196, 116)
        np.testing.assert_allclose(dec["boxes"][0], [132, 52, 196, 116], atol=0.5)
        self.assertAlmostEqual(float(dec["scores"][0]), 0.9, places=3)
        # kpt raw 0.75 → x = stride·(2·0.75 − 0.5) + 164 = 8 + 164 = 172; y = 8 + 84 = 92
        k = dec["keypoints"][0]
        self.assertEqual(k.shape, (17, 2))
        np.testing.assert_allclose(k[0], [172, 92], atol=0.5)
        self.assertGreater(float(dec["joint_scores"][0][0]), 0.9)
        self.assertFalse(dec["scores_sigmoid_applied"], "in-range scores must not be re-sigmoided")

    def test_no_detections_below_threshold(self):
        dec = yolov8_pose_decode(_pose_outputs(score=0.1), score_thres=0.3)
        self.assertEqual(len(dec["scores"]), 0)

    def test_logit_scores_get_sigmoid(self):
        # A REAL logit map has large-negative background, not zero — build one.
        outs = _pose_outputs(score=4.0)
        for name, t in outs.items():
            if t.shape[-1] == 1:  # the score tensors
                bg = t == 0.0
                t[bg] = -10.0
        dec = yolov8_pose_decode(outs, score_thres=0.3)
        self.assertTrue(dec["scores_sigmoid_applied"])
        self.assertEqual(len(dec["scores"]), 1)
        self.assertAlmostEqual(float(dec["scores"][0]), 1 / (1 + np.exp(-4.0)), places=3)


def _seg_outputs(cell_rc=(10, 20), bin_index=4, score=0.9, num_classes=80):
    outs = {}
    for stride, hw in ((8, 80), (16, 40), (32, 20)):
        box = np.zeros((1, hw, hw, 64), dtype=np.float32)
        box[..., :] = np.tile(_peaked_dfl(0), 4)
        sc = np.zeros((1, hw, hw, num_classes), dtype=np.float32)
        cf = np.zeros((1, hw, hw, 32), dtype=np.float32)
        if stride == 8:
            r, c = cell_rc
            box[0, r, c] = np.tile(_peaked_dfl(bin_index), 4)
            sc[0, r, c, min(3, num_classes - 1)] = score  # class 3 (or 0 single-class)
            cf[0, r, c, 0] = 5.0  # selects proto channel 0
        outs[f"model/conv{stride}_box"] = box
        outs[f"model/conv{stride}_score"] = sc
        outs[f"model/conv{stride}_coeff"] = cf
    proto = np.zeros((1, 160, 160, 32), dtype=np.float32)
    proto[..., 0] = 4.0  # sigmoid(5·4) ≈ 1 everywhere → mask = the box crop
    outs["model/proto"] = proto
    return outs


class TestSegDecode(unittest.TestCase):
    def test_box_class_and_mask_crop(self):
        dec = yolov8_seg_decode(_seg_outputs(), num_classes=80, score_thres=0.3)
        self.assertEqual(len(dec["scores"]), 1)
        self.assertEqual(int(dec["classes"][0]), 3)
        np.testing.assert_allclose(dec["boxes"][0], [132, 52, 196, 116], atol=0.5)
        m = dec["masks"][0]
        self.assertEqual(m.shape, (640, 640))
        self.assertTrue(m[84, 164], "mask must cover the box center")
        self.assertFalse(m[300, 300], "mask must be cropped outside the box")
        # area ≈ box area (proto ≈ 1 inside the crop)
        self.assertAlmostEqual(m.sum() / (64 * 64), 1.0, delta=0.15)

    def test_single_class_fastsam_shape(self):
        dec = yolov8_seg_decode(_seg_outputs(num_classes=1), num_classes=1, score_thres=0.3)
        self.assertEqual(len(dec["scores"]), 1)
        self.assertEqual(int(dec["classes"][0]), 0)


class TestWhisperMel(unittest.TestCase):
    def _filters(self):
        # torch-free fallback: a plausible 80×201 filterbank is enough for
        # shape tests; parity test builds the real mel comparison only w/ torch.
        return np.eye(80, 201, dtype=np.float32)

    def test_shape_five_seconds(self):
        from whisper_npu import _log_mel
        mel = _log_mel(np.zeros(80000, dtype=np.float32), self._filters())
        self.assertEqual(mel.shape, (80, 500))
        self.assertEqual(mel.dtype, np.float32)

    def test_torch_parity(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available for the parity arm")
        from whisper_npu import HOP_LENGTH, N_FFT, _log_mel
        rng = np.random.default_rng(7)
        audio = rng.standard_normal(80000).astype(np.float32) * 0.1
        filters = (rng.random((80, 201)) * 0.01).astype(np.float32)
        # reference: hailo-apps audio_utils.log_mel_spectrogram (torch)
        t_audio = torch.from_numpy(audio)
        window = torch.hann_window(N_FFT)
        stft = torch.stft(t_audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        mel = torch.from_numpy(filters) @ mag
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        ref = ((log_spec + 4.0) / 4.0).numpy()
        got = _log_mel(audio, filters)
        np.testing.assert_allclose(got, ref, atol=1e-4)


class TestWhisperPureHelpers(unittest.TestCase):
    def test_repetition_penalty_divides_recent_non_punct(self):
        from whisper_npu import _apply_repetition_penalty
        logits = np.ones((1, 100), dtype=np.float32) * 2.0
        out = _apply_repetition_penalty(logits.copy(), generated=[7, 11, 13, 42], penalty=2.0)
        self.assertAlmostEqual(float(out[7]), 1.0)   # penalized
        self.assertAlmostEqual(float(out[42]), 1.0)  # penalized
        self.assertAlmostEqual(float(out[11]), 2.0)  # punctuation exempt
        self.assertAlmostEqual(float(out[13]), 2.0)  # punctuation exempt
        self.assertAlmostEqual(float(out[50]), 2.0)  # untouched

    def test_repetition_penalty_window(self):
        from whisper_npu import _apply_repetition_penalty
        logits = np.ones((1, 100), dtype=np.float32) * 2.0
        generated = [5] + [20] * 8  # token 5 fell out of the 8-token window
        out = _apply_repetition_penalty(logits.copy(), generated, penalty=2.0, window=8)
        self.assertAlmostEqual(float(out[5]), 2.0)
        self.assertAlmostEqual(float(out[20]), 1.0)

    def test_clean_transcription_collapses_repeats(self):
        from whisper_npu import clean_transcription
        self.assertEqual(
            clean_transcription("The fox jumps. The fox jumps. Something else."),
            "The fox jumps.",
        )
        self.assertEqual(clean_transcription("One thing. Another thing."),
                         "One thing. Another thing.")
        self.assertEqual(clean_transcription("no terminal punctuation"),
                         "no terminal punctuation.")


class TestSidecarRegistry(unittest.TestCase):
    def test_new_tools_registered(self):
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("hailo_http_probe", str(_HERE / "http_server.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as e:
            # server.py needs the sidecar venv (dotenv/mcp); on a bare dev box
            # fall back to a TEXT-level pin so the registry is still guarded.
            src = (_HERE / "http_server.py").read_text(encoding="utf-8")
            for tool in ("pose", "segment", "text_embed", "zero_shot", "transcribe"):
                self.assertIn(f'"{tool}":', src, f"sidecar must expose {tool} (import skipped: {e})")
            return
        for tool in ("pose", "segment", "text_embed", "zero_shot", "transcribe"):
            self.assertIn(tool, mod.TOOLS, f"sidecar must expose {tool}")


if __name__ == "__main__":
    unittest.main()
