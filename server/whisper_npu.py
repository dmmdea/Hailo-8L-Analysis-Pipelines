"""Whisper-base on the Hailo-8L NPU — encoder + decoder HEFs, host glue.

Port of hailo-apps' speech_recognition standalone app (MIT,
hailo-ai/hailo-apps main @ 2026-08-23: whisper_pipeline.py, audio_utils.py,
postprocessing.py) onto this repo's HailoRuntime, with two adaptations:

- torch is replaced by pure numpy: the STFT reproduces torch.stft's exact
  layout (periodic Hann window, center reflect-padding, last frame dropped),
  and the mel filterbank comes from the same 4 KB mel_filters.npz asset.
- the transformers AutoTokenizer is replaced by the `tokenizers` package
  reading whisper_tokenizer.json from the assets dir — special-token ids
  (<|startoftranscript|>, language tags, <|transcribe|>, <|notimestamps|>,
  <|endoftext|>) are RESOLVED from the tokenizer file, never hardcoded, so a
  wrong-order language table cannot exist here.

The decoder HEFs were compiled WITHOUT the tokenization Gather/Add operators,
so the token-embedding lookup + positional add run on host from the two npy
assets hailo-apps publishes (token_embedding_weight_base.npy,
onnx_add_input_base.npy). Greedy decode with a repetition penalty, 5-second
chunks (the encoder's input length), 60 s cap per call.
"""
from __future__ import annotations

import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from accel_activity import device_call

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
MAX_DURATION_SEC = 60
VARIANT = "base"

# Punctuation tokens excluded from the repetition penalty (hailo-apps values).
_PUNCT_TOKENS = (11, 13)


def _assets_dir(models_dir: Path) -> Path:
    return models_dir / "assets"


def required_assets(models_dir: Path) -> list[str]:
    """Asset paths this module needs; missing ones (as strings) for error text."""
    a = _assets_dir(models_dir)
    needed = [
        a / f"token_embedding_weight_{VARIANT}.npy",
        a / f"onnx_add_input_{VARIANT}.npy",
        a / "whisper_tokenizer.json",
        a / "mel_filters.npz",
    ]
    return [str(p) for p in needed if not p.exists()]


def _load_audio(path: Path) -> np.ndarray:
    """Mono float32 waveform at 16 kHz. ffmpeg when available (any container),
    stdlib wave ONLY when ffmpeg is absent (PCM16 WAV).

    The two failure modes are kept apart on purpose: ffmpeg MISSING falls back
    to wave, but ffmpeg FAILING means the file itself is bad and its stderr is
    the correct diagnosis — retrying with a strictly weaker decoder would
    convert "truncated mp4" into a misleading "not a RIFF file" error."""
    from hailo_runtime import InvalidInput  # local import: avoid a cycle at module load

    try:
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
             "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    except FileNotFoundError:
        pass  # no ffmpeg on this box — the wave fallback below is legitimate
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode(errors="replace")[-500:]
        raise InvalidInput(f"ffmpeg could not decode {path}: {tail}") from e
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise InvalidInput(f"{path}: only PCM16 WAV is supported without ffmpeg")
            data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels()).mean(axis=1)
            if w.getframerate() != SAMPLE_RATE:
                # linear resample — good enough for speech at these rates
                n = int(round(len(data) * SAMPLE_RATE / w.getframerate()))
                data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
            return data
    except (wave.Error, EOFError) as e:
        raise InvalidInput(f"{path}: not a valid PCM WAV file ({e}); install ffmpeg for other formats") from e


def _log_mel(audio: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """torch.stft-compatible log-mel: periodic Hann, center reflect pad,
    |stft|^2 with the LAST frame dropped, then whisper's log/clip/scale."""
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    frames = x[idx] * window[None, :]
    spec = np.fft.rfft(frames, axis=1)  # (frames, 201)
    mag2 = (np.abs(spec) ** 2).T[:, :-1]  # (201, frames-1) — torch drops the last frame
    mel = filters @ mag2  # (80, frames-1)
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _improve(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio * (target_peak / peak)
    return audio


def _apply_repetition_penalty(logits: np.ndarray, generated: list[int], penalty: float = 1.5, window: int = 8) -> np.ndarray:
    logits = np.squeeze(logits, axis=0)
    for token in set(generated[-window:]):
        if token not in _PUNCT_TOKENS:
            logits[token] /= penalty
    return logits


def clean_transcription(text: str) -> str:
    """hailo-apps' repeated-sentence collapse."""
    sentences = re.split(r"(?<=[.?])\s+", text)
    unique: list[str] = []
    for sentence in sentences:
        norm = sentence.lower().strip()
        for u in unique:
            nu = u.lower().strip()
            if norm and (norm in nu or nu in norm):
                result = " ".join(unique)
                if not result.endswith((".", "?")):
                    result += "."
                return result
        unique.append(sentence.strip())
    result = " ".join(unique)
    if result and not result.endswith((".", "?")):
        result += "."
    return result


def transcribe_file(runtime: Any, audio_path: Path, language: str = "en") -> dict[str, Any]:
    """Encoder-once + greedy decoder loop per 5 s chunk; returns
    {text, chunks, language, duration_sec}."""
    from hailo_runtime import (  # local import: avoid a cycle at module load
        HEF_WHISPER_DECODER,
        HEF_WHISPER_ENCODER,
        DependencyMissing,
        HailoDeviceError,
        HEFMissing,
        InvalidInput,
        _models_dir,
    )

    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise DependencyMissing("the `tokenizers` package is not installed in the sidecar venv — pip install tokenizers") from e

    # PLATFORM SHORT-CIRCUIT (measured 2026-08-23): under Windows HailoRT 4.24
    # + Hailo-8L, BOTH whisper-base encoder builds time out on a minimal direct
    # feed — and attempting the InferModel path can ABORT the hosting process
    # natively (post-reboot verification: the transcribe attempt killed the
    # sidecar). So on Windows this raises the typed diagnosis BEFORE any device
    # work. Upstream validates whisper on Linux/RPi; Windows whisper support
    # exists only for Hailo-10H on HailoRT 5.x. Remove this gate when a
    # HailoRT 5.x Windows release adds 8L whisper support (or on Linux hosting,
    # where the code below is the faithful hailo-apps port, ready to run).
    import sys as _sys
    if _sys.platform == "win32":
        raise HailoDeviceError(
            "whisper-on-NPU is PLATFORM-BLOCKED on Windows HailoRT 4.24 + Hailo-8L: both "
            "encoder HEF builds time out (Linux/RPi-validated upstream; Windows whisper = "
            "Hailo-10H + HailoRT 5.x), and the inference attempt can crash the host process. "
            "The GPU whisper seat remains this box's STT path. Revisit on HailoRT 5.x "
            "Windows with 8L whisper support or Linux hosting."
        )

    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    missing = required_assets(_models_dir())
    if missing:
        raise HEFMissing(f"whisper decoder assets missing: {missing} — see README (assets are hailo-apps' npy files)")

    a = _assets_dir(_models_dir())
    table = np.load(a / f"token_embedding_weight_{VARIANT}.npy")
    add_input = np.load(a / f"onnx_add_input_{VARIANT}.npy")
    with np.load(a / "mel_filters.npz", allow_pickle=False) as f:
        filters = f["mel_80"]
    tok = Tokenizer.from_file(str(a / "whisper_tokenizer.json"))

    def tid(t: str) -> int:
        i = tok.token_to_id(t)
        if i is None:
            raise InvalidInput(f"token {t!r} not in the whisper tokenizer — unsupported language?")
        return i

    forced = [tid("<|startoftranscript|>"), tid(f"<|{language}|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
    eos = tid("<|endoftext|>")

    runtime.ensure_initialized()
    import hailo_platform as hpf

    d = _models_dir()
    enc_path, dec_path = d / HEF_WHISPER_ENCODER, d / HEF_WHISPER_DECODER
    for p in (enc_path, dec_path):
        if not p.exists():
            raise HEFMissing(f"{p.name} not present in {d}")
    enc_hef = hpf.HEF(str(enc_path))
    dec_hef = hpf.HEF(str(dec_path))
    enc_in_shape = tuple(enc_hef.get_input_vstream_infos()[0].shape)
    chunk_frames = int(np.prod(enc_in_shape)) // 80  # e.g. 500 mel frames = 5 s
    chunk_sec = max(1, chunk_frames // 100)
    dec_name = dec_hef.get_network_group_names()[0]
    sorted_out = list(dec_hef.get_sorted_output_names())
    dec_out_names = [n for n in sorted_out if "conv" in n]
    if not dec_out_names:
        # Guessing here would concatenate arbitrary tensors and argmax fluent
        # nonsense out of them — a wrong answer presented as success. Refuse.
        raise HailoDeviceError(
            "whisper decoder HEF has no 'conv'-named logit outputs — got "
            f"{sorted_out}; this port matches the hailo-apps "
            "fixed-sequence-matmul-split decoder graph only"
        )
    seq_len = int(dec_hef.get_output_vstream_infos()[0].shape[1])

    audio = _improve(_load_audio(audio_path))
    duration = len(audio) / SAMPLE_RATE
    texts: list[str] = []
    seg_samples = chunk_sec * SAMPLE_RATE
    audio = audio[: MAX_DURATION_SEC * SAMPLE_RATE]

    # Inference rides the InferModel API with explicit bindings on an OWN
    # VDevice with group_id "SHARED" — the EXACT flow of the reference
    # (hailo-apps whisper_pipeline.py). Two prior divergences both hung on
    # device (measured 2026-08-23): InferVStreams timed out the encoder's
    # streams, and create_infer_model on the runtime's legacy-configured
    # VDevice deadlocked the scheduler. The runtime's VDevice is SHARED-group
    # too, so the two coexist through the scheduler.
    timeout_ms = 60_000  # a real bound — the reference's 1e8 ms turns any hang into forever
    params = hpf.VDevice.create_params()
    params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = "SHARED"
    vdevice = hpf.VDevice(params)
    enc_model = vdevice.create_infer_model(str(enc_path))
    dec_model = vdevice.create_infer_model(str(dec_path))
    enc_model.input().set_format_type(hpf.FormatType.FLOAT32)
    enc_model.output().set_format_type(hpf.FormatType.FLOAT32)
    dec_model.input(f"{dec_name}/input_layer1").set_format_type(hpf.FormatType.FLOAT32)
    dec_model.input(f"{dec_name}/input_layer2").set_format_type(hpf.FormatType.FLOAT32)
    for name in sorted_out:
        dec_model.output(name).set_format_type(hpf.FormatType.FLOAT32)

    with enc_model.configure() as enc_cfg, dec_model.configure() as dec_cfg:
        enc_b = enc_cfg.create_bindings()
        dec_b = dec_cfg.create_bindings()
        for start in range(0, len(audio), seg_samples):
            chunk = audio[start : start + seg_samples]
            if not chunk.size:
                break
            if len(chunk) < seg_samples:
                chunk = np.pad(chunk, (0, seg_samples - len(chunk)))
            mel = _log_mel(chunk, filters)  # (80, chunk_frames)
            # reference layout: (1, 1, frames, 80) NHWC via expand+transpose
            input_mel = np.transpose(np.expand_dims(np.expand_dims(mel, 0), 2), (0, 2, 3, 1))
            input_mel = np.ascontiguousarray(input_mel.astype(np.float32))
            enc_b.input().set_buffer(input_mel)
            enc_buf = np.zeros(enc_model.output().shape, dtype=np.float32)
            enc_b.output().set_buffer(enc_buf)
            try:
                with device_call():
                    enc_cfg.run([enc_b], timeout_ms)
            except Exception as e:  # HailoRTTimeout and friends
                if "imeout" in type(e).__name__ or "imeout" in str(e):
                    raise HailoDeviceError(
                        "whisper encoder timed out on this platform. MEASURED 2026-08-23: both the "
                        "model-zoo and hailo-apps whisper-base encoder HEFs time out on a minimal "
                        "direct feed under Windows HailoRT 4.24 + Hailo-8L, while every vision HEF "
                        "runs — upstream validates whisper on Linux/RPi (Windows whisper support "
                        "exists only for Hailo-10H on HailoRT 5.x). PLATFORM-BLOCKED, not a host "
                        "bug: revisit on a HailoRT 5.x Windows release with 8L whisper support or "
                        "Linux hosting. The GPU whisper seat remains this box's STT path."
                    ) from e
                raise
            encoded = np.ascontiguousarray(enc_b.output().get_buffer())

            dec_ids = np.zeros((1, seq_len), dtype=np.int64)
            for k, t in enumerate(forced):
                dec_ids[0][k] = t
            generated: list[int] = []
            for i in range(len(forced) - 1, seq_len - 1):
                gather = table[dec_ids]  # (1, seq, width)
                add_output = gather + add_input
                tok_embed = np.transpose(np.expand_dims(add_output, axis=0), (0, 2, 1, 3)).astype(np.float32)
                dec_b.input(f"{dec_name}/input_layer1").set_buffer(encoded)
                dec_b.input(f"{dec_name}/input_layer2").set_buffer(np.ascontiguousarray(tok_embed))
                for name in sorted_out:
                    dec_b.output(name).set_buffer(np.zeros(dec_model.output(name).shape, dtype=np.float32))
                with device_call():
                    dec_cfg.run([dec_b], timeout_ms)
                parts = [np.asarray(dec_b.output(n).get_buffer(), dtype=np.float32).reshape((1, seq_len, -1))
                         for n in dec_out_names]
                logits_all = np.concatenate(parts, axis=2)
                vocab = tok.get_vocab_size(with_added_tokens=True)
                if logits_all.shape[-1] < vocab:
                    raise HailoDeviceError(
                        f"decoder logit width {logits_all.shape[-1]} < tokenizer vocab {vocab} "
                        "— output concat is wrong for this HEF"
                    )
                # A compiler may PAD the logit width past the vocab; argmax must
                # never pick a padding column, so slice to the real vocab.
                logits_all = logits_all[..., :vocab]
                logits = _apply_repetition_penalty(logits_all[:, i], generated)
                next_token = int(np.argmax(logits))
                generated.append(next_token)
                dec_ids[0][i + 1] = next_token
                if next_token == eos:
                    break
            texts.append(tok.decode([t for t in generated if t != eos], skip_special_tokens=True).strip())

    vdevice.release()
    text = clean_transcription(" ".join(t for t in texts if t))
    return {"text": text, "chunks": len(texts), "language": language, "duration_sec": round(duration, 2)}
