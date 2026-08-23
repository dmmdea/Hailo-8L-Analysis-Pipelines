"""Provision the host-side assets for the NPU text/zero-shot and whisper tools.

Downloads once into <HAILO_MODELS_DIR>/assets/:
  tinyclip.npz            token_embeddings (49408x512) + projection_layer — parsed
                          straight out of the HF safetensors with numpy (no torch)
  tinyclip_tokenizer.json HF fast-tokenizer spec (read by `tokenizers`)
  siglip2.npz             token_embeddings for the siglip2 text tower (--siglip2)
  siglip2_tokenizer.json
  whisper_tokenizer.json  openai/whisper-base tokenizer (decode + special ids)
  token_embedding_weight_base.npy / onnx_add_input_base.npy
                          hailo-apps' whisper decoder tokenization assets
  mel_filters.npz         whisper's 4 KB mel filterbank (from hailo-apps)

Usage:  python scripts/extract_text_assets.py [--siglip2] [--models-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

TINYCLIP = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
SIGLIP2 = "google/siglip2-base-patch32-256"
WHISPER = "openai/whisper-base"
HAILO_NPY_BASE = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/base/decoder_tokenization"
MEL_FILTERS_URL = "https://raw.githubusercontent.com/hailo-ai/hailo-apps/main/hailo_apps/python/standalone_apps/speech_recognition/assets/mel_filters.npz"

_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "U8": np.uint8}


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest.name}")
        return dest
    print(f"  fetch   {dest.name}  <- {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def _safetensors_tensor(path: Path, name: str) -> np.ndarray:
    """Read ONE tensor from a .safetensors file with numpy only.
    Format: u64le header length, JSON header {name: {dtype, shape, data_offsets}}, raw data.
    BF16 is upconverted via a 16-bit left shift into float32."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        if name not in header:
            candidates = [k for k in header if name.split(".")[-2] in k]
            raise KeyError(f"{name} not in {path.name}; near matches: {candidates[:8]}")
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + hlen + start)
        raw = f.read(end - start)
    shape = tuple(meta["shape"])
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u16.view(np.float32).reshape(shape).copy()
    if dt not in _DTYPES:
        raise ValueError(f"unsupported safetensors dtype {dt} for {name}")
    return np.frombuffer(raw, dtype=_DTYPES[dt]).reshape(shape).astype(np.float32, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default=os.environ.get("HAILO_MODELS_DIR", r"D:\Dev\hailo-models"))
    ap.add_argument("--siglip2", action="store_true", help="also provision the (large) siglip2 text table")
    a = ap.parse_args()
    assets = Path(a.models_dir) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    hf = "https://huggingface.co/{repo}/resolve/main/{file}"

    print("tinyclip text assets:")
    st = _fetch(hf.format(repo=TINYCLIP, file="model.safetensors"), assets / "tinyclip_model.safetensors")
    _fetch(hf.format(repo=TINYCLIP, file="tokenizer.json"), assets / "tinyclip_tokenizer.json")
    out = assets / "tinyclip.npz"
    if not out.exists():
        table = _safetensors_tensor(st, "text_model.embeddings.token_embedding.weight")
        proj = _safetensors_tensor(st, "text_projection.weight")
        np.savez(out, text_embeddings=table, projection_layer=proj)
        print(f"  wrote   {out.name}  table {table.shape}  projection {proj.shape}")
    else:
        print(f"  exists  {out.name}")

    print("whisper assets:")
    _fetch(hf.format(repo=WHISPER, file="tokenizer.json"), assets / "whisper_tokenizer.json")
    _fetch(f"{HAILO_NPY_BASE}/token_embedding_weight_base.npy", assets / "token_embedding_weight_base.npy")
    _fetch(f"{HAILO_NPY_BASE}/onnx_add_input_base.npy", assets / "onnx_add_input_base.npy")
    _fetch(MEL_FILTERS_URL, assets / "mel_filters.npz")

    if a.siglip2:
        print("siglip2 text assets (large):")
        st2 = _fetch(hf.format(repo=SIGLIP2, file="model.safetensors"), assets / "siglip2_model.safetensors")
        _fetch(hf.format(repo=SIGLIP2, file="tokenizer.json"), assets / "siglip2_tokenizer.json")
        out2 = assets / "siglip2.npz"
        if not out2.exists():
            table2 = _safetensors_tensor(st2, "text_model.embeddings.token_embedding.weight")
            np.savez(out2, text_embeddings=table2)
            print(f"  wrote   {out2.name}  table {table2.shape}")
        else:
            print(f"  exists  {out2.name}")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
