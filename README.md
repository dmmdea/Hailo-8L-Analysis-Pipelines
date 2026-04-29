# Hailo-8L vision lane for the OpenClaw YouTube analyst

> Production NPU offload for thumbnail intelligence — face detection, OCR, vehicle detection, CLIP embeddings, and a content-addressed cache that turns 8-second pipeline runs into 8-millisecond cache hits. **Measured 1060× speedup on repeat scans.**

This repository is the Hailo-8L half of a two-repo split:

- **[`OpenClaw-Youtube-Analyst-Skills`](https://github.com/dmmdea/OpenClaw-Youtube-Analyst-Skills)** — the analyst pipeline (channel scans, packaging analysis, matched-pair statistics, ad-spend joins). Runs on any node.
- **`hailo-youtube-stack-mcp`** *(this repo)* — the Hailo-8L NPU lane that the analyst pipeline calls into for vision features, plus the `hailo-vision` MCP server, the content-addressed Vision Cache, the DKMS kernel patch needed on Linux 6.12+, and the operational skill.

The dependency direction is one-way and optional: the analyst pipeline calls `maybe_hailo_backend()` and gets a working accelerator if this repo is installed and a Hailo-8L is present, or `None` otherwise. The OpenCV-only fallback path produces the same feature-dict schema, so downstream matched-pair stats are unaffected by which side computed them. **Nodes without a Hailo can ignore this repo entirely.**

---

## Why this exists

A YouTube channel analyst pipeline produces ~100 features per thumbnail across geometry, color, contrast, faces, OCR, vehicles, and CLIP embeddings. The expensive part is the inference: SCRFD + PaddleOCR-v5 + YOLOv5m + TinyCLIP + Real-ESRGAN, end-to-end ~8 seconds per thumbnail on a Hailo-8L. Running 210 thumbnails takes ~28 minutes cold; rerunning them after a config tweak shouldn't.

This repo packages four things that, together, make the Hailo lane production-grade:

1. **A thin runtime wrapper** (`hailo-vision/hailo_runtime.py`) — one `HailoRuntime` per process, persistent VDevice, ROUND_ROBIN scheduler, six HEFs hot, three deterministic OCR modes.
2. **A content-addressed cache** (`openclaw_shared/cache/vision_cache.py`) — keys feature dicts by `(sha256(asset_bytes), model_version, pipeline_version)`. Same image + same models + same pipeline rev = read from SQLite (and Parquet for embeddings). Otherwise recompute. **Measured 1060× speedup.**
3. **The kernel patch HailoRT 4.23.0 ships without** (`hailo-patch/`) — a 2-line `mmap_read_lock` wrap around `find_vma` that's required on Linux 6.12+. Without it, the first inference hangs for ~30 minutes with no recovery. Backported locally via DKMS so it survives kernel upgrades.
4. **A Claude Code skill** (`Hailo-Stack-Skill/SKILL.md`) — the operational runbook for everything above: device verification, driver lifecycle, HEF management, OCR mode semantics, troubleshooting matrix.

---

## Headline numbers

Live measurement on a Dell OptiPlex 7060 SFF (i5-8500, 64 GB DDR4) with Hailo-8L in the WLAN M.2 slot, against a 103,903-byte JPEG thumbnail. Pipeline: SCRFD-2.5g face detect + YOLOv5m vehicles + PaddleOCR-v5 production multi-scale (det+rec) + TinyCLIP image encoder + 22 OpenCV features.

| Call | Time | What it did |
|---|---|---|
| 1st (cold) | **8010.2 ms** | Full Hailo + OpenCV pipeline, SQLite write, Parquet append |
| 2nd (warm) | **7.6 ms** | Single SQLite read + Parquet point-read |
| **Speedup** | **1060.2×** | |

Verified properties:

- All 54 feature keys identical between cold and warm calls
- 512-dimensional CLIP embedding round-trips bit-exactly through the Parquet sidecar
- Bumping `model_version` (HEF swap) invalidates the cache as expected
- Bumping `pipeline_version` (code change) invalidates the cache as expected
- Failed extractions (`image_ok=False`) are intentionally **not** cached — a transient device error doesn't poison subsequent runs

Reproduce on your own node: see [`tests/live_phase_h_check.py`](tests/live_phase_h_check.py).

---

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  YouTube analyst pipeline   (sibling repo)                  │
│  scripts that call extract_thumbnail_features(path,         │
│                                              backend=...,   │
│                                              cache=...)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │  optional cache=, backend=
                         ▼
┌────────────────────────────────────┐    ┌────────────────────────┐
│  openclaw_shared.cache.            │    │  openclaw_shared.      │
│    VisionCache                     │    │    backends.hailo      │
│  ─ SQLite (vision_facts table)     │    │    maybe_hailo_backend │
│  ─ Parquet (embeddings sidecar)    │    │  ─ env-gated factory   │
│  ─ Keyed by (sha256, model, pipe)  │    │  ─ Returns None when   │
└────────────────────────────────────┘    │    Hailo unavailable    │
                                          └────────────┬───────────┘
                                                       │
                                                       ▼
                          ┌────────────────────────────────────────┐
                          │  hailo-vision/hailo_runtime.py         │
                          │  ─ Persistent VDevice                  │
                          │  ─ ROUND_ROBIN scheduler               │
                          │  ─ 6 HEFs:  scrfd, paddleocr-det,      │
                          │             paddleocr-rec, tinyclip,   │
                          │             real_esrgan, yolov5m       │
                          │  ─ OCR modes: fast / production /      │
                          │              research                  │
                          └─────────────┬──────────────────────────┘
                                        │
                                        ▼
                          ┌────────────────────────────────────────┐
                          │  /dev/hailo0  ←  hailo_pci DKMS 4.23.0 │
                          │                  + mmap_read_lock      │
                          │                    patch (kernel ≥6.12)│
                          └────────────────────────────────────────┘
```

---

## Repository layout

```
hailo-youtube-stack-mcp/
├── README.md                        ← you are here
├── LICENSE                          ← MIT
├── pyproject.toml                   ← installable as `openclaw-hailo`
│
├── Hailo-Stack-Skill/
│   └── SKILL.md                     ← Claude Code operational skill (~300 LoC)
│
├── hailo-vision/                    ← MCP server source
│   ├── server.py                    ← MCP entrypoint
│   ├── hailo_runtime.py             ← HailoRuntime + 4-tool surface
│   ├── pyrightconfig.json
│   ├── scripts/                     ← demo + benchmark + calibration scripts
│   └── tests/                       ← runtime + OCR-mode unit tests
│
├── openclaw_shared/                 ← PEP 420 namespace package
│   ├── cache/
│   │   ├── __init__.py
│   │   └── vision_cache.py          ← VisionCache + fingerprint helpers
│   └── backends/
│       ├── __init__.py
│       └── hailo.py                 ← maybe_hailo_backend() factory
│
├── tests/                           ← cache + wiring + live-device checks
│   ├── test_vision_cache.py         ← 24 unit tests
│   ├── test_thumbnail_cache_wiring.py  ← 7 wiring tests (needs sibling repo)
│   └── live_phase_h_check.py        ← end-to-end against /dev/hailo0
│
├── hailo-patch/                     ← kernel ≥6.12 VDMA patch
│   ├── 0001-mmap-read-lock-around-find-vma.patch
│   └── README.md                    ← apply procedure + lifecycle
│
└── docs/
    ├── HAILO_MAXIMUM_CAPABILITY_PLAN.md   ← post-audit phase plan
    ├── FINAL_REPORT.md                    ← closed 9-phase OCR plan, what worked
    ├── phase_H_done.marker                ← Phase H landing record
    └── phase_H_metrics.json               ← live measurement
```

---

## Installation

This repo expects a Linux host (Ubuntu 24.04+ tested) with a Hailo-8L installed and the HailoRT debs from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) already in place (HailoRT 4.23.0 + apps-infrastructure 25.10.0).

### 1. Apply the kernel patch (required on Linux 6.12+)

See [`hailo-patch/README.md`](hailo-patch/README.md) for the full procedure. Short version:

```bash
sudo cp hailo-patch/0001-*.patch /usr/src/hailo_pci-4.23.0/patches/
echo 'PATCH[0]="0001-mmap-read-lock-around-find-vma.patch"' \
  | sudo tee -a /usr/src/hailo_pci-4.23.0/dkms.conf
sudo dpkg-reconfigure hailort-pcie-driver
ls /dev/hailo*                          # expect /dev/hailo0
hailortcli fw-control identify          # expect Board=Hailo-8, Firmware=4.23.0
```

### 2. Set up the Hailo Python venv

The HailoRT wheel is C-extension-heavy and pinned to Python 3.12. `numpy<2` is mandatory (HailoRT 4.23.0 was compiled against the NumPy 1.x ABI).

```bash
python3.12 -m venv ~/openclaw-venvs/hailo
source ~/openclaw-venvs/hailo/bin/activate
pip install --upgrade pip
pip install ~/Downloads/hailort-4.23.0-cp312-cp312-linux_x86_64.whl
pip install "opencv-python<4.10" "numpy<2" pandas pyarrow openpyxl scipy \
            torch --index-url https://download.pytorch.org/whl/cpu \
            open-clip-torch
```

### 3. Drop the HEFs into `/home/hailo/models/`

All six HEFs are public on the Hailo Model Zoo S3 bucket (no auth):

```bash
BASE=https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l
for hef in scrfd_2.5g paddle_ocr_v5_mobile_detection \
           paddle_ocr_v5_mobile_recognition \
           tinyclip_vit_40m_32_text_19m_laion400m_image_encoder \
           real_esrgan_x2 yolov5m_vehicles ; do
    curl -fL "$BASE/$hef.hef" \
      | sudo install -o hailo -g hailo -m 644 /dev/stdin /home/hailo/models/$hef.hef
done
```

### 4. Install this repo's Python package

```bash
git clone https://github.com/dmmdea/hailo-youtube-stack-mcp.git
cd hailo-youtube-stack-mcp
pip install -e .[embeddings]              # `embeddings` adds pyarrow for the Parquet sidecar
```

### 5. (Optional) Install the Claude Code skill

```bash
cp -r Hailo-Stack-Skill ~/.claude/skills/
# Then restart your Claude Code session and say "verify hailo".
```

### 6. Verify

```bash
python -c "
from openclaw_shared.backends.hailo import maybe_hailo_backend
import os; os.environ['HAILO_VISION_ENABLED'] = '1'
rt = maybe_hailo_backend()
print(rt.embed('/path/to/any/thumbnail.jpg')[:5])  # expect 5 floats
"
```

---

## Quick start: cache-accelerated feature extraction

```python
from openclaw_shared.cache.vision_cache import (
    VisionCache, fingerprint_hef_dir, fingerprint_pipeline,
)
from openclaw_shared.backends.hailo import maybe_hailo_backend

# Sibling repo (OpenClaw-Youtube-Analyst-Skills) supplies extract_thumbnail_features.
# Without it, you can use VisionCache.get_or_compute(path, your_compute_fn) directly.
from openclaw_shared.features.thumbnail import extract_thumbnail_features

cache = VisionCache(
    cache_dir="~/openclaw-output/hailo-vision-cache",  # default if omitted
    model_version=fingerprint_hef_dir("/home/hailo/models"),
    pipeline_version=fingerprint_pipeline("/path/to/this/repo"),
)
backend = maybe_hailo_backend()  # None on a host without Hailo

# First call: ~8 s on a Hailo-8L (full pipeline + cache write).
# Every subsequent call on the same image with the same model+pipeline rev: ~7 ms.
features = extract_thumbnail_features("thumbnail.jpg", backend=backend, cache=cache)

# Inspect cache health
print(cache.stats())
# {'rows_total': 1234, 'rows_current_version': 980}
# rows_total >= rows_current_version means there are legacy rows from prior
# model/pipeline versions. They're shadowed (never returned) but kept on disk
# for archival queries. Wipe with `rm -rf ~/openclaw-output/hailo-vision-cache/`
# if you want a clean slate.
```

---

## OCR mode semantics

`HAILO_OCR_MODE` (env var, also accepted as `runtime.ocr(mode=...)`) chooses one of three deterministic configurations. The four orthogonal knobs (super-resolution / multi-scale detection / per-crop TTA / beam-search CTC) are entangled into three explicit modes after the [closed 9-phase OCR plan](docs/FINAL_REPORT.md) showed only multi-scale was monotonically positive on the deployed `paddle_ocr_v5_mobile_recognition` HEF:

| Mode | SR | Multi-scale | TTA | Beam | Use when |
|---|:---:|:---:|:---:|:---:|---|
| `fast` | – | – | – | – | Bulk scans where char-recall doesn't matter |
| `production` *(default)* | ✓ | ✓ | – | greedy | Default for `extract_thumbnail_features` and analyst-pipeline matched-pair tests |
| `research` | ✓ | ✓ | ✓ | beam + optional KenLM | Offline ablation runs only — regresses year extraction on the current rec HEF; kept for calibration sweeps |

Mode validation runs **before** `ensure_initialized()`, so a misconfigured value raises `ValueError` without touching the device.

---

## What's in [`docs/`](docs)

- **[`HAILO_MAXIMUM_CAPABILITY_PLAN.md`](docs/HAILO_MAXIMUM_CAPABILITY_PLAN.md)** — the post-audit, three-cache, strategic-repetition plan that frames Hailo's role as the deterministic vision side of a learning pipeline (not "an OCR project"). Phases B0/B/B-bis/H done; I/J/K queued; D/A/C gated on Hailo Developer Zone DFC v3.33.1.
- **[`FINAL_REPORT.md`](docs/FINAL_REPORT.md)** — the honest accounting of a closed 9-phase OCR quality plan: which knobs moved metrics, which didn't, and what the structural ceilings are on the current rec HEF (softmax collapse, accent-blindness, logo-only brand-recall gap).
- **[`phase_H_done.marker`](docs/phase_H_done.marker)** + **[`phase_H_metrics.json`](docs/phase_H_metrics.json)** — checkpoint trail for the Vision Cache landing.

---

## Status

| Phase | Description | Status |
|---|---|:---:|
| B0 | OCR mode split (`fast` / `production` / `research`) | ✅ 2026-04-24 |
| B / B-bis | Vehicle detection benchmark + `MODEL_TO_BRAND` lookup | ✅ 2026-04-24 |
| **H** | **Content-addressed Vision Cache** | ✅ **2026-04-28** |
| I | Shorts frame selector | ⏳ next |
| J | Long-form thumbnail validator | ⏳ pending |
| K | Benchmark expansion 90 → 300–500 | ⏳ pending |
| D | Auto-brand logo classifier (custom-train + DFC compile) | 🔒 gated on DFC v3.33.1 |
| A1→A2→C | DFC install + OCR recalibration | 🔒 gated |
| E | Synthetic OCR fine-tune | ◯ conditional |
| G | Production cutover + skill shipping | ◔ partial — this repo is the shipping piece |

---

## Hardware reference

This stack was developed and verified on:

- **Compute host:** Dell OptiPlex 7060 SFF — Intel i5-8500 (6C/6T, 65 W), 64 GB DDR4-2666, NVIDIA RTX 5060 8 GB, Samsung 980 1 TB NVMe
- **NPU:** Hailo-8L `HM21LB1C2KAE` (A+E key M.2 2230, ET grade, 13 TOPS INT8, ~1.5–2.5 W typical) in the OptiPlex's WLAN M.2 slot (PCIe-wired on the SFF/Tower variants; CNVio-only on the Micro)
- **OS:** Ubuntu 24.04.4 LTS Server, kernel 6.17.0-20-generic HWE
- **Hailo stack:** HailoRT 4.23.0 + apps-infrastructure 25.10.0 + DKMS driver 4.23.0 (with the patch in [`hailo-patch/`](hailo-patch))
- **Power envelope:** Hailo-8L draws 1.5–2.5 W typical, well inside the OptiPlex's WLAN-slot budget. No thermal mods required on the SFF chassis.

The same stack is portable to any Linux 24.04+ host with PCIe access to the Hailo silicon — Raspberry Pi 5 with the AI HAT+, mini-PCs with a free M.2, custom edge boxes. The OptiPlex was a budget choice, not a dependency.

---

## License

MIT for everything in this repo. The kernel patch is essentially a 2-line locking-API fix already present on Hailo's master branch; we redistribute it for the `hailo8` branch under public-domain / CC0 terms (see [`hailo-patch/README.md`](hailo-patch/README.md)).

The HEFs themselves are not in this repo — fetch them from the public Hailo Model Zoo S3 bucket (no auth required for the `v2.18.0/hailo8l/` path).

---

## Sibling repository

The YouTube-side analyst pipeline (the consumer of this lane) lives at **[dmmdea/OpenClaw-Youtube-Analyst-Skills](https://github.com/dmmdea/OpenClaw-Youtube-Analyst-Skills)** — Claude Code skills + (eventually) the analyst MCP server source. That repo's `Youtube-Analyst-Runbook` skill defers everything below `maybe_hailo_backend()` to the Hailo-Stack-Skill in this repo.

## Acknowledgements

- [Hailo](https://hailo.ai/) for the Hailo-8L silicon, the Model Zoo, and the open hailort-drivers source (master PR #26 was the source of the patch we backported here).
- [Anthropic](https://www.anthropic.com/) for Claude Code, which co-authored most of this codebase.
