---
name: Hailo-Stack-Skill
description: Operational runbook for the Hailo-8L NPU lane on the OpenClaw worker node — driver lifecycle (DKMS + kernel 6.12+ VDMA patch), HEF management, HailoRT venv, the `hailo-vision` MCP, content-addressed Vision Cache, OCR mode semantics, and verification steps. Use when the user says "verify hailo", "hailo not loading", "rebuild hailo driver", "check /dev/hailo0", "swap a hef", "warm the vision cache", "rerun packaging with hailo", "explain hailo ocr modes", "fix hailo after kernel upgrade", or anything about the Hailo accelerator beneath the analyst pipeline (the `Youtube-Analyst-Runbook` skill consumes this lane via `maybe_hailo_backend()` and stays out of the runtime).
---

# Hailo-8L Stack Skill

Operational runbook for the Hailo-8L NPU lane that backs the analyst pipeline's vision features (faces, OCR, vehicle detect, CLIP embeddings, super-resolution). Sibling skill `Youtube-Analyst-Runbook` consumes this lane via `maybe_hailo_backend()`; it never reaches into the device, driver, or HEF set directly. This skill is where that depth lives.

## When to use

- Verifying the device after a boot, kernel upgrade, or driver bump
- Diagnosing a runtime that initialized but produces zero detections / wrong outputs
- Swapping or recompiling a HEF (model file)
- Warming or invalidating the content-addressed Vision Cache (Phase H)
- Choosing the right OCR mode (`fast` / `production` / `research`) for a workload
- Surviving a kernel ≥ 6.12 upgrade (VDMA patch lifecycle)
- Triaging "Hailo lane fell back to OpenCV" symptoms surfaced by the Runbook

**Out of scope:** the analyst pipeline itself (channels, library scans, matched-pair stats) — that's the `Youtube-Analyst-Runbook` skill. Anything Hailo-tangential (hardware install, BIOS settings, M.2 slot routing) lives in the node deployment plan, not here.

## Hardware + driver invariants

This skill assumes a Dell OptiPlex 7060 SFF (or equivalent) with Hailo-8L installed in the WLAN M.2 slot:

- Module: HM21LB1C2KAE (A+E key 2230, ET grade, 13 TOPS INT8, ~1.5–2.5 W typical)
- PCIe enumeration: `02:00.0` Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01) — `lspci` reports "Hailo-8" even on the 8L silicon (same die, lower TOPS cap)
- Software stack: HailoRT 4.23.0 + Apps Infrastructure v25.10.0 + DKMS driver 4.23.0
- OS target: Ubuntu 24.04+ on kernel 6.17 HWE (DKMS module patch required for kernel ≥ 6.12 — see "Kernel patch lifecycle" below)
- Charset/dictionary assets: `/home/hailo/models/charsets/`, `/home/hailo/models/language_models/`

### Where things live

```
/dev/hailo0                                        char device (root:root, 666)
/usr/bin/hailortcli                                CLI
/usr/src/hailo_pci-4.23.0/                         DKMS source tree (patches/ subdir)
/var/lib/dkms/hailo_pci/4.23.0/                    DKMS build artifacts
/home/hailo/models/                                HEF directory (HailoRT-deb default)
~/openclaw-venvs/hailo/                            Py 3.12 venv with hailort==4.23.0
~/openclaw-mcp-servers/hailo-vision/               MCP wrapper (server.py, hailo_runtime.py)
~/openclaw-mcp-servers/_shared/openclaw_shared/    Shared package: backends/, cache/, features/
~/openclaw-output/hailo-ocr-quality-plan/          Phase markers, metrics, plan docs
~/openclaw-output/hailo-vision-cache/              Phase H cache: cache.db + embeddings.parquet
```

Note the **`/home/hailo/models/`** location: HailoRT's apps-infrastructure deb owns this directory. Earlier project docs referenced `/mnt/ai/hailo/models/` — that path is historical and was retired 2026-04-28 (`hailo_runtime.DEFAULT_MODELS_DIR` updated). If you see it in older runbooks, treat it as `/home/hailo/models/`.

## Verifying the device

Three checks, in order, take less than five seconds:

```bash
ls /dev/hailo*                          # expect /dev/hailo0
hailortcli fw-control identify          # expect Board=Hailo-8, Firmware=4.23.0
dkms status hailo_pci                   # expect "installed" for the running kernel
```

If `hailortcli` reports the firmware but `lspci` doesn't list the device, the kernel module isn't loaded — `sudo modprobe hailo_pci` and check `dmesg | tail` for VDMA / mmap errors (see "Kernel patch lifecycle"). If `dkms status` says `built` but not `installed`, run `sudo dkms install -m hailo_pci -v 4.23.0 -k $(uname -r)`.

Round-trip a one-shot inference to prove the runtime is healthy end-to-end:

```bash
source ~/openclaw-venvs/hailo/bin/activate
python -c "
import sys; sys.path.insert(0, '/home/dmmdea/openclaw-mcp-servers/hailo-vision')
from hailo_runtime import HailoRuntime
rt = HailoRuntime(); rt.ensure_initialized()
print(rt.embed('/home/dmmdea/openclaw-output/youtube-analyst/thumbnails/al_vazquez/-Maf601rtkk.jpg')[:5])
"
```

A 5-element float vector prints in ~1.5 s on a warm device. Anything else (exception, hang > 30 s, all-zeros vector) is a symptom — see Troubleshooting.

## HEF management

The deployed HEF set (all on `/home/hailo/models/`):

| HEF | Role | Source | Throughput on 8L |
|---|---|---|---|
| `scrfd_2.5g.hef` | face detection | Hailo Model Zoo v2.18.0 | 311 FPS |
| `paddle_ocr_v5_mobile_detection.hef` | OCR stage 1 (boxes) | Hailo Model Zoo v2.18.0 | 4.59 FPS (bottleneck) |
| `paddle_ocr_v5_mobile_recognition.hef` | OCR stage 2 (text) | Hailo Model Zoo v2.18.0 | 62.4 FPS |
| `tinyclip_vit_40m_32_text_19m_laion400m_image_encoder.hef` | CLIP image embedding (512-d) | Hailo Model Zoo v2.18.0 | 35 FPS |
| `real_esrgan_x2.hef` | Super-resolution (512×512 → 1024×1024) | Hailo Model Zoo v2.18.0 | n/a (one-shot) |
| `yolov5m_vehicles.hef` | Vehicle detection (single class) | Hailo Model Zoo v2.18.0 | n/a (one-shot) |
| `lprnet.hef` | License-plate recognition | Hailo Model Zoo v2.18.0 — **Chinese-plate trained, not used** | — |

Re-fetch any HEF from the public S3 bucket — no auth, no developer-zone gate:

```bash
HEF=scrfd_2.5g.hef
curl -fL -o /tmp/$HEF \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/$HEF
sha256sum /tmp/$HEF                                      # capture the hash
sudo install -o hailo -g hailo -m 644 /tmp/$HEF /home/hailo/models/
```

**Important:** swapping a HEF changes the `model_version` fingerprint in the Vision Cache (it hashes every `.hef` byte in the directory). Cached entries from the prior fingerprint stay readable but are never returned for the new fingerprint — they're shadowed, not deleted. To purge, see "Vision Cache" → "Eviction".

## HailoRT venv

The HailoRT Python wheel is C-extension-heavy and pinned to Python 3.12 on Ubuntu 24.04. The venv is **not** shared with the analyst pipeline's youtube-data venv:

```
~/openclaw-venvs/hailo/                  this skill's domain
~/openclaw-mcp-servers/youtube-data/.venv/   analyst skill's domain
```

Recreating the venv on a Python or Ubuntu upgrade:

```bash
deactivate 2>/dev/null || true
rm -rf ~/openclaw-venvs/hailo
python3.12 -m venv ~/openclaw-venvs/hailo
source ~/openclaw-venvs/hailo/bin/activate
pip install --upgrade pip
pip install ~/Downloads/hailort-4.23.0-cp312-cp312-linux_x86_64.whl
pip install "opencv-python<4.10" "numpy<2" pandas pyarrow openpyxl scipy \
            torch --index-url https://download.pytorch.org/whl/cpu \
            open-clip-torch
python -c "import hailo_platform; print(hailo_platform.__version__)"   # expect 4.23.0
```

`numpy < 2` is mandatory: HailoRT 4.23 was compiled against NumPy 1.x ABI. opencv-python ships its own numpy pin; we cap below 4.10 to keep that pin compatible.

## hailo-vision MCP layer

The MCP at `~/openclaw-mcp-servers/hailo-vision/` exposes 4 tools:

| Tool | Backend method | Returns |
|---|---|---|
| `hailo_face_detect` | `face_detect(image_path)` | list of `FaceBox(x, y, w, h, score)` |
| `hailo_ocr` | `ocr(image_path, mode=...)` | `OcrResult(text, boxes)` |
| `hailo_embed` | `embed(image_path)` | 512-d list of floats |
| `hailo_status` | runtime introspection | dict of HEFs loaded + device firmware + cache stats |

Plus an undocumented-but-public `vehicle_detect(image_path)` returning `VehicleBox`, and an SR helper `super_resolve(image_path)` for the OCR multi-scale path.

The MCP gates every tool on `HAILO_VISION_ENABLED=1`. With the env unset (or `0`), all tools return a structured error dict so the MCP is registrable on a Hailo-less host without crashing.

```
~/.openclaw/secrets.env
  HAILO_VISION_ENABLED=1
```

## Vision Cache (Phase H)

Content-addressed cache for Hailo-derived features. Same media + same models + same pipeline rev = read from cache; otherwise recompute.

**Key:** `(sha256(asset_bytes), model_version, pipeline_version)` where:

- `sha256(asset_bytes)` is a streaming SHA-256 of the image file
- `model_version` = `fingerprint_hef_dir('/home/hailo/models')` — SHA-256 over `basename:sha256` for every `.hef`, lex-sorted
- `pipeline_version` = `git rev-parse --short=12 HEAD` of the hailo-vision repo, or pinned `v0.1.0` fallback

**Storage:** SQLite at `~/openclaw-output/hailo-vision-cache/cache.db` table `vision_facts` + Parquet sidecar `embeddings.parquet` for the CLIP 512-d vectors (kept out of SQLite to keep row size small — JSON payload stays ~10 KB per asset).

**Wiring:** `extract_thumbnail_features` accepts an optional `cache=` kwarg. Backwards-compatible: callers that don't pass a cache see no cache traffic.

```python
from openclaw_shared.cache.vision_cache import (
    VisionCache, fingerprint_hef_dir, fingerprint_pipeline,
)
from openclaw_shared.backends.hailo import maybe_hailo_backend
from openclaw_shared.features.thumbnail import extract_thumbnail_features

cache = VisionCache(
    model_version=fingerprint_hef_dir("/home/hailo/models"),
    pipeline_version=fingerprint_pipeline("/home/dmmdea/openclaw-mcp-servers/hailo-vision"),
)
features = extract_thumbnail_features(path, backend=maybe_hailo_backend(), cache=cache)
```

**Measured speedup on Dell `/dev/hailo0`** (54-key feature dict including SCRFD + YOLOv5 vehicles + PaddleOCR-v5 production multi-scale + TinyCLIP embed + 22 OpenCV features):

| Call | Time |
|---|---|
| Cold (full Hailo pipeline) | 8010 ms |
| Warm (cache hit) | 7.6 ms |
| **Speedup** | **1060×** |

**Failure semantics:** failed extractions (`image_ok=False`) are intentionally **not** cached, so a transient device error doesn't poison subsequent runs. `cache.put()` exceptions are swallowed silently in the wiring so a full disk never blocks the pipeline output — degrades gracefully to recompute.

**Eviction:** none. Vision-fact rows are ~10 KB JSON; embeddings are ~2 KB Parquet — 10k assets ≈ 100 MB. To force a fresh compute (e.g., HEF swap with no version bump):

```bash
rm -rf ~/openclaw-output/hailo-vision-cache/
```

Or to invalidate selectively, bump the `model_version` argument (different fingerprint → all rows shadowed) — old rows stay readable for archival queries but aren't returned.

**Stats:**

```python
cache.stats()
# {"rows_total": 1234, "rows_current_version": 980}
```

`rows_total` ≥ `rows_current_version` indicates legacy rows from prior model/pipeline versions still on disk.

## OCR mode semantics

`HAILO_OCR_MODE` (env var, also accepted as `ocr(mode=...)`) selects the OCR pipeline shape. The four orthogonal knobs (super-resolution / multi-scale detection / per-crop TTA / beam-search CTC) are entangled into three explicit modes after the FINAL_REPORT showed only multi-scale was monotonically positive:

| Mode | SR | Multi-scale | TTA | Beam | Use when |
|---|---|---|---|---|---|
| `fast` | off | off | off | off | Bulk scans where char-recall doesn't matter |
| `production` *(default)* | on | on | off | greedy | Default for `extract_thumbnail_features` and the analyst pipeline. FINAL_REPORT-recommended. |
| `research` | on | on | on | beam + optional KenLM | Offline ablation runs only. Regresses year extraction in current rec HEF; do not use in production. |

Validation runs before `ensure_initialized()` so a misconfigured mode raises `ValueError` without touching the device.

When the mode changes mid-pipeline (e.g., switching from `production` to `research` for a calibration sweep), the cached vision facts under `(content_hash, model_version, pipeline_version)` may not reflect the new mode's outputs. Bump `pipeline_version` (commit the change, push, re-fingerprint) or use a separate `cache_dir` for research runs.

## Kernel patch lifecycle

**Stock HailoRT 4.23.0 driver crashes on kernel ≥ 6.12.** The `hailo_vdma_buffer_map` helper calls `find_vma` without holding `mmap_read_lock` — a 37-minute hang on first inference, no recovery without reboot. Backport from master PR #26 to the `hailo8` branch is pending as PR #44 upstream; until it ships, we apply locally:

```
/usr/src/hailo_pci-4.23.0/
  patches/0001-mmap-read-lock-around-find-vma.patch    2-line wrap
  dkms.conf                                              PATCH[0] line on its own
```

**Verifying the patch is live:**

```bash
grep PATCH /usr/src/hailo_pci-4.23.0/dkms.conf
# expect: PATCH[0]="0001-mmap-read-lock-around-find-vma.patch"
ls /usr/src/hailo_pci-4.23.0/patches/
# expect: 0001-mmap-read-lock-around-find-vma.patch
```

The `PATCH[0]=...` line MUST be on its own line in `dkms.conf`. An earlier bug had it appended via `tee -a` without a trailing newline, merging it with `AUTOINSTALL=yes` and DKMS silently ignored the patch — yielding the same hang.

**On every kernel bump,** DKMS re-runs `dkms install` and re-applies the patch automatically. No manual intervention required. Verify with `dmesg | grep hailo` after the first inference on the new kernel — no `find_vma` warnings = patch is live.

**Removing the patch** (when Hailo ships 4.23.1 or PR #44 lands in `hailo8`):

```bash
sudo rm /usr/src/hailo_pci-4.23.0/patches/0001-mmap-read-lock-around-find-vma.patch
sudo sed -i '/^PATCH\[0\]=/d' /usr/src/hailo_pci-4.23.0/dkms.conf
sudo dpkg-reconfigure hailort-pcie-driver
```

## Reinstalling after a kernel or driver upgrade

`hailort-pcie-driver` postinst has an interactive `read` under `set -e` that fails with EOF when dpkg has no TTY. Always pipe `yes`:

```bash
yes Y | sudo dpkg -i ~/Downloads/hailort-pcie-driver_4.23.0_all.deb
```

Same applies to `hailort_4.23.0_amd64.deb`. After install:

```bash
sudo dkms status hailo_pci
sudo modprobe hailo_pci
ls /dev/hailo*
```

If `/dev/hailo0` is missing, `dmesg | tail -50` is the first stop — the VDMA error message is unmistakable.

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `/dev/hailo0` missing after boot | DKMS didn't rebuild for new kernel | `sudo dkms install -m hailo_pci -v 4.23.0 -k $(uname -r)` then `sudo modprobe hailo_pci` |
| First inference hangs ~37 min, no recovery | VDMA `find_vma` patch absent or PATCH[0] line malformed | Re-apply patch (see "Kernel patch lifecycle"), `sudo dpkg-reconfigure hailort-pcie-driver` |
| `Missing HEFs under /mnt/ai/hailo/models` | Old default path | Update `hailo_runtime.DEFAULT_MODELS_DIR` to `/home/hailo/models` (already shipped 2026-04-28) or `export HAILO_MODELS_DIR=/home/hailo/models` |
| `hailo_platform` import fails | Wrong venv active, or numpy ≥ 2 installed | `source ~/openclaw-venvs/hailo/bin/activate`; if numpy 2.x, recreate venv |
| All embeddings come back zero-filled | Image preprocessing failed silently (wrong color space) | Ensure inputs are RGB uint8; HSV/BGR mismatches yield zero embeddings without raising |
| OCR returns garbled chars on Spanish brand banners | rec HEF is accent-blind by design (LM corpus stripped accents) | Don't fix it via beam — replace `paddle_ocr_v5_mobile_recognition.hef` with a recalibrated build (Phase A1→A2→C, gated on Hailo Developer Zone) |
| Vision cache returns stale features after HEF swap | `model_version` fingerprint didn't change because env override pinned to old dir | Verify `fingerprint_hef_dir('/home/hailo/models')` returns the new hash; bump cache or `rm -rf ~/openclaw-output/hailo-vision-cache/` |
| Cache hits but `hailo_clip_embedding` is `None` | pyarrow not installed in current venv | `pip install pyarrow` in the calling venv (the cache writes Parquet only when an embedding crosses the boundary) |
| `dpkg -i hailort-pcie-driver` exits 1 with no error | Postinst `read` got EOF | `yes Y | sudo dpkg -i ...` |

## Operational discipline

- **Never bypass `maybe_hailo_backend()` from analyst scripts.** The Runbook skill explicitly forbids reaching into `~/openclaw-mcp-servers/hailo-vision/` directly. The fallback contract (Hailo missing → OpenCV-only with the same dict schema) is what keeps the analyst pipeline standalone-first.
- **Cache is opt-in.** Existing call sites that don't pass `cache=` see no behavior change. New code that wants the speedup constructs a `VisionCache` once and passes it through.
- **Don't cache failures.** `extract_thumbnail_features` only puts to the cache when `image_ok=True`. A transient device error must produce a recompute next time, not a stuck null payload.
- **Bump `pipeline_version` on any feature-extraction code change.** A new commit hash invalidates everything; that's the point. Don't try to selectively invalidate — the rebuild cost is one-time per asset and the alternative (cherry-picked invalidation) is silently wrong when a code change affects an unexpected feature.
- **Regression sentinel for the rec HEF.** Any change to `paddle_ocr_v5_mobile_recognition.hef` (recalibration, retrain, swap) re-runs the 90-thumbnail benchmark before going to production. The bar is `year F1 ≥ 0.930`, `brand F1 ≥ 0.667` (current best with `MODEL_TO_BRAND` lookup); regressions revert.

## Phase tracking

The Hailo work follows a numbered phase plan (`HAILO_MAXIMUM_CAPABILITY_PLAN.md`, on the worker node). Status as of skill v1:

| Phase | Description | Status |
|---|---|---|
| B0 | OCR mode split (`fast` / `production` / `research`) | DONE 2026-04-24 |
| B / B-bis | Vehicle detection benchmark + `MODEL_TO_BRAND` lookup | DONE 2026-04-24 |
| **H** | **Content-addressed Vision Cache** | **DONE 2026-04-28** |
| I | Shorts frame selector | next |
| J | Long-form thumbnail validator | pending |
| K | Benchmark expansion 90 → 300–500 | pending |
| D | Auto-brand logo classifier (custom-train + DFC compile) | gated on Hailo Developer Zone |
| A1→A2→C | DFC v3.33.1 install + OCR recalibration | gated |
| E | Synthetic OCR fine-tune | conditional |
| G | Production cutover + skill shipping | partial — this skill is the shipping piece |

## Sibling skills

- `Youtube-Analyst-Runbook` — the analyst pipeline that consumes this lane via `maybe_hailo_backend()`. Runbook line ~187 references this skill as the source of truth for everything below the backend boundary.
- `Youtube-Analyzer-Setup-Skill` — bootstrap for the analyst MCPs. Hailo is independent of that bootstrap; this skill assumes the device is already physically present and the HailoRT debs are installed.
