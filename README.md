# Hailo-8L Analysis Pipelines

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Hailo-8L](https://img.shields.io/badge/NPU-Hailo--8L_M.2-0091EA) ![MCP Server](https://img.shields.io/badge/MCP-server-8A2BE2) ![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB) ![Edge AI](https://img.shields.io/badge/inference-100%25_on--device-orange)

Production vision pipelines running entirely on a **Hailo-8L M.2 NPU module** (HM21LB1C2KAE): an MCP server that exposes the accelerator to any MCP-capable agent, a vendored feature library with Spanish-aware OCR correction, a measured multi-phase OCR quality program, and the Linux kernel driver patch that keeps the whole thing alive on modern kernels.

Everything here ran as a working, productive pipeline for thumbnail and image analysis at scale. No cloud inference anywhere: every model executes on the 8L.

## MCP server

`server/server.py` exposes 14 tools over MCP (the same set the HTTP sidecar serves):

| Tool | Model | Notes |
|---|---|---|
| `hailo_face_detect(image_path)` | SCrFD-2.5G | ~311 FPS, 5 landmarks per face |
| `hailo_face_embed(image_path)` | SCrFD → ArcFace (landmark-aligned) | 512-d identity vector per face |
| `hailo_ocr(image_path)` | PaddleOCR v5 (det + rec) | ~4.59 FPS (detection-bound) |
| `hailo_embed(image_path)` | TinyCLIP ViT-61M image encoder | 512-d vector |
| `hailo_object_detect(image_path)` | YOLOv8s (on-chip NMS) | 80 COCO classes |
| `hailo_person_embed(image_path)` | YOLOv8s → OSNet | 512-d re-id, works with no visible face |
| `hailo_depth(image_path)` | Depth-Anything-V2 ViT-S | 224 px relative depth PNG |
| `hailo_enhance_low_light(image_path)` | Zero-DCE | original-resolution brightening |
| `hailo_pose(image_path)` | YOLOv8s-pose (raw head, host decode) | 17 COCO keypoints per person |
| `hailo_segment(image_path, everything=)` | YOLOv8s-seg / FastSAM-s (raw head) | instance-id mask PNG; `everything=True` = class-agnostic |
| `hailo_text_embed(text, space=)` | TinyCLIP text tower ON the NPU (or siglip2) | text vectors in the image-embedding space |
| `hailo_zero_shot(image_path, labels)` | TinyCLIP or SigLIP2 pair, both towers on-NPU | free-text labels → ranked similarities |
| `hailo_transcribe(audio_path)` | Whisper-base encoder+decoder HEFs | **PLATFORM-BLOCKED on Windows HailoRT 4.24** (both HEF builds time out; upstream validates Linux/RPi only) — the port is complete and returns a typed diagnosis; revisit on HailoRT 5.x Windows or Linux hosting |
| `hailo_status()` | n/a | device + runtime state, safe to call with the driver down |

Raw-head decoding (pose/seg/FastSAM) is the model zoo's own postprocessing math ported to
plain numpy (`server/decoders.py`); the text towers run their transformer ON the NPU with
host-side tokenization + EOT-gather/projection (`tokenizers` package). The Whisper pair is
hailo-apps' h8l build with its decode loop ported torch-free (`server/whisper_npu.py` —
the mel path is unit-tested for numerical parity against the torch reference).

**Extra host-side assets** (text/zero-shot/transcribe only): run
`python scripts/extract_text_assets.py` once — it provisions token-embedding tables (parsed
from HF safetensors with numpy, no torch), tokenizer files, whisper decoder npys and the mel
filterbank into `<HAILO_MODELS_DIR>/assets/`, and needs `pip install tokenizers` in the
serving venv. `--siglip2` adds the (large) SigLIP2 text table.

The server starts even when `HAILO_VISION_ENABLED != '1'`: tools return structured error dicts describing the disabled state, and flipping the env var makes them live without restarting the MCP host. Secrets load from `~/.hailo-vision/secrets.env` when present.

## Repository layout

```
server/            MCP server, HailoRT runtime wrapper, tests
server/scripts/    Analysis pipelines: ground-truth building, CLIP calibration
                   (quantized-vs-float agreement), thumbnail similarity,
                   visual outliers vs performance, phase-6 grid search,
                   end-to-end demo, prediction runs
shared/            vision_shared: vendored feature library
                   - backends/hailo.py: NPU backend selection
                   - features/: thumbnail features, OCR correction (SymSpell +
                     14 MB Spanish dictionary), OCR entity extraction, title
                     and video-type features
                   - metrics/: outlier scoring
quality-program/   The OCR maximum-capability program: plan, final report, and
                   per-phase metrics JSON (baseline through phase 7)
kernel-patch/      The VDMA stability patch (see below)
```

## OCR quality program

A phased, measurement-driven program to reach maximum OCR capability on the 8L for real-world thumbnails (stylized typography, Spanish text): baseline benchmark, decoding-mode comparison, SymSpell correction against a 14 MB Spanish frequency dictionary, entity extraction, calibration checks of the quantized CLIP encoder against a float reference, and grid-searched parameter tuning. Each phase's metrics are committed as JSON in `quality-program/`; the plan and final report document method and results.

## Kernel patch

`kernel-patch/0001-mmap-read-lock-around-find-vma.patch` wraps `find_vma` in `hailo_vdma_buffer_map` with `mmap_read_lock` / `mmap_read_unlock`.

Without it, on recent kernels the VDMA buffer-map ioctl triggers a `find_vma` kernel oops that leaves the system unresponsive until a forced reboot. With it, the ioctl completes cleanly.

- Tested on Ubuntu 24.04.2 LTS, HWE kernel 6.17.0-20-generic, Hailo-8L M.2 (HM21LB1C2KAE), firmware 4.23.0
- Submitted upstream to the Hailo driver repository (refs: upstream master PR #26, hailo8 backport #44)

## Requirements

- Hailo-8L M.2 module with HailoRT installed and the PCIe driver loaded (on Linux, the patched driver — see above)
- Python 3.10–3.13 with `mcp<2`, `python-dotenv`, `opencv-python`, `numpy`, and the `hailort` Python bindings; `hailo_ocr` additionally needs `pyctcdecode`
- The compiled `.hef` model binaries (not included in this repo — but free to download; see below)

Analysis scripts read their data root from `HAILO_PIPELINES_DATA`.

## Installation

These steps were verified end-to-end on Windows 11 with a Hailo-8L (HM21LB1C2KAE) and
HailoRT 4.24.0. Each subsection names a dead end that a reasonable reader walks into,
because the obvious path is wrong in three places.

### 1. The `.hef` models — download them, do not compile them

You do **not** need the Hailo Dataflow Compiler. Every model this project uses is a
precompiled, freely downloadable HEF in the public Hailo Model Zoo bucket — no account,
no login:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/<name>.hef
```

Required (the filenames already match what `server/hailo_runtime.py` expects):

```
scrfd_2.5g.hef
paddle_ocr_v5_mobile_detection.hef
paddle_ocr_v5_mobile_recognition.hef
tinyclip_vit_40m_32_text_19m_laion400m_image_encoder.hef
```

Optional (the runtime handles their absence): `real_esrgan_x2.hef`, `yolov5m_vehicles.hef`.
All six total ~342 MB.

Put them in one directory and point `HAILO_MODELS_DIR` at it.

> **Use the `hailo8l` path segment, not `hailo8`.** The bucket carries both, and they are
> different builds. A HAILO8 HEF is rejected by an 8L device at load time with an
> architecture-mismatch error. Verify any HEF before trusting it:
>
> ```
> hailortcli parse-hef <file>     # must print: Architecture HEF was compiled for: HAILO8L
> ```
>
> A nonexistent object in this bucket returns HTTP **403**, not 404 — so when probing for
> a model, test for a 200 rather than for the absence of a 404.

### 2. The `hailort` Python bindings — they are inside the installer you already have

`import hailo_platform` fails on a fresh HailoRT install, and `C:\Program Files\HailoRT`
contains no Python package, which makes it look like the bindings must be fetched from the
Hailo Developer Zone. They do not. On Windows the wheels ship **inside the HailoRT `.msi`**
and are simply not deployed to `Program Files`. Extract the installer without running it:

```
msiexec /a hailort_4.24.0_windows_installer.msi /qn TARGETDIR=C:\hailort-extract
```

and the wheels are at `HailoRT\python\`, one per interpreter, named like
`hailort-4.24.0-cp311-cp311-win_amd64.whl` (cp310, cp311, cp312, cp313 are present).
Pick the one matching your interpreter. (This is also why the supported Python range is
3.10–3.13: it is the set of wheels in the MSI.)

> **Install it with `--no-deps`.** The wheel declares a dependency on `netifaces`, which has
> no prebuilt wheel for current Pythons and tries to compile — failing with
> *"Microsoft Visual C++ 14.0 or greater is required"*. `netifaces` only enumerates network
> interfaces for **Ethernet-attached** Hailo devices; an M.2 (PCIe) module does not use it.
>
> ```
> pip install argcomplete contextlib2 future netaddr
> pip install --no-deps hailort-4.24.0-cp311-cp311-win_amd64.whl
> ```

The bindings are imported lazily inside `hailo_runtime.py`, so the MCP server starts and
`hailo_status()` answers even before this step — it is the honest probe for whether the
stack is complete.

### 3. Python dependencies

```
pip install "mcp<2" python-dotenv opencv-python numpy
pip install pyctcdecode        # only for hailo_ocr
```

> **Pin `mcp<2`.** `mcp` 2.0 removed `mcp.server.fastmcp`, which `server/server.py` imports;
> on 2.x the server fails at import time. 1.29 is known-good.
>
> `pyctcdecode` downgrades `numpy` to 1.x. The pipelines work on both; just do not be
> surprised by the version change.

### 4. Enable and verify

Windows (cmd):

```
set HAILO_MODELS_DIR=<your models directory>
set HAILO_VISION_ENABLED=1
python server/server.py
```

Linux / macOS:

```
export HAILO_MODELS_DIR=<your models directory>
export HAILO_VISION_ENABLED=1
python server/server.py
```

`hailo_status()` should report `enabled: true` and `hefs_missing: []`. Then run a real
inference — `hailo_embed` on any image returns a 512-element vector. The first call pays
the HEF load (several seconds); subsequent calls are fast.

### Windows vs. Linux

The kernel patch in `kernel-patch/` is for a **bare-metal Linux** host. On Windows, HailoRT's
own driver is stable and the patch does not apply.

Do not try to run this inside WSL2 on a Windows host: WSL2 has no PCIe passthrough, so the
NPU is unreachable from the Linux side. Use the Windows HailoRT path above.

## Device-busy activity file

HailoRT 4.24 on Windows has no busy counter and `hailortcli monitor` is unsupported there,
so nothing outside the process can tell how busy the 8L is. The process that runs inference
therefore publishes it: every device call (`pipe.infer` on each `InferVStreams` pipe via the
`timed_infer` proxy, and Whisper's `enc_cfg.run` / `dec_cfg.run`) is timed by
`server/accel_activity.py`, and a daemon thread rewrites a duty-cycle file every 500 ms. Both
the MCP server and the HTTP sidecar are covered, since both run inference through
`hailo_runtime.py`.

- **Where:** `$NVPAIR_ACCEL_ACTIVITY_DIR` if set; else `%ProgramData%\nvpair\accel-activity`
  on Windows; else `/run/nvpair/accel-activity`, falling back to
  `$XDG_RUNTIME_DIR/nvpair/accel-activity` when `/run` is not writable. File name `hailo.json`.
- **Schema 1** (UTF-8 JSON, no BOM, all numbers integers):

  ```json
  {"schema":1,"device":"hailo-8l","pid":1234,"started_ms":1790000000000,
   "updated_ms":1790000012345,"busy_ms":4321,"inflight":0}
  ```

  `busy_ms` is the cumulative wall time during which at least one device call was in flight
  (overlapping calls count once), including the in-flight portion up to `updated_ms`,
  measured on a monotonic clock; `started_ms`/`updated_ms` are epoch ms. A reader gets the
  duty cycle from two samples: `Δbusy_ms / Δupdated_ms`.
- **Lifecycle:** the writer starts on the first device call (a process that never inferred
  writes no file) and makes a final write with `inflight: 0` at interpreter exit. Writes are
  atomic (temp file + `os.replace`); a tick that collides with a reader holding the file open
  is skipped and the next one catches up. The inference path only updates two counters under
  a lock (about 2 µs per call); every file error is swallowed.
- **Disable:** `HAILO_ACTIVITY_DISABLE=1` turns it off entirely (no thread, no file).

## License

MIT

## HTTP sidecar (for the offload-harness)

`hailo-http.cmd` serves the same tools over loopback JSON-over-HTTP so a non-Python
caller (the offload-harness's `internal/hailoclient`) can use the NPU:

    GET  http://127.0.0.1:18813/health                 -> hailo_status()
    POST http://127.0.0.1:18813/v1/face_embed  {"image_path": "..."}  -> the tool's dict

The process exits on its own after `HAILO_SIDECAR_IDLE_SEC` (default 300 s) idle; the
harness starts it on demand, so nothing runs when AI features are not in use.
It refuses to bind anything but loopback — it is not an authenticated service.
