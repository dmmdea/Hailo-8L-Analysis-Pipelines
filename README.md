# Hailo-8L Analysis Pipelines

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Hailo-8L](https://img.shields.io/badge/NPU-Hailo--8L_M.2-0091EA) ![MCP Server](https://img.shields.io/badge/MCP-server-8A2BE2) ![Python](https://img.shields.io/badge/Python-3.12-3776AB) ![Edge AI](https://img.shields.io/badge/inference-100%25_on--device-orange)

Production vision pipelines running entirely on a **Hailo-8L M.2 NPU module** (HM21LB1C2KAE): an MCP server that exposes the accelerator to any MCP-capable agent, a vendored feature library with Spanish-aware OCR correction, a measured multi-phase OCR quality program, and the Linux kernel driver patch that keeps the whole thing alive on modern kernels.

Everything here ran as a working, productive pipeline for thumbnail and image analysis at scale. No cloud inference anywhere: every model executes on the 8L.

## MCP server

`server/server.py` exposes 4 tools over MCP:

| Tool | Model | Measured on the 8L |
|---|---|---|
| `hailo_face_detect(image_path)` | SCrFD-2.5G | ~311 FPS |
| `hailo_ocr(image_path)` | PaddleOCR v5 (det + rec) | ~4.59 FPS (detection-bound) |
| `hailo_embed(image_path)` | TinyCLIP image encoder | 512-d vector |
| `hailo_status()` | n/a | device + runtime state, safe to call with the driver down |

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

- Hailo-8L M.2 module with HailoRT and the (patched) PCIe driver
- Python 3.12, `mcp` (FastMCP), `dotenv`; OCR pipelines additionally use PaddleOCR-compiled HEF models and `pyctcdecode`
- Compiled `.hef` model binaries for SCrFD-2.5G, PaddleOCR v5, and TinyCLIP (not included; compile with the Hailo Dataflow Compiler for the 8L)

Analysis scripts read their data root from `HAILO_PIPELINES_DATA`.

## License

MIT
