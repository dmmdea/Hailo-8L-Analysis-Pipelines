# Hailo-8L Production YouTube Visual Intelligence Layer (HMC v2)

**Reframed 2026-04-24 after external audit (GPT 5.5).** This is not an
"OCR project." It's the production-grade visual-intelligence side of
the YouTube analyst pipeline: Hailo extracts stable visual facts ONCE
per piece of media, YouTube analytics updates continuously, and the
learning layer joins them to surface which visual patterns help the
channel. OCR is one signal among several; logos, vehicle detection,
shot-type classification, and frame ranking matter equally.


> **Phase H — DONE 2026-04-28.** Content-addressed VisionCache shipped at
> ``openclaw_shared/cache/vision_cache.py``; ``extract_thumbnail_features``
> wired through it. Live measurement: 8010 ms cold → 7.6 ms warm
> (1060× speedup). See ``phase_H_done.marker`` and ``phase_H_metrics.json``.
> 
> **Path correction.** HEFs live at ``/home/hailo/models/`` (HailoRT apps-
> infrastructure deb default), not ``/mnt/ai/hailo/models/`` as referenced
> throughout this plan. ``hailo_runtime.DEFAULT_MODELS_DIR`` updated 2026-04-28.
> Other ``/mnt/ai/hailo/...`` references (calib/, datasets/, alls/, corpora/)
> still describe target locations for Phases A/C/D/E and may be relocated
> when those phases land.
The plan stacks. Phase B (off-the-shelf HEFs) can be wired immediately
with no DFC; Phase C (OCR recalibration) is high-upside but treat as
research until benchmarked; Phases D-G ship the production-value
features (model→brand inference, Shorts frame selection, thumbnail
validator, logo classifier) that close the brand-recall gap and
make the system actually useful for the user's channel.

## Mission upgrade — three caches + strategic repetition

Production architecture has three layers:

1. **Immutable Vision Cache** (Hailo-heavy). Per-asset visual facts
   keyed by `content_hash + model_version + pipeline_version`. OCR,
   faces, vehicles, CLIP embeddings, logo predictions, shot types,
   frame rankings. Recompute only when media changes or model/
   pipeline version bumps. Storage: SQLite at
   `/home/dmmdea/openclaw-output/hailo-vision-cache/cache.db` plus
   a Parquet sidecar for embeddings.
2. **Mutable Analytics Cache** (nightly). Views, watch-time, retention,
   traffic sources, comments, A/B results, title/thumbnail changes.
   Refreshed via the YouTube Data + Analytics MCPs.
3. **Learning Join Layer.** Joins (1) with (2) on `(channel, video_id,
   asset_revision)`. Pattern mining + personal-feedback recommendations
   live here. Output: ranked actionable suggestions with confidence
   intervals, NOT scoring-as-astrology.

Strategic repetition rule (also from audit):
- New or changed media → analyze fully.
- New Hailo model or pipeline version → invalidate affected cache rows.
- Regression sentinel set → re-run on every code change.
- Fresh YouTube metrics → ingest nightly.
- Static unchanged media with same pipeline → **do not re-run**.

## Audit corrections applied (2026-04-24)

| Audit finding | Action |
|---|---|
| `ocr(mode="quality")` still runs TTA + beam + LM despite FINAL_REPORT showing they regressed | **Shipped 2026-04-24.** Split into three explicit modes: `fast`, `production`, `research`. `production` is what the FINAL_REPORT recommends (Phase 3 multi-scale detection only). `research` keeps the experimental TTA+beam+LM stack for offline benchmarking. See Phase B0 below for the landed change. |
| Phase B vehicle detection code-shipped but no `phase_B_done.marker` / metrics / predictions artifact | Re-run the 90-thumb prediction with vehicle features active and write all three artifacts. Phase B is not "done" until checkpoints exist. |
| OCR recompilation framed as production-critical | Re-prioritized as research until benchmarked. Production value comes faster from model→brand lookup + logo classifier + frame ranking. |
| Logo classifier scoped only to text-detection crops | Also feed it: vehicle bboxes (grille/badge regions), saliency crops (whole-image OpenCV saliency), and the whole thumbnail. Five crop sources, not one. |
| DFC version | Update to **v3.33.1** per Model Zoo v2.18.0 release notes (not v3.33.1). Use Model Zoo v2.18.0 tag, not master. |
| Plan was OCR-centric | Reframed: production YouTube visual intelligence layer. OCR is one of many signals. |

## Strategic premises (verified by deep research 2026-04-24)

- **There is no off-the-shelf rec replacement on Hailo-8L.** Every
  alternative text-recognition HEF (PaddleOCR-server, ParSeq, TrOCR,
  ABINet, SVTR, CRNN) returns 403 from the Hailo Model Zoo bucket on
  both 8 and 8L paths.
- **The fix to softmax collapse is at COMPILE step, not at training.**
  The PP-OCRv5 mobile rec architecture is fine; what kills calibrated
  probabilities is Hailo's ICDAR2015-calibrated INT8 quantization on
  the 18,385-wide CTC head with no precision override.
- **The structural brand-recall gap is logo recognition, not OCR
  quality.** 12 of 37 brand FN are logo-only thumbnails (VW emblem,
  CHEVROLET bowtie, HONDA grille). No OCR HEF — calibrated or not —
  closes that. A custom logo classifier head does.
- **CLIP/ViT does not fit Hailo-8L** (Hailo-10H/15H only). Keep
  TinyCLIP exactly as deployed.
- **CUDA + RTX 5060 (Blackwell) is not on Hailo's supported GPU list**
  for DFC v3.33.1. Compile on CPU (`--use-cpu`) or migrate compile
  workload to Aorus if it has Ampere/Ada.
- **PP-OCRv5 mobile rec HEF was compiled for Hailo-15L by Hailo's team.**
  Our deployed HEF runs on 8L via backward-compat. Recompiling against
  hailo8l target with our own .alls gives us the right artifact for
  our silicon class.
- **Hailo zoo's LPRNet is Chinese-plate trained** (11-class output, not
  36-class Latin alphanumeric). Skipping it; license-plate recognition
  on auto-review thumbnails would need either a Latin LPRNet swap from
  a future zoo release or a custom train.

## Target end-state pipeline

A single thumbnail produces these signals through HailoRuntime, in
roughly this order:

| Stage | HEF | Source | Output |
|---|---|---|---|
| Optional: Real-ESRGAN x2 SR | `real_esrgan_x2.hef` | Hailo zoo (have) | 2× upscaled thumbnail |
| Multi-scale text detection | `paddle_ocr_v5_mobile_detection.hef` | Hailo zoo (have) | text bounding boxes |
| **Recalibrated** OCR rec | `paddle_ocr_v5_mobile_rec_a16w16.hef` | **CUSTOM (Phase C)** | per-box text + calibrated softmax |
| Vehicle detection | `yolov5m_vehicles.hef` | Hailo zoo (Phase B1, **shipped**) | vehicle bboxes (single class) |
| Face detection | `scrfd_2.5g.hef` | Hailo zoo (have) | face count + area |
| Image embed | `tinyclip...hef` | Hailo zoo (have) | 512-d embedding |
| **Logo classifier** | `auto_logos_resnet18.hef` | **CUSTOM (Phase D)** | top-K brand probs per crop |

LPRNet is intentionally absent — the available Hailo-8L HEF is Chinese-
plate trained (11-class output), not the 36-class Latin alphanumeric
needed for Spanish/US/EU plates.

End-state disk: ~225 MB total HEFs at `/mnt/ai/hailo/models/`. Trivial
vs 342 GB free.

---

## Phase A — Foundation (gated on USER action)

### A1. Hailo Developer Zone registration **[USER ACTION REQUIRED]**

1. Go to https://hailo.ai/developer-zone/.
2. Register a free account using `dmmdea@hotmail.com`.
3. Wait for manual approval (community reports 24-48 h queue).
4. After approval, navigate to **Software Downloads → AI Software Suite**.
5. Download **Hailo Dataflow Compiler v3.33.1** for Linux x86_64
   (`.whl`). **Do NOT download v5.x** — that targets Hailo-10H/15,
   not 8/8L.
6. Drop the wheel at
   `/home/dmmdea/Downloads/hailo_dataflow_compiler-3.33.0-*-linux_x86_64.whl`.
7. Tell the assistant when it's there. Subsequent steps automate.

### A2. DFC install (assistant runs)

```bash
mkdir -p /home/dmmdea/hailo-dfc && cd /home/dmmdea/hailo-dfc
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install /home/dmmdea/Downloads/hailo_dataflow_compiler-3.33.0-*-linux_x86_64.whl
hailo --version
```

If `hailo --version` exits clean we're done. If GPU init complains,
pass `--use-cpu` to subsequent `hailo optimize` calls.

### A3. Calibration corpus assembly (assistant runs)

Build three calibration sets:

1. **OCR rec calibration** — 1,024 in-domain text-region crops:
   - Take the 90 ground-truth thumbnails through current detection
   - Save 5-10 highest-confidence text crops per thumbnail
   - Pad/normalize to 48×320 NHWC float32
   - Mix in 200-400 ICDAR2015 crops to keep general English/Latin coverage
   - Output: `/mnt/ai/hailo/calib/ocr_rec_calib_1024.npy`

2. **Logo classifier calibration** — 1,024 logo crops at 224×224.
   Built from the synthetic logo corpus once Phase D-1 finishes.

3. **Vehicle detector calibration** — 1,024 thumbnail crops at 640×640.
   Reuse the existing 210-thumb pool; resize and normalize.

### A4. Dataset acquisition (assistant runs)

```bash
mkdir -p /mnt/ai/hailo/datasets && cd /mnt/ai/hailo/datasets
git clone https://github.com/filippofilip95/car-logos-dataset.git
git clone https://github.com/GeneralBlockchain/vehicle-logos-dataset.git
git clone --depth 1 https://github.com/google/fonts.git google-fonts
```

Disk hit: ~2 GB (Google Fonts is the bulk).

---

## Phase B0 — OCR mode-semantics fix **[CODE LANDED 2026-04-24]**

`hailo_runtime.py:ocr(mode=...)` previously accepted only `fast` and
`quality`. `quality` entangled Phase 3 multi-scale detection AND Phase
4 TTA AND Phase 5 beam AND Phase 6 LM rescoring — the FINAL_REPORT
showed the last three regressed or were no-ops on the current rec HEF,
yet production callers got the experimental stack by default.

Fix shipped:

1. `VALID_OCR_MODES = ("fast", "production", "research")` is the new
   contract. Mode validation runs BEFORE `ensure_initialized()` so
   bad mode strings raise ValueError without touching the device.
2. `production` activates ONLY Phase 3 multi-scale detection
   (SR + multi-scale + SR-sourced crops + greedy CTC). No TTA, no beam,
   no LM. The four boolean knobs (`use_sr`, `use_multi_scale`,
   `use_tta`, `use_beam`) are now decoupled inside `ocr()`.
3. `research` is byte-equivalent to the old `quality` semantics —
   production stack PLUS per-crop TTA + beam-search CTC with optional
   KenLM rescoring (env-gated). Kept for offline ablation only.
4. `fast` unchanged.
5. `extract_thumbnail_features._merge_hailo_features` now reads +
   validates `HAILO_OCR_MODE` UP-FRONT (outside the per-feature try/
   except blocks) so misconfigured modes fail loud while device errors
   still stay isolated. Default flipped to `production`.
6. `run_predictions.py --mode` choices = `(fast, production, research)`,
   default `production`. Old `--mode quality` is gone (clean break).
7. `grid_search_phase6.py` switched to `--mode research` — Phase 6 is
   the beam/LM ablation track, which lives under research.
8. FINAL_REPORT.md production stanza references `HAILO_OCR_MODE=production`.
9. Tests pinned at `hailo-vision/tests/test_ocr_modes.py` (4 tests) and
   `_shared/tests/test_thumbnail_ocr_mode_env.py` (3 tests). 7/7 green.

Verification status:
- ✓ 90-thumb prediction in `production` mode against the live device
  reproduced Phase 3 baseline to 3 decimals (year F1=0.930,
  brand F1=0.575, model F1=0.654). See B3 below + `phase_B_done.marker`.
- ⏸ 90-thumb prediction in `research` mode reproducing today's old
  `quality` behavior. Deferred — research is offline-only, not on the
  production path; run on demand if the next rec HEF needs it.

## Phase B — Off-the-shelf wins (NO DFC needed; runs immediately)

This is the fastest path to value. Adds brand-new signals the current
pipeline lacks, no calibration / no training / no DFC required.

### B1. Vehicle detection — `yolov5m_vehicles.hef` (37.9 MB)  **[CODE SHIPPED 2026-04-24]**

Implementation surprises vs the original spec:
- Input is **1920×1080**, not 640×640. (Network internally letterboxes
  to 640.) For our 1280×720 thumbnails, direct cv2.resize to 1920×1080
  preserves 16:9 aspect ratio with no quality loss.
- Output is **single-class "vehicle"**, not multi-class car/truck/bus.
  The `hailo_vehicle_classes` field from the original spec is dropped.
- Built-in on-device NMS — no manual YOLO-grid decode needed. Output
  arrives as a list of `(1, N_detections, 5)` float32 arrays where
  each row is `(y1, x1, y2, x2, score)` in normalized [0, 1] coords.
- Net inference cost: ~0.07-0.09 s per thumbnail (not measurable on
  the 90-thumb baseline given total pass time is dominated by SR).

Code in `hailo_runtime.py`:
- `HEF_VEHICLE_DETECT` constant added to `OPTIONAL_HEFS`.
- New `VehicleBox` dataclass mirrors `FaceBox`.
- `HailoRuntime.vehicle_detect(image_path, score_threshold=0.3)`
  returns `list[VehicleBox]`.
- `extract_thumbnail_features` (in `openclaw_shared/features/thumbnail.py`)
  now emits four new feature keys when the backend has
  `vehicle_detect`: `hailo_vehicle_count`, `hailo_has_vehicle`,
  `hailo_largest_vehicle_area_ratio`, `hailo_vehicle_score_max`.
  Failure isolation: any `vehicle_detect` exception is swallowed by
  the same try/except pattern used for face/OCR/embed.

Smoke-test results on 5 anchor thumbs (2026-04-24):

| thumbnail | content | vehicles found |
|---|---|---|
| `Aox2tWDpIs4` | Tesla + Cadillac front shots | 2 |
| `BUk2ybi9_AM` | Forester + Palisade SUVs | 2 |
| `DcpVp-S4Vqo` | shorts / cockpit interior | 0 (correct) |
| `KlGmOL3p2vA` | Lamborghini + Phantom + Bentley + bg car | 4 |
| `Ao-QVHnFlmo` | 3 EVs (IONIQ 9, Subaru, ID Buzz) | 3 |

### B2. License plate rec — `lprnet.hef` — **DEFERRED (alphabet mismatch)**

Initial plan called for wiring this in. After downloading and probing
the HEF, the output shape is `1×19×11` UINT8 — only **11 output classes**.
That alphabet size is consistent with **Chinese license plates**
(province char + sub-alphabet of common chars) rather than the standard
36-class Latin alphanumeric (10 digits + 26 letters) needed for
Spanish / US / EU plates that show up on YouTube auto-review thumbnails.

Decision: skip B2. Costs disk + load time with no expected gain on
non-Chinese plates. Re-evaluate if a Latin LPRNet variant appears in a
future Hailo Model Zoo release.

The HEF stays at `/mnt/ai/hailo/models/lprnet.hef` (10.5 MB) for
reproducibility but is not in `OPTIONAL_HEFS` and won't load.

### B3. Re-benchmark + checkpoint  **[DONE 2026-04-24]**

Run completed on 90 thumbs in 690.2 s (0.13 thumb/s) — within 4 s of the
Phase 3 baseline (vehicle-detect overhead is in the noise). Production
mode reproduces the FINAL_REPORT P3 row to 3 decimals, empirically
confirming the B0 split was non-regressing:

  brand F1=0.575  model F1=0.654  year F1=0.930
  text strict CER=0.5619  WER=0.9405

Vehicle features populate on all 90 rows (4/4 keys, 0 missing). 150
vehicles total, mean 1.67/thumb, 15 zero-vehicle thumbs (Shorts +
interior). 5/5 anchor smoke-counts match the B1 spec.

Artifacts in `/home/dmmdea/openclaw-output/hailo-ocr-quality-plan/`:
- `predictions_phase_B.jsonl` (90 rows, 0 errors)
- `phase_B_metrics.json`
- `phase_B_done.marker` (full report)
- `phase_B_run.log`

---

## Phase B-bis — model→brand lookup  **[DONE 2026-04-24]**

Shipped as planned: `MODEL_TO_BRAND` dict in `openclaw_shared/features/
title.py` (~120 mappings, conservative — BYD short-words `tang/song/
han/yuan/destroyer` and Mitsubishi `eclipse cross` omitted to avoid
common-word collisions). `extract_entities` does a two-pass union with
provenance tracking via new `brand_sources` dict; `brands` stays a
plain sorted list (back-compat with benchmark scoring). Inference
NEVER overrides explicit OCR brand evidence.

Benchmark (replay over predictions_phase_B.jsonl — OCR text
deterministic):

  Brand F1:    0.575 → 0.667  (+0.092)
  Brand P:     1.000 → 0.892
  Brand R:     0.403 → 0.532  (+0.129)
  Model F1:    0.654 → 0.654  (Δ=0, as required)
  Year F1:     0.930 → 0.930  (Δ=0, as required)
  Strict CER:  0.5619 → 0.5619  (Δ=0)

8 new brand TPs from clean inferences. 4 new brand FPs inspected by
video_id — all trace to upstream OCR/model-extraction noise (Corolla
Cross OCR'd as TCROSS triggering VW; "AUTO" OCR'd as "ATO" triggering
BYD Atto; one GT-labeling judgment call on 4Runner→Toyota). All 4 FPs
co-occur with TPs on the same row, so downstream consumers gating on
`brand_sources` can still recover the correct brand.

Tests: 11 in `_shared/tests/test_model_to_brand.py` (mapping-table
integrity + 20 pinned mappings + the explicit-OCR-evidence-wins
invariant + back-compat). 64/64 across the suite.

Artifacts in `/home/dmmdea/openclaw-output/hailo-ocr-quality-plan/`:
- `predictions_phase_B_bis.jsonl` (90 rows, replayed)
- `phase_B_bis_metrics.json`
- `phase_B_bis_done.marker` (full report incl. FP inspection)

## Phase C — OCR rec recalibration (research-track until benchmarked)

This is the single most important phase. If C succeeds, Phases 4-6 of
the prior plan (TTA, beam, LM) become net-positive instead of no-ops,
AND we keep the existing rec architecture (no fine-tune compute spend).

### C1. Acquire stock PP-OCRv5 mobile rec ONNX

Per the research: the Hailo Model Zoo HEF is built from
`PP-OCRv5_mobile_rec_48x320_sim.onnx` (2025-08-10). Either:
- Download from Hailo's published S3 bucket (URL is in
  `cfg/networks/paddle_ocr_v5_mobile_recognition.yaml` of the Hailo
  Model Zoo repo), OR
- Re-export from PaddleOCR v3.5.0 release artifacts:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && git checkout v3.5.0
python tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=<stock_ckpt> Global.save_inference_dir=./infer_v5_mobile_rec
paddlex --paddle2onnx --paddle_model_dir ./infer_v5_mobile_rec \
    --onnx_model_dir ./onnx_v5_mobile_rec --opset_version 14
python -m onnxsim ./onnx_v5_mobile_rec/inference.onnx \
    ./PP-OCRv5_mobile_rec_48x320_sim.onnx --overwrite-input-shape 1,3,48,320
```

### C2. Bake temperature scaling INTO the ONNX

The single most reliable anti-collapse trick. Insert a constant divide
of T=2.0 into the ONNX graph immediately before the final softmax /
argmax via `onnx.helper.make_node('Div', ...)` and rewire the softmax
input. The exact node-rewire depends on the source ONNX graph; the
research agent flagged that `tools/export_model.py` produces a graph
where the Softmax node is clearly identifiable as the last op before
the output.

### C3. Author the recalibration .alls

Custom `.alls` with the documented anti-collapse compile-time tweaks:

```
model_optimization_flavor(optimization_level=2, compression_level=0)
pre_quantization_optimization(activation_clipping(layers={*fc_out*, *logits*, *softmax*}, mode=percentile, clipping_values=[0.001, 99.9]))
quantization_param({*fc_out*, *logits*}, precision_mode=a16_w16)
post_quantization_optimization(adaquant, policy=enabled)
post_quantization_optimization(finetune, lr=0.00005, epochs=8, dataset_size=4096, loss_layer_names=[<final 4 conv/fc layer names from parser output>])
```

### C4. Compile

```bash
cd /home/dmmdea/hailo-dfc && source .venv/bin/activate
hailo parser onnx PP-OCRv5_mobile_rec_48x320_T2_sim.onnx --hw-arch hailo8l --har-path rec_v5.har
hailo optimize rec_v5.har \
    --hw-arch hailo8l --use-cpu \
    --calib-set-path /mnt/ai/hailo/calib/ocr_rec_calib_1024.npy \
    --model-script /mnt/ai/hailo/alls/rec_v5_a16w16.alls \
    --output-har-path rec_v5_opt.har
hailo compiler rec_v5_opt.har --hw-arch hailo8l \
    --output-dir /mnt/ai/hailo/models/
```

### C5. Softmax sanity check

Feed 20 thumbnails through the new HEF, log top-1 probability per
timestep, average across all timesteps. Pass criterion: `top1_avg <
0.85` → softmax is calibrated enough for beam+LM.

If `top1_avg` stays >0.95, the recalibrate didn't take. Iterate the
.alls (more aggressive activation_clipping, optimization_level=4,
expand the a16_w16 layer set further back into the network).

### C6. Re-run prior plan's Phases 4 + 5 + 6 against new rec HEF

These were no-ops on the broken HEF. With a calibrated softmax, they
should each contribute monotonically. Record `phase_C_metrics.json`.

### C7. Production cutover

If brand/model F1 jumps materially, swap the production runtime to use
the new rec HEF by default. Keep the old as
`paddle_ocr_v5_mobile_recognition.hef.bak`.

---

## Phase D — Auto-brand logo classifier (custom train + compile)

Closes the structural brand-recall gap. Output: per-crop top-3 brand
probabilities. Pipes into entity extraction so a thumbnail showing
only a Hyundai logo (no text) still yields `hyundai` in the brand list.

**Crop sources (per audit correction):** the classifier runs on FIVE
crop sources, not just text detection boxes. Brand logos appear in
multiple regions:

1. **Vehicle bboxes** (from yolov5m_vehicles) — the most common
   logo location is the grille/badge of a detected vehicle. Crop
   the upper-center 30% of each vehicle bbox.
2. **Center-of-vehicle saliency crops** — for each vehicle bbox,
   compute OpenCV saliency map (`cv2.saliency.StaticSaliencySpectralResidual`)
   and crop the top saliency region.
3. **Text-detection bboxes** — when brand name appears in text and
   logo overlays the text region.
4. **Whole thumbnail at 224×224** — catches logo-only thumbnails
   where neither vehicle nor text detection fires (e.g., close-up
   of a steering wheel boss with VW logo).
5. **Saliency crops on the whole thumbnail** — top-3 saliency regions.

Aggregate top-K brand votes across all 5 crop sources per thumbnail.

### D1. Synthetic logo corpus

For each of 35 brands (Toyota, Honda, Hyundai, Kia, Subaru, Mazda,
Nissan, Mitsubishi, Suzuki, Lexus, VW, BMW, Mercedes, Audi, Porsche,
Mini, Opel, Seat, Renault, Peugeot, Citroën, Fiat, Alfa Romeo, Ferrari,
Lamborghini, Maserati, Ford, Chevrolet, Cadillac, GMC, RAM, Jeep,
Tesla, Rivian, BYD):

- Source vector logo from filippofilip95 (387 brands, MIT licensed)
- Render at 5 sizes (96, 160, 256, 384, 512 px) with 5 rotations
  (-15° to +15°)
- Composite onto 100 random backgrounds drawn from the existing
  210-thumb pool
- Apply: random JPEG compression [40, 95], random brightness ±20%,
  motion blur σ ∈ [0, 1.5], perspective warp ±8°
- Target: 1,000 imgs/brand × 35 brands = 35,000 training images
- Plus 5,000 validation images (held-out background pool)
- Plus 1,000 calibration set for Hailo DFC

Output dir:
`/mnt/ai/hailo/datasets/auto_logos_synth/{train,val,calib}/<brand>/...`

### D2. Train ResNet-18 head

PyTorch fine-tune from ImageNet-pretrained ResNet-18, replace final FC
with 35 output classes. Standard recipe:

- AdamW, lr=1e-4, weight_decay=1e-4
- 50 epochs, batch 64, AMP, on RTX 5060
- Augmentation pipeline: RandomResizedCrop(224), HorizontalFlip,
  ColorJitter(0.2,0.2,0.2,0.05), Normalize ImageNet stats
- Track val top-1 accuracy; stop when it plateaus

### D3. Export ONNX + compile

PyTorch script: load ResNet-18 with 35 classes, set inference mode via
`torch.no_grad()` + `model.train(False)`, `torch.onnx.export` to opset
14, then `onnxsim`. Bake T=1.5 temperature for calibrated output (same
Div node trick as Phase C). Then standard hailo parser → optimize →
compile path.

### D4. Wire into HailoRuntime

Add `classify_logo(crop) -> {brand: prob}` method. Call it on every
text-detection box in quality mode (logos often live in the same
regions as text on auto-review thumbnails). Aggregate top-K brand
votes across all boxes; emit `hailo_logo_brands` field with `[{brand,
prob, source}]`.

### D5. Benchmark

Run vs ground truth. Expected impact: HYUNDAI / VW / CHEVROLET miss
rate should drop substantially on logo-only thumbnails. Year/model
F1s unchanged (this is a brand-only signal).

Write `phase_D_done.marker` + `phase_D_metrics.json`.

---

## Phase E — Synthetic OCR fine-tune (only if C alone is insufficient)

**Skip this phase if Phase C alone produced acceptable brand/model F1.**

If after Phase C the stylized banner cases still garble (e.g., the
`8nP7NrrmTxU` ESTO ES ENTRY LEVEL anchor stays as `CESTOESEHIRYLEVEP`),
then a fine-tune on stylized synth is the next step.

### E1. Synthetic OCR corpus

Build a 300k-crop synthetic corpus matching YouTube-banner aesthetic:

- Word source: Spanish Wikipedia (already at
  `/mnt/ai/hailo/corpora/eswiki/eswiki_upper_noaccent.txt`)
- Add automotive vocab + clickbait tokens (`INCREÍBLE`, `LO MEJOR`,
  model names from CAR_MODELS, year tokens)
- 12 banner fonts from Google Fonts (Bebas Neue, Anton, Oswald Heavy,
  Montserrat Black, Archivo Black, Bangers, Bungee, Fjalla One,
  Impact, Luckiest Guy, Roboto Black, Teko)
- TRDG generates base crops (`pip install trdg`)
- Pillow post-processor adds: stroke 2-6 px, drop shadow 0-8 px blur,
  gradient fill, perspective warp ±8°, JPEG compression 40-85
- 250k uppercase Spanish + 50k uppercase English (the rec model's
  charset already covers both). Mix 80% synth + 20% real (~5,000
  hand-cropped from the 210-thumb pool with the existing OCR's output
  as draft labels, manually corrected).

Output: `/mnt/ai/hailo/datasets/banner_ocr_synth/{train,val}/...` plus
`train_list.txt` / `val_list.txt` in PaddleOCR's `image_path\tlabel`
format.

### E2. PaddleOCR fine-tune

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && git checkout v3.5.0
pip install -e .
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=<stock_ckpt> \
       Global.save_model_dir=./output/v5_mobile_rec_banner \
       Train.dataset.label_file_list=[/mnt/ai/hailo/datasets/banner_ocr_synth/train_list.txt] \
       Eval.dataset.label_file_list=[/mnt/ai/hailo/datasets/banner_ocr_synth/val_list.txt] \
       Optimizer.lr.learning_rate=0.0005 \
       Train.loader.batch_size_per_card=128 \
       Global.epoch_num=30
```

The research agent flagged: PaddlePaddle GPU support for RTX 50-series
is fresh as of Paddle 3.2.0; needs `paddlepaddle-gpu==3.1.1+` built
against CUDA 12.6/12.8. Smoke-test the install before committing to a
long fine-tune.

### E3. Export + recompile

Same Phase C2-C5 path with the fine-tuned checkpoint. New artifact:
`paddle_ocr_v5_mobile_rec_banner_a16w16.hef`.

### E4. Benchmark

Expect material lift on stylized banner thumbnails. Write
`phase_E_done.marker` + `phase_E_metrics.json`.

---

## Phase F — Detection retraining (optional, almost certainly skip)

Skip unless current detection misses entire banner regions on stylized
thumbnails. The benchmark already shows detection is fine.

## Phase H — Content-addressed Vision Cache (audit recommendation)

Per the three-cache architecture: every Hailo-derived feature for an
asset is keyed by `(content_hash, model_version, pipeline_version)`.
Same media, same models = read from cache. Different media or model
bump = recompute.

Implementation:

1. New module `openclaw_shared/cache/vision_cache.py`. SQLite table
   `vision_facts(asset_path TEXT, content_hash TEXT, model_version
   TEXT, pipeline_version TEXT, payload TEXT JSON, computed_at TIMESTAMP,
   PRIMARY KEY(content_hash, model_version, pipeline_version))`.
   Embeddings (CLIP 512-d float arrays) go to a sidecar Parquet file
   keyed by content_hash to keep the SQLite row size manageable.
2. `vision_cache.get_or_compute(asset_path, runtime, mode)` —
   compute SHA-256 of file bytes, look up; on miss, run the pipeline,
   serialize the feature dict to JSON, insert.
3. `model_version` fingerprint = SHA-256 of all loaded HEFs' bytes.
4. `pipeline_version` fingerprint = git rev-parse HEAD of the
   hailo-vision MCP repo, or fallback to a pinned semver string.
5. `extract_thumbnail_features` becomes a thin wrapper that reads
   from the cache.
6. Eviction: none for now — vision facts are ~10 KB per thumb in
   JSON form, so 10k assets = 100 MB. Negligible.

This unblocks repeated analysis without re-running Hailo, AND makes
the system honest about reuse: any feature snapshot has a clear
provenance fingerprint.

## Phase I — Shorts frame selector (production-useful capability)

Audit insight: YouTube Shorts use the auto-selected representative
frame from the video, NOT a creator-uploaded thumbnail. So our
"thumbnail analysis" pipeline doesn't help Shorts unless we can
recommend WHICH frame YouTube should pick (or which timestamp the
creator should re-upload from).

Implementation:

1. New tool `hailo_select_short_frame(video_path, top_k=5)`:
   - Decode the video at 1 fps via ffmpeg
   - Run each frame through the production-mode HailoRuntime
     (vehicle detect + face detect + multi-scale OCR detection +
     CLIP embed)
   - Score each frame by a weighted combination: vehicle area
     ratio, face presence, text-region count, CLIP similarity to
     the channel's recent-good-thumbnails centroid
   - Return top-K timestamps with feature breakdown

2. CLIP centroid for the channel: compute once on the channel's
   recent top-decile-performing videos' thumbnails. Update nightly
   from the analytics cache.

3. Integrates with the YouTube analyst pipeline at `youtube-data` MCP
   layer.

## Phase J — Long-form thumbnail candidate validator

Audit insight: before publishing, run a candidate thumbnail through
multi-signal validation. NOT scoring-as-astrology — concrete checks
the user can act on.

Checks:
- Face presence + area (matched against channel's outlier thumbs)
- Vehicle presence + dominant vehicle class
- Text-region count and total text area ratio
- Contrast measure (`hailo_runtime.embed` then PCA variance)
- Brightness mean (the well-known Danmar 82/255 vs US peers 128/255 finding)
- Logo presence (Phase D)
- Saliency-map "crowdedness" — top-3 saliency regions IoU; if all
  three overlap heavily, layout is too cluttered

Output: per-thumbnail report card with each check's value and a
plain-language verdict ("text area at 18% — within Danmar's
high-performer band of 12-22%; brightness at 78 — below your top
performers' 110-135 band; consider boosting contrast"). No 0-100
score.

## Phase K — Benchmark expansion to 300-500 labels

Audit recommends expanding the 90-label benchmark before training the
logo classifier. Reasons: the per-class brand sample size on the
current 90 is < 10 for most brands; classifier validation against
that sparse a set is statistically unreliable.

Plan:
1. Stratify across the channel's full 4-market peer set (US Hispanic,
   Spain, Mexico, Colombia)
2. Sample 50-70 thumbs per peer channel × 6 peers + 30 Danmar = 330-450
3. Use the model→brand lookup (Phase B-bis) to auto-pre-fill brand
   labels, then manually verify
4. Use Claude (this assistant in a labeling session) for first-pass
   labels, hand-verify the close calls
5. Same JSONL schema as `ground_truth.jsonl`

---

## Phase G — Integration + shipping

### G1. Final fused pipeline

Decide production config based on Phase B/C/D/E results:

- Always-on: B (vehicle + plate), C (recalibrated rec), D (logo classifier)
- Conditional: E (banner-fine-tuned rec) if it beats C on the benchmark
  by ≥0.05 F1 on brand or model

Update `HailoRuntime` to load all required HEFs at boot, run them
through the same `VDevice` ROUND_ROBIN scheduler.

### G2. Final benchmark + before/after report

Generate `HMC_FINAL_REPORT.md` with:

- Brand / model / year / keyword F1 trajectory across all phases
- Per-thumbnail spotchecks on the two anchor cases
- Wall-clock cost per thumbnail in fast vs quality vs fused mode
- Recommended production env vars

### G3. Skill packaging (per `feedback_skill_shipping_protocol.md`)

The custom HEF compile + dataset assembly pipeline becomes a skill at
`~/AI Ecosystem/OpenClaw/skills/Hailo-Maximum-Capability-Setup/`,
shipped per protocol:

1. Drive upload to `_current/` folder under user's skills directory
2. Verify Drive listing
3. Public GitHub repo `dmmdea/openclaw-hailo-maximum-capability`
4. Memory update (this file's project memory)

### G4. Regression guard

Add `tests/test_hailo_pipeline_regression.py` to the hailo-vision MCP:
runs 10 thumbnails through the fused pipeline, asserts:
- ≥3 brand hits
- ≥4 model hits
- ≥3 year hits
- Logo classifier confidence on the Hyundai/VW/Chevy logo thumbnails > 0.5
- Vehicle detector finds ≥1 vehicle on every non-shorts thumbnail

Fails CI / manual run if regression sneaks in.

---

## Critical path (revised after audit)

Production build order — execute in this sequence:

1. **B0** — fix OCR mode semantics (`fast` / `production` / `research`)
2. **B3** — finish Phase B vehicle-detection benchmark + checkpoint
3. **B-bis** — model→brand lookup table (cheap, immediate brand-recall lift)
4. **H** — content-addressed Vision Cache infrastructure
5. **I** — Shorts frame selector
6. **J** — long-form thumbnail candidate validator
7. **K** — expand benchmark from 90 → 300-500 labels
8. **D** — auto-brand logo classifier (training + DFC compile, gated on A1)
9. **A1 → A2 → C** — DFC registration + install + OCR recalibration (research track until benchmark proves a win)
10. **E** — synthetic OCR fine-tune (only if C alone insufficient)
11. **G** — production cutover + final report + skill shipping

Cheap-wins (B0, B3, B-bis, H, I, J, K) need no DFC and can ship now.
Heavy-cost branches (D for compile, C for retrain+compile) are gated
on the user's Hailo Developer Zone registration.

## DFC version (audit correction)

Hailo Dataflow Compiler **v3.33.1** ships with Model Zoo v2.18.0
(NOT v3.33.1 as the earlier draft of this plan said). Use the v2.18.0
zoo tag for HEFs and the v3.33.1 wheel for compiles. Confirmed in
the model-zoo release notes.

## What success looks like

- Production OCR brand F1 ≥ 0.75 (vs 0.575 baseline)
- Logo classifier closes ≥80% of the logo-only brand FN gap (so ~10
  of 12 anchor logo failures recover)
- Vehicle count + plate signals add new packaging features the
  matched-pair stats can use
- CER under 0.30 strict on the 78-text-bearing thumbnails
- Fused pipeline inference cost in same order of magnitude as today
  (the additional HEFs add small inference time vs the SR pass which
  dominates)

## When to ask Daniel

- Before A1: confirm registration email is `dmmdea@hotmail.com`
- After A1 approval: Daniel pastes the wheel into Downloads, signals ready
- After C5 entropy probe: "softmax now sits at top-1 mean X.XX, do we
  ship that as production rec or push for a more aggressive .alls?"
- Before E: "Phase C delivered Y on brand F1; is it worth the
  fine-tune for the additional banner cases or are we good?"

---

**Status flag for the next session:** if the assistant resumes this
work, read this file plus `FINAL_REPORT.md` first, then check
`/home/dmmdea/openclaw-output/hailo-ocr-quality-plan/phase_*_done.marker`
to see what's already shipped.
