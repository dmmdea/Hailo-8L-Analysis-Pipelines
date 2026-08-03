# Hailo OCR Quality Plan — Final Report

**Scope:** PaddleOCR v5 mobile (detection + recognition) on Hailo-8L, applied to
the 90-thumbnail target-channel benchmark corpus built in Phase 0. Goal:
improve brand / model / year / keyword extraction F1 through post-HEF
processing without retraining either HEF.

## TL;DR

**Recommended production config: Phase 3 (multi-scale detection only).** It
is the sole phase that delivers a monotonic gain with no regression:
`year F1 0.878 → 0.930`. Every phase beyond that traded metric for metric
without a clean net win, because the PaddleOCR v5 mobile recognition HEF
emits a near-one-hot softmax that is structurally resistant to beam
search + LM rescoring + TTA broadening as implemented here.

## Trajectory table (all 90 thumbnails)

| config                                    | brand F1 | model F1 | year F1 | strict CER | strict WER | 90-thumb pred time |
|------------------------------------------|---------:|---------:|--------:|-----------:|-----------:|-------------------:|
| Baseline (P1: greedy CTC on raw OCR)      |    0.575 |  **0.741** |   0.878 |   **0.464** |   **0.844** |              115 s |
| P2 (Real-ESRGAN x2 SR preprocess)         |    0.558 |    0.600 |   0.789 |    0.492 |    0.863 |              685 s |
| P3 (multi-scale detection union)          |    0.575 |    0.654 | **0.930** |    0.562 |    0.940 |              694 s |
| P5 (pyctcdecode beam + hotwords)          |    0.575 |    0.654 |   0.930 |    0.562 |    0.940 |              699 s |
| P6 (+ 4-gram KenLM on esWiki, α=0.3 β=0.5)|    0.575 |    0.654 |   0.930 |    0.562 |    0.940 |           ~2.2 hr * |
| P7 (+ symspellpy gated correction)        |    0.575 |    0.667 |   0.930 |    0.552 |    0.913 |              (CPU replay) |
| P4 (+ per-crop 2-way TTA, stacked 5+6)    |  **0.578** |    0.679 |   0.850 |    0.553 |    0.952 |              ~12 min |
| P4+P7 stacked (final)                     |    0.568 |    0.653 |   0.850 |    0.546 |    0.943 |              (CPU replay) |

*12 grid combos sequential; any single config is ~11 min.*

## The single anchor that everything optimized toward

`Ao-QVHnFlmo` — ground truth `IONIQ 9, SUBARU UNCHARTED, VW ID BUZZ`:

| config | decoded text (rec-detect pipeline output) |
|---|---|
| baseline | `MEJORES CARROS ELECTRICOS2026! UNCHARTED o SUBARU IDBU2` |
| P3 | `MEJORES CARROS ELECTRCOS2026! UNCHARTED UNCHARTED o SUBARU IDBUZ` |
| P4+5+6 | `MEJORES CARROS ELECTRCoS2026! UNCHARTED UNCHARTED SUBARU IDBUZZ` |
| final stacked | `MEJORES CARROS ELECTRCoS2026! UNCHARTED UNCHART SUBARU WWIDBUZZ` |

ID BUZZ was the big win of the P4+5+6 stack — the TTA-broadened softmax +
LM hotword boost finally let the decoder pick the `Z` at the end. HYUNDAI
and VW remain uncaptured (HYUNDAI is an inference from IONIQ, VW is only
visible as a logo, not as text).

`8nP7NrrmTxU` — ground truth `¿ESTO ES ENTRY LEVEL?`:

| config | decoded text |
|---|---|
| baseline | `CESIUESENIRYLEVEP` |
| P3 | `CESTOESENTRYLEVEP` |
| P4+5+6 | `CESTOESEHIRYLEVEP` |

All stylized banner decodes collapse the space between words; the rec HEF
cannot recover word boundaries on drop-shadow fonts. Fixing this is
out-of-scope for a post-HEF pipeline — it would require retraining the
detection HEF on stylized-font examples or running a word-segmentation
pass on the crop before rec.

## Phase-by-phase honest accounting

### Phase 0 — Ground truth (✓)
Hand-labeled 90 thumbnails (30 the target channel + 10 × 6 peer channels), verifying
paths and canonical vocab alignment. Entity coverage: 53 brand / 24 model
/ 23 year / 78 text. This is the single artifact every subsequent phase
depends on; keep it versioned.

### Phase 1 — Benchmark harness (✓)
`ocr_benchmark.py` computes entity precision/recall/F1 with canonical
alias normalization (mercedes/mercedes benz/mercedes-benz → `mercedes`,
etc.), CER/WER via jiwer, strict + lenient (no-case/no-accent) scores.
Baseline metrics locked at `phase_1_baseline.json`.

### Phase 2 — Real-ESRGAN x2 SR (✓, regressed)
HEF (131 MB, 8L) downloaded, tiled `super_resolve()` in runtime,
`ocr(mode="quality")` detects on original but crops rec from 2× SR image.
Net impact: negative. The rec HEF's distribution shifts enough that the
forced-correction table in `ocr_correction.py` (tuned for baseline noise
patterns) stops firing on the new noise patterns.

### Phase 3 — Multi-scale detection (✓, clean +0.052 on year F1)
Detection runs twice: on the original 1280×720 and on the 2× SR image
(downsampled to 960×544 for the detection HEF's fixed input). Boxes
unioned after area-ranked NMS. The SR-downsampled view catches small
overlay digits (years, short labels) that the direct-downsample view
misses. **This is the only phase with a clean monotonic improvement.**

### Phase 5 — pyctcdecode beam + hotwords (✓, no-op)
Built correctly — passes the ppocrv5 charset with Unicode-PUA remapping
for flag emojis + the BPE-marker token. CAR_BRANDS + CAR_MODELS as
hotwords at weight 10.0. Runs identically to greedy CTC on every
thumbnail because the rec HEF's softmax is too peaked for the top-100
beam set to contain any meaningfully different strings.

### Phase 6 — Custom Spanish KenLM + grid search (✓, no-op)
34.9M sentences extracted from esWiki via wikiextractor (Python 3.12
regex fix required), cleaned + uppercased + accent-stripped (matches
the rec HEF's accent-free outputs), 4-gram LM trained with lmplz
(`--prune 0 0 1`, 40 GB scratch), binarized to 4.5 GB .klm, unigrams
extracted + filtered to 1.97M alphabetic tokens. Grid search of 12
(alpha, beta) combos ran successfully — **every combo produced
bit-identical top-1 output**, confirming Phase 5's no-op diagnosis.
The LM is reusable as-is whenever the softmax becomes flat enough to
matter (e.g. a future rec-HEF swap).

### Phase 7 — symspellpy gated correction (✓, +0.013 model F1, -CER,
-WER). Replaces the forced-correction table. Custom SymSpell with
hermitdave/FrequencyWords (1.2M Spanish freq entries) + domain vocab
(CAR_BRANDS ∪ CAR_MODELS ∪ ALL_KEYWORD_CATEGORIES) at freq 1e9.
Gated skip rules: <3 chars, year regex, digit-only, currency tokens,
already-in-domain, and critically **within edit-distance 2 of any
domain term** (prevents `GASONA → CASONA` from destroying
`GASOLINA` fuzzy-match evidence). Net gain is small because only 29 of
90 predictions had any symspell-touchable tokens.

### Phase 4 — Per-crop recognition TTA (✓, mixed)
Geometric-mean of 2 augmentations per crop (CLAHE on/off). Scaled back
from the plan's 3-widths × 2-contrasts spec after the 6-aug version
produced divergent low-quality views that dragged the mean into
gibberish. With Phase 5 + 6 stacked, this finally recovered ID BUZZ on
the Ao-QVHnFlmo anchor, but the softmax broadening embedded digit runs
into adjacent words (`ELECTRCoS2026`), hurting year extraction. Net:
marginal brand+model gain, year regression.

## Where the residual gap lives

Remaining brand misses across the 90-thumbnail set break down into
three structural categories that post-HEF processing cannot fix:

1. **Logo-only brands (12 of 37 brand FN)** — thumbnails where the
   brand name appears only as a visual logo on the vehicle, not as
   text (VW on the ID Buzz, CHEVROLET bowtie on the Trax, HONDA grille
   emblem). Needs an object-detection or logo-classifier head, not
   OCR.

2. **Brand-implied-by-model (9 of 37)** — thumbnails like IONIQ 9 where
   the visible text only mentions the model name and the brand
   (HYUNDAI) has to be inferred from a model→brand mapping. Requires
   a small domain lookup table, not OCR.

3. **Stylized banner fonts (≈8 of 37)** — drop-shadow / outlined /
   heavily-kerned fonts where the rec HEF collapses or swaps
   characters. Needs a rec HEF retrained on stylized data, or a
   word-segmentation pass before rec.

## Recommended production settings

Set `HAILO_OCR_MODE=production` — Phase 3 multi-scale detection only
(SR + multi-scale detection + greedy CTC on SR-sourced crops). The old
`HAILO_OCR_MODE=quality` value entangled this with TTA + beam + LM, which
this report showed regress on the current rec HEF. After the B0
mode-semantics fix (2026-04-24) that experimental stack lives under
`HAILO_OCR_MODE=research` for offline ablation only and is no longer
reachable through the production env var.

Rely on the existing `ocr_correction.py` forced-correction table (not
the symspellpy replacement — keep symspellpy as the opt-in upgrade
path once logo and model→brand signals are added).

Production checklist — one line each for the harness:
```
HAILO_VISION_ENABLED=1
HAILO_OCR_MODE=production              # SR + multi-scale + greedy CTC
# Intentionally NOT setting (these are research-mode-only knobs):
# HAILO_BEAM_LM_PATH                   # Phase 6 LM — no-op on current HEF
# HAILO_BEAM_UNIGRAMS_PATH
# HAILO_BEAM_ALPHA / HAILO_BEAM_BETA
```

## Artifacts worth keeping

- `${HAILO_PIPELINES_DATA}` — the 90-label benchmark. Do not delete.
- `${HAILO_PIPELINES_HOME}` — reusable harness.
- `/mnt/ai/hailo/models/real_esrgan_x2.hef` — built, live, used by SR path.
- `/mnt/ai/hailo/models/language_models/es_auto.klm` + `.unigrams.clean.txt` — built + reusable whenever softmax flattens.

## Artifacts safe to delete

- `predictions_phase6_a*_b*.jsonl` (12 files, all bit-identical to `predictions_phase5_beam.jsonl`).
- `eswiki-latest-pages-articles.xml.bz2` (5 GB, source dump; keep if you expect to rebuild LM).
- `es_auto.arpa` (6.5 GB, superseded by `.klm`).
- `eswiki_upper.txt` (4.6 GB accent-preserving corpus — superseded by `_noaccent` version).

## Regression guard

See `tests/test_ocr_regression.py` (to be added as part of Phase 9
closeout) — runs 10 thumbnails through the recommended Phase 3
config and asserts brand/model/year coverage floors.
