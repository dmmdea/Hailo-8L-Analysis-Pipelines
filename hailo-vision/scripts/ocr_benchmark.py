"""Phase 1 — OCR benchmark harness.

Consumes a ground-truth JSONL (from Phase 0) and a predictions JSONL (from any
pipeline pass) and emits per-entity precision/recall/F1 + CER/WER.

Predictions schema (flexible):
  video_id                       — required, joins to ground truth
  hailo_ocr_text_corrected OR    — preferred text source
  hailo_ocr_text                 — fallback
  hailo_ocr_brands, models, years, years_ambiguous (lists)
  hailo_ocr_keyword_<cat>        — per-category keyword lists

Ground-truth schema (from phase_0):
  video_id, ground_truth_text, ground_truth_brands/models/years,
  ground_truth_years_ambiguous, ground_truth_keywords (dict of cat→list)

Usage:
  python ocr_benchmark.py \
      --ground-truth ground_truth.jsonl \
      --predictions predictions_baseline.jsonl \
      --output phase_1_baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer

# Brand/model aliases collapsed to a single canonical form so scoring is
# equivalence-class aware (e.g., "mercedes-benz" == "mercedes benz" == "mercedes").
BRAND_CANONICAL: dict[str, str] = {
    "mercedes": "mercedes",
    "mercedes benz": "mercedes",
    "mercedes-benz": "mercedes",
    "vw": "vw",
    "volkswagen": "vw",
    "alfa": "alfa romeo",
    "alfa romeo": "alfa romeo",
    "citroen": "citroen",
    "citroën": "citroen",
    "rolls royce": "rolls royce",
    "rolls-royce": "rolls royce",
    "chevrolet": "chevrolet",
    "chevy": "chevrolet",
    "aston": "aston martin",
    "aston martin": "aston martin",
}

MODEL_CANONICAL: dict[str, str] = {
    "crv": "cr-v",
    "cr-v": "cr-v",
    "id buzz": "id buzz",
    "id.buzz": "id buzz",
}

KEYWORD_CANONICAL: dict[str, str] = {
    "opinión": "opinion",
    "opinion": "opinion",
    "revisión": "revision",
    "revision": "revision",
    "electrico": "electrico",
    "electricos": "electrico",
    "electrica": "electrico",
    "electricas": "electrico",
    "hibrido": "hibrido",
    "hibrida": "hibrido",
    "hibridos": "hibrido",
    "hibridas": "hibrido",
    "hybrid": "hibrido",
    "nuevo": "nuevo",
    "nuevos": "nuevo",
    "nueva": "nuevo",
    "nuevas": "nuevo",
    "mejor": "mejor",
    "mejores": "mejor",
    "peor": "peor",
    "peores": "peor",
}


def _canon(s: str, table: dict[str, str]) -> str:
    return table.get(s.lower(), s.lower())


def _canon_set(items, table: dict[str, str]) -> set[str]:
    return {_canon(str(x).lower(), table) for x in (items or []) if str(x).strip()}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _normalize_text_for_cer(s: str) -> str:
    """Lowercase + NFKD-strip accents + collapse whitespace. CER/WER operate
    on this normalized form so accent-mismatches aren't flagged as errors when
    testing anything that doesn't explicitly carry accents."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strict_text(s: str) -> str:
    """Whitespace-collapsed only, case + accent preserved."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows[r["video_id"]] = r
    return rows


def _extract_pred_keywords(pred: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for k, v in pred.items():
        if not k.startswith("hailo_ocr_keyword_"):
            continue
        cat = k.replace("hailo_ocr_keyword_", "")
        out[cat] = _canon_set(v, KEYWORD_CANONICAL)
    return out


def _extract_gt_keywords(gt: dict) -> dict[str, set[str]]:
    raw = gt.get("ground_truth_keywords", {}) or {}
    return {cat: _canon_set(items, KEYWORD_CANONICAL) for cat, items in raw.items()}


def evaluate(gt_rows: dict[str, dict], pred_rows: dict[str, dict]) -> dict:
    joined = [(gt_rows[vid], pred_rows[vid]) for vid in gt_rows if vid in pred_rows]
    missing_preds = [vid for vid in gt_rows if vid not in pred_rows]
    unexpected_preds = [vid for vid in pred_rows if vid not in gt_rows]

    # ---- Entity F1 (micro) ----
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    per_video: list[dict] = []

    for gt, pred in joined:
        gt_brands = _canon_set(gt.get("ground_truth_brands"), BRAND_CANONICAL)
        gt_models = _canon_set(gt.get("ground_truth_models"), MODEL_CANONICAL)
        gt_years = set(int(y) for y in (gt.get("ground_truth_years") or []))

        pr_brands = _canon_set(pred.get("hailo_ocr_brands"), BRAND_CANONICAL)
        pr_models = _canon_set(pred.get("hailo_ocr_models"), MODEL_CANONICAL)
        pr_years = set(int(y) for y in (pred.get("hailo_ocr_years") or []))

        for entity, gt_set, pr_set in (
            ("brand", gt_brands, pr_brands),
            ("model", gt_models, pr_models),
            ("year", gt_years, pr_years),
        ):
            tp[entity] += len(gt_set & pr_set)
            fp[entity] += len(pr_set - gt_set)
            fn[entity] += len(gt_set - pr_set)

        # Per-keyword-category
        gt_kw = _extract_gt_keywords(gt)
        pr_kw = _extract_pred_keywords(pred)
        cats = set(gt_kw) | set(pr_kw)
        for cat in cats:
            gt_c = gt_kw.get(cat, set())
            pr_c = pr_kw.get(cat, set())
            key = f"kw:{cat}"
            tp[key] += len(gt_c & pr_c)
            fp[key] += len(pr_c - gt_c)
            fn[key] += len(gt_c - pr_c)

        per_video.append({
            "video_id": gt["video_id"],
            "channel": gt.get("channel"),
            "brand_miss": sorted(gt_brands - pr_brands),
            "brand_fp": sorted(pr_brands - gt_brands),
            "model_miss": sorted(gt_models - pr_models),
            "model_fp": sorted(pr_models - gt_models),
            "year_miss": sorted(gt_years - pr_years),
            "year_fp": sorted(pr_years - gt_years),
            "gt_text": gt.get("ground_truth_text", ""),
            "pred_text_corrected": pred.get("hailo_ocr_text_corrected") or pred.get("hailo_ocr_text", ""),
        })

    entity_scores = {}
    for entity in list(tp.keys()) + list(fp.keys()) + list(fn.keys()):
        if entity in entity_scores:
            continue
        p, r, f1 = _prf(tp[entity], fp[entity], fn[entity])
        entity_scores[entity] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp[entity],
            "fp": fp[entity],
            "fn": fn[entity],
        }

    # ---- CER / WER ----
    # Build pairs of (reference, hypothesis). Skip thumbnails where GT has no text
    # (pure product shots) from CER/WER to avoid divide-by-zero inflation.
    strict_refs, strict_hyps = [], []
    lenient_refs, lenient_hyps = [], []
    for gt, pred in joined:
        gt_text = gt.get("ground_truth_text", "") or ""
        pr_text = pred.get("hailo_ocr_text_corrected") or pred.get("hailo_ocr_text", "") or ""
        if not gt_text.strip():
            continue
        strict_refs.append(_strict_text(gt_text))
        strict_hyps.append(_strict_text(pr_text))
        lenient_refs.append(_normalize_text_for_cer(gt_text))
        lenient_hyps.append(_normalize_text_for_cer(pr_text))

    text_scores = {}
    if strict_refs:
        text_scores["strict"] = {
            "n": len(strict_refs),
            "cer": round(float(jiwer.cer(strict_refs, strict_hyps)), 4),
            "wer": round(float(jiwer.wer(strict_refs, strict_hyps)), 4),
        }
        text_scores["lenient_nocase_noaccent"] = {
            "n": len(lenient_refs),
            "cer": round(float(jiwer.cer(lenient_refs, lenient_hyps)), 4),
            "wer": round(float(jiwer.wer(lenient_refs, lenient_hyps)), 4),
        }

    # ---- Highlighted failures: known anchor thumbnails ----
    anchors = ("Ao-QVHnFlmo", "8nP7NrrmTxU")
    anchor_detail = {}
    pv_by_id = {p["video_id"]: p for p in per_video}
    for a in anchors:
        if a in pv_by_id:
            anchor_detail[a] = pv_by_id[a]

    return {
        "n_videos_in_gt": len(gt_rows),
        "n_videos_in_pred": len(pred_rows),
        "n_joined": len(joined),
        "missing_predictions": missing_preds,
        "unexpected_predictions": unexpected_preds,
        "entity_scores": entity_scores,
        "text_scores": text_scores,
        "anchor_detail": anchor_detail,
        "per_video": per_video,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    gt = _load_jsonl(args.ground_truth)
    pred = _load_jsonl(args.predictions)
    report = evaluate(gt, pred)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"videos: gt={report['n_videos_in_gt']} pred={report['n_videos_in_pred']} joined={report['n_joined']}")
    print()
    print("entity F1 (micro):")
    for ent, s in sorted(report["entity_scores"].items()):
        if s["tp"] + s["fn"] == 0 and s["fp"] == 0:
            continue
        print(f"  {ent:<22}  P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  (tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    print()
    print("text scores:")
    for mode, s in report["text_scores"].items():
        print(f"  {mode:<22}  CER={s['cer']:.4f}  WER={s['wer']:.4f}  (n={s['n']})")
    print()
    print("anchor detail:")
    for a, d in report["anchor_detail"].items():
        print(f"  {a}: miss_brand={d['brand_miss']} miss_model={d['model_miss']} miss_year={d['year_miss']}")
        print(f"         fp_brand={d['brand_fp']} fp_model={d['model_fp']}")
        print(f"         gt_text={d['gt_text']!r}")
        print(f"       pred_text={d['pred_text_corrected']!r}")
    print()
    print(f"wrote report → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
