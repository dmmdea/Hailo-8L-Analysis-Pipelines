"""Phase 6 grid search: for each (alpha, beta) in the plan's grid, run a
prediction pass with the KenLM beam decoder and benchmark vs ground truth.

Winner selection: maximize brand F1 (primary), then year F1 (tiebreak), then
-CER (tiebreak). Writes all metrics to phase_6_grid_metrics.json and the
winning config to phase_6_metrics.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/dmmdea/openclaw-output/hailo-ocr-quality-plan")
SAMPLE = ROOT / "sample_list.jsonl"
GT = ROOT / "ground_truth.jsonl"
LM_PATH = "/mnt/ai/hailo/models/language_models/es_auto.klm"
UNIGRAMS_PATH = "/mnt/ai/hailo/models/language_models/es_auto.unigrams.clean.txt"
PREDICT_SCRIPT = Path("/home/dmmdea/openclaw-mcp-servers/hailo-vision/scripts/run_predictions.py")
BENCHMARK_SCRIPT = Path("/home/dmmdea/openclaw-mcp-servers/hailo-vision/scripts/ocr_benchmark.py")

# Plan spec: alpha ∈ {0.3, 0.5, 0.7, 1.0}, beta ∈ {0.5, 1.0, 2.0} = 12 combos.
ALPHAS = (0.3, 0.5, 0.7, 1.0)
BETAS = (0.5, 1.0, 2.0)


def run_combo(alpha: float, beta: float) -> dict:
    """Run one prediction + benchmark pair at given (alpha, beta)."""
    tag = f"a{alpha}_b{beta}"
    preds = ROOT / f"predictions_phase6_{tag}.jsonl"
    metrics = ROOT / f"phase_6_metrics_{tag}.json"

    env = os.environ.copy()
    env["HAILO_VISION_ENABLED"] = "1"
    env["HAILO_BEAM_LM_PATH"] = LM_PATH
    env["HAILO_BEAM_UNIGRAMS_PATH"] = UNIGRAMS_PATH
    env["HAILO_BEAM_ALPHA"] = str(alpha)
    env["HAILO_BEAM_BETA"] = str(beta)
    env["PYTHONPATH"] = "/home/dmmdea/openclaw-mcp-servers/_shared"

    print(f"\n========= combo alpha={alpha} beta={beta} =========", flush=True)
    # Phase 6 is the beam/LM ablation track — research mode is the only one
    # that activates the beam decoder + KenLM rescoring this script tunes.
    subprocess.run(
        [sys.executable, str(PREDICT_SCRIPT),
         "--sample", str(SAMPLE),
         "--output", str(preds),
         "--mode", "research"],
        env=env, check=True,
    )
    subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT),
         "--ground-truth", str(GT),
         "--predictions", str(preds),
         "--output", str(metrics)],
        env=env, check=True,
    )
    with metrics.open() as f:
        data = json.load(f)
    data["alpha"] = alpha
    data["beta"] = beta
    data["tag"] = tag
    return data


def score_for_ranking(m: dict) -> tuple:
    """Higher is better. brand F1 primary, then year F1, then -CER."""
    ents = m.get("entity_scores", {})
    brand = ents.get("brand", {}).get("f1", 0.0)
    year = ents.get("year", {}).get("f1", 0.0)
    text = m.get("text_scores", {}).get("strict", {})
    cer = text.get("cer", 1.0)
    return (brand, year, -cer)


def main() -> int:
    results: list[dict] = []
    for alpha in ALPHAS:
        for beta in BETAS:
            try:
                r = run_combo(alpha, beta)
            except subprocess.CalledProcessError as exc:
                print(f"[grid] combo a={alpha} b={beta} failed: {exc}", file=sys.stderr)
                continue
            results.append(r)
            # Flush partial results so an interrupted run keeps what it had
            with (ROOT / "phase_6_grid_metrics.json").open("w") as f:
                json.dump(results, f, indent=2, default=str)

    if not results:
        print("[grid] no combos completed successfully", file=sys.stderr)
        return 1

    ranked = sorted(results, key=score_for_ranking, reverse=True)
    winner = ranked[0]

    summary = {
        "winner": {k: winner[k] for k in ("alpha", "beta", "tag", "entity_scores", "text_scores")},
        "all_combos": [
            {"alpha": r["alpha"], "beta": r["beta"],
             "brand_f1": r.get("entity_scores", {}).get("brand", {}).get("f1", 0.0),
             "model_f1": r.get("entity_scores", {}).get("model", {}).get("f1", 0.0),
             "year_f1":  r.get("entity_scores", {}).get("year", {}).get("f1", 0.0),
             "cer": r.get("text_scores", {}).get("strict", {}).get("cer", 1.0)}
            for r in ranked
        ],
    }
    with (ROOT / "phase_6_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[grid] winner: alpha={winner['alpha']} beta={winner['beta']} tag={winner['tag']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
