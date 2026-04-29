"""Phase 7 replay: take any predictions JSONL that has `hailo_ocr_text`
(raw decoder output) and re-run the symspellpy correction + entity
extraction on top. Produces a new predictions JSONL ready for the benchmark
harness.

Pure CPU, no Hailo device — safe to run in parallel with Phase 6 grid.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# allow the shared package to import when run from any cwd
sys.path.insert(0, "/home/dmmdea/openclaw-mcp-servers/_shared")

from openclaw_shared.features.ocr_correction_symspell import correct_text
from openclaw_shared.features.ocr_entities import extract_entities


def replay(src: Path, dst: Path) -> tuple[int, int]:
    n_rows = 0
    n_changed = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            raw = row.get("hailo_ocr_text", "") or ""
            corrected = correct_text(raw)
            ents = extract_entities(corrected)

            row["hailo_ocr_text_corrected"] = corrected
            row["hailo_ocr_char_count"] = len(raw)
            row["hailo_ocr_has_digit"] = bool(re.search(r"\d", raw))

            row["hailo_ocr_brands"] = ents["brands"]
            row["hailo_ocr_models"] = ents["models"]
            row["hailo_ocr_years"] = ents["years"]
            row["hailo_ocr_years_ambiguous"] = ents.get("years_ambiguous", [])
            row["hailo_ocr_brand_count"] = ents["brand_count"]
            row["hailo_ocr_model_count"] = ents["model_count"]
            row["hailo_ocr_has_year"] = bool(ents["years"]) or bool(ents.get("years_ambiguous"))
            row["hailo_ocr_keyword_superlative"] = ents["keywords_superlative"]
            row["hailo_ocr_keyword_freshness"] = ents["keywords_freshness"]
            row["hailo_ocr_keyword_powertrain"] = ents["keywords_powertrain"]
            row["hailo_ocr_keyword_category"] = ents["keywords_category"]
            row["hailo_ocr_keyword_review_format"] = ents["keywords_review_format"]

            if corrected != raw:
                n_changed += 1
            n_rows += 1
            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")
    return n_rows, n_changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input predictions jsonl")
    ap.add_argument("--output", required=True, help="output predictions jsonl")
    args = ap.parse_args()
    src = Path(args.source)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, c = replay(src, dst)
    print(f"[replay] wrote {n} rows → {dst}  ({c} had corrected text differing from raw)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
