"""Extract high-value semantic entities from noisy OCR text.

For content analysis on creator thumbnails, character-perfect transcription matters
less than *what* the thumbnail mentions. A Hailo-quantized OCR pass produces strings
like "MEJORES WARRO) ELECTPICoS 20V6 ID BUZZ" — the human-legible intent is clear
(Mejores Carros Eléctricos 2026 ID Buzz) even though 4 characters are wrong.

This module maps noisy OCR output → a stable entity block:
  brands, models, years (normalized), category/powertrain/review keywords.

Fuzzy matching via difflib (stdlib; no new dep). Reuses CAR_BRANDS + CAR_MODELS
from title.py so vocab stays in one place.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from vision_shared.features.title import CAR_BRANDS, CAR_MODELS, MODEL_TO_BRAND, REVIEW_KEYWORDS

# Domain keyword vocab — Spanish auto-review context.
SUPERLATIVE_KEYWORDS: tuple[str, ...] = (
    "mejor", "mejores", "peor", "peores", "top", "mas",
)
FRESHNESS_KEYWORDS: tuple[str, ...] = (
    "nuevo", "nueva", "nuevos", "nuevas", "lanzamiento", "estreno",
)
POWERTRAIN_KEYWORDS: tuple[str, ...] = (
    "electrico", "electrica", "electricos", "electricas", "ev", "bev",
    "hibrido", "hibrida", "hibridos", "hibridas", "hybrid", "phev",
    "diesel", "gasolina", "turbo",
)
CATEGORY_KEYWORDS: tuple[str, ...] = (
    "suv", "sedan", "crossover", "coupe", "hatchback", "pickup", "minivan",
    "convertible", "wagon", "familiar",
)

ALL_KEYWORD_CATEGORIES: dict[str, tuple[str, ...]] = {
    "superlative": SUPERLATIVE_KEYWORDS,
    "freshness": FRESHNESS_KEYWORDS,
    "powertrain": POWERTRAIN_KEYWORDS,
    "category": CATEGORY_KEYWORDS,
    "review_format": REVIEW_KEYWORDS,
}

# OCR substitution table — per-character plausible digit readings.
# Applied only when the enclosing pattern looks year-shaped. Multi-valued per char:
# a single OCR glyph can legitimately be multiple digits depending on font.
_YEAR_CHAR_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "0": ("0",),
    "1": ("1",),
    "2": ("2",),
    "3": ("3",),
    "4": ("4",),
    "5": ("5",),
    "6": ("6",),
    "7": ("7",),
    "8": ("8",),
    "9": ("9",),
    "o": ("0",),
    "q": ("0",),
    "d": ("0",),
    "i": ("1",),
    "l": ("1",),
    "|": ("1",),
    "z": ("2",),
    "r": ("2",),
    "e": ("3",),
    "a": ("4",),
    "s": ("5",),
    "g": ("6", "9"),
    "b": ("6", "8"),
    "t": ("7",),
    "v": ("0", "2", "6"),  # V is wide — font-dependent reading
    "y": ("7", "1"),
    "n": ("0",),
}
_YEAR_FULL_RANGE = range(1970, 2031)

_NONWORD_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = _NONWORD_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return [t for t in s.split(" ") if t]


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _extract_fuzzy_matches(tokens: list[str], vocab: tuple[str, ...], min_ratio: float = 0.82) -> list[str]:
    """Return deduplicated sorted list of vocab items found in tokens via fuzzy match.

    For single-word vocab items: checks each token.
    For multi-word vocab items: checks consecutive bigram/trigram token joins.
    """
    found: set[str] = set()

    for item in vocab:
        item_lower = item.lower()
        n_words = len(item_lower.split())

        if n_words > 1:
            # Multi-word: sliding window over tokens, join, fuzzy-compare
            target_len = len(item_lower)
            window = tokens
            for i in range(len(window) - n_words + 1):
                span = " ".join(window[i:i + n_words])
                if _similar(item_lower, span) >= min_ratio:
                    found.add(item_lower)
                    break
            # Also check if the item (space-less) appears as a substring in a long token
            item_compact = item_lower.replace(" ", "")
            for t in tokens:
                if len(t) >= target_len - 1 and _similar(item_compact, t) >= min_ratio:
                    found.add(item_lower)
                    break
        else:
            # Single-word: fuzzy match against each token
            is_short = len(item_lower) < 4
            for t in tokens:
                if len(t) < max(3, len(item_lower) - 2):
                    continue
                if _similar(item_lower, t) >= min_ratio:
                    found.add(item_lower)
                    break
                # substring fuzzy — e.g., "mejor" inside "mejores".
                # Disable for short items (<4 chars) — "ev" would match "nivel", "LEVEE", etc.
                if not is_short and item_lower in t:
                    found.add(item_lower)
                    break

    return sorted(found)


def _expand_year_candidates(raw: str) -> list[int]:
    """For a 4-char run, return ALL plausible year readings by expanding OCR alternatives."""
    if len(raw) != 4:
        return []
    chars = list(raw.lower())
    # Collect alternatives per position
    per_pos: list[tuple[str, ...]] = []
    for ch in chars:
        alts = _YEAR_CHAR_ALTERNATIVES.get(ch)
        if alts is None:
            return []
        per_pos.append(alts)
    # Cartesian product
    from itertools import product
    results: set[int] = set()
    for combo in product(*per_pos):
        try:
            y = int("".join(combo))
        except ValueError:
            continue
        if y in _YEAR_FULL_RANGE:
            results.add(y)
    return sorted(results)


def _detect_years(text: str) -> dict[str, list[int]]:
    """Return both unambiguous and ambiguous year candidates.

    `exact`: 4-digit runs matching 19XX/20XX exactly. High confidence.
    `ambiguous`: OCR-corrupt patterns that COULD be years. All plausible candidates
                 kept so the analyst / downstream logic can disambiguate with context.
    """
    exact: set[int] = set()
    ambiguous: set[int] = set()

    # Exact digit years 1970-2030 anywhere
    for m in re.finditer(r"(?:19[7-9]\d|20[0-2]\d|203[0])", text):
        exact.add(int(m.group()))

    # OCR-corrupt year candidates: starts with digit-or-confusion for 1/2, then any 3 chars
    lower = text.lower()
    for m in re.finditer(r"(?:[12zir][09onod][\w]{2})", lower):
        span = m.group()
        candidates = _expand_year_candidates(span)
        for c in candidates:
            if c not in exact:
                ambiguous.add(c)

    return {
        "exact": sorted(exact),
        "ambiguous": sorted(ambiguous - exact),
    }


def extract_entities(text: str) -> dict[str, object]:
    """Extract brands/models/years/keywords from noisy OCR text.

    Brand extraction is a two-pass union:
      1. Fuzzy match against CAR_BRANDS (explicit OCR evidence).
      2. Inference from extracted models via MODEL_TO_BRAND. Inference
         only ADDS brands; explicit OCR brand evidence is never
         overridden, and provenance is tracked in `brand_sources`.

    The `brands` field stays a plain sorted list (back-compat with
    benchmark scoring); `brand_sources` is a parallel dict mapping
    each brand to its origin ("ocr" | "model_inference").
    """
    if not text:
        return {
            "brands": [],
            "brand_sources": {},
            "models": [],
            "years": [],
            "keywords_superlative": [],
            "keywords_freshness": [],
            "keywords_powertrain": [],
            "keywords_category": [],
            "keywords_review_format": [],
            "brand_count": 0,
            "model_count": 0,
        }

    norm = _normalize(text)
    toks = _tokens(norm)

    ocr_brands = _extract_fuzzy_matches(toks, CAR_BRANDS, min_ratio=0.85)
    models = _extract_fuzzy_matches(toks, CAR_MODELS, min_ratio=0.85)

    # Brand inference from model. Conservative — MODEL_TO_BRAND only
    # contains 1:1 unambiguous mappings; collisions like "VW Polo" vs
    # "Marco Polo" are filtered at the dict level. Explicit OCR brand
    # evidence wins: a brand already present from the fuzzy pass keeps
    # its "ocr" provenance even if a model would also infer it.
    brand_sources: dict[str, str] = {b: "ocr" for b in ocr_brands}
    for m in models:
        inferred = MODEL_TO_BRAND.get(m)
        if inferred and inferred not in brand_sources:
            brand_sources[inferred] = "model_inference"
    brands = sorted(brand_sources.keys())

    years_map = _detect_years(norm)

    out: dict[str, object] = {
        "brands": brands,
        "brand_sources": brand_sources,
        "models": models,
        "years": years_map["exact"],  # back-compat: the confident set
        "years_ambiguous": years_map["ambiguous"],
        "brand_count": len(brands),
        "model_count": len(models),
    }
    for cat, vocab in ALL_KEYWORD_CATEGORIES.items():
        matches = _extract_fuzzy_matches(toks, vocab, min_ratio=0.85)
        out[f"keywords_{cat}"] = matches
    return out
