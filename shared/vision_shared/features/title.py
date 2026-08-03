"""
Title text feature extractor — Spanish-aware lexical features.

Designed for YouTube video titles. Produces a flat dict per title with features that
capture the packaging patterns the playbook calls out (price framing, question hooks,
emotional intensifiers, year mentions, brand/model density, etc.).

All features are deterministic and computable without ML deps — matches Week 2's
"CPU only, no VLM" scope. Sentence-transformer embeddings + spacy POS are deferred
to a later pass when those deps are worth the install weight.

Features emitted per title (all floats/ints/bools for easy pandas.DataFrame consumption):
  char_count          Total characters (including spaces).
  word_count          Whitespace-delimited tokens.
  sentence_count      Approximate sentence count (splits on .!?¿¡).
  avg_word_length     Mean chars per token.
  digit_count         Count of digits [0-9].
  has_digit           Bool: at least one digit present.
  uppercase_word_count    Tokens that are ≥3 chars AND fully uppercase (excludes "I", "DE", etc.).
  uppercase_word_ratio    uppercase_word_count / word_count.
  has_question_mark   Bool: contains "?" or "¿".
  starts_with_question Bool: title starts with "¿" (Spanish interrogative opener).
  exclamation_count   Count of "!" or "¡".
  has_emoji           Bool: contains any non-ASCII pictograph (approx via Unicode category).
  emoji_count         Count of emoji code points.
  has_year            Bool: title mentions a 4-digit year in 2010–2030 range.
  year_mentions       List of years found (as a count since we emit flat dicts).
  has_currency_cop    Bool: "70 millones", "COP", "$" followed by digits, "millones de pesos".
  has_currency_usd    Bool: "USD", "dolares", "dólares", "us$", or "$X USD" pattern.
  has_price_framing   Bool: any currency OR explicit price framing keyword ("cuesta", "precio").
  has_superlative     Bool: contains "MEJOR", "PEOR", "MAYOR", "MENOR", "TOP", "#1" (case-insens).
  has_negation        Bool: "NO COMPRES", "NO", "NUNCA" (case-insens; leading context).
  has_comparative     Bool: "vs", "versus", "comparado", "vs.".
  has_list_number     Bool: starts with "N X" or "X autos" pattern (listicle indicator).
  has_brand_mention   Bool: contains any recognizable car brand name.
  brand_count         Count of distinct brands mentioned.
  has_model_mention   Bool: contains any recognizable car model name (e.g. Ioniq, RAV4, Model Y, CR-V).
  model_count         Count of distinct models mentioned.
  has_brand_or_model  Bool: union of has_brand_mention and has_model_mention. Useful when a
                      title names only models with no brand ("Ioniq 9, Uncharted, Trailseeker").
  has_model_year      Bool: brand OR model mention AND 20XX year present in title.
  has_review_keyword  Bool: "review", "reseña", "prueba", "test drive", "POV", "análisis".
  has_curiosity_hook  Bool: "no te imaginas", "descubre", "la verdad", "secreto", "increíble",
                      "asombroso", "revelación", "misterio", etc.
  pipe_section_count  Count of "|" pipes (YouTube title segment delimiters).
  hashtag_count       Count of "#tag" patterns.
  is_collab           Bool: looks like a collaboration with another creator — "@"-mention
                      of another channel OR an explicit collab keyword ("saludo especial",
                      "colaboración", "featuring", "ft.", "feat."). Packaging analyses
                      should typically exclude these from matched-pair cohorts because
                      collab reach is a different experiment type (guest-driven traffic).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


# Known automotive brands likely to appear in Spanish auto-review titles.
# Conservative list — avoids overfitting to rare brands; case-insensitive match.
CAR_BRANDS: tuple[str, ...] = (
    "toyota", "honda", "nissan", "mazda", "hyundai", "kia", "chevrolet", "chevy",
    "ford", "dodge", "jeep", "ram", "chrysler", "buick", "cadillac", "gmc",
    "volkswagen", "vw", "audi", "bmw", "mercedes", "mercedes-benz", "mercedes benz",
    "porsche", "mini", "opel", "seat", "peugeot", "renault", "citroën", "citroen",
    "fiat", "alfa romeo", "alfa", "ferrari", "lamborghini", "maserati", "bentley",
    "rolls-royce", "rolls royce", "aston martin", "aston", "mclaren", "jaguar",
    "land rover", "range rover", "volvo", "saab", "subaru", "suzuki", "mitsubishi",
    "lexus", "acura", "infiniti", "tesla", "rivian", "lucid", "byd", "mg", "jac",
    "chery", "great wall", "haval", "geely",
)

# Common car model names across brands. Curated to avoid short/ambiguous names that
# false-positive on Spanish filler words ("GT", "R", "A3"-style trim codes skipped).
# Match is done with non-word-char lookarounds so multi-word and hyphenated names work.
CAR_MODELS: tuple[str, ...] = (
    # Toyota
    "camry", "corolla", "rav4", "highlander", "4runner", "tacoma", "tundra", "prius",
    "bz4x", "supra", "sequoia", "sienna", "avanza", "yaris", "hilux", "fortuner",
    # Honda
    "civic", "accord", "cr-v", "crv", "hr-v", "hrv", "pilot", "odyssey", "passport",
    "fit", "jazz", "ridgeline",
    # Nissan
    "altima", "sentra", "versa", "rogue", "pathfinder", "murano", "titan", "frontier",
    "kicks", "leaf", "maxima", "armada", "qashqai",
    # Hyundai
    "ioniq", "tucson", "santa fe", "elantra", "sonata", "accent", "palisade", "venue",
    "creta", "kona", "santa cruz",
    # Kia
    "sorento", "sportage", "seltos", "telluride", "forte", "stinger", "k5",
    "ev3", "ev4", "ev5", "ev6", "ev9",
    # Ford
    "f-150", "f150", "mustang", "explorer", "escape", "edge", "ranger", "bronco",
    "maverick", "expedition", "fusion", "focus", "lightning",
    # Chevrolet
    "silverado", "tahoe", "suburban", "equinox", "traverse", "blazer", "camaro",
    "corvette", "malibu", "impala", "trailblazer", "onix", "trax", "tracker",
    "trailseeker", "cruze", "sonic", "spin", "colorado",
    # Cadillac
    "escalade", "lyriq", "vistiq", "celestiq", "optiq", "ct4", "ct5", "ct6",
    "xt4", "xt5", "xt6",
    # Buick / GMC
    "enclave", "encore", "envision", "regal", "sierra", "acadia", "terrain",
    "yukon", "canyon",
    # Tesla
    "model s", "model 3", "model x", "model y", "cybertruck", "cybercab", "robotaxi",
    "roadster",
    # Volkswagen
    "id.3", "id.4", "id.7", "id buzz", "id.buzz", "taos", "atlas", "tiguan",
    "jetta", "passat", "golf", "polo", "nivus", "t-cross", "tcross",
    # Subaru
    "outback", "forester", "crosstrek", "impreza", "ascent", "legacy", "brz", "uncharted",
    # Jeep
    "wrangler", "grand cherokee", "gladiator", "compass", "renegade", "cherokee",
    "wagoneer", "grand wagoneer",
    # BYD
    "atto", "seal", "dolphin", "yuan", "song", "tang", "han", "destroyer",
    # BMW (electric + common — digit-containing names only; 2-letter trim codes dropped)
    "i4", "i5", "i7", "ix1", "ix3",
    # Mazda (hyphenated trims only — short 2-letter ones dropped)
    "cx-3", "cx-30", "cx-5", "cx-50", "cx-70", "cx-9", "cx-90", "mx-5", "mx-30",
    # Mitsubishi
    "outlander", "l200", "montero", "eclipse cross",
)
# NOTE: deliberately omitted short Lexus/BMW trim codes ("is", "es", "ls", "nx", "rx")
# because "es" collides with the Spanish verb "es" (=is) and would false-positive heavily.
# Lexus/Acura/Infiniti full-name coverage would need a brand-prefix check, deferred.


# Model→brand inference table — closes the brand-implied-by-model FN class
# (IONIQ visible in OCR, HYUNDAI is not — but the analyst still wants
# `hyundai` in the brand list). Conservative: 1:1 unambiguous mappings only.
#
# Inference rule (in ocr_entities.extract_entities): for every model in
# the extracted list, look up MODEL_TO_BRAND[model] and ADD it to the
# brand list — never override explicit OCR brand evidence. Inference is
# tracked in `brand_sources` for downstream provenance.
#
# Conservative omissions:
#   - BYD short-name models that collide with common English/Spanish
#     words: `tang`, `song`, `han`, `yuan`, `destroyer`. The fuzzy
#     matcher might emit these on noisy OCR ("DESTROZA" → "destroyer")
#     and the brand inference would compound the FP into a brand FP.
#   - `eclipse cross` (Mitsubishi) — the bigram requirement reduces
#     direct FP risk, but cross-brand connotation of "eclipse" alone
#     makes us want to see the bigram match before risking an FP.
# All other models in CAR_MODELS map deterministically to one brand.
MODEL_TO_BRAND: dict[str, str] = {
    # Toyota
    "camry": "toyota", "corolla": "toyota", "rav4": "toyota",
    "highlander": "toyota", "4runner": "toyota", "tacoma": "toyota",
    "tundra": "toyota", "prius": "toyota", "bz4x": "toyota",
    "supra": "toyota", "sequoia": "toyota", "sienna": "toyota",
    "avanza": "toyota", "yaris": "toyota", "hilux": "toyota",
    "fortuner": "toyota",
    # Honda
    "civic": "honda", "accord": "honda", "cr-v": "honda", "crv": "honda",
    "hr-v": "honda", "hrv": "honda", "pilot": "honda", "odyssey": "honda",
    "passport": "honda", "fit": "honda", "jazz": "honda", "ridgeline": "honda",
    # Nissan
    "altima": "nissan", "sentra": "nissan", "versa": "nissan",
    "rogue": "nissan", "pathfinder": "nissan", "murano": "nissan",
    "titan": "nissan", "frontier": "nissan", "kicks": "nissan",
    "leaf": "nissan", "maxima": "nissan", "armada": "nissan",
    "qashqai": "nissan",
    # Hyundai
    "ioniq": "hyundai", "tucson": "hyundai", "santa fe": "hyundai",
    "elantra": "hyundai", "sonata": "hyundai", "accent": "hyundai",
    "palisade": "hyundai", "venue": "hyundai", "creta": "hyundai",
    "kona": "hyundai", "santa cruz": "hyundai",
    # Kia
    "sorento": "kia", "sportage": "kia", "seltos": "kia",
    "telluride": "kia", "forte": "kia", "stinger": "kia", "k5": "kia",
    "ev3": "kia", "ev4": "kia", "ev5": "kia", "ev6": "kia", "ev9": "kia",
    # Ford
    "f-150": "ford", "f150": "ford", "mustang": "ford", "explorer": "ford",
    "escape": "ford", "edge": "ford", "ranger": "ford", "bronco": "ford",
    "maverick": "ford", "expedition": "ford", "fusion": "ford",
    "focus": "ford", "lightning": "ford",
    # Chevrolet
    "silverado": "chevrolet", "tahoe": "chevrolet", "suburban": "chevrolet",
    "equinox": "chevrolet", "traverse": "chevrolet", "blazer": "chevrolet",
    "camaro": "chevrolet", "corvette": "chevrolet", "malibu": "chevrolet",
    "impala": "chevrolet", "trailblazer": "chevrolet", "onix": "chevrolet",
    "trax": "chevrolet", "tracker": "chevrolet", "trailseeker": "chevrolet",
    "cruze": "chevrolet", "sonic": "chevrolet", "spin": "chevrolet",
    "colorado": "chevrolet",
    # Cadillac
    "escalade": "cadillac", "lyriq": "cadillac", "vistiq": "cadillac",
    "celestiq": "cadillac", "optiq": "cadillac",
    "ct4": "cadillac", "ct5": "cadillac", "ct6": "cadillac",
    "xt4": "cadillac", "xt5": "cadillac", "xt6": "cadillac",
    # Buick
    "enclave": "buick", "encore": "buick", "envision": "buick",
    "regal": "buick",
    # GMC
    "sierra": "gmc", "acadia": "gmc", "terrain": "gmc",
    "yukon": "gmc", "canyon": "gmc",
    # Tesla
    "model s": "tesla", "model 3": "tesla", "model x": "tesla",
    "model y": "tesla", "cybertruck": "tesla", "cybercab": "tesla",
    "robotaxi": "tesla", "roadster": "tesla",
    # Volkswagen
    "id.3": "volkswagen", "id.4": "volkswagen", "id.7": "volkswagen",
    "id buzz": "volkswagen", "id.buzz": "volkswagen",
    "taos": "volkswagen", "atlas": "volkswagen", "tiguan": "volkswagen",
    "jetta": "volkswagen", "passat": "volkswagen", "golf": "volkswagen",
    "polo": "volkswagen", "nivus": "volkswagen",
    "t-cross": "volkswagen", "tcross": "volkswagen",
    # Subaru
    "outback": "subaru", "forester": "subaru", "crosstrek": "subaru",
    "impreza": "subaru", "ascent": "subaru", "legacy": "subaru",
    "brz": "subaru", "uncharted": "subaru",
    # Jeep
    "wrangler": "jeep", "grand cherokee": "jeep", "gladiator": "jeep",
    "compass": "jeep", "renegade": "jeep", "cherokee": "jeep",
    "wagoneer": "jeep", "grand wagoneer": "jeep",
    # BYD (only the more-distinctive 4+ char names; short-words dropped)
    "atto": "byd", "seal": "byd", "dolphin": "byd",
    # BMW
    "i4": "bmw", "i5": "bmw", "i7": "bmw", "ix1": "bmw", "ix3": "bmw",
    # Mazda
    "cx-3": "mazda", "cx-30": "mazda", "cx-5": "mazda", "cx-50": "mazda",
    "cx-70": "mazda", "cx-9": "mazda", "cx-90": "mazda",
    "mx-5": "mazda", "mx-30": "mazda",
    # Mitsubishi
    "outlander": "mitsubishi", "l200": "mitsubishi", "montero": "mitsubishi",
}

CURIOSITY_HOOKS: tuple[str, ...] = (
    "no te imaginas", "descubre", "la verdad", "secreto", "increíble", "increible",
    "asombroso", "revelación", "revelacion", "misterio", "sorprendente",
    "no lo creerás", "no lo creeras", "lo que no sabías", "lo que no sabias",
)

REVIEW_KEYWORDS: tuple[str, ...] = (
    "review", "reseña", "resena", "prueba", "test drive", "testdrive",
    "pov", "análisis", "analisis", "opinión", "opinion", "revisión", "revision",
)

SUPERLATIVES: tuple[str, ...] = ("mejor", "peor", "mayor", "menor", "top")

NEGATIONS: tuple[str, ...] = ("no compres", "no compre", "nunca compres", "nunca compre", "jamás")

COMPARATIVES: tuple[str, ...] = (" vs ", " vs. ", "versus ", "comparado ", "comparativa")

# Collab-indicator keywords. Kept conservative to avoid false-positives on common
# Spanish connectives ("con", "junto" alone aren't sufficient — they're too generic).
COLLAB_KEYWORDS: tuple[str, ...] = (
    "saludo especial", "colaboración", "colaboracion", "colab", "colabo",
    "featuring", "ft.", "feat.", "en colaboración", "en colaboracion",
)
# "@Handle" mention is the strongest signal — YouTube tags another channel explicitly.
_COLLAB_AT_MENTION_RE = re.compile(r"(?<!\w)@\w{2,}")


_DIGIT_RE = re.compile(r"\d")
_YEAR_RE = re.compile(r"(?<!\d)(20[1-3]\d)(?!\d)")
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
_LIST_START_RE = re.compile(r"^\s*(\d+|top\s*\d+|\d+\s*autos?|\d+\s*carros?|\d+\s*suvs?)\b", re.IGNORECASE)
_COP_RE = re.compile(
    r"\$\s?\d|cop\b|colombian?os?\b|\bpesos?\b|\bmillones?\b",
    re.IGNORECASE,
)
_USD_RE = re.compile(r"\bus\s?\$|\busd\b|\bdolares\b|\bdólares\b|\buss?\$?\s*\d", re.IGNORECASE)
_PRICE_KEYWORD_RE = re.compile(r"\b(cuesta|precio|cuanto|cuánto|millones?)\b", re.IGNORECASE)
_QUESTION_OPENER_RE = re.compile(r"^\s*[¿?]")


def _build_model_matcher() -> re.Pattern[str]:
    """Compile a single regex that matches any CAR_MODELS entry as a standalone token.

    Uses non-word-char lookarounds (not \\b) so hyphenated names like "cr-v", "f-150",
    "cx-5" and dotted names like "id.3" match correctly — \\b would treat the hyphen
    or dot as a word boundary and split the match.
    """
    # Sort longest-first so multi-word names ("grand cherokee", "model y") match before
    # any prefix model ("cherokee", "model 3") would greedily consume.
    sorted_models = sorted(CAR_MODELS, key=len, reverse=True)
    escaped = [re.escape(m) for m in sorted_models]
    alt = "|".join(escaped)
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)


_MODEL_RE = _build_model_matcher()


def _count_emojis(s: str) -> int:
    """Count characters that land in Unicode categories typical for pictographs.

    Uses the `So` (Symbol, other) category + specific surrogate ranges. Sufficient for
    typical YouTube-title emojis without pulling in an emoji library dep.
    """
    count = 0
    for ch in s:
        cp = ord(ch)
        # Pictographic ranges — covers emoji, symbols, transport, etc.
        if (
            0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0x1F000 <= cp <= 0x1F2FF
            or unicodedata.category(ch) == "So"
        ):
            count += 1
    return count


def _uppercase_tokens(tokens: list[str]) -> list[str]:
    """Tokens that are fully uppercase AND at least 3 chars. Excludes short connectives."""
    return [t for t in tokens if len(t) >= 3 and t.isupper() and any(c.isalpha() for c in t)]


def extract_title_features(title: str) -> dict[str, Any]:
    """Extract all features for a single title. Empty/None input → features with safe defaults."""
    if not title or not isinstance(title, str):
        return _empty_features()

    s = title.strip()
    low = s.lower()
    tokens = re.findall(r"\S+", s)

    digits = _DIGIT_RE.findall(s)
    year_mentions = _YEAR_RE.findall(s)
    hashtags = _HASHTAG_RE.findall(s)
    up_tokens = _uppercase_tokens(tokens)

    brands_found = {b for b in CAR_BRANDS if b in low}
    models_found = {m.group(0).lower() for m in _MODEL_RE.finditer(low)}

    has_superlative = any(sl in low for sl in SUPERLATIVES) or "#1" in low or "#1" in s
    has_negation = any(ng in low for ng in NEGATIONS)
    has_comparative = any(cp in low for cp in COMPARATIVES)
    has_review_kw = any(kw in low for kw in REVIEW_KEYWORDS)
    has_curiosity = any(hk in low for hk in CURIOSITY_HOOKS)

    has_cop = bool(_COP_RE.search(s))
    has_usd = bool(_USD_RE.search(s))
    has_price_kw = bool(_PRICE_KEYWORD_RE.search(s))

    has_at_mention = bool(_COLLAB_AT_MENTION_RE.search(s))
    has_collab_keyword = any(k in low for k in COLLAB_KEYWORDS)
    is_collab = has_at_mention or has_collab_keyword

    # has_model_year = brand OR model mention AND year mention within the same title
    has_model_year = (bool(brands_found) or bool(models_found)) and bool(year_mentions)

    sentence_count = len(re.findall(r"[.!?¿¡]+", s)) or 1
    avg_word_length = (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0

    features: dict[str, Any] = {
        "char_count": len(s),
        "word_count": len(tokens),
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 3),
        "digit_count": len(digits),
        "has_digit": bool(digits),
        "uppercase_word_count": len(up_tokens),
        "uppercase_word_ratio": round(len(up_tokens) / len(tokens), 3) if tokens else 0.0,
        "has_question_mark": "?" in s or "¿" in s,
        "starts_with_question": bool(_QUESTION_OPENER_RE.match(s)),
        "exclamation_count": s.count("!") + s.count("¡"),
        "has_emoji": _count_emojis(s) > 0,
        "emoji_count": _count_emojis(s),
        "has_year": bool(year_mentions),
        "year_mentions": len(year_mentions),
        "has_currency_cop": has_cop,
        "has_currency_usd": has_usd,
        "has_price_framing": has_cop or has_usd or has_price_kw,
        "has_superlative": has_superlative,
        "has_negation": has_negation,
        "has_comparative": has_comparative,
        "has_list_number": bool(_LIST_START_RE.match(s)),
        "has_brand_mention": bool(brands_found),
        "brand_count": len(brands_found),
        "has_model_mention": bool(models_found),
        "model_count": len(models_found),
        "has_brand_or_model": bool(brands_found) or bool(models_found),
        "has_model_year": has_model_year,
        "has_review_keyword": has_review_kw,
        "has_curiosity_hook": has_curiosity,
        "pipe_section_count": s.count("|"),
        "hashtag_count": len(hashtags),
        "is_collab": is_collab,
    }
    return features


def _empty_features() -> dict[str, Any]:
    return {
        "char_count": 0, "word_count": 0, "sentence_count": 0, "avg_word_length": 0.0,
        "digit_count": 0, "has_digit": False, "uppercase_word_count": 0,
        "uppercase_word_ratio": 0.0, "has_question_mark": False,
        "starts_with_question": False, "exclamation_count": 0, "has_emoji": False,
        "emoji_count": 0, "has_year": False, "year_mentions": 0,
        "has_currency_cop": False, "has_currency_usd": False, "has_price_framing": False,
        "has_superlative": False, "has_negation": False, "has_comparative": False,
        "has_list_number": False, "has_brand_mention": False, "brand_count": 0,
        "has_model_mention": False, "model_count": 0, "has_brand_or_model": False,
        "has_model_year": False, "has_review_keyword": False, "has_curiosity_hook": False,
        "pipe_section_count": 0, "hashtag_count": 0, "is_collab": False,
    }


def extract_title_features_batch(titles: list[str]) -> list[dict[str, Any]]:
    """Convenience: apply extract_title_features to a list of titles."""
    return [extract_title_features(t) for t in titles]
