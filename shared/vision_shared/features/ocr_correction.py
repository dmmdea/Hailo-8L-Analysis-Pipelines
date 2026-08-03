"""Domain-aware OCR post-correction for Spanish automotive thumbnails.

Design principles:
  1. Never "correct" a word that's already in our domain vocab (brand/model/keyword).
     Blind pyspellchecker corrects SUBARU→subir, CADILLAC→cadillar — destroying signal.
  2. Correct only when the candidate has LOWER edit distance to a known-good word
     than it does to any other vocab word (not-too-ambiguous).
  3. Restore Spanish diacritics as a side effect when the spellcheck candidate is
     close enough to the original — e.g., `electrico → eléctrico`.
  4. Preserve uppercase / casing of the original when correcting.

The domain vocab is loaded from CAR_BRANDS + CAR_MODELS + keyword lists, cached.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

from vision_shared.features.title import CAR_BRANDS, CAR_MODELS, REVIEW_KEYWORDS

try:
    from spellchecker import SpellChecker  # type: ignore[import-untyped]
    _HAS_SPELLCHECKER = True
except ImportError:
    _HAS_SPELLCHECKER = False


# Additional automotive domain vocab (commonly present, often missed by general dict)
_EXTRA_DOMAIN_WORDS: tuple[str, ...] = (
    # Spanish market/locale + common automotive plurals the base dict lacks
    "mejor", "mejores", "peor", "peores", "nuevo", "nueva", "nuevos", "nuevas",
    "prueba", "pruebas", "review", "reviews", "análisis", "opinión", "revisión",
    "carro", "carros", "auto", "autos", "coche", "coches", "vehículo", "vehículos",
    "camioneta", "camionetas", "moto", "motos",
    # Powertrain/category (singular + plurals)
    "eléctrico", "eléctrica", "eléctricos", "eléctricas",
    "híbrido", "híbrida", "híbridos", "híbridas",
    "suv", "sedan", "sedán", "pickup", "coupé", "crossover", "hatchback",
    # Brand/model extras not always in CAR_BRANDS (short names we dropped)
    "vw", "mg", "byd", "jac", "gm",
    "ioniq", "uncharted", "trailseeker", "idbuzz", "idbuz",
    # Automotive jargon
    "awd", "4wd", "rwd", "fwd", "abs", "tpms", "cvt", "dsg",
    "pov", "dueño", "dueña",
)

# Force-correction rules — applied BEFORE spellchecker. Maps raw OCR → canonical
# domain term when the pattern is unambiguous enough that spellcheck suggestion
# ranking would be unreliable (e.g., when distance to a second domain word is close).
_FORCED_DOMAIN_CORRECTIONS: dict[str, str] = {
    "cadillae": "cadillac",
    "cadilac": "cadillac",
    "hyundia": "hyundai",
    "hyunday": "hyundai",
    "toyoto": "toyota",
    "tayota": "toyota",
    "hbrida": "híbrida",
    "hbridas": "híbridas",
    "hbrido": "híbrido",
    "hbridos": "híbridos",
    "hbrda": "híbrida",
    "idbu2": "idbuzz",
    "idbuz": "idbuzz",
    "idbuzz": "id buzz",
    "electpicos": "eléctricos",
    "electrpcos": "eléctricos",
    "palsade": "palisade",
    "foresteer": "forester",
    "dodgi": "dodge",
    "fore5ter": "forester",
}

_NONWORD_RE = re.compile(r"[^\w\s\-']", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _domain_words() -> set[str]:
    """Full set of domain-known words — should never be "corrected" away."""
    words: set[str] = set()
    for v in (CAR_BRANDS, CAR_MODELS, REVIEW_KEYWORDS, _EXTRA_DOMAIN_WORDS):
        for w in v:
            words.add(w.lower())
            # For multi-word items, also add the space-less form ("santa fe" → "santafe")
            if " " in w:
                words.add(w.lower().replace(" ", ""))
    return words


@lru_cache(maxsize=1)
def _spellchecker():
    if not _HAS_SPELLCHECKER:
        return None
    sc = SpellChecker(language='es', case_sensitive=False)
    # Teach it our domain vocab so "subaru" doesn't get "corrected" to "subir"
    sc.word_frequency.load_words(list(_domain_words()))
    return sc


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _match_case(original: str, corrected: str) -> str:
    """Carry over the original word's casing pattern onto the correction."""
    if original.isupper():
        return corrected.upper()
    if original.istitle():
        return corrected.title()
    return corrected


def correct_word(word: str, min_similarity: float = 0.82) -> str:
    """Return a Spanish-diacritic-restored or OCR-corrected version of a single word.

    Rules (applied in order; first match wins):
      1. Forced correction rule fires → apply (strongest signal for known OCR → domain)
      2. Word is in domain vocab → unchanged (including plurals, jargon)
      3. Word is < 3 chars → unchanged (too short to correct reliably)
      4. Exists in Spanish dictionary exactly → unchanged
      5. Has a correction close enough (ratio >= min_similarity AND within 1 edit of domain
         OR strictly closer than runner-up) → apply, preserve casing
      6. Otherwise → unchanged (don't hallucinate)
    """
    if not word or len(word) < 3:
        return word
    low = word.lower()

    forced = _FORCED_DOMAIN_CORRECTIONS.get(low)
    if forced is not None:
        return _match_case(word, forced)

    if low in _domain_words():
        return word

    sc = _spellchecker()
    if sc is None:
        return word

    if low in sc.word_frequency.dictionary:
        return word

    # First: is there a domain word within edit distance 2? If so prefer it over general-dict.
    domain_candidates = [
        (d, _similar(low, d)) for d in _domain_words()
        if abs(len(d) - len(low)) <= 2 and _similar(low, d) >= min_similarity
    ]
    if domain_candidates:
        best_domain, best_score = max(domain_candidates, key=lambda x: x[1])
        # Check no runner-up is too close (ambiguous domain match)
        scored = sorted(domain_candidates, key=lambda x: -x[1])
        if len(scored) == 1 or scored[1][1] <= best_score - 0.04:
            return _match_case(word, best_domain)

    candidates = sc.candidates(low)
    if not candidates:
        return word

    scored = sorted(
        ((cand, _similar(low, cand.lower())) for cand in candidates),
        key=lambda x: -x[1],
    )
    best, best_score = scored[0]
    if best_score < min_similarity:
        return word

    if len(scored) > 1 and scored[1][1] >= best_score - 0.03:
        return word

    return _match_case(word, best)


def correct_text(text: str) -> str:
    """Apply correct_word to each whitespace-separated token; preserve punctuation."""
    if not text:
        return text
    sc = _spellchecker()
    if sc is None:
        return text

    # Tokenize keeping punctuation boundaries
    tokens = re.findall(r"[\wáéíóúñüÁÉÍÓÚÑÜ'-]+|[^\s\w]", text, re.UNICODE)
    corrected_parts: list[str] = []
    for tok in tokens:
        if re.match(r"[\wáéíóúñüÁÉÍÓÚÑÜ'-]+", tok):
            corrected_parts.append(correct_word(tok))
        else:
            corrected_parts.append(tok)

    # Rejoin with single spaces (close to original whitespace — exact whitespace preservation
    # isn't needed for downstream entity extraction)
    return " ".join(corrected_parts)
