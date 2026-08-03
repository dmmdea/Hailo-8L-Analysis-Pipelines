"""symspellpy-backed OCR correction with gated rules — Phase 7 replacement
for the forced-table approach in ocr_correction.py.

Why symspellpy beats pyspellchecker here:
 - O(1) lookup vs O(n) — pyspellchecker's slow on a 1.2M-word dictionary,
   symspellpy is ~1000× faster via its symmetric-delete hash index
 - `transfer_casing=True` restores original case on the corrected token, so
   "ELECTRICOS" → "ELÉCTRICOS" retains uppercase
 - Custom-dict boost: automotive brands + models merged at frequency=1e9
   dominate all natural-language competitors, preserving them verbatim

Gated correction:
 1. Year-shaped tokens (19XX / 20XX) — skip (OCR digit confusion elsewhere
    is handled by the entity extractor's year-substitution table).
 2. Tokens exactly matching CAR_BRANDS or CAR_MODELS (lowercased) — skip.
    Prevents symspell from rewriting "IONIQ" into a dictionary word.
 3. Tokens shorter than 3 chars — skip. Two-letter tokens are mostly
    prepositions/abbreviations; symspell's edit-distance-1 rewrites them
    to high-freq words like "de" that destroy information.
 4. Tokens that are purely numeric or contain $/€/USD/COP — skip.
    Price/currency shouldn't be "corrected".
 5. Everything else: symspell lookup with max_edit_distance=1,
    Verbosity.CLOSEST, transfer_casing=True. If no suggestion within
    distance 1, leave token untouched.

This is the first of the Phase 7+ changes that care about preserving raw
OCR signal through correction — a "do no harm" policy that lets Phase 6
LM rescoring and downstream fuzzy entity matching work against clean-ish
text instead of over-corrected nonsense.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

from vision_shared.features.title import CAR_BRANDS, CAR_MODELS
from vision_shared.features.ocr_entities import ALL_KEYWORD_CATEGORIES

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FREQ_DICT_PATH = _DATA_DIR / "es_full.txt"

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_ALL_DIGITS_RE = re.compile(r"^\d+$")
_CURRENCY_RE = re.compile(r"[\$€¢£¥]|USD|COP|EUR|MXN|COL\$|US\$", re.IGNORECASE)
_TOKENIZE_RE = re.compile(r"(\s+|[^\w\s]+)", re.UNICODE)

def _collect_keywords() -> list[str]:
    acc: list[str] = []
    for cat, words in ALL_KEYWORD_CATEGORIES.items():
        acc.extend(words)
    return acc


_DOMAIN_TERMS_LOWER: frozenset[str] = frozenset(
    list(CAR_BRANDS) + list(CAR_MODELS) + _collect_keywords() + [
        # extra terms that live outside the canonical vocab but appear on
        # auto-review thumbnails and must not be rewritten by symspell
        "rav4", "cr-v", "crv", "f-150", "f150", "id", "buzz", "idbuzz",
        # English loanwords Spanish auto YouTubers use verbatim
        "review", "test", "drive", "new", "vs", "versus",
        # common powertrain/category shorthand not already in the keyword vocab
        "gasolina", "gasolinas", "diesel", "electrico", "electrica",
        "electricos", "electricas",
    ]
)


@functools.lru_cache(maxsize=1)
def _get_domain_sym_spell() -> Any:
    """Second SymSpell instance holding ONLY domain vocabulary. Used to probe
    whether a noisy OCR token is a plausible mis-spelling of a domain term
    (e.g. "GASONA" within edit-distance 2 of "gasolina"); in that case we
    skip the main correction so the downstream entity extractor's fuzzy
    matcher keeps its signal."""
    from symspellpy import SymSpell  # type: ignore[import-untyped]
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for term in _DOMAIN_TERMS_LOWER:
        sym.create_dictionary_entry(term, 1_000_000_000)
    return sym


@functools.lru_cache(maxsize=1)
def _get_sym_spell() -> Any:
    """Lazy-build the SymSpell instance + load Spanish frequency list.

    Domain terms (CAR_BRANDS + CAR_MODELS) get frequency 1e9 so they
    outweigh any natural-language alternative and stay verbatim through
    correction. The base list is hermitdave/FrequencyWords 2018 Spanish
    (1.2M entries).
    """
    from symspellpy import SymSpell  # type: ignore[import-untyped]
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    if _FREQ_DICT_PATH.exists():
        ok = sym.load_dictionary(str(_FREQ_DICT_PATH), term_index=0, count_index=1, encoding="utf-8")
        if not ok:
            raise RuntimeError(f"symspellpy could not load {_FREQ_DICT_PATH}")

    # Boost domain terms so they dominate corrections
    for term in _DOMAIN_TERMS_LOWER:
        sym.create_dictionary_entry(term, 1_000_000_000)
    return sym


def _should_skip(token: str) -> bool:
    """Return True if this token must not be touched by correction.

    The fuzzy-domain branch is the critical "do no harm" gate: if the token
    is within edit-distance 2 of any CAR_BRAND / CAR_MODEL / keyword, we
    preserve it verbatim so the downstream entity extractor's fuzzy matcher
    can still lock onto it. Without this guard, symspell's 1-edit rewrites
    common tokens like "GASONA" → "CASONA" (regular Spanish word) and
    destroy the evidence for GASOLINA (edit-distance 2 from the noisy
    form). Iter 4 tried forcing domain corrections instead, but that
    produced model false positives by over-rewriting noise.
    """
    if len(token) < 3:
        return True
    if _YEAR_RE.match(token):
        return True
    if _ALL_DIGITS_RE.match(token):
        return True
    if _CURRENCY_RE.search(token):
        return True
    lower = token.lower()
    if lower in _DOMAIN_TERMS_LOWER:
        return True
    from symspellpy import Verbosity  # type: ignore[import-untyped]
    try:
        hits = _get_domain_sym_spell().lookup(
            lower, Verbosity.CLOSEST, max_edit_distance=2,
        )
    except Exception:
        hits = []
    if hits:
        return True
    return False


def _correct_token(sym: Any, token: str) -> str:
    """Return the single best correction for `token`, or `token` unchanged if
    no suggestion is within edit distance 1. Tokens close to domain terms
    are filtered out before this point by _should_skip()."""
    from symspellpy import Verbosity  # type: ignore[import-untyped]

    suggestions = sym.lookup(
        token, Verbosity.CLOSEST, max_edit_distance=1, transfer_casing=True,
    )
    if not suggestions:
        return token
    return suggestions[0].term


def correct_text(text: str) -> str:
    """Token-by-token correction of `text`. Whitespace + punctuation preserved
    verbatim; only word-shaped tokens are candidates for correction.
    """
    if not text:
        return text
    sym = _get_sym_spell()
    # Tokenize — keep delimiters so we can reassemble with original spacing
    parts = _TOKENIZE_RE.split(text)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.isspace() or not any(c.isalpha() for c in part):
            out.append(part)
            continue
        if _should_skip(part):
            out.append(part)
            continue
        out.append(_correct_token(sym, part))
    return "".join(out)
