"""Unicode normalization and Hindi/English text normalizer.

Normalize every passage with Unicode normalization and whitespace cleanup.
Preserve Devanagari text, punctuation, numbers, names, and identifiers.
Add an internal transliteration field for queries typed in Latin characters,
but keep the original dataset text as the authoritative passage.

Per HHGOA Task 2 architecture document.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Compiled patterns (fast, module-level)
# ---------------------------------------------------------------------------

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Collapse multiple whitespace runs (preserves newlines as spaces)
_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")

# Strip leading/trailing junk that is NOT Devanagari, Latin, digits, or sentence-ending punctuation
_LEADING_JUNK_RE = re.compile(r"^[^\w\u0900-\u097F]+")
_TRAILING_JUNK_RE = re.compile(r"[^\w\u0900-\u097F।?!]+$")

# Latin-only words for transliteration alias extraction
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]{2,}")

# Simple Hinglish/transliteration alias map (common Indian proper nouns in Roman script)
_TRANSLIT_ALIASES: dict[str, str] = {
    "india": "भारत",
    "bharat": "भारत",
    "delhi": "दिल्ली",
    "mumbai": "मुंबई",
    "hindi": "हिंदी",
    "bengaluru": "बेंगलुरु",
    "bangalore": "बेंगलुरु",
    "kolkata": "कोलकाता",
    "chennai": "चेन्नई",
    "hyderabad": "हैदराबाद",
    "rajasthan": "राजस्थान",
    "gujarat": "गुजरात",
}


def normalize_text(text: str, preserve_devanagari: bool = True) -> str:
    """Apply NFC Unicode normalization and whitespace cleanup to text.

    Parameters
    ----------
    text:
        Raw passage or query text (Hindi, English, or code-mixed).
    preserve_devanagari:
        If True, Devanagari punctuation (।, ॥) and numerals are preserved.
        Always True for corpus passages; may be False for diagnostic use only.

    Returns
    -------
    str
        NFC-normalized, whitespace-collapsed, stripped text. Original dataset
        content is preserved; only normalization form and whitespace change.
    """
    if not text:
        return ""

    # Step 1: NFC normalization (canonical decomposition then canonical composition)
    normalized = unicodedata.normalize("NFC", text)

    # Step 2: Collapse whitespace runs to single space
    normalized = _WHITESPACE_RE.sub(" ", normalized)

    # Step 3: Strip leading/trailing whitespace only (never strip Devanagari content)
    normalized = normalized.strip()

    return normalized


def normalize_for_bm25(text: str) -> str:
    """Normalize text specifically for BM25 tokenization.

    Applies NFC normalization, lowercases Latin text, and strips characters
    that would generate spurious tokens. Devanagari content is NFC-normalized
    but NOT lowercased (Devanagari has no case). Numerals and identifiers are
    preserved.

    Parameters
    ----------
    text:
        Raw chunk text or query text.

    Returns
    -------
    str
        BM25-ready normalized string ready for `.split()` tokenization.
    """
    if not text:
        return ""

    # NFC first
    out = unicodedata.normalize("NFC", text)

    # Lowercase only ASCII/Latin characters; Devanagari has no case
    out = _lowercase_latin(out)

    # Collapse whitespace
    out = _WHITESPACE_RE.sub(" ", out)
    out = out.strip()

    return out


def _lowercase_latin(text: str) -> str:
    """Lowercase only Latin letters in a mixed-script string."""
    result = []
    for ch in text:
        cp = ord(ch)
        # ASCII letters
        if 0x41 <= cp <= 0x5A:
            result.append(chr(cp + 32))
        # Extended Latin (U+00C0–U+024F common Latin Extended)
        elif 0x00C0 <= cp <= 0x024F:
            result.append(ch.lower())
        else:
            result.append(ch)
    return "".join(result)


def tokenize_for_bm25(text: str) -> list[str]:
    """Normalize and tokenize text for BM25 index and query matching.

    Strips Devanagari sentence-end punctuation (।॥) and other non-word
    separators before splitting, so they do not become standalone tokens.
    Preserves Devanagari words, Hindi numerals, and Latin words as tokens.

    Parameters
    ----------
    text:
        Raw text to tokenize.

    Returns
    -------
    list[str]
        Token list ready for BM25Okapi or inverted index.
    """
    if not text:
        return []

    normalized = normalize_for_bm25(text)

    # Replace Devanagari sentence-end and common punctuation with spaces
    # so they don't attach to words as spurious suffixes
    normalized = normalized.replace("।", " ").replace("॥", " ")
    normalized = re.sub(r"[,\"'()[\]{}<>!?;:]+", " ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    tokens = [t for t in normalized.split() if t]
    return tokens


def detect_script(text: str) -> str:
    """Detect the primary script of a text string.

    Returns
    -------
    str
        One of: ``"hi"`` (Devanagari-dominant), ``"en"`` (Latin-dominant),
        ``"mixed"`` (significant presence of both).
    """
    if not text:
        return "en"

    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total = devanagari_chars + latin_chars

    if total == 0:
        return "en"

    devanagari_ratio = devanagari_chars / total
    if devanagari_ratio >= 0.7:
        return "hi"
    elif devanagari_ratio <= 0.3:
        return "en"
    else:
        return "mixed"


def build_transliteration_aliases(query: str) -> list[str]:
    """Extract Devanagari equivalents for recognized Latin words in a query.

    Used to expand BM25 query tokens with Hindi aliases so that a user typing
    ``"india ki rajdhani"`` can still match Hindi passages containing ``भारत``.

    Parameters
    ----------
    query:
        Raw query text (may be Hinglish / Latin-script Hindi).

    Returns
    -------
    list[str]
        Extra Hindi tokens to add to the BM25 query token list.
    """
    extra_tokens: list[str] = []
    latin_words = _LATIN_WORD_RE.findall(query.lower())
    for word in latin_words:
        hindi = _TRANSLIT_ALIASES.get(word)
        if hindi:
            extra_tokens.append(hindi)
    return extra_tokens
