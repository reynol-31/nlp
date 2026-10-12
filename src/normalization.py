"""Unicode normalisation for Hindi (Devanagari) word forms.

Every string that takes part in matching -- user input, lexicon lemmas, suffix
rules and gold-standard annotations -- goes through ``normalize_word`` so that
both analyzers and the evaluator see exactly the same symbols.

Steps (in order):

1. Unicode NFC.  Precomposed nukta letters such as U+095B (ज़) are composition
   exclusions, so NFC rewrites them to base letter + U+093C NUKTA.  After this
   step "दरवाज़ा" typed either way is the same code-point sequence.
2. Remove invisible format characters (ZWJ, ZWNJ, ZWSP, BOM, soft hyphen).
3. Strip surrounding whitespace and punctuation (including danda । and ॥).
4. Optional nukta folding for the Perso-Arabic letters क़ ख़ ग़ ज़ फ़, which
   writers frequently omit ("खराब" vs "ख़राब").  ड़ and ढ़ are *not* folded:
   they are distinct native phonemes and are written consistently.
5. Optional nasal folding: chandrabindu (ँ) -> anusvara (ं).  Everyday Hindi
   text uses both for the same suffixes (लड़कियाँ / लड़कियां).  This is lossy
   for rare minimal pairs (हँस / हंस) -- none occur in our lexicon.
6. Optional y-glide folding: after a vowel (including a consonant's inherent
   vowel), "ये/यी" -> "ए/ई" (खाये -> खाए, नयी -> नई, गये -> गए).  Both spellings are common; the Central Hindi
   Directorate standard prefers ए/ई.
"""

from __future__ import annotations

import re
import unicodedata

NUKTA = "\u093c"
CHANDRABINDU = "\u0901"
ANUSVARA = "\u0902"
VIRAMA = "\u094d"

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_EDGE_PUNCT = " \t\r\n\u00a0\u0964\u0965.,!?;:\"'()[]{}<>\u2026\u2013\u2014-\u2018\u2019\u201c\u201d"
_FOLDABLE_NUKTA = re.compile("([\u0915\u0916\u0917\u091c\u092b])" + NUKTA)  # क ख ग ज फ
# preceded by a vowel sign (ा..ौ, not virama), an independent vowel letter, or a
# consonant / nukta carrying the inherent vowel (नयी -> नई); never after virama (क्यों)
_Y_GLIDE = re.compile("(?<=[\u093c\u093e-\u094c\u0905-\u0914\u0915-\u0939])\u092f([\u0947\u0940])")  # य + े/ी
_Y_GLIDE_MAP = {"\u0947": "\u090f", "\u0940": "\u0908"}  # े->ए, ी->ई

DEVANAGARI_RANGE = (0x0900, 0x097F)


def normalize_word(
    text: str,
    *,
    fold_nukta: bool = True,
    fold_nasal: bool = True,
    fold_y_glide: bool = True,
) -> str:
    """Return the canonical matching form of a single Hindi word."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.translate(_INVISIBLE)
    s = s.strip(_EDGE_PUNCT)
    if fold_nukta:
        s = _FOLDABLE_NUKTA.sub(r"\1", s)
    if fold_nasal:
        s = s.replace(CHANDRABINDU, ANUSVARA)
    if fold_y_glide:
        s = _Y_GLIDE.sub(lambda m: _Y_GLIDE_MAP[m.group(1)], s)
    return s


def is_devanagari_word(text: str) -> bool:
    """True if *text* is non-empty and every character is in the Devanagari block."""
    if not text:
        return False
    lo, hi = DEVANAGARI_RANGE
    return all(lo <= ord(ch) <= hi for ch in text)


def split_words(text: str) -> list[str]:
    """Split free text into normalised word tokens (whitespace-delimited)."""
    tokens = [normalize_word(t) for t in (text or "").split()]
    return [t for t in tokens if t]


def codepoints(text: str) -> list[str]:
    """Human-readable code-point listing, useful in the UI and in tests."""
    return [f"{ch} U+{ord(ch):04X} {unicodedata.name(ch, '?')}" for ch in text]
