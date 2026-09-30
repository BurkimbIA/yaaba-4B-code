"""Unicode normalization shared by every path that compares two texts.

One table, one place. Three diverging copies once let seven held-out verses stay
in training because `«` was missing from all of them.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Attested confusables plus typographic variants. Hyphens are unified but never
# removed: 47.4 % of Moore texts carry a compound word. Tone marks are
# combining and meaning-bearing, so NFC attaches them rather than stripping.
CONFUSABLES: Final = str.maketrans({
    "ű": "ũ", "û": "ũ", "ɑ": "a", "ɡ": "g", "ʊ": "ʋ", "ʉ": "ʋ", "ε": "ɛ", "ι": "ɩ",
    "‐": "-", "‑": "-", "–": "-", "—": "-",
    "‘": "’", "‛": "’", "′": "’",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "″": '"',
    " ": " ", " ": " ", " ": " ",
})

_WHITESPACE: Final = re.compile(r"\s+")
_NON_WORD: Final = re.compile(r"[^\w]", re.UNICODE)


def clean(text: str) -> str:
    """NFC, confusables, collapsed whitespace."""
    folded = unicodedata.normalize("NFC", str(text)).translate(CONFUSABLES)
    return _WHITESPACE.sub(" ", folded).strip()


def key(text: str) -> str:
    """Relaxed comparison key: case, punctuation and whitespace erased.

    Use to *verify* a removal, never to perform one.
    """
    folded = unicodedata.normalize("NFC", str(text)).casefold()
    return _NON_WORD.sub("", folded)
