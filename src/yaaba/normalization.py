"""Unicode normalization shared by every path that compares two texts.

It is defined once because separate copies drifted: seven held-out verses
stayed in training when `«` was missing from all three copies of the table.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Confusables seen in the corpus, plus typographic variants. Hyphens are
# unified and kept, since 47.4 % of Moore texts contain a compound word. Tone
# marks carry meaning; NFC attaches them to their letter.
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

    For checking a removal. Removal itself uses exact text.
    """
    folded = unicodedata.normalize("NFC", str(text)).casefold()
    return _NON_WORD.sub("", folded)
