"""Keeping evaluation data out of training.

A filter that compared raw text to whitespace-collapsed text ran for weeks
without ever matching, leaving 399 evaluation documents in the mixture.
Removal may be exact; the *check* must be wider than the removal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

from .normalization import key

# Which mixture part is checked against which evaluation capabilities. Parts are
# not crossed with everything: code has nothing to do with Moore, and crossing
# all with all raises false positives on short texts.
TARGETS: Final[dict[str, tuple[str, ...]]] = {
    "moore.jsonl": ("moore_humain", "moore_whisper"),
    "francais.jsonl": ("francais", "francais_parallele"),
    "anglais.jsonl": ("anglais",),
    "code.jsonl": ("code",),
    "maths.jsonl": ("maths",),
    "C/francais.jsonl": ("francais", "francais_parallele"),
    "P/bilingue.jsonl": ("moore_humain", "moore_whisper", "francais",
                         "francais_parallele"),
    "P2/bilingue.jsonl": ("moore_humain", "moore_whisper", "francais",
                          "francais_parallele"),
}

# Below this length a substring match is likely coincidental.
MIN_CONTAINMENT: Final = 40

Leak = Literal["equal", "contained"]


def load(path: Path | str) -> dict[str, set[str]]:
    """Held-out texts by capability, stored as relaxed keys.

    Storing keys rather than texts makes it impossible to forget to relax the
    comparison at call time.
    """
    by_capability: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_capability.setdefault(record["capacite"], set()).add(key(record["text"]))
    return by_capability


# A part that is not in `TARGETS` gets an empty forbidden set, so it is checked
# against nothing and reports clean. That is right for a part with no held-out
# counterpart, and it is also what a typo produces. Pass this instead when the
# text belongs to no part and every capability applies -- an SFT turn, say,
# which is assembled from all of them.
EVERYTHING: Final = "*"


class Guard:
    """Leak detector for one mixture part, or for `EVERYTHING`."""

    __slots__ = ("forbidden", "long")

    def __init__(self, part: str, held_out: dict[str, set[str]]) -> None:
        capabilities = (tuple(held_out) if part == EVERYTHING
                        else TARGETS.get(part, ()))
        self.forbidden = set().union(*(held_out.get(c, set()) for c in capabilities)) \
            if capabilities else set()
        self.long = {k for k in self.forbidden if len(k) >= MIN_CONTAINMENT}

    def inspect(self, text: str) -> Leak | None:
        """Return how `text` leaks, or None. Equality first, then containment."""
        k = key(text)
        if k in self.forbidden:
            return "equal"
        if any(held in k for held in self.long):
            return "contained"
        return None
