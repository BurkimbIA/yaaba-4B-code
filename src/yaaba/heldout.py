"""Keeping evaluation data out of training.

Texts are compared on relaxed keys (case and whitespace folded). A filter that
compared raw text with whitespace-collapsed text never matched, and 399
evaluation documents stayed in the mixture. Removal can be exact, but the
check has to be wider than the removal.
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
    """Held-out texts by capability, stored as relaxed keys, so that a caller cannot
    compare them unrelaxed.
    """
    by_capability: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_capability.setdefault(record["capacite"], set()).add(key(record["text"]))
    return by_capability


# Text that belongs to no single part, such as an SFT turn, is checked against
# every capability. Any other unknown part is rejected: treating a typo as an
# empty target set silently disables the leak detector.
EVERYTHING: Final = "*"


class Guard:
    """Leak detector for one mixture part, or for `EVERYTHING`."""

    __slots__ = ("forbidden", "long")

    def __init__(self, part: str, held_out: dict[str, set[str]]) -> None:
        if part == EVERYTHING:
            capabilities = tuple(held_out)
        else:
            try:
                capabilities = TARGETS[part]
            except KeyError as error:
                known = ", ".join(sorted(TARGETS))
                raise ValueError(
                    f"unknown mixture part {part!r}; refusing to skip the "
                    f"held-out check (known parts: {known})"
                ) from error
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
