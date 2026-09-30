"""Find a data file by name, wherever `data/` currently files it.

`data/` is filed into buckets by state (`source/`, `travail/`, `produit/`,
`attente/`), and a file moves bucket when its state changes. Callers name the
file, never the bucket, so a move does not break them. An absent file raises:
a reader that returns nothing for a missing file rebuilds a smaller corpus
without any error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
DATA: Final = ROOT / "data"
BUCKETS: Final = ("travail", "source", "produit", "attente")


def path(name: str) -> Path:
    """`"raamde.jsonl"` -> `data/travail/raamde.jsonl`.

    Raises rather than returning a path that does not exist: a caller that
    swallows absence turns a moved file into a smaller corpus.
    """
    for bucket in BUCKETS:
        candidate = DATA / bucket / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{name} is in no bucket of data/ ({', '.join(BUCKETS)})")


def find(name: str) -> Path | None:
    """Same, for the callers where absence is a valid answer."""
    return next((p for b in BUCKETS if (p := DATA / b / name).exists()), None)
