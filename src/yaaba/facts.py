"""Score a factual answer in one place, so that the scripts cannot disagree.

The probe looks for strings in an answer. Two scripts once scored the same
answer differently: one tested the trap first, the other the expected answer.
Neither order works for every answer (X is expected, Y is the trap):

    "X covers more than Y"   right; mentions Y to rule it out
    "X, no, Y"               trapped; mentions X, then retracts it

A substring match cannot tell a rebuttal from a retraction, so an answer
holding both strings gets its own verdict, `ambigu`, and a human reads it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# These values are stored in evaluation JSON already on disk and read back by
# `compare`, so they stay in French like the other stored keys.
RIGHT: Final = "juste"
TRAPPED: Final = "piege"        # it learned something else
UNCLEAR: Final = "ambigu"       # both strings present; a matcher cannot tell
UNKNOWN: Final = "faux"         # it does not know

VERDICTS: Final = (RIGHT, TRAPPED, UNCLEAR, UNKNOWN)

# One budget for every script. 120 and 128 tokens cut step-by-step arithmetic
# before the result.
BUDGET: Final = 320


def budget_of(point: dict) -> int:
    """The token budget a measured point's answers were generated under.

    Read from `budget_faits` when present. The earliest points lack it and are
    dated by their fields: `complet` came with the 128-token budget, and a
    point without it was scored at 48.
    """
    if "budget_faits" in point:
        return int(point["budget_faits"])
    return 128 if point.get("faits") and "complet" in point["faits"][0] else 48


def answers_the_probe(point: dict, probe: list[dict], budget: int = BUDGET) -> bool:
    """Whether a measured point already answered this probe, at this budget.

    A notebook that re-answers the facts skips the points that are up to date.
    Testing the budget alone would skip every point after the probe grew from
    20 to 52 items. Same rule as `evaluate_checkpoint.same_fact_budget`.
    """
    if point.get("budget_faits") != budget:
        return False
    return {f["id"] for f in point.get("faits", ())} == {f["id"] for f in probe}


def normalise(text: str) -> str:
    """Casefold, and unify the apostrophes that split one word in two.

    Models write both `’` and `'`, sometimes within one run. Accents are kept:
    in French `piege` and `piège` are different words, and the expected lists
    carry accents.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    for fancy in ("’", "ʼ", "‘"):
        text = text.replace(fancy, "'")
    return text


def contains(needle: str, haystack: str) -> bool:
    """Match on word boundaries. A trailing `*` in `needle` leaves the end open.

    A bare substring match let `no` match inside `nord`, and `12` inside `120`.
    The star is plain data, not a regex: `abc*` matches `abc` and `abcdef`.
    """
    stem, prefix = (needle[:-1], True) if needle.endswith("*") else (needle, False)
    pattern = r"\b" + re.escape(normalise(stem)) + ("" if prefix else r"\b")
    return re.search(pattern, haystack) is not None


def score(answer: str, fact: dict, complete: bool = True) -> str:
    """One of `VERDICTS`, from the `attendu` and `piege` lists of a fact.

    `complete` is False when generation hit its token budget. A cut answer that
    has not reached the expected string is scored `ambigu` rather than wrong,
    so a long answer is not counted as forgetting (one answer was cut at "56",
    one character before "563").
    """
    lowered = normalise(answer)
    right = any(contains(good, lowered) for good in fact["attendu"])
    trapped = any(contains(trap, lowered) for trap in fact.get("piege", ()))
    if right and trapped:
        return UNCLEAR
    if right:
        return RIGHT
    if trapped:
        return TRAPPED
    return UNKNOWN if complete else UNCLEAR
