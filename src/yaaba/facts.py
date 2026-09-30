"""Scoring a factual answer, once, so two scripts cannot disagree about it.

The probe looks for a string in an answer. That is coarse, and the coarseness
produced a real contradiction: `evaluate_checkpoint` tested the trap first,
`measure_facts` tested the expected value first, each with a written
justification, and they scored the same answer differently.

Both justifications hold, on different answers:

    "the Pacific Ocean covers more than the Atlantic"   right, mentions the trap
    "Ouagadougou, no, Bobo-Dioulasso"                   trapped, mentions the right one

**No ordering separates them**, because a substring matcher cannot see that one
sentence rules the trap out and the other retracts the right answer. So the
answer here is not an order, it is a fourth outcome: when both strings are
present, say so and let a human look. Picking silently is how a right answer
came to be scored as a regression, and the CPT exit criterion rejects a
checkpoint on exactly that signal.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# The values are data: they are written into evaluation JSON already on disk and
# read back by `compare`, so they stay French like the other stored keys.
RIGHT: Final = "juste"
TRAPPED: Final = "piege"        # it learned something else
UNCLEAR: Final = "ambigu"       # both strings present; a matcher cannot tell
UNKNOWN: Final = "faux"         # it does not know

VERDICTS: Final = (RIGHT, TRAPPED, UNCLEAR, UNKNOWN)

# Lives here for the same reason `score` does: `measure_facts` had raised it
# to 320 and written down why, `evaluate_checkpoint` still passed 128, and the
# same twenty items were answered under two different budgets. 120 cut
# step-by-step arithmetic short; so does 128.
BUDGET: Final = 320


def budget_of(point: dict) -> int:
    """The token budget a measured point's answers were generated under.

    Read from `budget_faits` when it is there. It is not there on the earliest
    points, so those are dated by what they carry:
    `complet` arrived with the 128-token budget, and a point without it was
    scored at 48.

    Lives here rather than in a script because it existed in three copies, one
    of them hardcoded to 128, which is how a re-measured point kept reporting
    the old budget in the exported curve.
    """
    if "budget_faits" in point:
        return int(point["budget_faits"])
    return 128 if point.get("faits") and "complet" in point["faits"][0] else 48


def answers_the_probe(point: dict, probe: list[dict], budget: int = BUDGET) -> bool:
    """Whether a measured point already answered THIS probe, at this budget.

    A notebook that re-answers the facts skips a point it considers up to date,
    and the skip used to test the budget alone. When the probe went from 20 to
    52 items, every stored point still read as up to date
    and the whole re-run would have done nothing, printing success.

    Same rule as `evaluate_checkpoint.same_fact_budget`, and for the same
    reason: a tally is comparable only to a tally taken on the same items at
    the same budget.
    """
    if point.get("budget_faits") != budget:
        return False
    return {f["id"] for f in point.get("faits", ())} == {f["id"] for f in probe}


def normalise(text: str) -> str:
    """Casefold, and fold the apostrophes that split one same word in two.

    `n’a pas d’acces` and `n'a pas d'acces` are the same answer, and the models
    emit both, sometimes inside one run. Accents are kept: in French `piege` and
    `piège` are two words, and the expected lists are written with accents.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    for fancy in ("’", "ʼ", "‘"):
        text = text.replace(fancy, "'")
    return text


def contains(needle: str, haystack: str) -> bool:
    """Word-anchored match. A trailing `*` in `needle` leaves the end free.

    A bare substring was too loose to be right: `no`, expected for `bf04`,
    matched inside `nord` and `notamment` and scored two checkpoints RIGHT on
    answers that never said the Burkina Faso is landlocked. The same looseness
    would let `12` match inside `120`.

    The star is data, not a regex: `enclav*` still catches `enclave` and
    `enclavement`, which is why that list carried a prefix in the first place.
    """
    stem, prefix = (needle[:-1], True) if needle.endswith("*") else (needle, False)
    pattern = r"\b" + re.escape(normalise(stem)) + ("" if prefix else r"\b")
    return re.search(pattern, haystack) is not None


def score(answer: str, fact: dict, complete: bool = True) -> str:
    """One of `VERDICTS`, from the `attendu` and `piege` lists of a fact.

    `complete` is False when generation hit its token budget instead of
    stopping. A cut answer that has not yet said the expected string has not
    said anything either way, and calling that UNKNOWN counts verbosity as
    forgetting. Run A lost `cal03` that way: it answered 1000 - 437 in a LaTeX
    array and was cut at "56", one character short of "563".
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
