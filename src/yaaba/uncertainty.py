"""How much of a bits-per-character gap is real, and how much is the sample.

Step 07 has to answer "which arm is better" from four numbers. Without an
interval around each one, a gap of 0.01 bpc and a gap of 0.5 bpc look alike, and
the answer is whatever the reader already believed.

## Why bootstrap and not a standard error

Bits per character is a **ratio of two sums**, not a mean:

    bpc = sum(nll over texts) / ln(2) / sum(characters over texts)

The usual standard error of a mean does not apply, and the two sums are not
independent: a long text contributes to both. Resampling the `(nll, chars)`
pairs together and recomputing the ratio makes no distributional assumption and
gets the dependence right for free.

## Why the comparison is paired, and why that is not a detail

Two checkpoints are measured on **the same held-out texts**. Comparing them
through two independent intervals throws that away: the texts a model finds hard
are the texts the other model finds hard too, and that shared difficulty is
noise common to both. Resampling the same text indices for both cancels it.

The practical consequence is large. Two intervals can overlap while the paired
difference is nowhere near zero, so reading overlap as "no difference" is a
mistake this module exists to prevent. `overlap` reports both, and the paired
one is the one that decides.

## What this cannot do

It measures sampling noise on 200 texts per capability, nothing else. It does
not see a held-out set that leaked, a capability whose texts are unlike what the
name suggests, or a tokenizer difference. An interval is a floor on the
uncertainty, never a certificate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Final, NamedTuple

# Fixed, so an interval recomputed tomorrow is the interval quoted today.
SEED: Final = 20260831
RESAMPLES: Final = 2000
LEVEL: Final = 0.95


class Interval(NamedTuple):
    """A value and the range the sample supports for it."""

    value: float
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def __str__(self) -> str:
        return f"{self.value:.4f} +/- {self.half_width:.4f}"


def bits_per_character(per_text: Iterable[tuple[float, int]]) -> float:
    """The ratio of sums, in bits: total negative log-likelihood over characters.

    `per_text` holds `(nll_in_nats, characters)` for each text.
    """
    total_nll = total_chars = 0.0
    for nll, chars in per_text:
        total_nll += nll
        total_chars += chars
    return total_nll / math.log(2) / total_chars if total_chars else float("nan")


def interval(per_text: list[tuple[float, int]], resamples: int = RESAMPLES,
             level: float = LEVEL) -> Interval:
    """Percentile bootstrap around one measurement.

    The pairs are resampled together: splitting them would break the link
    between a text's likelihood and its length, which is the whole reason the
    ratio is not a mean.
    """
    point = bits_per_character(per_text)
    if len(per_text) < 2:
        return Interval(point, point, point)

    draw = random.Random(SEED)
    size = len(per_text)
    values = []
    for _ in range(resamples):
        sample = [per_text[draw.randrange(size)] for _ in range(size)]
        values.append(bits_per_character(sample))
    values.sort()
    tail = (1 - level) / 2
    return Interval(point,
                    values[int(tail * resamples)],
                    values[min(int((1 - tail) * resamples), resamples - 1)])


def difference(before: list[tuple[float, int]], after: list[tuple[float, int]],
               resamples: int = RESAMPLES, level: float = LEVEL) -> Interval:
    """Paired bootstrap of `after - before`, on the same texts in the same order.

    Negative means `after` is better, since bits per character is a cost.

    An interval that excludes zero is a difference the sample supports. One that
    contains it is not "no difference": it is a difference this many texts cannot
    resolve, which is a statement about the measurement and not about the models.
    """
    if len(before) != len(after):
        raise ValueError(
            f"paired comparison needs the same texts: {len(before)} vs {len(after)}")

    point = bits_per_character(after) - bits_per_character(before)
    if len(before) < 2:
        return Interval(point, point, point)

    draw = random.Random(SEED)
    size = len(before)
    values = []
    for _ in range(resamples):
        # The same indices on both sides: that is what makes it paired.
        picked = [draw.randrange(size) for _ in range(size)]
        values.append(bits_per_character([after[i] for i in picked])
                      - bits_per_character([before[i] for i in picked]))
    values.sort()
    tail = (1 - level) / 2
    return Interval(point,
                    values[int(tail * resamples)],
                    values[min(int((1 - tail) * resamples), resamples - 1)])


def overlap(first: Interval, second: Interval) -> bool:
    """Whether two independent intervals overlap.

    Reported for what it is and never used to conclude: two overlapping
    intervals routinely hide a paired difference that excludes zero, because the
    shared difficulty of the texts inflates both. Use `difference` to decide.
    """
    return first.low <= second.high and second.low <= first.high
