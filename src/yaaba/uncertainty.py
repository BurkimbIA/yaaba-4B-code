"""How much of a bits-per-character gap is real, and how much is the sample.

Comparing arms means comparing a handful of numbers. Without an interval, a gap
of 0.01 bits per character and a gap of 0.5 look alike.

Bits per character is a ratio of two sums, not a mean:

    bpc = sum(nll over texts) / ln(2) / sum(characters over texts)

The standard error of a mean does not apply, and the two sums are dependent,
since a long text adds to both. Resampling the `(nll, chars)` pairs together
and recomputing the ratio needs no distributional assumption and keeps that
dependence.

Two checkpoints are measured on the same held-out texts. The texts one model
finds hard, the other usually finds hard too, so comparing two separate
intervals keeps that shared difficulty as noise. Resampling the same text
indices for both cancels it. Two intervals can therefore overlap while the
paired difference excludes zero; `difference` is the one to decide with.

The interval covers sampling noise on 200 texts per capability and nothing
else: it does not see a leaked held-out text, a capability whose texts do not
match its name, or a tokenizer effect. It is a lower bound on the uncertainty.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Final, NamedTuple

# Fixed, so a recomputed interval matches the one already reported.
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

    The `(nll, chars)` pairs are resampled together, keeping the link between
    a text's likelihood and its length.
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

    Negative means `after` is better, since bits per character is a cost. An
    interval that contains zero means these texts cannot resolve the
    difference; it does not show that there is none.
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
        # The same indices on both sides make the comparison paired.
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

    For reporting only: overlapping intervals can hide a paired difference that
    excludes zero. Use `difference` to decide.
    """
    return first.low <= second.high and second.low <= first.high
