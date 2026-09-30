"""Paired bootstrap on FLORES+ devtest chrF, from the stored hypotheses. No GPU.

Resamples the 1,012 sentences with one shared index per draw, so both systems see
the same resample, and recomputes corpus chrF from summed sentence statistics.

    python scripts/measure/flores_bootstrap.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "measure"))
EVAL = ROOT / "evaluation" / "flores"
DRAWS = 1000
COMPARISONS = [("P2_sft", "A_sft"), ("P2_sft", "B_sft"), ("P2_sft", "C_sft"),
               ("P2_sft", "base_sft"), ("A_sft", "B_sft"), ("A_sft", "C_sft"),
               ("C_sft", "B_sft"), ("yaaba-4b", "base_sft-651"), ("yaaba-4b", "qwen3-4b"),
               ("yaaba-4b", "nllb-200-600m")]
CHRF_ = CHRF()


def sentence_stats(hypotheses: list[str], references: list[str]) -> np.ndarray:
    return np.array(CHRF_._extract_corpus_statistics(hypotheses, [references]), dtype=float)


def corpus_chrf(stats: np.ndarray) -> float:
    return CHRF_._compute_f_score(list(stats.sum(0)))


def interval(a: np.ndarray, b: np.ndarray, draws: list[np.ndarray]) -> tuple[float, float, float]:
    """Difference a - b and its 95% paired interval."""
    diffs = sorted(corpus_chrf(a[i]) - corpus_chrf(b[i]) for i in draws)
    return corpus_chrf(a) - corpus_chrf(b), diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def main() -> int:
    import flores

    pairs = flores.load_devtest()
    references = {"french_to_moore": [m for _, m in pairs],
                  "moore_to_french": [f for f, _ in pairs]}
    rng = random.Random(20260825)
    draws = [np.array([rng.randrange(len(pairs)) for _ in pairs]) for _ in range(DRAWS)]
    for direction, refs in references.items():
        print(direction)
        stats = {}
        for name in {n for pair in COMPARISONS for n in pair}:
            result = json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))
            stats[name] = sentence_stats(result[direction]["hypotheses"], refs)
        for a, b in COMPARISONS:
            diff, lo, hi = interval(stats[a], stats[b], draws)
            print(f"  {a:9s} - {b:13s} {diff:+6.2f}  [{lo:+.2f}, {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
