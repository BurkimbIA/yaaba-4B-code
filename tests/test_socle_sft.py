"""The start-point guard, on the defect that produced numbers instead of an error.

`A + SFT` was once measured at 4.7883 bits per character of Moore, worse than
the base and four times the 1.2759 of `A` alone. The model had not regressed:
the SFT adapter, trained on `base + CPT`, had been put on the bare base. No
error was raised, and all five arms had to be measured again.

These tests touch neither GPU nor network: the guard sits BEFORE the torch
import on purpose, so it can be tested and fails fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "measure"))

import evaluate_checkpoint as ec  # noqa: E402


@pytest.mark.parametrize("name", [
    "A-6005-checkpoint-1191", "C-4820-checkpoint-1191",
    "C-6025-checkpoint-1191", "P2-6120-checkpoint-1191",
    "B-2755-checkpoint-1191",
])
def test_an_arm_point_without_its_start_refuses(name):
    with pytest.raises(SystemExit, match="socle"):
        ec.load(name)


@pytest.mark.parametrize("name", [
    "base", "checkpoint-6005", "checkpoint-1191", "base-checkpoint-1191",
])
def test_what_is_not_an_arm_point_passes_the_guard(name):
    """`checkpoint-6005` is a CPT point, rightly measured on the base.

    `base-checkpoint-1191` is the SFT control: its start IS the bare base, and
    it was the only one of the seven measurements that was right.
    """
    assert not ec.POINT_DE_BRAS.match(name), name
