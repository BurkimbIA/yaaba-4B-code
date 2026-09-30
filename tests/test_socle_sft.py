"""The guard that refuses an SFT point loaded without its CPT adapter.

`A + SFT` was once measured at 4.7883 Moore bits per character, against 1.2759
for `A` alone: the SFT adapter, trained on `base + CPT`, had been loaded on the
bare base, and no error was raised.

The guard runs before torch is imported, so these tests need no GPU or network.
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
    """`checkpoint-6005` is a CPT point and is measured on the base.

    `base-checkpoint-1191` is the SFT control, whose start is the bare base.
    """
    assert not ec.POINT_DE_BRAS.match(name), name
