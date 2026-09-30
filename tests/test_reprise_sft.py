"""A resume across a change of set must refuse to run.

Both SFT runs once resumed old checkpoints instead of training on the repaired
set. `base` trained 119 steps on the new set on top of 1,191 on the old one;
`P2` resumed at step 1,776 under a ceiling of 1,310 and trained none. Nothing
raised, and the cell reported both runs as done. These tests run without torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))

from train_sft import resume_refused  # noqa: E402


def test_a_checkpoint_already_past_the_ceiling_refuses():
    """The `P2` case: no steps left to train."""
    why = resume_refused({"global_step": 1776, "max_steps": 1965}, 1310)
    assert why and "nothing left to train" in why


def test_a_checkpoint_built_for_another_ceiling_refuses():
    """The `base` case: the set changed, so the steps per epoch did too."""
    why = resume_refused({"global_step": 1191, "max_steps": 1191}, 1310)
    assert why and "has changed since" in why


def test_a_real_resume_passes():
    """A run interrupted halfway through its own ceiling continues."""
    assert resume_refused({"global_step": 655, "max_steps": 1310}, 1310) is None


def test_a_state_without_a_ceiling_only_blocks_on_steps_done():
    """Without `max_steps` in the state, only the steps already done can refuse."""
    assert resume_refused({"global_step": 655}, 1310) is None
    assert resume_refused({"global_step": 1310}, 1310) is not None
