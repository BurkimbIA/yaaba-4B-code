from __future__ import annotations

import pytest

from yaaba.heldout import EVERYTHING, Guard

HELD_OUT = {
    "moore_humain": {"heldoutmoore"},
    "anglais": {"heldoutenglish"},
}


def test_unknown_part_fails_closed():
    with pytest.raises(ValueError, match="unknown mixture part"):
        Guard("mooree.jsonl", HELD_OUT)


def test_everything_checks_every_capability():
    guard = Guard(EVERYTHING, HELD_OUT)
    assert guard.inspect("held out Moore") == "equal"
    assert guard.inspect("held out English") == "equal"


def test_known_part_only_checks_its_declared_capabilities():
    guard = Guard("moore.jsonl", HELD_OUT)
    assert guard.inspect("held out Moore") == "equal"
    assert guard.inspect("held out English") is None
