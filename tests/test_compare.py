"""`--compare` and `--arms` must apply the same stop rule.

`compare` once carried its own copy of the rule. The copy cut the curve at a
point whose facts were merely not comparable, and cut every arm at the first
signal of any arm, so it kept nothing where `--arms` kept one point per arm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "measure"))

import evaluate_checkpoint as ec  # noqa: E402

CAPABILITIES = ec.SHOULD_DROP + ec.WATCH_APART + ec.SHOULD_HOLD


def point(name: str, arm: str | None, moore: float, facts: list[str],
          english: float = 1.0) -> dict:
    measures = {c: {"bits_par_caractere": 1.0} for c in CAPABILITIES}
    measures["moore_humain"] = {"bits_par_caractere": moore}
    measures["anglais"] = {"bits_par_caractere": english}
    return {"point": name,
            "chemin": f"/content/cpt/{arm}-20260101-0000/{name}" if arm else "base",
            "mesures": measures, "budget_faits": 320,
            "faits": [{"id": i, "note": "juste"} for i in facts],
            "faits_compte": {"juste": len(facts), "piege": 0}}


def write(folder: Path, points: list[tuple[str, dict]]) -> str:
    for stem, content in points:
        (folder / f"{stem}.json").write_text(json.dumps(content), encoding="utf-8")
    return str(folder / "*.json")


def test_facts_not_comparable_do_not_cut_the_curve(tmp_path, capsys):
    pattern = write(tmp_path, [
        ("base", point("base", None, 2.0, ["f1", "f2"])),
        # measured on another item set: a note, not a stop signal
        ("A-checkpoint-100", point("checkpoint-100", "A", 1.8, ["f1"])),
        ("A-checkpoint-200", point("checkpoint-200", "A", 1.5, ["f1", "f2"])),
        ("A-checkpoint-300", point("checkpoint-300", "A", 1.6, ["f1", "f2"])),
    ])
    assert ec.compare(pattern, answers={}) == 0
    out = capsys.readouterr().out
    assert "BEST A: checkpoint-200" in out
    assert "from checkpoint-300 onward excluded" in out


def test_a_signal_in_one_arm_does_not_cut_another(tmp_path, capsys):
    pattern = write(tmp_path, [
        ("base", point("base", None, 2.0, ["f1"])),
        # B loses English at its first point: B is cut, A is not
        ("B-checkpoint-50", point("checkpoint-50", "B", 1.9, ["f1"], english=1.2)),
        ("A-checkpoint-100", point("checkpoint-100", "A", 1.8, ["f1"])),
        ("A-checkpoint-200", point("checkpoint-200", "A", 1.5, ["f1"])),
    ])
    assert ec.compare(pattern, answers={}) == 1       # B has nothing to keep
    captured = capsys.readouterr()
    assert "BEST A: checkpoint-200" in captured.out
    assert "B: NO eligible checkpoint" in captured.err


def test_compare_and_arms_keep_the_same_point(tmp_path, capsys):
    pattern = write(tmp_path, [
        ("base", point("base", None, 2.0, ["f1"])),
        ("A-checkpoint-100", point("checkpoint-100", "A", 1.8, ["f1"])),
        ("A-checkpoint-200", point("checkpoint-200", "A", 1.5, ["f1"])),
        ("A-checkpoint-300", point("checkpoint-300", "A", 1.7, ["f1"])),
    ])
    kept, _, _ = ec.keepable(json.loads((tmp_path / "base.json").read_text()),
                             [json.loads((tmp_path / f"A-checkpoint-{s}.json").read_text())
                              for s in (100, 200, 300)], {})
    ec.compare(pattern, answers={})
    assert f"BEST A: {kept['point']}" in capsys.readouterr().out
