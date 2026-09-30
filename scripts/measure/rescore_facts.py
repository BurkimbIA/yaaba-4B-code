"""Re-score the stored factual answers, without a GPU.

A verdict in `evaluation/points/*.json` is a function of three things: the
answer, the expected list, and the scorer. Only the answer costs GPU time. When
the other two are wrong -- and both were -- re-running the models would be
paying for a measurement that is already on disk.

Three defects were repaired at once, all found by reading the stored answers:

    `bf04`  expected `non|no|enclav|landlocked` and missed the most natural
            phrasing of the right answer, `n'a pas d'acces a la mer`.
    `no`    matched inside `nord` and `notamment`, scoring two checkpoints RIGHT
            on answers that never addressed the question.
    `af02`  trapped on `senegal` but not `Senegal` with its accent, so three
            points that had learned the wrong river were filed as ignorant.

What this script cannot repair is the fourth defect, truncation: an answer cut
at the token budget stays cut. That one needs the GPU again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yaaba.facts import VERDICTS, score  # noqa: E402

EVAL = ROOT / "evaluation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="rewrite the point files; without it, only report")
    args = parser.parse_args(argv)

    facts = {f["id"]: f for f in
             (json.loads(line) for line in (EVAL / "faits.jsonl").open(encoding="utf-8"))}

    changed_points = 0
    for path in sorted((EVAL / "points").glob("*.json")):
        point = json.loads(path.read_text(encoding="utf-8"))
        moves = []
        for answered in point["faits"]:
            before = answered["note"]
            after = score(answered["reponse"], facts[answered["id"]],
                          answered.get("complet", True))
            if after != before:
                moves.append(f"{answered['id']} {before}->{after}")
                answered["note"] = after
        if not moves:
            continue
        changed_points += 1
        point["faits_compte"] = {kind: sum(f["note"] == kind for f in point["faits"])
                                 for kind in VERDICTS}
        print(f"{path.stem:24s} {point['faits_compte']['juste']:>2d}"
              f"/{len(point['faits'])} juste   "
              + "  ".join(moves))
        if args.write:
            path.write_text(json.dumps(point, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    verb = "rescored" if args.write else "would change"
    print(f"\n{verb}: {changed_points} points")
    if not args.write:
        print("re-run with --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
