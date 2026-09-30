"""Re-score the stored factual answers, without a GPU.

A verdict in `evaluation/points/*.json` depends on the answer, the expected
list and the scorer. Only the answer needs the GPU, so after a fix to the
expected lists or to the scorer, the stored answers are scored again here.

Fixes made this way, all found by reading stored answers:

- an expected list that missed a common phrasing of the right answer;
- `no` matching inside `nord` and `notamment`, which scored two checkpoints
  right on answers that never addressed the question;
- a trap written without its accent, which filed three answers giving the
  trap as not known.

An answer cut at the token budget stays cut; that needs a new generation.
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
