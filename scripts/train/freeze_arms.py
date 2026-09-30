"""Freeze each arm into an immutable parquet set, identified by its content.

An arm used to be a recipe, resolved at training time. That made an early run
of arm `A` impossible to reproduce: within two days the source data changed
three times (a leak repaired, a share topped up, the recipes rebuilt), and the
same recipe then yielded another mixture without any warning. The recipe says
how to compose; the frozen set records what the model saw, which is what lets
two arms trained days apart be compared.

The file is a parquet readable by `datasets`, stored on S3 rather than the
Hub, since freezing does not require publishing and the licences of every
source have not been checked for redistribution.

    import datasets
    arm = datasets.load_dataset("parquet", data_files="arm-A.parquet")["train"]

The folder name is the first twelve characters of the sha256 of the texts, in
training order. The same mixture always gets the same folder, and different
mixtures get different ones, which a timestamp would not guarantee.

    s3://burkimbia-store/text/moore-assistant/bras/A/<fingerprint>/
        data.parquet     the texts, in order
        carte.json       the recipe, the counts, the licences, the date

The pointer `bras/A/dernier.json` names the latest fingerprint, so the notebook
does not need to know it. The S3 key names are in French because readers of
the existing files depend on them.

The card holds what an audit needs: the recipe, the count per share, the ratio
obtained, the leaks found at freezing time, the licences and the script
version.

    python scripts/train/freeze_arms.py --arm B            # write, no deposit
    python scripts/train/freeze_arms.py --arm A --deposit
    python scripts/train/freeze_arms.py --all --deposit
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from yaaba import mixture, packing  # noqa: E402
from yaaba import s3 as s3_store
from yaaba.heldout import TARGETS  # noqa: E402
from yaaba.heldout import load as load_held_out

ROOT = Path(__file__).resolve().parents[2]
MIXTURE, OUT = ROOT / "melange", ROOT / "melange" / "bras"
REMOTE = "s3://burkimbia-store/text/moore-assistant/bras"
ARMS = ("A", "B", "C", "P", "P2")

# Which file feeds which share, to label every row of the parquet. The keys are
# file names on disk and the values are card fields, so both stay French.
SHARE = {"moore.jsonl": "moore", "francais.jsonl": "francais",
         "anglais.jsonl": "anglais", "code.jsonl": "code", "maths.jsonl": "maths",
         "C/francais.jsonl": "francais_aligne", "P/bilingue.jsonl": "bilingue",
         "P2/bilingue.jsonl": "bilingue"}


def licences() -> dict[str, str]:
    """Licences per source, read from the manifest so the card follows its changes."""
    manifest = ROOT / "data" / "produit" / "reinjection_manifeste.csv"
    if not manifest.exists():
        return {}
    return {row.get("part", "?"): row["licence"]
            for row in csv.DictReader(manifest.open(encoding="utf-8"))
            if row.get("licence")}


def freeze(arm: str, deposit: bool) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    recipe = json.loads(
        (MIXTURE / "recettes" / f"{arm}.json").read_text(encoding="utf-8"))
    plan = mixture.plan(recipe)

    # The held-out texts are checked again at freezing time, on the texts that
    # will actually be trained on.
    held_out = load_held_out(ROOT / "evaluation" / "tenu_a_lecart.jsonl")
    loaded = mixture.load_recipe(recipe, MIXTURE, held_out)
    texts = loaded.texts
    per_file = {name: vars(report) for name, report in loaded.per_file.items()}

    if loaded.leaks:
        print(f"FAILED: {loaded.leaks} held-out leaks, not freezing", file=sys.stderr)
        return 1

    # Rebuild the share label in the same order the loader used.
    shares, sources = [], []
    for name in plan:
        count = per_file[name]["documents"]
        shares += [SHARE.get(name, "?")] * count
        sources += [name] * count
    assert len(shares) == len(texts), (len(shares), len(texts))

    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    fingerprint = digest.hexdigest()[:12]

    folder = OUT / arm / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts, "part": shares, "source": sources}),
                   folder / "data.parquet", compression="zstd")
    size = (folder / "data.parquet").stat().st_size

    total = sum(report["tokens"] for report in per_file.values())
    per_share: dict[str, int] = {}
    for name in plan:
        label = SHARE.get(name, "?")
        per_share[label] = per_share.get(label, 0) + per_file[name]["tokens"]

    # The card's keys stay in French: existing S3 tooling and the licence page
    # read them.
    card = {
        "bras": arm,
        "empreinte": fingerprint,
        "fige_le": dt.date.today().isoformat(),
        "documents": len(texts),
        "tokens": total,
        "tokens_par_part": per_share,
        "ratio_obtenu": {k: round(100 * v / total, 2) for k, v in per_share.items()},
        "sequences_1024": packing.sequence_count(total, len(texts)),
        "fuites_tenu_a_lecart": 0,
        "capacites_controlees": {k: list(v) for k, v in TARGETS.items() if k in plan},
        "par_fichier": per_file,
        "recette": recipe,
        "licences": licences(),
        "note": ("Fige depuis le reservoir a cette date. Une recette dit comment "
                 "composer, ce fichier dit ce qui a ete vu."),
    }
    (folder / "carte.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ARM {arm}  fingerprint {fingerprint}")
    print(f"  {len(texts):>9,} documents   {total:>12,} tokens")
    print(f"  parquet {size/1e6:>6.1f} Mo   ({size/max(total,1)*1000:.2f} B/kilotoken)")
    for label, tokens in sorted(per_share.items(), key=lambda pair: -pair[1]):
        print(f"    {label:16s} {tokens:>11,}  {100*tokens/total:5.2f} %")
    print(f"  -> {folder}")

    if not deposit:
        print("  (not deposited: run again with --deposit)")
        return 0

    uri = f"{REMOTE}/{arm}/{fingerprint}"
    s3_store.upload_dir(folder, uri)
    # The pointer, so the notebook does not have to know the fingerprint.
    pointer = OUT / arm / "dernier.json"
    pointer.write_text(json.dumps({"bras": arm, "empreinte": fingerprint,
                                   "uri": uri, "fige_le": card["fige_le"]},
                                  indent=2), encoding="utf-8")
    s3_store.upload_dir(pointer.parent, f"{REMOTE}/{arm}")

    # Read back what was just written to S3 and compare the hash, since an upload
    # that reports success can still serve a stale object.
    served = s3_store.matches(f"{uri}/data.parquet", folder / "data.parquet")
    if served is not True:
        print(f"  read back from S3: "
              f"{'ABSENT' if served is None else 'DIFFERENT'}", file=sys.stderr)
        return 1
    print("  read back from S3: identical")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--deposit", action="store_true")
    args = parser.parse_args()
    if not args.arm and not args.all:
        parser.error("either --arm or --all is required")
    for arm in (ARMS if args.all else (args.arm,)):
        if freeze(arm, args.deposit):
            return 1
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
