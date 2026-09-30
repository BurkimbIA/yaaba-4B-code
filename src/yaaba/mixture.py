"""Resolving a training arm: from a recipe, or from a frozen dataset.

A recipe says how to compose the mixture; a frozen dataset records what was
seen. Only the frozen dataset lets two arms trained days apart be compared,
because the source data changed three times in as many days and the same
recipe then gave different mixtures.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .heldout import Guard
from .normalization import key

Take = Literal["all", "unpaired", "fill"]

FROZEN_URI = "s3://burkimbia-store/text/moore-assistant/bras"


@dataclass(frozen=True)
class Source:
    """One file of the mixture, and how much of it this arm takes."""

    path: str
    take: Take
    budget: int | None = None


@dataclass
class Report:
    """What loading actually produced, per file."""

    documents: int = 0
    tokens: int = 0
    dropped_paired: int = 0
    leaks: int = 0
    budget: int | None = None


@dataclass
class Arm:
    texts: list[str]
    per_file: dict[str, Report] = field(default_factory=dict)
    fingerprint: str = "recipe"

    @property
    def tokens(self) -> int:
        return sum(r.tokens for r in self.per_file.values())

    @property
    def documents(self) -> int:
        return sum(r.documents for r in self.per_file.values())

    @property
    def leaks(self) -> int:
        return sum(r.leaks for r in self.per_file.values())

def witness(mixture: Path,
            exclude: str | tuple[str, ...] = ()) -> Iterator[dict]:
    """The mixture, minus the `registre` values the caller writes into it.

    A generator: callers read it once, and a list of 303,000 dicts takes
    187 MB.

    An extractor that deduplicates against the mixture must exclude its own
    output, or a second run finds its texts already present and keeps almost
    nothing, without any error. On `nllb_mono` a rerun kept 3,305 lines instead
    of 35,542.

    `exclude` is the `registre` value the caller writes, or several of them.
    """
    excluded = {exclude} if isinstance(exclude, str) else set(exclude)
    if not mixture.exists():
        return
    for line in mixture.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("registre") not in excluded:
            yield row


def witness_keys(mixture: Path, exclude: str | tuple[str, ...] = ()) -> set[str]:
    """`witness` reduced to relaxed keys, which is what most extractors need.

    Built from the generator, so only the set is held in memory (a list first
    cost 257 MB at peak on a 303,000-row mixture).
    """
    return {key(row["text"]) for row in witness(mixture, exclude)}



def plan(recipe: dict) -> dict[str, Source]:
    """Recipe -> one Source per file, deduplicated.

    A bilingual file appears twice in a recipe, once per language, but it is one
    document; loading it twice would double it and skew the ratio. Two
    different takes on one file raise an error, since the result would depend
    on read order.
    """
    sources: dict[str, Source] = {}
    for part in recipe["parts"].values():
        for entry in part["sources"]:
            path, take = entry["fichier"], entry["prendre"]
            if take.startswith("sans la part appariee"):
                source = Source(path, "unpaired")
            elif take.startswith("cote") or take == "tout":
                source = Source(path, "all")
            elif take == "completer":
                source = Source(path, "fill", entry["tokens"])
            else:
                raise ValueError(f"unknown take: {take!r}")
            if path in sources:
                if sources[path] != source:
                    raise ValueError(
                        f"{path} appears twice with different takes: "
                        f"{sources[path]} and {source}")
                continue
            sources[path] = source
    return sources


def paired_keys(mixture: Path) -> set[str]:
    """Moore keys that have a French counterpart, from the canonical `P` file.

    Always from `P`, even for `P2`: `P2` groups the same pairs by document, so
    its keys are not separable.
    """
    keys = set()
    for line in (mixture / "P" / "bilingue.jsonl").open(encoding="utf-8"):
        moore, _, _ = json.loads(line)["text"].partition("\n\n")
        keys.add(key(moore))
    return keys


def load_recipe(recipe: dict, mixture: Path,
                held_out: dict[str, set[str]] | None = None) -> Arm:
    """Load an arm's texts in plan order, optionally checking them again for leaks.

    The check here runs on the texts the model will see, after composition.
    """
    sources = plan(recipe)
    paired = (paired_keys(mixture)
              if any(s.take == "unpaired" for s in sources.values()) else set())

    arm = Arm(texts=[])
    for path, source in sources.items():
        file = mixture / path
        if not file.exists():
            raise FileNotFoundError(file)
        guard = Guard(path, held_out) if held_out else None
        report = Report(budget=source.budget)

        for line in file.open(encoding="utf-8"):
            record = json.loads(line)
            if source.take == "unpaired" and key(record["text"]) in paired:
                report.dropped_paired += 1
                continue
            if source.budget is not None and report.tokens >= source.budget:
                break
            if guard and guard.inspect(record["text"]):
                report.leaks += 1
                continue
            arm.texts.append(record["text"])
            report.documents += 1
            report.tokens += record["tokens"]
        arm.per_file[path] = report
    return arm


def frozen_card(name: str, mixture: Path, cache: Path) -> dict | None:
    """The frozen arm's card: counts, ratio, leaks, recipe. No data is read.

    A frozen arm is verified from its card, since resolving the recipe again
    needs mixture files that a fresh runtime does not have.
    """
    from . import s3

    pointer = mixture / "bras" / name / "dernier.json"
    if not pointer.exists():
        pointer = cache / f"{name}-dernier.json"
        if not s3.download_file(f"{FROZEN_URI}/{name}/dernier.json", pointer):
            return None
    meta = json.loads(pointer.read_text(encoding="utf-8"))

    card = mixture / "bras" / name / meta["empreinte"] / "carte.json"
    if not card.exists():
        card = cache / f"{name}-carte.json"
        if not s3.download_file(f"{meta['uri']}/carte.json", card):
            return None
    return json.loads(card.read_text(encoding="utf-8"))


def load_frozen(name: str, mixture: Path, cache: Path) -> Arm | None:
    """Load an arm from its frozen parquet, or None if there is none.

    Looks for the local pointer first, then S3, so a fresh runtime works.
    """
    import pyarrow.parquet as pq

    from . import s3

    pointer = mixture / "bras" / name / "dernier.json"
    if not pointer.exists():
        # A single key: `download_dir` would list `.../x.json/`, find nothing,
        # and fall back to the recipe on a runtime that has no mixture files.
        pointer = cache / f"{name}-dernier.json"
        if not s3.download_file(f"{FROZEN_URI}/{name}/dernier.json", pointer):
            return None

    meta = json.loads(pointer.read_text(encoding="utf-8"))
    fingerprint = meta["empreinte"]

    folder = mixture / "bras" / name / fingerprint
    if not (folder / "data.parquet").exists():
        folder = cache / f"bras-{name}-{fingerprint}"
        if not (folder / "data.parquet").exists():
            s3.download_dir(meta["uri"], folder)
    if not (folder / "data.parquet").exists():
        return None

    texts = pq.read_table(folder / "data.parquet",
                          columns=["text"]).column("text").to_pylist()
    return Arm(texts=texts, fingerprint=fingerprint)
