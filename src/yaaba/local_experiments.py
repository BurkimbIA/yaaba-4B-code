"""Shared, dependency-light contracts for local Mac experiments.

The training and evaluation entry points import heavyweight runtimes lazily.
This module keeps data preparation, provenance, hashes and resume checks
testable on CPU without MLX, Torch or model downloads.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .heldout import EVERYTHING, Guard
from .heldout import load as load_heldout

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
}
ROLES = {"system", "user", "assistant", "tool"}
MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path | str, value: Any) -> None:
    """Write JSON atomically so an interrupted run cannot leave half a manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def append_jsonl(path: Path | str, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def _messages(record: Mapping[str, Any], source: str) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}: missing non-empty `messages` list")
    clean: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source}: message {index} is not an object")
        role, content = message.get("role"), message.get("content")
        if role not in ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{source}: message {index} needs a known role and non-empty content"
            )
        clean.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in clean):
        raise ValueError(f"{source}: no assistant message to supervise")
    return clean


def assistant_prefixes(messages: list[dict[str, str]]) -> Iterator[list[dict[str, str]]]:
    """Yield one chat per assistant turn, with that turn as the completion.

    MLX-LM's ``mask_prompt`` and the MPS encoder both supervise only the final
    assistant message. Expanding a dialogue this way therefore gives every
    assistant turn a real assistant-only mask without placing user tokens in
    the loss.
    """
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            yield messages[: index + 1]


def assistant_completion_tokens(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int,
) -> dict[str, list[int]] | None:
    """Tokenize a chat and mask every token before its final assistant answer.

    The prefix equality check matters: subtracting token counts from separately
    templated strings is only valid when the generation prompt is an actual
    token prefix of the completed chat.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("assistant-only encoding requires a final assistant message")

    def templated(value: list[dict[str, str]], generation: bool) -> list[int]:
        options = {
            "tokenize": True,
            "add_generation_prompt": generation,
            "enable_thinking": False,
            "return_dict": False,
        }
        try:
            encoded = tokenizer.apply_chat_template(value, **options)
        except (TypeError, ValueError):
            options.pop("enable_thinking")
            encoded = tokenizer.apply_chat_template(value, **options)
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    prompt = templated(messages[:-1], True)
    full = templated(messages, False)
    if len(full) > max_length:
        return None
    if len(prompt) >= len(full) or full[: len(prompt)] != prompt:
        raise ValueError(
            "chat template generation prompt is not an exact prefix of the "
            "completed chat; refusing an approximate assistant mask"
        )
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def _assigned_split(
    record: Mapping[str, Any], *, seed: int, group: str,
    default_split: str | None, valid_percent: float, test_percent: float,
) -> tuple[str, str]:
    raw = record.get("split", default_split)
    if raw is not None:
        try:
            return SPLIT_ALIASES[str(raw).casefold()], "source"
        except KeyError as error:
            raise ValueError(f"unknown split {raw!r} for group {group!r}") from error
    if valid_percent < 0 or test_percent < 0 or valid_percent + test_percent >= 100:
        raise ValueError("valid_percent and test_percent must be non-negative and sum < 100")
    bucket = int(hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()[:16], 16)
    position = bucket / 0xFFFFFFFFFFFFFFFF * 100
    if position < test_percent:
        return "test", "sha256-group"
    if position < test_percent + valid_percent:
        return "valid", "sha256-group"
    return "train", "sha256-group"


def _nested_value(record: Mapping[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def prepare_chat_dataset(
    source: Path | str,
    destination: Path | str,
    *,
    seed: int,
    heldout_path: Path | str | None,
    default_split: str | None = None,
    group_field: str = "group_id",
    valid_percent: float = 5.0,
    test_percent: float = 5.0,
    limits: Mapping[str, int | None] | None = None,
    include_variants: set[str] | None = None,
    exclude_variants: set[str] | None = None,
    allow_ungrouped_technical_pilot: bool = False,
) -> dict[str, Any]:
    """Validate, leak-check, split and write MLX-LM chat JSONL files."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if heldout_path is None:
        raise ValueError("held-out file is required for a real training dataset")
    heldout_path = Path(heldout_path).resolve()
    guard = Guard(EVERYTHING, load_heldout(heldout_path))

    groups: dict[str, str] = {}
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_ALIASES.values()}
    source_records = 0
    excluded_records = 0
    ungrouped_records = 0
    variant_counts: dict[str, int] = {}
    split_methods: set[str] = set()
    lineage_envelopes = 0
    for line_number, source_record in read_jsonl(source):
        source_records += 1
        where = f"{source}:{line_number}"
        envelope = source_record
        if isinstance(source_record.get("record"), Mapping):
            record = dict(source_record["record"])
            lineage_envelopes += 1
            lineage = source_record.get("lineage")
            if isinstance(lineage, Mapping) and lineage.get("split") is not None:
                if (record.get("split") is not None and
                        str(record["split"]).casefold() !=
                        str(lineage["split"]).casefold()):
                    raise ValueError(
                        f"{where}: record.split conflicts with lineage.split"
                    )
                record["split"] = lineage["split"]
        else:
            record = source_record
        if include_variants is not None or exclude_variants is not None:
            included = include_variants or set()
            excluded = exclude_variants or set()
            if included & excluded:
                raise ValueError("included and excluded variants overlap")
            variant = record.get("variant")
            if not isinstance(variant, str):
                raise ValueError(f"{where}: variant is required by the filter")
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            if variant not in included | excluded:
                raise ValueError(
                    f"{where}: variant {variant!r} is neither included nor excluded"
                )
            if variant in excluded:
                excluded_records += 1
                continue
        messages = _messages(record, where)
        raw_group = _nested_value(envelope, group_field)
        if raw_group in (None, ""):
            raw_group = _nested_value(record, group_field)
        if raw_group in (None, ""):
            ungrouped_records += 1
            if not allow_ungrouped_technical_pilot:
                raise ValueError(
                    f"{where}: missing explicit {group_field!r}; refusing to invent "
                    "a per-line group. Use the technical-pilot override only for a "
                    "bounded software/backward-pass check."
                )
            if record.get("split", default_split) is None:
                raise ValueError(
                    f"{where}: ungrouped data cannot be assigned to a new hash split"
                )
            group = f"UNVERIFIED-line-{line_number}"
        else:
            group = str(raw_group)
        split, method = _assigned_split(
            record, seed=seed, group=group, default_split=default_split,
            valid_percent=valid_percent, test_percent=test_percent,
        )
        split_methods.add(method)
        previous = groups.setdefault(group, split)
        if previous != split:
            raise ValueError(
                f"group {group!r} crosses splits ({previous!r}, {split!r})"
            )
        for message_index, message in enumerate(messages):
            leak = guard.inspect(message["content"])
            if leak:
                raise ValueError(
                    f"{where}: held-out {leak} match in message {message_index}; "
                    "refusing to prepare training data"
                )
        for assistant_index, prefix in enumerate(assistant_prefixes(messages)):
            examples[split].append({
                "messages": prefix,
                "_provenance": {
                    "source_line": line_number,
                    "group_id": group,
                    "assistant_turn": assistant_index,
                    **({
                        "origin_path": envelope["lineage"].get("origin_path"),
                        "origin_line": envelope["lineage"].get("origin_line"),
                        "record_hash": envelope["lineage"].get("record_hash"),
                        **({"group_kind": envelope["lineage"]["group_kind"]}
                           if "group_kind" in envelope["lineage"] else {}),
                        **({"group_evidence_fields":
                            envelope["lineage"]["group_evidence_fields"]}
                           if "group_evidence_fields" in envelope["lineage"] else {}),
                    } if isinstance(envelope.get("lineage"), Mapping) else {}),
                },
            })

    limits = limits or {}
    before_limits = {name: len(rows) for name, rows in examples.items()}
    for offset, (name, rows) in enumerate(examples.items()):
        random.Random(seed + offset).shuffle(rows)
        limit = limits.get(name)
        if limit is not None:
            if limit < 1:
                raise ValueError(f"{name} sample limit must be positive")
            del rows[limit:]
        path = destination / f"{name}.jsonl"
        if rows:
            destination.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif path.exists():
            path.unlink()

    result = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "heldout": str(heldout_path),
        "heldout_sha256": sha256_file(heldout_path),
        "seed": seed,
        "split_method": sorted(split_methods),
        "group_field": group_field,
        "source_records": source_records,
        "lineage_envelopes": lineage_envelopes,
        "excluded_records": excluded_records,
        "variant_counts": variant_counts,
        "included_variants": sorted(include_variants) if include_variants is not None else None,
        "excluded_variants": sorted(exclude_variants) if exclude_variants is not None else None,
        "groups": len(groups),
        "grouping_verified": ungrouped_records == 0,
        "ungrouped_records": ungrouped_records,
        "assistant_examples_before_limits": before_limits,
        "assistant_examples": {name: len(rows) for name, rows in examples.items()},
        "files": {
            name: {"path": str(destination / f"{name}.jsonl"),
                   "sha256": sha256_file(destination / f"{name}.jsonl")}
            for name, rows in examples.items() if rows
        },
        "assistant_mask": "final assistant completion only; multi-turn expanded by prefix",
        "heldout_matches": 0,
    }
    write_json(destination / "provenance.json", result)
    return result


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_identity(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            text=True, capture_output=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_paths": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def model_identity(model: Path | str, revision: str | None) -> dict[str, Any]:
    """Return an exact local identity, or a required remote revision."""
    path = Path(model).expanduser()
    if not path.exists():
        if not revision:
            raise ValueError(
                "a remote model id requires --model-revision; use a local snapshot "
                "path for offline and exact execution"
            )
        return {"model": str(model), "revision": revision, "local": False}
    if not path.is_dir():
        raise ValueError(f"model path is not a directory: {path}")
    files = []
    for candidate in sorted(p for p in path.rglob("*") if p.is_file()):
        if candidate.name in MODEL_FILES or candidate.suffix == ".safetensors":
            files.append({
                "path": str(candidate.relative_to(path)),
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            })
    if not files:
        raise ValueError(f"no model/config/tokenizer files found under {path}")
    snapshot_revision = None
    parts = path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            snapshot_revision = parts[index + 1]
    return {
        "model": str(path.resolve()),
        "revision": revision or snapshot_revision,
        "local": True,
        "files": files,
        "identity_sha256": canonical_hash(files),
    }


def model_with_chat_template(
    model: Path | str, destination: Path | str, chat_template: Path | str | None,
) -> tuple[Path, dict[str, Any]]:
    """Return a model directory whose tokenizer has an explicit chat template.

    Some otherwise usable checkpoints omit ``chat_template``. An overlay keeps
    every model file as a symlink and only writes a patched tokenizer config;
    weights are neither copied nor converted.
    """
    model = Path(model).resolve()
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    embedded = config.get("chat_template")
    if chat_template is None:
        if not embedded:
            raise ValueError(
                f"{model} has no embedded chat template; pass --chat-template"
            )
        return model, {
            "source": "model tokenizer_config.json",
            "sha256": canonical_hash(embedded),
        }

    template_path = Path(chat_template).resolve()
    template = template_path.read_text(encoding="utf-8")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for child in model.iterdir():
        target = destination / child.name
        if child.name == "tokenizer_config.json" or target.exists():
            continue
        target.symlink_to(child, target_is_directory=child.is_dir())
    config["chat_template"] = template
    write_json(destination / "tokenizer_config.json", config)
    shutil.copy2(template_path, destination / "chat_template.jinja")
    return destination, {
        "source": str(template_path),
        "sha256": sha256_file(template_path),
        "overlay": str(destination),
        "base_model": str(model),
    }


def validate_prepared_assistant_masks(
    data_directory: Path | str, tokenizer: Any, max_length: int,
) -> dict[str, Any]:
    """Drop chats that do not fully fit and prove every kept answer has labels."""
    directory = Path(data_directory)
    report: dict[str, Any] = {"max_length": max_length, "splits": {}}
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        if not path.exists():
            continue
        kept: list[dict[str, Any]] = []
        dropped = 0
        minimum_answer_tokens: int | None = None
        for _, record in read_jsonl(path):
            encoded = assistant_completion_tokens(
                tokenizer, record["messages"], max_length
            )
            if encoded is None:
                dropped += 1
                continue
            answer_tokens = sum(label != -100 for label in encoded["labels"])
            if answer_tokens < 1:
                raise ValueError(
                    f"{path}: assistant has no supervised token within max_length"
                )
            minimum_answer_tokens = (
                answer_tokens if minimum_answer_tokens is None
                else min(minimum_answer_tokens, answer_tokens)
            )
            kept.append(record)
        if not kept:
            path.unlink()
        else:
            with path.open("w", encoding="utf-8") as handle:
                for record in kept:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        report["splits"][split] = {
            "kept": len(kept), "dropped_overflow": dropped,
            "minimum_supervised_assistant_tokens": minimum_answer_tokens,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return report


def ensure_resume_contract(path: Path | str, contract: Mapping[str, Any], resume: bool) -> str:
    """Persist the immutable run contract and reject incompatible resumes."""
    path = Path(path)
    digest = canonical_hash(contract)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError(f"run already exists at {path.parent}; pass --resume")
        if stored.get("sha256") != digest:
            raise ValueError(
                "resume contract changed (model, data, seed or training config); "
                "start a new run directory"
            )
    else:
        if resume:
            raise ValueError(f"cannot resume: missing {path}")
        write_json(path, {"sha256": digest, "contract": contract})
    return digest


def latest_mlx_checkpoint(adapter_dir: Path | str) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []
    for path in Path(adapter_dir).glob("*_adapters.safetensors"):
        prefix = path.name.partition("_")[0]
        if prefix.isdigit():
            candidates.append((int(prefix), path))
    if not candidates:
        return None, 0
    step, path = max(candidates)
    return path, step


def normalized_text(text: str) -> str:
    return " ".join(str(text).casefold().split())
