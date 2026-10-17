"""Fail-closed, text-free registry for teacher proposal candidates.

The registry validates provenance and isolation metadata only.  It never reads
or stores source, proposal, reference, or gold text, and an eligible record is
still only eligible for review: it is not a human-accepted training label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final = 1
CONTRACT_VERSION: Final = "yaaba.teacher-proposal.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {"candidate_id", "source", "teacher", "generation", "screening"}
)
_SOURCE_FIELDS = frozenset(
    {
        "source_id",
        "group_ids",
        "source_sha256",
        "split",
        "provenance",
        "rights",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {"dataset_id", "dataset_revision", "record_id", "record_sha256"}
)
_RIGHTS_FIELDS = frozenset({"status", "basis", "license_id", "allowed_uses"})
_TEACHER_FIELDS = frozenset(
    {"model_id", "revision", "model_sha256", "tokenizer_sha256"}
)
_GENERATION_FIELDS = frozenset(
    {
        "run_id",
        "manifest_sha256",
        "code_sha256",
        "prompt_sha256",
        "proposal_sha256",
        "decoding_sha256",
    }
)
_SCREENING_FIELDS = frozenset({"reject_reasons"})

# These reasons are based only on proposal mechanics.  No reference score or
# linguistic quality proxy is an admissible automatic filter.
TECHNICAL_REJECT_REASONS: Final = frozenset(
    {
        "duplicate_proposal",
        "empty_proposal",
        "extreme_repetition",
        "invalid_characters",
        "length_out_of_bounds",
        "prompt_artifact",
        "source_copy",
        "truncated",
    }
)

# Reasons are emitted in this order so output is stable across Python/runtime
# details and independently of the ordering in input arrays.
_REASON_ORDER: Final = (
    "unknown_top_level_field",
    "unknown_source_field",
    "unknown_provenance_field",
    "unknown_rights_field",
    "unknown_teacher_field",
    "unknown_generation_field",
    "unknown_screening_field",
    "missing_candidate_id",
    "missing_source_id",
    "missing_group_ids",
    "invalid_group_ids",
    "missing_source_sha256",
    "invalid_source_sha256",
    "missing_split",
    "split_forbidden",
    "split_not_allowed",
    "source_id_forbidden",
    "source_hash_forbidden",
    "group_forbidden",
    "missing_provenance",
    "missing_dataset_id",
    "missing_dataset_revision",
    "missing_record_id",
    "missing_record_sha256",
    "invalid_record_sha256",
    "missing_rights",
    "rights_status_unknown",
    "rights_not_authorized",
    "missing_rights_basis",
    "missing_license_id",
    "invalid_rights_uses",
    "missing_required_rights_use",
    "missing_teacher",
    "missing_teacher_model_id",
    "missing_teacher_revision",
    "missing_teacher_model_sha256",
    "invalid_teacher_model_sha256",
    "missing_teacher_tokenizer_sha256",
    "invalid_teacher_tokenizer_sha256",
    "missing_generation",
    "missing_generation_run_id",
    "missing_generation_manifest_sha256",
    "invalid_generation_manifest_sha256",
    "missing_generation_code_sha256",
    "invalid_generation_code_sha256",
    "missing_prompt_sha256",
    "invalid_prompt_sha256",
    "missing_proposal_sha256",
    "invalid_proposal_sha256",
    "missing_decoding_sha256",
    "invalid_decoding_sha256",
    "invalid_screening_reasons",
    "unsupported_reject_reason",
)
_REASON_RANK = {reason: index for index, reason in enumerate(_REASON_ORDER)}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _digest(value: object) -> str | None:
    normalized = _text(value)
    if normalized is None:
        return None
    normalized = normalized.lower()
    return normalized if _SHA256.fullmatch(normalized) else None


def _normalize_name(value: str) -> str:
    return re.sub(r"[-\s]+", "_", value.strip().lower())


def _sorted_reasons(reasons: set[str]) -> list[str]:
    return sorted(reasons, key=lambda item: (_REASON_RANK.get(item, 10_000), item))


def _unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], reason: str, reasons: set[str]
) -> None:
    if set(value) - allowed:
        reasons.add(reason)


def _require_digest(
    value: object, *, missing: str, invalid: str, reasons: set[str]
) -> str | None:
    raw = _text(value)
    if raw is None:
        reasons.add(missing)
        return None
    digest = _digest(raw)
    if digest is None:
        reasons.add(invalid)
    return digest


@dataclass(frozen=True)
class ContractPolicy:
    """Immutable isolation and rights policy for a registry run."""

    policy_id: str
    allowed_splits: frozenset[str]
    forbidden_splits: frozenset[str]
    forbidden_group_ids: frozenset[str]
    forbidden_source_ids: frozenset[str]
    forbidden_source_sha256: frozenset[str]
    authorized_rights_statuses: frozenset[str]
    required_rights_uses: frozenset[str]
    policy_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContractPolicy:
        allowed_fields = {
            "schema_version",
            "policy_id",
            "allowed_splits",
            "forbidden_splits",
            "forbidden_group_ids",
            "forbidden_source_ids",
            "forbidden_source_sha256",
            "authorized_rights_statuses",
            "required_rights_uses",
        }
        unknown = set(value) - allowed_fields
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"policy schema_version must be {SCHEMA_VERSION}")

        policy_id = _text(value.get("policy_id"))
        if policy_id is None:
            raise ValueError("policy_id must be a non-empty string")

        def names(
            field: str, *, required: bool = False, normalize: bool = True
        ) -> frozenset[str]:
            raw = value.get(field)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"{field} must be an array of strings")
            parsed = frozenset(
                (_normalize_name(item) if normalize else item.strip())
                for item in raw
                if isinstance(item, str) and item.strip()
            )
            if len(parsed) != len(raw):
                raise ValueError(f"{field} must contain unique non-empty strings")
            if required and not parsed:
                raise ValueError(f"{field} must not be empty")
            return parsed

        allowed_splits = names("allowed_splits", required=True)
        forbidden_splits = names("forbidden_splits", required=True)
        if allowed_splits & forbidden_splits:
            raise ValueError("allowed_splits and forbidden_splits must be disjoint")
        mandatory_forbidden = {"dev", "val", "validation", "test", "gold"}
        if not mandatory_forbidden <= forbidden_splits:
            missing = sorted(mandatory_forbidden - forbidden_splits)
            raise ValueError(f"forbidden_splits omits mandatory values: {missing}")

        forbidden_group_ids = names("forbidden_group_ids", normalize=False)
        forbidden_source_ids = names("forbidden_source_ids", normalize=False)
        authorized_rights_statuses = names("authorized_rights_statuses", required=True)
        required_rights_uses = names("required_rights_uses", required=True)

        raw_hashes = value.get("forbidden_source_sha256")
        if not isinstance(raw_hashes, Sequence) or isinstance(raw_hashes, (str, bytes)):
            raise ValueError("forbidden_source_sha256 must be an array of SHA-256")
        forbidden_hashes = frozenset(str(item).strip().lower() for item in raw_hashes)
        if len(forbidden_hashes) != len(raw_hashes) or any(
            not _SHA256.fullmatch(item) for item in forbidden_hashes
        ):
            raise ValueError(
                "forbidden_source_sha256 must contain unique valid SHA-256 values"
            )

        canonical = {
            "schema_version": SCHEMA_VERSION,
            "policy_id": policy_id,
            "allowed_splits": sorted(allowed_splits),
            "forbidden_splits": sorted(forbidden_splits),
            "forbidden_group_ids": sorted(forbidden_group_ids),
            "forbidden_source_ids": sorted(forbidden_source_ids),
            "forbidden_source_sha256": sorted(forbidden_hashes),
            "authorized_rights_statuses": sorted(authorized_rights_statuses),
            "required_rights_uses": sorted(required_rights_uses),
        }
        return cls(
            policy_id=policy_id,
            allowed_splits=allowed_splits,
            forbidden_splits=forbidden_splits,
            forbidden_group_ids=forbidden_group_ids,
            forbidden_source_ids=forbidden_source_ids,
            forbidden_source_sha256=forbidden_hashes,
            authorized_rights_statuses=authorized_rights_statuses,
            required_rights_uses=required_rights_uses,
            policy_sha256=_sha256_canonical(canonical),
        )


def evaluate_candidate(
    candidate: Mapping[str, Any], policy: ContractPolicy
) -> dict[str, Any]:
    """Return one deterministic, text-free eligibility decision."""

    reasons: set[str] = set()
    _unknown_fields(candidate, _TOP_LEVEL_FIELDS, "unknown_top_level_field", reasons)

    candidate_id = _text(candidate.get("candidate_id"))
    if candidate_id is None:
        reasons.add("missing_candidate_id")

    source = _mapping(candidate.get("source"))
    _unknown_fields(source, _SOURCE_FIELDS, "unknown_source_field", reasons)
    source_id = _text(source.get("source_id"))
    raw_group_ids = source.get("group_ids")
    group_ids = (
        {
            item.strip()
            for item in raw_group_ids
            if isinstance(item, str) and item.strip()
        }
        if isinstance(raw_group_ids, Sequence)
        and not isinstance(raw_group_ids, (str, bytes))
        else set()
    )
    split_raw = _text(source.get("split"))
    split = _normalize_name(split_raw) if split_raw else None
    if source_id is None:
        reasons.add("missing_source_id")
    if not group_ids:
        reasons.add("missing_group_ids")
    if not isinstance(raw_group_ids, Sequence) or isinstance(
        raw_group_ids, (str, bytes)
    ):
        reasons.add("invalid_group_ids")
    elif len(group_ids) != len(raw_group_ids):
        reasons.add("invalid_group_ids")
    source_sha256 = _require_digest(
        source.get("source_sha256"),
        missing="missing_source_sha256",
        invalid="invalid_source_sha256",
        reasons=reasons,
    )
    if split is None:
        reasons.add("missing_split")
    elif split in policy.forbidden_splits:
        reasons.add("split_forbidden")
    elif split not in policy.allowed_splits:
        reasons.add("split_not_allowed")
    if source_id and source_id in policy.forbidden_source_ids:
        reasons.add("source_id_forbidden")
    if source_sha256 and source_sha256 in policy.forbidden_source_sha256:
        reasons.add("source_hash_forbidden")
    if group_ids & policy.forbidden_group_ids:
        reasons.add("group_forbidden")

    provenance = _mapping(source.get("provenance"))
    if not provenance:
        reasons.add("missing_provenance")
    _unknown_fields(provenance, _PROVENANCE_FIELDS, "unknown_provenance_field", reasons)
    dataset_id = _text(provenance.get("dataset_id"))
    dataset_revision = _text(provenance.get("dataset_revision"))
    record_id = _text(provenance.get("record_id"))
    if dataset_id is None:
        reasons.add("missing_dataset_id")
    if dataset_revision is None:
        reasons.add("missing_dataset_revision")
    if record_id is None:
        reasons.add("missing_record_id")
    record_sha256 = _require_digest(
        provenance.get("record_sha256"),
        missing="missing_record_sha256",
        invalid="invalid_record_sha256",
        reasons=reasons,
    )

    rights = _mapping(source.get("rights"))
    if not rights:
        reasons.add("missing_rights")
    _unknown_fields(rights, _RIGHTS_FIELDS, "unknown_rights_field", reasons)
    rights_status_raw = _text(rights.get("status"))
    rights_status = _normalize_name(rights_status_raw) if rights_status_raw else None
    if rights_status is None or rights_status == "unknown":
        reasons.add("rights_status_unknown")
    elif rights_status not in policy.authorized_rights_statuses:
        reasons.add("rights_not_authorized")
    rights_basis = _text(rights.get("basis"))
    license_id = _text(rights.get("license_id"))
    if rights_basis is None:
        reasons.add("missing_rights_basis")
    if license_id is None:
        reasons.add("missing_license_id")
    raw_uses = rights.get("allowed_uses")
    allowed_uses = (
        {
            _normalize_name(item)
            for item in raw_uses
            if isinstance(item, str) and item.strip()
        }
        if isinstance(raw_uses, Sequence) and not isinstance(raw_uses, (str, bytes))
        else set()
    )
    if not isinstance(raw_uses, Sequence) or isinstance(raw_uses, (str, bytes)):
        reasons.add("invalid_rights_uses")
    elif len(allowed_uses) != len(raw_uses):
        reasons.add("invalid_rights_uses")
    if not policy.required_rights_uses <= allowed_uses:
        reasons.add("missing_required_rights_use")

    teacher = _mapping(candidate.get("teacher"))
    if not teacher:
        reasons.add("missing_teacher")
    _unknown_fields(teacher, _TEACHER_FIELDS, "unknown_teacher_field", reasons)
    model_id = _text(teacher.get("model_id"))
    revision = _text(teacher.get("revision"))
    if model_id is None:
        reasons.add("missing_teacher_model_id")
    if revision is None:
        reasons.add("missing_teacher_revision")
    model_sha256 = _require_digest(
        teacher.get("model_sha256"),
        missing="missing_teacher_model_sha256",
        invalid="invalid_teacher_model_sha256",
        reasons=reasons,
    )
    tokenizer_sha256 = _require_digest(
        teacher.get("tokenizer_sha256"),
        missing="missing_teacher_tokenizer_sha256",
        invalid="invalid_teacher_tokenizer_sha256",
        reasons=reasons,
    )

    generation = _mapping(candidate.get("generation"))
    if not generation:
        reasons.add("missing_generation")
    _unknown_fields(generation, _GENERATION_FIELDS, "unknown_generation_field", reasons)
    run_id = _text(generation.get("run_id"))
    if run_id is None:
        reasons.add("missing_generation_run_id")
    generation_digests: dict[str, str | None] = {}
    for field, missing, invalid in (
        (
            "manifest_sha256",
            "missing_generation_manifest_sha256",
            "invalid_generation_manifest_sha256",
        ),
        (
            "code_sha256",
            "missing_generation_code_sha256",
            "invalid_generation_code_sha256",
        ),
        ("prompt_sha256", "missing_prompt_sha256", "invalid_prompt_sha256"),
        ("proposal_sha256", "missing_proposal_sha256", "invalid_proposal_sha256"),
        (
            "decoding_sha256",
            "missing_decoding_sha256",
            "invalid_decoding_sha256",
        ),
    ):
        generation_digests[field] = _require_digest(
            generation.get(field), missing=missing, invalid=invalid, reasons=reasons
        )

    screening = _mapping(candidate.get("screening"))
    _unknown_fields(screening, _SCREENING_FIELDS, "unknown_screening_field", reasons)
    raw_reject_reasons = screening.get("reject_reasons", [])
    if not isinstance(raw_reject_reasons, Sequence) or isinstance(
        raw_reject_reasons, (str, bytes)
    ):
        reasons.add("invalid_screening_reasons")
        technical_reasons: set[str] = set()
    else:
        technical_reasons = {
            _normalize_name(item)
            for item in raw_reject_reasons
            if isinstance(item, str) and item.strip()
        }
        if len(technical_reasons) != len(raw_reject_reasons):
            reasons.add("invalid_screening_reasons")
        if not technical_reasons <= TECHNICAL_REJECT_REASONS:
            reasons.add("unsupported_reject_reason")
        reasons.update(technical_reasons & TECHNICAL_REJECT_REASONS)

    if candidate_id is None:
        sanitized_seed = {
            "source_id": source_id,
            "group_ids": sorted(group_ids),
            "source_sha256": source_sha256,
            "proposal_sha256": generation_digests["proposal_sha256"],
        }
        candidate_id = f"unidentified-{_sha256_canonical(sanitized_seed)[:16]}"

    decision_reasons = _sorted_reasons(reasons)
    if not decision_reasons:
        decision_reasons = ["automatic_contract_checks_passed"]
        decision_status = "eligible_for_review"
    else:
        decision_status = "rejected"

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "candidate_id": candidate_id,
        "source": {
            "source_id": source_id,
            "group_ids": sorted(group_ids),
            "source_sha256": source_sha256,
            "split": split,
            "provenance": {
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "record_id": record_id,
                "record_sha256": record_sha256,
            },
            "rights": {
                "status": rights_status,
                "basis": rights_basis,
                "license_id": license_id,
                "allowed_uses": sorted(allowed_uses),
            },
        },
        "teacher": {
            "model_id": model_id,
            "revision": revision,
            "model_sha256": model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
        },
        "generation": {
            "run_id": run_id,
            **generation_digests,
        },
        "label_origin": "teacher_proposal",
        "decision": {
            "status": decision_status,
            "reasons": decision_reasons,
            "human_reviewed": False,
            "training_admitted": False,
        },
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_sha256,
    }
    record["record_sha256"] = _sha256_canonical(record)
    return record


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(parsed, Mapping):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(parsed)
    return records


def _write_if_same_or_new(path: Path, content: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_same_or_absent(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"refusing to replace different registry file: {path}")


def register_candidates(
    candidates: Sequence[Mapping[str, Any]],
    policy: ContractPolicy,
    output_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate and atomically register candidates plus a deterministic manifest."""

    if not candidates:
        raise ValueError("candidate registry must not be empty")
    records = [evaluate_candidate(candidate, policy) for candidate in candidates]
    records.sort(key=lambda item: item["candidate_id"])
    ids = [record["candidate_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")

    output = Path(output_path)
    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_name(f"{output.stem}.manifest.json")
    )
    if output.resolve() == manifest_output.resolve():
        raise ValueError("registry and manifest paths must be different")
    output_bytes = b"".join(
        (_canonical_json(record) + "\n").encode("utf-8") for record in records
    )
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    counts = {
        "eligible_for_review": sum(
            record["decision"]["status"] == "eligible_for_review" for record in records
        ),
        "rejected": sum(
            record["decision"]["status"] == "rejected" for record in records
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_sha256,
        "input_identity_sha256": _sha256_canonical(
            sorted(record["record_sha256"] for record in records)
        ),
        "registry_sha256": output_sha256,
        "records": len(records),
        "counts": counts,
        "contains_raw_text": False,
        "references_used_for_filtering": False,
        "labels_are_gold": False,
        "admission_scope": "technical_eligibility_for_review_only",
        "identity_comparison": "exact_utf8_after_outer_whitespace",
        "split_and_rights_normalization": "lowercase_then_hyphen_or_space_to_underscore",
    }
    manifest["manifest_sha256"] = _sha256_canonical(manifest)
    manifest_bytes = (_canonical_json(manifest) + "\n").encode("utf-8")

    # Validate both destinations before creating either one, so a conflicting
    # companion manifest cannot leave a partial new registry behind.
    _require_same_or_absent(output, output_bytes)
    _require_same_or_absent(manifest_output, manifest_bytes)
    _write_if_same_or_new(output, output_bytes)
    _write_if_same_or_new(manifest_output, manifest_bytes)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and register text-free teacher proposal metadata."
    )
    parser.add_argument("--input", type=Path, required=True, help="candidate JSONL")
    parser.add_argument("--policy", type=Path, required=True, help="policy JSON")
    parser.add_argument("--output", type=Path, required=True, help="registry JSONL")
    parser.add_argument("--manifest", type=Path, help="optional manifest JSON path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    policy_raw = json.loads(args.policy.read_text(encoding="utf-8"))
    if not isinstance(policy_raw, Mapping):
        raise ValueError("policy must contain a JSON object")
    policy = ContractPolicy.from_mapping(policy_raw)
    manifest = register_candidates(
        _read_jsonl(args.input),
        policy,
        args.output,
        manifest_path=args.manifest,
    )
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
