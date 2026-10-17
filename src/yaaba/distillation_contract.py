"""Fail-closed registry without raw content fields for teacher proposals.

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
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+~-]{0,127}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
IMMUTABLE_FORBIDDEN_SPLITS: Final = frozenset(
    {
        "benchmark",
        "dev",
        "development",
        "eval",
        "evaluation",
        "gold",
        "held_out",
        "heldout",
        "hold_out",
        "holdout",
        "human_review",
        "review",
        "test",
        "testing",
        "val",
        "validation",
    }
)
IMMUTABLE_REQUIRED_RIGHTS_USES: Final = frozenset(
    {"model_training", "teacher_generation"}
)
_EVALUATION_SPLIT_PREFIXES: Final = (
    "benchmark",
    "dev",
    "eval",
    "gold",
    "held",
    "hold",
    "review",
    "test",
    "val",
)
_TOP_LEVEL_FIELDS = frozenset(
    {"candidate_id", "source", "teacher", "generation", "screening"}
)
_SOURCE_FIELDS = frozenset(
    {
        "source_id",
        "source_parent_ids",
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
        "contract_id",
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
    "invalid_identifier",
    "missing_source_id",
    "missing_source_parent_ids",
    "invalid_source_parent_ids",
    "missing_group_ids",
    "invalid_group_ids",
    "missing_required_group_kind",
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
    "invalid_dataset_revision",
    "dataset_identity_not_allowed",
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
    "invalid_teacher_revision",
    "missing_teacher_model_sha256",
    "invalid_teacher_model_sha256",
    "missing_teacher_tokenizer_sha256",
    "invalid_teacher_tokenizer_sha256",
    "missing_generation",
    "missing_generation_contract_id",
    "generation_contract_not_allowed",
    "generation_contract_identity_mismatch",
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


def _identifier(value: object) -> str | None:
    normalized = _text(value)
    if normalized is None or not _IDENTIFIER.fullmatch(normalized):
        return None
    return normalized


def _revision(value: object) -> str | None:
    normalized = _text(value)
    if normalized is None:
        return None
    normalized = normalized.lower()
    return normalized if _IMMUTABLE_REVISION.fullmatch(normalized) else None


def _normalize_name(value: str) -> str:
    return re.sub(r"[-\s]+", "_", value.strip().lower())


def _name(value: object) -> str | None:
    raw = _text(value)
    if raw is None:
        return None
    normalized = _normalize_name(raw)
    return normalized if _NAME.fullmatch(normalized) else None


def _is_evaluation_split(value: str) -> bool:
    return value in IMMUTABLE_FORBIDDEN_SPLITS or any(
        token.startswith(_EVALUATION_SPLIT_PREFIXES) for token in value.split("_")
    )


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
    required_group_kinds: frozenset[str]
    allowed_datasets: Mapping[str, str]
    generation_contracts: Mapping[str, Mapping[str, Any]]
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
            "required_group_kinds",
            "allowed_datasets",
            "allowed_generation_contracts",
            "authorized_rights_statuses",
            "required_rights_uses",
        }
        unknown = set(value) - allowed_fields
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"policy schema_version must be {SCHEMA_VERSION}")

        policy_id = _identifier(value.get("policy_id"))
        if policy_id is None:
            raise ValueError("policy_id must be a canonical opaque identifier")

        def names(
            field: str, *, required: bool = False, identifiers: bool = False
        ) -> frozenset[str]:
            raw = value.get(field)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"{field} must be an array of strings")
            parser = _identifier if identifiers else _name
            values = [parser(item) for item in raw]
            if any(item is None for item in values):
                raise ValueError(f"{field} contains a non-canonical value")
            parsed = frozenset(values)
            if len(parsed) != len(values):
                raise ValueError(f"{field} must contain unique values")
            if required and not parsed:
                raise ValueError(f"{field} must not be empty")
            return parsed

        allowed_splits = names("allowed_splits", required=True)
        configured_forbidden_splits = names("forbidden_splits")
        forbidden_splits = configured_forbidden_splits | IMMUTABLE_FORBIDDEN_SPLITS
        if allowed_splits & forbidden_splits or any(
            _is_evaluation_split(split) for split in allowed_splits
        ):
            raise ValueError("allowed_splits intersects an evaluation split")

        forbidden_group_ids = names("forbidden_group_ids", identifiers=True)
        forbidden_source_ids = names("forbidden_source_ids", identifiers=True)
        required_group_kinds = names("required_group_kinds", required=True)
        authorized_rights_statuses = names("authorized_rights_statuses", required=True)
        if authorized_rights_statuses != {"authorized"}:
            raise ValueError(
                "authorized_rights_statuses must be exactly ['authorized']"
            )
        configured_rights_uses = names("required_rights_uses")
        required_rights_uses = configured_rights_uses | IMMUTABLE_REQUIRED_RIGHTS_USES

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

        raw_datasets = value.get("allowed_datasets")
        if (
            not isinstance(raw_datasets, Sequence)
            or isinstance(raw_datasets, (str, bytes))
            or not raw_datasets
        ):
            raise ValueError("allowed_datasets must be a non-empty array")
        allowed_datasets: dict[str, str] = {}
        for raw_dataset in raw_datasets:
            if not isinstance(raw_dataset, Mapping) or set(raw_dataset) != {
                "dataset_id",
                "dataset_revision",
            }:
                raise ValueError(
                    "each allowed dataset needs only dataset_id and dataset_revision"
                )
            dataset_id = _identifier(raw_dataset.get("dataset_id"))
            dataset_revision = _revision(raw_dataset.get("dataset_revision"))
            if dataset_id is None or dataset_revision is None:
                raise ValueError(
                    "allowed dataset identities need a canonical id and immutable revision"
                )
            if dataset_id in allowed_datasets:
                raise ValueError("allowed dataset ids must be unique")
            allowed_datasets[dataset_id] = dataset_revision

        raw_contracts = value.get("allowed_generation_contracts")
        if (
            not isinstance(raw_contracts, Sequence)
            or isinstance(raw_contracts, (str, bytes))
            or not raw_contracts
        ):
            raise ValueError("allowed_generation_contracts must be a non-empty array")
        generation_contracts: dict[str, Mapping[str, Any]] = {}
        for raw_contract in raw_contracts:
            if not isinstance(raw_contract, Mapping) or set(raw_contract) != {
                "contract_id",
                "teacher",
                "generation",
            }:
                raise ValueError(
                    "generation contracts need only contract_id, teacher, and generation"
                )
            contract_id = _identifier(raw_contract.get("contract_id"))
            teacher = _mapping(raw_contract.get("teacher"))
            generation = _mapping(raw_contract.get("generation"))
            if contract_id is None or set(teacher) != _TEACHER_FIELDS:
                raise ValueError("generation contract teacher identity is incomplete")
            expected_generation_fields = _GENERATION_FIELDS - {
                "contract_id",
                "proposal_sha256",
            }
            if set(generation) != expected_generation_fields:
                raise ValueError("generation contract run identity is incomplete")
            canonical_teacher = {
                "model_id": _identifier(teacher.get("model_id")),
                "revision": _revision(teacher.get("revision")),
                "model_sha256": _digest(teacher.get("model_sha256")),
                "tokenizer_sha256": _digest(teacher.get("tokenizer_sha256")),
            }
            canonical_generation = {
                "run_id": _identifier(generation.get("run_id")),
                **{
                    field: _digest(generation.get(field))
                    for field in expected_generation_fields - {"run_id"}
                },
            }
            if (
                None in canonical_teacher.values()
                or None in canonical_generation.values()
            ):
                raise ValueError(
                    "generation contract identities need canonical ids, immutable revision, and SHA-256 values"
                )
            if contract_id in generation_contracts:
                raise ValueError("generation contract ids must be unique")
            identity: dict[str, Any] = {
                "contract_id": contract_id,
                "teacher": canonical_teacher,
                "generation": canonical_generation,
            }
            identity["identity_sha256"] = _sha256_canonical(identity)
            generation_contracts[contract_id] = identity

        canonical = {
            "schema_version": SCHEMA_VERSION,
            "policy_id": policy_id,
            "allowed_splits": sorted(allowed_splits),
            "forbidden_splits": sorted(forbidden_splits),
            "forbidden_group_ids": sorted(forbidden_group_ids),
            "forbidden_source_ids": sorted(forbidden_source_ids),
            "forbidden_source_sha256": sorted(forbidden_hashes),
            "required_group_kinds": sorted(required_group_kinds),
            "allowed_datasets": [
                {"dataset_id": key, "dataset_revision": allowed_datasets[key]}
                for key in sorted(allowed_datasets)
            ],
            "allowed_generation_contracts": [
                generation_contracts[key] for key in sorted(generation_contracts)
            ],
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
            required_group_kinds=required_group_kinds,
            allowed_datasets=allowed_datasets,
            generation_contracts=generation_contracts,
            authorized_rights_statuses=authorized_rights_statuses,
            required_rights_uses=required_rights_uses,
            policy_sha256=_sha256_canonical(canonical),
        )


def evaluate_candidate(
    candidate: Mapping[str, Any], policy: ContractPolicy
) -> dict[str, Any]:
    """Return one deterministic eligibility decision without content fields."""

    reasons: set[str] = set()
    _unknown_fields(candidate, _TOP_LEVEL_FIELDS, "unknown_top_level_field", reasons)

    candidate_id_raw = _text(candidate.get("candidate_id"))
    candidate_id = _identifier(candidate_id_raw)
    if candidate_id_raw is None:
        reasons.add("missing_candidate_id")
    elif candidate_id is None:
        reasons.add("invalid_identifier")

    source = _mapping(candidate.get("source"))
    _unknown_fields(source, _SOURCE_FIELDS, "unknown_source_field", reasons)
    source_id_raw = _text(source.get("source_id"))
    source_id = _identifier(source_id_raw)
    raw_parent_ids = source.get("source_parent_ids")
    source_parent_ids = (
        {
            item
            for item in (_identifier(raw) for raw in raw_parent_ids)
            if item is not None
        }
        if isinstance(raw_parent_ids, Sequence)
        and not isinstance(raw_parent_ids, (str, bytes))
        else set()
    )
    raw_group_ids = source.get("group_ids")
    group_ids: dict[str, str] = {}
    if isinstance(raw_group_ids, Mapping):
        for raw_kind, raw_identifier in raw_group_ids.items():
            kind = _name(raw_kind)
            identifier = _identifier(raw_identifier)
            if kind is None or identifier is None or kind in group_ids:
                reasons.add("invalid_group_ids")
                continue
            group_ids[kind] = identifier
    else:
        reasons.add("invalid_group_ids")
    split_raw = _text(source.get("split"))
    split = _name(split_raw)
    if source_id_raw is None:
        reasons.add("missing_source_id")
    elif source_id is None:
        reasons.add("invalid_identifier")
    if not source_parent_ids:
        reasons.add("missing_source_parent_ids")
    if not isinstance(raw_parent_ids, Sequence) or isinstance(
        raw_parent_ids, (str, bytes)
    ):
        reasons.add("invalid_source_parent_ids")
    elif len(source_parent_ids) != len(raw_parent_ids):
        reasons.add("invalid_source_parent_ids")
    if not group_ids:
        reasons.add("missing_group_ids")
    if not policy.required_group_kinds <= set(group_ids):
        reasons.add("missing_required_group_kind")
    source_sha256 = _require_digest(
        source.get("source_sha256"),
        missing="missing_source_sha256",
        invalid="invalid_source_sha256",
        reasons=reasons,
    )
    if split_raw is None:
        reasons.add("missing_split")
    elif split is None:
        reasons.add("invalid_identifier")
    elif _is_evaluation_split(split) or split in policy.forbidden_splits:
        reasons.add("split_forbidden")
    elif split not in policy.allowed_splits:
        reasons.add("split_not_allowed")
    if source_id and source_id in policy.forbidden_source_ids:
        reasons.add("source_id_forbidden")
    if source_sha256 and source_sha256 in policy.forbidden_source_sha256:
        reasons.add("source_hash_forbidden")
    if (set(group_ids.values()) | source_parent_ids) & policy.forbidden_group_ids:
        reasons.add("group_forbidden")

    provenance = _mapping(source.get("provenance"))
    if not provenance:
        reasons.add("missing_provenance")
    _unknown_fields(provenance, _PROVENANCE_FIELDS, "unknown_provenance_field", reasons)
    dataset_id_raw = _text(provenance.get("dataset_id"))
    dataset_id = _identifier(dataset_id_raw)
    dataset_revision_raw = _text(provenance.get("dataset_revision"))
    dataset_revision = _revision(dataset_revision_raw)
    record_id_raw = _text(provenance.get("record_id"))
    record_id = _identifier(record_id_raw)
    if dataset_id_raw is None:
        reasons.add("missing_dataset_id")
    elif dataset_id is None:
        reasons.add("invalid_identifier")
    if dataset_revision_raw is None:
        reasons.add("missing_dataset_revision")
    elif dataset_revision is None:
        reasons.add("invalid_dataset_revision")
    if record_id_raw is None:
        reasons.add("missing_record_id")
    elif record_id is None:
        reasons.add("invalid_identifier")
    if (
        dataset_id is not None
        and dataset_revision is not None
        and policy.allowed_datasets.get(dataset_id) != dataset_revision
    ):
        reasons.add("dataset_identity_not_allowed")
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
    rights_status = _name(rights_status_raw)
    if rights_status is None or rights_status == "unknown":
        reasons.add("rights_status_unknown")
    elif rights_status not in policy.authorized_rights_statuses:
        reasons.add("rights_not_authorized")
    rights_basis_raw = _text(rights.get("basis"))
    rights_basis = _identifier(rights_basis_raw)
    license_id_raw = _text(rights.get("license_id"))
    license_id = _identifier(license_id_raw)
    if rights_basis_raw is None:
        reasons.add("missing_rights_basis")
    elif rights_basis is None:
        reasons.add("invalid_identifier")
    if license_id_raw is None:
        reasons.add("missing_license_id")
    elif license_id is None:
        reasons.add("invalid_identifier")
    raw_uses = rights.get("allowed_uses")
    allowed_uses = (
        {parsed for item in raw_uses if (parsed := _name(item)) is not None}
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
    model_id_raw = _text(teacher.get("model_id"))
    model_id = _identifier(model_id_raw)
    revision_raw = _text(teacher.get("revision"))
    revision = _revision(revision_raw)
    if model_id_raw is None:
        reasons.add("missing_teacher_model_id")
    elif model_id is None:
        reasons.add("invalid_identifier")
    if revision_raw is None:
        reasons.add("missing_teacher_revision")
    elif revision is None:
        reasons.add("invalid_teacher_revision")
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
    contract_id_raw = _text(generation.get("contract_id"))
    contract_id = _identifier(contract_id_raw)
    if contract_id_raw is None:
        reasons.add("missing_generation_contract_id")
    elif contract_id is None:
        reasons.add("invalid_identifier")
    expected_contract = policy.generation_contracts.get(contract_id or "")
    if contract_id is not None and expected_contract is None:
        reasons.add("generation_contract_not_allowed")
    run_id_raw = _text(generation.get("run_id"))
    run_id = _identifier(run_id_raw)
    if run_id_raw is None:
        reasons.add("missing_generation_run_id")
    elif run_id is None:
        reasons.add("invalid_identifier")
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

    actual_teacher = {
        "model_id": model_id,
        "revision": revision,
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
    actual_generation_identity = {
        "run_id": run_id,
        **{
            field: generation_digests[field]
            for field in generation_digests
            if field != "proposal_sha256"
        },
    }
    if expected_contract is not None and (
        actual_teacher != expected_contract["teacher"]
        or actual_generation_identity != expected_contract["generation"]
    ):
        reasons.add("generation_contract_identity_mismatch")

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
            "source_parent_ids": sorted(source_parent_ids),
            "group_ids": group_ids,
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
            "source_parent_ids": sorted(source_parent_ids),
            "group_ids": {key: group_ids[key] for key in sorted(group_ids)},
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
            "contract_id": contract_id,
            "contract_identity_sha256": (
                expected_contract["identity_sha256"]
                if expected_contract is not None
                else None
            ),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # link(2) fails with EEXIST and therefore never replaces a file
            # created by a concurrent writer between preflight and commit.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    f"refusing to replace different registry file: {path}"
                ) from None
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
    contract_counts: dict[str, dict[str, int]] = {}
    dataset_counts: dict[tuple[str, str], dict[str, int]] = {}
    for record in records:
        status = record["decision"]["status"]
        contract_id = record["generation"]["contract_id"]
        if contract_id in policy.generation_contracts:
            by_status = contract_counts.setdefault(
                contract_id, {"eligible_for_review": 0, "rejected": 0}
            )
            by_status[status] += 1
        dataset_id = record["source"]["provenance"]["dataset_id"]
        dataset_revision = record["source"]["provenance"]["dataset_revision"]
        if (
            dataset_id in policy.allowed_datasets
            and policy.allowed_datasets[dataset_id] == dataset_revision
        ):
            key = (dataset_id, dataset_revision)
            by_status = dataset_counts.setdefault(
                key, {"eligible_for_review": 0, "rejected": 0}
            )
            by_status[status] += 1
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
        "allowed_generation_contracts_referenced": [
            {
                **policy.generation_contracts[key],
                "counts": contract_counts[key],
            }
            for key in sorted(contract_counts)
        ],
        "allowed_dataset_identities_referenced": [
            {
                "dataset_id": dataset_id,
                "dataset_revision": revision,
                "counts": dataset_counts[(dataset_id, revision)],
            }
            for dataset_id, revision in sorted(dataset_counts)
        ],
        "raw_content_fields_permitted": False,
        "identifier_syntax": _IDENTIFIER.pattern,
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
        description="Validate teacher proposal metadata without content fields."
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
