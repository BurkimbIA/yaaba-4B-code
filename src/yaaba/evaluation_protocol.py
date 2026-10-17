"""Fail-closed metadata guard for evaluation split roles.

This module can reject missing or contradictory declarations.  It cannot prove
that a registry lists every real-world exposure or every related source group.
That distinction is intentional: a successful check authorizes only the claim
recorded in its receipt, not a general claim of dataset independence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .local_experiments import canonical_hash

SCHEMA_VERSION = 1
ROLES = frozenset(
    {
        "training",
        "annotation_pilot",
        "development_frozen",
        "test_sealed",
        "diagnostic_exposed",
    }
)
PURPOSES = frozenset({"diagnostic", "development", "final_test"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class EvaluationProtocolError(ValueError):
    """Raised when a protocol cannot support the requested evaluation claim."""


def _sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if not _SHA256.fullmatch(digest):
        raise EvaluationProtocolError(f"{field} must be a SHA-256 digest")
    return digest


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise EvaluationProtocolError(f"{field} must be a string")
    identifier = value
    if not _IDENTIFIER.fullmatch(identifier):
        raise EvaluationProtocolError(f"{field} is not a valid opaque identifier")
    return identifier


def role_assignment_sha256(splits: Sequence[Mapping[str, Any]]) -> str:
    """Hash the complete, order-independent role assignment declaration."""

    ordered = sorted(
        (dict(item) for item in splits), key=lambda item: str(item.get("split_id"))
    )
    return canonical_hash(ordered)


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise EvaluationProtocolError(f"evaluation protocol does not exist: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationProtocolError(
            f"invalid evaluation protocol JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise EvaluationProtocolError("evaluation protocol must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationProtocolError(
            f"evaluation protocol schema_version must be {SCHEMA_VERSION}"
        )
    if value.get("status") != "frozen":
        raise EvaluationProtocolError("evaluation protocol status must be `frozen`")
    _identifier(value.get("protocol_id"), "protocol_id")
    return value, hashlib.sha256(raw).hexdigest()


def _validated_splits(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_splits = protocol.get("splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        raise EvaluationProtocolError(
            "evaluation protocol needs a non-empty splits list"
        )
    if not all(isinstance(item, dict) for item in raw_splits):
        raise EvaluationProtocolError("every split declaration must be an object")
    expected_assignment = role_assignment_sha256(raw_splits)
    if (
        _sha256(protocol.get("role_assignment_sha256"), "role_assignment_sha256")
        != expected_assignment
    ):
        raise EvaluationProtocolError("role_assignment_sha256 does not match splits")

    by_id: dict[str, dict[str, Any]] = {}
    data_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    for index, item in enumerate(raw_splits):
        prefix = f"splits[{index}]"
        split_id = _identifier(item.get("split_id"), f"{prefix}.split_id")
        if split_id in by_id:
            raise EvaluationProtocolError(f"duplicate split_id: {split_id}")
        role = item.get("role")
        if role not in ROLES:
            raise EvaluationProtocolError(
                f"{prefix}.role must be one of {sorted(ROLES)}"
            )
        data_sha256 = _sha256(item.get("data_sha256"), f"{prefix}.data_sha256")
        split_manifest_sha256 = _sha256(
            item.get("split_manifest_sha256"), f"{prefix}.split_manifest_sha256"
        )
        group_evidence_sha256 = _sha256(
            item.get("group_evidence_sha256"), f"{prefix}.group_evidence_sha256"
        )
        owner = data_owners.get(data_sha256)
        if owner is not None:
            raise EvaluationProtocolError(
                f"data_sha256 is assigned more than once: {owner} and {split_id}"
            )
        group_ids = item.get("group_ids")
        if not isinstance(group_ids, list) or not group_ids:
            raise EvaluationProtocolError(
                f"{prefix}.group_ids must be a non-empty list"
            )
        normalized_groups: list[str] = []
        for group_index, value in enumerate(group_ids):
            group_id = _identifier(value, f"{prefix}.group_ids[{group_index}]")
            owner = group_owners.get(group_id)
            if owner is not None:
                raise EvaluationProtocolError(
                    f"group_id is assigned more than once: {owner} and {split_id}"
                )
            normalized_groups.append(group_id)
            group_owners[group_id] = split_id
        history = item.get("exposure_history")
        if not isinstance(history, list):
            raise EvaluationProtocolError(f"{prefix}.exposure_history must be a list")
        for history_index, exposure in enumerate(history):
            history_prefix = f"{prefix}.exposure_history[{history_index}]"
            if not isinstance(exposure, dict):
                raise EvaluationProtocolError(f"{history_prefix} must be an object")
            _identifier(exposure.get("purpose"), f"{history_prefix}.purpose")
            _sha256(
                exposure.get("system_or_protocol_sha256"),
                f"{history_prefix}.system_or_protocol_sha256",
            )
            if (
                not isinstance(exposure.get("observed_at"), str)
                or not exposure["observed_at"].strip()
            ):
                raise EvaluationProtocolError(
                    f"{history_prefix}.observed_at must be a non-empty string"
                )
        normalized = dict(item)
        normalized.update(
            {
                "data_sha256": data_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "group_evidence_sha256": group_evidence_sha256,
                "group_ids": normalized_groups,
            }
        )
        by_id[split_id] = normalized
        data_owners[data_sha256] = split_id
    return by_id


def authorize_evaluation(
    *,
    protocol_path: Path | None,
    expected_protocol_sha256: str | None,
    split_id: str | None,
    purpose: str,
    questions_sha256: str,
    system_contract_sha256: str,
) -> dict[str, Any]:
    """Validate an evaluation role and return a manifest-ready receipt.

    An invocation without a registry remains available for historical commands,
    but is irreversibly labelled diagnostic and cannot claim development or final
    test evidence.
    """

    if purpose not in PURPOSES:
        raise EvaluationProtocolError(f"purpose must be one of {sorted(PURPOSES)}")
    questions_sha256 = _sha256(questions_sha256, "questions_sha256")
    system_contract_sha256 = _sha256(system_contract_sha256, "system_contract_sha256")
    if protocol_path is None:
        if expected_protocol_sha256 is not None:
            raise EvaluationProtocolError(
                "--evaluation-protocol-sha256 requires --evaluation-protocol"
            )
        if split_id is not None:
            raise EvaluationProtocolError("--split-id requires --evaluation-protocol")
        if purpose != "diagnostic":
            raise EvaluationProtocolError(
                "development and final_test require a frozen evaluation protocol"
            )
        return {
            "purpose": "diagnostic",
            "split_id": None,
            "split_role": "diagnostic_exposed",
            "role_source": "default_fail_closed",
            "independent_test_claim": False,
            "independence_status": "not_claimed",
            "protocol": None,
            "guard_scope": (
                "unregistered diagnostic input; no claim of split independence"
            ),
        }
    if split_id is None:
        raise EvaluationProtocolError("--evaluation-protocol requires --split-id")
    if expected_protocol_sha256 is None:
        raise EvaluationProtocolError(
            "--evaluation-protocol requires an independently pinned "
            "--evaluation-protocol-sha256"
        )

    protocol_path = protocol_path.resolve()
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    if (
        _sha256(expected_protocol_sha256, "evaluation_protocol_sha256")
        != protocol_sha256
    ):
        raise EvaluationProtocolError(
            "evaluation protocol content does not match its pinned SHA-256"
        )
    splits = _validated_splits(protocol)
    selected_id = _identifier(split_id, "split_id")
    try:
        selected = splits[selected_id]
    except KeyError as error:
        raise EvaluationProtocolError(
            f"split_id is not declared: {selected_id}"
        ) from error
    if selected["data_sha256"] != questions_sha256:
        raise EvaluationProtocolError(
            "questions SHA-256 does not match the declared split content"
        )

    role = selected["role"]
    if role == "training":
        raise EvaluationProtocolError("training data cannot be used by the evaluator")
    if purpose == "final_test":
        if role != "test_sealed":
            raise EvaluationProtocolError(
                f"final_test requires role test_sealed, got {role}"
            )
        if selected["exposure_history"]:
            raise EvaluationProtocolError(
                "a test_sealed split with prior exposure cannot be a final test"
            )
        if protocol.get("systems_frozen") is not True:
            raise EvaluationProtocolError("final_test requires systems_frozen=true")
        allowed = protocol.get("allowed_system_contract_sha256")
        if not isinstance(allowed, list) or not allowed:
            raise EvaluationProtocolError(
                "final_test requires allowed_system_contract_sha256"
            )
        allowed_hashes = {
            _sha256(value, "allowed_system_contract_sha256[]") for value in allowed
        }
        if system_contract_sha256 not in allowed_hashes:
            raise EvaluationProtocolError(
                "runtime system contract is absent from the frozen allowlist"
            )
    elif purpose == "development":
        if role != "development_frozen":
            raise EvaluationProtocolError(
                f"development requires role development_frozen, got {role}"
            )
    else:
        if role == "test_sealed":
            raise EvaluationProtocolError(
                "test_sealed data may only be opened by a final_test invocation"
            )

    return {
        "purpose": purpose,
        "split_id": selected_id,
        "split_role": role,
        "role_source": "frozen_protocol",
        "independent_test_claim": purpose == "final_test",
        "independence_status": (
            "declared_and_metadata_checked_not_empirically_proven"
            if purpose == "final_test"
            else "not_claimed"
        ),
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "protocol_id": protocol["protocol_id"],
            "role_assignment_sha256": protocol["role_assignment_sha256"],
            "split_manifest_sha256": selected["split_manifest_sha256"],
            "group_evidence_sha256": selected["group_evidence_sha256"],
        },
        "questions_sha256": questions_sha256,
        "system_contract_sha256": system_contract_sha256,
        "guard_scope": (
            "metadata consistency guard; does not prove complete exposure or "
            "real-world source-group independence"
        ),
    }
