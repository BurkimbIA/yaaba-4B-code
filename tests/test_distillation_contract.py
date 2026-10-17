from __future__ import annotations

import json
from copy import deepcopy

import pytest

from yaaba.distillation_contract import (
    ContractPolicy,
    evaluate_candidate,
    register_candidates,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


@pytest.fixture
def policy() -> ContractPolicy:
    return ContractPolicy.from_mapping(
        {
            "schema_version": 1,
            "policy_id": "private-pool-v1",
            "allowed_splits": ["teacher_pool"],
            "forbidden_splits": [
                "dev",
                "val",
                "validation",
                "test",
                "gold",
                "eval",
                "heldout",
            ],
            "forbidden_group_ids": ["eval-page-9"],
            "forbidden_source_ids": ["dev-32-item-1"],
            "forbidden_source_sha256": [SHA_F],
            "authorized_rights_statuses": ["authorized"],
            "required_rights_uses": ["teacher_generation", "model_training"],
        }
    )


@pytest.fixture
def candidate() -> dict:
    return {
        "candidate_id": "proposal-001",
        "source": {
            "source_id": "source-001",
            "group_ids": ["document-001", "page-001"],
            "source_sha256": SHA_A,
            "split": "teacher_pool",
            "provenance": {
                "dataset_id": "authorized-private-source",
                "dataset_revision": "revision-20261017",
                "record_id": "record-001",
                "record_sha256": SHA_B,
            },
            "rights": {
                "status": "authorized",
                "basis": "owner_permission",
                "license_id": "private-research-license-v1",
                "allowed_uses": ["model_training", "teacher_generation"],
            },
        },
        "teacher": {
            "model_id": "teacher/model",
            "revision": "revision-abc",
            "model_sha256": SHA_C,
            "tokenizer_sha256": SHA_D,
        },
        "generation": {
            "run_id": "run-001",
            "manifest_sha256": SHA_E,
            "code_sha256": SHA_A,
            "prompt_sha256": SHA_B,
            "proposal_sha256": SHA_C,
            "decoding_sha256": SHA_D,
        },
        "screening": {"reject_reasons": []},
    }


def reasons(record: dict) -> set[str]:
    return set(record["decision"]["reasons"])


def test_valid_candidate_is_only_eligible_for_review(policy, candidate):
    record = evaluate_candidate(candidate, policy)

    assert record["decision"] == {
        "status": "eligible_for_review",
        "reasons": ["automatic_contract_checks_passed"],
        "human_reviewed": False,
        "training_admitted": False,
    }
    assert record["label_origin"] == "teacher_proposal"
    assert "source_text" not in json.dumps(record)


@pytest.mark.parametrize("split", ["dev", "VAL", "validation", "test", "gold"])
def test_evaluation_or_gold_split_is_rejected(policy, candidate, split):
    candidate["source"]["split"] = split

    record = evaluate_candidate(candidate, policy)

    assert record["decision"]["status"] == "rejected"
    assert "split_forbidden" in reasons(record)


def test_forbidden_group_is_rejected_exactly(policy, candidate):
    candidate["source"]["group_ids"] = ["document-parent", "eval-page-9"]

    record = evaluate_candidate(candidate, policy)

    assert "group_forbidden" in reasons(record)


def test_similar_but_not_exact_group_is_not_rejected(policy, candidate):
    candidate["source"]["group_ids"] = ["Eval-Page-9"]

    record = evaluate_candidate(candidate, policy)

    assert record["decision"]["status"] == "eligible_for_review"


def test_forbidden_source_identity_is_rejected(policy, candidate):
    candidate["source"]["source_id"] = "dev-32-item-1"
    candidate["source"]["source_sha256"] = SHA_F

    record = evaluate_candidate(candidate, policy)

    assert {"source_id_forbidden", "source_hash_forbidden"} <= reasons(record)


def test_missing_teacher_revision_and_prompt_hash_fail_closed(policy, candidate):
    del candidate["teacher"]["revision"]
    del candidate["generation"]["prompt_sha256"]

    record = evaluate_candidate(candidate, policy)

    assert {"missing_teacher_revision", "missing_prompt_sha256"} <= reasons(record)


def test_unknown_rights_are_rejected(policy, candidate):
    candidate["source"]["rights"]["status"] = "unknown"

    record = evaluate_candidate(candidate, policy)

    assert "rights_status_unknown" in reasons(record)


def test_missing_training_rights_are_rejected(policy, candidate):
    candidate["source"]["rights"]["allowed_uses"] = ["teacher_generation"]

    record = evaluate_candidate(candidate, policy)

    assert "missing_required_rights_use" in reasons(record)


def test_gold_or_raw_text_field_is_rejected_and_not_registered(policy, candidate):
    candidate["gold"] = "must never be retained"

    record = evaluate_candidate(candidate, policy)

    assert "unknown_top_level_field" in reasons(record)
    assert "must never be retained" not in json.dumps(record)


def test_reference_based_filter_reason_is_not_supported(policy, candidate):
    candidate["screening"]["reject_reasons"] = ["low_chrf_to_reference"]

    record = evaluate_candidate(candidate, policy)

    assert "unsupported_reject_reason" in reasons(record)


def test_registry_is_deterministic_across_input_order(tmp_path, policy, candidate):
    second = deepcopy(candidate)
    second["candidate_id"] = "proposal-002"
    second["source"]["source_id"] = "source-002"
    second["source"]["source_sha256"] = SHA_B
    second["generation"]["proposal_sha256"] = SHA_D

    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    first_manifest = register_candidates([second, candidate], policy, first_output)
    second_manifest = register_candidates([candidate, second], policy, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest["contains_raw_text"] is False
    assert first_manifest["references_used_for_filtering"] is False
    assert first_manifest["labels_are_gold"] is False


def test_registry_refuses_to_replace_different_content(tmp_path, policy, candidate):
    output = tmp_path / "registry.jsonl"
    register_candidates([candidate], policy, output)
    changed = deepcopy(candidate)
    changed["generation"]["proposal_sha256"] = SHA_F

    with pytest.raises(FileExistsError, match="refusing to replace"):
        register_candidates([changed], policy, output)


def test_empty_registry_is_rejected(tmp_path, policy):
    with pytest.raises(ValueError, match="must not be empty"):
        register_candidates([], policy, tmp_path / "registry.jsonl")


def test_policy_must_forbid_all_core_evaluation_splits():
    with pytest.raises(ValueError, match="omits mandatory"):
        ContractPolicy.from_mapping(
            {
                "schema_version": 1,
                "policy_id": "unsafe",
                "allowed_splits": ["teacher_pool"],
                "forbidden_splits": ["test"],
                "forbidden_group_ids": [],
                "forbidden_source_ids": [],
                "forbidden_source_sha256": [],
                "authorized_rights_statuses": ["authorized"],
                "required_rights_uses": ["model_training"],
            }
        )
