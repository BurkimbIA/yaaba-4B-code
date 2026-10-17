from __future__ import annotations

import json
from copy import deepcopy

import pytest

import yaaba.distillation_contract as contract_module
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
REV_A = "1" * 40
REV_B = "2" * 40


def policy_mapping() -> dict:
    return {
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
        "required_group_kinds": ["document", "page"],
        "allowed_datasets": [
            {
                "dataset_id": "authorized-private-source",
                "dataset_revision": REV_A,
            }
        ],
        "allowed_generation_contracts": [
            {
                "contract_id": "teacher-run-1",
                "teacher": {
                    "model_id": "teacher/model",
                    "revision": REV_B,
                    "model_sha256": SHA_C,
                    "tokenizer_sha256": SHA_D,
                },
                "generation": {
                    "run_id": "run-001",
                    "manifest_sha256": SHA_E,
                    "code_sha256": SHA_A,
                    "prompt_sha256": SHA_B,
                    "decoding_sha256": SHA_D,
                },
            }
        ],
        "authorized_rights_statuses": ["authorized"],
        "required_rights_uses": ["teacher_generation", "model_training"],
    }


@pytest.fixture
def policy() -> ContractPolicy:
    return ContractPolicy.from_mapping(policy_mapping())


@pytest.fixture
def candidate() -> dict:
    return {
        "candidate_id": "proposal-001",
        "source": {
            "source_id": "source-001",
            "source_parent_ids": ["parent-001"],
            "group_ids": {"document": "document-001", "page": "page-001"},
            "source_sha256": SHA_A,
            "split": "teacher_pool",
            "provenance": {
                "dataset_id": "authorized-private-source",
                "dataset_revision": REV_A,
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
            "revision": REV_B,
            "model_sha256": SHA_C,
            "tokenizer_sha256": SHA_D,
        },
        "generation": {
            "contract_id": "teacher-run-1",
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


@pytest.mark.parametrize(
    "split",
    [
        "dev",
        "development",
        "dev32",
        "VAL",
        "validation",
        "test",
        "gold",
        "held_out",
    ],
)
def test_evaluation_or_gold_split_is_rejected(policy, candidate, split):
    candidate["source"]["split"] = split

    record = evaluate_candidate(candidate, policy)

    assert record["decision"]["status"] == "rejected"
    assert "split_forbidden" in reasons(record)


def test_forbidden_group_is_rejected_exactly(policy, candidate):
    candidate["source"]["group_ids"]["page"] = "eval-page-9"

    record = evaluate_candidate(candidate, policy)

    assert "group_forbidden" in reasons(record)


def test_similar_but_not_exact_group_is_not_rejected(policy, candidate):
    candidate["source"]["group_ids"]["page"] = "Eval-Page-9"

    record = evaluate_candidate(candidate, policy)

    assert record["decision"]["status"] == "eligible_for_review"


def test_forbidden_source_parent_is_rejected(policy, candidate):
    candidate["source"]["source_parent_ids"] = ["eval-page-9"]

    record = evaluate_candidate(candidate, policy)

    assert "group_forbidden" in reasons(record)


def test_required_group_role_cannot_be_omitted(policy, candidate):
    del candidate["source"]["group_ids"]["document"]

    record = evaluate_candidate(candidate, policy)

    assert "missing_required_group_kind" in reasons(record)


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


def test_mobile_revision_and_identity_drift_are_rejected(policy, candidate):
    candidate["teacher"]["revision"] = "main"
    candidate["generation"]["prompt_sha256"] = SHA_F

    record = evaluate_candidate(candidate, policy)

    assert "invalid_teacher_revision" in reasons(record)
    assert "generation_contract_identity_mismatch" in reasons(record)


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


def test_technical_reject_reasons_are_deterministic(policy, candidate):
    candidate["screening"]["reject_reasons"] = ["truncated", "empty_proposal"]
    first = evaluate_candidate(candidate, policy)
    candidate["screening"]["reject_reasons"].reverse()
    second = evaluate_candidate(candidate, policy)

    assert first == second
    assert first["decision"]["status"] == "rejected"


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
    assert first_manifest["raw_content_fields_permitted"] is False
    assert first_manifest["references_used_for_filtering"] is False
    assert first_manifest["labels_are_gold"] is False
    assert first_manifest["generation_contracts_observed"][0]["records"] == 2


def test_registry_refuses_to_replace_different_content(tmp_path, policy, candidate):
    output = tmp_path / "registry.jsonl"
    register_candidates([candidate], policy, output)
    changed = deepcopy(candidate)
    changed["generation"]["proposal_sha256"] = SHA_F

    with pytest.raises(FileExistsError, match="refusing to replace"):
        register_candidates([changed], policy, output)


def test_concurrent_writer_cannot_be_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "registry.jsonl"

    def concurrent_link(_source, destination):
        destination = type(output)(destination)
        destination.write_bytes(b"concurrent\n")
        raise FileExistsError(destination)

    monkeypatch.setattr(contract_module.os, "link", concurrent_link)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        contract_module._write_if_same_or_new(output, b"ours\n")
    assert output.read_bytes() == b"concurrent\n"


def test_empty_registry_is_rejected(tmp_path, policy):
    with pytest.raises(ValueError, match="must not be empty"):
        register_candidates([], policy, tmp_path / "registry.jsonl")


def test_policy_cannot_allow_an_evaluation_alias():
    raw = policy_mapping()
    raw["allowed_splits"] = ["development"]
    raw["forbidden_splits"] = []

    with pytest.raises(ValueError, match="intersects an evaluation split"):
        ContractPolicy.from_mapping(raw)


def test_policy_cannot_allow_a_prefixed_development_split():
    raw = policy_mapping()
    raw["allowed_splits"] = ["dev32_private"]

    with pytest.raises(ValueError, match="intersects an evaluation split"):
        ContractPolicy.from_mapping(raw)


def test_policy_cannot_weaken_required_distillation_rights():
    raw = policy_mapping()
    raw["required_rights_uses"] = ["reading"]

    parsed = ContractPolicy.from_mapping(raw)

    assert {"teacher_generation", "model_training", "reading"} == set(
        parsed.required_rights_uses
    )


def test_policy_rejects_mobile_teacher_revision():
    raw = policy_mapping()
    raw["allowed_generation_contracts"][0]["teacher"]["revision"] = "main"

    with pytest.raises(ValueError, match="immutable revision"):
        ContractPolicy.from_mapping(raw)


def test_policy_rejects_mobile_dataset_revision():
    raw = policy_mapping()
    raw["allowed_datasets"][0]["dataset_revision"] = "latest"

    with pytest.raises(ValueError, match="immutable revision"):
        ContractPolicy.from_mapping(raw)


def test_identifier_fields_cannot_retain_free_text(policy, candidate):
    private_sentence = "une phrase privée avec espaces"
    candidate["source"]["source_id"] = private_sentence

    record = evaluate_candidate(candidate, policy)

    assert record["decision"]["status"] == "rejected"
    assert "invalid_identifier" in reasons(record)
    assert private_sentence not in json.dumps(record)


def test_multiple_explicit_teacher_contracts_are_aggregated(candidate, tmp_path):
    raw = policy_mapping()
    second_contract = deepcopy(raw["allowed_generation_contracts"][0])
    second_contract["contract_id"] = "teacher-run-2"
    second_contract["teacher"]["model_id"] = "second/teacher"
    second_contract["teacher"]["revision"] = "3" * 40
    second_contract["teacher"]["model_sha256"] = SHA_F
    second_contract["generation"]["run_id"] = "run-002"
    second_contract["generation"]["manifest_sha256"] = SHA_C
    raw["allowed_generation_contracts"].append(second_contract)
    policy = ContractPolicy.from_mapping(raw)
    second = deepcopy(candidate)
    second["candidate_id"] = "proposal-002"
    second["teacher"] = deepcopy(second_contract["teacher"])
    second["generation"].update(second_contract["generation"])
    second["generation"]["contract_id"] = "teacher-run-2"
    second["generation"]["proposal_sha256"] = SHA_F

    manifest = register_candidates(
        [candidate, second], policy, tmp_path / "registry.jsonl"
    )

    observed = manifest["generation_contracts_observed"]
    assert [item["contract_id"] for item in observed] == [
        "teacher-run-1",
        "teacher-run-2",
    ]
