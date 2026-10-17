from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from yaaba.evaluation_protocol import (
    EvaluationProtocolError,
    authorize_evaluation,
    finalize_evaluation,
    role_assignment_sha256,
)
from yaaba.local_experiments import canonical_hash, sha256_file

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "local"
sys.path.insert(0, str(SCRIPTS))

import evaluate_local  # noqa: E402

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SYSTEM_SHA = "d" * 64


def _split(
    split_id: str,
    role: str,
    data_sha256: str,
    *,
    exposure_history: list[dict] | None = None,
) -> dict:
    return {
        "split_id": split_id,
        "role": role,
        "data_sha256": data_sha256,
        "split_manifest_sha256": SHA_B,
        "group_evidence_sha256": SHA_C,
        "group_ids": [f"group:{split_id}"],
        "expected_record_count": 1,
        "exposure_history": exposure_history or [],
    }


def _protocol(path: Path, splits: list[dict], **overrides) -> Path:
    value = {
        "schema_version": 1,
        "protocol_id": "eval-protocol-v1",
        "status": "frozen",
        "systems_frozen": True,
        "allowed_system_contract_sha256": [SYSTEM_SHA],
        "splits": splits,
        "role_assignment_sha256": role_assignment_sha256(splits),
        **overrides,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _authorize(path: Path | None, split_id: str | None, purpose: str, data=SHA_A):
    return authorize_evaluation(
        protocol_path=path,
        expected_protocol_sha256=sha256_file(path) if path else None,
        split_id=split_id,
        purpose=purpose,
        questions_sha256=data,
        questions_record_count=1,
        sample_limit=1,
        system_contract_sha256=SYSTEM_SHA,
    )


def test_unregistered_input_is_diagnostic_only():
    receipt = _authorize(None, None, "diagnostic")
    assert receipt["split_role"] == "diagnostic_exposed"
    assert receipt["role_source"] == "default_fail_closed"
    assert receipt["final_test_authorized"] is False
    with pytest.raises(EvaluationProtocolError, match="require a frozen"):
        _authorize(None, None, "final_test")


def test_final_test_accepts_exact_sealed_split_and_frozen_system(tmp_path):
    protocol = _protocol(
        tmp_path / "protocol.json", [_split("final-v1", "test_sealed", SHA_A)]
    )
    receipt = _authorize(protocol, "final-v1", "final_test")
    assert receipt["final_test_authorized"] is True
    assert receipt["final_test_completed"] is False
    assert receipt["independence_status"] == "protocol_local_authorization_only"
    assert receipt["split_role"] == "test_sealed"
    assert receipt["protocol"]["sha256"] == sha256_file(protocol)


@pytest.mark.parametrize("role", ["training", "development_frozen"])
def test_non_test_roles_cannot_be_claimed_as_final(tmp_path, role):
    protocol = _protocol(tmp_path / "protocol.json", [_split("split-v1", role, SHA_A)])
    match = "training data" if role == "training" else "requires role test_sealed"
    with pytest.raises(EvaluationProtocolError, match=match):
        _authorize(protocol, "split-v1", "final_test")


def test_development_exposure_never_becomes_final_test(tmp_path):
    history = [
        {
            "purpose": "model_selection",
            "observed_at": "2026-10-17T18:00:00+02:00",
            "system_or_protocol_sha256": SYSTEM_SHA,
        }
    ]
    protocol = _protocol(
        tmp_path / "protocol.json",
        [_split("renamed-final", "test_sealed", SHA_A, exposure_history=history)],
    )
    with pytest.raises(EvaluationProtocolError, match="prior exposure"):
        _authorize(protocol, "renamed-final", "final_test")


def test_same_content_cannot_be_aliased_across_roles(tmp_path):
    protocol = _protocol(
        tmp_path / "protocol.json",
        [
            _split("dev-v1", "development_frozen", SHA_A),
            _split("test-v1", "test_sealed", SHA_A),
        ],
    )
    with pytest.raises(EvaluationProtocolError, match="assigned more than once"):
        _authorize(protocol, "test-v1", "final_test")


def test_declared_group_cannot_cross_roles(tmp_path):
    development = _split("dev-v1", "development_frozen", SHA_A)
    final = _split("test-v1", "test_sealed", SHA_B)
    final["group_ids"] = development["group_ids"]
    protocol = _protocol(tmp_path / "protocol.json", [development, final])
    with pytest.raises(EvaluationProtocolError, match="group_id is assigned"):
        _authorize(protocol, "test-v1", "final_test", data=SHA_B)


def test_final_test_requires_exact_questions_and_system_contract(tmp_path):
    protocol = _protocol(
        tmp_path / "protocol.json", [_split("final-v1", "test_sealed", SHA_A)]
    )
    with pytest.raises(EvaluationProtocolError, match="questions SHA-256"):
        _authorize(protocol, "final-v1", "final_test", data=SHA_B)
    with pytest.raises(EvaluationProtocolError, match="frozen allowlist"):
        authorize_evaluation(
            protocol_path=protocol,
            expected_protocol_sha256=sha256_file(protocol),
            split_id="final-v1",
            purpose="final_test",
            questions_sha256=SHA_A,
            questions_record_count=1,
            sample_limit=1,
            system_contract_sha256="e" * 64,
        )


def test_protocol_requires_an_out_of_band_content_pin(tmp_path):
    protocol = _protocol(
        tmp_path / "protocol.json", [_split("final-v1", "test_sealed", SHA_A)]
    )
    with pytest.raises(EvaluationProtocolError, match="independently pinned"):
        authorize_evaluation(
            protocol_path=protocol,
            expected_protocol_sha256=None,
            split_id="final-v1",
            purpose="final_test",
            questions_sha256=SHA_A,
            questions_record_count=1,
            sample_limit=1,
            system_contract_sha256=SYSTEM_SHA,
        )
    with pytest.raises(EvaluationProtocolError, match="does not match"):
        authorize_evaluation(
            protocol_path=protocol,
            expected_protocol_sha256="f" * 64,
            split_id="final-v1",
            purpose="final_test",
            questions_sha256=SHA_A,
            questions_record_count=1,
            sample_limit=1,
            system_contract_sha256=SYSTEM_SHA,
        )


def test_final_test_requires_complete_input_and_successful_completion(tmp_path):
    sealed = _split("final-v1", "test_sealed", SHA_A)
    sealed["expected_record_count"] = 2
    protocol = _protocol(tmp_path / "protocol.json", [sealed])
    with pytest.raises(EvaluationProtocolError, match="record count"):
        authorize_evaluation(
            protocol_path=protocol,
            expected_protocol_sha256=sha256_file(protocol),
            split_id="final-v1",
            purpose="final_test",
            questions_sha256=SHA_A,
            questions_record_count=1,
            sample_limit=2,
            system_contract_sha256=SYSTEM_SHA,
        )
    receipt = authorize_evaluation(
        protocol_path=protocol,
        expected_protocol_sha256=sha256_file(protocol),
        split_id="final-v1",
        purpose="final_test",
        questions_sha256=SHA_A,
        questions_record_count=2,
        sample_limit=2,
        system_contract_sha256=SYSTEM_SHA,
    )
    with pytest.raises(EvaluationProtocolError, match="incomplete"):
        finalize_evaluation(receipt, completed=1, stop_reason="walltime_limit")
    with pytest.raises(EvaluationProtocolError, match="sample_limit"):
        authorize_evaluation(
            protocol_path=protocol,
            expected_protocol_sha256=sha256_file(protocol),
            split_id="final-v1",
            purpose="final_test",
            questions_sha256=SHA_A,
            questions_record_count=2,
            sample_limit=1,
            system_contract_sha256=SYSTEM_SHA,
        )
    completed = finalize_evaluation(
        receipt, completed=2, stop_reason="sample_limit_or_input_end"
    )
    assert completed["final_test_completed"] is True
    assert completed["independence_status"] == (
        "protocol_declared_complete_metadata_consistent_not_empirically_proven"
    )


def test_role_assignment_hash_is_verified(tmp_path):
    protocol = _protocol(
        tmp_path / "protocol.json",
        [_split("dev-v1", "development_frozen", SHA_A)],
        role_assignment_sha256="f" * 64,
    )
    with pytest.raises(EvaluationProtocolError, match="does not match"):
        _authorize(protocol, "dev-v1", "development")


def test_evaluator_legacy_run_records_non_independent_diagnostic(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"id": "q1", "prompt": "question", "reference": "answer"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    assert (
        evaluate_local.main(
            [
                "--backend",
                "mock",
                "--model",
                "mock-model",
                "--model-revision",
                "mock-revision",
                "--questions",
                str(questions),
                "--output",
                str(output),
                "--max-samples",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    receipt = manifest["evaluation_protocol"]
    assert receipt["purpose"] == "diagnostic"
    assert receipt["final_test_authorized"] is False
    assert manifest["system_contract"]["metric_versions"]["sacrebleu"]
    assert set(manifest["system_contract"]["component_sha256"]) == {
        "evaluate_local.py",
        "evaluation_protocol.py",
        "local_experiments.py",
    }
    assert manifest["system_contract_sha256"] == canonical_hash(
        manifest["system_contract"]
    )


def test_evaluator_system_contract_covers_runtime_and_adapter_config(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "q1", "prompt": "question"}) + "\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"fixed-weights")
    (adapter / "adapter_config.json").write_text('{"rank": 4}')

    def run(output: Path) -> dict:
        assert (
            evaluate_local.main(
                [
                    "--backend",
                    "mock",
                    "--model",
                    "mock-model",
                    "--model-revision",
                    "mock-revision",
                    "--adapter",
                    str(adapter),
                    "--questions",
                    str(questions),
                    "--output",
                    str(output),
                    "--max-samples",
                    "1",
                ]
            )
            == 0
        )
        return json.loads((output / "manifest.json").read_text())

    first = run(tmp_path / "first")
    (adapter / "adapter_config.json").write_text('{"rank": 8}')
    second = run(tmp_path / "second")
    assert first["system_contract_sha256"] != second["system_contract_sha256"]
    assert first["system_contract"]["backend"] == "mock"
    assert set(first["system_contract"]["runtime_versions"]) == {
        "mlx",
        "mlx-lm",
        "torch",
        "transformers",
        "peft",
        "sacrebleu",
    }


def test_evaluator_claims_completion_only_after_full_success(tmp_path, monkeypatch):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "q1", "prompt": "question"}) + "\n")
    common = [
        "--backend",
        "mock",
        "--model",
        "mock-model",
        "--model-revision",
        "mock-revision",
        "--questions",
        str(questions),
        "--max-samples",
        "1",
    ]
    diagnostic_output = tmp_path / "diagnostic"
    assert evaluate_local.main([*common, "--output", str(diagnostic_output)]) == 0
    diagnostic = json.loads((diagnostic_output / "manifest.json").read_text())
    sealed = _split("sealed-v1", "test_sealed", sha256_file(questions))
    protocol = _protocol(
        tmp_path / "protocol.json",
        [sealed],
        allowed_system_contract_sha256=[diagnostic["system_contract_sha256"]],
    )
    final_args = [
        *common,
        "--evaluation-purpose",
        "final_test",
        "--evaluation-protocol",
        str(protocol),
        "--evaluation-protocol-sha256",
        sha256_file(protocol),
        "--split-id",
        "sealed-v1",
    ]

    failed_output = tmp_path / "failed"
    with monkeypatch.context() as patch:
        patch.setattr(
            evaluate_local,
            "_run_evaluation",
            lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
        )
        with pytest.raises(RuntimeError, match="synthetic failure"):
            evaluate_local.main([*final_args, "--output", str(failed_output)])
    failed = json.loads((failed_output / "manifest.json").read_text())
    assert failed["evaluation_protocol"]["final_test_authorized"] is True
    assert failed["evaluation_protocol"]["final_test_completed"] is False

    success_output = tmp_path / "success"
    assert evaluate_local.main([*final_args, "--output", str(success_output)]) == 0
    success = json.loads((success_output / "manifest.json").read_text())
    assert success["evaluation_protocol"]["final_test_completed"] is True
    assert success["evaluation_protocol"]["independence_status"].startswith(
        "protocol_declared_complete"
    )
