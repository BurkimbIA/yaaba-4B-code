from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from yaaba.local_experiments import (
    assistant_completion_tokens,
    assistant_prefixes,
    ensure_resume_contract,
    prepare_chat_dataset,
    validate_prepared_assistant_masks,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "local"
sys.path.insert(0, str(SCRIPTS))

import evaluate_local  # noqa: E402
import train_mlx  # noqa: E402


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class ExactPrefixTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=False, return_dict=False):
        assert tokenize and enable_thinking is False
        assert return_dict is False
        tokens = []
        for message in messages:
            tokens += [10 if message["role"] == "user" else 20]
            tokens += [ord(char) for char in message["content"]]
            tokens += [99]
        if add_generation_prompt:
            tokens += [20]
        return tokens


class BrokenPrefixTokenizer(ExactPrefixTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=False, return_dict=False):
        result = super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            return_dict=return_dict,
        )
        return ([777] + result) if not add_generation_prompt else result


def test_multi_turn_expands_to_one_completion_per_assistant():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    prefixes = list(assistant_prefixes(messages))
    assert [len(prefix) for prefix in prefixes] == [2, 4]
    assert [prefix[-1]["content"] for prefix in prefixes] == ["a1", "a2"]


def test_assistant_mask_has_no_prompt_labels():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    encoded = assistant_completion_tokens(ExactPrefixTokenizer(), messages, 100)
    assert encoded is not None
    first_label = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
    assert encoded["labels"][:first_label] == [-100] * first_label
    assert encoded["labels"][first_label:] == encoded["input_ids"][first_label:]


def test_assistant_mask_refuses_approximate_template_boundary():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    with pytest.raises(ValueError, match="exact prefix"):
        assistant_completion_tokens(BrokenPrefixTokenizer(), messages, 100)


def test_preflight_drops_overflow_and_keeps_supervised_assistant_token(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _jsonl(data / "train.jsonl", [
        {"messages": [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "yes"},
        ]},
        {"messages": [
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "too long"},
        ]},
    ])
    report = validate_prepared_assistant_masks(
        data, ExactPrefixTokenizer(), max_length=20
    )
    assert report["splits"]["train"]["kept"] == 1
    assert report["splits"]["train"]["dropped_overflow"] == 1
    assert report["splits"]["train"]["minimum_supervised_assistant_tokens"] >= 1
    assert len((data / "train.jsonl").read_text().splitlines()) == 1


def test_prepare_is_grouped_deterministic_and_leak_checked(tmp_path):
    source = _jsonl(tmp_path / "source.jsonl", [{
        "group_id": "dialogue-1", "split": "train",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
    }])
    heldout = _jsonl(tmp_path / "heldout.jsonl", [
        {"capacite": "anglais", "text": "independent forbidden text"},
    ])
    first = prepare_chat_dataset(source, tmp_path / "one", seed=7,
                                 heldout_path=heldout)
    second = prepare_chat_dataset(source, tmp_path / "two", seed=7,
                                  heldout_path=heldout)
    assert first["assistant_examples"]["train"] == 2
    assert first["files"]["train"]["sha256"] == second["files"]["train"]["sha256"]
    rows = [json.loads(line) for line in
            (tmp_path / "one" / "train.jsonl").read_text().splitlines()]
    assert {row["_provenance"]["group_id"] for row in rows} == {"dialogue-1"}

    leaking = _jsonl(tmp_path / "leaking.jsonl", [{
        "split": "train",
        "messages": [
            {"role": "user", "content": "independent forbidden text"},
            {"role": "assistant", "content": "answer"},
        ],
    }])
    with pytest.raises(ValueError, match="held-out equal"):
        prepare_chat_dataset(leaking, tmp_path / "leak-out", seed=7,
                             heldout_path=heldout,
                             allow_ungrouped_technical_pilot=True)


def test_variant_filter_requires_an_explicit_decision(tmp_path):
    source = _jsonl(tmp_path / "source.jsonl", [{
        "variant": "new_variant", "split": "train",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }])
    heldout = _jsonl(tmp_path / "heldout.jsonl", [
        {"capacite": "anglais", "text": "independent"},
    ])
    with pytest.raises(ValueError, match="neither included nor excluded"):
        prepare_chat_dataset(
            source, tmp_path / "out", seed=1, heldout_path=heldout,
            include_variants={"kept"}, exclude_variants={"excluded"},
        )


def test_missing_group_is_only_allowed_for_an_existing_split_technical_pilot(tmp_path):
    source = _jsonl(tmp_path / "source.jsonl", [{
        "split": "train",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }])
    heldout = _jsonl(tmp_path / "heldout.jsonl", [
        {"capacite": "anglais", "text": "independent"},
    ])
    with pytest.raises(ValueError, match="refusing to invent"):
        prepare_chat_dataset(source, tmp_path / "refused", seed=1,
                             heldout_path=heldout)
    provenance = prepare_chat_dataset(
        source, tmp_path / "pilot", seed=1, heldout_path=heldout,
        allow_ungrouped_technical_pilot=True,
    )
    assert provenance["grouping_verified"] is False
    assert provenance["ungrouped_records"] == 1


def test_lineage_envelope_uses_attested_group_and_nested_record(tmp_path):
    source = _jsonl(tmp_path / "pilot.jsonl", [{
        "lineage": {
            "group_hash": "group-one", "origin_path": "source/page.jsonl",
            "origin_line": 7, "record_hash": "record-sha",
        },
        "record": {
            "split": "train", "variant": "mos_mos",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        },
    }])
    heldout = _jsonl(tmp_path / "heldout.jsonl", [
        {"capacite": "anglais", "text": "independent"},
    ])
    provenance = prepare_chat_dataset(
        source, tmp_path / "prepared", seed=1, heldout_path=heldout,
        group_field="lineage.group_hash", include_variants={"mos_mos"},
        exclude_variants=set(),
    )
    assert provenance["lineage_envelopes"] == 1
    assert provenance["grouping_verified"] is True
    row = json.loads((tmp_path / "prepared" / "train.jsonl").read_text())
    assert row["_provenance"] == {
        "assistant_turn": 0, "group_id": "group-one", "origin_line": 7,
        "origin_path": "source/page.jsonl", "record_hash": "record-sha",
        "source_line": 1,
    }


def test_resume_contract_rejects_changed_data(tmp_path):
    path = tmp_path / "resume.json"
    ensure_resume_contract(path, {"data": "one"}, resume=False)
    assert ensure_resume_contract(path, {"data": "one"}, resume=True)
    with pytest.raises(ValueError, match="contract changed"):
        ensure_resume_contract(path, {"data": "two"}, resume=True)


def test_mlx_config_is_bounded_and_masks_prompt(tmp_path):
    args = train_mlx.parser().parse_args([
        "--model", str(tmp_path), "--data", str(tmp_path / "data"),
        "--output", str(tmp_path / "out"), "--heldout", str(tmp_path / "held"),
        "--max-steps", "9", "--save-every", "4",
    ])
    config, _ = train_mlx._mlx_config(args, tmp_path, tmp_path / "out", 4, None)
    assert config["iters"] == 5
    assert config["mask_prompt"] is True
    assert config["save_every"] == 4
    assert config["grad_checkpoint"] is True
    assert config["steps_per_report"] == 5
    assert config["clear_cache_threshold"] == 2 * 1024**3


def test_explicit_mock_smokes_write_honest_artifacts(tmp_path):
    training = tmp_path / "training"
    assert train_mlx.main([
        "--model", "ignored", "--data", "ignored", "--output", str(training),
        "--mock-smoke", "--max-steps", "2", "--walltime-minutes", "1",
    ]) == 0
    train_manifest = json.loads((training / "manifest.json").read_text())
    assert train_manifest["evidence_kind"] == "TECHNICAL_MOCK"
    assert "no model quality evidence" in train_manifest["claim_scope"]

    questions = _jsonl(tmp_path / "questions.jsonl", [{
        "id": "q1", "prompt": "What is this?", "reference": "A test."
    }])
    model = tmp_path / "mock-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"explicit-mock"}')
    evaluation = tmp_path / "evaluation"
    assert evaluate_local.main([
        "--backend", "mock", "--model", str(model),
        "--model-revision", "explicit-mock", "--questions", str(questions),
        "--output", str(evaluation), "--max-samples", "1",
    ]) == 0
    eval_manifest = json.loads((evaluation / "manifest.json").read_text())
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert eval_manifest["evidence_kind"] == "TECHNICAL_MOCK"
    assert metrics["completed"] == 1
    assert "not a human quality score" in metrics["metric_scope"]
