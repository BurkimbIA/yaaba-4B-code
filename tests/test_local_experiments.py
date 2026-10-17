from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
import mlx_lora_bounded  # noqa: E402
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


class ThinkingDefaultTokenizer(ExactPrefixTokenizer):
    def __init__(self):
        self.thinking_values = []

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=False,
        enable_thinking=True, return_dict=False, tools=None,
    ):
        self.thinking_values.append(enable_thinking)
        tokens = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            return_dict=return_dict,
        )
        if enable_thinking and add_generation_prompt:
            tokens += [701, 702, 703, 704]
        return tokens


class FailingThinkingTokenizer(ThinkingDefaultTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        if "enable_thinking" in kwargs:
            raise ValueError("template rendering failed")
        return super().apply_chat_template(*args, **kwargs)


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


def test_mlx_chat_mask_disables_thinking_and_matches_preflight():
    tokenizer = ThinkingDefaultTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    dataset = type(
        "Dataset", (), {
            "chat_key": "messages", "mask_prompt": True,
            "tokenizer": tokenizer,
        },
    )()
    tokens, offset = mlx_lora_bounded._exact_chat_process(
        dataset, {"messages": messages}
    )
    expected = assistant_completion_tokens(tokenizer, messages, 100)
    assert expected is not None
    expected_offset = next(
        index for index, label in enumerate(expected["labels"]) if label != -100
    )
    assert tokens == expected["input_ids"]
    assert offset == expected_offset
    assert all(value is False for value in tokenizer.thinking_values)


def test_mlx_chat_mask_does_not_hide_template_value_error():
    tokenizer = FailingThinkingTokenizer()
    with pytest.raises(ValueError, match="template rendering failed"):
        mlx_lora_bounded._apply_chat_template(
            tokenizer, [], return_dict=False
        )
    with pytest.raises(ValueError, match="template rendering failed"):
        assistant_completion_tokens(
            tokenizer,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            100,
        )


def test_mlx_chat_mask_does_not_fallback_when_kwarg_is_unsupported():
    class UnsupportedTokenizer:
        def apply_chat_template(self, messages, *, return_dict=False):
            return []

    with pytest.raises(TypeError, match="enable_thinking"):
        mlx_lora_bounded._apply_chat_template(
            UnsupportedTokenizer(), [], return_dict=False
        )


def test_evaluation_template_disables_thinking():
    class EvaluationTokenizer:
        def __init__(self):
            self.thinking_values = []

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, enable_thinking
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            self.thinking_values.append(enable_thinking)
            return "rendered prompt"

    tokenizer = EvaluationTokenizer()
    messages = [{"role": "user", "content": "question"}]
    prompt, options = evaluate_local._templated(tokenizer, messages)
    assert isinstance(prompt, str)
    assert options == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert tokenizer.thinking_values == [False]


def test_evaluation_template_does_not_hide_renderer_failure():
    tokenizer = FailingThinkingTokenizer()
    messages = [{"role": "user", "content": "question"}]
    with pytest.raises(ValueError, match="template rendering failed"):
        evaluate_local._templated(tokenizer, messages)


def test_evaluation_template_does_not_fallback_when_kwarg_is_unsupported():
    class UnsupportedTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return []

    messages = [{"role": "user", "content": "question"}]
    with pytest.raises(TypeError, match="enable_thinking"):
        evaluate_local._templated(UnsupportedTokenizer(), messages)


def test_mlx_load_dataset_uses_exact_mask_and_source_order_before_seed(
    tmp_path, monkeypatch,
):
    pytest.importorskip("mlx.core")
    import numpy as np
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset, load_dataset
    from mlx_lm.tuner.trainer import iterate_batches

    rows = [
        {
            "messages": [
                {"role": "user", "content": chr(65 + index)},
                {"role": "assistant", "content": "answer"},
            ],
            "_provenance": {"source_index": index},
        }
        for index in range(5)
    ]
    _jsonl(tmp_path / "train.jsonl", rows)
    tokenizer = ThinkingDefaultTokenizer()
    config = SimpleNamespace(
        data=str(tmp_path), train=True, test=False, mask_prompt=True,
        hf_dataset=False,
    )
    with monkeypatch.context() as patch:
        # Register the original method for automatic restoration before the
        # production installer replaces it process-wide.
        patch.setattr(ChatDataset, "process", ChatDataset.process)
        mlx_lora_bounded._install_exact_chat_mask()
        train, _, _ = load_dataset(config, tokenizer)
        for index, row in enumerate(rows):
            tokens, offset = train.process(row)
            expected = assistant_completion_tokens(tokenizer, row["messages"], 100)
            assert expected is not None
            expected_offset = next(
                i for i, label in enumerate(expected["labels"]) if label != -100
            )
            assert tokens == expected["input_ids"]
            assert offset == expected_offset
            assert tokens[1] == 65 + index

        cached = CacheDataset(train)
        assert [cached.itemlen(index) for index in range(5)] == [2] * 5
        batches = iterate_batches(
            cached, batch_size=1, max_seq_length=100, loop=False, seed=7
        )
        observed = [int(batch[0][0, 1].item()) - 65 for batch in batches]
        expected_order = np.random.RandomState(7).permutation(5).tolist()
        assert observed == expected_order


def test_assistant_target_mask_excludes_only_fill_position_for_variable_lengths():
    steps = np.arange(1, 9)
    lengths = np.array([[3, 6], [2, 7]])
    mask = mlx_lora_bounded._assistant_target_mask(steps, lengths, np)

    assert np.flatnonzero(mask[0]).tolist() == [2, 3, 4]
    assert np.flatnonzero(mask[1]).tolist() == [1, 2, 3, 4, 5]


def test_mlx_real_batch_loss_contract_keeps_internal_zero_and_eos(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset, load_dataset
    from mlx_lm.tuner.trainer import iterate_batches

    class TerminatedTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=False,
            tools=None,
        ):
            assert tokenize and enable_thinking is False and return_dict is False
            assert tools is None
            tokens = [10, 11]
            if add_generation_prompt:
                return tokens + [20]
            content = [0, 21] if messages[-1]["content"] == "long" else [21]
            return tokens + [20] + content + [99, 198]

    rows = [
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": answer},
            ]
        }
        for answer in ("short", "long")
    ]
    _jsonl(tmp_path / "train.jsonl", rows)
    config = SimpleNamespace(
        data=str(tmp_path),
        train=True,
        test=False,
        mask_prompt=True,
        hf_dataset=False,
    )
    with monkeypatch.context() as patch:
        patch.setattr(ChatDataset, "process", ChatDataset.process)
        mlx_lora_bounded._install_exact_chat_mask()
        train, _, _ = load_dataset(config, TerminatedTokenizer())
        batch, lengths = next(
            iterate_batches(
                CacheDataset(train),
                batch_size=2,
                max_seq_length=32,
                loop=False,
                seed=7,
            )
        )

    batch_np = np.array(batch)
    lengths_np = np.array(lengths)
    targets = batch_np[:, 1:]
    steps = np.arange(1, targets.shape[1] + 1)
    mask = mlx_lora_bounded._assistant_target_mask(steps, lengths_np, np)
    selected = [targets[index][mask[index]].tolist() for index in range(2)]

    assert sorted(selected, key=len) == [[21, 99, 198], [0, 21, 99, 198]]
    for index, (offset, length) in enumerate(lengths_np):
        assert selected[index] == batch_np[index, offset:length].tolist()
        assert batch_np[index, length] == 0
        assert len(selected[index]) == length - offset

    class ZeroLogitModel:
        def __call__(self, inputs):
            return mx.zeros((*inputs.shape, 256))

    loss, token_count = mlx_lora_bounded._assistant_loss_without_batch_fill(
        ZeroLogitModel(), batch, lengths
    )
    mx.eval(loss, token_count)
    assert int(token_count.item()) == sum(len(tokens) for tokens in selected)
    assert np.isfinite(float(loss.item()))


def test_scoped_assistant_loss_injects_and_restores_on_success_and_failure():
    calls = {"train": [], "evaluate": []}
    state = {"fail": False}

    def original_train(
        model,
        optimizer,
        train_dataset,
        val_dataset=None,
        args=None,
        loss=None,
    ):
        calls["train"].append(loss)
        if state["fail"]:
            raise RuntimeError("training failed")
        return "trained"

    def original_evaluate(
        model,
        dataset,
        batch_size,
        num_batches,
        max_seq_length=2048,
        loss=None,
    ):
        calls["evaluate"].append(loss)
        return 1.0

    lora_module = SimpleNamespace(
        train=original_train, evaluate=original_evaluate
    )
    dataset = [([10, 20, 99], 1)]
    training_args = SimpleNamespace(max_seq_length=8)
    with mlx_lora_bounded._scoped_assistant_loss(lora_module):
        assert lora_module.train(
            None, None, dataset, dataset, args=training_args
        ) == "trained"
        assert lora_module.evaluate(None, dataset, 1, 1, max_seq_length=8) == 1.0
        assert calls == {
            "train": [mlx_lora_bounded._assistant_loss_without_batch_fill],
            "evaluate": [mlx_lora_bounded._assistant_loss_without_batch_fill],
        }
    assert lora_module.train is original_train
    assert lora_module.evaluate is original_evaluate

    state["fail"] = True
    with pytest.raises(RuntimeError, match="training failed"):
        with mlx_lora_bounded._scoped_assistant_loss(lora_module):
            lora_module.train(None, None, dataset, dataset, args=training_args)
    assert lora_module.train is original_train
    assert lora_module.evaluate is original_evaluate


def test_scoped_assistant_loss_refuses_an_explicit_different_loss():
    def original_train(
        model,
        optimizer,
        train_dataset,
        val_dataset=None,
        args=None,
        loss=None,
    ):
        return loss

    def original_evaluate(
        model,
        dataset,
        batch_size,
        num_batches,
        max_seq_length=2048,
        loss=None,
    ):
        return loss

    lora_module = SimpleNamespace(
        train=original_train, evaluate=original_evaluate
    )
    dataset = [([10, 20, 99], 1)]
    with mlx_lora_bounded._scoped_assistant_loss(lora_module):
        with pytest.raises(ValueError, match="explicitly supplied"):
            lora_module.train(
                None,
                None,
                dataset,
                args=SimpleNamespace(max_seq_length=8),
                loss=lambda: None,
            )
        with pytest.raises(ValueError, match="explicitly supplied"):
            lora_module.evaluate(
                None, dataset, 1, 1, max_seq_length=8, loss=lambda: None
            )


@pytest.mark.parametrize(
    "mode", ["omitted", "keyword_none", "positional_none", "custom_identical"]
)
def test_scoped_assistant_loss_normalizes_supported_call_forms(mode):
    def original_train(
        model,
        optimizer,
        train_dataset,
        val_dataset=None,
        args=None,
        loss=None,
    ):
        return loss

    def original_evaluate(
        model,
        dataset,
        batch_size,
        num_batches,
        max_seq_length=2048,
        loss=None,
    ):
        return loss

    lora_module = SimpleNamespace(
        train=original_train, evaluate=original_evaluate
    )
    dataset = [([10, 20, 99], 1)]
    training_args = SimpleNamespace(max_seq_length=8)
    custom = mlx_lora_bounded._assistant_loss_without_batch_fill
    with mlx_lora_bounded._scoped_assistant_loss(lora_module):
        if mode == "omitted":
            train_loss = lora_module.train(
                None, None, dataset, dataset, args=training_args
            )
            test_loss = lora_module.evaluate(
                None, dataset, 1, 1, max_seq_length=8
            )
        elif mode == "keyword_none":
            train_loss = lora_module.train(
                None, None, dataset, dataset, args=training_args, loss=None
            )
            test_loss = lora_module.evaluate(
                None, dataset, 1, 1, max_seq_length=8, loss=None
            )
        elif mode == "positional_none":
            train_loss = lora_module.train(
                None, None, dataset, dataset, training_args, None
            )
            test_loss = lora_module.evaluate(None, dataset, 1, 1, 8, None)
        else:
            train_loss = lora_module.train(
                None, None, dataset, dataset, training_args, custom
            )
            test_loss = lora_module.evaluate(None, dataset, 1, 1, 8, custom)
    assert train_loss is custom
    assert test_loss is custom


@pytest.mark.parametrize(
    ("tokens", "offset", "max_seq_length"),
    [
        ([10, 20], 2, 8),
        ([10, 20, 30, 40], 3, 3),
        ([10], 0, 8),
    ],
)
def test_loss_dataset_validation_rejects_rows_without_real_target(
    tokens, offset, max_seq_length
):
    with pytest.raises(ValueError, match="no real loss target"):
        mlx_lora_bounded._validate_loss_dataset(
            [(tokens, offset)], max_seq_length, "train"
        )


def test_loss_dataset_validation_accepts_internal_zero_target():
    mlx_lora_bounded._validate_loss_dataset(
        [([10, 0, 99, 198], 1)], 8, "train"
    )


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
    # Public max_steps is optimizer updates; MLX iters is micro-batches.
    assert config["iters"] == 5 * args.grad_accumulation
    assert config["mask_prompt"] is True
    assert config["save_every"] == 4 * args.grad_accumulation
    assert config["grad_checkpoint"] is True
    assert config["steps_per_report"] == 5 * args.grad_accumulation
    assert config["clear_cache_threshold"] == 2 * 1024**3
    assert config["val_batches"] == 0
    valid = tmp_path / "out" / "data" / "valid.jsonl"
    valid.parent.mkdir(parents=True)
    valid.write_text('{}\n', encoding="utf-8")
    config_with_validation, _ = train_mlx._mlx_config(
        args, tmp_path, tmp_path / "out", 4, None
    )
    assert config_with_validation["val_batches"] == -1


def test_mlx_checkpoint_counts_only_complete_optimizer_updates(tmp_path):
    segment = tmp_path / "adapters" / "segment-0000002"
    segment.mkdir(parents=True)
    (segment / "0000007_adapters.safetensors").touch()
    assert train_mlx._latest_segment_checkpoint(tmp_path, 8) == (None, 0)
    complete = segment / "0000008_adapters.safetensors"
    complete.touch()
    assert train_mlx._latest_segment_checkpoint(tmp_path, 8) == (complete, 3)


def test_mlx_032_table_metrics_are_parsed_as_optimizer_steps(tmp_path):
    log = tmp_path / "mlx.log"
    log.write_text(
        "\x1b[1m   8    \x1b[32m4.151 ▼\x1b[0m       3      0.0k\n"
        "  16    3.070 ▼       40      0.1k\n",
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.jsonl"
    train_mlx._metrics_from_log(log, metrics, offset=2, grad_accumulation=8)
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    losses = [row for row in rows if row["name"] == "train_loss"]
    assert [(row["step"], row["value"]) for row in losses] == [
        (3, 4.151), (4, 3.070),
    ]


def test_tracked_resume_is_coupled_to_persisted_run_id(tmp_path):
    args = train_mlx.parser().parse_args([
        "--model", str(tmp_path), "--data", str(tmp_path / "data"),
        "--output", str(tmp_path), "--heldout", str(tmp_path / "held"),
        "--resume", "--track",
    ])
    (tmp_path / "tracking_state.json").write_text(
        json.dumps({"mlflow_run_id": "persisted-id"}), encoding="utf-8"
    )
    train_mlx._couple_tracking_resume(args, tmp_path)
    assert args.mlflow_run_id == "persisted-id"
    args.mlflow_run_id = "different-id"
    with pytest.raises(ValueError, match="differs"):
        train_mlx._couple_tracking_resume(args, tmp_path)


@pytest.mark.parametrize("status", [
    "process_failed", "rss_memory_limit", "system_memory_floor",
])
def test_failed_result_raises_inside_tracking_context(status):
    with pytest.raises(train_mlx.TrainingRunFailed, match=status):
        train_mlx._raise_for_failed_result({
            "status": status, "latest_adapter": "checkpoint",
        })


@pytest.mark.parametrize("checkpoint", [None, "checkpoint"])
def test_walltime_marks_tracking_failed_even_when_recoverable(checkpoint):
    seen = []

    class RecordingContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _traceback):
            seen.append(exc_type)

    with pytest.raises(train_mlx.TrainingRunFailed, match="walltime_limit"):
        with RecordingContext():
            train_mlx._raise_for_failed_result({
                "status": "walltime_limit", "latest_adapter": checkpoint,
            })
    assert seen == [train_mlx.TrainingRunFailed]


def test_tracking_logs_only_new_metric_lines(tmp_path):
    class FakeTracked:
        def __init__(self):
            self.metrics = []

        def log_metric(self, name, value, *, kind, step):
            self.metrics.append((name, value, kind, step))

        def log_artifact(self, *_args, **_kwargs):
            pass

        def log_checkpoint(self, *_args, **_kwargs):
            pass

    _jsonl(tmp_path / "metrics.jsonl", [
        {"name": "loss", "value": 2.0, "kind": "ml", "step": 1},
        {"name": "loss", "value": 1.0, "kind": "ml", "step": 2},
    ])
    (tmp_path / "tracking_state.json").write_text(json.dumps({
        "mlflow_run_id": "run", "metrics_lines_logged": 1,
    }), encoding="utf-8")
    tracked = FakeTracked()
    train_mlx._log_tracking(tracked, tmp_path, {})
    assert tracked.metrics == [("loss", 1.0, "ml", 2)]
    state = json.loads((tmp_path / "tracking_state.json").read_text())
    assert state["metrics_lines_logged"] == 2
    train_mlx._log_tracking(tracked, tmp_path, {})
    assert tracked.metrics == [("loss", 1.0, "ml", 2)]


def test_explicit_mock_smokes_write_honest_artifacts(tmp_path):
    training = tmp_path / "training"
    assert train_mlx.main([
        "--model", "ignored", "--data", "ignored", "--output", str(training),
        "--mock-smoke", "--max-steps", "2", "--walltime-minutes", "1",
    ]) == 0
    train_manifest = json.loads((training / "manifest.json").read_text())
    assert train_manifest["evidence_kind"] == "TECHNICAL_MOCK"
    assert "no model quality evidence" in train_manifest["claim_scope"]
    resume_contract = json.loads(
        (training / "resume_contract.json").read_text()
    )["contract"]
    assert resume_contract["code"] == train_manifest["code"]
    assert resume_contract["versions"] == train_manifest["versions"]

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
