#!/usr/bin/env python3
"""Set MLX's hard allocator limit before entering the official LoRA CLI."""

from __future__ import annotations

import argparse
import inspect
import sys
from contextlib import contextmanager
from functools import wraps


def _apply_chat_template(tokenizer, messages, **kwargs):
    """Apply a chat template with a fail-closed non-thinking contract."""
    return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)


def _exact_chat_process(dataset, record):
    """Process one MLX chat with the same template contract as preflight."""
    messages = record[dataset.chat_key]
    tools = record.get("tools")
    tokens = list(
        _apply_chat_template(
            dataset.tokenizer, messages, tools=tools, return_dict=False
        )
    )
    if not dataset.mask_prompt:
        return tokens, 0

    add_generation_prompt = messages[-1].get("role") == "assistant"
    prompt = list(
        _apply_chat_template(
            dataset.tokenizer,
            messages[:-1],
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
    )
    if len(prompt) >= len(tokens) or tokens[: len(prompt)] != prompt:
        raise ValueError(
            "chat template generation prompt is not an exact prefix of the "
            "completed chat; refusing an approximate MLX assistant mask"
        )
    return tokens, len(prompt)


def _install_exact_chat_mask() -> None:
    """Make MLX ChatDataset use the preflight's non-thinking mask boundary."""
    from mlx_lm.tuner.datasets import ChatDataset

    ChatDataset.process = _exact_chat_process


def _assistant_target_mask(steps, lengths, array_api):
    """Mask assistant targets while excluding the batch-fill position.

    MLX target step ``n`` predicts token ``n``. A sequence of length ``L`` has
    real target positions through ``L - 1``; position ``L`` is the first value
    from the zero-initialized batch buffer and must not contribute to loss.
    """
    return array_api.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])


def _assistant_loss_without_batch_fill(model, batch, lengths):
    """MLX-LM causal loss with the exact assistant boundary and no fill target."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = _assistant_target_mask(steps, lengths, mx)
    ntoks = mask.sum()
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ce = ce.astype(mx.float32).sum() / ntoks
    return ce, ntoks


def _validate_loss_dataset(dataset, max_seq_length: int, name: str) -> None:
    """Fail before compilation when a dataset row has no real target token."""
    if dataset is None:
        return
    if max_seq_length < 2:
        raise ValueError("max_seq_length must leave room for a causal target")
    for index in range(len(dataset)):
        item = dataset[index]
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name} row {index}: expected (tokens, offset)")
        tokens, offset = item
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(f"{name} row {index}: invalid target offset {offset!r}")
        effective_length = min(len(tokens), max_seq_length)
        first_target_step = max(offset, 1)
        if first_target_step >= effective_length:
            raise ValueError(
                f"{name} row {index}: no real loss target remains after "
                f"truncation (offset={offset}, effective_length={effective_length})"
            )


@contextmanager
def _scoped_assistant_loss(lora_module):
    """Inject corrected train/evaluation loss and always restore the MLX API."""
    original_train = lora_module.train
    original_evaluate = lora_module.evaluate
    train_signature = inspect.signature(original_train)
    evaluate_signature = inspect.signature(original_evaluate)
    if not {"args", "train_dataset", "val_dataset", "loss"}.issubset(
        train_signature.parameters
    ):
        raise RuntimeError("unsupported mlx_lm train API for exact loss injection")
    if not {"dataset", "max_seq_length", "loss"}.issubset(
        evaluate_signature.parameters
    ):
        raise RuntimeError("unsupported mlx_lm evaluate API for exact loss injection")

    @wraps(original_train)
    def train_with_assistant_loss(*args, **kwargs):
        bound = train_signature.bind_partial(*args, **kwargs)
        supplied = bound.arguments.get("loss")
        bound.apply_defaults()
        if supplied is not None and supplied is not _assistant_loss_without_batch_fill:
            raise ValueError("refusing to replace an explicitly supplied MLX loss")
        training_args = bound.arguments["args"]
        _validate_loss_dataset(
            bound.arguments["train_dataset"], training_args.max_seq_length, "train"
        )
        _validate_loss_dataset(
            bound.arguments["val_dataset"], training_args.max_seq_length, "validation"
        )
        bound.arguments["loss"] = _assistant_loss_without_batch_fill
        return original_train(*bound.args, **bound.kwargs)

    @wraps(original_evaluate)
    def evaluate_with_assistant_loss(*args, **kwargs):
        bound = evaluate_signature.bind_partial(*args, **kwargs)
        supplied = bound.arguments.get("loss")
        bound.apply_defaults()
        if supplied is not None and supplied is not _assistant_loss_without_batch_fill:
            raise ValueError("refusing to replace an explicitly supplied MLX loss")
        _validate_loss_dataset(
            bound.arguments["dataset"], bound.arguments["max_seq_length"], "test"
        )
        bound.arguments["loss"] = _assistant_loss_without_batch_fill
        return original_evaluate(*bound.args, **bound.kwargs)

    lora_module.train = train_with_assistant_loss
    lora_module.evaluate = evaluate_with_assistant_loss
    try:
        yield
    finally:
        lora_module.train = original_train
        lora_module.evaluate = original_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import lora

    mx.set_memory_limit(args.memory_limit_bytes)
    _install_exact_chat_mask()
    sys.argv = ["mlx_lm.lora", "--config", args.config]
    with _scoped_assistant_loss(lora):
        lora.main()


if __name__ == "__main__":
    main()
