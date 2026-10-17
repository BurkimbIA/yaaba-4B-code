#!/usr/bin/env python3
"""Set MLX's hard allocator limit before entering the official LoRA CLI."""

from __future__ import annotations

import argparse
import sys


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
    lora.main()


if __name__ == "__main__":
    main()
