#!/usr/bin/env python3
"""Set MLX's hard allocator limit before entering the official LoRA CLI."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import lora

    mx.set_memory_limit(args.memory_limit_bytes)
    sys.argv = ["mlx_lm.lora", "--config", args.config]
    lora.main()


if __name__ == "__main__":
    main()
