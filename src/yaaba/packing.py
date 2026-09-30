"""Packing documents into fixed-length training sequences.

Moore documents are short, 46 tokens at the median (a verse, a proverb, a
pair). One document per sequence would fill a 1024-token window to 4.5 % and
spend the rest on padding. Documents are concatenated with `eos` between them
and cut every `window` tokens, so a long document is split across sequences
and never truncated. As usual in pretraining, a document can see the end of
the previous one inside a sequence.
"""

from __future__ import annotations

from pathlib import Path

WINDOW = 1024


def pack(texts: list[str], cache: Path, model: str, window: int = WINDOW) -> Path:
    """Tokenize, concatenate with `eos`, cut every `window`. Cached on disk.

    The cache path encodes the mixture's identity, so two mixtures never share
    a cache.
    """
    import numpy as np
    from transformers import AutoTokenizer

    if cache.exists():
        count = cache.stat().st_size // 4 // window
        print(f"  cache hit: {count:,} sequences ({cache})", flush=True)
        return cache

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos = tokenizer.eos_token_id
    flat: list[int] = []
    for start in range(0, len(texts), 1000):
        batch = texts[start:start + 1000]
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            flat.extend(ids)
            flat.append(eos)
        if start % 50_000 == 0:
            print(f"  tokenized {start:,}/{len(texts):,} docs, {len(flat):,} tokens",
                  flush=True)

    count = len(flat) // window          # the tail, under one sequence, is dropped
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(flat[:count * window], dtype=np.uint32).tofile(cache)
    print(f"  packed: {count:,} sequences, {len(flat) - count * window} tokens dropped",
          flush=True)
    return cache


def sequence_count(tokens: int, documents: int, window: int = WINDOW) -> int:
    """Sequences a mixture will yield, counting one `eos` per document.

    Leaving out the separators under-counted arm A by 313 sequences, ten steps
    per epoch.
    """
    return (tokens + documents) // window
