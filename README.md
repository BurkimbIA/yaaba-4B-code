# yaaba-4B: training and evaluation code

Code and measurements behind *yaaba-4B: Teaching a Small LLM Mooré with Fifteen
Million Tokens* (AfriLang AI 2026). yaaba-4B is Qwen3-4B, continued-pretrained
on about 15 million tokens of Mooré (`mos`) and then instruction-tuned, with
LoRA throughout.

| What | Where | Access |
|------|-------|--------|
| Instruct model | [burkimbia/yaaba-4B](https://huggingface.co/burkimbia/yaaba-4B) | gated, Apache-2.0 |
| CPT base (before instruction tuning) | [burkimbia/yaaba-4B-base](https://huggingface.co/burkimbia/yaaba-4B-base) | gated, CC BY-NC 4.0 |
| Instruction set | [burkimbia/yaaba-instruct](https://huggingface.co/datasets/burkimbia/yaaba-instruct) | gated, CC BY-NC 4.0 |
| Evaluation items | [burkimbia/yaaba-4B-eval](https://huggingface.co/datasets/burkimbia/yaaba-4B-eval) | gated, do not train on it |

The evaluation items (the held-out texts, the 52 fact items, the 12 held-out
questions and the generative probes) are not in this repository.
Text published in plain view can end up in training corpora, and a model trained
on a test set can no longer be measured with it. The items are in the gated
dataset above, which carries a canary string.

## What reruns from this repository alone

No GPU is needed for any of these.

```bash
uv sync
uv run pytest                                      # 33 tests
uv run python scripts/figures/make_figures.py      # the paper's three figures, into figures/
uv run python scripts/measure/flores.py --table    # FLORES+ devtest chrF table
uv run python scripts/measure/flores_bootstrap.py  # paired bootstrap between systems
```

The figures read only `evaluation/`. The FLORES+ commands read the stored
hypotheses in `evaluation/flores/` and need the references from
[openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus),
which is gated on the Hub: accept its terms and log in with `huggingface-cli login`.

## What needs the gated evaluation set

`scripts/measure/evaluate_checkpoint.py` measures a checkpoint: bits per
character on the held-out texts of seven capabilities, the fact probe, and the
generative probes. It reads its items from `evaluation/`, where the gated
dataset's files go. The point files in `evaluation/points*/` here keep the
numbers and drop the answers to the fact items and probes; the full point files are in the gated
dataset, and `--compare` and `--arms` need them. Once your access is granted:

```bash
huggingface-cli download burkimbia/yaaba-4B-eval --repo-type dataset --local-dir evaluation
uv run python scripts/measure/evaluate_checkpoint.py --arms
```

## What does not rerun from public material

The continued pre-training corpus is not released, so `scripts/train/freeze_arms.py`
and `scripts/train/train_cpt.py` are published to show exactly what was done,
not to be rerun as they stand. The paper says the same in its limitations.

`scripts/train/train_sft.py` trains on the instruction set. `yaaba-instruct`
holds the same turns with phone numbers masked and English field names, so a
rerun needs a small conversion back to the layout the script reads.

Training ran on a single A100 40 GB on Colab. The settings are in `configs/`,
and each run copies its config into its output folder. Checkpoints were
synced to an S3 bucket through the standard AWS variables (see `src/yaaba/s3.py`);
the bucket paths in the scripts are ours and are not public.

## Layout

```text
configs/            CPT and SFT settings
src/yaaba/          shared code: Mooré text checks, bootstrap intervals, S3, checkpoints
scripts/train/      freeze the CPT data arms, CPT, SFT
scripts/measure/    checkpoint evaluation, fact probe, FLORES+
scripts/figures/    the paper's figures, from evaluation/
evaluation/         measured points (numbers only) and FLORES+ outputs
tests/
```

Some identifiers and data keys are in French (`mesures`, `faits`, `juste`,
`piege`): they match the files already written and are kept so old and new
measurements stay readable by the same code.

## Citation

```bibtex
@inproceedings{sawadogo2026yaaba,
  title     = {yaaba-4B: Teaching a Small LLM Moor{\'e} with Fifteen Million Tokens},
  author    = {Sawadogo, Salif and Nikiema, Mahamadi and Kabor{\'e}, Josias and Oubda, David},
  booktitle = {AfriLang AI 2026},
  series    = {Proceedings of Machine Learning Research},
  volume    = {314},
  year      = {2026}
}
```

## License

Code: Apache-2.0. The models and datasets carry their own licenses, listed above.
