"""Continued pre-training for one arm, resuming by default.

    python scripts/train/train_cpt.py --arm A --verify   # here, no torch needed
    python scripts/train/train_cpt.py --arm A --smoke    # 30 steps on GPU
    python scripts/train/train_cpt.py --arm A            # the run
    python scripts/train/train_cpt.py --arm A            # after a crash: same
    python scripts/train/train_cpt.py --arm A --fresh    # start over
    python scripts/train/train_cpt.py --arm P2 --resume-from P2-.../checkpoint-3172

Resuming is the default because the wrong move should be the one that costs:
when a runtime dies you rerun the same cell, and silently starting over would
lose the hours already paid for without showing it.

Setup lives in `configs/cpt.json` and is copied into the run directory, so a
checkpoint always carries the settings that produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yaaba import checkpoints, mixture, packing, s3  # noqa: E402
from yaaba.heldout import load as load_held_out  # noqa: E402

MIXTURE = ROOT / "melange"
CONFIG = json.loads((ROOT / "configs" / "cpt.json").read_text(encoding="utf-8"))
RUNS_URI = "s3://burkimbia-store/text/moore-assistant/cpt"

BATCH, ACCUM = CONFIG["lot"], CONFIG["accumulation"]
WINDOW = CONFIG["fenetre"]
ARMS = ("A", "C", "P", "P2")


def recipe_for(arm: str) -> dict:
    return json.loads((MIXTURE / "recettes" / f"{arm}.json").read_text(encoding="utf-8"))


def configure_wandb(arm: str, run: str, fingerprint: str) -> bool:
    """Group runs by arm and reattach a resume to the same curve.

    Without a stable run id a 12 h run dying three times yields four
    disconnected curves, and neither epochs nor arms can be compared. Since
    resuming is the default, that is the normal case.
    """
    if os.environ.get("WANDB_DISABLED"):
        return False
    # A key without the package fails at step 0, after the card is allocated and
    # the mixture packed. Degrade instead: the S3 log stays the source of truth.
    if importlib.util.find_spec("wandb") is None:
        print("W&B: key present but package missing, tracking off", file=sys.stderr)
        return False

    # A cell that crashed leaves wandb alive in the kernel, and the next attempt
    # inherits it: every variable set below is ignored and `init` fails on
    # "run ID <the previous one> is in use", naming a run the caller never
    # mentioned. Clear it here rather than after the card is allocated.
    #
    # `finish` alone does not do it. It closes the *run*; the run id is pinned
    # by the *session* singleton, which is what says "your wandb session has
    # already started". Only `teardown` drops that, and closing the run first
    # keeps it from being marked crashed.
    #
    # And `find_spec` finding the package does not mean it imports: a partial
    # install raises on `import wandb`, which is the same fatal-at-step-0 shape
    # the check above exists to prevent. Reporting is not worth a run.
    try:
        import wandb

        if wandb.run is not None:
            print(f"W&B: closing {wandb.run.id}, left open by a previous attempt",
                  file=sys.stderr)
            wandb.finish()
        clear = getattr(wandb, "teardown", None)
        if clear is None:  # before the public name existed
            from wandb.sdk import wandb_setup

            clear = wandb_setup._teardown
        clear()
    except Exception as broken:  # noqa: BLE001
        print(f"W&B: package present but unusable ({broken}), tracking off",
              file=sys.stderr)
        return False

    os.environ.setdefault("WANDB_PROJECT", CONFIG["projet_wandb"])
    os.environ["WANDB_RUN_GROUP"] = f"arm-{arm}"
    os.environ["WANDB_TAGS"] = f"arm-{arm},data-{fingerprint}"
    os.environ["WANDB_RUN_ID"] = run
    os.environ["WANDB_RESUME"] = "allow"
    return True


def verify(arm: str, out: Path) -> int:
    """Everything checkable without a GPU. Must print READY before paying.

    Prefers the frozen arm's card: it records what was frozen, and re-resolving
    the recipe needs mixture files a fresh runtime does not have.
    """
    card = mixture.frozen_card(arm, MIXTURE, out)
    if card:
        sequences = card["sequences_1024"]
        hours = (sequences * WINDOW * CONFIG["epoques"]
                 / CONFIG["debit_mesure_tokens_s"] / 3600)
        print(f"ARM {arm} frozen {card['empreinte']}, {card['fige_le']}")
        print(f"  {card['documents']:>12,} documents")
        print(f"  {card['tokens']:>12,} tokens")
        for part, share in sorted(card["ratio_obtenu"].items(),
                                  key=lambda kv: -kv[1]):
            print(f"    {part:16s} {share:5.2f} %  "
                  f"{card['tokens_par_part'][part]:>12,}")
        print(f"  packed    {sequences:>12,} sequences of {WINDOW}")
        print(f"  steps     {sequences // (BATCH * ACCUM):>12,} per epoch "
              f"({BATCH * ACCUM * WINDOW:,} tokens per step)")
        print(f"  held-out leaks when frozen: {card['fuites_tenu_a_lecart']}")
        print(f"  {CONFIG['epoques']} epochs {hours:>10.1f} h   "
              f"{hours * 1.89:.2f} $ on A100")
        ready = card["fuites_tenu_a_lecart"] == 0 and card["documents"] > 0
        print(f"\n{'READY' if ready else 'NOT READY'}")
        return 0 if ready else 1

    print(f"no frozen arm for {arm}, falling back to the recipe", flush=True)
    recipe = recipe_for(arm)
    held_out = load_held_out(ROOT / "evaluation" / "tenu_a_lecart.jsonl")
    loaded = mixture.load_recipe(recipe, MIXTURE, held_out)

    print(f"ARM {arm}: {recipe['quoi']}")
    print(f"  {'file':22s} {'take':14s} {'docs':>9s} {'tokens':>12s}")
    for path, report in loaded.per_file.items():
        source = mixture.plan(recipe)[path]
        extra = (f"  ({report.dropped_paired:,} paired dropped)"
                 if report.dropped_paired else "")
        print(f"  {path:22s} {source.take:14s} {report.documents:>9,} "
              f"{report.tokens:>12,}{extra}")

    declared = recipe["total_tokens"]
    drift = abs(loaded.tokens - declared) / declared
    sequences = packing.sequence_count(loaded.tokens, loaded.documents, WINDOW)
    hours = (sequences * WINDOW * CONFIG["epoques"]
             / CONFIG["debit_mesure_tokens_s"] / 3600)

    print(f"\n  loaded    {loaded.tokens:>12,} tokens")
    print(f"  declared  {declared:>12,} tokens   drift {100 * drift:.3f} %")
    print(f"  packed    {sequences:>12,} sequences of {WINDOW}")
    print(f"  steps     {sequences // (BATCH * ACCUM):>12,} per epoch "
          f"({BATCH * ACCUM * WINDOW:,} tokens per step)")
    print(f"  held-out leaks in what would be loaded: {loaded.leaks}")
    print(f"  {CONFIG['epoques']} epochs {hours:>10.1f} h   "
          f"{hours * 1.89:.2f} $ on A100")

    # A drift can mean two very different things: an inconsistent recipe, or one
    # asking for more than the reservoir holds, which the recipe already says.
    short = {k: v for k, v in recipe.get("manque", {}).items() if v}
    if short:
        print("\n  recipe declares missing: "
              + ", ".join(f"{k} {v:,}" for k, v in short.items()))
        print("  -> python scripts/build/completer_melange.py --depose")

    ready = drift < 0.01 and loaded.leaks == 0 and loaded.texts
    print(f"\n{'READY' if ready else 'NOT READY'}")
    return 0 if ready else 1


def resume_named(name: str, out: Path) -> tuple[Path, str]:
    """Resume from a run named by hand: `<run>` or `<run>/checkpoint-N`.

    Automatic resume asks S3 which runs exist, and a prefix listing there can be
    stale for hours: `P2-20260902-1725` had four checkpoints deposited
    and appeared in no listing, flat or delimited, while `head_object` on its
    keys answered. The run restarted from zero, losing 2.8 epochs.

    Naming the run skips the listing entirely: every lookup here is by key.
    """
    run, _, point = name.strip("/").partition("/")
    folder = out / run
    if not point:
        found = s3.latest_checkpoint(f"{RUNS_URI}/{run}")
        if not found:
            raise SystemExit(
                f"no checkpoint listed under {run}. The listing may be stale: "
                f"name the point too, as '{run}/checkpoint-N'.")
        point = found[0]

    target = folder / point
    if not (target / s3.REQUIRED_TO_RESUME).exists():
        s3.download_dir(f"{RUNS_URI}/{run}/{point}", target)
    if not (target / s3.REQUIRED_TO_RESUME).exists():
        raise SystemExit(f"{run}/{point}: no {s3.REQUIRED_TO_RESUME} on S3, "
                         "it cannot be resumed from")

    print(f"RESUMING as asked: {run}/{point}", flush=True)
    absent = [f for f in s3.RESUME_STATE if not (target / f).exists()]
    if absent:
        print(f"  WARNING: {', '.join(absent)} missing, resuming without "
              f"optimizer state", file=sys.stderr)
    return folder, str(target)


def resolve_run(arm: str, out: Path, fresh: bool,
                resume_from: str = "") -> tuple[Path, str | None]:
    """Pick the run directory and the checkpoint to resume from.

    Resolved before loading anything: the previous version read 322k documents
    and packed 36 M tokens before discovering there was nothing to resume.
    """
    if resume_from and not fresh:
        return resume_named(resume_from, out)
    if not fresh:
        local = sorted((d for d in out.glob(f"{arm}-*")
                        if d.is_dir() and any(d.glob("checkpoint-*"))),
                       key=lambda d: d.name)
        if local:
            folder = local[-1]
            resume = max(folder.glob("checkpoint-*"),
                         key=lambda d: int(d.name.rpartition("-")[2]))
            print(f"RESUMING locally from {resume}", flush=True)
            return folder, str(resume)

        # A dead Colab runtime wipes /content, which is the normal case: if it
        # had not died there would be nothing to resume.
        for name in reversed(s3.list_runs(RUNS_URI, f"{arm}-")):
            checkpoint = s3.latest_checkpoint(f"{RUNS_URI}/{name}")
            if not checkpoint:
                continue
            folder = out / name
            s3.download_dir(f"{RUNS_URI}/{name}/{checkpoint[0]}",
                            folder / checkpoint[0])

            # A checkpoint without `trainer_state.json` cannot be resumed: the
            # Trainer raises before the first step, after the card is allocated
            # and the mixture packed. Early runs uploaded the adapter alone,
            # so this is not hypothetical.
            if not (folder / checkpoint[0] / s3.REQUIRED_TO_RESUME).exists():
                print(f"  {name}/{checkpoint[0]}: no {s3.REQUIRED_TO_RESUME}, "
                      f"cannot resume from it (adapter kept for evaluation)",
                      file=sys.stderr)
                continue

            print(f"RESUMING from S3: {name}/{checkpoint[0]}", flush=True)
            absent = [f for f in s3.RESUME_STATE
                      if not (folder / checkpoint[0] / f).exists()]
            if absent:
                print(f"  WARNING: {', '.join(absent)} missing, resuming without "
                      f"optimizer state", file=sys.stderr)
            return folder, str(folder / checkpoint[0])

    folder = out / f"{arm}-{dt.datetime.now():%Y%m%d-%H%M}"
    print(f"FRESH RUN: {folder.name}"
          + ("" if fresh else f"  (nothing to resume for {arm})"), flush=True)
    return folder, None


def train(arm: str, epochs: int, out: Path, fresh: bool, upload: bool,
          smoke: int, from_recipe: bool, resume_from: str = "") -> int:
    import gc
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, Trainer, TrainerCallback,
                              TrainingArguments, default_data_collator)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    folder, resume = resolve_run(arm, out, fresh or bool(smoke), resume_from)
    run = folder.name
    uri = f"{RUNS_URI}/{run}"

    # Two seconds, before the card is allocated. Uploads only warn when they
    # fail, which is right for a blip mid-run and wrong for credentials that
    # never worked: a run then trains for hours and leaves nothing.
    if upload and not smoke:
        why = s3.writable(uri)
        if why:
            raise SystemExit(
                f"S3 refuses a write to {uri}: {why}. Nothing would survive "
                "this run: check the AWS secrets before paying for the card.")
        print(f"S3 writable: {uri}", flush=True)

    loaded = None if from_recipe else mixture.load_frozen(arm, MIXTURE, out)
    if loaded is None:
        loaded = mixture.load_recipe(recipe_for(arm), MIXTURE)
        print(f"ARM {arm} from recipe: {len(loaded.texts):,} documents", flush=True)
    else:
        print(f"ARM {arm} frozen {loaded.fingerprint}: {len(loaded.texts):,} documents",
              flush=True)

    cache = packing.pack(
        loaded.texts,
        out / f"packed-{arm}-{loaded.fingerprint}-{WINDOW}.u32",
        CONFIG["base"], WINDOW)
    packed = np.memmap(cache, dtype=np.uint32, mode="r").reshape(-1, WINDOW)

    class Sequences(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(packed)

        def __getitem__(self, index: int) -> dict:
            ids = torch.from_numpy(packed[index].astype(np.int64))
            return {"input_ids": ids, "labels": ids.clone(),
                    "attention_mask": torch.ones_like(ids)}

    model = AutoModelForCausalLM.from_pretrained(CONFIG["base"],
                                                 dtype=torch.bfloat16,
                                                 device_map="auto")
    model.config.use_cache = False
    # Frozen base weights mean the replayed forward pass starts from inputs that
    # need no gradient, cutting the chain before LoRA. One line, and it only
    # shows after the card is allocated.
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **CONFIG["lora"]))
    model.print_trainable_parameters()

    # Packing is done per file, so a sequence is homogeneous. It is the sampler's
    # shuffle that makes a step see all five languages: the ratio holds over the
    # whole mixture, not inside a window.
    steps_per_epoch = -(-len(packed) // (BATCH * ACCUM))
    total_steps = smoke or steps_per_epoch * epochs
    print(f"  {len(packed):,} sequences, {steps_per_epoch:,} steps/epoch, "
          f"{total_steps:,} total", flush=True)

    log = folder / "progression.jsonl"

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            if not logs:
                return
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pas": state.global_step,
                                         "epoque": round(state.epoch or 0, 3),
                                         **logs}, ensure_ascii=False) + "\n")

    if smoke:
        # Thirty steps are worth no W&B run, and without a key `wandb.init`
        # blocks on stdin forever.
        os.environ["WANDB_DISABLED"] = "true"
        print(f"SMOKE RUN: {smoke} steps, no checkpoint, no upload", flush=True)

    tracking = configure_wandb(arm, run, loaded.fingerprint)
    print(f"W&B: {'project ' + os.environ['WANDB_PROJECT'] + ', run ' + run}"
          if tracking else "W&B: disabled", flush=True)

    settings = dict(
        output_dir=str(folder),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        # Activation recomputation is the one throughput lever left: it costs
        # 20 to 30 %. It is in the config because it is measurable, and because
        # a checkpoint must carry the setting that produced it.
        gradient_checkpointing=CONFIG["recalcul_activations"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=CONFIG["optimiseur"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=max(1, round(CONFIG["warmup_ratio"] * total_steps)),
        bf16=True,
        logging_steps=CONFIG["logging_steps"],
        # Saves are paced on the runtime's life expectancy, not on the shape of
        # training. One fifth of an epoch puts epoch boundaries exactly on saves
        #.
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // CONFIG["sauvegardes_par_epoque"]),
        save_total_limit=CONFIG["save_total_limit_disque"],
        report_to=["wandb"] if tracking else [],
        run_name=run,
    )
    if smoke:
        settings |= {"max_steps": smoke, "save_strategy": "no",
                     "logging_steps": max(1, smoke // 6)}

    callbacks: list = [Log()]
    if upload and not smoke:
        callbacks.append(checkpoints.callback(uri, log, steps_per_epoch))

    Trainer(model=model, args=TrainingArguments(**settings),
            train_dataset=Sequences(), data_collator=default_data_collator,
            callbacks=callbacks).train(resume_from_checkpoint=resume)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cpt.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\n-> {folder}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--verify", action="store_true",
                        help="resolve and check the recipe, without torch")
    parser.add_argument("--epochs", type=int, default=CONFIG["epoques"])
    parser.add_argument("--out", default="/content/cpt")
    parser.add_argument("--fresh", action="store_true",
                        help="start over instead of resuming")
    parser.add_argument("--from-recipe", action="store_true",
                        help="resolve the recipe instead of using the frozen arm")
    parser.add_argument("--smoke", nargs="?", type=int, const=30, default=0,
                        metavar="STEPS",
                        help="test the torch path on STEPS steps, no checkpoint")
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    args = parser.parse_args()

    if args.verify:
        return verify(args.arm, Path(args.out))
    return train(args.arm, args.epochs, Path(args.out), args.fresh,
                 args.upload, args.smoke, args.from_recipe,
                 args.resume_from)


if __name__ == "__main__":
    sys.exit(main())
