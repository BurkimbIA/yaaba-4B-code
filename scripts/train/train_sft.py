"""The instruction stage, on the base alone or on a CPT arm.

    python scripts/train/train_sft.py --start base --verify   # here, no torch
    python scripts/train/train_sft.py --start A    --verify
    python scripts/train/train_sft.py --start A    --smoke    # 20 steps on GPU
    python scripts/train/train_sft.py --start A               # the run
    python scripts/train/train_sft.py --start A               # after a crash: same

`--start base` is the control and it is not optional. `tengsoaba-4b` is the
score to beat but it was trained on a different, far larger instruction
set, so comparing it to `A + SFT` would confound the continued pre-training with
the size of the instruction set. Running the same 12 702 turns on the base and on
each arm isolates what the CPT actually bought.

Setup lives in `configs/sft.json` and is copied into the run directory.

## Three things this stage can get silently wrong

**The adapter loaded frozen.** `PeftModel.from_pretrained` defaults to
`is_trainable=False`. A run then trains, shows a falling loss and learns
nothing. Here the arm's adapter is *merged* rather than continued, so the trap
does not apply, but the guard that catches it stays: trainable parameters are
counted and zero raises.

**The loss computed on the question.** Without a mask the model is trained to
produce the prompt as much as the answer, which on `mos_mos` means training it
to echo. Only the assistant tokens carry a label here, and a test checks it.

**A turn silently truncated.** A long tale fills the window and its answer falls
off the end, leaving an example with nothing to learn from. Such turns are
dropped and counted, never truncated.
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

from yaaba import checkpoints, data, heldout, s3  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "sft.json").read_text(encoding="utf-8"))
RUNS_URI = "s3://burkimbia-store/text/moore-assistant/sft"
CPT_URI = "s3://burkimbia-store/text/moore-assistant/cpt"

BATCH, ACCUM = CONFIG["lot"], CONFIG["accumulation"]
WINDOW = CONFIG["fenetre"]
ARMS = ("A", "B", "C", "P", "P2")
STARTS = ("base",) + ARMS

ADAPTER = "adapter_model.safetensors"
IGNORE = -100  # what the loss skips


def turns(split: str = "train") -> list[dict]:
    """The assembled turns of one split, filtered to the chosen variants.

    The variant says which language the question is in and which the answer is
    in. Choosing them is an experimental choice, so it is read from the config
    and never hardcoded here.
    """
    keep = set(CONFIG["variantes"])
    chosen = []
    with data.path(CONFIG["jeu"]).open(encoding="utf-8") as handle:
        for line in handle:
            turn = json.loads(line)
            if turn.get("split") == split and turn.get("variant") in keep:
                chosen.append(turn)
    return chosen


def encode(turn: dict, tokenizer, window: int = WINDOW) -> dict | None:
    """One turn as ids and labels, or None if it does not fit.

    The prompt is templated twice: once without the answer to find where it
    ends, once with. Everything before the answer is masked, so the loss only
    ever sees what the model is meant to produce.

    `enable_thinking=False` is not optional on Qwen3: the mode is on by default,
    reasons in Chinese, and spends the whole budget.
    """
    messages = turn["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None

    def template(upto, generation_prompt):
        try:
            return tokenizer.apply_chat_template(
                upto, tokenize=False, add_generation_prompt=generation_prompt,
                enable_thinking=False)
        except (TypeError, ValueError):
            return tokenizer.apply_chat_template(
                upto, tokenize=False, add_generation_prompt=generation_prompt)

    prompt = tokenizer(template(messages[:-1], True), add_special_tokens=False)
    full = tokenizer(template(messages, False), add_special_tokens=False)
    if len(full.input_ids) > window or len(prompt.input_ids) >= len(full.input_ids):
        return None

    labels = list(full.input_ids)
    labels[:len(prompt.input_ids)] = [IGNORE] * len(prompt.input_ids)
    return {"input_ids": full.input_ids, "attention_mask": full.attention_mask,
            "labels": labels}


def start_from(start: str, out: Path) -> tuple[str, str | None]:
    """The base model id, and the adapter to merge into it, or None.

    It is *merged*, so what trains afterwards is a fresh adapter over a fixed
    starting point: the SFT stage stays identical whatever the start, which is
    the only way the comparison says something about the CPT.

    **The last checkpoint is not the best one**, which is the whole reason for
    measuring a curve. `C` turns between epoch 4 and epoch 5, on Moore and on the
    facts alike, so taking its last point would carry an overfit into
    the SFT stage and blame the result on the CPT recipe.

    So an arm may name its point: `C@checkpoint-4820`. Without a point, the last
    checkpoint of the most recent run is used, which is right for an arm whose
    curve is still going down and wrong for one that has turned.
    """
    if start == "base":
        return CONFIG["base"], None
    start, _, wanted = start.partition("@")
    runs = s3.list_runs(CPT_URI, f"{start}-")
    if not runs:
        raise SystemExit(f"no CPT run for arm {start} under {CPT_URI}")
    run = runs[-1]
    if wanted:
        name, step = wanted, int(wanted.rpartition("-")[2])
    else:
        found = s3.latest_checkpoint(f"{CPT_URI}/{run}")
        if not found:
            raise SystemExit(f"no checkpoint in {run}")
        name, step = found
    local = out / "depart" / f"{run}-{name}"
    if not (local / ADAPTER).exists():
        s3.download_dir(f"{CPT_URI}/{run}/{name}", local)
    # Checked after the download, so a mistyped point fails here rather than
    # inside `PeftModel.from_pretrained` twenty minutes of setup later.
    if not (local / ADAPTER).exists():
        raise SystemExit(f"{CPT_URI}/{run}/{name} has no {ADAPTER}")
    print(f"start: {CONFIG['base']} + {run}/{name} (step {step:,})", flush=True)
    return CONFIG["base"], str(local)


def configure_wandb(start: str, run: str) -> bool:
    """Same contract as the CPT stage, including closing a leftover session."""
    if os.environ.get("WANDB_DISABLED"):
        return False
    if importlib.util.find_spec("wandb") is None:
        print("W&B: key present but package missing, tracking off", file=sys.stderr)
        return False
    try:
        import wandb

        if wandb.run is not None:
            print(f"W&B: closing {wandb.run.id}, left open by a previous attempt",
                  file=sys.stderr)
            wandb.finish()
        clear = getattr(wandb, "teardown", None)
        if clear is None:
            from wandb.sdk import wandb_setup

            clear = wandb_setup._teardown
        clear()
    except Exception as broken:  # noqa: BLE001
        print(f"W&B: package present but unusable ({broken}), tracking off",
              file=sys.stderr)
        return False

    os.environ.setdefault("WANDB_PROJECT", CONFIG["projet_wandb"])
    os.environ["WANDB_RUN_GROUP"] = f"sft-{label(start)}"
    os.environ["WANDB_TAGS"] = f"sft,start-{label(start)}"
    os.environ["WANDB_RUN_ID"] = run
    os.environ["WANDB_RESUME"] = "allow"
    return True


def held_out_in(chosen: list[dict]) -> list[tuple[int, str, str, str]]:
    """Turns whose text is in the evaluation instrument, with how they match.

    The CPT mixture is screened part by part (`heldout.TARGETS`); the SFT set
    never was, and three of its 4 265 turns are word for word in the held-out
    `moore_humain`, from `enquetes` and `spg_series`. Training on them and then
    measuring on them would put memorisation into the final comparison.

    An SFT turn belongs to no mixture part -- it is assembled from all of them
    -- so it is checked against `EVERYTHING`, the widest control there is.
    """
    guard = heldout.Guard(heldout.EVERYTHING,
                          heldout.load(ROOT / "evaluation" / "tenu_a_lecart.jsonl"))
    found = []
    for index, turn in enumerate(chosen):
        for message in turn["messages"]:
            leak = guard.inspect(message["content"])
            if leak:
                found.append((index, turn.get("task", "?"),
                              turn.get("source", "?"), leak))
                break
    return found


def start_missing(start: str) -> str | None:
    """Why the start cannot be resolved on S3, or None. No download, no GPU.

    `verify` used to print READY without ever asking S3, so a checkpoint that
    did not exist passed the check. In a notebook chaining six runs, finding out
    at the fifth hour that one is missing costs the session.
    """
    if start == "base":
        return None
    arm, _, wanted = start.partition("@")
    runs = s3.list_runs(CPT_URI, f"{arm}-")
    if not runs:
        return f"no CPT run for arm {arm} under {CPT_URI}"
    names = [name for name, _ in s3.checkpoint_names(f"{CPT_URI}/{runs[-1]}")]
    if not names:
        return f"no checkpoint in {runs[-1]}"
    if wanted and wanted not in names:
        return f"{runs[-1]} has no {wanted}; it has {', '.join(names[-3:])}"
    return None


def unlisted_variants() -> dict[str, int]:
    """Variants present in the set that `configs/sft.json` names nowhere.

    **A variant absent from `variantes` is dropped silently**, and that cost the
    whole verbosity work: `fr_long` was invented by two builders and none of its
    turns reached the run that was meant to train them. Nothing failed,
    nothing warned; the turns simply were not there.

    So an exclusion now has to be written down. A variant belongs either to
    `variantes`, which trains it, or to `variantes_ecartees`, which is a
    decision someone made and can defend. Anything else stops the run.
    """
    known = set(CONFIG["variantes"]) | set(CONFIG.get("variantes_ecartees", ()))
    counts: dict[str, int] = {}
    with data.path(CONFIG["jeu"]).open(encoding="utf-8") as handle:
        for line in handle:
            name = json.loads(line).get("variant", "?")
            if name not in known:
                counts[name] = counts.get(name, 0) + 1
    return counts


def demo() -> None:
    """The unlisted-variant guard, checked. A guard that cannot fail guards
    nothing.

    `verify` covers the rest of this file and needs the set on disk; this one
    needs nothing, so it runs in `pytest` and in a hook.
    """
    kept = list(CONFIG["variantes"])
    try:
        # Every variant on disk is named somewhere, or `verify` would refuse.
        assert unlisted_variants() == {}, unlisted_variants()
        # Drop one, and the guard must see exactly it. This is the shape of the
        # defect: `fr_long` was in neither list and its turns vanished unsaid.
        CONFIG["variantes"] = [v for v in kept if v != "fr_long"]
        seen = unlisted_variants()
        assert set(seen) == {"fr_long"}, seen
        assert seen["fr_long"] > 0
    finally:
        CONFIG["variantes"] = kept
    print(f"train_sft : ok | guard sees {seen['fr_long']:,} orphaned turns")


def verify(start: str) -> int:
    """Everything checkable without a GPU. Must print READY before paying."""
    chosen = turns("train")
    held = turns("val")
    by_variant: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for turn in chosen:
        by_variant[turn["variant"]] = by_variant.get(turn["variant"], 0) + 1
        by_task[turn.get("task", "?")] = by_task.get(turn.get("task", "?"), 0) + 1

    print(f"START {start}   set {CONFIG['jeu']}")
    print(f"  variants kept: {', '.join(CONFIG['variantes'])}")
    for name, count in sorted(by_variant.items(), key=lambda kv: -kv[1]):
        print(f"    {name:10s} {count:>6,}")
    print("  tasks:")
    for name, count in sorted(by_task.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {name:24s} {count:>6,}")
    steps = -(-len(chosen) // (BATCH * ACCUM))
    print(f"  {len(chosen):,} train turns, {len(held):,} held back for reading")
    print(f"  steps     {steps:>8,} per epoch, {steps * CONFIG['epoques']:,} total")

    # A turn longer than the window is dropped, so the count belongs here, before
    # the card is paid for, and not in a log nobody reads afterwards.
    longest = max((sum(len(m["content"]) for m in t["messages"]) for t in chosen),
                  default=0)
    print(f"  longest turn {longest:,} characters "
          f"(window {WINDOW:,} tokens; turns that overflow are dropped)")

    dropped_on_purpose = CONFIG.get("variantes_ecartees", ())
    if dropped_on_purpose:
        print(f"  set aside on purpose: {', '.join(dropped_on_purpose)}")
    orphans = unlisted_variants()
    if orphans:
        print("\n  VARIANTS NAMED NOWHERE IN configs/sft.json:")
        for name, count in sorted(orphans.items(), key=lambda kv: -kv[1]):
            print(f"    {name:12s} {count:>6,} turns would vanish without a word")

    leaks = held_out_in(chosen + held)
    if leaks:
        print(f"\n  HELD OUT IN THE SET: {len(leaks)} turn(s)")
        for index, task, source, how in leaks[:10]:
            print(f"    turn {index:>5,}  {how:9s} {task:18s} {source}")
    else:
        print("  held out: 0 turn touches the evaluation instrument")

    absent = start_missing(start)
    print(f"  start: {absent or start + ' is on S3'}")

    ready = (bool(chosen) and bool(CONFIG["variantes"])
             and not leaks and not absent and not orphans)
    print(f"\n{'READY' if ready else 'NOT READY'}")
    return 0 if ready else 1


GRAINE_TENUES = 20260912


def questions_tenues(combien: int = 12) -> list[dict]:
    """The turns the exit criterion is read on: Moore questions, held back.

    Written here rather than in the notebook because three places need the same
    twelve turns: the notebook that generates the answers, the review sheet that
    collects the verdicts, and any later rerun. A rule copied into three files
    is a rule that will differ in three files.

    **The family is `expliquer_proverbe`, and that is the whole point.** The
    criterion has been measured on three different item sets and none of them
    could be graded:

      - the first twelve turns in file order, 7 `condenser` and 5 `raconter`,
        where the prompt carried a proverb-shaped line and asked which proverb
        says it, so repeating the prompt was a defensible answer;
      - the `repondre` turns carrying a question mark, which came from
        broadcast and interview transcript: the question is a conversational
        turn and the expected answer the next one, encoding what only the
        interviewee knows. `B sãn n kẽ n gʋʋls tɩ bõe?` expects a list of proper
        nouns from one show, and all four checkpoints echoed the question.

    `expliquer_proverbe` was picked next because the proverb is quoted in full
    and the turn asks `võor yaa bõe?`, what does it mean, so the answer looked
    determinable from the prompt. **Measured, it is not**: the expected
    gloss shares 18 % of its words with the proverb it explains, the level of
    function words. It is a second maxim, one defensible reading among several,
    and the four checkpoints recover their question two to four times more than
    their reference, `checkpoint-872` worst of all. A reviewer holding that
    reference crosses out any other correct gloss, so the item scores « landed
    on the site's wording », not « answered relevantly ».

    **These twelve therefore stay as the set that produced the numbers already
    on disk, and nothing more.** Swapping the family a fourth time would repeat
    what has failed three times: each problem was found by a speaker after the
    fact, never by the code, and each replacement was chosen from whatever the
    val set happened to hold. Of the eleven families there, only `titrer`
    grades **without a reference** -- the text is in the prompt, so a speaker
    judges whether the title fits it -- and its prompts run 275 words.
    """
    import random

    pose = [t for t in turns("val")
            if t["variant"].startswith("mos") and t["task"] == "expliquer_proverbe"]
    if len(pose) < combien:
        raise SystemExit(f"{len(pose)} gradable Moore questions in val, "
                         f"{combien} needed")
    return random.Random(GRAINE_TENUES).sample(pose, combien)


def label(start: str) -> str:
    """The start, as a run name: `C@checkpoint-4820` becomes `C-4820`.

    The point belongs in the name. Two SFT runs from `C` at epoch 4 and at
    epoch 5 are two different experiments, and a name that hid the difference
    would let the second resume from the first's checkpoints.
    """
    arm, _, point = start.partition("@")
    return f"{arm}-{point.rpartition('-')[2]}" if point else arm


def resume_refused(state: dict, total_steps: int) -> str | None:
    """Why this checkpoint cannot continue under the current ceiling, or None.

    A resume that crosses a change of set or of epoch count is invisible. Once,
    `base` resumed an older run and trained **119 steps** on the repaired set,
    while `P2` resumed a checkpoint at step 1,776 under a ceiling of 1,310 and
    therefore trained **zero**. Nothing failed: transformers printed a mismatch
    warning among the download bars and the cell reported both runs done.

    `trainer_state.json` carries the ceiling the resumed run was built with,
    which is `steps_per_epoch * epochs`. It moves as soon as the set grows or
    shrinks, or the epoch cap changes, so comparing it catches both without
    fingerprinting the set separately.
    """
    done = state.get("global_step", 0)
    before = state.get("max_steps")
    if done >= total_steps:
        return (f"it stopped at step {done:,} and this run caps at "
                f"{total_steps:,}: there is nothing left to train")
    if before is not None and before != total_steps:
        return (f"it was built for {before:,} steps and this run caps at "
                f"{total_steps:,}: the set or the epoch count has changed since")
    return None


def resolve_run(start: str, out: Path, fresh: bool) -> tuple[Path, str | None]:
    """Where this run lives, and the checkpoint to resume from."""
    start = label(start)
    if not fresh:
        for folder in sorted(out.glob(f"sft-{start}-*"), reverse=True):
            local = list(folder.glob("checkpoint-*"))
            if local:
                resume = max(local, key=lambda d: int(d.name.rpartition("-")[2]))
                print(f"RESUMING locally from {resume}", flush=True)
                return folder, str(resume)
        for name in reversed(s3.list_runs(RUNS_URI, f"sft-{start}-")):
            found = s3.latest_checkpoint(f"{RUNS_URI}/{name}")
            if not found:
                continue
            folder = out / name
            s3.download_dir(f"{RUNS_URI}/{name}/{found[0]}", folder / found[0])
            if not (folder / found[0] / s3.REQUIRED_TO_RESUME).exists():
                print(f"  {name}/{found[0]}: no {s3.REQUIRED_TO_RESUME}, "
                      f"cannot resume (adapter kept for evaluation)",
                      file=sys.stderr)
                continue
            print(f"RESUMING from S3: {name}/{found[0]}", flush=True)
            return folder, str(folder / found[0])

    folder = out / f"sft-{start}-{dt.datetime.now():%Y%m%d-%H%M}"
    print(f"FRESH RUN: {folder.name}"
          + ("" if fresh else f"  (nothing to resume for {start})"), flush=True)
    return folder, None


def train(start: str, epochs: int, out: Path, fresh: bool, upload: bool,
          smoke: int) -> int:
    import gc

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    folder, resume = resolve_run(start, out, fresh or bool(smoke))
    run = folder.name
    uri = f"{RUNS_URI}/{run}"

    if upload and not smoke:
        why = s3.writable(uri)
        if why:
            raise SystemExit(
                f"S3 refuses a write to {uri}: {why}. Nothing would survive "
                "this run: check the AWS secrets before paying for the card.")
        print(f"S3 writable: {uri}", flush=True)

    base_id, adapter = start_from(start, out)
    tokenizer = AutoTokenizer.from_pretrained(base_id)

    def prepare(split: str) -> tuple[list, int]:
        kept, lost = [], 0
        for turn in turns(split):
            example = encode(turn, tokenizer)
            if example is None:
                lost += 1
            else:
                kept.append(example)
        return kept, lost

    encoded, dropped = prepare("train")
    if not encoded:
        raise SystemExit("no turn survived encoding: check `variantes`")
    print(f"  {len(encoded):,} turns encoded, {dropped:,} dropped "
          f"(longer than {WINDOW:,} tokens, or malformed)", flush=True)

    # The held-back split was loaded by `verify` to be counted and then thrown
    # away, so one run produced **zero** evaluation points.
    # Without it the only visible signal is the training loss, which falls at
    # every epoch boundary whether the model generalises or memorises. That
    # shape is readable only in hindsight; an evaluation loss says it while the
    # card is still running.
    held, held_dropped = prepare("val")
    print(f"  {len(held):,} turns held back for evaluation, "
          f"{held_dropped:,} dropped", flush=True)

    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16,
                                                 device_map="auto")
    if adapter:
        from peft import PeftModel

        # Merged, not continued: see the module docstring and `configs/sft.json`.
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **CONFIG["lora"]))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        # A frozen adapter trains, shows a falling loss, and learns nothing.
        raise SystemExit("no trainable parameter: the adapter loaded frozen")
    model.print_trainable_parameters()

    steps_per_epoch = -(-len(encoded) // (BATCH * ACCUM))
    total_steps = smoke or steps_per_epoch * epochs
    print(f"  {steps_per_epoch:,} steps/epoch, {total_steps:,} total", flush=True)

    if resume and not smoke:
        pourquoi = resume_refused(
            json.loads((Path(resume) / s3.REQUIRED_TO_RESUME).read_text(
                encoding="utf-8")), total_steps)
        if pourquoi:
            raise SystemExit(
                f"refusing to resume {Path(resume).name}: {pourquoi}.\n"
                f"This run has {len(encoded):,} turns over {epochs} epochs. "
                "Pass fresh=True to open a new run, or restore the previous "
                "set and ceiling to continue this one.")

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
        os.environ["WANDB_DISABLED"] = "true"
        print(f"SMOKE RUN: {smoke} steps, no checkpoint, no upload", flush=True)

    tracking = configure_wandb(start, run)
    print(f"W&B: {'project ' + os.environ['WANDB_PROJECT'] + ', run ' + run}"
          if tracking else "W&B: disabled", flush=True)

    settings = dict(
        output_dir=str(folder),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        gradient_checkpointing=CONFIG["recalcul_activations"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=CONFIG["optimiseur"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=max(1, round(CONFIG["warmup_ratio"] * total_steps)),
        bf16=True,
        logging_steps=CONFIG["logging_steps"],
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // CONFIG["sauvegardes_par_epoque"]),
        save_total_limit=CONFIG["save_total_limit_disque"],
        # The evaluation lands **on** the save, not between two: that way each
        # checkpoint carries the number that says whether it was worth keeping.
        # One cadence rather than two also means one fewer knob to set wrong.
        #
        # `load_best_model_at_end` is here for one reason, and it is not the
        # loading: it makes `save_total_limit` **exempt the argmin from the
        # rotation**. Without it the cadences agree and the pruning still eats
        # the best point, because the pruning keeps the LAST n, not the best.
        # Measured on two runs of the same recipe: on the first the minimum was
        # step 588 and survived; on the second it was step 591 and was deleted,
        # leaving step 788 at eval 1.3036 against the 1.2649 that had actually
        # been reached. A run produces a curve and the point is chosen by
        # measuring; a curve whose lowest point is thrown away
        # cannot be measured, and the loss is invisible: the run looks
        # complete and every file is where it should be.
        **({"eval_strategy": "steps",
            "eval_steps": max(1, steps_per_epoch // CONFIG["sauvegardes_par_epoque"]),
            "per_device_eval_batch_size": BATCH,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False} if held else {}),
        report_to=["wandb"] if tracking else [],
        run_name=run,
        remove_unused_columns=False,
    )
    if smoke:
        # `load_best_model_at_end` goes with the saves: transformers refuses an
        # evaluation cadence of `steps` with a save cadence of `no`, rightly,
        # since there is no point to load.
        settings |= {"max_steps": smoke, "save_strategy": "no",
                     "load_best_model_at_end": False,
                     "logging_steps": max(1, smoke // 5)}

    callbacks: list = [Log()]
    if upload and not smoke:
        callbacks.append(checkpoints.callback(uri, log, steps_per_epoch))

    Trainer(model=model, args=TrainingArguments(**settings),
            train_dataset=encoded,
            eval_dataset=held or None,
            data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE),
            callbacks=callbacks).train(resume_from_checkpoint=resume)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sft.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\n-> {folder}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not `choices=STARTS`: an arm may name its point, `C@checkpoint-4820`,
    # because the last checkpoint is not the best one.
    parser.add_argument("--start", required=True, metavar="START",
                        help="'base' for the control, or a CPT arm to build on, "
                             "optionally with its point: C@checkpoint-4820")
    parser.add_argument("--verify", action="store_true",
                        help="resolve and check the set, without torch")
    parser.add_argument("--demo", action="store_true",
                        help="check the unlisted-variant guard, needs nothing "
                             "on disk but the set")
    parser.add_argument("--smoke", type=int, default=0, metavar="STEPS")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/content/sft"))
    parser.add_argument("--epochs", type=int, default=CONFIG["epoques"])
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    arm = args.start.partition("@")[0]
    if arm not in STARTS:
        parser.error(f"unknown start {arm!r}, expected one of {', '.join(STARTS)}"
                     " (optionally followed by @checkpoint-NNNN)")

    if args.demo:
        demo()
        return 0
    if args.verify:
        return verify(args.start)
    return train(args.start, args.epochs, args.out, args.fresh,
                 not args.no_upload, args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
