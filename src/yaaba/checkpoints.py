"""Trainer callback that persists checkpoints as they are written.

Kept apart from `yaaba.s3` so that module stays importable without transformers.

Two kinds of save, for two different purposes. *Interval* saves exist so a dead
runtime does not cost hours; only the newest is kept. *Epoch* saves are the
material of the measurement curve; all are kept, minus their optimizer
state, since evaluation never resumes from them.

An epoch save is the **first save landing in a new epoch**, not a save falling
exactly on the boundary: with 1,201 steps per epoch and a save every 219, no
save ever falls on a boundary, and a rule asking for one keeps nothing.

Keeping only the first save of each epoch is still wrong while the curve is
falling, because that save is the worst of its epoch. With 655 steps per epoch
and a save every 218, the end-of-first-epoch save, nearly always the minimum of
the evaluation loss, is exactly the one such a rule deletes. So `keep_all`
defaults to True: every save is kept, with its optimizer state pruned.
"""

from __future__ import annotations

from pathlib import Path

from . import s3


class CheckpointUploader:
    """Upload each checkpoint to `base_uri` the moment the Trainer writes it.

    Args:
        base_uri: `s3://.../cpt/<run>`; empty disables every upload.
        log: training log, uploaded on every logging step because it is small
            and makes the run observable from outside the runtime.
        steps_per_epoch: marks which saves open a new epoch. 0 keeps all.
        keep_all: keep every save instead of rotating the interval ones.
        False restores the old behaviour.
    """

    def __init__(self, base_uri: str, log: Path | None = None,
                 steps_per_epoch: int = 0, keep_all: bool = True) -> None:
        self.base = base_uri.rstrip("/")
        self.log = log
        self.steps_per_epoch = steps_per_epoch
        self.keep_all = keep_all
        self._previous: tuple[str, bool] | None = None
        self._epoch_kept = -1

    def opens_an_epoch(self, step: int) -> bool:
        """Whether this save is the first of a new epoch, and record it.

        Called once per save, in order. Kept apart from `on_save` so the rule
        can be checked without a Trainer.
        """
        if not self.steps_per_epoch:
            return False
        epoch = step // self.steps_per_epoch
        if epoch <= self._epoch_kept:
            return False
        self._epoch_kept = epoch
        return True

    # -- TrainerCallback surface -------------------------------------------

    def on_log(self, args, state, control, **kwargs) -> None:
        if self.base and self.log and self.log.exists():
            s3.upload_file(self.log, f"{self.base}/progression.jsonl")

    def on_save(self, args, state, control, **kwargs) -> None:
        if not self.base:
            return
        step = state.global_step
        name = f"checkpoint-{step}"
        directory = Path(args.output_dir) / name
        # `opens_an_epoch` is still called, in order, because it carries state
        # and because the printed line says which saves mark an epoch.
        is_epoch = self.opens_an_epoch(step) or self.keep_all

        if self.log and self.log.exists():
            # a checkpoint without its curve cannot be compared to anything
            import shutil

            shutil.copy2(self.log, directory / self.log.name)

        s3.upload_dir(directory, f"{self.base}/{name}")
        print(f"  {name} {'(epoch, kept)' if is_epoch else '(interval)'}", flush=True)

        if self._previous:
            previous, was_epoch = self._previous
            uri = f"{self.base}/{previous}"
            dropped = (s3.prune_resume_state(uri) if was_epoch
                       else s3.delete_checkpoint(uri))
            if dropped:
                verb = "pruned" if was_epoch else "deleted"
                print(f"  {verb} {previous} ({dropped} files)", flush=True)
        self._previous = (name, is_epoch)


def callback(base_uri: str, log: Path | None = None, steps_per_epoch: int = 0,
             keep_all: bool = True):
    """Build the callback, deriving from TrainerCallback lazily."""
    from transformers import TrainerCallback

    class _Callback(CheckpointUploader, TrainerCallback):
        pass

    return _Callback(base_uri, log, steps_per_epoch, keep_all)


def by_eval(run_uri: str) -> tuple[str, int, float] | None:
    """The kept checkpoint whose evaluation loss is lowest.

    The epoch count is a ceiling and the checkpoint is chosen by measuring the
    curve. The most trained checkpoint is often not the best one: on one run the
    evaluation loss bottomed at 1.2643 after one epoch and climbed to 1.4197 by
    the third while the training loss fell from 1.01 to 0.57.

    The answer is the argmin over the checkpoints whose weights still exist on
    S3: a step can be the curve's minimum and have no weights left to load.

    Returns `(name, step, eval_loss)`, or None when the run logged no
    evaluation.
    """
    import json

    raw = s3.read_bytes(f"{run_uri.rstrip('/')}/progression.jsonl")
    if not raw:
        return None
    kept = {step for _, step in s3.checkpoint_names(run_uri)}
    curve = [
        # The log is written by `Log`, whose step field is named `pas`.
        (row["eval_loss"], int(row["pas"]))
        for row in (json.loads(line) for line in
                    raw.decode("utf-8", "replace").splitlines() if line)
        if "eval_loss" in row and int(row.get("pas", 0)) in kept
    ]
    if not curve:
        return None
    loss, step = min(curve)
    return f"checkpoint-{step}", step, loss


def demo() -> None:
    """What `opens_an_epoch` and `by_eval` promise, without S3 or a Trainer."""
    # 1,201 steps per epoch and a save every 219: no save falls on an epoch
    # boundary, yet every epoch must keep one checkpoint, its first.
    up = CheckpointUploader("", steps_per_epoch=1201, keep_all=False)
    opening = [step for step in range(219, 6006, 219) if up.opens_an_epoch(step)]
    assert opening == [219, 1314, 2409, 3723, 4818], opening

    # While the curve still descends, the first save of an epoch is its worst.
    # Keeping only those kept 218, 872 and 1310 and discarded the minimum at 654,
    # hence `keep_all`.
    saved = [218, 436, 654, 872, 1090, 1308, 1310]
    first_only = CheckpointUploader("", steps_per_epoch=655, keep_all=False)
    assert [s for s in saved if first_only.opens_an_epoch(s)] == [218, 872, 1310]
    everything = CheckpointUploader("", steps_per_epoch=655)
    assert [s for s in saved
            if everything.opens_an_epoch(s) or everything.keep_all] == saved

    # `by_eval` takes the minimum, not the last point, among checkpoints whose
    # weights still exist.
    import json as _json

    curve = [{"pas": s, "eval_loss": e} for s, e in
             ((196, 1.32), (392, 1.29), (588, 1.26), (1176, 1.30),
              (1372, 1.20), (1764, 1.42))]
    raw = "\n".join(_json.dumps(r) for r in curve).encode()
    real_read, real_names = s3.read_bytes, s3.checkpoint_names
    try:
        s3.read_bytes = lambda _uri: raw
        # 1,372 is the curve's minimum but its weights were deleted.
        s3.checkpoint_names = lambda _uri: [
            ("checkpoint-196", 196), ("checkpoint-588", 588),
            ("checkpoint-1176", 1176), ("checkpoint-1764", 1764)]
        assert by_eval("s3://fake/run") == ("checkpoint-588", 588, 1.26)
        s3.read_bytes = lambda _uri: b""
        assert by_eval("s3://fake/run") is None  # a run with no evaluation
    finally:
        s3.read_bytes, s3.checkpoint_names = real_read, real_names
    print("yaaba.checkpoints: ok")
