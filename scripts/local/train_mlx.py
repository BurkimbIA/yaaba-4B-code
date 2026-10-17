#!/usr/bin/env python3
"""Run a bounded, provenance-tracked SFT experiment on Apple silicon.

MLX-LM is the primary backend. ``--backend mps`` is a slower Transformers/PEFT
fallback. Both consume the same prepared chats and supervise assistant tokens
only. A real run requires a gated held-out file so leak checking cannot be
silently skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yaaba.local_experiments import (  # noqa: E402
    append_jsonl,
    assistant_completion_tokens,
    canonical_hash,
    ensure_resume_contract,
    git_identity,
    model_identity,
    model_with_chat_template,
    package_versions,
    prepare_chat_dataset,
    read_jsonl,
    validate_prepared_assistant_masks,
    write_json,
)

TRAIN_LOSS = re.compile(
    r"Iter\s+(?P<step>\d+):\s+Train loss\s+(?P<loss>[0-9.eE+-]+).*?"
    r"It/sec\s+(?P<steps>[0-9.eE+-]+).*?Tokens/sec\s+(?P<tokens>[0-9.eE+-]+)"
)
VAL_LOSS = re.compile(r"Iter\s+(?P<step>\d+):\s+Val loss\s+(?P<loss>[0-9.eE+-]+)")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("mlx", "mps"), default="mlx")
    result.add_argument("--model", required=True,
                        help="local model/snapshot path; remote ids require --model-revision")
    result.add_argument("--model-revision")
    result.add_argument("--chat-template", type=Path,
                        help="explicit Jinja template when the checkpoint omits one")
    result.add_argument("--data", type=Path, required=True, help="source chat JSONL")
    result.add_argument("--heldout", type=Path,
                        help="gated tenu_a_lecart.jsonl; mandatory except --mock-smoke")
    result.add_argument("--output", type=Path, required=True,
                        help="new run directory, or existing one with --resume")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--mock-smoke", action="store_true",
                        help="prepare synthetic data and exercise contracts; no model/runtime")
    result.add_argument("--seed", type=int, default=20261017)
    result.add_argument("--default-split", choices=("train", "valid", "test"))
    result.add_argument("--group-field", default="group_id")
    result.add_argument(
        "--allow-ungrouped-technical-pilot", action="store_true",
        help="permit existing source splits without group provenance for a bounded "
             "software/backward-pass pilot; never a quality experiment",
    )
    result.add_argument(
        "--technical-pilot", action="store_true",
        help="label a lineage or other bounded pilot as technical-only",
    )
    result.add_argument(
        "--variants", default="fr_fr,fr_long,mos_mos,traduction,correction",
        help="comma-separated variants to train",
    )
    result.add_argument(
        "--excluded-variants", default="fr_mos,mos_fr",
        help="comma-separated variants explicitly excluded; unknown variants block",
    )
    result.add_argument("--valid-percent", type=float, default=5.0)
    result.add_argument("--test-percent", type=float, default=5.0)
    result.add_argument("--max-train-samples", type=_positive, default=2048)
    result.add_argument("--max-valid-samples", type=_positive, default=256)
    result.add_argument("--max-test-samples", type=_positive, default=256)
    result.add_argument("--max-steps", type=_positive, default=200)
    result.add_argument("--walltime-minutes", type=_positive, default=60)
    result.add_argument("--memory-limit-gb", type=_positive, default=32)
    result.add_argument("--minimum-available-gb", type=_positive, default=4)
    result.add_argument("--max-seq-length", type=_positive, default=1024)
    result.add_argument("--batch-size", type=_positive, default=1)
    result.add_argument("--grad-accumulation", type=_positive, default=8)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--save-every", type=_positive, default=25)
    result.add_argument("--eval-every", type=_positive, default=25)
    result.add_argument("--report-every", type=_positive, default=10)
    result.add_argument("--num-layers", type=int, default=8,
                        help="MLX: last N layers; -1 means every layer")
    result.add_argument("--lora-rank", type=_positive, default=16)
    result.add_argument("--lora-scale", type=float, default=32.0)
    result.add_argument("--lora-dropout", type=float, default=0.05)
    result.add_argument("--track", action="store_true",
                        help="require the Yaaba MLflow contract and local server")
    result.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    result.add_argument("--experiment-name", default="yaaba-mac-sft")
    result.add_argument("--run-name")
    result.add_argument("--mlflow-run-id")
    return result


def _mock_files(output: Path) -> tuple[Path, Path, Path]:
    fixture = output / "mock-input"
    fixture.mkdir(parents=True, exist_ok=True)
    data = fixture / "train.jsonl"
    heldout = fixture / "heldout.jsonl"
    model = fixture / "mock-model"
    model.mkdir(exist_ok=True)
    data.write_text(
        json.dumps({
            "id": "mock-1", "split": "train",
            "messages": [
                {"role": "user", "content": "mock question"},
                {"role": "assistant", "content": "mock answer"},
            ],
        }) + "\n",
        encoding="utf-8",
    )
    heldout.write_text(
        json.dumps({"capacite": "anglais", "text": "independent held out text"})
        + "\n", encoding="utf-8",
    )
    (model / "config.json").write_text(
        json.dumps({"model_type": "explicit-mock-no-weights"}), encoding="utf-8"
    )
    return data, heldout, model


def _resolve_remote(model: str, revision: str | None) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()
    if not revision:
        raise ValueError("remote model requires --model-revision")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision)).resolve()


def _metrics_from_log(log: Path, metrics: Path, offset: int = 0) -> None:
    if not log.exists():
        return
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := TRAIN_LOSS.search(line):
            step = offset + int(match.group("step"))
            append_jsonl(metrics, {
                "step": step, "kind": "ml", "name": "train_loss",
                "value": float(match.group("loss")), "source": "mlx-lm stdout",
            })
            append_jsonl(metrics, {
                "step": step, "kind": "technical", "name": "tokens_per_second",
                "value": float(match.group("tokens")), "source": "mlx-lm stdout",
            })
        if match := VAL_LOSS.search(line):
            append_jsonl(metrics, {
                "step": offset + int(match.group("step")), "kind": "ml",
                "name": "validation_loss", "value": float(match.group("loss")),
                "source": "mlx-lm stdout",
            })


def _monitor(pid: int, path: Path, stop: threading.Event, *,
             memory_limit_bytes: int, minimum_available_bytes: int,
             state: dict) -> None:
    try:
        import psutil
        process = psutil.Process(pid)
    except (ImportError, OSError):
        return
    low_available_samples = 0
    while not stop.wait(2):
        try:
            children = process.children(recursive=True)
            memory = process.memory_info().rss + sum(
                child.memory_info().rss for child in children
            )
            available = psutil.virtual_memory().available
            append_jsonl(path, {
                "time_unix": time.time(), "rss_bytes": memory,
                "system_available_bytes": available,
            })
            low_available_samples = (
                low_available_samples + 1 if available < minimum_available_bytes else 0
            )
            if memory > memory_limit_bytes:
                state["reason"] = "rss_memory_limit"
            elif low_available_samples >= 3:
                state["reason"] = "system_memory_floor"
            if state.get("reason"):
                try:
                    os.killpg(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return


def _run_bounded(command: list[str], log: Path, resources: Path,
                 walltime_seconds: int, *, memory_limit_bytes: int,
                 minimum_available_bytes: int) -> tuple[int, str, float]:
    started = time.monotonic()
    stop = threading.Event()
    monitor_state: dict[str, str] = {}
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        monitor = threading.Thread(
            target=_monitor, args=(process.pid, resources, stop),
            kwargs={"memory_limit_bytes": memory_limit_bytes,
                    "minimum_available_bytes": minimum_available_bytes,
                    "state": monitor_state},
            daemon=True,
        )
        monitor.start()
        try:
            code = process.wait(timeout=walltime_seconds)
            reason = monitor_state.get(
                "reason", "completed" if code == 0 else "process_failed"
            )
        except subprocess.TimeoutExpired:
            reason = "walltime_limit"
            os.killpg(process.pid, signal.SIGINT)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
        finally:
            stop.set()
            monitor.join(timeout=3)
    return code, reason, time.monotonic() - started


def _latest_segment_checkpoint(run: Path) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []
    for segment in (run / "adapters").glob("segment-*"):
        try:
            offset = int(segment.name.rpartition("-")[2])
        except ValueError:
            continue
        for checkpoint in segment.glob("*_adapters.safetensors"):
            local = checkpoint.name.partition("_")[0]
            if local.isdigit():
                candidates.append((offset + int(local), checkpoint))
        final = segment / "adapters.safetensors"
        state = segment / "segment_state.json"
        if final.exists() and state.exists():
            completed = int(json.loads(state.read_text())["completed_steps"])
            candidates.append((completed, final))
    if not candidates:
        return None, 0
    step, path = max(candidates)
    return path, step


def _mlx_config(args: argparse.Namespace, model: Path, run: Path,
                completed: int, resume_file: Path | None) -> tuple[dict, Path]:
    remaining = args.max_steps - completed
    if remaining <= 0:
        raise ValueError(f"run already reached max_steps={args.max_steps}")
    adapter = run / "adapters" / f"segment-{completed:07d}"
    config = {
        "model": str(model),
        "train": True,
        "fine_tune_type": "lora",
        "data": str(run / "data"),
        "seed": args.seed,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": remaining,
        "val_batches": 8,
        "learning_rate": args.learning_rate,
        "steps_per_report": min(args.report_every, remaining),
        "steps_per_eval": min(args.eval_every, remaining),
        "grad_accumulation_steps": args.grad_accumulation,
        "resume_adapter_file": str(resume_file) if resume_file else None,
        "adapter_path": str(adapter),
        "save_every": min(args.save_every, remaining),
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": True,
        "clear_cache_threshold": 2 * 1024**3,
        "mask_prompt": True,
        "lora_parameters": {
            "rank": args.lora_rank, "scale": args.lora_scale,
            "dropout": args.lora_dropout,
            # No explicit keys: MLX-LM discovers all linear modules in the last
            # N layers, including Qwen3.5 linear-attention projections.
        },
        "report_to": None,
    }
    return config, adapter


def _train_mlx(args: argparse.Namespace, model: Path, run: Path) -> dict:
    resume_file, completed = _latest_segment_checkpoint(run)
    if args.resume and resume_file is None:
        raise ValueError("no saved MLX adapter checkpoint to resume")
    if not args.resume:
        resume_file, completed = None, 0
    config, adapter = _mlx_config(args, model, run, completed, resume_file)
    config_path = run / f"mlx-config-{completed:07d}.json"
    write_json(config_path, config)
    command = [
        sys.executable, str(Path(__file__).with_name("mlx_lora_bounded.py")),
        "--memory-limit-bytes", str(args.memory_limit_gb * 1024**3),
        "--config", str(config_path),
    ]
    write_json(run / "command.json", command)
    if args.dry_run:
        return {"status": "dry_run", "command": command, "completed_steps": completed}

    log = run / f"mlx-{completed:07d}.log"
    code, reason, duration = _run_bounded(
        command, log, run / "resources.jsonl", args.walltime_minutes * 60,
        memory_limit_bytes=args.memory_limit_gb * 1024**3,
        minimum_available_bytes=args.minimum_available_gb * 1024**3,
    )
    _metrics_from_log(log, run / "metrics.jsonl", offset=completed)
    latest, total = _latest_segment_checkpoint(run)
    if code == 0 and (adapter / "adapters.safetensors").exists():
        total = args.max_steps
        latest = adapter / "adapters.safetensors"
        write_json(adapter / "segment_state.json", {"completed_steps": total})
    append_jsonl(run / "metrics.jsonl", {
        "step": total, "kind": "technical", "name": "wall_seconds",
        "value": duration, "source": "runner monotonic clock",
    })
    return {
        "status": reason, "exit_code": code, "duration_seconds": duration,
        "completed_steps": total, "latest_adapter": str(latest) if latest else None,
        "resume_semantics": "adapter weights restored; optimizer state resets per MLX-LM",
    }


def _train_mps(args: argparse.Namespace, model_path: Path, run: Path) -> dict:
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

    if not torch.backends.mps.is_available():
        raise RuntimeError("Torch MPS is unavailable")
    try:
        import psutil

        physical_memory = psutil.virtual_memory().total
    except ImportError:
        psutil = None
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    torch.mps.set_per_process_memory_fraction(
        min(1.0, args.memory_limit_gb * 1024**3 / physical_memory)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def encoded(split: str) -> list[dict]:
        path = run / "data" / f"{split}.jsonl"
        rows = []
        if not path.exists():
            return rows
        for _, record in read_jsonl(path):
            item = assistant_completion_tokens(
                tokenizer, record["messages"], args.max_seq_length
            )
            if item is not None:
                rows.append(item)
        return rows

    train_set, valid_set = encoded("train"), encoded("valid")
    if not train_set:
        raise ValueError("no training example survived tokenization")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", target_modules="all-linear", r=args.lora_rank,
        lora_alpha=args.lora_scale, lora_dropout=args.lora_dropout,
    ))

    started = time.monotonic()

    class BoundAndLog(TrainerCallback):
        low_available_samples = 0

        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            for name, value in (logs or {}).items():
                if isinstance(value, (int, float)):
                    append_jsonl(run / "metrics.jsonl", {
                        "step": state.global_step,
                        "kind": "ml" if "loss" in name else "technical",
                        "name": name, "value": float(value), "source": "Trainer",
                    })

        def on_step_end(self, _args, _state, control, **_kwargs):
            if time.monotonic() - started >= args.walltime_minutes * 60:
                control.should_save = True
                control.should_training_stop = True
            if psutil is not None:
                process = psutil.Process()
                rss = process.memory_info().rss + sum(
                    child.memory_info().rss for child in process.children(recursive=True)
                )
                available = psutil.virtual_memory().available
                append_jsonl(run / "resources.jsonl", {
                    "time_unix": time.time(), "rss_bytes": rss,
                    "system_available_bytes": available,
                })
                self.low_available_samples = (
                    self.low_available_samples + 1
                    if available < args.minimum_available_gb * 1024**3 else 0
                )
                if (rss > args.memory_limit_gb * 1024**3 or
                        self.low_available_samples >= 3):
                    control.should_save = True
                    control.should_training_stop = True
            return control

    output = run / "mps-checkpoints"
    resume = None
    if args.resume:
        checkpoints = [p for p in output.glob("checkpoint-*") if p.is_dir()]
        if not checkpoints:
            raise ValueError("no MPS Trainer checkpoint to resume")
        resume = max(checkpoints, key=lambda p: int(p.name.rpartition("-")[2]))
    settings = TrainingArguments(
        output_dir=str(output), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate, seed=args.seed, data_seed=args.seed,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=min(args.report_every, args.max_steps), save_strategy="steps",
        save_steps=min(args.save_every, args.max_steps), save_total_limit=3,
        eval_strategy="steps" if valid_set else "no",
        eval_steps=min(args.eval_every, args.max_steps),
        report_to=[], remove_unused_columns=False,
    )
    write_json(run / "command.json", {"backend": "mps", "settings": settings.to_dict()})
    if args.dry_run:
        return {"status": "dry_run", "train_examples": len(train_set)}
    trainer = Trainer(
        model=model, args=settings, train_dataset=train_set,
        eval_dataset=valid_set or None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        callbacks=[BoundAndLog()],
    )
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    duration = time.monotonic() - started
    return {
        "status": "completed" if result.global_step >= args.max_steps else "walltime_limit",
        "completed_steps": result.global_step, "duration_seconds": duration,
        "latest_adapter": str(output),
        "resume_semantics": "Trainer checkpoint restores model, optimizer, scheduler and step",
    }


def _tracking_context(args: argparse.Namespace, manifest: dict, run: Path):
    if not args.track:
        return contextlib.nullcontext(None)
    try:
        from yaaba.experiment_tracking import (
            build_manifest,
            tracked_run,
            write_manifest,
        )
    except ImportError as error:
        raise RuntimeError(
            "--track requires src/yaaba/experiment_tracking.py from the MLOps branch"
        ) from error

    identity = manifest["model"]
    tracking_manifest = build_manifest(
        run_kind=("mac-sft-technical-smoke"
                  if args.mock_smoke or args.technical_pilot else "mac-sft"),
        evidence_kind="OBSERVED",
        config={
            "sha256": manifest["contract_sha256"],
            "resolved": manifest["training"],
        },
        data=[
            {"path": manifest["data"]["source"],
             "sha256": manifest["data"]["source_sha256"]},
            {"path": manifest["data"]["heldout"],
             "sha256": manifest["data"]["heldout_sha256"]},
        ],
        code=manifest["code"],
        model={**identity, "sha256": identity.get("identity_sha256",
                                                   canonical_hash(identity))},
        tokenizer={**manifest["chat_template"],
                   "sha256": manifest["chat_template"]["sha256"]},
        seed=args.seed,
        budget=manifest["limits"],
        stop_criteria={"steps": args.max_steps,
                       "walltime_seconds": args.walltime_minutes * 60},
        extra={"claim_scope": manifest["claim_scope"], "backend": args.backend},
    )
    write_manifest(tracking_manifest, run / "tracking_manifest.json")
    return tracked_run(
        tracking_manifest, experiment_name=args.experiment_name,
        run_name=args.run_name or run.name, tracking_uri=args.tracking_uri,
        run_id=args.mlflow_run_id,
    )


def _log_tracking(tracked, run: Path, result: dict) -> None:
    if tracked is None:
        return
    metrics = run / "metrics.jsonl"
    if metrics.exists():
        for _, row in read_jsonl(metrics):
            tracked.log_metric(
                row["name"], row["value"], kind=row["kind"],
                step=int(row.get("step", 0)),
            )
    for name in ("manifest.json", "tracking_manifest.json", "result.json",
                 "resources.jsonl", "command.json"):
        path = run / name
        if path.exists():
            tracked.log_artifact(path, artifact_path="run")
    latest = result.get("latest_adapter")
    if latest and Path(latest).exists():
        tracked.log_checkpoint(
            latest, step=int(result.get("completed_steps", 0)),
            metadata={"backend": result.get("backend", "training"),
                      "resume_semantics": result.get("resume_semantics")},
        )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)

    if args.mock_smoke:
        args.data, args.heldout, mock_model = _mock_files(run)
        args.model = str(mock_model)
        args.model_revision = "explicit-mock"
    elif args.heldout is None:
        raise SystemExit("--heldout is mandatory for a real training run")

    data_provenance = prepare_chat_dataset(
        args.data, run / "data", seed=args.seed, heldout_path=args.heldout,
        default_split=args.default_split, group_field=args.group_field,
        valid_percent=args.valid_percent, test_percent=args.test_percent,
        limits={
            "train": args.max_train_samples,
            "valid": args.max_valid_samples,
            "test": args.max_test_samples,
        },
        include_variants=(None if args.mock_smoke else
                          {value for value in args.variants.split(",") if value}),
        exclude_variants=(None if args.mock_smoke else
                          {value for value in args.excluded_variants.split(",") if value}),
        allow_ungrouped_technical_pilot=(
            args.mock_smoke or args.allow_ungrouped_technical_pilot
        ),
    )
    if not data_provenance["assistant_examples"]["train"]:
        raise SystemExit("prepared dataset has no train examples")

    if args.mock_smoke or args.dry_run:
        model = Path(args.model).expanduser().resolve() if Path(args.model).exists() else args.model
    else:
        model = _resolve_remote(args.model, args.model_revision)
    identity = model_identity(model, args.model_revision)
    if args.mock_smoke:
        runtime_model, template_identity = Path(model), {
            "source": "explicit mock", "sha256": canonical_hash("mock")
        }
    else:
        runtime_model, template_identity = model_with_chat_template(
            model, run / "model-overlay", args.chat_template
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            runtime_model, local_files_only=True, trust_remote_code=False
        )
        mask_report = validate_prepared_assistant_masks(
            run / "data", tokenizer, args.max_seq_length
        )
        data_provenance["post_tokenization_assistant_mask"] = mask_report
        data_provenance["assistant_examples"] = {
            split: details["kept"]
            for split, details in mask_report["splits"].items()
        }
        data_provenance["files"] = {
            split: {
                "path": str(run / "data" / f"{split}.jsonl"),
                "sha256": details["sha256"],
            }
            for split, details in mask_report["splits"].items()
            if details["kept"]
        }
        write_json(run / "data" / "provenance.json", data_provenance)
        if not data_provenance["assistant_examples"].get("train", 0):
            raise SystemExit(
                "no train example has an assistant token within --max-seq-length"
            )
    training = {
        "backend": args.backend, "seed": args.seed,
        "max_steps": args.max_steps, "walltime_minutes": args.walltime_minutes,
        "max_seq_length": args.max_seq_length, "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "learning_rate": args.learning_rate, "save_every": args.save_every,
        "eval_every": args.eval_every, "report_every": args.report_every,
        "num_layers": args.num_layers,
        "lora_rank": args.lora_rank, "lora_scale": args.lora_scale,
        "lora_dropout": args.lora_dropout,
        "memory_limit_gb": args.memory_limit_gb,
        "minimum_available_gb": args.minimum_available_gb,
    }
    contract = {
        "model": identity, "chat_template": template_identity,
        "data": data_provenance, "training": training,
        "evidence_kind": "TECHNICAL_MOCK" if args.mock_smoke else "OBSERVED_LOCAL_RUN",
    }
    contract_hash = ensure_resume_contract(run / "resume_contract.json", contract, args.resume)
    manifest = {
        **contract, "contract_sha256": contract_hash,
        "code": git_identity(ROOT),
        "versions": package_versions(
            ("mlx", "mlx-lm", "torch", "transformers", "peft", "sacrebleu")
        ),
        "limits": {
            "samples": data_provenance["assistant_examples"],
            "steps": args.max_steps, "walltime_seconds": args.walltime_minutes * 60,
            "process_memory_bytes": args.memory_limit_gb * 1024**3,
            "minimum_system_available_bytes": args.minimum_available_gb * 1024**3,
        },
        "claim_scope": (
            "software smoke only; no model quality evidence" if args.mock_smoke else
            "lineage technical pilot; backward-pass evidence only, no model quality "
            "or human Mooré judgment" if args.technical_pilot else
            "ungrouped technical pilot; backward-pass evidence only, no quality or "
            "independent-split evidence" if not data_provenance["grouping_verified"] else
            "automatic training metrics only; no human Mooré quality judgment"
        ),
    }
    write_json(run / "manifest.json", manifest)

    with _tracking_context(args, manifest, run) as tracked:
        if args.mock_smoke:
            result = {
                "status": "mock_smoke_passed", "completed_steps": 0,
                "backend": "explicit-mock-no-training",
            }
        elif args.backend == "mlx":
            result = _train_mlx(args, Path(runtime_model), run)
        else:
            result = _train_mps(args, Path(runtime_model), run)
        if tracked is not None:
            result["mlflow_run_id"] = tracked.run_id
        write_json(run / "result.json", result)
        _log_tracking(tracked, run, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") in {
        "process_failed", "rss_memory_limit", "system_memory_floor"
    } else 0


if __name__ == "__main__":
    raise SystemExit(main())
