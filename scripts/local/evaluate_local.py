#!/usr/bin/env python3
"""Generate a replayable local baseline without claiming human quality.

The evaluator stores exact model, adapter, input, chat-template and decoding
identities together with raw model responses. Reported scores are automatic
only (exact match and chrF when references exist).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yaaba.local_experiments import (  # noqa: E402
    canonical_hash,
    git_identity,
    model_identity,
    model_with_chat_template,
    normalized_text,
    package_versions,
    read_jsonl,
    sha256_file,
    write_json,
)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("mlx", "mps", "mock"), default="mlx")
    result.add_argument("--model", required=True)
    result.add_argument("--model-revision")
    result.add_argument("--chat-template", type=Path,
                        help="explicit Jinja template when the checkpoint omits one")
    result.add_argument("--adapter", type=Path)
    result.add_argument("--questions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--label", default="baseline")
    result.add_argument("--seed", type=int, default=20261017)
    result.add_argument("--max-samples", type=_positive, default=128)
    result.add_argument("--max-tokens", type=_positive, default=256)
    result.add_argument("--walltime-minutes", type=_positive, default=30)
    result.add_argument("--track", action="store_true")
    result.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    result.add_argument("--experiment-name", default="yaaba-local-evaluation")
    result.add_argument("--run-name")
    result.add_argument("--mlflow-run-id")
    return result


def _messages(record: dict[str, Any], line_number: int) -> list[dict[str, str]]:
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"line {line_number}: invalid messages")
        if messages[-1].get("role") == "assistant":
            messages = messages[:-1]
    elif isinstance(record.get("prompt"), str):
        messages = [{"role": "user", "content": record["prompt"]}]
    else:
        raise ValueError(f"line {line_number}: needs `messages` or string `prompt`")
    if not messages or messages[-1].get("role") == "assistant":
        raise ValueError(f"line {line_number}: prompt must end before assistant response")
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def _templated(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, dict]:
    options = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    prompt = tokenizer.apply_chat_template(messages, **options)
    return str(prompt), options


def _mock_generate(record_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    digest = hashlib.sha256(f"{record_id}\0{prompt}".encode()).hexdigest()[:16]
    response = f"[EXPLICIT MOCK RESPONSE {digest}]"
    return {
        "response": response, "prompt_tokens": len(prompt.split()),
        "generation_tokens": len(response.split()), "prompt_tps": None,
        "generation_tps": None, "peak_memory_gb": None,
        "finish_reason": "mock", "max_tokens": max_tokens,
    }


def _mlx_generator(model_path: Path, adapter: Path | None, seed: int):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    model, tokenizer = load(
        str(model_path), adapter_path=str(adapter) if adapter else None
    )
    sampler = make_sampler(temp=0.0)

    def generate(messages: list[dict[str, str]], max_tokens: int) -> tuple[dict, str, dict]:
        prompt, options = _templated(tokenizer, messages)
        text, final = "", None
        for response in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            text += response.text
            final = response
        if final is None:
            raise RuntimeError("MLX-LM returned no generation response")
        return ({
            "response": text.strip(), "prompt_tokens": final.prompt_tokens,
            "generation_tokens": final.generation_tokens,
            "prompt_tps": final.prompt_tps,
            "generation_tps": final.generation_tps,
            "peak_memory_gb": final.peak_memory,
            "finish_reason": final.finish_reason, "max_tokens": max_tokens,
        }, prompt, options)

    return generate


def _mps_generator(model_path: Path, adapter: Path | None, seed: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    def generate(messages: list[dict[str, str]], max_tokens: int) -> tuple[dict, str, dict]:
        prompt, options = _templated(tokenizer, messages)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        started = time.monotonic()
        with torch.no_grad():
            output = model.generate(
                **encoded, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        duration = time.monotonic() - started
        produced = output[0, encoded.input_ids.shape[1]:]
        count = int(produced.shape[0])
        return ({
            "response": tokenizer.decode(produced, skip_special_tokens=True).strip(),
            "prompt_tokens": int(encoded.input_ids.shape[1]),
            "generation_tokens": count, "prompt_tps": None,
            "generation_tps": count / duration if duration else None,
            "peak_memory_gb": None,
            "finish_reason": "length" if count == max_tokens else "stop",
            "max_tokens": max_tokens,
        }, prompt, options)

    return generate


def _records(path: Path, limit: int) -> Iterator[tuple[int, dict[str, Any]]]:
    for index, item in enumerate(read_jsonl(path)):
        if index >= limit:
            break
        yield item


def _tracking_context(args: argparse.Namespace, manifest: dict, output: Path):
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
        run_kind="local-generation-baseline",
        evidence_kind="OBSERVED",
        config={"decoding": manifest["decoding"],
                "sha256": canonical_hash(manifest["decoding"])},
        data={**manifest["questions"], "sha256": manifest["questions"]["sha256"]},
        code=manifest["code"],
        model={**identity, "sha256": identity.get("identity_sha256",
                                                   canonical_hash(identity))},
        tokenizer={**manifest["chat_template"],
                   "sha256": manifest["chat_template"]["sha256"]},
        seed=args.seed,
        budget={"max_samples": args.max_samples, "max_tokens": args.max_tokens,
                "walltime_seconds": args.walltime_minutes * 60},
        stop_criteria={"sample_limit": args.max_samples,
                       "walltime_seconds": args.walltime_minutes * 60},
        extra={"claim_scope": manifest["claim_scope"], "label": args.label,
               "adapter": manifest["adapter"]},
    )
    write_manifest(tracking_manifest, output / "tracking_manifest.json")
    return tracked_run(
        tracking_manifest, experiment_name=args.experiment_name,
        run_name=args.run_name or output.name, tracking_uri=args.tracking_uri,
        run_id=args.mlflow_run_id,
    )


def _log_tracking(tracked, output: Path, metrics: dict) -> None:
    if tracked is None:
        return
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None:
            kind = "ml" if name in {"normalized_exact_match", "chrf"} else "technical"
            tracked.log_metric(name, float(value), kind=kind, step=0)
    for name in ("manifest.json", "tracking_manifest.json", "outputs.jsonl",
                 "metrics.json"):
        path = output / name
        if path.exists():
            tracked.log_artifact(path, artifact_path="evaluation")


def _run_evaluation(args: argparse.Namespace, results_path: Path, generator):
    started_all = time.monotonic()
    automatic: list[tuple[str, str]] = []
    latencies: list[float] = []
    completed = 0
    stop_reason = "sample_limit_or_input_end"
    template_options: dict | None = None
    with results_path.open("w", encoding="utf-8") as handle:
        for line_number, record in _records(args.questions, args.max_samples):
            if time.monotonic() - started_all >= args.walltime_minutes * 60:
                stop_reason = "walltime_limit"
                break
            messages = _messages(record, line_number)
            record_id = str(record.get("id", f"line-{line_number}"))
            sample_started = time.monotonic()
            if args.backend == "mock":
                prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                generated = _mock_generate(record_id, prompt, args.max_tokens)
                options = {"mock_template": "canonical JSON messages"}
            else:
                generated, prompt, options = generator(messages, args.max_tokens)
            latency = time.monotonic() - sample_started
            latencies.append(latency)
            template_options = options
            reference = record.get("reference")
            if isinstance(reference, str):
                automatic.append((generated["response"], reference))
            row = {
                "id": record_id,
                "source_line": line_number,
                "messages_sha256": canonical_hash(messages),
                "templated_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": generated.pop("response"),
                "reference_present": isinstance(reference, str),
                "latency_seconds": latency,
                **generated,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed += 1

    exact = sum(normalized_text(hyp) == normalized_text(ref)
                for hyp, ref in automatic)
    metrics: dict[str, Any] = {
        "completed": completed, "references": len(automatic),
        "normalized_exact_match": exact / len(automatic) if automatic else None,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "wall_seconds": time.monotonic() - started_all,
        "stop_reason": stop_reason,
        "metric_scope": "automatic only; not a human quality score",
    }
    if automatic:
        try:
            import sacrebleu

            metrics["chrf"] = sacrebleu.corpus_chrf(
                [hyp for hyp, _ in automatic],
                [[ref for _, ref in automatic]],
            ).score
            metrics["chrf_signature"] = "sacrebleu.corpus_chrf defaults"
        except ImportError:
            metrics["chrf"] = None
    return metrics, template_options


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "outputs.jsonl"
    if results_path.exists():
        raise SystemExit(f"refusing to overwrite {results_path}")

    model_path = Path(args.model).expanduser()
    if args.backend != "mock" and not model_path.exists():
        raise SystemExit("evaluation requires a local exact model snapshot path")
    if args.adapter and not args.adapter.exists():
        raise SystemExit(f"adapter does not exist: {args.adapter}")
    identity = model_identity(model_path if model_path.exists() else args.model,
                              args.model_revision)
    if args.backend == "mock":
        runtime_model, template_identity = model_path, {
            "source": "explicit mock", "sha256": canonical_hash("mock")
        }
    else:
        runtime_model, template_identity = model_with_chat_template(
            model_path, output / "model-overlay", args.chat_template
        )
    adapter_identity = (
        model_identity(args.adapter, None) if args.adapter and args.adapter.is_dir()
        else ({"path": str(args.adapter), "sha256": sha256_file(args.adapter)}
              if args.adapter else None)
    )
    decoding = {
        "algorithm": "greedy_argmax", "temperature": 0.0,
        "max_tokens": args.max_tokens, "seed": args.seed,
        "thinking_requested": False, "draft_model": None,
    }
    manifest = {
        "label": args.label, "backend": args.backend,
        "evidence_kind": "TECHNICAL_MOCK" if args.backend == "mock" else "OBSERVED_LOCAL_RUN",
        "model": identity, "adapter": adapter_identity,
        "chat_template": template_identity,
        "questions": {"path": str(args.questions.resolve()),
                      "sha256": sha256_file(args.questions)},
        "decoding": decoding, "max_samples": args.max_samples,
        "walltime_seconds": args.walltime_minutes * 60,
        "code": git_identity(ROOT),
        "versions": package_versions(("mlx", "mlx-lm", "torch", "transformers", "peft")),
        "claim_scope": "automatic metrics and raw generations; no human Mooré score",
    }
    manifest["evaluation_identity_sha256"] = canonical_hash({
        "model": identity, "adapter": adapter_identity,
        "chat_template": template_identity,
        "questions_sha256": manifest["questions"]["sha256"],
        "decoding": decoding,
    })
    write_json(output / "manifest.json", manifest)

    if args.backend == "mlx":
        generator = _mlx_generator(runtime_model, args.adapter, args.seed)
    elif args.backend == "mps":
        generator = _mps_generator(runtime_model, args.adapter, args.seed)
    else:
        generator = None

    with _tracking_context(args, manifest, output) as tracked:
        metrics, template_options = _run_evaluation(args, results_path, generator)
        write_json(output / "metrics.json", metrics)
        manifest["chat_template_options_used"] = template_options
        manifest["results_sha256"] = sha256_file(results_path)
        if tracked is not None:
            manifest["mlflow_run_id"] = tracked.run_id
        write_json(output / "manifest.json", manifest)
        _log_tracking(tracked, output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
