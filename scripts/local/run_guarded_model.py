#!/usr/bin/env python3
"""Run one local model process under memory, time, health and exclusivity guards."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import psutil


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observations", type=Path, required=True)
    result.add_argument("--lock-file", type=Path, required=True)
    result.add_argument("--status-file", type=Path, required=True)
    result.add_argument("--owner", required=True)
    result.add_argument("--health-url", default="http://127.0.0.1:8765/healthz")
    result.add_argument("--max-rss-gib", type=_positive_float, default=32)
    result.add_argument("--minimum-available-gib", type=_positive_float, default=4)
    result.add_argument("--walltime-seconds", type=_positive_float, default=1800)
    result.add_argument("--poll-seconds", type=_positive_float, default=0.5)
    result.add_argument("--termination-grace-seconds", type=_positive_float, default=10)
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _health(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 local URL is explicit
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        payload = json.load(response)
    if payload.get("status") != "ok":
        raise RuntimeError(f"health status is not ok: {payload.get('status')!r}")
    if payload.get("service") != "burkimbia-mac":
        raise RuntimeError(f"unexpected health service: {payload.get('service')!r}")
    if type(payload.get("busy")) is not bool:  # bool is required, not 0/1
        raise RuntimeError("health busy field must be a boolean")
    if payload["busy"] is not False:
        raise RuntimeError(f"health endpoint is not idle: busy={payload.get('busy')!r}")
    if payload.get("loaded_services") != ["translation"]:
        raise RuntimeError(
            "health endpoint must expose only the translation context: "
            f"{payload.get('loaded_services')!r}"
        )
    return payload


def _process_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(
            process.memory_info().rss for process in processes if process.is_running()
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can transiently return EPERM while an orphaned process group is
        # being reaped. Treat that as indeterminate/present and keep waiting.
        return True
    return True


def _wait_group_gone(process_group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_exists(process_group_id):
            return True
        time.sleep(min(0.01, timeout))
    return not _group_exists(process_group_id)


def _terminate_group(process: subprocess.Popen, grace_seconds: float) -> bool:
    process_group_id = process.pid
    if not _group_exists(process_group_id):
        process.wait()
        return False
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        # The group may disappear during the grace interval. If it remains,
        # the post-SIGKILL existence check below fails closed.
        pass
    if not _wait_group_gone(process_group_id, grace_seconds):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        if not _wait_group_gone(process_group_id, grace_seconds):
            raise RuntimeError(f"process group {process_group_id} survived SIGKILL")
    process.wait()
    return True


def run(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing guarded model command")

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock_file.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        print(json.dumps({"guard": {"stop_reason": "exclusive_lock_busy"}}))
        return 72

    started_wall = time.time()
    started = time.monotonic()
    status = {
        "active": True,
        "guard_pid": os.getpid(),
        "owner": args.owner,
        "started_unix": started_wall,
    }
    _write_json(args.status_file, status)
    process: subprocess.Popen | None = None
    max_rss = 0
    min_available = 1 << 62
    reason = None
    returncode = 70
    try:
        _health(args.health_url)
        available_before_spawn = psutil.virtual_memory().available
        if available_before_spawn < args.minimum_available_gib * 1024**3:
            reason = "minimum_available_limit_preflight"
            raise RuntimeError(
                "available memory is below the preflight minimum: "
                f"{available_before_spawn} bytes"
            )
        process = subprocess.Popen(command, start_new_session=True)
        args.observations.parent.mkdir(parents=True, exist_ok=True)
        with args.observations.open("w", encoding="utf-8") as handle:
            while process.poll() is None:
                rss = _process_rss(process.pid)
                available = psutil.virtual_memory().available
                elapsed = time.monotonic() - started
                max_rss = max(max_rss, rss)
                min_available = min(min_available, available)
                try:
                    health = _health(args.health_url)
                    health_error = None
                except Exception as error:  # health failure is a stop condition
                    health = None
                    health_error = f"{type(error).__name__}: {error}"
                observation = {
                    "available_bytes": available,
                    "elapsed_seconds": elapsed,
                    "health": health,
                    "health_error": health_error,
                    "rss_bytes": rss,
                }
                handle.write(json.dumps(observation, sort_keys=True) + "\n")
                handle.flush()
                if health_error:
                    reason = "health_endpoint_not_idle_or_unhealthy"
                elif rss > args.max_rss_gib * 1024**3:
                    reason = "process_rss_limit"
                elif available < args.minimum_available_gib * 1024**3:
                    reason = "minimum_available_limit"
                elif elapsed > args.walltime_seconds:
                    reason = "walltime_limit"
                if reason:
                    _terminate_group(process, args.termination_grace_seconds)
                    break
                time.sleep(args.poll_seconds)
        returncode = process.wait()
    except Exception as error:
        reason = reason or f"preflight_or_guard_error:{type(error).__name__}"
        if process is not None and process.poll() is None:
            _terminate_group(process, args.termination_grace_seconds)
        print(f"guard error: {error}", file=sys.stderr)
    finally:
        if process is not None:
            try:
                cleaned_leftover = _terminate_group(
                    process, args.termination_grace_seconds
                )
                if cleaned_leftover and reason is None:
                    reason = "descendant_process_group_cleanup"
                    returncode = 70
            except Exception as error:
                reason = "process_group_cleanup_failed"
                returncode = 70
                print(f"guard cleanup error: {error}", file=sys.stderr)
        summary = {
            "active": False,
            "guard_pid": os.getpid(),
            "max_rss_gib": max_rss / 1024**3,
            "min_available_gib": (
                min_available / 1024**3 if min_available != 1 << 62 else None
            ),
            "owner": args.owner,
            "returncode": returncode,
            "stop_reason": reason or "process_exit",
            "wall_seconds": time.monotonic() - started,
        }
        _write_json(args.observations.with_suffix(".summary.json"), summary)
        _write_json(args.status_file, summary)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
        print(json.dumps({"guard": summary}), flush=True)
    if reason:
        return 70
    return returncode


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
