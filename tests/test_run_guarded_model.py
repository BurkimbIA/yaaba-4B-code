from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "local"
sys.path.insert(0, str(SCRIPTS))

import run_guarded_model  # noqa: E402

HEALTHY = {
    "status": "ok",
    "service": "burkimbia-mac",
    "busy": False,
    "loaded_services": ["translation"],
}


def _args(tmp_path: Path, command: list[str]) -> list[str]:
    return [
        "--observations",
        str(tmp_path / "observations.jsonl"),
        "--lock-file",
        str(tmp_path / "heavy.lock"),
        "--status-file",
        str(tmp_path / "status.json"),
        "--owner",
        "unit-test",
        "--poll-seconds",
        "0.01",
        "--termination-grace-seconds",
        "0.5",
        "--walltime-seconds",
        "5",
        "--",
        *command,
    ]


def test_guard_requires_idle_health_continuously(tmp_path, monkeypatch):
    calls = 0

    def health(_url):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("busy=true")
        return HEALTHY

    monkeypatch.setattr(run_guarded_model, "_health", health)
    result = run_guarded_model.main(
        _args(
            tmp_path,
            [sys.executable, "-c", "import time; time.sleep(2)"],
        )
    )
    assert result == 70
    summary = json.loads((tmp_path / "observations.summary.json").read_text())
    assert summary["stop_reason"] == "health_endpoint_not_idle_or_unhealthy"
    assert json.loads((tmp_path / "status.json").read_text())["active"] is False


def test_guard_refuses_an_exclusive_lock_already_held(tmp_path, monkeypatch):
    monkeypatch.setattr(run_guarded_model, "_health", lambda _url: HEALTHY)
    lock = (tmp_path / "heavy.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = run_guarded_model.main(
            _args(
                tmp_path,
                [sys.executable, "-c", "raise SystemExit(0)"],
            )
        )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert result == 72
    assert not (tmp_path / "status.json").exists()


def test_guard_records_success_and_releases_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(run_guarded_model, "_health", lambda _url: HEALTHY)
    result = run_guarded_model.main(
        _args(
            tmp_path,
            [sys.executable, "-c", "raise SystemExit(0)"],
        )
    )
    assert result == 0
    summary = json.loads((tmp_path / "status.json").read_text())
    assert summary["active"] is False
    assert summary["stop_reason"] == "process_exit"


def test_health_payload_must_be_idle(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'{"status":"ok","service":"burkimbia-mac",'
                b'"busy":true,"loaded_services":["translation"]}'
            )

    monkeypatch.setattr(
        run_guarded_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    with pytest.raises(RuntimeError, match="not idle"):
        run_guarded_model._health("http://127.0.0.1:8765/healthz")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_positive_float_rejects_non_finite_and_non_positive(value):
    with pytest.raises(run_guarded_model.argparse.ArgumentTypeError):
        run_guarded_model._positive_float(value)


def test_guard_checks_available_memory_before_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(run_guarded_model, "_health", lambda _url: HEALTHY)
    monkeypatch.setattr(
        run_guarded_model.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1),
    )
    child_marker = tmp_path / "child-started"
    result = run_guarded_model.main(
        _args(
            tmp_path,
            [sys.executable, "-c", f"open({str(child_marker)!r}, 'w').close()"],
        )
    )
    assert result == 70
    assert not child_marker.exists()
    summary = json.loads((tmp_path / "status.json").read_text())
    assert summary["stop_reason"] == "minimum_available_limit_preflight"


def test_guard_kills_child_that_ignores_term_on_health_exception(tmp_path, monkeypatch):
    calls = 0

    def health(_url):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("health changed")
        return HEALTHY

    monkeypatch.setattr(run_guarded_model, "_health", health)
    child_pid = tmp_path / "child.pid"
    child = (
        "import os,signal,time,pathlib;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    result = run_guarded_model.main(
        _args(
            tmp_path,
            [sys.executable, "-c", child],
        )
    )
    assert result == 70
    pid = int(child_pid.read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"guard left child {pid} alive")


def test_guard_cleans_process_group_after_root_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(run_guarded_model, "_health", lambda _url: HEALTHY)
    child_pid = tmp_path / "orphan-child.pid"
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(30)"
    )
    root_code = (
        "import pathlib,subprocess,sys;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid))"
    )
    result = run_guarded_model.main(
        _args(
            tmp_path,
            [sys.executable, "-c", root_code],
        )
    )
    assert result == 70
    pid = int(child_pid.read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"guard released its lock while descendant {pid} remained")
    summary = json.loads((tmp_path / "status.json").read_text())
    assert summary["stop_reason"] == "descendant_process_group_cleanup"
    assert summary["returncode"] == 70


def test_group_existence_treats_transient_eperm_as_indeterminate(monkeypatch):
    outcomes = [PermissionError(), PermissionError(), ProcessLookupError()]

    def killpg(_process_group_id, _signal):
        raise outcomes.pop(0)

    monkeypatch.setattr(run_guarded_model.os, "killpg", killpg)
    assert run_guarded_model._wait_group_gone(12345, 0.1) is True
    assert outcomes == []


def test_persistent_eperm_marks_cleanup_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_guarded_model, "_health", lambda _url: HEALTHY)
    original_terminate = run_guarded_model._terminate_group

    def persistent_eperm(process, grace_seconds):
        if process.poll() is None:
            return original_terminate(process, grace_seconds)
        raise PermissionError("persistent process-group uncertainty")

    monkeypatch.setattr(run_guarded_model, "_terminate_group", persistent_eperm)
    result = run_guarded_model.main(
        _args(tmp_path, [sys.executable, "-c", "raise SystemExit(0)"])
    )
    assert result == 70
    summary = json.loads((tmp_path / "status.json").read_text())
    assert summary["stop_reason"] == "process_group_cleanup_failed"
    assert summary["returncode"] == 70
