from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from yaaba.experiment_tracking import (
    MANIFEST_PARAM,
    ManifestMismatch,
    build_manifest,
    redact_secrets,
    sha256_path,
    tracked_run,
    verify_manifest,
    write_manifest,
)


def materials(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, content in {
        "config.json": '{"batch": 1, "api_key": "do-not-log"}',
        "data.jsonl": '{"text": "synthetic technical fixture"}\n',
        "runner.py": "print('technical fixture')\n",
        "model.txt": "not model weights\n",
        "tokenizer.json": '{"type": "technical fixture"}\n',
    }.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path
    return paths


def manifest_for(tmp_path: Path) -> dict:
    paths = materials(tmp_path)
    return build_manifest(
        run_kind="technical-smoke",
        evidence_kind="OBSERVED",
        config=paths["config.json"],
        data=paths["data.jsonl"],
        code=paths["runner.py"],
        model=paths["model.txt"],
        tokenizer=paths["tokenizer.json"],
        seed=17,
        budget={"max_steps": 1},
        stop_criteria={"technical_smoke_only": True},
    )


def test_sha256_path_tracks_file_names_and_content(tmp_path):
    folder = tmp_path / "tree"
    folder.mkdir()
    (folder / "a").write_text("one", encoding="utf-8")
    first = sha256_path(folder)
    assert first == sha256_path(folder)
    (folder / "a").write_text("two", encoding="utf-8")
    assert sha256_path(folder) != first


def test_symlink_manifest_tracks_resolved_blob_content(tmp_path):
    paths = materials(tmp_path / "inputs")
    target = tmp_path / "weights-v1.safetensors"
    target.write_bytes(b"first synthetic blob")
    overlay = tmp_path / "model.safetensors"
    overlay.symlink_to(target)

    common = dict(
        run_kind="technical-smoke",
        evidence_kind="OBSERVED",
        config=paths["config.json"],
        data=paths["data.jsonl"],
        code=paths["runner.py"],
        tokenizer=paths["tokenizer.json"],
    )
    first = build_manifest(model=overlay, **common)
    assert first["model"]["hash_scope"] == "resolved_file_content"
    assert first["model"]["sha256"] == sha256_path(target)

    target.write_bytes(b"second synthetic blob")
    second = build_manifest(model=overlay, **common)
    assert second["model"]["sha256"] == sha256_path(target)
    assert second["model"]["sha256"] != first["model"]["sha256"]
    assert second["manifest_sha256"] != first["manifest_sha256"]


def test_manifest_is_deterministic_immutable_and_redacted(tmp_path):
    manifest = manifest_for(tmp_path)
    assert verify_manifest(manifest) == manifest["manifest_sha256"]
    assert manifest["config"]["resolved"]["api_key"] == "__REDACTED__"

    destination = tmp_path / "run" / "manifest.json"
    write_manifest(manifest, destination)
    write_manifest(manifest, destination)
    changed = dict(manifest)
    changed["run_kind"] = "changed"
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_manifest(changed, destination)


def test_secret_detection_does_not_hide_token_counts_or_tokenizer():
    safe = redact_secrets(
        {
            "token": "hidden",
            "max_tokens": 128,
            "tokenizer": "Qwen",
            "nested": {"service_password": "hidden"},
        }
    )
    assert safe == {
        "token": "__REDACTED__",
        "max_tokens": 128,
        "tokenizer": "Qwen",
        "nested": {"service_password": "__REDACTED__"},
    }


def test_remote_identity_requires_an_explicit_mapping(tmp_path):
    paths = materials(tmp_path)
    common = dict(
        run_kind="evaluation",
        evidence_kind="HISTORICAL",
        config=paths["config.json"],
        data=paths["data.jsonl"],
        code=paths["runner.py"],
        tokenizer=paths["tokenizer.json"],
    )
    with pytest.raises(ValueError, match="not a local path"):
        build_manifest(model="org/remote-model", **common)
    manifest = build_manifest(
        model={"uri": "org/remote-model", "sha256": "a" * 64}, **common
    )
    assert manifest["model"]["hash_scope"] == "declared_content"


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    # Set YAABA_TEST_MLFLOW_URI to exercise the exact same tests through the
    # HTTP server.  SQLite matches the persistent backend used by the launcher.
    monkeypatch.chdir(tmp_path)
    database = (tmp_path / "mlflow.db").resolve()
    return os.environ.get("YAABA_TEST_MLFLOW_URI", f"sqlite:///{database}")


def test_run_lifecycle_metrics_artifacts_and_resume(tmp_path, tracking_uri):
    from mlflow import MlflowClient

    manifest = manifest_for(tmp_path / "inputs")
    artifact = tmp_path / "result.json"
    artifact.write_text('{"technical": true}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text('{"step": 1}\n', encoding="utf-8")

    with tracked_run(
        manifest,
        experiment_name="yaaba-technical-tests",
        run_name="cpu-smoke",
        tracking_uri=tracking_uri,
    ) as run:
        run_id = run.run_id
        assert run.client.get_run(run_id).info.status == "RUNNING"
        assert run.log_metric("cpu/seconds", 0.01, kind="technical", step=1) == (
            "technical/cpu/seconds"
        )
        run.log_metric("heldout/loss", 1.25, kind="ml", step=1)
        record = run.log_artifact(artifact, metadata={"api_key": "hidden"})
        point = run.log_checkpoint(checkpoint, step=1, metadata={"resumable": True})
        assert record.sha256 == sha256_path(artifact)
        assert point.step == 1

    client = MlflowClient(tracking_uri=tracking_uri)
    completed = client.get_run(run_id)
    assert completed.info.status == "FINISHED"
    assert completed.data.params[MANIFEST_PARAM] == manifest["manifest_sha256"]
    assert completed.data.metrics["technical/cpu/seconds"] == 0.01
    assert completed.data.metrics["ml/heldout/loss"] == 1.25
    assert completed.data.tags["yaaba.latest_checkpoint_step"] == "1"

    with pytest.raises(RuntimeError, match="intentional technical failure"):
        with tracked_run(
            manifest,
            experiment_name="yaaba-technical-tests",
            run_name="ignored-on-resume",
            tracking_uri=tracking_uri,
            run_id=run_id,
        ) as resumed:
            assert resumed.client.get_run(run_id).info.status == "RUNNING"
            raise RuntimeError("intentional technical failure")

    failed = client.get_run(run_id)
    assert failed.info.status == "FAILED"
    assert failed.data.tags["yaaba.failure_type"] == "RuntimeError"
    assert failed.data.tags["yaaba.resume_count"] == "1"


def test_resume_rejects_a_different_manifest_without_changing_status(
    tmp_path, tracking_uri
):
    from mlflow import MlflowClient

    manifest = manifest_for(tmp_path / "first")
    with tracked_run(
        manifest,
        experiment_name="yaaba-manifest-tests",
        run_name="immutable",
        tracking_uri=tracking_uri,
    ) as run:
        run_id = run.run_id

    other = manifest_for(tmp_path / "second")
    with pytest.raises(ManifestMismatch):
        with tracked_run(
            other,
            experiment_name="yaaba-manifest-tests",
            run_name="immutable",
            tracking_uri=tracking_uri,
            run_id=run_id,
        ):
            pass
    assert MlflowClient(tracking_uri=tracking_uri).get_run(run_id).info.status == (
        "FINISHED"
    )


def test_server_command_is_local_and_uses_explicit_storage(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "local" / "start_mlflow.py"
    spec = importlib.util.spec_from_file_location("start_mlflow", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.server_command(tmp_path / "state", 5001)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "5001"
    assert "sqlite:///" in command[command.index("--backend-store-uri") + 1]
    assert command[command.index("--artifacts-destination") + 1].startswith("file://")
    assert "0.0.0.0" not in command
