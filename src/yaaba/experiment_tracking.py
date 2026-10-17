"""Reproducible, local-first MLflow tracking for Yaaba experiments.

Manifest construction deliberately has no MLflow dependency.  Training code can
therefore always write its standalone manifest, even when tracking is unavailable.
Only :func:`tracked_run` loads the optional tracking dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
MANIFEST_ARTIFACT = "run-manifest/manifest.json"
MANIFEST_PARAM = "yaaba.manifest_sha256"
SCHEMA_VERSION = 1

EVIDENCE_KINDS = frozenset(
    {"OBSERVED", "HISTORICAL", "DECLARED", "HYPOTHESIS", "PROPOSED"}
)
METRIC_KINDS = frozenset({"technical", "ml"})

_SENSITIVE_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "access_key",
        "auth_token",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_key",
        "token",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_NAME = re.compile(r"^[A-Za-z0-9_. /-]+$")


class TrackingError(RuntimeError):
    """Base error for the Yaaba tracking contract."""


class TrackingUnavailable(TrackingError):
    """Raised when the optional MLflow client cannot be loaded or reached."""


class ManifestMismatch(TrackingError):
    """Raised when a run is resumed with a different immutable manifest."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sensitive_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return normalized in _SENSITIVE_PARTS or any(
        normalized.endswith(f"_{part}") for part in _SENSITIVE_PARTS
    )


def redact_secrets(value: Any) -> Any:
    """Return a JSON-safe copy with values of credential-like keys removed.

    This function never inspects the process environment.  Keys such as
    ``max_tokens`` and ``tokenizer`` are intentionally not treated as secrets.
    """

    if isinstance(value, Mapping):
        return {
            str(key): "__REDACTED__" if _is_sensitive_key(key) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("manifest values must not contain NaN or infinity")
        return value
    raise TypeError(f"manifest value is not JSON-compatible: {type(value).__name__}")


def sha256_path(path: str | os.PathLike[str]) -> str:
    """Hash resolved file content or a resolved directory tree deterministically.

    Symbolic links are followed deliberately: a manifest identity must change
    when the model/tokenizer blob behind a stable overlay path changes.  Cyclic
    directory links are rejected rather than assigned a misleading digest.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    def hash_file(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def hash_resolved_tree(directory: Path, active: set[tuple[int, int]]) -> str:
        resolved = directory.resolve(strict=True)
        identity = resolved.stat()
        inode = (identity.st_dev, identity.st_ino)
        if inode in active:
            raise ValueError(f"cyclic directory symlink: {directory}")
        active.add(inode)
        try:
            digest = hashlib.sha256()
            digest.update(b"directory\0")
            for child in sorted(resolved.iterdir(), key=lambda item: item.name):
                target = child.resolve(strict=True)
                name = child.name.encode("utf-8")
                if target.is_file():
                    digest.update(b"F\0" + name + b"\0")
                    digest.update(hash_file(target).encode("ascii") + b"\0")
                elif target.is_dir():
                    digest.update(b"D\0" + name + b"\0")
                    digest.update(
                        hash_resolved_tree(target, active).encode("ascii") + b"\0"
                    )
                else:
                    raise ValueError(f"unsupported filesystem entry: {child}")
            return digest.hexdigest()
        finally:
            active.remove(inode)

    resolved = source.resolve(strict=True)
    if resolved.is_file():
        return hash_file(resolved)
    if resolved.is_dir():
        return hash_resolved_tree(resolved, set())
    raise ValueError(f"unsupported filesystem entry: {source}")


def _validate_sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a 64-character SHA-256 hexadecimal digest")
    return digest


def _path_identity(path: Path, *, include_config: bool = False) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        hash_scope = "resolved_file_content" if path.is_symlink() else "file_content"
    elif resolved.is_dir():
        hash_scope = (
            "resolved_directory_tree" if path.is_symlink() else "directory_tree"
        )
    else:
        raise ValueError(f"unsupported filesystem entry: {path}")
    identity: dict[str, Any] = {
        "kind": "path",
        "path": str(path),
        "sha256": sha256_path(path),
        "hash_scope": hash_scope,
    }
    if include_config and path.is_file() and path.suffix.lower() == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        identity["resolved"] = redact_secrets(parsed)
    return identity


def _identity(value: Any, field: str, *, include_config: bool = False) -> Any:
    if isinstance(value, os.PathLike):
        return _path_identity(Path(value), include_config=include_config)
    if isinstance(value, str):
        path = Path(value)
        if not path.exists() and not path.is_symlink():
            raise ValueError(
                f"{field} string is not a local path: {value!r}; "
                "use a mapping with an explicit sha256 for a remote resource"
            )
        return _path_identity(path, include_config=include_config)
    if isinstance(value, Mapping):
        safe = redact_secrets(value)
        if "sha256" in safe:
            safe["sha256"] = _validate_sha256(safe["sha256"], f"{field}.sha256")
            safe.setdefault("hash_scope", "declared_content")
        else:
            safe["sha256"] = _sha256_bytes(_canonical_json(safe).encode("utf-8"))
            safe["hash_scope"] = "canonical_metadata"
        return safe
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError(f"{field} must not be empty")
        return [
            _identity(item, f"{field}[{index}]", include_config=include_config)
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field} must be a path, mapping, or non-empty sequence")


def build_manifest(
    *,
    run_kind: str,
    evidence_kind: str,
    config: Any,
    data: Any,
    code: Any,
    model: Any,
    tokenizer: Any,
    seed: int | None = None,
    budget: Mapping[str, Any] | None = None,
    stop_criteria: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest and attach its own SHA-256.

    Local paths are content-hashed.  Remote resources must be mappings with an
    explicit ``sha256``; mappings without one are honestly labelled as hashes of
    canonical metadata rather than hashes of remote content.
    """

    evidence = evidence_kind.upper()
    if evidence not in EVIDENCE_KINDS:
        raise ValueError(f"evidence_kind must be one of {sorted(EVIDENCE_KINDS)}")
    if not run_kind.strip():
        raise ValueError("run_kind must not be empty")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind.strip(),
        "evidence_kind": evidence,
        "config": _identity(config, "config", include_config=True),
        "data": _identity(data, "data"),
        "code": _identity(code, "code"),
        "model": _identity(model, "model"),
        "tokenizer": _identity(tokenizer, "tokenizer"),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if budget is not None:
        payload["budget"] = redact_secrets(budget)
    if stop_criteria is not None:
        payload["stop_criteria"] = redact_secrets(stop_criteria)
    if extra is not None:
        payload["extra"] = redact_secrets(extra)
    payload["manifest_sha256"] = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return payload


def _verify_identity(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        if "sha256" not in value:
            raise ValueError(f"{field} identity has no sha256")
        _validate_sha256(value["sha256"], f"{field}.sha256")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError(f"{field} identity must not be empty")
        for index, item in enumerate(value):
            _verify_identity(item, f"{field}[{index}]")
        return
    raise TypeError(f"{field} identity must be a mapping or non-empty sequence")


def verify_manifest(manifest: Mapping[str, Any]) -> str:
    """Validate the schema and checksum, returning the manifest digest."""

    safe = redact_secrets(manifest)
    required = {
        "schema_version",
        "run_kind",
        "evidence_kind",
        "config",
        "data",
        "code",
        "model",
        "tokenizer",
        "manifest_sha256",
    }
    absent = sorted(required - safe.keys())
    if absent:
        raise ValueError(f"manifest is missing: {', '.join(absent)}")
    if safe["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {safe['schema_version']}")
    if safe["evidence_kind"] not in EVIDENCE_KINDS:
        raise ValueError(f"invalid evidence kind: {safe['evidence_kind']!r}")
    if not isinstance(safe["run_kind"], str) or not safe["run_kind"].strip():
        raise ValueError("manifest run_kind must be a non-empty string")
    for field in ("config", "data", "code", "model", "tokenizer"):
        _verify_identity(safe[field], field)
    expected = _validate_sha256(safe.pop("manifest_sha256"), "manifest_sha256")
    actual = _sha256_bytes(_canonical_json(safe).encode("utf-8"))
    if actual != expected:
        raise ValueError(
            f"manifest checksum mismatch: expected {expected}, got {actual}"
        )
    return expected


def write_manifest(manifest: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a standalone manifest only if it would not replace other content."""

    verify_manifest(manifest)
    destination = Path(path)
    content = _canonical_json(redact_secrets(manifest)) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to replace immutable manifest: {destination}"
            ) from None
    return destination


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    sha256: str
    remote_path: str
    size_bytes: int
    step: int | None = None
    metadata: Mapping[str, Any] | None = None


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _load_client(tracking_uri: str):
    try:
        from mlflow import MlflowClient
    except ImportError as error:
        raise TrackingUnavailable(
            "MLflow is optional and is not installed. Install "
            "requirements-tracking.txt in the experiment virtual environment."
        ) from error
    return MlflowClient(tracking_uri=tracking_uri)


class TrackedRun:
    """Context-managed MLflow run with manifest and metric invariants."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        experiment_name: str,
        run_name: str,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        run_id: str | None = None,
    ) -> None:
        self.manifest = redact_secrets(manifest)
        self.manifest_sha256 = verify_manifest(self.manifest)
        self.experiment_name = experiment_name.strip()
        self.run_name = run_name.strip()
        self.tracking_uri = tracking_uri
        self.requested_run_id = run_id
        self.run_id: str | None = None
        self.client = None
        self._active = False
        if not self.experiment_name or not self.run_name:
            raise ValueError("experiment_name and run_name must not be empty")

    def _experiment_id(self) -> str:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(self.experiment_name)
        except Exception:  # another local process may have created it
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise
            return experiment.experiment_id

    def _manifest_text(self) -> str:
        return _canonical_json(self.manifest) + "\n"

    def _log_new_manifest(self) -> None:
        self.client.log_param(self.run_id, MANIFEST_PARAM, self.manifest_sha256)
        with tempfile.TemporaryDirectory(prefix="yaaba-manifest-") as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(self._manifest_text(), encoding="utf-8")
            self.client.log_artifact(self.run_id, str(path), "run-manifest")

    def _verify_remote_manifest(self, run: Any) -> None:
        recorded = run.data.params.get(MANIFEST_PARAM)
        if recorded != self.manifest_sha256:
            raise ManifestMismatch(
                f"run {self.run_id} has manifest {recorded!r}, "
                f"requested {self.manifest_sha256!r}"
            )
        with tempfile.TemporaryDirectory(prefix="yaaba-manifest-check-") as directory:
            downloaded = Path(
                self.client.download_artifacts(
                    self.run_id, MANIFEST_ARTIFACT, str(Path(directory).resolve())
                )
            )
            if downloaded.read_text(encoding="utf-8") != self._manifest_text():
                raise ManifestMismatch(
                    f"run {self.run_id} manifest artifact differs from its checksum"
                )

    def __enter__(self) -> TrackedRun:
        self.client = _load_client(self.tracking_uri)
        created = False
        status_started = False
        try:
            experiment_id = self._experiment_id()
            if self.requested_run_id:
                run = self.client.get_run(self.requested_run_id)
                self.run_id = run.info.run_id
                if run.info.experiment_id != experiment_id:
                    raise ManifestMismatch(
                        f"run {self.run_id} belongs to experiment "
                        f"{run.info.experiment_id}, not {experiment_id}"
                    )
                self._verify_remote_manifest(run)
                count = int(run.data.tags.get("yaaba.resume_count", "0")) + 1
                self.client.update_run(self.run_id, status="RUNNING")
                status_started = True
                self.client.set_tag(self.run_id, "yaaba.resume_count", str(count))
            else:
                tags = {
                    "yaaba.schema_version": str(SCHEMA_VERSION),
                    "yaaba.evidence_kind": self.manifest["evidence_kind"],
                    "yaaba.run_kind": self.manifest["run_kind"],
                    "yaaba.metric_contract": "technical-or-ml",
                    "yaaba.resume_count": "0",
                }
                run = self.client.create_run(
                    experiment_id, tags=tags, run_name=self.run_name
                )
                self.run_id = run.info.run_id
                created = True
                self._log_new_manifest()
            self._active = True
            return self
        except Exception:
            if (created or status_started) and self.run_id:
                try:
                    self.client.set_terminated(self.run_id, status="FAILED")
                except Exception:
                    pass
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._active:
            return False
        try:
            if exc_type is None:
                self.client.set_terminated(self.run_id, status="FINISHED")
            else:
                # The class name is useful and does not risk copying credentials
                # embedded in an exception message into MLflow.
                try:
                    self.client.set_terminated(self.run_id, status="FAILED")
                except Exception:
                    # Preserve the experiment exception if the tracking server
                    # itself is unavailable during cleanup.
                    pass
                try:
                    self.client.set_tag(
                        self.run_id, "yaaba.failure_type", exc_type.__name__
                    )
                except Exception:
                    pass
        finally:
            self._active = False
        return False

    def _require_active(self) -> None:
        if not self._active or not self.run_id:
            raise TrackingError(
                "tracking operations require an active tracked_run context"
            )

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        kind: str,
        step: int = 0,
        timestamp_ms: int | None = None,
    ) -> str:
        """Log one metric under ``technical/`` or ``ml/``."""

        self._require_active()
        if kind not in METRIC_KINDS:
            raise ValueError(f"metric kind must be one of {sorted(METRIC_KINDS)}")
        clean_name = name.strip().strip("/")
        if not clean_name or not _METRIC_NAME.fullmatch(clean_name):
            raise ValueError(f"invalid metric name: {name!r}")
        if clean_name.startswith("technical/") or clean_name.startswith("ml/"):
            raise ValueError("metric name must not include its kind prefix")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("metric value must be finite")
        if step < 0:
            raise ValueError("metric step must be non-negative")
        key = f"{kind}/{clean_name}"
        self.client.log_metric(
            self.run_id,
            key,
            numeric,
            timestamp=timestamp_ms or int(time.time() * 1000),
            step=int(step),
            synchronous=True,
        )
        return key

    def log_metrics(
        self,
        values: Mapping[str, float],
        *,
        kind: str,
        step: int = 0,
        timestamp_ms: int | None = None,
    ) -> list[str]:
        timestamp = timestamp_ms or int(time.time() * 1000)
        return [
            self.log_metric(name, value, kind=kind, step=step, timestamp_ms=timestamp)
            for name, value in values.items()
        ]

    def _log_path(
        self,
        path: str | os.PathLike[str],
        *,
        artifact_path: str,
        kind: str,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        self._require_active()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        digest = sha256_path(source)
        category = artifact_path.strip().strip("/")
        if not category or ".." in Path(category).parts:
            raise ValueError(f"invalid artifact path: {artifact_path!r}")
        destination = f"{category}/{digest}/{source.name}"
        if source.is_dir():
            self.client.log_artifacts(self.run_id, str(source), destination)
        else:
            self.client.log_artifact(self.run_id, str(source), f"{category}/{digest}")
        record = ArtifactRecord(
            kind=kind,
            sha256=digest,
            remote_path=destination,
            size_bytes=_tree_size(source),
            step=step,
            metadata=redact_secrets(metadata) if metadata is not None else None,
        )
        suffix = f"-step-{step}" if step is not None else ""
        index_name = f"artifact-index/{kind}{suffix}-{digest}.json"
        self.client.log_text(
            self.run_id, _canonical_json(asdict(record)) + "\n", index_name
        )
        return record

    def log_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        artifact_path: str = "artifacts",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        return self._log_path(
            path,
            artifact_path=artifact_path,
            kind="artifact",
            metadata=metadata,
        )

    def log_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        step: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if step < 0:
            raise ValueError("checkpoint step must be non-negative")
        record = self._log_path(
            path,
            artifact_path=f"checkpoints/step-{step}",
            kind="checkpoint",
            step=step,
            metadata=metadata,
        )
        run = self.client.get_run(self.run_id)
        prior = int(run.data.tags.get("yaaba.latest_checkpoint_step", "-1"))
        if step >= prior:
            self.client.set_tag(self.run_id, "yaaba.latest_checkpoint_step", str(step))
        return record


def tracked_run(
    manifest: Mapping[str, Any],
    *,
    experiment_name: str,
    run_name: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    run_id: str | None = None,
) -> TrackedRun:
    """Create a context-managed run; pass ``run_id`` for verified resumption."""

    return TrackedRun(
        manifest,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        run_id=run_id,
    )
