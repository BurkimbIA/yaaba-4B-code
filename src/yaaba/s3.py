"""Checkpoint persistence, driven by the standard AWS environment variables.

The same code runs on Colab, RunPod or a laptop:

    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL_S3
    S3_CHECKPOINT_URI   where checkpoints go; unset means every call is a no-op
    REQUIRE_S3_SYNC     "true" turns an upload failure into an error

Uploads are best effort by default, so a network error does not end a long
run. A checkpoint is uploaded whole, optimizer state included, because resuming
needs it and the local disk is usually gone by then.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Final

# Large, and only needed on the most recent checkpoint, so it is pruned from
# older ones.
RESUME_STATE: Final = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt")

# The Trainer cannot resume without this file. It is small and never pruned; a
# checkpoint without it starts a fresh run instead of crashing after the GPU is
# allocated.
REQUIRED_TO_RESUME: Final = "trainer_state.json"

DEFAULT_ENDPOINT: Final = "https://t3.storage.dev"


def _client():
    import boto3

    return boto3.client("s3",
                        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", DEFAULT_ENDPOINT))


def _split(uri: str) -> tuple[str, str]:
    """`s3://bucket/some/prefix` -> `("bucket", "some/prefix")`."""
    bucket, _, prefix = uri.removeprefix("s3://").strip("/").partition("/")
    return bucket, prefix


def sync_required() -> bool:
    return os.getenv("REQUIRE_S3_SYNC", "").lower() in {"1", "true", "yes"}


def _fail(message: str, error: Exception) -> None:
    if sync_required():
        raise RuntimeError(message) from error
    print(f"  WARNING: {message}", file=sys.stderr, flush=True)


def writable(uri: str) -> str | None:
    """None if `uri` accepts a write, else the reason it does not.

    A round trip of a few bytes, called once before a run starts. Uploads only
    warn on failure (see `_fail`), which suits a short network error but not
    credentials that never worked: the run would train for hours and save
    nothing.
    """
    if not uri:
        return None
    bucket, prefix = _split(uri)
    key = f"{prefix}/.ecriture-verifiee"
    try:
        client = _client()
        client.put_object(Bucket=bucket, Key=key, Body=b"ok")
        client.get_object(Bucket=bucket, Key=key)["Body"].read()
        client.delete_object(Bucket=bucket, Key=key)
        return None
    except Exception as error:  # noqa: BLE001
        # boto3 puts the key id in some errors, so only the error class and a
        # truncated message are returned.
        return f"{type(error).__name__}: {error}"[:300]


def upload_file(path: Path, uri: str) -> bool:
    """Upload a single file. Used for the training log, which is small and hot."""
    if not uri or not path.is_file():
        return False
    bucket, key = _split(uri)
    try:
        _client().upload_file(str(path), bucket, key)
        return True
    except Exception as error:  # noqa: BLE001 - persistence must not end a run
        _fail(f"upload failed ({path} -> {uri}): {error}", error)
        return False


def upload_dir(directory: Path, uri: str, quiet: bool = False) -> int:
    """Upload a directory tree. Returns the number of files sent."""
    if not uri or not directory.is_dir():
        return 0
    bucket, prefix = _split(uri)
    try:
        client = _client()
        count = 0
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                rel = path.relative_to(directory).as_posix()
                client.upload_file(str(path), bucket, f"{prefix}/{rel}")
                count += 1
        if not quiet:
            print(f"  -> {uri}  ({count} files)", flush=True)
        return count
    except Exception as error:  # noqa: BLE001
        _fail(f"upload failed ({directory} -> {uri}): {error}", error)
        return 0


def download_file(uri: str, path: Path) -> bool:
    """Download one object. `download_dir` lists a prefix, which finds nothing for a
    single key and yields an empty directory.
    """
    if not uri:
        return False
    bucket, key = _split(uri)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(bucket, key, str(path))
        return True
    except Exception as error:  # noqa: BLE001 - absence is a valid answer
        _fail(f"download failed ({uri} -> {path}): {error}", error)
        return False


def download_dir(uri: str, directory: Path) -> int:
    """Download a prefix into `directory`. Used to resume a run."""
    if not uri:
        return 0
    bucket, prefix = _split(uri)
    client = _client()
    directory.mkdir(parents=True, exist_ok=True)
    count, token = 0, None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{prefix}/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) + 1:]
            if not rel:
                continue
            target = directory / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(target))
            count += 1
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    print(f"  <- {uri}  ({count} files)", flush=True)
    return count


def prune_resume_state(uri: str) -> int:
    """Drop the resume state of a checkpoint, keeping its adapter."""
    return _delete(uri, RESUME_STATE)


def delete_checkpoint(uri: str) -> int:
    """Drop a checkpoint entirely. Used for superseded intermediate saves."""
    if not uri:
        return 0
    bucket, prefix = _split(uri)
    try:
        client = _client()
        keys = [{"Key": o["Key"]} for o in
                client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/")
                .get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        return len(keys)
    except Exception as error:  # noqa: BLE001
        _fail(f"delete failed ({uri}): {error}", error)
        return 0


def _delete(uri: str, names: tuple[str, ...]) -> int:
    if not uri:
        return 0
    bucket, prefix = _split(uri)
    try:
        client = _client()
        count = 0
        for name in names:
            try:
                client.delete_object(Bucket=bucket, Key=f"{prefix}/{name}")
                count += 1
            except Exception:  # noqa: BLE001 - absent keys are fine
                continue
        return count
    except Exception as error:  # noqa: BLE001
        _fail(f"prune failed ({uri}): {error}", error)
        return 0


def read_bytes(uri: str) -> bytes | None:
    """One object, in memory. `None` when it is not there.

    Only for small objects: a card, a pointer. Use `digest` on anything whose
    size you do not control.
    """
    if not uri:
        return None
    bucket, key = _split(uri)
    try:
        return _client().get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as error:  # noqa: BLE001 - absence is a valid answer
        _fail(f"read failed ({uri}): {error}", error)
        return None


def _digest(stream) -> str:
    """sha256 of a stream, one megabyte at a time.

    Objects are never read whole: a 53 MB parquet and its local copy would take
    106 MB per comparison.
    """
    running = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        running.update(block)
    return running.hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return _digest(handle)


def matches(uri: str, path: Path) -> bool | None:
    """`True` identical, `False` different, `None` when S3 has nothing there.

    Absent and different call for different actions (upload again, or find
    what went wrong), so they are kept apart.

    The object is read back and hashed because Tigris serves a per-region
    cache: a key rewritten from one machine can still reach a reader elsewhere
    as the old object.
    """
    if not uri or not path.is_file():
        return None
    bucket, key = _split(uri)
    try:
        served = _client().get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as error:  # noqa: BLE001 - absence is a valid answer
        _fail(f"read failed ({uri}): {error}", error)
        return None
    return _digest(served) == digest(path)


def run_names(uri: str) -> list[str]:
    """The run directories directly under `uri`, alphabetically."""
    if not uri:
        return []
    bucket, prefix = _split(uri)
    try:
        page = _client().list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/",
                                         Delimiter="/")
        return sorted(entry["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                      for entry in page.get("CommonPrefixes", []))
    except Exception as error:  # noqa: BLE001
        _fail(f"listing failed ({uri}): {error}", error)
        return []


def checkpoint_names(uri: str) -> list[tuple[str, int]]:
    """Every `checkpoint-N` under `uri`, in training order.

    Sorted by number: in string order `checkpoint-542` comes after
    `checkpoint-2168`, which would resume from the wrong epoch.
    """
    if not uri:
        return []
    bucket, prefix = _split(uri)
    try:
        page = _client().list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/",
                                         Delimiter="/")
        steps = []
        for entry in page.get("CommonPrefixes", []):
            name = entry["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            _, _, digits = name.partition("checkpoint-")
            if name.startswith("checkpoint-") and digits.isdigit():
                steps.append((name, int(digits)))
        return sorted(steps, key=lambda pair: pair[1])
    except Exception as error:  # noqa: BLE001
        _fail(f"listing failed ({uri}): {error}", error)
        return []


def latest_checkpoint(uri: str) -> tuple[str, int] | None:
    """Most advanced `checkpoint-N` under `uri`, to resume from."""
    found = checkpoint_names(uri)
    return found[-1] if found else None


def list_runs(uri: str, prefix: str) -> list[str]:
    """Run directories under `uri` starting with `prefix`, oldest first.

    Run names are timestamped, so string order is chronological for them. It is
    not for checkpoint names (see `checkpoint_names`).
    """
    if not uri:
        return []
    bucket, base = _split(uri)
    try:
        page = _client().list_objects_v2(Bucket=bucket, Prefix=f"{base}/{prefix}",
                                         Delimiter="/")
        return sorted(entry["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                      for entry in page.get("CommonPrefixes", []))
    except Exception as error:  # noqa: BLE001
        _fail(f"listing failed ({uri}): {error}", error)
        return []
