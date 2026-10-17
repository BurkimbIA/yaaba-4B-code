"""Start the local-only MLflow server in the foreground."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 5001


def server_command(state_dir: Path, port: int) -> list[str]:
    state = state_dir.expanduser().resolve()
    database = state / "mlflow.db"
    artifacts = state / "artifacts"
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        HOST,
        "--port",
        str(port),
        "--backend-store-uri",
        f"sqlite:///{database.as_posix()}",
        "--artifacts-destination",
        artifacts.as_uri(),
        "--serve-artifacts",
        "--workers",
        "1",
    ]


def port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run persistent local MLflow storage on 127.0.0.1 only."
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        type=Path,
        help="persistent directory that will hold mlflow.db and artifacts/",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the argv as JSON and exit"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    state = args.state_dir.expanduser().resolve()
    artifacts = state / "artifacts"
    state.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        state.chmod(0o700)
        artifacts.chmod(0o700)
    except OSError:
        pass

    command = server_command(state, args.port)
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False))
        return 0
    if not port_is_available(args.port):
        parser.error(f"{HOST}:{args.port} is already in use")

    print(f"MLflow metadata: {state / 'mlflow.db'}", flush=True)
    print(f"MLflow artifacts: {artifacts}", flush=True)
    print(f"MLflow URI: http://{HOST}:{args.port}", flush=True)
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
