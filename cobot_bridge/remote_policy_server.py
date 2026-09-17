"""Deploy and run the persistent SmolVLA policy server on the robot GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess


SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=5",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--local-workspace", type=Path, required=True)
    parser.add_argument("--remote-workspace", type=PurePosixPath, required=True)
    parser.add_argument("--local-policy-path", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--remote-python", type=PurePosixPath, required=True)
    parser.add_argument("--bind", default="tcp://0.0.0.0:5560")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cameras", nargs="+", default=("front", "right"))
    return parser


def _validated_policy_id(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("policy id must be a relative path inside the checkpoint root")
    return path


def _ssh_command(host: str, remote_command: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, host, remote_command]


def _rsync_ssh() -> str:
    return shlex.join(["ssh", *SSH_OPTIONS])


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _vlm_metadata_path(policy_path: Path) -> Path:
    config = json.loads((policy_path / "config.json").read_text())
    model_id = str(config.get("vlm_model_name", ""))
    direct = Path(model_id).expanduser()
    if direct.is_dir():
        return direct.resolve()
    if not model_id:
        raise ValueError("checkpoint config does not define vlm_model_name")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, local_files_only=True)).resolve()


def main() -> None:
    args = build_parser().parse_args()
    workspace = args.local_workspace.resolve()
    local_policy = args.local_policy_path.resolve()
    policy_id = _validated_policy_id(args.policy_id)
    if not (workspace / "src" / "lerobot").is_dir():
        raise FileNotFoundError(workspace / "src" / "lerobot")
    if not (workspace / "cobot_bridge" / "policy_server.py").is_file():
        raise FileNotFoundError(workspace / "cobot_bridge" / "policy_server.py")
    if not (local_policy / "model.safetensors").is_file():
        raise FileNotFoundError(local_policy / "model.safetensors")
    local_vlm_metadata = _vlm_metadata_path(local_policy)

    remote_workspace = args.remote_workspace
    remote_policy = remote_workspace / "checkpoints" / policy_id
    mkdir_command = "mkdir -p " + " ".join(
        shlex.quote(str(path))
        for path in (
            remote_workspace / "src" / "lerobot",
            remote_workspace / "cobot_bridge",
            remote_policy,
            remote_workspace / "vlm_metadata",
        )
    )
    _run(_ssh_command(args.ssh_host, mkdir_command))

    common_rsync = ["rsync", "-az", "--partial", "-e", _rsync_ssh()]
    _run(
        [
            *common_rsync,
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            f"{workspace / 'src' / 'lerobot'}/",
            f"{args.ssh_host}:{remote_workspace / 'src' / 'lerobot'}/",
        ]
    )
    _run(
        [
            *common_rsync,
            "--exclude=__pycache__",
            "--exclude=web_static",
            f"{workspace / 'cobot_bridge'}/",
            f"{args.ssh_host}:{remote_workspace / 'cobot_bridge'}/",
        ]
    )
    _run(
        [
            *common_rsync,
            f"{local_policy}/",
            f"{args.ssh_host}:{remote_policy}/",
        ]
    )
    _run(
        [
            *common_rsync,
            "--copy-links",
            "--exclude=model.safetensors",
            "--exclude=*.bin",
            f"{local_vlm_metadata}/",
            f"{args.ssh_host}:{remote_workspace / 'vlm_metadata'}/",
        ]
    )

    server = [
        str(args.remote_python),
        "-m",
        "cobot_bridge.policy_server",
        "--bind",
        args.bind,
        "--policy-path",
        str(remote_policy),
        "--device",
        args.device,
        "--vlm-model-path",
        str(remote_workspace / "vlm_metadata"),
        "--cameras",
        *args.cameras,
    ]
    pythonpath = f"{remote_workspace / 'src'}:{remote_workspace}"
    remote_command = (
        f"cd {shlex.quote(str(remote_workspace))} && "
        f"PYTHONPATH={shlex.quote(pythonpath)} exec {shlex.join(server)}"
    )
    os.execvp("ssh", _ssh_command(args.ssh_host, remote_command))


if __name__ == "__main__":
    main()
