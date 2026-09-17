"""Run a Cobot rollout on the robot and keep it managed by the PC dashboard."""

from __future__ import annotations

import argparse
import os
from pathlib import PurePosixPath
import re
import shlex
import signal
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
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
RUNNER_MODULES = {
    "standard": "cobot_bridge.run_lerobot_policy",
    "smooth_v2": "cobot_bridge.run_lerobot_policy_smooth",
    "smooth_v3": "cobot_bridge.run_lerobot_policy_smooth_v3",
    "edge_v4": "cobot_bridge.run_lerobot_policy_edge_v4",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--remote-workspace", type=PurePosixPath, required=True)
    parser.add_argument("--remote-python", type=PurePosixPath, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--policy-server-endpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--action-space", choices=("joint", "eef_delta", "eef_target"), default="joint"
    )
    parser.add_argument(
        "--model-eef-frame",
        choices=("link6", "gripper_center"),
        default="gripper_center",
        help="EEF frame used by the checkpoint; SDK feedback/targets remain link6",
    )
    parser.add_argument(
        "--eef-ik-backend",
        choices=("pink", "sdk"),
        default="sdk",
        help="EEF target backend used by the rollout client",
    )
    parser.add_argument("--pink-urdf", default=None)
    parser.add_argument("--runner", choices=tuple(RUNNER_MODULES), default="standard")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="actions consumed per prediction; defaults to the complete model chunk",
    )
    parser.add_argument("--cameras", nargs="+", default=("front", "right"))
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument("--startup-init-hold", action="store_true")
    parser.add_argument("--diagnostic-output", type=PurePosixPath)
    return parser


def _ssh(host: str, remote_command: str, *, tty: bool = False) -> list[str]:
    return ["ssh", *(('-tt',) if tty else ()), *SSH_OPTIONS, host, remote_command]


def _tmux_command(args: argparse.Namespace) -> tuple[str, str]:
    if not SESSION_PATTERN.fullmatch(args.session):
        raise ValueError("invalid remote tmux session name")
    command = [
        str(args.remote_python),
        "-m",
        RUNNER_MODULES[args.runner],
        "--endpoint",
        args.endpoint,
        "--policy-server-endpoint",
        args.policy_server_endpoint,
        "--task",
        args.task,
        "--side",
        args.side,
        "--fps",
        str(args.fps),
        "--duration",
        str(args.duration),
        "--mode",
        str(args.mode),
        "--action-space",
        getattr(args, "action_space", "joint"),
        "--model-eef-frame",
        getattr(args, "model_eef_frame", "gripper_center"),
        "--eef-ik-backend",
        getattr(args, "eef_ik_backend", "sdk"),
        "--cameras",
        *args.cameras,
    ]
    if getattr(args, "pink_urdf", None):
        command.extend(("--pink-urdf", str(args.pink_urdf)))
    if args.n_action_steps is not None:
        command.extend(("--n-action-steps", str(args.n_action_steps)))
    if args.send_actions:
        command.append("--send-actions")
    if args.startup_init_hold:
        command.append("--startup-init-hold")
    if args.diagnostic_output is not None:
        command.extend(("--diagnostic-output", str(args.diagnostic_output)))
    pythonpath = f"{args.remote_workspace / 'src'}:{args.remote_workspace}"
    rollout = (
        f"cd {shlex.quote(str(args.remote_workspace))} && "
        f"PYTHONUNBUFFERED=1 PYTHONPATH={shlex.quote(pythonpath)} exec {shlex.join(command)}"
    )
    start = shlex.join(["tmux", "new-session", "-d", "-s", args.session, rollout])
    attach = shlex.join(["tmux", "attach-session", "-t", args.session])
    return start, attach


def main() -> None:
    args = build_parser().parse_args()
    start_command, attach_command = _tmux_command(args)
    interrupted = False
    attach: subprocess.Popen[bytes] | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        if attach is not None and attach.poll() is None:
            attach.terminate()

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        subprocess.run(_ssh(args.ssh_host, start_command), check=True)
        if interrupted:
            return
        attach = subprocess.Popen(_ssh(args.ssh_host, attach_command, tty=True))
        returncode = attach.wait()
    finally:
        # The session normally disappears when rollout exits. Always issue an
        # exact-session cleanup as protection against SSH disconnects or a stop
        # arriving during startup.
        kill_command = shlex.join(["tmux", "kill-session", "-t", args.session])
        subprocess.run(_ssh(args.ssh_host, kill_command), check=False)
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
    if interrupted:
        return
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
