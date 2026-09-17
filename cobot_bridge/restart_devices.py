"""Safely restart the AGX Cobot Magic arm and camera ROS device stacks.

Also restarts the AGX-local JPEG preview server so the dashboard preview is
brought back together with the devices it depends on.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
import time
from pathlib import Path

from .client import CobotBridgeClient


CAN_INTERFACES = ("can_left_m", "can_left_s", "can_right_m", "can_right_s")
CAMERAS = ("front", "left", "right")
SIDES = ("left", "right")
REALSENSE_SERIALS = {
    "front": "042222071914",
    "left": "337122071405",
    "right": "337322073221",
}
PREVIEW_PYTHON = "/home/agilex/miniforge3/envs/robot_bridge/bin/python"
PREVIEW_PORT = 8091
PREVIEW_LOG = "/home/agilex/lerobot/mjpeg_cam_server.log"


class DeviceRestartError(RuntimeError):
    def __init__(self, message: str, status: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.status = status or {}


def _ssh(
    host: str,
    arguments: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        "ssh",
        "-T",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        host,
        shlex.join(arguments),
    ]
    result = subprocess.run(
        command,
        text=True,
        input=input_text,
        capture_output=True,
        timeout=20,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if check and result.returncode != 0:
        raise DeviceRestartError(
            f"remote command failed ({result.returncode}): {shlex.join(arguments)}"
        )
    return result


def _bash(
    host: str,
    script: str,
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return _ssh(host, ["bash", "-lc", script], check=check, input_text=input_text)


def _tmux_shell(command: str) -> str:
    return shlex.join(["bash", "-lc", command])


def _window_exists(host: str, name: str) -> bool:
    result = _ssh(
        host,
        ["tmux", "list-windows", "-t", "cobot_magic", "-F", "#{window_name}"],
        check=False,
    )
    return result.returncode == 0 and name in result.stdout.splitlines()


def _can_ready(host: str) -> bool:
    checks = " && ".join(
        f"ip -br link show {shlex.quote(interface)} | grep -q ' UP '"
        for interface in CAN_INTERFACES
    )
    return _bash(host, checks, check=False).returncode == 0


def _load_sudo_password(path: Path | None) -> str:
    if path is None:
        raise DeviceRestartError(
            "CAN setup requires AGX sudo credentials; configure COBOT_ROBOT_SUDO_PASSWORD_FILE"
        )
    try:
        info = path.stat()
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise DeviceRestartError(
                f"sudo credential file must be owner-only (chmod 600): {path}"
            )
        password = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DeviceRestartError(f"cannot read AGX sudo credential file {path}: {exc}") from exc
    if not password:
        raise DeviceRestartError(f"AGX sudo credential file is empty: {path}")
    return password


def _ensure_can(host: str, workspace: Path, sudo_password_file: Path | None) -> None:
    if _can_ready(host):
        print("All four logical CAN interfaces are already UP.", flush=True)
        return
    configure = workspace / "scripts" / "configure_cobot_magic_can.sh"
    sudo_password = _load_sudo_password(sudo_password_file)
    result = _bash(
        host,
        (
            "IFS= read -r sudo_password; "
            "printf '%s\\n' \"$sudo_password\" | "
            "sudo -S -p '' env CONFIGURE_BASE_CAN=0 "
            f"bash {shlex.quote(str(configure))}"
        ),
        check=False,
        # stdin, not command line or log: the remote shell forwards this only
        # to sudo. The dashboard never stores or displays the password.
        input_text=sudo_password + "\n",
    )
    if result.returncode != 0 or not _can_ready(host):
        raise DeviceRestartError(
            "CAN interfaces are not UP and passwordless limited CAN setup is unavailable"
        )


def _stop_remote_rollouts(host: str) -> None:
    result = _ssh(
        host,
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        check=False,
    )
    if result.returncode != 0:
        return
    for session in result.stdout.splitlines():
        if session.startswith("cobot_rollout_"):
            _ssh(host, ["tmux", "kill-session", "-t", session], check=False)
            print(f"Stopped remote rollout session {session}.", flush=True)


def _ensure_realsense_cameras(host: str) -> None:
    result = _ssh(host, ["rs-enumerate-devices", "-s"], check=False)
    if result.returncode != 0:
        raise DeviceRestartError("rs-enumerate-devices failed on the robot")
    detected = set(result.stdout.split())
    missing = [
        f"{name}={serial}"
        for name, serial in REALSENSE_SERIALS.items()
        if serial not in detected
    ]
    if missing:
        raise DeviceRestartError(
            "configured RealSense camera(s) not detected: " + ", ".join(missing)
        )
    cameras = ", ".join(
        f"{name}={serial}" for name, serial in REALSENSE_SERIALS.items()
    )
    print(f"Detected all configured RealSense cameras: {cameras}.", flush=True)


def _camera_command(workspace: Path) -> str:
    return "; ".join(
        (
            "source /opt/ros/humble/setup.bash",
            "source /home/agilex/ros2_ws/install/setup.bash",
            "unset ROS_DOMAIN_ID",
            "export ROS_LOCALHOST_ONLY=1",
            f"cd {shlex.quote(str(workspace))}",
            "exec ros2 launch scripts/cobot_magic_mixed_camera.launch.py",
        )
    )


def _stop_camera_processes(host: str) -> None:
    # The camera stack may have been started outside ``cobot_magic`` (for
    # example after a manual AGX reboot).  A tmux-only cleanup would leave its
    # RealSense drivers alive and make the replacement launch silently lose
    # the devices.  The bracketed pattern cannot match this remote shell.
    camera_patterns = (
        "cobot_magic_mixed_camera.launch.[p]y",
        "realsense2_camera_nod[e].*__node:=camera_[flr]",
    )
    for camera_pattern in camera_patterns:
        _bash(host, f"pkill -INT -f '{camera_pattern}' || true", check=False)
    time.sleep(2)
    for camera_pattern in camera_patterns:
        _bash(host, f"pkill -KILL -f '{camera_pattern}' || true", check=False)


def _restart_camera_window(host: str, workspace: Path) -> None:
    # Release every UVC node before probing. `rs-enumerate-devices` otherwise
    # reports false "not detected" failures when a stale node owns /dev/video.
    _stop_camera_processes(host)
    _ensure_realsense_cameras(host)
    camera_command = _camera_command(workspace)
    # Clean up both current and legacy camera window names so stale drivers
    # cannot retain a USB device and race the replacement launch.
    for camera_window in ("camera", "cameras", "arm"):
        if _window_exists(host, camera_window):
            _ssh(host, ["tmux", "kill-window", "-t", f"cobot_magic:{camera_window}"])
    _ssh(
        host,
        [
            "tmux",
            "new-window",
            "-d",
            "-t",
            "cobot_magic",
            "-n",
            "camera",
            _tmux_shell(camera_command),
        ],
    )
    print("Restarted AGX camera tmux window.", flush=True)


def _restart_tmux_stacks(host: str, workspace: Path) -> None:
    arm_command = "; ".join(
        (
            "source /opt/ros/humble/setup.bash",
            "source /home/agilex/piper_ros/install/setup.bash",
            "unset ROS_DOMAIN_ID",
            "export ROS_LOCALHOST_ONLY=1",
            f"cd {shlex.quote(str(workspace))}",
            "exec ros2 launch scripts/cobot_magic_master_puppet.launch.py live_output:=true",
        )
    )
    has_session = (
        _ssh(host, ["tmux", "has-session", "-t", "cobot_magic"], check=False).returncode
        == 0
    )
    if not has_session:
        _ssh(
            host,
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                "cobot_magic",
                "-n",
                "arms",
                _tmux_shell(arm_command),
            ],
        )
    else:
        if _window_exists(host, "arms"):
            _ssh(host, ["tmux", "kill-window", "-t", "cobot_magic:arms"])
        _ssh(
            host,
            ["tmux", "new-window", "-d", "-t", "cobot_magic", "-n", "arms", _tmux_shell(arm_command)],
        )

    _restart_camera_window(host, workspace)
    print("Restarted AGX arms and camera tmux windows.", flush=True)


def _restart_preview_server(host: str, endpoint: str) -> None:
    # Each ssh call below keeps the bracketed pattern out of its own command
    # line except in calls where no kill runs, so pkill/pgrep never match the
    # remote shell itself. The nohup line (plain module name) is a separate
    # call, so it is never present alongside a kill.
    pattern = "cobot_mjpeg_cam_serve[r]"
    _bash(host, f"pkill -f '{pattern}' || true")
    time.sleep(2)
    # aiortc's graceful shutdown can hang on stale connections, so force kill
    # anything that is still around before taking the port again.
    _bash(host, f"pkill -9 -f '{pattern}' || true")
    time.sleep(1)
    # "cd" is a separate statement (not "cd ... && daemon &"): backgrounding an
    # AND-list forks a subshell that inherits the ssh channel's fds and waits
    # on the daemon forever, so ssh would hang past the 20s helper timeout.
    _bash(
        host,
        (
            f"cd /home/agilex/lerobot || exit 1; "
            f"setsid nohup {PREVIEW_PYTHON} -m cobot_bridge.cobot_mjpeg_cam_server "
            f"--port {PREVIEW_PORT} --bridge-endpoint {endpoint} "
            f">> {PREVIEW_LOG} 2>&1 < /dev/null & "
            "echo launched"
        ),
    )
    time.sleep(8)
    result = _bash(host, f"pgrep -f '{pattern}'", check=False)
    pids = [pid for pid in result.stdout.split() if pid.strip()]
    if not pids:
        raise DeviceRestartError(
            f"AGX-local JPEG preview server failed to start on port {PREVIEW_PORT}"
        )
    print(
        f"Restarted AGX-local JPEG preview server on port {PREVIEW_PORT} (pid {pids[0]}).",
        flush=True,
    )


def _wait_ready(
    endpoint: str, timeout_s: float, stable_seconds: float = 5.0
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, object] = {}
    ready_since: float | None = None
    while time.monotonic() < deadline:
        try:
            with CobotBridgeClient(endpoint, timeout_s=1.5) as client:
                raw, images = client.get_observation(sides=SIDES, images=CAMERAS)
            arms = {
                f"{role}_{side}": bool(
                    (sample := raw.get(role, {}).get(side))
                    and sample.get("fresh", False)
                    and len(sample.get("position", [])) == 7
                )
                for role in ("master", "puppet")
                for side in SIDES
            }
            image_meta = raw.get("images", {})
            cameras = {
                name: bool(
                    (metadata := image_meta.get(name))
                    and metadata.get("fresh", False)
                    and images.get(name)
                )
                for name in CAMERAS
            }
            control = raw.get("control", {})
            last = {
                "arms": arms,
                "cameras": cameras,
                "mode": control.get("effective_mode"),
                "teach_active": bool(control.get("teach_active", False)),
                "output_enabled": bool(control.get("output_enabled", False)),
            }
            all_ready = (
                all(arms.values())
                and all(cameras.values())
                and not last["teach_active"]
            )
            if all_ready:
                ready_since = ready_since or time.monotonic()
                if time.monotonic() - ready_since >= stable_seconds:
                    print(json.dumps(last, indent=2), flush=True)
                    return last
            else:
                ready_since = None
        except Exception as exc:
            last = {"error": str(exc)}
        time.sleep(1.0)
    raise DeviceRestartError(
        f"devices did not become ready within {timeout_s:.0f}s: {last}", status=last
    )


def restart_devices(
    host: str,
    workspace: Path,
    endpoint: str,
    timeout_s: float,
    sudo_password_file: Path | None,
) -> None:
    _ensure_can(host, workspace, sudo_password_file)
    _stop_remote_rollouts(host)
    _restart_tmux_stacks(host, workspace)
    _restart_preview_server(host, endpoint)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    for attempt in range(1, 4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            _wait_ready(endpoint, min(20.0, remaining), stable_seconds=5.0)
            return
        except DeviceRestartError as exc:
            last_error = exc
            if attempt == 3:
                break
            camera_status = exc.status.get("cameras", {})
            if isinstance(camera_status, dict) and all(camera_status.values()):
                print(
                    "Cameras are stable; arm feedback is still missing, so not restarting cameras again.",
                    flush=True,
                )
            else:
                print(f"Camera stack was not stable; retrying ({attempt}/3).", flush=True)
                _restart_camera_window(host, workspace)
    raise DeviceRestartError(f"device restart did not stabilize: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="agilex@192.168.250.2")
    parser.add_argument("--remote-workspace", type=Path, default=Path("/home/agilex/lerobot"))
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--sudo-password-file",
        type=Path,
        default=(Path(os.environ["COBOT_ROBOT_SUDO_PASSWORD_FILE"])
                 if os.environ.get("COBOT_ROBOT_SUDO_PASSWORD_FILE") else None),
        help="owner-only local file used only to authenticate the AGX CAN setup",
    )
    args = parser.parse_args()
    restart_devices(
        args.ssh_host,
        args.remote_workspace,
        args.endpoint,
        args.timeout,
        args.sudo_password_file,
    )


if __name__ == "__main__":
    main()
