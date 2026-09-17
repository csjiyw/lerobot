"""Move all Cobot Magic arms to the verified non-singular ready pose.

The former all-zero joint pose is singular for Piper Cartesian IK.  The
verified ready pose below is used consistently for master-left, master-right,
puppet-left and puppet-right; it is reached through canonical Mode 2
individual joint alignment rather than IK.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from typing import Any

import numpy as np

from .client import CobotBridgeClient


LOG = logging.getLogger("cobot_bridge.move_init_pose")
# ``cobot_magic_il_2_eef`` episode 31, frame 0: a real task-start pose for
# ``pick and place the object``.  It keeps the TCP and gripper state within
# the policy's start-of-episode distribution while avoiding the all-zero
# Cartesian-IK singularity (Jacobian condition number ~=74, warning=1000).
# The same joint convention/pose is used for the left side.
INIT_POSE = np.asarray(
    [0.0670, 0.0040, -0.3450, -0.1100, 0.5710, -0.1280, 0.0],
    dtype=np.float32,
)
DEFAULT_JOINT_SPEED = 0.48
DEFAULT_MIN_SECONDS = 1.0
SIDES = ("left", "right")
ROLES = ("master", "puppet")
STOP = False


def _stop(_signal: int, _frame: Any) -> None:
    global STOP
    STOP = True


def smoothstep_chunk(start: list[float], duration: float, waypoint_hz: float) -> list[list[float]]:
    count = max(2, int(math.ceil(duration * waypoint_hz)) + 1)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
    begin = np.asarray(start, dtype=np.float64)
    return [row.tolist() for row in begin + smooth[:, None] * (INIT_POSE - begin)]


def _accepted(label: str, result: dict[str, Any]) -> None:
    if not result.get("accepted", False):
        raise RuntimeError(f"{label} rejected: {result}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--joint-speed", type=float, default=DEFAULT_JOINT_SPEED)
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--waypoint-hz", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.joint_speed <= 0 or args.min_seconds <= 0 or args.waypoint_hz <= 0:
        raise ValueError("speed, duration and waypoint rate must be positive")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    with CobotBridgeClient(args.endpoint, timeout_s=2.0) as client:
        raw, _ = client.get_observation(sides=SIDES, images=())
        if not raw.get("ready", False):
            raise RuntimeError(f"robot feedback is not ready: {raw.get('missing_or_stale')}")
        control = raw.get("control", {})
        if control.get("teach_active", False):
            raise RuntimeError("Teach is active")
        starts = {
            role: {side: list(raw[role][side]["position"]) for side in SIDES}
            for role in ROLES
        }
        max_joint_delta = max(
            abs(starts[role][side][joint] - INIT_POSE[joint])
            for role in ROLES
            for side in SIDES
            for joint in range(7)
        )
        duration = max(args.min_seconds, max_joint_delta / args.joint_speed)
        if duration + 3.0 > args.timeout:
            raise RuntimeError(
                f"init plan needs {duration + 3.0:.1f}s but timeout is {args.timeout:.1f}s"
            )
        LOG.info(
            "four-arm init plan target=%s estimated_duration=%.3fs starts=%s",
            INIT_POSE.tolist(),
            duration,
            starts,
        )
        if not args.execute:
            LOG.info("dry run only; no command sent")
            return

        control_started = False
        try:
            _accepted("Mode 2", client.set_mode(2))
            control_started = True
            _accepted("output enable", client.set_output(True))
            for _ in range(20):
                time.sleep(0.05)
                current, _ = client.get_observation(sides=SIDES, images=())
                control = current.get("control", {})
                if control.get("teach_active", False):
                    raise RuntimeError("init motion interrupted by Teach")
                if (
                    control.get("effective_mode_id") == 5
                    and control.get("output_enabled", False)
                ):
                    break
            else:
                raise RuntimeError(f"ALIGN_SYNC/output feedback missing: {control}")

            epoch = int(current.get("control", {}).get("epoch", -1))
            if epoch < 0:
                raise RuntimeError(f"alignment epoch missing: {current.get('control')}")
            alignment: dict[str, Any] = {}
            for side in SIDES:
                result = client.set_action(
                    side,
                    INIT_POSE.tolist(),
                    epoch=epoch,
                    sequence=1,
                    alignment_speed=args.joint_speed,
                )
                _accepted(f"{side} alignment target", result)
                alignment.update(result)

            motion_started = time.monotonic()
            actual_duration = float(alignment.get("duration", duration))
            deadline = motion_started + min(args.timeout, actual_duration + 3.0)
            last = current
            aligned = False
            while not STOP and time.monotonic() < deadline:
                time.sleep(0.05)
                last, _ = client.get_observation(sides=SIDES, images=())
                mode = int(last.get("control", {}).get("effective_mode_id", -1))
                if last.get("control", {}).get("teach_active", False) or mode in (3, 4):
                    raise RuntimeError("init motion interrupted by Teach")
                if mode == 2:
                    aligned = True
                    break
                if mode != 5:
                    raise RuntimeError(f"init alignment entered unexpected mode: {last.get('control')}")
            if STOP:
                raise KeyboardInterrupt
            if not aligned:
                raise RuntimeError(f"four-arm init alignment timed out: {last.get('control')}")
            errors = {
                f"{role}_{side}": np.abs(
                    np.asarray(last[role][side]["position"]) - INIT_POSE
                )
                for role in ROLES
                for side in SIDES
            }
            if any(float(np.max(error[:6])) > 0.04 or float(error[6]) > 0.012 for error in errors.values()):
                raise RuntimeError(
                    "init feedback did not reach target: "
                    + str({arm: error.round(4).tolist() for arm, error in errors.items()})
                )
            LOG.info("all four master/puppet arms reached init pose")
        finally:
            if control_started:
                try:
                    LOG.info("disable_output=%s", client.set_output(False))
                finally:
                    LOG.info("safe_mode=%s", client.set_mode(0))


if __name__ == "__main__":
    main()
