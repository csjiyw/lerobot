"""Replay recorded puppet command targets through the canonical policy path.

The recorded waypoints are interpolated using their PC timestamps. The script
is computation-only unless ``--execute`` is supplied. Puppet mode performs a
local smooth approach in Mode 1. Sync mode delegates the initial independent
master/puppet alignment to the robot arbiter before streaming shared actions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import h5py
import numpy as np

from .client import BridgeError, CobotBridgeClient
from .record_raw import FORMAT
from .remote_robot import CobotMagicRemoteRobot


LOG = logging.getLogger("cobot_bridge.replay_raw")
SAFE = 0
POLICY_PUPPET = 1
RLT_SYNC = 2
ALIGN_SYNC = 5
TEACH = 3
RESYNC_REQUIRED = 4


class TeachEpisodeComplete(RuntimeError):
    """The operator completed a Teach intervention; policy must shut down."""


def longest_valid_run(valid: np.ndarray) -> slice:
    """Return the longest contiguous valid command interval."""

    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("command validity mask must be one-dimensional")
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    if len(starts) == 0:
        raise ValueError("raw episode contains no valid puppet_command interval")
    lengths = stops - starts
    best = int(np.argmax(lengths))
    if lengths[best] < 2:
        raise ValueError("valid puppet_command interval must contain at least two frames")
    return slice(int(starts[best]), int(stops[best]))


def load_trajectory(path: Path, source: str, waypoint_hz: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as raw:
        if raw.attrs.get("format") != FORMAT or not bool(raw.attrs.get("completed", False)):
            raise ValueError("raw episode is missing, unsupported, or incomplete")
        timestamps = np.asarray(raw["frames/pc_timestamp_ns"][:], dtype=np.int64)
        positions = np.asarray(raw[f"frames/{source}"][:], dtype=np.float64)
        if source == "puppet_command":
            selected = longest_valid_run(raw["frames/puppet_command_valid"][:])
            timestamps = timestamps[selected]
            positions = positions[selected]
    if positions.ndim != 2 or positions.shape[1] != 7 or not np.isfinite(positions).all():
        raise ValueError("trajectory must be finite Nx7")
    source_time = (timestamps - timestamps[0]) / 1e9
    if len(source_time) < 2 or not np.all(np.diff(source_time) > 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    target_time = np.arange(0.0, source_time[-1] + 0.5 / waypoint_hz, 1.0 / waypoint_hz)
    target = np.column_stack(
        [np.interp(target_time, source_time, positions[:, joint]) for joint in range(7)]
    )
    return target_time, target


def alignment_trajectory(
    start: np.ndarray,
    target: np.ndarray,
    *,
    control_hz: float,
    max_speed: float,
    minimum_seconds: float,
) -> tuple[float, np.ndarray]:
    maximum = float(np.max(np.abs(target[:6] - start[:6])))
    duration = max(minimum_seconds, maximum / max_speed)
    steps = max(2, math.ceil(duration * control_hz) + 1)
    phase = np.linspace(0.0, 1.0, steps)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    return duration, start[None, :] + smooth[:, None] * (target - start)[None, :]


def trim_stationary_holds(
    timestamps: np.ndarray,
    trajectory: np.ndarray,
    *,
    threshold: float,
    settle_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Remove leading/trailing holds while preserving settle time around motion."""

    if threshold <= 0 or settle_seconds < 0:
        raise ValueError("trim threshold must be positive and settle seconds non-negative")
    movement = np.max(np.abs(np.diff(trajectory, axis=0)), axis=1)
    moving = np.flatnonzero(movement > threshold)
    if len(moving) == 0:
        return timestamps, trajectory, {
            "trimmed": False,
            "start_hold_removed_s": 0.0,
            "end_hold_removed_s": 0.0,
            "original_waypoints": len(trajectory),
            "trimmed_waypoints": len(trajectory),
        }

    first_motion_time = float(timestamps[moving[0]])
    last_motion_time = float(timestamps[moving[-1] + 1])
    keep_start_time = max(float(timestamps[0]), first_motion_time - settle_seconds)
    keep_end_time = min(float(timestamps[-1]), last_motion_time + settle_seconds)
    epsilon = max(1e-12, float(timestamps[-1] - timestamps[0]) * 1e-12)
    start_index = int(np.searchsorted(timestamps, keep_start_time - epsilon, side="left"))
    end_index = int(np.searchsorted(timestamps, keep_end_time + epsilon, side="right"))
    end_index = max(start_index + 2, min(end_index, len(trajectory)))
    clipped_time = timestamps[start_index:end_index] - timestamps[start_index]
    clipped = trajectory[start_index:end_index]
    return clipped_time, clipped, {
        "trimmed": start_index > 0 or end_index < len(trajectory),
        "start_hold_removed_s": float(timestamps[start_index] - timestamps[0]),
        "end_hold_removed_s": float(timestamps[-1] - timestamps[end_index - 1]),
        "original_waypoints": len(trajectory),
        "trimmed_waypoints": len(clipped),
    }


def require_accepted(label: str, result: dict[str, object]) -> None:
    if not result.get("accepted", False):
        raise BridgeError(f"{label} rejected: {result}")


def wait_for_teach_release(robot: CobotMagicRemoteRobot):
    LOG.warning("Teach intervention active; action chunk cleared, waiting for release")
    while True:
        observation = robot.observe()
        if (
            not observation.teach_active
            and observation.effective_mode == RESYNC_REQUIRED
        ):
            LOG.warning("Teach released; RESYNC_REQUIRED observed")
            return observation
        if not observation.teach_active and observation.effective_mode not in (
            TEACH,
            RESYNC_REQUIRED,
        ):
            raise BridgeError(f"unexpected mode after Teach: {observation.control}")
        time.sleep(0.05)


def establish_sync_mode(
    client: CobotBridgeClient,
    robot: CobotMagicRemoteRobot,
    first_target: np.ndarray,
    *,
    alignment_timeout: float,
    initial_request: bool,
):
    """Enter RLT_SYNC; any completed Teach intervention terminates policy."""

    if initial_request:
        require_accepted("Mode 2", robot.request_mode(RLT_SYNC))

    while True:
        current = robot.observe()
        if current.teach_active or current.effective_mode == TEACH:
            wait_for_teach_release(robot)
            raise TeachEpisodeComplete("Teach released before alignment")
        if current.effective_mode == RESYNC_REQUIRED:
            raise TeachEpisodeComplete("Teach release requires policy shutdown")
        if current.effective_mode != ALIGN_SYNC:
            raise BridgeError(f"expected ALIGN_SYNC before alignment: {current.control}")

        require_accepted("output enable", robot.set_output(True))
        for _ in range(20):
            time.sleep(0.05)
            current = robot.observe()
            if current.teach_active or current.effective_mode == TEACH:
                break
            if bool(current.control.get("output_enabled", False)):
                break
        if current.teach_active or current.effective_mode == TEACH:
            wait_for_teach_release(robot)
            raise TeachEpisodeComplete("Teach released while enabling alignment")
        if not bool(current.control.get("output_enabled", False)):
            raise BridgeError("output_enabled feedback did not become true")

        alignment = robot.send_alignment_target(first_target.tolist(), observation=current)
        require_accepted("alignment target", alignment)
        auxiliary_holds: dict[str, np.ndarray] = {}
        waiting_sides = alignment.get("waiting_sides", [])
        if not isinstance(waiting_sides, list):
            raise BridgeError(f"invalid alignment waiting_sides: {alignment}")
        for side in waiting_sides:
            if side == robot.side:
                raise BridgeError(f"selected side still missing after target submission: {alignment}")
            target = current.position("master", side).astype(np.float64, copy=True)
            alignment = robot.send_alignment_target_for_side(
                side, target.tolist(), observation=current
            )
            require_accepted(f"{side} alignment hold target", alignment)
            auxiliary_holds[side] = target
        if alignment.get("reason") == "sync_first_target_buffered" or "duration" not in alignment:
            raise BridgeError(f"alignment did not start after all targets: {alignment}")
        LOG.warning(
            "individual alignment started: duration=%.3fs max_delta=%.4frad epoch=%s holds=%s",
            float(alignment.get("duration", 0.0)),
            float(alignment.get("maximum_delta", 0.0)),
            alignment.get("epoch"),
            sorted(auxiliary_holds),
        )
        deadline = time.monotonic() + alignment_timeout
        while time.monotonic() < deadline:
            time.sleep(0.02)
            current = robot.observe()
            if current.teach_active or current.effective_mode in (TEACH, RESYNC_REQUIRED):
                if current.effective_mode != RESYNC_REQUIRED:
                    wait_for_teach_release(robot)
                raise TeachEpisodeComplete("Teach released during alignment")
            if current.effective_mode == RLT_SYNC:
                LOG.warning("individual alignment complete; RLT_SYNC owns control")
                return current, alignment, auxiliary_holds
            if current.effective_mode != ALIGN_SYNC:
                raise BridgeError(f"alignment interrupted: {current.control}")
        raise BridgeError(f"individual alignment timed out: {current.control}")


def queue_sync_chunks(
    client: CobotBridgeClient,
    *,
    selected_side: str,
    chunk: np.ndarray,
    auxiliary_holds: dict[str, np.ndarray],
    epoch: int,
    waypoint_hz: float,
) -> tuple[dict, dict[str, dict]]:
    """Queue one moving chunk plus equal-duration stationary chunks for other sync sides."""

    auxiliary_results: dict[str, dict] = {}
    for side, target in auxiliary_holds.items():
        hold_chunk = np.repeat(target[None, :], len(chunk), axis=0)
        result = client.set_action_chunk(
            side,
            hold_chunk.tolist(),
            epoch=epoch,
            sequence=time.time_ns(),
            waypoint_hz=waypoint_hz,
        )
        require_accepted(f"{side} RLT hold chunk", result)
        auxiliary_results[side] = result

    selected_result = client.set_action_chunk(
        selected_side,
        chunk.tolist(),
        epoch=epoch,
        sequence=time.time_ns(),
        waypoint_hz=waypoint_hz,
    )
    require_accepted("RLT chunk", selected_result)
    return selected_result, auxiliary_results


def monitor_sync_chunk(robot: CobotMagicRemoteRobot, duration: float):
    deadline = time.monotonic() + duration + 0.25
    last = robot.observe()
    while time.monotonic() < deadline:
        time.sleep(0.05)
        last = robot.observe()
        if last.teach_active or last.effective_mode in (TEACH, RESYNC_REQUIRED):
            return True, last
        if last.effective_mode != RLT_SYNC:
            raise BridgeError(f"replay interrupted outside Teach: {last.control}")
    return False, last


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--mode",
        choices=("puppet", "sync"),
        default="puppet",
        help="puppet uses mode 1; sync sends the identical action to master and puppet",
    )
    parser.add_argument(
        "--source",
        choices=("master_position", "puppet_position", "puppet_command"),
        default="puppet_command",
    )
    parser.add_argument("--waypoint-hz", type=int, default=30)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.5,
        help="replay speed in (0, 1]; 0.5 doubles replay duration while AGX stays at 200 Hz",
    )
    parser.add_argument("--alignment-speed", type=float, default=0.12)
    parser.add_argument("--alignment-min-seconds", type=float, default=2.0)
    parser.add_argument("--alignment-timeout", type=float, default=30.0)
    parser.add_argument(
        "--trim-stationary-holds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically remove leading and trailing stationary waypoints",
    )
    parser.add_argument("--trim-threshold-rad", type=float, default=1e-4)
    parser.add_argument("--trim-settle-seconds", type=float, default=0.5)
    parser.add_argument("--loop", action="store_true", help="loop sync replay until interrupted")
    parser.add_argument("--loop-transition-speed", type=float, default=0.12)
    parser.add_argument("--loop-transition-min-seconds", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.waypoint_hz <= 0 or args.waypoint_hz > 200 or args.alignment_speed <= 0:
        raise ValueError("waypoint_hz must be in [1, 200] and alignment_speed positive")
    if not 0.0 < args.speed_scale <= 1.0:
        raise ValueError("speed_scale must be in (0, 1]")
    if args.loop and args.mode != "sync":
        raise ValueError("loop replay currently requires --mode sync")
    if args.loop_transition_speed <= 0 or args.loop_transition_min_seconds <= 0:
        raise ValueError("loop transition speed and minimum seconds must be positive")
    target_time, trajectory = load_trajectory(args.raw, args.source, args.waypoint_hz)
    original_duration = float(target_time[-1])
    trim_info: dict[str, float | int | bool] = {
        "trimmed": False,
        "start_hold_removed_s": 0.0,
        "end_hold_removed_s": 0.0,
        "original_waypoints": len(trajectory),
        "trimmed_waypoints": len(trajectory),
    }
    if args.trim_stationary_holds:
        target_time, trajectory, trim_info = trim_stationary_holds(
            target_time,
            trajectory,
            threshold=args.trim_threshold_rad,
            settle_seconds=args.trim_settle_seconds,
        )
    playback_hz = args.waypoint_hz * args.speed_scale
    replay_duration = float(target_time[-1] / args.speed_scale)
    summary = {
        "raw": str(args.raw),
        "source": args.source,
        "loop": args.loop,
        "trim": trim_info,
        "waypoints": len(trajectory),
        "source_waypoint_hz": args.waypoint_hz,
        "playback_waypoint_hz": playback_hz,
        "speed_scale": args.speed_scale,
        "agx_output_hz": 200,
        "mode": args.mode,
        "original_source_duration_s": original_duration,
        "source_duration_s": float(target_time[-1]),
        "duration_s": replay_duration,
        "first": trajectory[0].round(6).tolist(),
        "last": trajectory[-1].round(6).tolist(),
        "execute": args.execute,
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        return

    with CobotBridgeClient(args.endpoint, timeout_s=2.0) as client:
        ping = client.ping()
        if not ping.get("allow_control", False):
            raise BridgeError("AGX bridge is read-only")
        robot = CobotMagicRemoteRobot(client, side=args.side, images=())
        initial = robot.observe()
        if not initial.ready or initial.teach_active:
            raise BridgeError("robot is not ready or Teach is active")

        try:
            if args.mode == "puppet":
                duration, alignment = alignment_trajectory(
                    initial.puppet_state.astype(np.float64),
                    trajectory[0],
                    # Keep the approach duration unchanged even when replay is slowed.
                    control_hz=playback_hz,
                    max_speed=args.alignment_speed,
                    minimum_seconds=args.alignment_min_seconds,
                )
                LOG.warning(
                    "live puppet replay: local align %.2fs, then replay %.2fs",
                    duration,
                    replay_duration,
                )
                require_accepted("Mode 1", robot.request_mode(POLICY_PUPPET))
                current = robot.observe()
                if current.effective_mode != POLICY_PUPPET:
                    raise BridgeError(f"Mode 1 did not become active: {current.control}")
                require_accepted("output enable", robot.set_output(True))
                for _ in range(20):
                    time.sleep(0.05)
                    current = robot.observe()
                    if bool(current.control.get("output_enabled", False)):
                        break
                else:
                    raise BridgeError("output_enabled feedback did not become true")
                chunk = np.concatenate((alignment[:-1], trajectory), axis=0)
                queued = client.set_action_chunk(
                    args.side,
                    chunk.tolist(),
                    epoch=current.epoch,
                    sequence=time.time_ns(),
                    waypoint_hz=playback_hz,
                )
                require_accepted("Mode 1 chunk", queued)
                deadline = time.monotonic() + float(queued["duration_s"]) + 0.25
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    current = robot.observe()
                    if current.teach_active or current.effective_mode != POLICY_PUPPET:
                        raise BridgeError(f"puppet replay interrupted: {current.control}")
                print(json.dumps({"pc_alignment_s": duration, "chunk": queued}, indent=2))
            else:
                divergence = float(
                    np.max(np.abs(initial.master_state[:6] - initial.puppet_state[:6]))
                )
                LOG.warning(
                    "live sync replay: divergence %.4frad, clipped replay %.2fs, loop=%s",
                    divergence,
                    replay_duration,
                    args.loop,
                )
                current, alignment_result, auxiliary_holds = establish_sync_mode(
                    client,
                    robot,
                    trajectory[0],
                    alignment_timeout=args.alignment_timeout,
                    initial_request=True,
                )
                transition_duration, transition = alignment_trajectory(
                    trajectory[-1],
                    trajectory[0],
                    control_hz=playback_hz,
                    max_speed=args.loop_transition_speed * args.speed_scale,
                    minimum_seconds=args.loop_transition_min_seconds / args.speed_scale,
                )
                LOG.warning(
                    "loop transition prepared: %.2fs, then %.2fs clipped replay",
                    transition_duration,
                    replay_duration,
                )
                cycle = 0
                restart_after_teach = True
                while True:
                    chunk = (
                        trajectory
                        if restart_after_teach
                        else np.concatenate((transition[:-1], trajectory), axis=0)
                    )
                    queued, hold_chunks = queue_sync_chunks(
                        client,
                        selected_side=args.side,
                        chunk=chunk,
                        auxiliary_holds=auxiliary_holds,
                        epoch=current.epoch,
                        waypoint_hz=playback_hz,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "cycle_started",
                                "cycle": cycle,
                                "after_teach": restart_after_teach and cycle > 0,
                                "robot_alignment": alignment_result,
                                "chunk": queued,
                                "hold_chunks": hold_chunks,
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                    interrupted, current = monitor_sync_chunk(
                        robot, float(queued["duration_s"])
                    )
                    if interrupted:
                        LOG.warning("human intervention detected during cycle %d", cycle)
                        if current.effective_mode != RESYNC_REQUIRED:
                            wait_for_teach_release(robot)
                        LOG.warning("Teach released; shutting down policy episode")
                        break

                    print(
                        json.dumps({"event": "cycle_complete", "cycle": cycle}),
                        flush=True,
                    )
                    cycle += 1
                    if not args.loop:
                        break
                    restart_after_teach = False
                    alignment_result = None
        except TeachEpisodeComplete as exc:
            LOG.warning("%s; shutting down policy episode", exc)
        except KeyboardInterrupt:
            LOG.warning("replay stopped by operator")
        finally:
            try:
                robot.set_output(False)
            finally:
                robot.request_mode(SAFE)


if __name__ == "__main__":
    main()
