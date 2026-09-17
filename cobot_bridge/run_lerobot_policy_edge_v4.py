"""Run time-aligned IL action chunks with asynchronous edge-only transitions.

V4 preserves every non-overlapping 30 FPS waypoint inside a policy chunk. A
new prediction is requested only near the current chunk boundary. Its actions
that refer to time already executed by the old chunk are skipped, and only the
adaptive edge window receives a decaying offset. The PC-side policy-rate
limiter enforces velocity and acceleration before each waypoint is sent. The
AGX bridge only interpolates consecutive waypoints on its 200 Hz scheduler.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .client import CobotBridgeClient
from .policy_client import PersistentPolicyClient
from .remote_robot import (
    POLICY_PUPPET,
    RESYNC_REQUIRED,
    RLT_SYNC,
    SAFE,
    TEACH,
    CobotMagicRemoteRobot,
)
from .run_lerobot_policy import INIT_POSE, _hold_startup_init, build_parser, resolve_action_steps
from .run_lerobot_policy_smooth import _write_diagnostic
from .synchronized_chunk_replanner import (
    DEFAULT_POLICY_ACCELERATION_LIMITS,
    DEFAULT_POLICY_VELOCITY_LIMITS,
    PolicyRateCommandLimiter,
)


LOG = logging.getLogger("cobot_bridge.run_lerobot_policy_edge_v4")
PROFILE_NAME = "Edge V4"


@dataclass(frozen=True)
class PredictedChunk:
    sequence: int
    observed_at: float
    ready_at: float
    latency_ms: float
    input_state: np.ndarray
    actions: np.ndarray


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def edge_blend_action(
    previous: np.ndarray,
    edge_start: np.ndarray,
    current: np.ndarray,
    *,
    step: int,
    blend_steps: int,
) -> tuple[np.ndarray, float]:
    """Blend only the first ``blend_steps`` actions of a fresh chunk."""

    previous = np.asarray(previous, dtype=np.float64)
    edge_start = np.asarray(edge_start, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != (7,) or edge_start.shape != (7,) or current.shape != (7,):
        raise ValueError("edge actions must have shape (7,)")
    if blend_steps < 0 or step < 0:
        raise ValueError("edge step values must be non-negative")
    if blend_steps == 0 or step >= blend_steps:
        return current.copy(), 1.0
    if blend_steps == 1:
        return current.copy(), 1.0
    alpha = smoothstep(step / (blend_steps - 1))
    # Decay a constant arm-joint edge offset instead of scaling the new
    # trajectory. This preserves local deltas and leaves gripper timing raw.
    blended = current.copy()
    blended[:6] += (1.0 - alpha) * (previous[:6] - edge_start[:6])
    return blended, alpha


def adaptive_edge_blend_steps(
    previous: np.ndarray,
    edge_start: np.ndarray,
    *,
    min_steps: int,
    max_steps: int,
    max_joint_step: float,
    action_hz: float | None = None,
    velocity_limits: np.ndarray | None = None,
    acceleration_limits: np.ndarray | None = None,
) -> tuple[int, float]:
    """Choose an edge window from gap size and optional v/a limits.

    The old V4 ``min_steps..max_steps`` behavior remains the fallback.  When
    limits are supplied, the window also grows with the number of policy ticks
    needed by the largest joint gap, so the overlap is not a fixed three-frame
    patch.
    """

    previous = np.asarray(previous, dtype=np.float64)
    edge_start = np.asarray(edge_start, dtype=np.float64)
    if previous.shape != (7,) or edge_start.shape != (7,):
        raise ValueError("edge actions must have shape (7,)")
    if min_steps < 0 or max_steps < min_steps:
        raise ValueError("invalid adaptive edge blend range")
    if max_joint_step <= 0:
        raise ValueError("max joint step must be positive")
    if action_hz is not None and action_hz <= 0:
        raise ValueError("action_hz must be positive")
    gap = float(np.max(np.abs(previous[:6] - edge_start[:6])))
    required = 0 if gap == 0.0 else int(np.ceil(gap / max_joint_step)) + 1
    if action_hz is not None:
        velocity_limits = (
            DEFAULT_POLICY_VELOCITY_LIMITS
            if velocity_limits is None
            else np.asarray(velocity_limits, dtype=np.float64)
        )
        acceleration_limits = (
            DEFAULT_POLICY_ACCELERATION_LIMITS
            if acceleration_limits is None
            else np.asarray(acceleration_limits, dtype=np.float64)
        )
        if velocity_limits.shape != (7,) or acceleration_limits.shape != (7,):
            raise ValueError("kinematic limits must have shape (7,)")
        arm_gap = np.abs(previous[:6] - edge_start[:6])
        dt = 1.0 / float(action_hz)
        # Smoothstep has peak slope 1.5/T and peak acceleration 6/T^2.
        required_velocity = int(
            np.max(np.ceil(1.5 * arm_gap / (velocity_limits[:6] * dt)))
        ) + 1
        required_acceleration = int(
            np.max(
                np.ceil(
                    np.sqrt(6.0 * arm_gap / (acceleration_limits[:6] * dt**2))
                )
            )
        ) + 1
        required = max(required, required_velocity, required_acceleration)
    return min(max(required, min_steps), max_steps), gap


def time_aligned_start_step(
    observed_at: float,
    playback_at: float,
    *,
    fps: float,
    chunk_length: int,
) -> int:
    """Return the first chunk index whose timestamp has not already passed."""

    if fps <= 0 or chunk_length < 1:
        raise ValueError("fps and chunk length must be positive")
    elapsed_steps = int(round(max(0.0, playback_at - observed_at) * fps))
    if elapsed_steps >= chunk_length:
        raise RuntimeError("prefetched Edge V4 chunk expired before playback")
    return elapsed_steps


def prefetch_trigger_step(chunk_length: int, prefetch_steps: int) -> int:
    if chunk_length < 1 or prefetch_steps < 1:
        raise ValueError("chunk length and prefetch steps must be positive")
    return max(0, chunk_length - prefetch_steps)


def main() -> None:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--edge-blend-steps",
        type=int,
        default=3,
        help="minimum number of new-chunk arm actions modified at each edge",
    )
    parser.add_argument(
        "--max-edge-blend-steps",
        type=int,
        default=30,
        help="maximum adaptive edge blend window; grows when the gap needs more policy ticks",
    )
    parser.add_argument(
        "--edge-max-joint-step",
        type=float,
        default=0.08,
        help="arm-joint gap in radians used to grow the edge window",
    )
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=7,
        help="request the next chunk this many playback steps before the edge",
    )
    parser.add_argument("--first-prediction-timeout", type=float, default=10.0)
    parser.add_argument("--diagnostic-output", type=Path)
    args = parser.parse_args()
    if not args.policy_server_endpoint:
        parser.error("Edge V4 requires --policy-server-endpoint")
    if (
        args.edge_blend_steps < 0
        or args.max_edge_blend_steps < args.edge_blend_steps
        or args.edge_max_joint_step <= 0
        or args.prefetch_steps < 1
    ):
        parser.error("invalid edge blend or prefetch configuration")
    if args.first_prediction_timeout <= 0:
        parser.error("first prediction timeout must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    period = 1.0 / args.fps
    request_queue: queue.Queue[int] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[PredictedChunk] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []
    prediction_latencies: list[float] = []

    def inference_worker() -> None:
        policy = PersistentPolicyClient(
            args.policy_server_endpoint, timeout_s=args.policy_server_timeout
        )
        try:
            policy.reset()
            with CobotBridgeClient(args.endpoint, timeout_s=5.0) as bridge:
                robot = CobotMagicRemoteRobot(
                    bridge, side=args.side, images=tuple(args.cameras)
                )
                while not stop_event.is_set():
                    try:
                        sequence = request_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    observation = robot.observe()
                    observed_at = time.monotonic()
                    if observation.teach_active or observation.effective_mode in (
                        TEACH,
                        RESYNC_REQUIRED,
                    ):
                        stop_event.set()
                        break
                    if not observation.ready:
                        raise RuntimeError(
                            f"V4 inference observation not ready: "
                            f"{observation.raw.get('missing_or_stale')}"
                        )
                    started = time.monotonic()
                    actions = policy.predict_chunk(
                        state=observation.puppet_state,
                        images=observation.image_bytes,
                        cameras=tuple(args.cameras),
                        task=args.task,
                    )
                    latency_ms = (time.monotonic() - started) * 1000.0
                    count = resolve_action_steps(args.n_action_steps, len(actions))
                    ready_at = time.monotonic()
                    chunk = PredictedChunk(
                        sequence=sequence,
                        observed_at=observed_at,
                        ready_at=ready_at,
                        latency_ms=latency_ms,
                        input_state=observation.puppet_state.astype(
                            np.float64, copy=True
                        ),
                        actions=np.asarray(actions[:count], dtype=np.float64).copy(),
                    )
                    prediction_latencies.append(latency_ms)
                    _write_diagnostic(
                        args.diagnostic_output,
                        {
                            "event": "prediction",
                            "profile": PROFILE_NAME,
                            "sequence": sequence,
                            "observed_at": observed_at,
                            "ready_at": ready_at,
                            "latency_ms": latency_ms,
                            "input_state": chunk.input_state.tolist(),
                            "raw_chunk": chunk.actions.tolist(),
                        },
                    )
                    result_queue.put(chunk)
        except BaseException as exc:
            worker_errors.append(exc)
            stop_event.set()
        finally:
            policy.close()

    sent = 0
    ticks = 0
    boundary_holds = 0
    time_aligned_skips = 0
    sequence = 0
    pending = False
    worker: threading.Thread | None = None
    control_started = False
    with CobotBridgeClient(args.endpoint, timeout_s=5.0) as bridge:
        robot = CobotMagicRemoteRobot(bridge, side=args.side, images=())
        initial = robot.observe()
        if not initial.ready or initial.teach_active:
            raise RuntimeError("robot is not ready or Teach is active")
        auxiliary_holds: dict[str, np.ndarray] = {}
        try:
            if args.send_actions:
                result = robot.request_mode(args.mode)
                if not result.get("accepted", False):
                    raise RuntimeError(f"mode request rejected: {result}")
                result = robot.set_output(True)
                if not result.get("accepted", False):
                    raise RuntimeError(f"output enable rejected: {result}")
                control_started = True
                if args.startup_init_hold:
                    auxiliary_holds = _hold_startup_init(
                        robot, sync_sides=tuple(args.sync_sides)
                    )

            worker = threading.Thread(
                target=inference_worker, name="edge-v4-inference", daemon=True
            )
            worker.start()
            request_queue.put(sequence)
            sequence += 1
            pending = True

            current: PredictedChunk | None = None
            prefetched: PredictedChunk | None = None
            chunk_step = 0
            edge_step = 0
            active_blend_steps = 0
            aligned_start_step = 0
            edge_gap_max_joint = 0.0
            previous_target = (
                INIT_POSE.astype(np.float64, copy=True)
                if args.startup_init_hold
                else initial.puppet_state.astype(np.float64, copy=True)
            )
            last_command = previous_target.copy()
            edge_start_action = previous_target.copy()
            policy_limiter = PolicyRateCommandLimiter(
                action_hz=args.fps,
                velocity_limits=DEFAULT_POLICY_VELOCITY_LIMITS,
                acceleration_limits=DEFAULT_POLICY_ACCELERATION_LIMITS,
            )
            policy_limiter.reset(last_command)
            started_at = time.monotonic()
            deadline = started_at + args.duration
            first_deadline = started_at + args.first_prediction_timeout

            while time.monotonic() < deadline and not stop_event.is_set():
                tick_started = time.monotonic()
                observation = robot.observe()
                if args.send_actions:
                    expected_mode = POLICY_PUPPET if args.mode == 1 else RLT_SYNC
                    if observation.teach_active or observation.effective_mode in (
                        TEACH,
                        RESYNC_REQUIRED,
                    ):
                        LOG.warning("Teach intervention ended Edge V4 rollout")
                        break
                    if observation.effective_mode != expected_mode:
                        raise RuntimeError(
                            f"Edge V4 lost control ownership: {observation.control}"
                        )
                try:
                    ready_chunk = result_queue.get_nowait()
                except queue.Empty:
                    ready_chunk = None
                if ready_chunk is not None:
                    pending = False
                    if current is None:
                        current = ready_chunk
                        chunk_step = 0
                        edge_step = 0
                        aligned_start_step = 0
                        previous_target = last_command.copy()
                        edge_start_action = current.actions[0].copy()
                        active_blend_steps, edge_gap_max_joint = (
                            adaptive_edge_blend_steps(
                                previous_target,
                                edge_start_action,
                                min_steps=args.edge_blend_steps,
                                max_steps=args.max_edge_blend_steps,
                                max_joint_step=args.edge_max_joint_step,
                                action_hz=args.fps,
                                velocity_limits=DEFAULT_POLICY_VELOCITY_LIMITS,
                                acceleration_limits=DEFAULT_POLICY_ACCELERATION_LIMITS,
                            )
                        )
                    else:
                        prefetched = ready_chunk

                if current is None:
                    if time.monotonic() >= first_deadline:
                        raise RuntimeError("timed out waiting for first Edge V4 chunk")
                    command_target = last_command
                    raw_action = None
                    blend_alpha = 0.0
                    boundary_hold = True
                elif chunk_step >= len(current.actions):
                    if prefetched is None:
                        command_target = last_command
                        raw_action = None
                        blend_alpha = 0.0
                        boundary_hold = True
                        boundary_holds += 1
                    else:
                        previous_target = last_command.copy()
                        current = prefetched
                        prefetched = None
                        playback_at = time.monotonic()
                        chunk_step = time_aligned_start_step(
                            current.observed_at,
                            playback_at,
                            fps=args.fps,
                            chunk_length=len(current.actions),
                        )
                        aligned_start_step = chunk_step
                        edge_step = 0
                        edge_start_action = current.actions[chunk_step].copy()
                        time_aligned_skips += chunk_step
                        active_blend_steps, edge_gap_max_joint = (
                            adaptive_edge_blend_steps(
                                previous_target,
                                edge_start_action,
                                min_steps=args.edge_blend_steps,
                                max_steps=args.max_edge_blend_steps,
                                max_joint_step=args.edge_max_joint_step,
                                action_hz=args.fps,
                                velocity_limits=DEFAULT_POLICY_VELOCITY_LIMITS,
                                acceleration_limits=DEFAULT_POLICY_ACCELERATION_LIMITS,
                            )
                        )
                        raw_action = current.actions[chunk_step]
                        command_target, blend_alpha = edge_blend_action(
                            previous_target,
                            edge_start_action,
                            raw_action,
                            step=edge_step,
                            blend_steps=active_blend_steps,
                        )
                        boundary_hold = False
                else:
                    raw_action = current.actions[chunk_step]
                    command_target, blend_alpha = edge_blend_action(
                        previous_target,
                        edge_start_action,
                        raw_action,
                        step=edge_step,
                        blend_steps=active_blend_steps,
                    )
                    boundary_hold = False

                # All physical kinematic limiting happens at the policy
                # waypoint rate.  The bridge receives two consecutive points
                # below and only interpolates them at its 200 Hz timer.
                requested_target = command_target.copy()
                command_target = policy_limiter.step(requested_target)

                if (
                    current is not None
                    and chunk_step >= prefetch_trigger_step(
                        len(current.actions), args.prefetch_steps
                    )
                    and prefetched is None
                    and not pending
                ):
                    request_queue.put(sequence)
                    sequence += 1
                    pending = True

                send_result: dict[str, Any] | None = None
                if args.send_actions:
                    for side, target in auxiliary_holds.items():
                        result = robot.send_action_for_side(
                            side, target, observation=observation
                        )
                        if not result.get("accepted", False):
                            raise RuntimeError(f"{side} hold rejected: {result}")
                    send_result = robot.send_action_chunk(
                        [last_command, command_target],
                        waypoint_hz=args.fps,
                        observation=observation,
                    )
                    if not send_result.get("accepted", False):
                        raise RuntimeError(f"Edge V4 action rejected: {send_result}")
                    sent += 1
                ticks += 1
                last_command = command_target.copy()
                active_sequence = None if current is None else current.sequence
                active_step = None if current is None else chunk_step
                active_edge_step = None if raw_action is None else edge_step
                _write_diagnostic(
                    args.diagnostic_output,
                    {
                        "event": "action",
                        "profile": PROFILE_NAME,
                        "tick": ticks,
                        "elapsed_s": time.monotonic() - started_at,
                        "monotonic_time": time.monotonic(),
                        "chunk_sequence": active_sequence,
                        "chunk_step": active_step,
                        "aligned_start_step": aligned_start_step,
                        "chunk_observation_age_steps": (
                            None
                            if current is None
                            else int(
                                round(
                                    max(0.0, time.monotonic() - current.observed_at)
                                    * args.fps
                                )
                            )
                        ),
                        "edge_step": active_edge_step,
                        "edge_blend_steps": active_blend_steps,
                        "edge_gap_max_joint": edge_gap_max_joint,
                        "chunk_length": None if current is None else len(current.actions),
                        "edge_blend": bool(
                            raw_action is not None
                            and active_edge_step is not None
                            and active_edge_step < active_blend_steps
                        ),
                        "blend_alpha": blend_alpha,
                        "boundary_hold": boundary_hold,
                        "raw_action": None if raw_action is None else raw_action.tolist(),
                        "requested_target": requested_target.tolist(),
                        "command_target": command_target.tolist(),
                        "sent_target": command_target.tolist() if args.send_actions else None,
                        "policy_rate_hz": args.fps,
                        "policy_velocity_ratio": float(
                            np.max(
                                np.abs(policy_limiter.last_velocity)
                                / policy_limiter.velocity_limits
                            )
                        ),
                        "policy_acceleration_ratio": float(
                            np.max(
                                np.abs(policy_limiter.last_acceleration)
                                / policy_limiter.acceleration_limits
                            )
                        ),
                        "wire_interpolation_hz": 200.0,
                        "wire_waypoints": 2,
                        "measured_state": observation.puppet_state.astype(
                            np.float64
                        ).tolist(),
                        "applied_command": (
                            None
                            if observation.executed_puppet_action() is None
                            else observation.executed_puppet_action().astype(
                                np.float64
                            ).tolist()
                        ),
                        "send_result": send_result,
                    },
                )
                if raw_action is not None:
                    chunk_step += 1
                    edge_step += 1
                delay = period - (time.monotonic() - tick_started)
                if delay > 0:
                    stop_event.wait(delay)
            if worker_errors:
                raise worker_errors[0]
        finally:
            stop_event.set()
            if control_started:
                try:
                    robot.set_output(False)
                finally:
                    robot.request_mode(SAFE)
            if worker is not None:
                worker.join(timeout=max(2.0, args.policy_server_timeout + 1.0))

    LOG.info(
        "Edge V4 complete: ticks=%d sent=%d predictions=%d boundary_holds=%d "
        "time_aligned_skips=%d median_inference_ms=%s",
        ticks,
        sent,
        len(prediction_latencies),
        boundary_holds,
        time_aligned_skips,
        None if not prediction_latencies else float(np.median(prediction_latencies)),
    )


if __name__ == "__main__":
    main()
