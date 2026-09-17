"""Run a Cobot policy with fixed-rate control and overlapping chunk ensembling.

Inference runs in a background thread. Each prediction is timestamped at its
observation and contributes the action corresponding to the current control
time. Overlapping absolute-position predictions are exponentially ensembled,
then lightly low-pass filtered before the existing AGX-side 200 Hz safety
limiter receives them.
"""

from __future__ import annotations

import json
import logging
import math
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


LOG = logging.getLogger("cobot_bridge.run_lerobot_policy_smooth")
_DIAGNOSTIC_LOCK = threading.Lock()


@dataclass(frozen=True)
class TimedActionChunk:
    sequence: int
    observed_at: float
    ready_at: float
    actions: np.ndarray


@dataclass(frozen=True)
class SmoothRunnerProfile:
    name: str = "Smooth V2"
    target_filter_tau: float = 0.05
    consistency_scale: float = 0.0
    chunk_smoothing_passes: int = 0
    max_joint_velocity: float = 0.0
    max_joint_acceleration: float = 0.0
    max_joint_jerk: float = 0.0
    max_gripper_velocity: float = 0.0
    max_gripper_acceleration: float = 0.0
    max_gripper_jerk: float = 0.0
    target_tracking_omega: float = 6.0


V2_PROFILE = SmoothRunnerProfile()


def smooth_action_chunk(actions: np.ndarray, *, passes: int) -> np.ndarray:
    """Denoise arm waypoints without changing the gripper trajectory."""

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (7,) or not np.isfinite(values).all():
        raise ValueError(f"invalid action chunk: {values.shape}")
    if passes < 0:
        raise ValueError("chunk smoothing passes must be non-negative")
    result = values.copy()
    for _ in range(passes):
        padded = np.pad(result[:, :6], ((1, 1), (0, 0)), mode="edge")
        result[:, :6] = (
            0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        )
    return result


class TemporalEnsembleBuffer:
    """Thread-safe, time-aligned ensemble of overlapping action chunks."""

    def __init__(
        self, *, action_hz: float, decay: float, consistency_scale: float = 0.0
    ) -> None:
        if not math.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("ensemble decay must be non-negative")
        if not math.isfinite(consistency_scale) or consistency_scale < 0:
            raise ValueError("consistency scale must be non-negative")
        self.action_hz = float(action_hz)
        self.decay = float(decay)
        self.consistency_scale = float(consistency_scale)
        self._chunks: list[TimedActionChunk] = []
        self._lock = threading.Lock()
        self._latest_ready_at: float | None = None

    @property
    def latest_ready_at(self) -> float | None:
        with self._lock:
            return self._latest_ready_at

    def add(self, chunk: TimedActionChunk) -> None:
        actions = np.asarray(chunk.actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1:] != (7,) or not np.isfinite(actions).all():
            raise ValueError(f"invalid action chunk: {actions.shape}")
        stored = TimedActionChunk(
            sequence=int(chunk.sequence),
            observed_at=float(chunk.observed_at),
            ready_at=float(chunk.ready_at),
            actions=actions.copy(),
        )
        with self._lock:
            self._chunks.append(stored)
            self._chunks.sort(key=lambda item: item.observed_at)
            self._latest_ready_at = stored.ready_at

    def action(self, now: float) -> tuple[np.ndarray | None, dict[str, Any]]:
        now = float(now)
        proposals: list[np.ndarray] = []
        weights: list[float] = []
        sequences: list[int] = []
        phases: list[float] = []
        ages: list[float] = []
        retained: list[TimedActionChunk] = []
        with self._lock:
            for chunk in self._chunks:
                phase = max(0.0, (now - chunk.observed_at) * self.action_hz)
                if phase > len(chunk.actions) - 1:
                    continue
                retained.append(chunk)
                lower = min(int(math.floor(phase)), len(chunk.actions) - 1)
                upper = min(lower + 1, len(chunk.actions) - 1)
                fraction = phase - lower
                proposal = (
                    (1.0 - fraction) * chunk.actions[lower]
                    + fraction * chunk.actions[upper]
                )
                age = max(0.0, now - chunk.observed_at)
                proposals.append(proposal)
                weights.append(math.exp(-self.decay * age))
                sequences.append(chunk.sequence)
                phases.append(phase)
                ages.append(age)
            self._chunks = retained

        if not proposals:
            return None, {"active_chunks": 0, "sequences": []}
        values = np.stack(proposals)
        age_weights = np.asarray(weights, dtype=np.float64)
        consistency_weights = np.ones_like(age_weights)
        residuals = np.zeros_like(age_weights)
        if self.consistency_scale > 0 and len(values) > 1:
            robust_center = np.median(values[:, :6], axis=0)
            residuals = np.linalg.norm(values[:, :6] - robust_center[None, :], axis=1)
            consistency_weights = 1.0 / (
                1.0 + np.square(residuals / self.consistency_scale)
            )
        normalized = age_weights * consistency_weights
        normalized /= normalized.sum()
        result = np.sum(values * normalized[:, None], axis=0)
        dispersion = np.linalg.norm(values[:, :6] - result[None, :6], axis=1)
        return result, {
            "active_chunks": len(proposals),
            "sequences": sequences,
            "max_joint_dispersion": float(dispersion.max(initial=0.0)),
            "proposals": [
                {
                    "sequence": sequence,
                    "phase": phase,
                    "age_s": age,
                    "weight": float(weight),
                    "age_weight": float(age_weight),
                    "consistency_weight": float(consistency_weight),
                    "consistency_residual": float(residual),
                    "target": proposal.tolist(),
                }
                for sequence, phase, age, weight, age_weight, consistency_weight, residual, proposal in zip(
                    sequences,
                    phases,
                    ages,
                    normalized,
                    age_weights,
                    consistency_weights,
                    residuals,
                    values,
                    strict=True,
                )
            ],
        }


class TargetLowPass:
    def __init__(self, initial: np.ndarray, *, tau: float, now: float | None = None) -> None:
        values = np.asarray(initial, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("invalid initial target")
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("target filter tau must be non-negative")
        self.value = values.copy()
        self.tau = float(tau)
        self.last_time = None if now is None else float(now)

    def update(self, target: np.ndarray, now: float) -> np.ndarray:
        values = np.asarray(target, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("invalid target")
        now = float(now)
        if self.last_time is None or self.tau == 0:
            alpha = 1.0
        else:
            dt = max(0.0, now - self.last_time)
            alpha = 1.0 - math.exp(-dt / self.tau)
        self.value += alpha * (values - self.value)
        self.last_time = now
        return self.value.copy()


class JerkLimitedTargetFilter:
    """Track an absolute target while bounding velocity, acceleration and jerk."""

    def __init__(
        self,
        initial: np.ndarray,
        *,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        now: float,
        tracking_omega: float = 6.0,
    ) -> None:
        values = np.asarray(initial, dtype=np.float64)
        self.max_velocity = self._limits(max_velocity, "velocity")
        self.max_acceleration = self._limits(max_acceleration, "acceleration")
        self.max_jerk = self._limits(max_jerk, "jerk")
        if not math.isfinite(tracking_omega) or tracking_omega <= 0:
            raise ValueError("invalid target tracking omega")
        self.tracking_omega = float(tracking_omega)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("invalid initial target")
        self.position = values.copy()
        self.velocity = np.zeros(7, dtype=np.float64)
        self.acceleration = np.zeros(7, dtype=np.float64)
        self.last_time = float(now)

    @staticmethod
    def _limits(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (7,) or not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError(f"invalid target {name} limits")
        return result

    def update(self, target: np.ndarray, now: float) -> np.ndarray:
        values = np.asarray(target, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("invalid target")
        now = float(now)
        dt = now - self.last_time
        if not math.isfinite(dt) or dt <= 0:
            return self.position.copy()
        if dt > 0.25:
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            dt = min(dt, 1.0 / 30.0)

        omega = self.tracking_omega
        desired_jerk = (
            omega**3 * (values - self.position)
            - 3.0 * omega**2 * self.velocity
            - 3.0 * omega * self.acceleration
        )
        limited_jerk = np.clip(desired_jerk, -self.max_jerk, self.max_jerk)
        next_position = (
            self.position
            + self.velocity * dt
            + 0.5 * self.acceleration * dt**2
            + (1.0 / 6.0) * limited_jerk * dt**3
        )
        next_velocity = np.clip(
            self.velocity
            + self.acceleration * dt
            + 0.5 * limited_jerk * dt**2,
            -self.max_velocity,
            self.max_velocity,
        )
        next_acceleration = np.clip(
            self.acceleration + limited_jerk * dt,
            -self.max_acceleration,
            self.max_acceleration,
        )

        self.position = next_position
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        self.last_time = now
        return self.position.copy()


def _write_diagnostic(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    with _DIAGNOSTIC_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)


def main(profile: SmoothRunnerProfile = V2_PROFILE) -> None:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--inference-hz", type=float, default=5.0)
    parser.add_argument("--ensemble-decay", type=float, default=2.0)
    parser.add_argument("--target-filter-tau", type=float, default=profile.target_filter_tau)
    parser.add_argument("--consistency-scale", type=float, default=profile.consistency_scale)
    parser.add_argument(
        "--chunk-smoothing-passes", type=int, default=profile.chunk_smoothing_passes
    )
    parser.add_argument(
        "--target-max-joint-velocity", type=float, default=profile.max_joint_velocity
    )
    parser.add_argument(
        "--target-max-joint-acceleration",
        type=float,
        default=profile.max_joint_acceleration,
    )
    parser.add_argument(
        "--target-max-joint-jerk", type=float, default=profile.max_joint_jerk
    )
    parser.add_argument(
        "--target-max-gripper-velocity",
        type=float,
        default=profile.max_gripper_velocity,
    )
    parser.add_argument(
        "--target-max-gripper-acceleration",
        type=float,
        default=profile.max_gripper_acceleration,
    )
    parser.add_argument(
        "--target-max-gripper-jerk", type=float, default=profile.max_gripper_jerk
    )
    parser.add_argument(
        "--target-tracking-omega", type=float, default=profile.target_tracking_omega
    )
    parser.add_argument("--first-prediction-timeout", type=float, default=10.0)
    parser.add_argument("--max-prediction-gap", type=float, default=0.75)
    parser.add_argument("--diagnostic-output", type=Path)
    args = parser.parse_args()
    if not args.policy_server_endpoint:
        parser.error(f"{profile.name} requires --policy-server-endpoint")
    if args.inference_hz <= 0 or args.inference_hz > args.fps:
        parser.error("--inference-hz must be in (0, fps]")
    if args.first_prediction_timeout <= 0 or args.max_prediction_gap <= 0:
        parser.error("prediction timeouts must be positive")
    if args.consistency_scale < 0 or args.chunk_smoothing_passes < 0:
        parser.error("consistency scale and chunk smoothing passes must be non-negative")
    if args.target_tracking_omega <= 0:
        parser.error("target tracking omega must be positive")
    target_limits = (
        args.target_max_joint_velocity,
        args.target_max_joint_acceleration,
        args.target_max_joint_jerk,
        args.target_max_gripper_velocity,
        args.target_max_gripper_acceleration,
        args.target_max_gripper_jerk,
    )
    jerk_filter_enabled = any(value > 0 for value in target_limits)
    if jerk_filter_enabled and not all(value > 0 for value in target_limits):
        parser.error("all target velocity, acceleration and jerk limits must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    control_period = 1.0 / args.fps
    inference_period = 1.0 / args.inference_hz
    buffer = TemporalEnsembleBuffer(
        action_hz=args.fps,
        decay=args.ensemble_decay,
        consistency_scale=args.consistency_scale,
    )
    stop_event = threading.Event()
    first_prediction = threading.Event()
    worker_errors: list[BaseException] = []
    inference_stats: list[float] = []

    def inference_worker() -> None:
        sequence = 0
        policy_client = PersistentPolicyClient(
            args.policy_server_endpoint, timeout_s=args.policy_server_timeout
        )
        try:
            policy_client.reset()
            with CobotBridgeClient(args.endpoint, timeout_s=5.0) as bridge:
                robot = CobotMagicRemoteRobot(
                    bridge, side=args.side, images=tuple(args.cameras)
                )
                while not stop_event.is_set():
                    cycle_started = time.monotonic()
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
                            f"smooth inference observation not ready: "
                            f"{observation.raw.get('missing_or_stale')}"
                        )
                    started = time.monotonic()
                    actions = policy_client.predict_chunk(
                        state=observation.puppet_state,
                        images=observation.image_bytes,
                        cameras=tuple(args.cameras),
                        task=args.task,
                    )
                    latency = time.monotonic() - started
                    count = resolve_action_steps(args.n_action_steps, len(actions))
                    ready_at = time.monotonic()
                    raw_chunk = np.asarray(actions[:count], dtype=np.float64)
                    processed_chunk = smooth_action_chunk(
                        raw_chunk, passes=args.chunk_smoothing_passes
                    )
                    buffer.add(
                        TimedActionChunk(
                            sequence=sequence,
                            observed_at=observed_at,
                            ready_at=ready_at,
                            actions=processed_chunk,
                        )
                    )
                    inference_stats.append(latency)
                    _write_diagnostic(
                        args.diagnostic_output,
                        {
                            "event": "prediction",
                            "sequence": sequence,
                            "bridge_time_ns": int(
                                observation.raw.get("bridge_time_ns", 0)
                            ),
                            "observation_sequence": int(
                                observation.raw.get("sequence", -1)
                            ),
                            "observed_at": observed_at,
                            "ready_at": ready_at,
                            "latency_ms": latency * 1000.0,
                            "input_state": observation.puppet_state.astype(
                                np.float64
                            ).tolist(),
                            "profile": profile.name,
                            "raw_chunk": raw_chunk.tolist(),
                            "processed_chunk": processed_chunk.tolist(),
                            "image_stamps_ns": {
                                name: int((observation.raw.get("images", {}).get(name) or {}).get("stamp_ns", 0))
                                for name in args.cameras
                            },
                        },
                    )
                    sequence += 1
                    first_prediction.set()
                    stop_event.wait(max(0.0, inference_period - (time.monotonic() - cycle_started)))
        except BaseException as exc:
            worker_errors.append(exc)
            stop_event.set()
            first_prediction.set()
        finally:
            policy_client.close()

    sent = 0
    ticks = 0
    holds = 0
    control_started = False
    worker: threading.Thread | None = None
    with CobotBridgeClient(args.endpoint, timeout_s=5.0) as client:
        robot = CobotMagicRemoteRobot(client, side=args.side, images=())
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
                target=inference_worker, name="smooth-policy-inference", daemon=True
            )
            worker.start()
            startup_target = INIT_POSE.copy() if args.startup_init_hold else initial.puppet_state.copy()
            prediction_deadline = time.monotonic() + args.first_prediction_timeout
            while not first_prediction.wait(timeout=control_period):
                if time.monotonic() >= prediction_deadline:
                    raise RuntimeError("timed out waiting for first smooth prediction")
                if args.send_actions:
                    observation = robot.observe()
                    for side, target in auxiliary_holds.items():
                        robot.send_action_for_side(side, target, observation=observation)
                    robot.send_action(startup_target, observation=observation)
            if worker_errors:
                raise worker_errors[0]

            observation = robot.observe()
            target_filter = TargetLowPass(
                observation.puppet_state,
                tau=args.target_filter_tau,
                now=time.monotonic(),
            )
            target_jerk_filter = None
            if jerk_filter_enabled:
                max_velocity = np.asarray(
                    [args.target_max_joint_velocity] * 6
                    + [args.target_max_gripper_velocity],
                    dtype=np.float64,
                )
                max_acceleration = np.asarray(
                    [args.target_max_joint_acceleration] * 6
                    + [args.target_max_gripper_acceleration],
                    dtype=np.float64,
                )
                max_jerk = np.asarray(
                    [args.target_max_joint_jerk] * 6
                    + [args.target_max_gripper_jerk],
                    dtype=np.float64,
                )
                target_jerk_filter = JerkLimitedTargetFilter(
                    observation.puppet_state,
                    max_velocity=max_velocity,
                    max_acceleration=max_acceleration,
                    max_jerk=max_jerk,
                    now=time.monotonic(),
                    tracking_omega=args.target_tracking_omega,
                )
            started_at = time.monotonic()
            deadline = started_at + args.duration
            last_target = observation.puppet_state.astype(np.float64, copy=True)
            while time.monotonic() < deadline and not stop_event.is_set():
                tick_started = time.monotonic()
                observation = robot.observe()
                if args.send_actions:
                    expected_mode = POLICY_PUPPET if args.mode == 1 else RLT_SYNC
                    if observation.teach_active or observation.effective_mode in (
                        TEACH,
                        RESYNC_REQUIRED,
                    ):
                        LOG.warning("Teach intervention ended %s rollout", profile.name)
                        break
                    if observation.effective_mode != expected_mode:
                        raise RuntimeError(
                            f"{profile.name} lost control ownership: {observation.control}"
                        )

                now = time.monotonic()
                action, ensemble = buffer.action(now)
                if action is None:
                    holds += 1
                    action = last_target
                else:
                    last_target = action
                if (
                    buffer.latest_ready_at is None
                    or now - buffer.latest_ready_at > args.max_prediction_gap
                ):
                    raise RuntimeError(f"{profile.name} prediction stream became stale")
                low_pass_target = target_filter.update(action, now)
                sent_target = (
                    low_pass_target
                    if target_jerk_filter is None
                    else target_jerk_filter.update(low_pass_target, now)
                )
                ticks += 1
                send_result: dict[str, Any] | None = None
                puppet_sample = (
                    observation.raw.get("puppet", {}).get(args.side) or {}
                )
                command_sample = (
                    observation.raw.get("commands", {})
                    .get("puppet", {})
                    .get(args.side)
                    or {}
                )
                applied_command = observation.executed_puppet_action()
                if args.send_actions:
                    for side, target in auxiliary_holds.items():
                        result = robot.send_action_for_side(
                            side, target, observation=observation
                        )
                        if not result.get("accepted", False):
                            raise RuntimeError(f"{side} hold rejected: {result}")
                    send_result = robot.send_action(sent_target, observation=observation)
                    if not send_result.get("accepted", False):
                        raise RuntimeError(f"{profile.name} action rejected: {send_result}")
                    sent += 1
                if ticks == 1 or ticks % max(1, int(args.fps)) == 0:
                    LOG.info(
                        "smooth tick=%d active_chunks=%d dispersion=%.4f holds=%d",
                        ticks,
                        ensemble.get("active_chunks", 0),
                        ensemble.get("max_joint_dispersion", 0.0),
                        holds,
                    )
                _write_diagnostic(
                    args.diagnostic_output,
                    {
                        "event": "action",
                        "profile": profile.name,
                        "tick": ticks,
                        "elapsed_s": now - started_at,
                        "monotonic_time": now,
                        "wall_time_ns": time.time_ns(),
                        "bridge_time_ns": int(observation.raw.get("bridge_time_ns", 0)),
                        "observation_sequence": int(observation.raw.get("sequence", -1)),
                        "effective_mode": observation.effective_mode,
                        "epoch": observation.epoch,
                        "active_chunks": ensemble.get("active_chunks", 0),
                        "dispersion": ensemble.get("max_joint_dispersion", 0.0),
                        "proposals": ensemble.get("proposals", []),
                        "ensemble_target": action.tolist(),
                        "low_pass_target": low_pass_target.tolist(),
                        "command_target": sent_target.tolist(),
                        "sent_target": sent_target.tolist() if args.send_actions else None,
                        "measured_state": observation.puppet_state.astype(
                            np.float64
                        ).tolist(),
                        "measured_state_stamp_ns": int(
                            puppet_sample.get("stamp_ns", 0)
                        ),
                        "measured_state_age_ms": float(
                            puppet_sample["age_ms"]
                        ) if "age_ms" in puppet_sample else None,
                        "applied_command": (
                            None
                            if applied_command is None
                            else applied_command.astype(np.float64).tolist()
                        ),
                        "applied_command_stamp_ns": int(
                            command_sample.get("stamp_ns", 0)
                        ),
                        "applied_command_age_ms": float(
                            command_sample["age_ms"]
                        ) if "age_ms" in command_sample else None,
                        "send_result": send_result,
                    },
                )
                delay = control_period - (time.monotonic() - tick_started)
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

    median_latency = (
        float(np.median(inference_stats) * 1000.0) if inference_stats else None
    )
    LOG.info(
        "%s complete: sent=%d holds=%d predictions=%d median_inference_ms=%s",
        profile.name,
        sent,
        holds,
        len(inference_stats),
        median_latency,
    )


if __name__ == "__main__":
    main()
