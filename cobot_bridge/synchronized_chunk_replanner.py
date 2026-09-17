"""Rollout-side phase-synchronized retiming for absolute action chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


DEFAULT_POLICY_VELOCITY_LIMITS = np.asarray(
    [0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 0.12], dtype=np.float64
)
DEFAULT_POLICY_ACCELERATION_LIMITS = np.asarray(
    [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.4], dtype=np.float64
)


@dataclass(frozen=True)
class ReplanDiagnostics:
    time_scale: float
    edge_gap: float
    max_velocity_ratio: float
    max_acceleration_ratio: float
    transition_steps: int = 0
    output_steps: int = 0
    duration_s: float = 0.0


class PolicyRateCommandLimiter:
    """Apply common-rate kinematic limits at the policy/waypoint rate.

    This is deliberately a position-command limiter, not a robot-side servo.
    Each call produces one legal policy-rate waypoint.  The bridge can then
    linearly interpolate the pair of consecutive waypoints at 200 Hz without
    adding another velocity or acceleration policy.
    """

    def __init__(
        self,
        *,
        action_hz: float,
        velocity_limits: np.ndarray = DEFAULT_POLICY_VELOCITY_LIMITS,
        acceleration_limits: np.ndarray = DEFAULT_POLICY_ACCELERATION_LIMITS,
    ) -> None:
        if not math.isfinite(action_hz) or action_hz <= 0.0:
            raise ValueError("action_hz must be positive")
        self.action_hz = float(action_hz)
        self.dt = 1.0 / self.action_hz
        self.velocity_limits = self._limits(velocity_limits, "velocity")
        self.acceleration_limits = self._limits(
            acceleration_limits, "acceleration"
        )
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(7, dtype=np.float64)
        self.last_velocity = np.zeros(7, dtype=np.float64)
        self.last_acceleration = np.zeros(7, dtype=np.float64)

    @staticmethod
    def _limits(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (7,) or not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError(f"{name} limits must be finite positive shape-(7,) values")
        return result

    def reset(
        self,
        position: np.ndarray,
        *,
        velocity: np.ndarray | None = None,
    ) -> None:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (7,) or not np.isfinite(position).all():
            raise ValueError("position must be a finite shape-(7,) vector")
        self.position = position.copy()
        if velocity is None:
            self.velocity.fill(0.0)
        else:
            velocity = np.asarray(velocity, dtype=np.float64)
            if velocity.shape != (7,) or not np.isfinite(velocity).all():
                raise ValueError("velocity must be a finite shape-(7,) vector")
            self.velocity = np.clip(
                velocity, -self.velocity_limits, self.velocity_limits
            )
        self.last_velocity = self.velocity.copy()
        self.last_acceleration.fill(0.0)

    def step(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("target must be a finite shape-(7,) vector")
        if self.position is None:
            self.reset(target)
            return target.astype(np.float32)

        error = target - self.position
        # The stopping-speed envelope prevents a new edge target from being
        # approached too quickly to stop on the following policy tick.
        stopping_speed = np.sqrt(
            2.0 * self.acceleration_limits * np.abs(error)
        )
        desired_velocity = np.sign(error) * np.minimum(
            self.velocity_limits, stopping_speed
        )
        # Leave a small numerical margin because the wire command is later
        # represented as float32 before it reaches the bridge.
        max_delta_velocity = self.acceleration_limits * self.dt * 0.999
        next_velocity = np.clip(
            desired_velocity,
            self.velocity - max_delta_velocity,
            self.velocity + max_delta_velocity,
        )
        next_velocity = np.clip(
            next_velocity, -self.velocity_limits, self.velocity_limits
        )
        next_position = self.position + next_velocity * self.dt

        # A stop may require a tiny overshoot when the previous command
        # velocity cannot be reduced to zero within one policy tick.  Do not
        # snap to the target in that case: the snap would create an
        # acceleration impulse.  The stopping-speed envelope makes the
        # overshoot converge on subsequent ticks.
        command_velocity = (next_position - self.position) / self.dt
        command_acceleration = (command_velocity - self.last_velocity) / self.dt
        self.position = next_position
        self.velocity = command_velocity
        self.last_velocity = command_velocity
        self.last_acceleration = command_acceleration
        return next_position.astype(np.float32)


class SynchronizedChunkReplanner:
    """Retimes absolute 30 Hz action waypoints onto a variable 200 Hz path.

    Every segment has one shared duration for all seven dimensions. A segment
    starts at one policy tick and grows until its quintic trajectory satisfies
    all configured velocity/acceleration limits. The output length is therefore
    intentionally variable; there is no legacy fixed-length fallback.
    """

    def __init__(
        self,
        *,
        action_hz: float,
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        maximum_time_scale: float = 8.0,
    ) -> None:
        if not math.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if not math.isfinite(maximum_time_scale) or maximum_time_scale < 1:
            raise ValueError("maximum_time_scale must be at least one")
        self.action_hz = float(action_hz)
        self.dt = 1.0 / self.action_hz
        self.velocity_limits = self._limits(velocity_limits, "velocity")
        self.acceleration_limits = self._limits(acceleration_limits, "acceleration")
        self.maximum_time_scale = float(maximum_time_scale)
        self.velocity = np.zeros(7, dtype=np.float64)
        self.acceleration = np.zeros(7, dtype=np.float64)

    @staticmethod
    def _limits(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (7,) or not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError(f"{name} limits must be finite positive shape-(7,) values")
        return result

    def reset(self) -> None:
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)

    @staticmethod
    def _quintic_coefficients(
        start: np.ndarray,
        start_velocity: np.ndarray,
        start_acceleration: np.ndarray,
        end: np.ndarray,
        end_velocity: np.ndarray,
        end_acceleration: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Return per-axis quintic coefficients for a C2 trajectory.

        The returned array has shape ``(7, 6)`` and stores coefficients in
        ascending time order.  Keeping this in the rollout-side planner lets
        the robot-side 200 Hz scheduler remain a pure trajectory follower.
        """

        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("trajectory duration must be positive")
        t = float(duration)
        matrix = np.asarray(
            [
                [t**3, t**4, t**5],
                [3.0 * t**2, 4.0 * t**3, 5.0 * t**4],
                [6.0 * t, 12.0 * t**2, 20.0 * t**3],
            ],
            dtype=np.float64,
        )
        rhs = np.stack(
            [
                end - start - start_velocity * t - 0.5 * start_acceleration * t**2,
                end_velocity - start_velocity - start_acceleration * t,
                end_acceleration - start_acceleration,
            ],
            axis=0,
        )
        tail = np.linalg.solve(matrix, rhs).T
        return np.column_stack(
            [start, start_velocity, 0.5 * start_acceleration, tail]
        )

    @staticmethod
    def _evaluate_quintic(
        coefficients: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate position, velocity and acceleration for a quintic path."""

        t = np.asarray(times, dtype=np.float64)[:, None]
        c0, c1, c2, c3, c4, c5 = [coefficients[:, i][None, :] for i in range(6)]
        position = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
        velocity = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
        acceleration = (
            2.0 * c2
            + 6.0 * c3 * t
            + 12.0 * c4 * t**2
            + 20.0 * c5 * t**3
        )
        return position, velocity, acceleration

    def _fit_segment(
        self,
        start: np.ndarray,
        start_velocity: np.ndarray,
        start_acceleration: np.ndarray,
        end: np.ndarray,
        end_velocity: np.ndarray,
        end_acceleration: np.ndarray,
        *,
        output_hz: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Fit one dynamically timed C2 segment under v/a limits.

        The initial duration is one policy tick.  If the positional gap is
        too large, the duration is increased until both velocity and
        acceleration constraints are satisfied.  The duration is quantized
        to the 200 Hz output grid so the bridge can follow the result without
        doing a second physical limiter.
        """

        # The policy endpoint velocity is only a preference.  In particular,
        # a chunk may begin at rest while its first predicted waypoint already
        # implies a non-zero 30 Hz velocity.  Forcing that endpoint derivative
        # into a one-tick quintic can be physically impossible even though the
        # position itself is close.  Search the output grid and select the
        # closest *feasible* endpoint velocity independently per joint.  This
        # keeps one common segment duration for the whole 7D path while never
        # silently violating one joint's limits.
        minimum_steps = max(1, int(math.ceil(self.dt * output_hz - 1e-12)))
        maximum_steps = max(
            minimum_steps,
            int(math.floor(self.dt * self.maximum_time_scale * output_hz + 1e-12)),
        )
        desired_velocity = np.clip(
            np.asarray(end_velocity, dtype=np.float64),
            -self.velocity_limits,
            self.velocity_limits,
        )
        last_velocity_ratio = 0.0
        last_acceleration_ratio = 0.0

        for output_steps in range(minimum_steps, maximum_steps + 1):
            duration = output_steps / output_hz
            dense_times = np.linspace(0.0, duration, max(65, output_steps + 1))
            terminal_velocity = np.empty(7, dtype=np.float64)

            for axis in range(7):
                limit_v = float(self.velocity_limits[axis])
                limit_a = float(self.acceleration_limits[axis])
                if abs(desired_velocity[axis]) <= 1e-9:
                    # A stop/reversal waypoint is intentional.  Do not choose
                    # a non-zero substitute merely to make the current short
                    # duration pass; that would push the velocity into the
                    # following zero-distance segment.
                    candidates = np.asarray([0.0], dtype=np.float64)
                else:
                    candidates = np.unique(
                        np.concatenate(
                            (
                                np.asarray([desired_velocity[axis], 0.0]),
                                np.linspace(-limit_v, limit_v, 65),
                            )
                        )
                    )
                feasible: list[float] = []
                best_velocity_ratio = math.inf
                best_acceleration_ratio = math.inf
                for candidate in candidates:
                    coefficients = self._quintic_coefficients(
                        np.asarray([start[axis]]),
                        np.asarray([start_velocity[axis]]),
                        np.asarray([start_acceleration[axis]]),
                        np.asarray([end[axis]]),
                        np.asarray([candidate]),
                        np.asarray([end_acceleration[axis]]),
                        duration,
                    )
                    _, velocity, acceleration = self._evaluate_quintic(
                        coefficients, dense_times
                    )
                    velocity_ratio = float(np.max(np.abs(velocity[:, 0])) / limit_v)
                    acceleration_ratio = float(
                        np.max(np.abs(acceleration[:, 0])) / limit_a
                    )
                    best_velocity_ratio = min(best_velocity_ratio, velocity_ratio)
                    best_acceleration_ratio = min(
                        best_acceleration_ratio, acceleration_ratio
                    )
                    if velocity_ratio <= 1.0 + 1e-6 and acceleration_ratio <= 1.0 + 1e-6:
                        feasible.append(float(candidate))
                last_velocity_ratio = max(last_velocity_ratio, best_velocity_ratio)
                last_acceleration_ratio = max(
                    last_acceleration_ratio, best_acceleration_ratio
                )
                if not feasible:
                    break
                terminal_velocity[axis] = min(
                    feasible,
                    key=lambda value: abs(value - desired_velocity[axis]),
                )
            else:
                coefficients = self._quintic_coefficients(
                    start,
                    start_velocity,
                    start_acceleration,
                    end,
                    terminal_velocity,
                    end_acceleration,
                    duration,
                )
                _, dense_velocity, dense_acceleration = self._evaluate_quintic(
                    coefficients, dense_times
                )
                velocity_ratio = float(
                    np.max(np.abs(dense_velocity) / self.velocity_limits[None, :])
                )
                acceleration_ratio = float(
                    np.max(np.abs(dense_acceleration) / self.acceleration_limits[None, :])
                )
                if velocity_ratio <= 1.0 + 1e-6 and acceleration_ratio <= 1.0 + 1e-6:
                    wire_times = np.arange(output_steps + 1, dtype=np.float64) / output_hz
                    positions, velocities, accelerations = self._evaluate_quintic(
                        coefficients, wire_times
                    )
                    # Avoid accumulating a floating point endpoint error over
                    # many variable-duration segments.
                    positions[-1] = end
                    velocities[-1] = terminal_velocity
                    accelerations[-1] = end_acceleration
                    return positions, velocities, accelerations, duration

        raise RuntimeError(
            "trajectory segment cannot satisfy velocity/acceleration limits "
            f"within maximum_time_scale={self.maximum_time_scale:.2f}; "
            f"best_velocity_ratio={last_velocity_ratio:.3f}, "
            f"best_acceleration_ratio={last_acceleration_ratio:.3f}"
        )

    def plan_trajectory(
        self,
        current: np.ndarray,
        actions: np.ndarray,
        *,
        output_hz: float = 200.0,
    ) -> tuple[np.ndarray, ReplanDiagnostics]:
        """Create a variable-duration, 200 Hz wire trajectory from 30 Hz actions.

        ``actions`` remains the policy-time sequence.  Each adjacent pair is
        connected with a quintic segment whose duration starts at one 30 Hz
        tick and grows only when the per-axis velocity or acceleration limit
        would be violated.  Therefore a small chunk keeps its original
        timing while a large overlap gap automatically consumes more policy
        frames.  The returned trajectory is sampled at ``output_hz`` and is
        suitable for a robot-side position follower.

        This method intentionally does not apply jerk clipping.  The quintic
        path is C2-continuous, and retaining the old V4 timing behaviour avoids
        adding a second source of phase delay during contact.
        """

        current = np.asarray(current, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if current.shape != (7,) or actions.ndim != 2 or actions.shape[1:] != (7,):
            raise ValueError("current/actions must have shapes (7,) and (N, 7)")
        if len(actions) < 2 or not np.isfinite(current).all() or not np.isfinite(actions).all():
            raise ValueError("action chunk must contain at least two finite actions")
        if not math.isfinite(output_hz) or output_hz <= 0.0 or output_hz > 200.0:
            raise ValueError("output_hz must be in (0, 200]")

        waypoints = np.concatenate((current[None, :], actions), axis=0)
        # Estimate only the desired policy-time endpoint velocity.  Acceleration
        # is intentionally zero at each waypoint: it gives C2 continuity
        # between consecutive fitted segments and avoids injecting a clipped,
        # one-tick acceleration estimate at every action boundary.
        desired_velocity = np.diff(waypoints, axis=0) / self.dt
        segment_start_velocity = np.clip(
            self.velocity, -self.velocity_limits, self.velocity_limits
        )
        zero_acceleration = np.zeros(7, dtype=np.float64)

        output_segments: list[np.ndarray] = []
        segment_durations: list[float] = []
        max_velocity_ratio = 0.0
        max_acceleration_ratio = 0.0
        segment_start = current.copy()
        for index in range(len(actions)):
            # Stop at a repeated waypoint or a direction reversal instead of
            # carrying a forward endpoint velocity into a zero-distance
            # segment.  Otherwise a mathematically smooth curve would need to
            # stop at the same position while still moving, which is not
            # possible under a finite acceleration limit.
            segment_end_velocity = desired_velocity[index].copy()
            if index + 1 < len(actions):
                next_velocity = desired_velocity[index + 1]
                same_direction = desired_velocity[index] * next_velocity > 0.0
                segment_end_velocity = np.where(
                    same_direction, segment_end_velocity, 0.0
                )
            positions, velocities, accelerations, duration = self._fit_segment(
                segment_start,
                segment_start_velocity,
                zero_acceleration,
                actions[index],
                segment_end_velocity,
                zero_acceleration,
                output_hz=output_hz,
            )
            output_segments.append(positions[1:])
            segment_durations.append(duration)
            max_velocity_ratio = max(
                max_velocity_ratio,
                float(np.max(np.abs(velocities) / self.velocity_limits[None, :])),
            )
            max_acceleration_ratio = max(
                max_acceleration_ratio,
                float(
                    np.max(np.abs(accelerations) / self.acceleration_limits[None, :])
                ),
            )
            segment_start = actions[index].copy()
            segment_start_velocity = velocities[-1].copy()

        trajectory = np.concatenate(output_segments, axis=0).astype(np.float32)
        self.velocity = segment_start_velocity.copy()
        self.acceleration = np.zeros(7, dtype=np.float64)
        first_duration = segment_durations[0]
        transition_steps = max(1, int(round(first_duration * self.action_hz)))
        total_duration = float(sum(segment_durations))
        return trajectory, ReplanDiagnostics(
            time_scale=float(max(segment_durations) / self.dt),
            edge_gap=float(np.max(np.abs(actions[0] - current))),
            max_velocity_ratio=max_velocity_ratio,
            max_acceleration_ratio=max_acceleration_ratio,
            transition_steps=transition_steps,
            output_steps=int(len(trajectory)),
            duration_s=total_duration,
        )
