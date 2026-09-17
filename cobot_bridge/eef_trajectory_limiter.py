"""PC-side Cartesian smoothing for absolute 7D EEF policy targets.

The limiter intentionally operates before IK.  Its input/output pose is
``[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]`` and every returned pose is
safe to use as the next 200 Hz Pink target.  It does not publish robot output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CartesianLimits:
    control_hz: float = 200.0
    max_linear_velocity: float = 0.25       # m/s
    max_linear_acceleration: float = 1.20   # m/s^2
    max_angular_velocity: float = 1.20      # rad/s
    max_angular_acceleration: float = 6.00  # rad/s^2
    max_gripper_velocity: float = 0.10
    max_gripper_acceleration: float = 0.50

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class CartesianTrajectoryLimiter:
    """Stateful velocity/acceleration bounded SE(3) target generator."""

    def __init__(self, limits: CartesianLimits | None = None) -> None:
        self.limits = limits or CartesianLimits()
        self.limits.validate()
        self.dt = 1.0 / self.limits.control_hz
        self.pose: np.ndarray | None = None
        self.linear_velocity = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.gripper_velocity = 0.0

    @staticmethod
    def _bounded_velocity(
        current: np.ndarray, target: np.ndarray, velocity: np.ndarray, *, vmax: float, amax: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        error = target - current
        desired = np.clip(error / dt, -vmax, vmax)
        next_velocity = velocity + np.clip(desired - velocity, -amax * dt, amax * dt)
        next_position = current + next_velocity * dt
        # Never pass a stationary target merely due to finite integration.
        crossed = (error * (target - next_position)) <= 0.0
        next_position = np.where(crossed, target, next_position)
        next_velocity = np.where(crossed, 0.0, next_velocity)
        return next_position, next_velocity

    def reset(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("EEF reset pose must be finite shape (7,)")
        self.pose = pose.copy()
        self.linear_velocity.fill(0.0)
        self.angular_velocity.fill(0.0)
        self.gripper_velocity = 0.0

    def step(self, target: np.ndarray) -> np.ndarray:
        if self.pose is None:
            raise RuntimeError("CartesianTrajectoryLimiter must be reset before step")
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("EEF target must be finite shape (7,)")
        position, self.linear_velocity = self._bounded_velocity(
            self.pose[:3], target[:3], self.linear_velocity,
            vmax=self.limits.max_linear_velocity, amax=self.limits.max_linear_acceleration, dt=self.dt,
        )
        current_rotation = Rotation.from_rotvec(self.pose[3:6])
        target_rotation = Rotation.from_rotvec(target[3:6])
        rotation_error = (current_rotation.inv() * target_rotation).as_rotvec()
        delta_rotation, self.angular_velocity = self._bounded_velocity(
            np.zeros(3), rotation_error, self.angular_velocity,
            vmax=self.limits.max_angular_velocity, amax=self.limits.max_angular_acceleration, dt=self.dt,
        )
        rotation = current_rotation * Rotation.from_rotvec(delta_rotation)
        gripper, gripper_velocity = self._bounded_velocity(
            np.asarray([self.pose[6]]), np.asarray([target[6]]), np.asarray([self.gripper_velocity]),
            vmax=self.limits.max_gripper_velocity, amax=self.limits.max_gripper_acceleration, dt=self.dt,
        )
        self.gripper_velocity = float(gripper_velocity[0])
        self.pose = np.concatenate((position, rotation.as_rotvec(), gripper))
        return self.pose.astype(np.float32, copy=True)

    def advance(self, target: np.ndarray, ticks: int) -> list[np.ndarray]:
        if ticks < 1:
            raise ValueError("ticks must be positive")
        return [self.step(target) for _ in range(ticks)]


def blend_eef_chunk_edge(
    previous_end: np.ndarray | None,
    actions: np.ndarray,
    *,
    policy_hz: float,
    max_steps: int = 12,
    limits: CartesianLimits | None = None,
) -> tuple[np.ndarray, int]:
    """Blend a newly predicted absolute EEF chunk from the last chunk endpoint.

    The blend length is chosen from the actual Cartesian/rotation boundary gap,
    rather than using a fixed chunk-edge frame count.  Pose interpolation is
    SE(3): linear translation, exponential-map rotation, scalar gripper.
    """

    result = np.asarray(actions, dtype=np.float32).copy()
    if previous_end is None or result.ndim != 2 or result.shape[1] != 7 or len(result) == 0:
        return result, 0
    if not np.isfinite(result).all() or not np.isfinite(previous_end).all():
        raise ValueError("EEF edge inputs must be finite")
    if policy_hz <= 0 or max_steps < 1:
        raise ValueError("policy_hz and max_steps must be positive")
    profile = limits or CartesianLimits()
    start = np.asarray(previous_end, dtype=np.float64).reshape(7)
    first = result[0].astype(np.float64)
    linear_gap = float(np.linalg.norm(first[:3] - start[:3]))
    angular_gap = float(np.linalg.norm((Rotation.from_rotvec(start[3:6]).inv() * Rotation.from_rotvec(first[3:6])).as_rotvec()))
    gripper_gap = abs(float(first[6] - start[6]))
    # Smoothstep has peak velocity 1.5/T and peak acceleration 6/T^2.
    duration = max(
        1.5 * linear_gap / profile.max_linear_velocity,
        np.sqrt(6.0 * linear_gap / profile.max_linear_acceleration),
        1.5 * angular_gap / profile.max_angular_velocity,
        np.sqrt(6.0 * angular_gap / profile.max_angular_acceleration),
        1.5 * gripper_gap / profile.max_gripper_velocity,
        np.sqrt(6.0 * gripper_gap / profile.max_gripper_acceleration),
    )
    steps = min(len(result), max_steps, max(1, int(np.ceil(duration * policy_hz))))
    start_rotation = Rotation.from_rotvec(start[3:6])
    for index in range(steps):
        phase = (index + 1) / steps
        alpha = phase * phase * (3.0 - 2.0 * phase)
        target = result[index].astype(np.float64)
        target_rotation = Rotation.from_rotvec(target[3:6])
        result[index, :3] = (1.0 - alpha) * start[:3] + alpha * target[:3]
        result[index, 3:6] = (start_rotation * Rotation.from_rotvec(
            alpha * (start_rotation.inv() * target_rotation).as_rotvec()
        )).as_rotvec()
        result[index, 6] = (1.0 - alpha) * start[6] + alpha * target[6]
    return result, steps
