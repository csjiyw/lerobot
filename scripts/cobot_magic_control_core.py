#!/usr/bin/env python3
"""Pure state-machine logic for Cobot Magic control ownership.

This module intentionally has no ROS dependency so that the safety transitions
can be unit tested without connecting to a robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Sequence


class ControlMode(IntEnum):
    SAFE = 0
    POLICY_PUPPET = 1
    RLT_SYNC = 2
    TEACH = 3
    RESYNC_REQUIRED = 4
    ALIGN_SYNC = 5


REQUESTABLE_MODES = {
    ControlMode.SAFE,
    ControlMode.POLICY_PUPPET,
    ControlMode.RLT_SYNC,
}


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    effective_mode: ControlMode
    epoch: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "effective_mode": self.effective_mode.name,
            "effective_mode_id": int(self.effective_mode),
            "epoch": self.epoch,
        }


class VelocityAccelerationLimiter:
    """Stateful per-axis position command limiter.

    The limiter tracks a bounded velocity toward each requested position and
    bounds the velocity change on every update. A long command gap resets the
    trajectory to measured feedback so a new owner cannot resume stale motion.
    """

    def __init__(
        self,
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        *,
        reset_after_s: float = 0.25,
    ) -> None:
        self.max_velocity = self._positive_limits(max_velocity, "max_velocity")
        self.max_acceleration = self._positive_limits(
            max_acceleration, "max_acceleration"
        )
        if len(self.max_velocity) != len(self.max_acceleration):
            raise ValueError("motion_limit_size_mismatch")
        if not math.isfinite(reset_after_s) or reset_after_s <= 0.0:
            raise ValueError("invalid_reset_after_s")
        self.reset_after_s = float(reset_after_s)
        self.position: list[float] | None = None
        self.velocity: list[float] | None = None
        self.last_time: float | None = None

    @staticmethod
    def _positive_limits(values: Sequence[float], name: str) -> list[float]:
        result = [float(value) for value in values]
        if not result or not all(math.isfinite(value) and value > 0.0 for value in result):
            raise ValueError(f"invalid_{name}")
        return result

    def reset(self) -> None:
        self.position = None
        self.velocity = None
        self.last_time = None

    def step(
        self,
        target: Sequence[float],
        *,
        now: float,
        measured_position: Sequence[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        target_values = [float(value) for value in target]
        if len(target_values) != len(self.max_velocity) or not all(
            math.isfinite(value) for value in target_values
        ):
            raise ValueError("invalid_motion_target")

        measured_values = None
        if measured_position is not None:
            measured_values = [float(value) for value in measured_position]
            if len(measured_values) != len(target_values) or not all(
                math.isfinite(value) for value in measured_values
            ):
                raise ValueError("invalid_measured_position")

        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("invalid_motion_timestamp")
        gap = None if self.last_time is None else timestamp - self.last_time
        if (
            self.position is None
            or self.velocity is None
            or gap is None
            or gap <= 0.0
            or gap > self.reset_after_s
        ):
            self.position = list(measured_values or target_values)
            self.velocity = [0.0] * len(target_values)
            self.last_time = timestamp
            return list(self.position), list(self.velocity)

        dt = gap
        next_position: list[float] = []
        next_velocity: list[float] = []
        for index, target_value in enumerate(target_values):
            position = self.position[index]
            velocity = self.velocity[index]
            error = target_value - position
            acceleration_limit = self.max_acceleration[index]

            if abs(error) <= 1e-12 and abs(velocity) <= acceleration_limit * dt:
                next_position.append(target_value)
                next_velocity.append(0.0)
                continue

            direction = 1.0 if error >= 0.0 else -1.0
            stopping_velocity = math.sqrt(2.0 * acceleration_limit * abs(error))
            desired_velocity = direction * min(
                self.max_velocity[index], stopping_velocity
            )
            velocity_delta = desired_velocity - velocity
            max_velocity_delta = acceleration_limit * dt
            velocity_delta = max(
                -max_velocity_delta, min(max_velocity_delta, velocity_delta)
            )
            limited_velocity = max(
                -self.max_velocity[index],
                min(self.max_velocity[index], velocity + velocity_delta),
            )
            limited_position = position + 0.5 * (velocity + limited_velocity) * dt

            # Snap only when the prior velocity can physically stop within
            # this tick. Otherwise preserve the acceleration bound and allow
            # the bounded tracker to correct a small overshoot.
            if (
                error != 0.0
                and (target_value - limited_position) * error <= 0.0
                and abs(velocity) <= max_velocity_delta + 1e-12
            ):
                limited_position = target_value
                limited_velocity = 0.0

            next_position.append(limited_position)
            next_velocity.append(limited_velocity)

        self.position = next_position
        self.velocity = next_velocity
        self.last_time = timestamp
        return list(next_position), list(next_velocity)


class OneShotTeachAlignmentGate:
    """Allow at most one slow alignment during one physical Teach press.

    Once the initial alignment has completed (or was unnecessary), subsequent
    master motion must stay in direct-follow mode until the button is released.
    """

    def __init__(self) -> None:
        self.follow_ready = False

    def reset(self) -> None:
        self.follow_ready = False

    def needs_alignment(self, maximum_error: float, threshold: float) -> bool:
        if self.follow_ready:
            return False
        if maximum_error <= threshold:
            self.follow_ready = True
            return False
        return True

    def finish_alignment(self) -> None:
        self.follow_ready = True


class ControlStateMachine:
    """Owns mode transitions and rejects stale/conflicting actions.

    ``dry_run`` is a hard process-level safety gate. Even if logical output is
    enabled, no action can be authorized while dry-run is active.
    """

    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = bool(dry_run)
        self.output_enabled = False
        self.requested_mode = ControlMode.SAFE
        self.effective_mode = ControlMode.SAFE
        self.teach_active = False
        self.emergency_latched = False
        self.resync_required = False
        self.epoch = 0
        self.reason = "startup_safe"

    @staticmethod
    def parse_mode(value: ControlMode | int | str) -> ControlMode:
        if isinstance(value, ControlMode):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            aliases = {
                "POLICY": "POLICY_PUPPET",
                "PUPPET": "POLICY_PUPPET",
                "RLT": "RLT_SYNC",
                "SYNC": "RLT_SYNC",
            }
            normalized = aliases.get(normalized, normalized)
            if normalized.isdigit():
                return ControlMode(int(normalized))
            return ControlMode[normalized]
        return ControlMode(int(value))

    def _decision(self, accepted: bool, reason: str) -> Decision:
        self.reason = reason
        return Decision(accepted, reason, self.effective_mode, self.epoch)

    def request_mode(self, value: ControlMode | int | str) -> Decision:
        try:
            mode = self.parse_mode(value)
        except (KeyError, ValueError, TypeError):
            return self._decision(False, "invalid_mode")
        if mode not in REQUESTABLE_MODES:
            return self._decision(False, "teach_is_hardware_owned")
        if self.emergency_latched:
            return self._decision(False, "emergency_latched")

        self.requested_mode = mode
        self.epoch += 1
        if mode == ControlMode.SAFE:
            self.resync_required = False
            self.effective_mode = ControlMode.TEACH if self.teach_active else ControlMode.SAFE
            return self._decision(True, "safe_requested")

        if self.teach_active:
            self.effective_mode = ControlMode.TEACH
            return self._decision(True, "mode_deferred_until_teach_release")

        self.resync_required = False
        if mode == ControlMode.RLT_SYNC:
            self.effective_mode = ControlMode.ALIGN_SYNC
            return self._decision(True, "waiting_sync_first_target")
        self.effective_mode = mode
        return self._decision(True, "mode_changed")

    def set_teach_active(self, active: bool) -> Decision:
        active = bool(active)
        if active == self.teach_active:
            return self._decision(True, "teach_unchanged")

        self.teach_active = active
        self.epoch += 1
        if active:
            self.effective_mode = ControlMode.TEACH
            self.resync_required = False
            return self._decision(True, "teach_active")

        if self.emergency_latched:
            self.requested_mode = ControlMode.SAFE
            self.effective_mode = ControlMode.SAFE
            self.resync_required = False
            return self._decision(True, "teach_released_emergency_latched")

        if self.requested_mode in (ControlMode.POLICY_PUPPET, ControlMode.RLT_SYNC):
            self.resync_required = True
            self.effective_mode = ControlMode.RESYNC_REQUIRED
            return self._decision(True, "teach_released_resync_required")

        self.resync_required = False
        self.effective_mode = ControlMode.SAFE
        return self._decision(True, "teach_released_safe")

    def set_output_enabled(self, enabled: bool) -> Decision:
        if enabled and self.emergency_latched:
            return self._decision(False, "emergency_latched")
        self.output_enabled = bool(enabled)
        if not self.output_enabled:
            return self._decision(True, "output_disabled")
        if self.dry_run:
            return self._decision(True, "output_logically_enabled_dry_run")
        return self._decision(True, "output_enabled")

    def emergency_safe(self) -> Decision:
        """Latch automatic control off without overriding physical Teach."""
        self.epoch += 1
        self.output_enabled = False
        self.requested_mode = ControlMode.SAFE
        self.effective_mode = ControlMode.TEACH if self.teach_active else ControlMode.SAFE
        self.resync_required = False
        self.emergency_latched = True
        return self._decision(True, "emergency_safe_latched")

    def clear_emergency(self) -> Decision:
        self.epoch += 1
        self.output_enabled = False
        self.requested_mode = ControlMode.SAFE
        self.effective_mode = ControlMode.TEACH if self.teach_active else ControlMode.SAFE
        self.resync_required = False
        self.emergency_latched = False
        return self._decision(True, "emergency_cleared")

    def resync(self) -> Decision:
        if self.teach_active:
            return self._decision(False, "teach_active")
        if self.requested_mode not in (ControlMode.POLICY_PUPPET, ControlMode.RLT_SYNC):
            return self._decision(False, "no_policy_mode_requested")
        self.epoch += 1
        self.resync_required = False
        if self.requested_mode == ControlMode.RLT_SYNC:
            self.effective_mode = ControlMode.ALIGN_SYNC
            return self._decision(True, "resynced_waiting_sync_first_target")
        self.effective_mode = self.requested_mode
        return self._decision(True, "resynced_waiting_fresh_action")

    def authorize_alignment_target(self, *, epoch: int | None = None) -> Decision:
        if self.teach_active or self.effective_mode == ControlMode.TEACH:
            return self._decision(False, "teach_active")
        if self.requested_mode != ControlMode.RLT_SYNC or self.effective_mode != ControlMode.ALIGN_SYNC:
            return self._decision(False, "not_align_sync")
        if epoch is not None and int(epoch) != self.epoch:
            return self._decision(False, "stale_epoch")
        if not self.output_enabled:
            return self._decision(False, "output_disabled")
        if self.dry_run:
            return self._decision(False, "dry_run")
        return self._decision(True, "sync_first_target_accepted")

    def mark_alignment_started(self) -> Decision:
        if self.effective_mode != ControlMode.ALIGN_SYNC:
            return self._decision(False, "not_align_sync")
        return self._decision(True, "individual_alignment_started")

    def finish_alignment(self) -> Decision:
        if self.teach_active:
            return self._decision(False, "teach_active")
        if self.requested_mode != ControlMode.RLT_SYNC or self.effective_mode != ControlMode.ALIGN_SYNC:
            return self._decision(False, "not_align_sync")
        self.effective_mode = ControlMode.RLT_SYNC
        return self._decision(True, "individual_alignment_complete")

    def fail_safe(self, reason: str) -> Decision:
        self.epoch += 1
        self.requested_mode = ControlMode.SAFE
        self.effective_mode = ControlMode.SAFE
        self.resync_required = False
        return self._decision(False, reason)

    def authorize_action(self, *, epoch: int | None = None) -> Decision:
        if self.teach_active or self.effective_mode == ControlMode.TEACH:
            return self._decision(False, "teach_active")
        if self.resync_required or self.effective_mode == ControlMode.RESYNC_REQUIRED:
            return self._decision(False, "resync_required")
        if self.effective_mode == ControlMode.ALIGN_SYNC:
            return self._decision(False, "align_sync_in_progress")
        if self.effective_mode not in (ControlMode.POLICY_PUPPET, ControlMode.RLT_SYNC):
            return self._decision(False, "owner_safe")
        if epoch is not None and int(epoch) != self.epoch:
            return self._decision(False, "stale_epoch")
        if not self.output_enabled:
            return self._decision(False, "output_disabled")
        if self.dry_run:
            return self._decision(False, "dry_run")
        return self._decision(True, "accepted")

    def policy_targets(self) -> tuple[str, ...]:
        if self.effective_mode == ControlMode.POLICY_PUPPET:
            return ("puppet",)
        if self.effective_mode == ControlMode.RLT_SYNC:
            return ("master", "puppet")
        return ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode.name,
            "requested_mode_id": int(self.requested_mode),
            "effective_mode": self.effective_mode.name,
            "effective_mode_id": int(self.effective_mode),
            "teach_active": self.teach_active,
            "emergency_latched": self.emergency_latched,
            "resync_required": self.resync_required,
            "output_enabled": self.output_enabled,
            "dry_run": self.dry_run,
            "epoch": self.epoch,
            "reason": self.reason,
        }


def validate_joint_action(values: Iterable[Any], *, joint_count: int = 7) -> list[float]:
    import math

    result = [float(value) for value in values]
    if len(result) != joint_count:
        raise ValueError(f"expected_{joint_count}_values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("non_finite_action")
    if any(abs(value) > 12.5 for value in result[:6]):
        raise ValueError("joint_position_out_of_protocol_range")
    return result


def validate_eef_target(values: Iterable[Any]) -> list[float]:
    """Validate an absolute Piper EEF target in ``xyz+rotvec+gripper`` form."""

    result = [float(value) for value in values]
    if len(result) != 7:
        raise ValueError("expected_7_values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("non_finite_eef_target")
    if any(abs(value) > 2.0 for value in result[:3]):
        raise ValueError("eef_position_out_of_protocol_range")
    if any(abs(value) > 4.0 * math.pi for value in result[3:6]):
        raise ValueError("eef_rotation_out_of_protocol_range")
    if not 0.0 <= result[6] <= 0.2:
        raise ValueError("eef_gripper_out_of_protocol_range")
    return result


def quaternion_to_rotvec(x: float, y: float, z: float, w: float) -> list[float]:
    """Convert a ROS quaternion to a shortest-axis rotation vector."""

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid_eef_quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half <= 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, max(1e-12, w))
    scale = angle / sin_half
    return [x * scale, y * scale, z * scale]


def rotvec_to_euler_xyz(rotvec: Sequence[float]) -> tuple[float, float, float]:
    """Convert a rotation vector to Piper's extrinsic XYZ RPY convention.

    Piper defines ``RX, RY, RZ`` as fixed/base-frame rotations, so the
    rotation matrix is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    rx, ry, rz = (float(value) for value in rotvec)
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta <= 1e-12:
        return 0.0, 0.0, 0.0
    x, y, z = rx / theta, ry / theta, rz / theta
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    r11 = c + x * x * one_c
    r12 = x * y * one_c - z * s
    r13 = x * z * one_c + y * s
    r21 = y * x * one_c + z * s
    r22 = c + y * y * one_c
    r23 = y * z * one_c - x * s
    r31 = z * x * one_c - y * s
    r32 = z * y * one_c + x * s
    r33 = c + z * z * one_c
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll), Piper's XYZ extrinsic convention.
    pitch = math.asin(max(-1.0, min(1.0, -r31)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(r32, r33)
        yaw = math.atan2(r21, r11)
    else:
        # Pick yaw=0 at the gimbal lock singularity and keep roll stable.
        roll = math.atan2(-r23, r22)
        yaw = 0.0
    return roll, pitch, yaw


def euler_xyz_to_rotvec(euler: Sequence[float]) -> list[float]:
    """Convert PosCmd Euler XYZ angles back to a rotation vector."""

    roll, pitch, yaw = (float(value) for value in euler)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Piper XYZ extrinsic: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    r11 = cp * cy
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cr * cp
    trace = max(-1.0, min(3.0, r11 + r22 + r33))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta <= 1e-8:
        return [0.0, 0.0, 0.0]
    sin_theta = math.sin(theta)
    if abs(sin_theta) <= 1e-8:
        # This rare branch is only needed near pi; choose the largest diagonal
        # and recover a stable axis without introducing a dependency on scipy.
        diagonal = (r11, r22, r33)
        index = max(range(3), key=lambda item: diagonal[item])
        axis = [0.0, 0.0, 0.0]
        axis[index] = 1.0
        return [theta * value for value in axis]
    scale = theta / (2.0 * sin_theta)
    return [
        scale * (r32 - r23),
        scale * (r13 - r31),
        scale * (r21 - r12),
    ]
