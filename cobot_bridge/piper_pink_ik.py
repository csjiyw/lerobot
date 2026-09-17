"""Differential Pink IK for one six-axis Piper arm.

This module is deliberately independent from ROS and robot output.  It turns a
target end-effector pose into a *bounded next joint target*.  The existing
Cobot Magic arbiter remains responsible for ownership, Teach preemption and
physical command publication.

Pink and Pinocchio are imported lazily so the normal bridge can run in an
environment that does not install the optional kinematics dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PIPER_ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))

# Piper's ROS JointState feedback and ``joint_ctrl_single`` command use the
# same joint convention as the official URDF.  A recorded 20 s replay confirms
# this directly: FK from the unmodified ROS joints matches ``end_pose`` with
# 0.14 mm RMSE, while flipping J2/J3 produces 254 mm RMSE.  Keep the explicit
# boundary helpers because callers rely on them, but the transform is identity.
PIPER_HARDWARE_TO_PINK_SIGN = np.ones(6, dtype=float)


def _validate_arm_joint_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite shape-(6,) vector, got {result.shape}")
    return result


def hardware_to_pink_joints(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert Piper ROS/SDK arm joints to the Pink/URDF convention."""

    return _validate_arm_joint_vector(values, name="hardware joints") * PIPER_HARDWARE_TO_PINK_SIGN


def pink_to_hardware_joints(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert Pink/URDF arm joints to Piper ROS/SDK command coordinates."""

    return _validate_arm_joint_vector(values, name="Pink joints") * PIPER_HARDWARE_TO_PINK_SIGN


@dataclass(frozen=True)
class PiperIKSettings:
    control_hz: float = 200.0
    max_joint_velocity: float = 0.45
    # Zero disables the Pink-side hard acceleration constraint.  The Cobot
    # Magic arbiter already enforces its own hard acceleration bound; applying
    # both creates avoidable lag and can make a QP infeasible at a joint limit.
    max_joint_acceleration: float = 0.0
    position_cost: float = 1.0
    orientation_cost: float = 0.7
    posture_cost: float = 1e-3
    acceleration_cost: float = 1e-3
    lm_damping: float = 1e-3
    qp_damping: float = 1e-8
    singular_value_warning: float = 1e-3
    condition_warning: float = 1_000.0
    # The URDF J5 upper limit remains a planner soft limit.  The real right
    # Piper can report up to roughly 0.1 rad beyond it when manually dragged
    # to its physical end stop, so measured J5 feedback gets this additional
    # acceptance margin before clipping into the Pink configuration.
    measured_feedback_tolerance_rad: float = 0.02
    joint5_feedback_tolerance_rad: float = 0.10

    def validate(self) -> None:
        positive = {
            "control_hz": self.control_hz,
            "max_joint_velocity": self.max_joint_velocity,
            "position_cost": self.position_cost,
            "orientation_cost": self.orientation_cost,
            "lm_damping": self.lm_damping,
            "qp_damping": self.qp_damping,
            "singular_value_warning": self.singular_value_warning,
            "condition_warning": self.condition_warning,
        }
        invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"IK settings must be finite and positive: {invalid}")
        for name in (
            "max_joint_acceleration",
            "posture_cost",
            "acceleration_cost",
            "measured_feedback_tolerance_rad",
            "joint5_feedback_tolerance_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class PiperIKDiagnostics:
    position_error_m: float
    orientation_error_rad: float
    maximum_joint_step_rad: float
    maximum_joint_velocity_rad_s: float
    minimum_singular_value: float
    jacobian_condition: float
    manipulability: float
    near_singularity: bool
    finite: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class PiperIKStep:
    joints: np.ndarray
    velocity: np.ndarray
    diagnostics: PiperIKDiagnostics


class PiperPinkIK:
    """Stateful 200 Hz differential IK adapter for one Piper arm."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        end_effector_frame: str = "link6",
        arm_joint_names: Sequence[str] = PIPER_ARM_JOINTS,
        settings: PiperIKSettings | None = None,
        solver: str | None = None,
    ) -> None:
        self.settings = settings or PiperIKSettings()
        self.settings.validate()
        self.dt = 1.0 / self.settings.control_hz
        self.end_effector_frame = str(end_effector_frame)
        self.arm_joint_names = tuple(str(name) for name in arm_joint_names)
        if len(self.arm_joint_names) != 6 or len(set(self.arm_joint_names)) != 6:
            raise ValueError("Piper IK requires six unique arm joint names")

        try:
            import pinocchio as pin
            import pink
            import qpsolvers
            from pink import solve_ik
            from pink.limits import AccelerationLimit
            from pink.tasks import FrameTask, LowAccelerationTask, PostureTask
        except ImportError as exc:
            raise ImportError(
                "PiperPinkIK requires pink, pinocchio and qpsolvers in the active environment"
            ) from exc

        self.pin = pin
        self.pink = pink
        self._solve_ik = solve_ik
        urdf_path = Path(urdf_path).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(urdf_path)

        full_model = pin.buildModelFromUrdf(str(urdf_path))
        missing = [name for name in self.arm_joint_names if not full_model.existJointName(name)]
        if missing:
            raise ValueError(f"URDF is missing Piper arm joints: {missing}")
        arm_joint_ids = {full_model.getJointId(name) for name in self.arm_joint_names}
        locked_joint_ids = [
            joint_id
            for joint_id in range(1, full_model.njoints)
            if joint_id not in arm_joint_ids
        ]
        neutral = pin.neutral(full_model)
        self.model = (
            pin.buildReducedModel(full_model, locked_joint_ids, neutral)
            if locked_joint_ids
            else full_model
        )
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError(
                f"reduced Piper model must be 6 DoF, got nq={self.model.nq} nv={self.model.nv}"
            )
        reduced_names = tuple(self.model.names[1:])
        if reduced_names != self.arm_joint_names:
            raise ValueError(
                f"unexpected reduced joint order {reduced_names}; expected {self.arm_joint_names}"
            )
        frame_id = self.model.getFrameId(self.end_effector_frame)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"URDF has no end-effector frame {self.end_effector_frame!r}")

        safe_velocity = np.full(self.model.nv, self.settings.max_joint_velocity, dtype=float)
        self.model.velocityLimit[:] = np.minimum(self.model.velocityLimit, safe_velocity)
        q0 = pin.neutral(self.model)
        self.configuration = pink.Configuration(self.model, self.model.createData(), q0)

        self.frame_task = FrameTask(
            self.end_effector_frame,
            position_cost=self.settings.position_cost,
            orientation_cost=self.settings.orientation_cost,
            lm_damping=self.settings.lm_damping,
        )
        self.posture_task = PostureTask(cost=self.settings.posture_cost)
        self.acceleration_task = LowAccelerationTask(cost=self.settings.acceleration_cost)
        self.tasks = [self.frame_task]
        if self.settings.posture_cost > 0:
            self.tasks.append(self.posture_task)
        if self.settings.acceleration_cost > 0:
            self.tasks.append(self.acceleration_task)

        self.acceleration_limit = (
            AccelerationLimit(
                self.model,
                np.full(self.model.nv, self.settings.max_joint_acceleration, dtype=float),
            )
            if self.settings.max_joint_acceleration > 0
            else None
        )
        self.limits = [
            self.model.configuration_limit,
            self.model.velocity_limit,
        ]
        if self.acceleration_limit is not None:
            self.limits.append(self.acceleration_limit)
        available = list(qpsolvers.available_solvers)
        if solver is None:
            solver = next(
                (name for name in ("daqp", "quadprog", "proxqp", "osqp") if name in available),
                available[0] if available else None,
            )
        if solver is None or solver not in available:
            raise RuntimeError(f"requested QP solver {solver!r} is unavailable; available={available}")
        self.solver = solver
        self.previous_velocity = np.zeros(self.model.nv, dtype=float)
        self.reset(q0)

    @property
    def lower_limits(self) -> np.ndarray:
        return self.model.lowerPositionLimit.copy()

    @property
    def upper_limits(self) -> np.ndarray:
        return self.model.upperPositionLimit.copy()

    def _validate_joints(self, joints: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"Piper arm joints must be a finite shape-(6,) vector, got {values.shape}")
        tolerance = 1e-6
        if np.any(values < self.model.lowerPositionLimit - tolerance) or np.any(
            values > self.model.upperPositionLimit + tolerance
        ):
            raise ValueError(
                "Piper joints exceed URDF limits: "
                f"q={values.tolist()} lower={self.lower_limits.tolist()} upper={self.upper_limits.tolist()}"
            )
        return np.clip(values, self.model.lowerPositionLimit, self.model.upperPositionLimit)

    def coerce_measured_joints(
        self,
        joints: Sequence[float] | np.ndarray,
        *,
        tolerance_rad: float | None = None,
    ) -> np.ndarray:
        """Project small encoder noise at a hard stop onto the URDF limits.

        This is only for measured feedback. Generated targets continue to use
        strict validation so a real out-of-range IK result cannot be hidden.
        """

        values = np.asarray(joints, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(
                f"measured Piper joints must be a finite shape-(6,) vector, got {values.shape}"
            )
        if tolerance_rad is None:
            tolerance = np.full(
                self.model.nq, self.settings.measured_feedback_tolerance_rad, dtype=float
            )
            tolerance[4] = self.settings.joint5_feedback_tolerance_rad
        else:
            if tolerance_rad < 0 or not np.isfinite(tolerance_rad):
                raise ValueError("measured joint tolerance must be finite and non-negative")
            tolerance = np.full(self.model.nq, tolerance_rad, dtype=float)
        if np.any(values < self.model.lowerPositionLimit - tolerance) or np.any(
            values > self.model.upperPositionLimit + tolerance
        ):
            raise ValueError(
                "measured Piper joints exceed URDF limits beyond feedback tolerance: "
                f"q={values.tolist()} tolerance={tolerance.tolist()}"
            )
        return np.clip(values, self.model.lowerPositionLimit, self.model.upperPositionLimit)

    def reset(self, joints: Sequence[float] | np.ndarray) -> None:
        values = self._validate_joints(joints)
        self.configuration.update(values)
        self.posture_task.set_target(values)
        self.previous_velocity = np.zeros(self.model.nv, dtype=float)
        self.acceleration_task.set_last_integration(self.previous_velocity, self.dt)
        if self.acceleration_limit is not None:
            self.acceleration_limit.set_last_integration(self.previous_velocity, self.dt)

    def set_posture_reference(self, joints: Sequence[float] | np.ndarray) -> None:
        self.posture_task.set_target(self._validate_joints(joints))

    def forward_kinematics(self, joints: Sequence[float] | np.ndarray) -> Any:
        values = self._validate_joints(joints)
        data = self.model.createData()
        self.pin.forwardKinematics(self.model, data, values)
        self.pin.updateFramePlacements(self.model, data)
        return data.oMf[self.model.getFrameId(self.end_effector_frame)].copy()

    def _as_se3(self, target_pose: Any) -> Any:
        if isinstance(target_pose, self.pin.SE3):
            return target_pose.copy()
        matrix = np.asarray(target_pose, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("target pose must be a finite Pinocchio SE3 or 4x4 matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("target pose has an invalid homogeneous final row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError("target pose rotation is not a valid SO(3) matrix")
        return self.pin.SE3(rotation, matrix[:3, 3])

    def _pose_errors(self, target: Any) -> tuple[float, float]:
        current = self.configuration.get_transform_frame_to_world(self.end_effector_frame)
        position_error = float(np.linalg.norm(target.translation - current.translation))
        orientation_error = float(
            np.linalg.norm(self.pin.log3(current.rotation.T @ target.rotation))
        )
        return position_error, orientation_error

    def _singularity_metrics(self) -> tuple[float, float, float, bool]:
        jacobian = self.configuration.get_frame_jacobian(self.end_effector_frame)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        minimum = float(singular_values[-1])
        maximum = float(singular_values[0])
        condition = float(maximum / minimum) if minimum > 0 else float("inf")
        manipulability = float(np.prod(singular_values))
        near = bool(
            minimum < self.settings.singular_value_warning
            or condition > self.settings.condition_warning
        )
        return minimum, condition, manipulability, near

    def diagnostics(
        self,
        target_pose: Any,
        *,
        previous_joints: Sequence[float] | np.ndarray | None = None,
        velocity: Sequence[float] | np.ndarray | None = None,
    ) -> PiperIKDiagnostics:
        target = self._as_se3(target_pose)
        position_error, orientation_error = self._pose_errors(target)
        joints = self.configuration.q
        if previous_joints is None:
            maximum_step = 0.0
        else:
            previous = self._validate_joints(previous_joints)
            maximum_step = float(np.max(np.abs(joints - previous)))
        velocity_values = (
            np.zeros(6, dtype=float)
            if velocity is None
            else np.asarray(velocity, dtype=float)
        )
        if velocity_values.shape != (6,):
            raise ValueError("velocity must have shape (6,)")
        minimum, condition, manipulability, near = self._singularity_metrics()
        finite = bool(
            np.isfinite(joints).all()
            and np.isfinite(velocity_values).all()
            and np.isfinite(position_error)
            and np.isfinite(orientation_error)
            and np.isfinite(minimum)
            and np.isfinite(manipulability)
        )
        return PiperIKDiagnostics(
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            maximum_joint_step_rad=maximum_step,
            maximum_joint_velocity_rad_s=float(np.max(np.abs(velocity_values))),
            minimum_singular_value=minimum,
            jacobian_condition=condition,
            manipulability=manipulability,
            near_singularity=near,
            finite=finite,
        )

    def step(
        self,
        target_pose: Any,
        *,
        measured_joints: Sequence[float] | np.ndarray | None = None,
    ) -> PiperIKStep:
        """Advance one bounded differential-IK tick without publishing it."""

        target = self._as_se3(target_pose)
        if measured_joints is not None:
            self.configuration.update(self._validate_joints(measured_joints))
        previous_joints = self.configuration.q.copy()
        self.frame_task.set_target(target)
        self.acceleration_task.set_last_integration(self.previous_velocity, self.dt)
        if self.acceleration_limit is not None:
            self.acceleration_limit.set_last_integration(self.previous_velocity, self.dt)
        velocity = self._solve_ik(
            self.configuration,
            self.tasks,
            self.dt,
            solver=self.solver,
            damping=self.settings.qp_damping,
            limits=self.limits,
        )
        if velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise RuntimeError(f"Pink returned an invalid velocity: {velocity}")
        next_joints = self.pin.integrate(
            self.model, self.configuration.q, velocity * self.dt
        )
        next_joints = self._validate_joints(next_joints)
        self.configuration.update(next_joints)
        self.previous_velocity = velocity.copy()
        diagnostics = self.diagnostics(
            target,
            previous_joints=previous_joints,
            velocity=velocity,
        )
        return PiperIKStep(next_joints.copy(), velocity.copy(), diagnostics)

    def solve_pose(
        self,
        target_pose: Any,
        *,
        current_joints: Sequence[float] | np.ndarray,
        maximum_iterations: int = 2_000,
        position_tolerance_m: float = 5e-4,
        orientation_tolerance_rad: float = 5e-3,
    ) -> tuple[PiperIKStep, int, bool]:
        """Iteratively solve one pose for offline validation only."""

        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        self.reset(current_joints)
        target = self._as_se3(target_pose)
        last_step: PiperIKStep | None = None
        for iteration in range(1, maximum_iterations + 1):
            last_step = self.step(target)
            if (
                last_step.diagnostics.position_error_m <= position_tolerance_m
                and last_step.diagnostics.orientation_error_rad <= orientation_tolerance_rad
            ):
                return last_step, iteration, True
        assert last_step is not None
        return last_step, maximum_iterations, False
