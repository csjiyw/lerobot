"""Offline Piper end-effector metrics from absolute joint-action chunks.

This module uses Pinocchio only for evaluation.  It does not initialize ROS,
connect to the robot, or participate in training/backpropagation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


PIPER_ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))
DEFAULT_PIPER_URDF = (
    Path(__file__).resolve().parents[1]
    / "pink_ik/robot/robot_descriptions/manipulator/AgileX/"
    "piper_description/urdf/piper.urdf"
)
DEFAULT_EEF_FRAMES = ("gripper_center", "link6")


class PiperForwardKinematics:
    """Reduced six-joint Piper FK model for one or more EEF frames."""

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_PIPER_URDF,
        *,
        frames: Sequence[str] = DEFAULT_EEF_FRAMES,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError("Piper EEF evaluation requires pinocchio") from exc

        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        if not frames or len(set(frames)) != len(frames):
            raise ValueError("EEF frame names must be non-empty and unique")
        self.frames = tuple(str(frame) for frame in frames)

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        missing_joints = [
            name for name in PIPER_ARM_JOINTS if not full_model.existJointName(name)
        ]
        if missing_joints:
            raise ValueError(f"Piper URDF is missing arm joints: {missing_joints}")
        arm_joint_ids = {full_model.getJointId(name) for name in PIPER_ARM_JOINTS}
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
        if self.model.nq != 6 or tuple(self.model.names[1:]) != PIPER_ARM_JOINTS:
            raise ValueError(
                "reduced Piper model must contain joint1..joint6 in order; "
                f"observed nq={self.model.nq} names={tuple(self.model.names[1:])}"
            )
        self.frame_ids: dict[str, int] = {}
        for frame in self.frames:
            frame_id = self.model.getFrameId(frame)
            if frame_id >= len(self.model.frames):
                raise ValueError(f"Piper URDF has no EEF frame {frame!r}")
            self.frame_ids[frame] = frame_id

    def forward(self, joints: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        values = np.asarray(joints, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"Piper FK joints must have shape (N, 6), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Piper FK joints contain non-finite values")

        outputs = {
            frame: (
                np.empty((values.shape[0], 3), dtype=np.float64),
                np.empty((values.shape[0], 3, 3), dtype=np.float64),
            )
            for frame in self.frames
        }
        data = self.model.createData()
        for row, configuration in enumerate(values):
            self.pin.forwardKinematics(self.model, data, configuration)
            self.pin.updateFramePlacements(self.model, data)
            for frame, frame_id in self.frame_ids.items():
                placement = data.oMf[frame_id]
                positions, rotations = outputs[frame]
                positions[row] = placement.translation
                rotations[row] = placement.rotation
        return outputs

    def joint_limit_summary(self, joints: np.ndarray) -> dict[str, Any]:
        values = np.asarray(joints, dtype=np.float64)
        lower = self.model.lowerPositionLimit.reshape(1, 6)
        upper = self.model.upperPositionLimit.reshape(1, 6)
        violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
        return {
            "violating_sample_fraction": float(np.any(violation > 1e-6, axis=1).mean()),
            "per_joint_violating_fraction": (violation > 1e-6).mean(axis=0).tolist(),
            "maximum_violation_rad": float(violation.max(initial=0.0)),
        }


def _scalar_error_summary(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float]:
    scaled = np.asarray(values, dtype=np.float64) * float(scale)
    return {
        "mean": float(scaled.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(scaled)))),
        "median": float(np.median(scaled)),
        "p95": float(np.quantile(scaled, 0.95)),
        "maximum": float(scaled.max(initial=0.0)),
    }


def _orientation_error_rad(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("nji,njk->nik", target, predicted)
    cosine = (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def eef_error_summary(
    predicted_normalized: np.ndarray,
    target_normalized: np.ndarray,
    valid: np.ndarray,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    kinematics: PiperForwardKinematics,
) -> dict[str, Any]:
    """Compare normalized action chunks in physical EEF space."""

    predicted = np.asarray(predicted_normalized, dtype=np.float64)
    target = np.asarray(target_normalized, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError(
            f"EEF action chunks must share shape (N, H, D), got {predicted.shape}/{target.shape}"
        )
    if mask.shape != predicted.shape[:2] or predicted.shape[2] < 6:
        raise ValueError("EEF valid mask or action dimension is incompatible")
    if not mask.any():
        raise ValueError("EEF evaluation has no valid action steps")
    mean = np.asarray(action_mean, dtype=np.float64).reshape(1, 1, -1)
    std = np.asarray(action_std, dtype=np.float64).reshape(1, 1, -1)
    if mean.shape[2] != predicted.shape[2] or std.shape != mean.shape:
        raise ValueError("action normalization statistics do not match action dimension")

    predicted_physical = predicted * std + mean
    target_physical = target * std + mean
    predicted_joints = predicted_physical[..., :6][mask]
    target_joints = target_physical[..., :6][mask]
    horizon_indices = np.broadcast_to(
        np.arange(predicted.shape[1], dtype=np.int64), mask.shape
    )[mask]
    predicted_fk = kinematics.forward(predicted_joints)
    target_fk = kinematics.forward(target_joints)

    frames: dict[str, Any] = {}
    for frame in kinematics.frames:
        predicted_position, predicted_rotation = predicted_fk[frame]
        target_position, target_rotation = target_fk[frame]
        position_error_m = np.linalg.norm(predicted_position - target_position, axis=1)
        orientation_error_rad = _orientation_error_rad(
            predicted_rotation, target_rotation
        )
        per_horizon = []
        for horizon in range(predicted.shape[1]):
            horizon_mask = horizon_indices == horizon
            per_horizon.append(
                {
                    "horizon": horizon,
                    "samples": int(horizon_mask.sum()),
                    "position_rmse_mm": (
                        float(
                            np.sqrt(np.mean(np.square(position_error_m[horizon_mask])))
                            * 1000.0
                        )
                        if horizon_mask.any()
                        else None
                    ),
                    "orientation_rmse_deg": (
                        float(
                            np.degrees(
                                np.sqrt(
                                    np.mean(np.square(orientation_error_rad[horizon_mask]))
                                )
                            )
                        )
                        if horizon_mask.any()
                        else None
                    ),
                }
            )
        frames[frame] = {
            "task_space_loss": {
                "position_mse_m2": float(np.mean(np.square(position_error_m))),
                "orientation_mse_rad2": float(
                    np.mean(np.square(orientation_error_rad))
                ),
            },
            "position_error_mm": _scalar_error_summary(position_error_m, scale=1000.0),
            "orientation_error_deg": _scalar_error_summary(
                orientation_error_rad, scale=180.0 / np.pi
            ),
            "per_horizon": per_horizon,
        }

    return {
        "samples": int(mask.sum()),
        "frames": frames,
        "predicted_joint_limits": kinematics.joint_limit_summary(predicted_joints),
        "target_joint_limits": kinematics.joint_limit_summary(target_joints),
    }
