"""Runtime Piper EEF frame conversion.

The Piper ROS ``end_pose`` feedback is the pose of ``link6``.  The existing
EEF datasets and checkpoints use the fixed ``gripper_center`` TCP from the
official URDF.  This module contains only the fixed TCP transform; it does
not run IK, ROS, CAN, or publish commands.

Keeping this conversion at the policy boundary is important:

``SDK link6 feedback -> model gripper_center state``
``model gripper_center target -> SDK link6 target``

The translation must be rotated with the current orientation.  A constant
world-frame XYZ offset is not equivalent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

import numpy as np


SDK_EEF_FRAME: Final[str] = "link6"
LEGACY_MODEL_EEF_FRAME: Final[str] = "gripper_center"
SUPPORTED_EEF_FRAMES: Final[tuple[str, str]] = (SDK_EEF_FRAME, LEGACY_MODEL_EEF_FRAME)

_LOCAL_URDF = (
    Path(__file__).resolve().parents[1]
    / "pink_ik/robot/robot_descriptions/manipulator/AgileX/"
    "piper_description/urdf/piper.urdf"
)
_DEFAULT_URDF_CANDIDATES: Final[tuple[Path, ...]] = (
    _LOCAL_URDF,
    Path("/home/agilex/smolvla_runtime/cobot_bridge/assets/piper_model.urdf"),
    Path("/home/agilex/piper_ros/src/piper_description/urdf/piper_description.urdf"),
    Path("/home/agilex/piper_ros/src/piper_description/urdf/piper.urdf"),
)


def resolve_piper_urdf(urdf_path: str | Path | None = None) -> Path:
    """Resolve an official Piper URDF on either the PC or AGX runtime."""

    requested = urdf_path or os.environ.get("COBOT_PIPER_URDF")
    candidates = (Path(requested).expanduser(),) if requested else _DEFAULT_URDF_CANDIDATES
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file():
            return path
    searched = ", ".join(str(Path(candidate).expanduser()) for candidate in candidates)
    raise FileNotFoundError(
        "Piper EEF conversion requires the official URDF; searched: " + searched
    )


def _parse_vector(value: str | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if value is None or not value.strip():
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(value, sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"invalid URDF vector: {value!r}")
    return values


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF roll-pitch-yaw to the child-to-parent rotation."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = (float(item) for item in value)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        # The first-order form avoids numerical loss for tiny policy deltas.
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    cross = _skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
            dtype=np.float64,
        ) * 0.5
    if np.pi - angle < 1e-6:
        # At pi the skew part vanishes.  The eigenvector of the symmetric
        # part is the stable axis; its sign is immaterial for this pose.
        _, vectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = vectors[:, -1]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    axis /= max(2.0 * np.sin(angle), 1e-12)
    return axis * angle


def eef_pose_delta(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a base-frame Cartesian delta from ``state`` to ``target``.

    Translation is expressed in the Piper base frame. Rotation is the spatial
    relative rotation ``R_target @ R_state.T`` rather than a subtraction of
    two axis-angle vectors (axis-angle coordinates are not additive).
    Gripper remains the absolute target command.
    """

    current = np.asarray(state, dtype=np.float64).reshape(-1)
    desired = np.asarray(target, dtype=np.float64).reshape(-1)
    if current.shape != (7,) or desired.shape != (7,):
        raise ValueError("EEF state and target must both have shape (7,)")
    if not np.isfinite(current).all() or not np.isfinite(desired).all():
        raise ValueError("EEF state or target contains non-finite values")
    result = np.empty(7, dtype=np.float32)
    result[:3] = desired[:3] - current[:3]
    state_rotation = _rotvec_to_matrix(current[3:6])
    target_rotation = _rotvec_to_matrix(desired[3:6])
    result[3:6] = _matrix_to_rotvec(target_rotation @ state_rotation.T)
    result[6] = desired[6]
    return result


def apply_eef_pose_delta(state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a base-frame Cartesian delta produced by :func:`eef_pose_delta`."""

    current = np.asarray(state, dtype=np.float64).reshape(-1)
    action = np.asarray(delta, dtype=np.float64).reshape(-1)
    if current.shape != (7,) or action.shape != (7,):
        raise ValueError("EEF state and delta must both have shape (7,)")
    if not np.isfinite(current).all() or not np.isfinite(action).all():
        raise ValueError("EEF state or delta contains non-finite values")
    result = np.empty(7, dtype=np.float32)
    result[:3] = current[:3] + action[:3]
    state_rotation = _rotvec_to_matrix(current[3:6])
    delta_rotation = _rotvec_to_matrix(action[3:6])
    result[3:6] = _matrix_to_rotvec(delta_rotation @ state_rotation)
    result[6] = action[6]
    return result


class PiperEEFFrameTransform:
    """Convert absolute Piper EEF poses between ``link6`` and TCP frames."""

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        self.urdf_path = resolve_piper_urdf(urdf_path)
        tree = ElementTree.parse(self.urdf_path)
        fixed = next(
            (
                joint
                for joint in tree.getroot().iter("joint")
                if joint.get("name") == "gripper_center_fixed"
            ),
            None,
        )
        if fixed is None or fixed.get("type") != "fixed":
            raise ValueError(
                f"{self.urdf_path} has no fixed gripper_center_fixed joint"
            )
        parent = fixed.find("parent")
        child = fixed.find("child")
        if parent is None or child is None or parent.get("link") != SDK_EEF_FRAME or child.get("link") != LEGACY_MODEL_EEF_FRAME:
            raise ValueError(
                "gripper_center_fixed must connect link6 -> gripper_center; "
                f"got parent={None if parent is None else parent.get('link')!r}, "
                f"child={None if child is None else child.get('link')!r}"
            )
        origin = fixed.find("origin")
        self.translation = _parse_vector(
            None if origin is None else origin.get("xyz"), default=(0.0, 0.0, 0.0)
        )
        rpy = _parse_vector(
            None if origin is None else origin.get("rpy"), default=(0.0, 0.0, 0.0)
        )
        self.rotation = _rpy_to_matrix(rpy)
        if not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all():
            raise ValueError("Piper fixed EEF transform is non-finite")

    def transform(
        self,
        pose: np.ndarray,
        *,
        source_frame: str,
        target_frame: str,
    ) -> np.ndarray:
        """Transform one ``[x,y,z,rx,ry,rz,gripper]`` pose."""

        if source_frame not in SUPPORTED_EEF_FRAMES or target_frame not in SUPPORTED_EEF_FRAMES:
            raise ValueError(
                f"unsupported EEF frame conversion {source_frame!r} -> {target_frame!r}"
            )
        value = np.asarray(pose, dtype=np.float64).reshape(-1)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"EEF pose must be a finite shape-(7,) vector, got {value.shape}")
        if source_frame == target_frame:
            return value.astype(np.float32, copy=True)

        source_rotation = _rotvec_to_matrix(value[3:6])
        if source_frame == SDK_EEF_FRAME and target_frame == LEGACY_MODEL_EEF_FRAME:
            target_rotation = source_rotation @ self.rotation
            target_position = value[:3] + source_rotation @ self.translation
        else:
            # T_base_link6 = T_base_gripper_center * inv(T_link6_gripper_center)
            target_rotation = source_rotation @ self.rotation.T
            target_position = value[:3] - target_rotation @ self.translation
        result = np.empty(7, dtype=np.float32)
        result[:3] = target_position
        result[3:6] = _matrix_to_rotvec(target_rotation)
        # Gripper command/state semantics are independent of the rigid TCP.
        result[6] = value[6]
        if not np.isfinite(result).all():
            raise ValueError("EEF frame conversion produced non-finite values")
        return result

    def link6_to_gripper_center(self, pose: np.ndarray) -> np.ndarray:
        return self.transform(
            pose, source_frame=SDK_EEF_FRAME, target_frame=LEGACY_MODEL_EEF_FRAME
        )

    def gripper_center_to_link6(self, pose: np.ndarray) -> np.ndarray:
        return self.transform(
            pose, source_frame=LEGACY_MODEL_EEF_FRAME, target_frame=SDK_EEF_FRAME
        )


__all__ = [
    "apply_eef_pose_delta",
    "eef_pose_delta",
    "LEGACY_MODEL_EEF_FRAME",
    "PiperEEFFrameTransform",
    "SDK_EEF_FRAME",
    "SUPPORTED_EEF_FRAMES",
    "resolve_piper_urdf",
]
