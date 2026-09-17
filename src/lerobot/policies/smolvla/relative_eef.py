"""GR00T-style chunk-relative EEF transforms.

Every pose in an action chunk is expressed relative to the single EEF state
that conditioned the policy: ``T_relative[h] = inv(T_state) @ T_target[h]``.
The gripper remains an absolute command.
"""

from __future__ import annotations

import numpy as np


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = value
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    cross = _skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        ) * 0.5
    if np.pi - angle < 1e-6:
        _, vectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = vectors[:, -1]
        return axis / max(float(np.linalg.norm(axis)), 1e-12) * angle
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    ) / max(2.0 * np.sin(angle), 1e-12)
    return axis * angle


def chunk_to_relative(state: np.ndarray, targets: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(7)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 7:
        raise ValueError(f"EEF targets must have shape (H, 7), got {targets.shape}")
    state_rotation = rotvec_to_matrix(state[3:6])
    result = np.empty_like(targets, dtype=np.float32)
    for index, target in enumerate(targets):
        target_rotation = rotvec_to_matrix(target[3:6])
        result[index, :3] = state_rotation.T @ (target[:3] - state[:3])
        result[index, 3:6] = matrix_to_rotvec(state_rotation.T @ target_rotation)
        result[index, 6] = target[6]
    return result


def relative_chunk_to_absolute(state: np.ndarray, relative: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(7)
    relative = np.asarray(relative, dtype=np.float64)
    if relative.ndim != 2 or relative.shape[1] != 7:
        raise ValueError(f"relative EEF chunk must have shape (H, 7), got {relative.shape}")
    state_rotation = rotvec_to_matrix(state[3:6])
    result = np.empty_like(relative, dtype=np.float32)
    for index, action in enumerate(relative):
        result[index, :3] = state[:3] + state_rotation @ action[:3]
        result[index, 3:6] = matrix_to_rotvec(state_rotation @ rotvec_to_matrix(action[3:6]))
        result[index, 6] = action[6]
    return result

