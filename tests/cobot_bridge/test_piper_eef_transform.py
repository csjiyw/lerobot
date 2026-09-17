from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pinocchio")

from cobot_bridge.piper_eef_metrics import DEFAULT_PIPER_URDF, PiperForwardKinematics
from cobot_bridge.piper_eef_transform import (
    LEGACY_MODEL_EEF_FRAME,
    PiperEEFFrameTransform,
    SDK_EEF_FRAME,
)


def _pose_from_fk(position: np.ndarray, rotation: np.ndarray, pin) -> np.ndarray:
    return np.concatenate(
        [position, np.asarray(pin.log3(rotation), dtype=np.float64).reshape(3), [0.08]]
    ).astype(np.float32)


def test_link6_to_model_frame_matches_official_fk() -> None:
    assert Path(DEFAULT_PIPER_URDF).is_file()
    fk = PiperForwardKinematics(frames=(SDK_EEF_FRAME, LEGACY_MODEL_EEF_FRAME))
    converter = PiperEEFFrameTransform(DEFAULT_PIPER_URDF)
    joints = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.15, 0.35, -0.45, 0.2, 0.4, -0.25],
            [-0.3, 0.6, -0.8, -0.35, 0.7, 0.2],
        ],
        dtype=np.float64,
    )
    poses = fk.forward(joints)
    link_positions, link_rotations = poses[SDK_EEF_FRAME]
    model_positions, model_rotations = poses[LEGACY_MODEL_EEF_FRAME]

    for index in range(len(joints)):
        link_pose = _pose_from_fk(link_positions[index], link_rotations[index], fk.pin)
        converted = converter.link6_to_gripper_center(link_pose)
        np.testing.assert_allclose(converted[:3], model_positions[index], atol=2e-7)
        # Round-tripping the pose is the public check for the orientation too.
        round_trip = converter.gripper_center_to_link6(converted)
        np.testing.assert_allclose(round_trip[:3], link_pose[:3], atol=2e-7)
        np.testing.assert_allclose(
            fk.pin.exp3(round_trip[3:6]), link_rotations[index], atol=2e-6
        )
        np.testing.assert_allclose(round_trip[6], link_pose[6], atol=1e-7)


def test_model_frame_translation_is_not_a_world_xyz_offset() -> None:
    converter = PiperEEFFrameTransform(DEFAULT_PIPER_URDF)
    pose_a = np.asarray([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 0.04], dtype=np.float32)
    pose_b = np.asarray([0.2, -0.1, 0.3, 0.0, np.pi / 2.0, 0.0, 0.04], dtype=np.float32)
    model_a = converter.link6_to_gripper_center(pose_a)
    model_b = converter.link6_to_gripper_center(pose_b)
    assert not np.allclose(model_a[:3] - pose_a[:3], model_b[:3] - pose_b[:3])


def test_same_frame_conversion_is_identity() -> None:
    converter = PiperEEFFrameTransform(DEFAULT_PIPER_URDF)
    pose = np.asarray([0.1, 0.2, 0.3, 0.1, -0.2, 0.3, 0.06], dtype=np.float32)
    np.testing.assert_array_equal(
        converter.transform(
            pose, source_frame=SDK_EEF_FRAME, target_frame=SDK_EEF_FRAME
        ),
        pose,
    )
