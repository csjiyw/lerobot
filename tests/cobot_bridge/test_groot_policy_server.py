from __future__ import annotations

import numpy as np

from cobot_bridge.groot_policy_server import _format_observation, _merge_action_chunk


def test_format_groot_observation_uses_right_arm_schema():
    images = {
        "front": np.zeros((12, 16, 3), dtype=np.uint8),
        "right": np.ones((12, 16, 3), dtype=np.uint8),
    }
    observation = _format_observation(
        np.arange(7, dtype=np.float32),
        images,
        cameras=("front", "right"),
        task="pick and place the object",
        pose_state_key="eef_pose",
    )

    assert observation["video"]["front"].shape == (1, 1, 12, 16, 3)
    assert observation["state"]["eef_pose"].shape == (1, 1, 6)
    assert observation["state"]["gripper"].shape == (1, 1, 1)
    assert observation["language"]["annotation.human.task_description"] == [
        ["pick and place the object"]
    ]


def test_merge_groot_eef_and_gripper_action_chunk():
    actions = {
        "eef_pose": np.zeros((1, 16, 6), dtype=np.float32),
        "gripper": np.full((1, 16, 1), 0.04, dtype=np.float32),
    }

    chunk = _merge_action_chunk(actions, pose_action_key="eef_pose")

    assert chunk.shape == (16, 7)
    np.testing.assert_allclose(chunk[:, 6], 0.04)
