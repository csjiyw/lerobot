from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cobot_bridge.groot_policy_server import (
    _format_observation,
    _merge_action_chunk,
    _resolve_camera_key_map,
    _resolve_checkpoint_embodiment_tag,
)


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


def test_embodiment_tag_comes_from_saved_run_config_not_robot_type(tmp_path: Path):
    processor = tmp_path / "processor_config.json"
    processor.write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "modality_configs": {
                        "cobotmagic": {"state": {"modality_keys": ["eef_pose", "gripper"]}},
                        "new_embodiment": {"state": {"modality_keys": ["arm"]}},
                    }
                }
            }
        )
    )
    run_config = tmp_path / "experiment_cfg"
    run_config.mkdir()
    (run_config / "conf.yaml").write_text(
        "data:\n  datasets:\n    - embodiment_tag: cobotmagic\n      dataset_type: physical_embodiment\n"
    )

    assert _resolve_checkpoint_embodiment_tag(tmp_path) == "cobotmagic"


def test_bridge_cameras_map_to_saved_checkpoint_video_keys():
    mapping = _resolve_camera_key_map(
        ("cam_high", "cam_right_wrist"), ("front", "right")
    )
    assert mapping == {"front": "cam_high", "right": "cam_right_wrist"}
    observation = _format_observation(
        np.zeros(7, dtype=np.float32),
        {
            "front": np.zeros((12, 16, 3), dtype=np.uint8),
            "right": np.ones((12, 16, 3), dtype=np.uint8),
        },
        cameras=("front", "right"),
        task="test",
        pose_state_key="eef_pose",
        camera_key_map=mapping,
    )
    assert set(observation["video"]) == {"cam_high", "cam_right_wrist"}
