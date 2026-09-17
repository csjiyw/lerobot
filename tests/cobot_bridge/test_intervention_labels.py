from __future__ import annotations

import json
import sys

import cv2
import h5py
import numpy as np
import pandas as pd

import cobot_bridge.record_raw as record_raw
from cobot_bridge.convert_raw_to_lerobot import (
    derive_control_labels,
    hybrid_state_t3_action,
    main as convert_main,
    trim_episode_bounds,
)
from cobot_bridge.record_raw import (
    CONTROL_ALIGNMENT,
    CONTROL_HUMAN,
    CONTROL_OTHER,
    CONTROL_POLICY,
    RawEpisodeWriter,
    classify_control_source,
    read_result_outcome,
)
from cobot_bridge.remote_robot import RemoteObservation


def test_control_source_classification():
    assert classify_control_source(teach_active=False, effective_mode=0) == CONTROL_OTHER
    assert classify_control_source(teach_active=False, effective_mode=2) == CONTROL_POLICY
    assert classify_control_source(teach_active=False, effective_mode=5) == CONTROL_ALIGNMENT
    assert classify_control_source(teach_active=True, effective_mode=2) == CONTROL_HUMAN
    assert classify_control_source(teach_active=False, effective_mode=3) == CONTROL_HUMAN


def test_pc_result_file_accepts_explicit_failure(tmp_path):
    result_file = tmp_path / "episode.result"
    result_file.write_text("failure")
    assert read_result_outcome(result_file) == "failure"


def test_pc_result_file_accepts_discard_command(tmp_path):
    result_file = tmp_path / "episode.result"
    result_file.write_text("discard")
    assert read_result_outcome(result_file) == "discard"


def test_recorder_discard_removes_all_episode_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "episode.raw.h5"
    result_file = tmp_path / "episode.raw.h5.result"
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    sample = {"position": [0.0] * 7, "stamp_ns": 123, "fresh": True}
    observation = RemoteObservation(
        raw={
            "ready": True,
            "bridge_time_ns": 100,
            "sequence": 1,
            "control": {"teach_active": False, "effective_mode_id": 0, "epoch": 1},
            "master": {"right": sample},
            "puppet": {"right": sample},
            "commands": {"puppet": {"right": sample}},
            "images": {"front": {"stamp_ns": 200}, "right": {"stamp_ns": 300}},
        },
        image_bytes={"front": jpeg.tobytes(), "right": jpeg.tobytes()},
        side="right",
    )

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class FakeRobot:
        def __init__(self, *_args, **_kwargs):
            pass

        def observe(self):
            result_file.write_text("discard")
            return observation

    monkeypatch.setattr(record_raw, "CobotBridgeClient", FakeClient)
    monkeypatch.setattr(record_raw, "CobotMagicRemoteRobot", FakeRobot)
    monkeypatch.setattr(record_raw, "STOP", False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_raw",
            "--endpoint",
            "tcp://127.0.0.1:1",
            "--output",
            str(output),
            "--stop-file",
            str(result_file),
            "--task",
            "discard test",
            "--side",
            "right",
            "--cameras",
            "front",
            "right",
            "--fps",
            "1000",
            "--min-frames",
            "1",
        ],
    )

    record_raw.main()

    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()
    assert not result_file.exists()


def test_intervention_edges_and_ids_are_explicit():
    teach = np.asarray([0, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    modes = np.asarray([2, 2, 3, 3, 4, 2, 3, 4], dtype=np.int64)

    labels = derive_control_labels(teach, modes)

    np.testing.assert_array_equal(labels["teach_event"], [0, 0, 1, 0, -1, 0, 1, -1])
    np.testing.assert_array_equal(labels["intervention_event"], [0, 0, 1, 0, -1, 0, 1, -1])
    np.testing.assert_array_equal(labels["intervention_id"], [0, 0, 1, 1, 1, 0, 2, 2])
    np.testing.assert_array_equal(labels["is_policy"], [1, 1, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(
        labels["is_human_intervention"], [0, 0, 1, 1, 0, 0, 1, 0]
    )


def test_effective_teach_mode_drives_intervention_boundary_before_delayed_topic():
    teach = np.asarray([0, 0, 0, 1, 1, 0], dtype=np.uint8)
    modes = np.asarray([2, 2, 3, 3, 3, 4], dtype=np.int64)

    labels = derive_control_labels(teach, modes)

    np.testing.assert_array_equal(labels["teach_event"], [0, 0, 0, 1, 0, -1])
    np.testing.assert_array_equal(labels["intervention_event"], [0, 0, 1, 0, 0, -1])
    np.testing.assert_array_equal(labels["intervention_id"], [0, 0, 1, 1, 1, 1])


def test_il_trim_removes_preteach_and_stationary_holds():
    timestamps = np.arange(12, dtype=np.int64) * 1_000_000_000
    trajectory = np.zeros((12, 7), dtype=np.float64)
    trajectory[4:7, 0] = [0.02, 0.05, 0.08]
    trajectory[7:, 0] = 0.1
    teach = np.zeros(12, dtype=np.uint8)
    teach[2:11] = 1

    start, end, info = trim_episode_bounds(
        timestamps,
        trajectory,
        teach,
        collection_type="IL",
        joint_tolerance=0.01,
        gripper_tolerance=0.002,
        settle_seconds=0.0,
    )

    assert (start, end) == (4, 8)
    assert info["initial_wait_removed_s"] == 2.0
    assert info["initial_steady_removed_s"] == 2.0
    assert info["final_steady_removed_s"] == 3.0
    assert info["post_teach_removed_s"] == 1.0


def test_hybrid_t3_action_uses_future_arm_and_current_gripper_command():
    puppet = np.arange(42, dtype=np.float32).reshape(6, 7) / 100.0
    executed = np.full((6, 7), -1.0, dtype=np.float32)
    executed[:, 6] = np.linspace(0.08, 0.03, 6, dtype=np.float32)
    valid = np.ones(6, dtype=np.uint8)

    action = hybrid_state_t3_action(
        puppet,
        executed,
        valid,
        index=1,
        future_index=4,
    )

    np.testing.assert_allclose(action[:6], puppet[4, :6])
    assert action[6] == executed[1, 6]


def test_hybrid_t3_action_rejects_missing_gripper_command():
    puppet = np.zeros((5, 7), dtype=np.float32)
    executed = np.zeros((5, 7), dtype=np.float32)
    valid = np.ones(5, dtype=np.uint8)
    valid[0] = 0

    with np.testing.assert_raises_regex(ValueError, "gripper command is invalid"):
        hybrid_state_t3_action(
            puppet,
            executed,
            valid,
            index=0,
            future_index=3,
        )


def test_raw_intervention_labels_convert_to_lerobot(tmp_path, monkeypatch):
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    raw_path = tmp_path / "episode.raw.h5"
    writer = RawEpisodeWriter(
        raw_path,
        side="right",
        cameras=("front", "right"),
        task="test intervention",
        fps=30,
        flush_frames=1,
        collection_type="RL",
    )
    sample = {"position": [0.0] * 7, "stamp_ns": 123, "fresh": True}
    for sequence, (teach, mode) in enumerate(((False, 2), (True, 3), (False, 4))):
        observation = RemoteObservation(
            raw={
                "ready": True,
                "bridge_time_ns": sequence + 100,
                "sequence": sequence,
                "control": {
                    "teach_active": teach,
                    "effective_mode_id": mode,
                    "epoch": sequence,
                },
                "master": {"right": sample},
                "puppet": {"right": sample},
                "commands": {"puppet": {"right": sample}},
                "images": {
                    "front": {"stamp_ns": sequence + 200},
                    "right": {"stamp_ns": sequence + 300},
                },
            },
            image_bytes={"front": jpeg.tobytes(), "right": jpeg.tobytes()},
            side="right",
        )
        writer.append(observation)
    writer.close(completed=True, stop_reason="teach_released")

    with h5py.File(raw_path, "r") as raw:
        np.testing.assert_array_equal(raw["frames/control_source"][:], [1, 2, 0])
        np.testing.assert_array_equal(raw["frames/teach_event"][:], [0, 1, -1])
        np.testing.assert_array_equal(raw["frames/intervention_id"][:], [0, 1, 1])
        np.testing.assert_array_equal(raw["frames/intervention_event"][:], [0, 1, -1])
        np.testing.assert_array_equal(raw["frames/learning_mask"][:], [1, 1, 0])
        np.testing.assert_array_equal(raw["frames/policy_action_valid"][:], [1, 0, 0])
        np.testing.assert_array_equal(raw["frames/human_action_valid"][:], [0, 1, 0])
        np.testing.assert_array_equal(raw["frames/executed_action_valid"][:], [1, 1, 1])
        np.testing.assert_array_equal(raw["frames/done"][:], [0, 0, 1])
        np.testing.assert_array_equal(raw["frames/success"][:], [-1, -1, 1])
        np.testing.assert_array_equal(raw["frames/reward_valid"][:], [0, 0, 1])
        np.testing.assert_allclose(raw["frames/reward"][:], [0.0, 0.0, 1.0])
        assert raw.attrs["stop_reason"] == "teach_released"
        assert raw.attrs["outcome"] == "success"
        assert raw.attrs["collection_type"] == "RL"
        assert raw.attrs["collection_type_id"] == 1

    root = tmp_path / "lerobot"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_raw_to_lerobot",
            "--raw",
            str(raw_path),
            "--repo-id",
            "local/test_intervention",
            "--root",
            str(root),
            "--task",
            "canonical intervention task",
            "--no-use-videos",
            "--image-writer-threads",
            "0",
        ],
    )
    convert_main()
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert tasks.index[0] == "canonical intervention task"
    conversion = json.loads((root / "meta" / "cobot_magic_conversion.json").read_text())
    assert conversion["episodes"][0]["raw_task"] == "test intervention"
    assert conversion["episodes"][0]["task"] == "canonical intervention task"
    info = json.loads((root / "meta" / "info.json").read_text())
    for key in (
        "observation.control_source",
        "observation.is_policy",
        "observation.is_human_intervention",
        "observation.intervention_id",
        "observation.teach_event",
        "observation.effective_mode",
        "observation.control_epoch",
        "rlt.reference_action",
        "rlt.policy_action",
        "rlt.human_action",
        "rlt.executed_action",
        "rlt.phase",
        "rlt.cycle_id",
        "rlt.learning_mask",
        "rlt.intervention_event",
        "rlt.reward",
        "rlt.done",
        "rlt.success",
        "rlt.collection_type",
    ):
        assert key in info["features"]
