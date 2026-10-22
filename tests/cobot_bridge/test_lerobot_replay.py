from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cobot_bridge.lerobot_replay import export_episode_to_raw


def _make_dataset(root: Path) -> np.ndarray:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    names = [
        "l-j1", "l-j2", "l-j3", "l-j4", "l-j5", "l-j6",
        "r-j1", "r-j2", "r-j3", "r-j4", "r-j5", "r-j6",
        "left_gripper", "right_gripper",
    ]
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 3,
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "action_joint": {"names": {"motors": names}},
                },
            }
        )
    )
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["insert the workpiece"]}) + "\n"
    )
    actions = np.arange(3 * len(names), dtype=np.float32).reshape(3, len(names))
    table = pa.table(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "action_joint": [row.tolist() for row in actions],
        }
    )
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(table, parquet)
    return actions


def test_export_episode_uses_named_right_arm_dimensions_and_raw_replay_schema(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_actions = _make_dataset(dataset_root)
    output = tmp_path / "replay.raw.h5"

    metadata = export_episode_to_raw(dataset_root, 0, output, side="right")

    expected = source_actions[:, [6, 7, 8, 9, 10, 11, 13]]
    assert metadata["frames"] == 3
    assert metadata["fps"] == 30
    assert metadata["task"] == "insert the workpiece"
    assert metadata["joint_dimensions"] == [
        "r-j1", "r-j2", "r-j3", "r-j4", "r-j5", "r-j6", "right_gripper"
    ]
    with h5py.File(output, "r") as raw:
        assert bool(raw.attrs["completed"]) is True
        assert raw["frames/puppet_command"].shape == (3, 7)
        np.testing.assert_array_equal(raw["frames/puppet_command"][:], expected)
        np.testing.assert_array_equal(
            raw["frames/pc_timestamp_ns"][:],
            np.asarray([0, 33_333_333, 66_666_666], dtype=np.int64),
        )
        assert raw["frames/puppet_command_valid"][:].tolist() == [1, 1, 1]


def test_export_episode_rejects_episode_index_out_of_range(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)

    with pytest.raises(ValueError, match="out of range"):
        export_episode_to_raw(dataset_root, 1, tmp_path / "invalid.raw.h5")
