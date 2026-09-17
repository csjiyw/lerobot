from __future__ import annotations

from argparse import Namespace

import cv2
import numpy as np

from cobot_bridge.collect_lerobot import _create_dataset, _frame, build_parser
from cobot_bridge.remote_robot import RemoteObservation


def test_pc_collection_writes_lerobot_episode(tmp_path):
    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    sample = {"position": [0.0] * 7, "fresh": True}
    observation = RemoteObservation(
        raw={
            "ready": True,
            "bridge_time_ns": 123,
            "control": {"teach_active": True, "effective_mode_id": 3, "epoch": 1},
            "master": {"left": sample},
            "puppet": {"left": sample},
            "commands": {"puppet": {"left": sample}},
        },
        image_bytes={"front": encoded.tobytes()},
        side="left",
    )
    args = Namespace(
        cameras=["front"],
        use_videos=False,
        repo_id="local/test_cobot_bridge",
        fps=20,
        root=tmp_path / "dataset",
        image_writer_threads=0,
        vcodec="h264",
    )
    dataset = _create_dataset(args, observation)
    dataset.add_frame(_frame(observation, np.zeros(7, dtype=np.float32), "test task", args.cameras))
    dataset.save_episode()
    dataset.finalize()

    assert (args.root / "meta" / "info.json").is_file()
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 1


def test_collection_episode_is_pc_controlled_by_default(tmp_path):
    args = build_parser().parse_args(
        [
            "--repo-id",
            "local/test",
            "--root",
            str(tmp_path / "dataset"),
            "--task",
            "test task",
        ]
    )

    assert args.episode_trigger == "pc"
    assert args.episode_seconds == 30.0
