#!/usr/bin/env python3
"""Create a one-frame LeRobot image dataset for an overfit sanity test.

The source action remains a per-frame action. With chunk30 training, LeRobot
will apply its normal end-of-episode padding to the one-frame episode. The
source full-dataset stats are retained so this test exercises the same
normalization convention as the real training run.
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-episode", type=int, default=0)
    parser.add_argument("--source-frame", type=int, default=0)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_root}")

    source = LeRobotDataset(
        "local/cobot_magic_il_cross_073_clean_arm_state_t3_cmd_gripper_v3_eef",
        root=args.source_root,
        episodes=[args.source_episode],
        video_backend="torchcodec",
    )
    item = source[args.source_frame]

    features = {}
    for key, feature in source.features.items():
        if key in DEFAULT_FEATURES:
            continue
        out_feature = copy.deepcopy(feature)
        if out_feature["dtype"] == "video":
            out_feature["dtype"] = "image"
            out_feature.pop("info", None)
        features[key] = out_feature

    tiny = LeRobotDataset.create(
        repo_id="local/cobot_magic_il_cross_073_single_frame_ep0",
        root=args.output_root,
        fps=source.fps,
        features=features,
        robot_type=source.meta.info.get("robot_type"),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    # LeRobot 0.4.0 initializes this only in __init__, not in create().
    # The one-frame fixture stores camera observations as PNG images.
    tiny.img_format = "png"

    frame = {
        "task": item["task"],
        "timestamp": float(item["timestamp"].item()),
    }
    for key in features:
        frame[key] = item[key]

    tiny.add_frame(frame)
    tiny.save_episode(parallel_encoding=False)
    tiny.finalize()

    source_stats = args.source_root / "meta" / "stats.json"
    if source_stats.exists():
        shutil.copy2(source_stats, args.output_root / "meta" / "stats.json")

    loaded = LeRobotDataset(
        "local/cobot_magic_il_cross_073_single_frame_ep0",
        root=args.output_root,
        video_backend="torchcodec",
    )
    print(f"created={args.output_root}")
    print(f"frames={len(loaded)} episodes={loaded.num_episodes} task={loaded[0]['task']}")
    for key in loaded.meta.camera_keys:
        print(f"{key}_shape={tuple(loaded[0][key].shape)}")


if __name__ == "__main__":
    main()
