"""Make a native LeRobot v3 right-arm joint view without re-encoding videos."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH, flatten_dict,
    write_episodes, write_info, write_stats, write_tasks,
)
from .repair_state_derived_gripper import dimension_names, statistics


def prepare(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("output must be a new directory outside the source")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("expected a v2.1 source")
    names = [f"r-j{i}" for i in range(1, 7)] + ["right_gripper"]
    indices = {key: [dimension_names(info, key).index(n) for n in names]
               for key in ("observation.state", "action")}
    cameras = {"observation.images.head": "observation.images.camera_f",
               "observation.images.hand_right": "observation.images.camera_r"}
    scalar_keys = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    features = {key: copy.deepcopy(info["features"][key]) for key in scalar_keys}
    for key in indices:
        features[key] = {"dtype": "float32", "shape": [7],
                         "names": [f"joint{i}" for i in range(7)]}
    for old, new in cameras.items():
        features[new] = copy.deepcopy(info["features"][old])
    tasks = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
    task_names = {t["task_index"]: t["task"] for t in tasks}
    output.mkdir(parents=True)
    write_tasks(pd.DataFrame({"task_index": [t["task_index"] for t in tasks]},
                            index=[t["task"] for t in tasks]), output)
    rows, values = [], {key: [] for key in indices}
    offset = 0
    paths = sorted((source / "data").rglob("*.parquet"))
    for path in paths:
        table = pq.read_table(path)
        episode = int(table["episode_index"][0].as_py())
        if episode != len(rows):
            raise ValueError("episode indices must be contiguous")
        data = {key: table[key] for key in scalar_keys}
        # The downloaded head-trimmed dataset restarts `index` at zero in
        # every episode. Native v3 uses a global contiguous row index.
        data["index"] = pa.array(np.arange(offset, offset + len(table)), type=pa.int64())
        assert np.array_equal(np.asarray(table["frame_index"].to_pylist()), np.arange(len(table)))
        episode_stats = {}
        for key, selected in indices.items():
            array = np.asarray(table[key].to_pylist(), dtype=np.float32)[:, selected]
            data[key] = pa.array(array.tolist(), type=pa.list_(pa.float32(), 7))
            episode_stats[key] = statistics(array)
            values[key].append(array)
        dest = output / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=episode)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), dest, compression="snappy")
        row = {"episode_index": episode, "length": len(table),
               "tasks": [task_names[i] for i in sorted(set(table["task_index"].to_pylist()))],
               "data/chunk_index": 0, "data/file_index": episode,
               "dataset_from_index": offset, "dataset_to_index": offset + len(table),
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
               **flatten_dict({"stats": episode_stats})}
        for old, new in cameras.items():
            old_path = source / info["video_path"].format(
                episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=old)
            if not old_path.is_file():
                raise FileNotFoundError(old_path)
            video_dest = output / DEFAULT_VIDEO_PATH.format(video_key=new, chunk_index=0, file_index=episode)
            video_dest.parent.mkdir(parents=True, exist_ok=True)
            video_dest.symlink_to(old_path.resolve())
            row.update({f"videos/{new}/chunk_index": 0,
                        f"videos/{new}/file_index": episode,
                        f"videos/{new}/from_timestamp": 0.0,
                        f"videos/{new}/to_timestamp": len(table) / info["fps"]})
        rows.append(row)
        offset += len(table)
    assert len(rows) == info["total_episodes"] and offset == info["total_frames"]
    write_episodes(Dataset.from_list(rows), output)
    write_stats({key: statistics(np.concatenate(arrays)) for key, arrays in values.items()}, output)
    new_info = {"codebase_version": "v3.0", "robot_type": "cobot_magic",
                "total_episodes": len(rows), "total_frames": offset,
                "total_tasks": len(tasks), "chunks_size": 1000, "fps": info["fps"],
                "splits": {"train": f"0:{len(rows)}"}, "data_path": DEFAULT_DATA_PATH,
                "video_path": DEFAULT_VIDEO_PATH, "features": features,
                "data_files_size_in_mb": 100, "video_files_size_in_mb": 500,
                "action_space": "joint_absolute",
                "source_dataset": str(source)}
    write_info(new_info, output)
    contract = {"source": str(source), "state_indices": indices["observation.state"],
                "action_indices": indices["action"], "dimension_names": names,
                "state": "measured right arm joints and gripper aperture",
                "action": "absolute right-arm state[t+3] joint targets + inferred gripper close target",
                "camera_mapping": cameras, "task_texts": tasks,
                "episodes": len(rows), "frames": offset,
                "trim": "no additional trimming; original head-only trim and complete tails retained",
                "stats": "recomputed from the exact projected state/action data",
                "videos": "original decoded pixel data, shared by symlink; no resize or re-encode",
                "gripper_repair_report": str(source / "meta/gripper_action_repair.json")}
    (output / "meta/training_view_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        "# SmolVLA abs joint 训练数据\n\n"
        f"源数据：`{source}`\n\n"
        f"全部 {len(rows)} 条、{offset} 帧，右臂 joint1–6 + gripper，30 FPS。"
        "action 是绝对关节目标，夹持段使用修正后的闭合目标；state 保留真实反馈。"
        "不再去头去尾，完整保留现有结束/归位段。\n\n"
        "图像使用 head + hand_right，映射为 camera_f + camera_r。视频为符号链接，"
        "训练时使用 256×256 等比缩放加 padding，保留数据的原始任务文本。\n\n"
        "数值数据与统计为独立副本；视频依赖源数据，迁移机器时需复制链接目标。"
        "字段选择和数据来源见 `meta/training_view_contract.json`。\n")
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False, indent=2))
