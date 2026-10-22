"""Export a LeRobot episode's joint targets to the raw replay format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .record_raw import FORMAT


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = info.get("features", {}).get(key, {})
    names = feature.get("names") if isinstance(feature, dict) else None
    if isinstance(names, dict):
        names = names.get("motors")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"dataset metadata is missing named dimensions for {key}")
    return names


def export_episode_to_raw(
    dataset_root: Path,
    episode_index: int,
    output_path: Path,
    *,
    side: str = "right",
) -> dict[str, Any]:
    """Write one LeRobot ``action_joint`` arm trajectory as a replayable H5.

    The replay vector is six absolute joint targets plus that arm's gripper
    target. The left/right dimension indices are resolved from metadata names,
    never assumed from column order.
    """

    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    root = dataset_root.resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"not a LeRobot dataset (missing {info_path})")
    info = json.loads(info_path.read_text())
    total_episodes = int(info.get("total_episodes", 0))
    if episode_index < 0 or episode_index >= total_episodes:
        raise ValueError(
            f"episode index {episode_index} out of range [0, {total_episodes})"
        )

    feature_key = "action_joint"
    dimension_names = _feature_names(info, feature_key)
    expected = [f"{side[0]}-j{joint}" for joint in range(1, 7)]
    gripper_name = f"{side}_gripper"
    missing = [name for name in [*expected, gripper_name] if name not in dimension_names]
    if missing:
        raise ValueError(f"{feature_key} does not contain required dimensions: {missing}")
    dimension_indices = [dimension_names.index(name) for name in [*expected, gripper_name]]

    fps = int(info.get("fps", 0))
    if fps <= 0:
        raise ValueError("dataset metadata has invalid fps")
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError("dataset metadata has invalid chunks_size")
    template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    relative_data = template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    parquet_path = (root / relative_data).resolve()
    try:
        parquet_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("dataset data_path escapes the dataset root") from exc
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - LeRobot runtime dependency
        raise RuntimeError("pyarrow is required to replay LeRobot datasets") from exc

    table = pq.read_table(parquet_path)
    if "episode_index" in table.column_names:
        episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
        mask = episode_ids == episode_index
        if not mask.any():
            raise ValueError(f"episode file contains no rows for episode {episode_index}")
        table = table.filter(mask)
    if feature_key not in table.column_names:
        raise ValueError(f"episode parquet is missing {feature_key}")
    if "frame_index" in table.column_names:
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
        table = table.take(np.argsort(frame_indices, kind="stable"))

    all_actions = np.asarray(table[feature_key].to_pylist(), dtype=np.float32)
    if all_actions.ndim != 2 or all_actions.shape[1] != len(dimension_names):
        raise ValueError(
            f"unexpected {feature_key} shape {all_actions.shape}; metadata declares "
            f"{len(dimension_names)} dimensions"
        )
    actions = all_actions[:, dimension_indices]
    if len(actions) < 2 or not np.isfinite(actions).all():
        raise ValueError("episode must contain at least two finite joint-action frames")

    episode_task = f"LeRobot episode {episode_index}"
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        with episodes_path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                episode_meta = json.loads(line)
                if int(episode_meta.get("episode_index", -1)) == episode_index:
                    tasks = episode_meta.get("tasks", [])
                    if tasks:
                        episode_task = str(tasks[0])
                    break

    timestamps_ns = np.arange(len(actions), dtype=np.int64) * (1_000_000_000 // fps)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as raw:
        raw.attrs["format"] = FORMAT
        raw.attrs["completed"] = True
        raw.attrs["outcome"] = "replay_source"
        raw.attrs["task"] = episode_task
        raw.attrs["side"] = side
        raw.attrs["fps"] = fps
        raw.attrs["num_frames"] = len(actions)
        raw.attrs["collection_type"] = "REPLAY"
        raw.attrs["source_dataset"] = str(root)
        raw.attrs["source_episode_index"] = int(episode_index)
        raw.attrs["source_feature"] = feature_key
        raw.attrs["source_side"] = side
        raw.attrs["source_fps"] = fps
        frames = raw.create_group("frames")
        frames.create_dataset("pc_timestamp_ns", data=timestamps_ns)
        frames.create_dataset("puppet_command", data=actions)
        frames.create_dataset(
            "puppet_command_valid", data=np.ones(len(actions), dtype=np.uint8)
        )

    return {
        "path": str(output_path),
        "frames": int(len(actions)),
        "fps": fps,
        "task": episode_task,
        "side": side,
        "source_feature": feature_key,
        "joint_dimensions": [*expected, gripper_name],
    }
