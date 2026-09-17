#!/usr/bin/env python3
"""Build h30 RLT replay bundles from contiguous 30 Hz SmolVLA features.

The feature cache is produced with the frozen h30 Stage-1 VLA.  Each retained
30 Hz frame becomes one policy-tick transition and uses the dataset's executed
action chunk as behavior.  Episode bundles preserve success/failure balancing
for the existing Cobot Magic Stage-2 trainer.  This command is offline-only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .rlt_stage2_data import ACTION_CHUNK_TIMEBASE, REPLAY_FORMAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _scalar(value: object) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar metadata value, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _atomic_save(path: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    cache_path = args.feature_cache.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    stage1 = str(args.stage1_checkpoint.expanduser().resolve())
    with np.load(cache_path, allow_pickle=False) as stored:
        required = {"z_rl", "state", "reference", "target", "valid", "episode_index"}
        missing = required - set(stored.files)
        if missing:
            raise ValueError(f"feature cache is missing {sorted(missing)}")
        cache = {name: np.asarray(stored[name]) for name in required}
        if "is_success" in stored.files:
            cache["is_success"] = np.asarray(stored["is_success"], dtype=np.float32).reshape(-1)
        else:
            cache["is_success"] = np.ones(cache["z_rl"].shape[0], dtype=np.float32)
    n = int(cache["z_rl"].shape[0])
    if any(value.shape[0] != n for value in cache.values()):
        raise ValueError("feature cache fields have inconsistent row counts")
    if cache["reference"].shape != cache["target"].shape:
        raise ValueError("reference and target chunk shapes differ")
    if cache["reference"].ndim != 3 or cache["reference"].shape[1] != 30:
        raise ValueError(f"expected h30 chunks, got {cache['reference'].shape}")

    # Read one cheap metadata column per row; no video is decoded here.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        dataset_root.name,
        root=dataset_root,
        download_videos=False,
    )
    raw = dataset.hf_dataset.data
    if raw.num_rows != n:
        raise ValueError(f"dataset rows={raw.num_rows} do not match cache rows={n}")
    if "observation.is_human_intervention" in raw.column_names:
        intervention = np.asarray(
            [_scalar(value) for value in raw.column("observation.is_human_intervention").to_pylist()],
            dtype=np.float32,
        )
    else:
        intervention = np.zeros(n, dtype=np.float32)

    episode_index = cache["episode_index"].astype(np.int64)
    episodes = np.unique(episode_index)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for episode in episodes:
        rows = np.flatnonzero(episode_index == episode)
        usable = rows[cache["valid"][rows].astype(bool).all(axis=1)]
        usable = usable[usable + 1 < n]
        usable = usable[episode_index[np.minimum(usable + 1, n - 1)] == episode]
        if usable.size < 2:
            continue
        current = usable[:-1]
        nxt = usable[1:]
        # The cache is contiguous; do not silently bridge a dropped row.
        contiguous = (nxt == current + 1)
        current = current[contiguous]
        nxt = nxt[contiguous]
        if current.size < 1:
            continue
        success = bool(cache["is_success"][rows[0]] > 0.5)
        outcome = "success" if success else "failure"
        rewards = np.zeros((current.size, 1), dtype=np.float32)
        dones = np.zeros((current.size, 1), dtype=np.float32)
        rewards[-1, 0] = float(success)
        dones[-1, 0] = 1.0
        intervention_rows = intervention[current, None].astype(np.float32)
        metadata = {
            "format": REPLAY_FORMAT,
            "created_at_ns": time.time_ns(),
            "decision_path": str(cache_path),
            "stage1_checkpoint": stage1,
            "task": "pick and place the cross object",
            "workflow_mode": 2 if bool(intervention_rows.any()) else 1,
            "collection_phase": "actor",
            "outcome": outcome,
            "transition_count": int(current.size),
            "intervention_transitions": int(intervention_rows.sum()),
            "action_source_counts": {"actor": int(current.size - intervention_rows.sum()), "human": int(intervention_rows.sum()), "reference": 0},
            "source_decisions": int(rows.size),
            "selected_decision_start": 0,
            "selected_decision_end": int(rows.size),
            "trimmed_prefix_decisions": 0,
            "trimmed_suffix_decisions": int(rows.size - current.size - 1),
            "stationary_trim_enabled": False,
            "stationary_state_delta": 0.0,
            "action_space": "smolvla_normalized",
            "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
            "action_chunk_frame_stride": 1,
            "z_dim": int(cache["z_rl"].shape[1]),
            "state_dim": int(cache["state"].shape[1]),
            "action_steps": int(cache["target"].shape[1]),
            "action_dim": int(cache["target"].shape[2]),
            "source_dataset": str(dataset_root),
        }
        arrays = {
            "z_rl": cache["z_rl"][current].astype(np.float32),
            "state": cache["state"][current].astype(np.float32),
            "actions": cache["target"][current].astype(np.float32),
            "ref_actions": cache["reference"][current].astype(np.float32),
            "rewards": rewards,
            "next_z_rl": cache["z_rl"][nxt].astype(np.float32),
            "next_state": cache["state"][nxt].astype(np.float32),
            "next_ref_actions": cache["reference"][nxt].astype(np.float32),
            "dones": dones,
            "is_intervention": intervention_rows,
            "is_demo": np.zeros((current.size, 1), dtype=np.float32),
        }
        output = output_root / f"steam_h30_episode_{int(episode):04d}_{outcome}.npz"
        if output.exists():
            raise FileExistsError(output)
        _atomic_save(output, metadata, arrays)
        manifest.append({"episode": int(episode), "outcome": outcome, "transitions": int(current.size), "path": str(output)})

    summary = {
        "format": "cobot_magic_h30_replay_manifest_v1",
        "feature_cache": str(cache_path),
        "dataset_root": str(dataset_root),
        "stage1_checkpoint": stage1,
        "episodes": len(manifest),
        "success_episodes": sum(item["outcome"] == "success" for item in manifest),
        "failure_episodes": sum(item["outcome"] == "failure" for item in manifest),
        "transitions": sum(item["transitions"] for item in manifest),
        "items": manifest,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
