"""Add successful IL demonstrations to a Cobot Magic Stage 2 replay store.

The Actor-BC feature cache already contains tensors produced by the exact
frozen Stage-1 policy and normalizer.  This command samples complete IL
episodes at the rollout decision rate and writes immutable replay bundles.
Intermediate rewards are zero; only the terminal transition receives reward 1.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .rlt_stage2_data import (
    ACTION_CHUNK_FRAME_STRIDE,
    ACTION_CHUNK_TIMEBASE,
    REPLAY_FORMAT,
    load_replay_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-episode-fraction", type=float, default=0.1)
    parser.add_argument(
        "--decision-stride-frames",
        type=int,
        default=30,
        help="IL frames between replay decisions (30 frames = 1 Hz at 30 FPS).",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help="Truncate cached action chunks to this horizon (e.g. 10 for the "
        "remote chunk10 recipe on a chunk30 VLA). Must be <= the cached "
        "chunk length.",
    )
    parser.add_argument(
        "--success-episodes",
        type=int,
        help="Number to add; by default fills success episodes up to failure count.",
    )
    parser.add_argument(
        "--include-validation-episodes",
        action="store_true",
        help="Use all IL episodes, including the Actor-BC validation split.",
    )
    return parser


def split_episode_ids(
    episode_ids: np.ndarray, *, seed: int, val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if unique.size < 2 or not 0 < val_fraction < 1:
        raise ValueError("need at least two episodes and val_fraction in (0, 1)")
    shuffled = np.random.default_rng(seed).permutation(unique)
    val_count = max(1, min(unique.size - 1, round(unique.size * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


def decision_rows(rows: np.ndarray, *, stride: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 2 or stride <= 0:
        raise ValueError("an IL demo requires at least two frames and positive stride")
    selected = rows[::stride]
    if selected[-1] != rows[-1]:
        selected = np.append(selected, rows[-1])
    if selected.size < 2:
        selected = rows[[0, -1]]
    return selected


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_bundle(
    output_path: Path,
    *,
    episode_id: int,
    selected: np.ndarray,
    cache: dict[str, np.ndarray],
    stage1_checkpoint: Path,
    dataset_root: Path,
    stride: int,
) -> dict[str, Any]:
    transition_count = int(selected.size - 1)
    current = selected[:-1]
    following = selected[1:]
    rewards = np.zeros((transition_count, 1), dtype=np.float32)
    dones = np.zeros((transition_count, 1), dtype=np.float32)
    rewards[-1, 0] = 1.0
    dones[-1, 0] = 1.0
    arrays = {
        "z_rl": cache["z_rl"][current],
        "state": cache["state"][current],
        # Ground-truth normalized IL chunks are both the replay behavior action
        # and the Actor BC target.  The frozen VLA remains a separate input.
        "actions": cache["target"][current],
        "ref_actions": cache["reference"][current],
        "rewards": rewards,
        "next_z_rl": cache["z_rl"][following],
        "next_state": cache["state"][following],
        "next_ref_actions": cache["reference"][following],
        "dones": dones,
        "is_intervention": np.zeros((transition_count, 1), dtype=np.float32),
        "is_demo": np.ones((transition_count, 1), dtype=np.float32),
    }
    metadata = {
        "format": REPLAY_FORMAT,
        "created_at_ns": time.time_ns(),
        "decision_path": None,
        "dataset_root": str(dataset_root),
        "source_episode_index": int(episode_id),
        "stage1_checkpoint": str(stage1_checkpoint),
        "task": "IL successful demonstration",
        "workflow_mode": 1,
        "collection_phase": "il_demo",
        "outcome": "success",
        "transition_count": transition_count,
        "intervention_transitions": 0,
        "demo_transitions": transition_count,
        "action_source_counts": {"reference": 0, "actor": 0, "human": 0, "demo": int(selected.size)},
        "source_decisions": int(selected.size),
        "selected_decision_start": 0,
        "selected_decision_end": int(selected.size),
        "trimmed_prefix_decisions": 0,
        "trimmed_suffix_decisions": 0,
        "stationary_trim_enabled": False,
        "decision_stride_frames": int(stride),
        "action_space": "smolvla_normalized",
        "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
        "action_chunk_frame_stride": ACTION_CHUNK_FRAME_STRIDE,
        "z_dim": int(cache["z_rl"].shape[1]),
        "state_dim": int(cache["state"].shape[1]),
        "action_steps": int(cache["target"].shape[1]),
        "action_dim": int(cache["target"].shape[2]),
    }
    temporary = output_path.with_name(output_path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(output_path.with_suffix(output_path.suffix + ".json"), metadata)
    return metadata


def _dataset_episode_index(dataset_root: Path) -> np.ndarray:
    parquet_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet data under {dataset_root}")
    parts = [
        np.asarray(pq.read_table(path, columns=["episode_index"])["episode_index"])
        .reshape(-1)
        .astype(np.int64)
        for path in parquet_paths
    ]
    return np.concatenate(parts)


def main() -> None:
    args = build_parser().parse_args()
    cache_path = args.cache_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    replay_root = args.replay_root.expanduser().resolve()
    if args.decision_stride_frames <= 0:
        raise ValueError("decision_stride_frames must be positive")
    if not 0 < args.val_episode_fraction < 1:
        raise ValueError("val_episode_fraction must be in (0, 1)")
    replay_root.mkdir(parents=True, exist_ok=True)

    with np.load(cache_path, allow_pickle=False) as stored:
        cache = {name: np.asarray(stored[name]) for name in stored.files}
    cache_metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_metadata_path.is_file():
        raise FileNotFoundError(f"Actor BC cache metadata is missing: {cache_metadata_path}")
    cache_metadata = json.loads(cache_metadata_path.read_text())
    if cache_metadata.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
        raise ValueError("Actor BC cache uses an incompatible action timebase")
    required = {"z_rl", "state", "reference", "target", "valid"}
    if not required.issubset(cache):
        raise ValueError(f"Actor BC cache is missing {sorted(required - set(cache))}")
    if args.action_steps is not None:
        native_steps = int(cache["target"].shape[1])
        if not 1 <= args.action_steps <= native_steps:
            raise ValueError(
                f"action-steps must be in [1, {native_steps}] (cached chunk "
                f"length), got {args.action_steps}"
            )
        cache["target"] = cache["target"][:, : args.action_steps, :]
        cache["reference"] = cache["reference"][:, : args.action_steps, :]
    episode_index = _dataset_episode_index(dataset_root)
    sample_count = int(cache["z_rl"].shape[0])
    if episode_index.shape != (sample_count,):
        raise ValueError(
            f"dataset/cache row mismatch: {episode_index.shape} versus {sample_count}"
        )
    if not np.asarray(cache["valid"]).all():
        raise ValueError("IL demo replay currently requires complete GT action chunks")

    existing = [
        load_replay_metadata(path)
        for path in sorted(replay_root.glob("*.npz"))
    ]
    successes = sum(meta.get("outcome") == "success" for meta in existing)
    failures = sum(meta.get("outcome") == "failure" for meta in existing)
    requested = (
        max(0, failures - successes)
        if args.success_episodes is None
        else int(args.success_episodes)
    )
    if requested < 0:
        raise ValueError("success_episodes cannot be negative")

    training_episodes, validation_episodes = split_episode_ids(
        episode_index, seed=args.seed, val_fraction=args.val_episode_fraction
    )
    already_added = {
        int(meta["source_episode_index"])
        for meta in existing
        if meta.get("collection_phase") == "il_demo"
    }
    candidate_pool = (
        np.unique(episode_index) if args.include_validation_episodes else training_episodes
    )
    candidates = [int(value) for value in candidate_pool if int(value) not in already_added]
    if requested > len(candidates):
        raise ValueError(
            f"need {requested} IL demos but only {len(candidates)} non-validation episodes remain"
        )

    created: list[dict[str, Any]] = []
    for episode_id in candidates[:requested]:
        rows = np.flatnonzero(episode_index == episode_id)
        selected = decision_rows(rows, stride=args.decision_stride_frames)
        output_path = replay_root / f"il_demo_episode_{episode_id:06d}.npz"
        if output_path.exists():
            raise FileExistsError(output_path)
        created.append(
            _write_bundle(
                output_path,
                episode_id=episode_id,
                selected=selected,
                cache=cache,
                stage1_checkpoint=stage1_checkpoint,
                dataset_root=dataset_root,
                stride=args.decision_stride_frames,
            )
        )

    result = {
        "existing_success_episodes": successes,
        "existing_failure_episodes": failures,
        "created_success_episodes": len(created),
        "created_transitions": sum(item["transition_count"] for item in created),
        "final_success_episodes": successes + len(created),
        "final_failure_episodes": failures,
        "validation_episode_ids": validation_episodes.astype(int).tolist(),
        "included_validation_episodes": bool(args.include_validation_episodes),
        "created_episode_ids": [item["source_episode_index"] for item in created],
        "decision_stride_frames": args.decision_stride_frames,
    }
    _atomic_json(replay_root / "il_demo_manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
