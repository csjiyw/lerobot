#!/usr/bin/env python3
"""Compare SmolVLA chunk-30 and chunk-10 predictions offline.

The evaluator uses recorded observations (teacher-forced state/images) and
never connects to a robot.  For each checkpoint it reports:

* relative-E​​EF action error against the dataset target chunk;
* absolute EEF error after restoring the predicted relative chunk;
* the predicted boundary gap when a fresh observation is used after the
  previous chunk (30->30 for the chunk-30 model, 10->10 for chunk-10).

The comparison intentionally loads each checkpoint's saved pre/post processor,
so the model's own action normalization and relative-E​​EF contract are used.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.relative_eef import chunk_to_relative, relative_chunk_to_absolute


def _parse_episodes(value: str | None, total: int) -> list[int]:
    if not value or value.strip().lower() == "all":
        return list(range(total))
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            episodes.extend(range(lo, hi + 1))
        else:
            episodes.append(int(part))
    return sorted(set(ep for ep in episodes if 0 <= ep < total))


def _load_bundle(
    checkpoint: Path,
    metadata: LeRobotDatasetMetadata,
    device: torch.device,
) -> tuple[Any, Any, Any, int]:
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = str(device)
    policy = make_policy(cfg, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        dataset_stats=metadata.stats,
        # Saved rollout processors commonly contain the training device
        # (usually ``cuda``).  An offline evaluator must override that step
        # explicitly; otherwise a CPU smoke test can feed CUDA tensors to a
        # CPU-loaded policy, and a GPU comparison can inherit the wrong GPU.
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    policy.eval()
    chunk_size = int(getattr(cfg, "chunk_size", 0))
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_size={chunk_size} in {checkpoint}")
    return policy, preprocessor, postprocessor, chunk_size


def _model_step(
    item: dict[str, Any],
    bundle: tuple[Any, Any, Any, int],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy, preprocessor, postprocessor, chunk_size = bundle
    observation: dict[str, Any] = {
        "observation.state": item["observation.state"],
        "observation.images.camera_f": item["observation.images.camera_f"],
        "observation.images.camera_r": item["observation.images.camera_r"],
        "task": item["task"],
    }
    # Dataset images are already CHW float tensors in [0, 1].  The saved
    # processor adds the batch dimension and moves tensors to the device.
    torch.manual_seed(seed)
    with torch.inference_mode():
        processed = preprocessor(observation)
        predicted_normalized = policy.predict_action_chunk(processed)
        predicted_relative = postprocessor(predicted_normalized)
    predicted_relative = predicted_relative.detach().cpu().numpy()[0, :chunk_size].astype(np.float64)
    state = item["observation.state"].detach().cpu().numpy().astype(np.float64)
    targets = item["action"].detach().cpu().numpy().astype(np.float64)[:chunk_size]
    target_relative = chunk_to_relative(state, targets).astype(np.float64)
    predicted_absolute = relative_chunk_to_absolute(state, predicted_relative).astype(np.float64)
    return predicted_relative, target_relative, predicted_absolute


def _accumulate_error(metrics: dict[str, list[float]], predicted: np.ndarray, target: np.ndarray) -> None:
    diff = predicted - target
    for name, sl in (("position", slice(0, 3)), ("rotation", slice(3, 6)), ("gripper", slice(6, 7))):
        values = diff[:, sl]
        metrics[f"{name}_rmse"].append(float(np.sqrt(np.mean(values * values))))
        metrics[f"{name}_mae"].append(float(np.mean(np.abs(values))))


def evaluate_checkpoint(
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    episodes: list[int],
    device: torch.device,
    seed: int,
    max_samples: int | None,
) -> dict[str, Any]:
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    # Read one target chunk per queried observation.  The action remains in
    # absolute EEF space on disk; the saved relative processor handles the
    # model-side conversion and normalization.
    probe_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    chunk_size = int(getattr(probe_cfg, "chunk_size", 0))
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps={"action": [i / metadata.fps for i in range(chunk_size)]},
        video_backend="torchcodec",
    )
    bundle = _load_bundle(checkpoint, metadata, device)

    selected: list[tuple[int, int]] = []
    for ep in episodes:
        ep_meta = metadata.episodes[ep]
        start = int(ep_meta["dataset_from_index"])
        end = int(ep_meta["dataset_to_index"])
        # Use a chunk boundary grid.  This is deliberately not a random frame
        # sample: it measures the same no-overlap open-loop cadence as rollout.
        selected.extend((ep, idx) for idx in range(start, end - chunk_size + 1, chunk_size))
    if max_samples is not None:
        selected = selected[:max_samples]

    errors: dict[str, list[float]] = defaultdict(list)
    boundary_gaps_mm: list[float] = []
    boundary_rot_gaps_rad: list[float] = []
    samples_by_episode: dict[int, int] = defaultdict(int)
    previous: dict[int, np.ndarray] = {}
    count = 0
    for ep, abs_idx in selected:
        item = dataset[abs_idx]
        predicted_relative, target_relative, predicted_absolute = _model_step(
            item, bundle, device, seed + count
        )
        _accumulate_error(errors, predicted_relative, target_relative)
        abs_target = item["action"].detach().cpu().numpy().astype(np.float64)[:chunk_size]
        abs_diff = predicted_absolute - abs_target
        errors["absolute_position_rmse"].append(float(np.sqrt(np.mean(abs_diff[:, :3] ** 2))))
        errors["absolute_position_mae"].append(float(np.mean(np.abs(abs_diff[:, :3]))))
        errors["absolute_rotation_rmse"].append(float(np.sqrt(np.mean(abs_diff[:, 3:6] ** 2))))
        errors["absolute_gripper_mae"].append(float(np.mean(np.abs(abs_diff[:, 6]))))

        if ep in previous:
            gap = predicted_absolute[0, :3] - previous[ep][-1, :3]
            boundary_gaps_mm.append(float(np.linalg.norm(gap) * 1000.0))
            # Rotation is represented as the magnitude of the relative
            # rotation vector difference here; it is a continuity diagnostic.
            boundary_rot_gaps_rad.append(float(np.linalg.norm(predicted_absolute[0, 3:6] - previous[ep][-1, 3:6])))
        previous[ep] = predicted_absolute
        samples_by_episode[ep] += 1
        count += 1

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def percentile(values: list[float], q: float) -> float | None:
        return float(np.quantile(values, q)) if values else None

    result = {
        "checkpoint": str(checkpoint),
        "chunk_size": chunk_size,
        "fps": metadata.fps,
        "episodes": episodes,
        "samples": count,
        "samples_by_episode": dict(samples_by_episode),
        "per_sample_error": {key: mean(values) for key, values in sorted(errors.items())},
        "boundary_gap_mm": {
            "count": len(boundary_gaps_mm),
            "mean": mean(boundary_gaps_mm),
            "p50": percentile(boundary_gaps_mm, 0.50),
            "p95": percentile(boundary_gaps_mm, 0.95),
            "max": max(boundary_gaps_mm) if boundary_gaps_mm else None,
        },
        "boundary_rotation_gap_rad": {
            "count": len(boundary_rot_gaps_rad),
            "mean": mean(boundary_rot_gaps_rad),
            "p50": percentile(boundary_rot_gaps_rad, 0.50),
            "p95": percentile(boundary_rot_gaps_rad, 0.95),
            "max": max(boundary_rot_gaps_rad) if boundary_rot_gaps_rad else None,
        },
    }
    del bundle
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--chunk30", required=True, type=Path)
    parser.add_argument("--chunk10", required=True, type=Path)
    parser.add_argument("--episodes", default="all", help="episode ids, e.g. 0,1,5-9; default all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata(repo_id=args.dataset_repo_id, root=args.dataset_root)
    episodes = _parse_episodes(args.episodes, metadata.total_episodes)
    device = torch.device(args.device)
    results = {
        "dataset_root": str(args.dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "episodes": episodes,
        "models": [
            evaluate_checkpoint(
                checkpoint,
                args.dataset_root,
                args.dataset_repo_id,
                episodes,
                device,
                args.seed + index * 100000,
                args.max_samples,
            )
            for index, checkpoint in enumerate((args.chunk30, args.chunk10))
        ],
    }
    output = args.output or Path("web_runs") / "open_loop_chunk_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
