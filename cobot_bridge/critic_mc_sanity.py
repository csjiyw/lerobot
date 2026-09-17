"""Success-conditioned Monte Carlo critic sanity check.

This is deliberately separate from RLT/TD3 training.  It trains a fresh
action-conditioned critic on bounded episode-quality targets:

* autonomous success: 0 -> +1;
* autonomous failure/abort: 0 -> -1;
* successful intervention: 0 -> -1 before intervention, then -1 -> +1;
* failed intervention: 0 -> -1 and stays at -1 after intervention.

The target is a diagnostic quality label, not a deployment reward.  Episodes
are split before training so that success/failure separation is measured on
held-out episodes.  No actor, policy server, bridge, or robot is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .rlt_stage2_train import discover_replay_bundles, load_replay_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--success-start", type=float, default=0.82)
    parser.add_argument("--random-seed", type=int, default=20260817)
    parser.add_argument("--log-freq", type=int, default=1_000)
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = ranks[positive].sum()
    n_positive = float(positive.sum())
    n_negative = float(negative.sum())
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1.0) / 2.0)
        / (n_positive * n_negative)
    )


def _metric_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    correlation = 0.0
    if np.std(prediction) > 1e-12 and np.std(target) > 1e-12:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "prediction_mean": float(np.mean(prediction)),
        "target_mean": float(np.mean(target)),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "pearson": correlation,
    }


def _episode_rows(episode_index: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for episode in np.unique(episode_index.astype(np.int64)):
        result[int(episode)] = np.flatnonzero(episode_index == episode)
    return result


def _make_quality_targets(
    arrays: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    *,
    success_start: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if not 0 < success_start < 1:
        raise ValueError("success_start must be in (0, 1)")
    episode_index = arrays["replay_episode_index"].reshape(-1).astype(np.int64)
    outcome_success = arrays["outcome_success"].reshape(-1) > 0.5
    intervention = arrays["is_intervention"].reshape(-1) > 0.5
    rows_by_episode = _episode_rows(episode_index)
    targets = np.zeros(episode_index.shape[0], dtype=np.float32)
    progress = np.zeros_like(targets)
    counts = {
        "autonomous_success": 0,
        "autonomous_failure": 0,
        "intervention_success": 0,
        "intervention_failure": 0,
    }
    for episode, rows in rows_by_episode.items():
        length = len(rows)
        episode_progress = np.linspace(0.0, 1.0, length, dtype=np.float32)
        progress[rows] = episode_progress
        success = bool(outcome_success[rows[0]])
        intervention_rows = np.flatnonzero(intervention[rows])
        if intervention_rows.size == 0:
            if success:
                targets[rows] = episode_progress
                counts["autonomous_success"] += 1
            else:
                targets[rows] = -episode_progress
                counts["autonomous_failure"] += 1
            continue

        intervention_start = int(intervention_rows[0])
        pre_progress = np.linspace(
            0.0,
            1.0,
            intervention_start + 1,
            dtype=np.float32,
        )
        targets[rows[: intervention_start + 1]] = -pre_progress
        if success:
            recovery_progress = np.linspace(
                0.0,
                1.0,
                length - intervention_start,
                dtype=np.float32,
            )
            targets[rows[intervention_start:]] = -1.0 + 2.0 * recovery_progress
            counts["intervention_success"] += 1
        else:
            targets[rows[intervention_start:]] = -1.0
            counts["intervention_failure"] += 1
    if sum(counts.values()) != len(metadata):
        raise RuntimeError(f"target episode accounting mismatch: {counts} vs {len(metadata)}")
    return targets, progress, counts


def _split_episodes(
    metadata: list[dict[str, Any]], *, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    labels = np.asarray([meta.get("outcome") == "success" for meta in metadata])
    validation: list[int] = []
    for outcome in (False, True):
        candidates = np.flatnonzero(labels == outcome)
        rng.shuffle(candidates)
        count = max(1, min(len(candidates) - 1, round(len(candidates) * val_fraction)))
        validation.extend(candidates[:count].tolist())
    validation_ids = np.asarray(sorted(validation), dtype=np.int64)
    all_ids = np.arange(len(metadata), dtype=np.int64)
    train_ids = np.setdiff1d(all_ids, validation_ids, assume_unique=True)
    return train_ids, validation_ids


def _predict(
    critic: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    action_key: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    critic.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            z = torch.as_tensor(arrays["z_rl"][selected], device=device)
            state = torch.as_tensor(arrays["state"][selected], device=device)
            action = torch.as_tensor(arrays[action_key][selected], device=device)
            q1, q2 = critic(z, state, action)
            values.append(torch.minimum(q1, q2).reshape(-1).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _progress_metrics(
    prediction: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    progress: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        mask = (progress >= lower) & (progress < upper)
        name = f"{lower:.2f}_{min(upper, 1.0):.2f}"
        result[name] = {
            "rows": int(mask.sum()),
            "target_success_mean": float(targets[mask & labels].mean())
            if (mask & labels).any()
            else None,
            "target_failure_mean": float(targets[mask & ~labels].mean())
            if (mask & ~labels).any()
            else None,
            "q_success_mean": float(prediction[mask & labels].mean())
            if (mask & labels).any()
            else None,
            "q_failure_mean": float(prediction[mask & ~labels].mean())
            if (mask & ~labels).any()
            else None,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.updates <= 0 or args.batch_size <= 0 or args.log_freq <= 0:
        raise ValueError("updates, batch_size and log_freq must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be positive and finite")
    architecture_checkpoint = args.architecture_checkpoint.expanduser().resolve()
    stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    replay_root = args.replay_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    architecture = torch.load(architecture_checkpoint, map_location="cpu", weights_only=False)
    if architecture.get("stage") != "rlt_stage2":
        raise ValueError("architecture checkpoint must be an RLT Stage 2 checkpoint")
    if Path(architecture["stage1_checkpoint"]).expanduser().resolve() != stage1_checkpoint:
        raise ValueError("architecture and Stage 1 checkpoints differ")
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLCriticMLP

    config_payload = dict(architecture["rlt_config"])
    config_payload["mode"] = "inference"
    config = SmolVLARLTConfig(**config_payload)
    replay_paths = discover_replay_bundles(replay_root)
    if not replay_paths:
        raise RuntimeError(f"no replay bundles under {replay_root}")
    arrays, metadata = load_replay_dataset(replay_paths, stage1_checkpoint=stage1_checkpoint)
    targets, progress, target_counts = _make_quality_targets(
        arrays, metadata, success_start=float(args.success_start)
    )
    episode_index = arrays["replay_episode_index"].reshape(-1).astype(np.int64)
    outcome = arrays["outcome_success"].reshape(-1) > 0.5
    rows_by_episode = _episode_rows(episode_index)
    rng = np.random.default_rng(args.random_seed)
    train_episodes, validation_episodes = _split_episodes(
        metadata, val_fraction=float(args.val_fraction), rng=rng
    )
    train_rows = np.concatenate([rows_by_episode[int(ep)] for ep in train_episodes])
    validation_rows = np.concatenate([rows_by_episode[int(ep)] for ep in validation_episodes])
    device = torch.device(args.device)
    critic = RLCriticMLP(config).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    episode_starts = {ep: rows[0] for ep, rows in rows_by_episode.items()}
    episode_lengths = {ep: len(rows) for ep, rows in rows_by_episode.items()}
    losses: list[float] = []
    started = time.monotonic()
    for update in range(1, int(args.updates) + 1):
        sampled_episodes = rng.choice(train_episodes, size=int(args.batch_size), replace=True)
        sampled_rows = np.asarray(
            [
                episode_starts[int(ep)]
                + int(rng.integers(episode_lengths[int(ep)]))
                for ep in sampled_episodes
            ],
            dtype=np.int64,
        )
        z = torch.as_tensor(arrays["z_rl"][sampled_rows], device=device)
        state = torch.as_tensor(arrays["state"][sampled_rows], device=device)
        action = torch.as_tensor(arrays["actions"][sampled_rows], device=device)
        target = torch.as_tensor(targets[sampled_rows, None], device=device)
        q1, q2 = critic(z, state, action)
        loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update == 1 or update % int(args.log_freq) == 0:
            rate = update / max(time.monotonic() - started, 1e-9)
            print(
                f"update={update} loss={losses[-1]:.6f} "
                f"updates_per_second={rate:.2f}"
            )

    validation_q = _predict(
        critic,
        arrays,
        validation_rows,
        action_key="actions",
        device=device,
        batch_size=int(args.batch_size),
    )
    validation_targets = targets[validation_rows]
    validation_labels = outcome[validation_rows]
    validation_progress = progress[validation_rows]
    validation_episode_index = episode_index[validation_rows]
    episode_q = np.asarray(
        [
            validation_q[validation_episode_index == episode]
            .mean()
            for episode in validation_episodes
        ],
        dtype=np.float32,
    )
    episode_labels = np.asarray(
        [metadata[int(episode)].get("outcome") == "success" for episode in validation_episodes],
        dtype=bool,
    )
    random_generator = np.random.default_rng(args.random_seed + 1)
    random02 = arrays["ref_actions"][validation_rows] + random_generator.normal(
        0.0, 0.02, size=arrays["ref_actions"][validation_rows].shape
    ).astype(np.float32)
    random05 = arrays["ref_actions"][validation_rows] + random_generator.normal(
        0.0, 0.05, size=arrays["ref_actions"][validation_rows].shape
    ).astype(np.float32)
    reference_q = _predict(
        critic,
        arrays,
        validation_rows,
        action_key="ref_actions",
        device=device,
        batch_size=int(args.batch_size),
    )
    temporary_arrays = dict(arrays)
    temporary_arrays["random02"] = random02
    temporary_arrays["random05"] = random05
    random02_q = _predict(
        critic,
        temporary_arrays,
        np.arange(len(validation_rows)),
        action_key="random02",
        device=device,
        batch_size=int(args.batch_size),
    )
    random05_q = _predict(
        critic,
        temporary_arrays,
        np.arange(len(validation_rows)),
        action_key="random05",
        device=device,
        batch_size=int(args.batch_size),
    )
    report: dict[str, Any] = {
        "stage": "critic_mc_sanity",
        "architecture_checkpoint": str(architecture_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "replay_root": str(replay_root),
        "episodes": len(metadata),
        "transitions": int(len(targets)),
        "train_episodes": train_episodes.astype(int).tolist(),
        "validation_episodes": validation_episodes.astype(int).tolist(),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "success_start": float(args.success_start),
        "target_counts": target_counts,
        "target_range": [float(targets.min()), float(targets.max())],
        "train_loss_last": float(losses[-1]),
        "train_loss_mean_last_100": float(np.mean(losses[-100:])),
        "validation": {
            "rows": int(len(validation_rows)),
            "episodes": int(len(validation_episodes)),
            "quality_target": _metric_summary(validation_q, validation_targets),
            "transition_success_failure_auc": _binary_auc(
                validation_labels.astype(np.int64), validation_q
            ),
            "episode_success_failure_auc": _binary_auc(
                episode_labels.astype(np.int64), episode_q
            ),
            "success_q_mean": float(validation_q[validation_labels].mean()),
            "failure_q_mean": float(validation_q[~validation_labels].mean()),
            "success_target_mean": float(validation_targets[validation_labels].mean()),
            "failure_target_mean": float(validation_targets[~validation_labels].mean()),
            "progress": _progress_metrics(
                validation_q,
                validation_targets,
                validation_labels,
                validation_progress,
            ),
            "reference_q_mean": float(reference_q.mean()),
            "random02_q_mean": float(random02_q.mean()),
            "random05_q_mean": float(random05_q.mean()),
            "random02_minus_reference_mean": float(np.mean(random02_q - reference_q)),
            "random05_minus_reference_mean": float(np.mean(random05_q - reference_q)),
        },
        "actor_allowed": False,
        "decision": "allow_actor_only_if_heldout_auc_and_progress_separation_are_positive",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": "critic_mc_sanity",
        "architecture_checkpoint": str(architecture_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "rlt_config": config_payload,
        "critic_state_dict": critic.state_dict(),
        "target_definition": "success_0_to_1_failure_0_to_minus1_intervention_recovery_minus1_to1",
        "report": report,
    }
    _atomic_torch_save(output_dir / "critic_mc_sanity.pt", checkpoint)
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
