"""Diagnose whether the MC critic uses action information.

This is an offline diagnostic only.  It trains a fresh state-only twin critic
on the same success-conditioned Monte Carlo targets used by
``critic_mc_sanity`` and evaluates existing full Q critics with original and
batch-shuffled actions.  It never updates an actor and never touches robot
control.
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
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .critic_mc_sanity import (
    _binary_auc,
    _episode_rows,
    _make_quality_targets,
    _metric_summary,
    _split_episodes,
)
from .rlt_stage2_train import discover_replay_bundles, load_replay_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--full-critic-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--success-start", type=float, default=0.82)
    parser.add_argument("--random-seed", type=int, default=20260817)
    parser.add_argument("--log-freq", type=int, default=2_000)
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


class _StateOnlyTwinCritic(nn.Module):
    """Twin critic with z_rl and proprio only; action is absent by design."""

    def __init__(self, config: Any):
        super().__init__()
        projection_dim = int(config.modality_projection_dim)
        self.projection_dim = projection_dim
        self.activation_name = str(config.modality_projection_activation)
        if projection_dim:
            self.z_projector = nn.Linear(int(config.rlt_hidden_dim), projection_dim)
            self.state_projector = nn.Linear(int(config.state_dim), projection_dim)
            input_dim = 2 * projection_dim
        else:
            self.z_projector = nn.Identity()
            self.state_projector = nn.Identity()
            input_dim = int(config.rlt_hidden_dim) + int(config.state_dim)
        self.q1 = self._build_tower(config, input_dim)
        self.q2 = self._build_tower(config, input_dim)

    @staticmethod
    def _build_tower(config: Any, input_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in config.critic_hidden_dims:
            layers.append(nn.Linear(previous, int(hidden_dim)))
            if config.critic_layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(
                nn.Tanh() if config.critic_activation == "tanh" else nn.SiLU()
            )
            previous = int(hidden_dim)
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    def _features(self, z_rl: Tensor, state: Tensor) -> Tensor:
        z_features = self.z_projector(z_rl)
        state_features = self.state_projector(state)
        if self.projection_dim:
            activation = torch.tanh if self.activation_name == "tanh" else F.silu
            z_features = activation(z_features)
            state_features = activation(state_features)
        return torch.cat([z_features, state_features], dim=-1)

    def forward(self, z_rl: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        features = self._features(z_rl, state)
        return self.q1(features), self.q2(features)


def _predict_state_only(
    critic: nn.Module,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
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
            q1, q2 = critic(z, state)
            values.append(torch.minimum(q1, q2).reshape(-1).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _predict_full(
    critic: nn.Module,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    actions: np.ndarray,
    *,
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
            action = torch.as_tensor(actions[start : start + len(selected)], device=device)
            q1, q2 = critic(z, state, action)
            values.append(torch.minimum(q1, q2).reshape(-1).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _quality_report(
    prediction: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    success = labels.astype(bool)
    return {
        "quality_target": _metric_summary(prediction, targets),
        "transition_auc": _binary_auc(labels.astype(np.int64), prediction),
        "success_q_mean": float(prediction[success].mean()),
        "failure_q_mean": float(prediction[~success].mean()),
        "success_failure_gap": float(prediction[success].mean() - prediction[~success].mean()),
    }


def _train_state_only(
    config: Any,
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    rows_by_episode: dict[int, np.ndarray],
    train_episodes: np.ndarray,
    *,
    device: torch.device,
    updates: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_seed: int,
    log_freq: int,
) -> tuple[_StateOnlyTwinCritic, list[float]]:
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)
    critic = _StateOnlyTwinCritic(config).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay)
    episode_starts = {ep: rows[0] for ep, rows in rows_by_episode.items()}
    episode_lengths = {ep: len(rows) for ep, rows in rows_by_episode.items()}
    rng = np.random.default_rng(random_seed)
    losses: list[float] = []
    started = time.monotonic()
    for update in range(1, updates + 1):
        sampled_episodes = rng.choice(train_episodes, size=batch_size, replace=True)
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
        target = torch.as_tensor(targets[sampled_rows, None], device=device)
        q1, q2 = critic(z, state)
        loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update == 1 or update % log_freq == 0:
            rate = update / max(time.monotonic() - started, 1e-9)
            print(f"state_only_update={update} loss={losses[-1]:.6f} updates_per_second={rate:.2f}")
    return critic, losses


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.updates <= 0 or args.batch_size <= 0 or args.log_freq <= 0:
        raise ValueError("updates, batch_size and log_freq must be positive")
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
    targets, _, target_counts = _make_quality_targets(
        arrays, metadata, success_start=float(args.success_start)
    )
    episode_index = arrays["replay_episode_index"].reshape(-1).astype(np.int64)
    outcome = arrays["outcome_success"].reshape(-1) > 0.5
    rows_by_episode = _episode_rows(episode_index)
    split_rng = np.random.default_rng(args.random_seed)
    train_episodes, validation_episodes = _split_episodes(
        metadata, val_fraction=float(args.val_fraction), rng=split_rng
    )
    validation_rows = np.concatenate([rows_by_episode[int(ep)] for ep in validation_episodes])
    validation_targets = targets[validation_rows]
    validation_labels = outcome[validation_rows]
    device = torch.device(args.device)

    state_only, losses = _train_state_only(
        config,
        arrays,
        targets,
        rows_by_episode,
        train_episodes,
        device=device,
        updates=int(args.updates),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        random_seed=int(args.random_seed),
        log_freq=int(args.log_freq),
    )
    state_prediction = _predict_state_only(
        state_only,
        arrays,
        validation_rows,
        device=device,
        batch_size=int(args.batch_size),
    )
    state_report = _quality_report(state_prediction, validation_targets, validation_labels)

    full_reports: dict[str, Any] = {}
    base_actions = arrays["actions"][validation_rows]
    shuffle_rng = np.random.default_rng(int(args.random_seed) + 2)
    shuffled_actions = base_actions[shuffle_rng.permutation(len(base_actions))]
    for checkpoint_path in args.full_critic_checkpoint:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("stage") != "critic_mc_sanity":
            raise ValueError(f"unsupported full critic checkpoint: {checkpoint_path}")
        critic = RLCriticMLP(config).to(device)
        critic.load_state_dict(payload["critic_state_dict"])
        original = _predict_full(
            critic,
            arrays,
            validation_rows,
            base_actions,
            device=device,
            batch_size=int(args.batch_size),
        )
        shuffled = _predict_full(
            critic,
            arrays,
            validation_rows,
            shuffled_actions,
            device=device,
            batch_size=int(args.batch_size),
        )
        full_reports[checkpoint_path.parent.name] = {
            "checkpoint": str(checkpoint_path),
            "original": _quality_report(original, validation_targets, validation_labels),
            "action_shuffle": _quality_report(shuffled, validation_targets, validation_labels),
            "shuffle_minus_original_mean": float(np.mean(shuffled - original)),
            "shuffle_minus_original_abs_mean": float(np.mean(np.abs(shuffled - original))),
            "shuffle_prediction_correlation": float(np.corrcoef(original, shuffled)[0, 1]),
        }

    report = {
        "stage": "critic_input_diagnostics",
        "architecture_checkpoint": str(architecture_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "replay_root": str(replay_root),
        "episodes": len(metadata),
        "transitions": int(len(targets)),
        "target_counts": target_counts,
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "validation_episodes": validation_episodes.astype(int).tolist(),
        "train_loss_last": float(losses[-1]),
        "train_loss_mean_last_100": float(np.mean(losses[-100:])),
        "state_only_z_rl_plus_proprio": state_report,
        "full_critic_action_shuffle": full_reports,
        "decision": "do_not_update_actor_until_action_shuffle_and_state_only_tests_show_action_conditioned_generalization",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        output_dir / "state_only_critic.pt",
        {
            "stage": "critic_input_diagnostics_state_only",
            "architecture_checkpoint": str(architecture_checkpoint),
            "stage1_checkpoint": str(stage1_checkpoint),
            "rlt_config": config_payload,
            "critic_state_dict": state_only.state_dict(),
            "report": report,
        },
    )
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
