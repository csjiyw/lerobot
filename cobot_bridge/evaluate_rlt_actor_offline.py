"""Evaluate an RLT actor on held-out IL and optional offline replay features.

This command only consumes cached normalized tensors. It does not load camera
streams, connect to the bridge, start a policy server, or publish robot actions.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .piper_eef_metrics import (
    DEFAULT_EEF_FRAMES,
    DEFAULT_PIPER_URDF,
    PiperForwardKinematics,
    eef_error_summary,
)
from .rlt_stage2_data import (
    ACTION_CHUNK_TIMEBASE,
    SUPPORTED_ACTOR_OUTPUT_SEMANTICS,
    compose_actor_action,
)
from .rlt_stage2_train import discover_replay_bundles, load_replay_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-cache", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-episode-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--piper-urdf", type=Path, default=DEFAULT_PIPER_URDF)
    parser.add_argument("--eef-frames", nargs="+", default=list(DEFAULT_EEF_FRAMES))
    parser.add_argument("--skip-eef-metrics", action="store_true")
    parser.add_argument(
        "--max-eef-regression-ratio",
        type=float,
        default=1.0,
        help="Maximum Actor/Reference EEF RMSE ratio for the evaluation verdict.",
    )
    return parser


def _mean_std_action_stats(policy_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray, Path]:
    config_path = policy_path / "config.json"
    policy_config = json.loads(config_path.read_text())
    action_mode = policy_config.get("normalization_mapping", {}).get("ACTION")
    if action_mode != "MEAN_STD":
        raise ValueError(
            "EEF evaluation currently requires ACTION=MEAN_STD normalization, "
            f"found {action_mode!r}"
        )
    candidates = sorted(
        policy_path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one policy normalizer under {policy_path}, found {len(candidates)}"
        )
    stats = load_file(candidates[0])
    mean = stats["action.mean"].cpu().numpy().astype(np.float64)
    std = stats["action.std"].cpu().numpy().astype(np.float64)
    if mean.shape != (action_dim,) or std.shape != (action_dim,) or np.any(std <= 0):
        raise ValueError("action mean/std are incompatible with the RLT action dimension")
    return mean, std, candidates[0]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _weighted_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    action_weights: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    error = np.square(prediction - target)
    weights = action_weights.reshape(1, 1, -1)
    if valid is not None:
        weights = weights * valid[..., None].astype(np.float32)
    return float(np.sum(error * weights) / max(float(np.sum(np.broadcast_to(weights, error.shape))), 1.0))


def _error_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    action_weights: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    error = prediction - target
    mask = np.ones(error.shape[:2], dtype=bool) if valid is None else valid.astype(bool)
    flattened = error[mask]
    chunk_l2 = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1)
    return {
        "weighted_mse": _weighted_mse(prediction, target, action_weights, valid),
        "unweighted_mse": float(np.mean(np.square(flattened))),
        "per_dim_mse": np.mean(np.square(flattened), axis=0).tolist(),
        "per_dim_mae": np.mean(np.abs(flattened), axis=0).tolist(),
        "chunk_l2_mean": float(np.mean(chunk_l2)),
        "chunk_l2_p95": float(np.quantile(chunk_l2, 0.95)),
    }


def _overlap_summary(
    chunks: np.ndarray,
    row_indices: np.ndarray,
    episode_index: np.ndarray,
) -> dict[str, Any] | None:
    if chunks.shape[1] < 2 or chunks.shape[0] < 2:
        return None
    adjacent = (row_indices[1:] == row_indices[:-1] + 1) & (
        episode_index[1:] == episode_index[:-1]
    )
    if not adjacent.any():
        return None
    left = chunks[:-1][adjacent, 1:]
    right = chunks[1:][adjacent, :-1]
    absolute = np.abs(left - right)
    return {
        "pairs": int(adjacent.sum()),
        "overlap_mae": float(absolute.mean()),
        "overlap_p95_abs": float(np.quantile(absolute, 0.95)),
        "per_dim_overlap_mae": absolute.mean(axis=(0, 1)).tolist(),
    }


@torch.inference_mode()
def _predict(
    actor: torch.nn.Module,
    z_rl: np.ndarray,
    state: np.ndarray,
    reference: np.ndarray,
    *,
    action_scale: torch.Tensor,
    output_semantics: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    actor.eval()
    for start in range(0, z_rl.shape[0], batch_size):
        stop = min(start + batch_size, z_rl.shape[0])
        raw, _ = actor(
            torch.as_tensor(z_rl[start:stop], device=device),
            torch.as_tensor(state[start:stop], device=device),
            torch.as_tensor(reference[start:stop], device=device),
            training=False,
        )
        reference_tensor = torch.as_tensor(reference[start:stop], device=device)
        output.append(
            compose_actor_action(
                raw,
                reference_tensor,
                action_scale,
                output_semantics=output_semantics,
                residual_scale=actor.config.actor_residual_scale,
            )
            .cpu()
            .float()
            .numpy()
        )
    return np.concatenate(output)


def _heldout_indices(
    episode_index: np.ndarray, *, seed: int, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_index.astype(np.int64))
    if episodes.size < 2:
        raise ValueError("held-out evaluation requires at least two episodes")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    count = max(1, min(episodes.size - 1, round(episodes.size * validation_fraction)))
    selected_episodes = shuffled[:count]
    return np.flatnonzero(np.isin(episode_index, selected_episodes)), selected_episodes


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or not 0 < args.val_episode_fraction < 1:
        raise ValueError("batch_size and val_episode_fraction are invalid")
    if not np.isfinite(args.max_eef_regression_ratio) or args.max_eef_regression_ratio < 1:
        raise ValueError("max_eef_regression_ratio must be finite and at least 1")
    actor_checkpoint = args.actor_checkpoint.expanduser().resolve()
    cache_path = args.actor_cache.expanduser().resolve()
    stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    checkpoint = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_stage1 = Path(checkpoint["stage1_checkpoint"]).expanduser().resolve()
    if checkpoint_stage1 != stage1_checkpoint:
        raise ValueError("actor and requested Stage 1 checkpoints differ")
    if checkpoint.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
        raise ValueError("actor checkpoint uses an incompatible action timebase")
    output_semantics = checkpoint.get("actor_output_semantics")
    if output_semantics not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS:
        raise ValueError("actor checkpoint uses unsupported output semantics")

    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP

    config_payload = dict(checkpoint["rlt_config"])
    config_payload["mode"] = "inference"
    config = SmolVLARLTConfig(**config_payload)
    device = torch.device(args.device)
    actor = RLActorMLP(config).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    action_scale = torch.as_tensor(
        checkpoint["actor_action_scale"], dtype=torch.float32, device=device
    ).reshape(1, 1, -1)
    action_weights = np.asarray(
        config.bc_action_weights or [1.0] * config.action_dim, dtype=np.float32
    )

    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    cache_metadata = json.loads(metadata_path.read_text())
    if Path(cache_metadata["stage1_checkpoint"]).resolve() != stage1_checkpoint:
        raise ValueError("Actor cache belongs to another Stage 1 checkpoint")
    with np.load(cache_path, allow_pickle=False) as stored:
        cache = {name: np.asarray(stored[name]) for name in stored.files}
    indices, validation_episodes = _heldout_indices(
        cache["episode_index"],
        seed=args.seed,
        validation_fraction=args.val_episode_fraction,
    )
    il_reference = cache["reference"][indices]
    il_prediction = _predict(
        actor,
        cache["z_rl"][indices],
        cache["state"][indices],
        il_reference,
        action_scale=action_scale,
        output_semantics=output_semantics,
        device=device,
        batch_size=args.batch_size,
    )
    il_target = cache["target"][indices]
    il_valid = cache["valid"][indices].astype(bool)
    zero_ref_prediction = _predict(
        actor,
        cache["z_rl"][indices],
        cache["state"][indices],
        np.zeros_like(il_reference),
        action_scale=action_scale,
        output_semantics=output_semantics,
        device=device,
        batch_size=args.batch_size,
    )
    il_episode_index = cache["episode_index"][indices].astype(np.int64)
    actor_il = _error_summary(il_prediction, il_target, action_weights, il_valid)
    reference_il = _error_summary(il_reference, il_target, action_weights, il_valid)
    eef_evaluation = None
    if not args.skip_eef_metrics:
        policy_path = Path(checkpoint["smolvla_path"]).expanduser().resolve()
        action_mean, action_std, normalizer_path = _mean_std_action_stats(
            policy_path, config.action_dim
        )
        kinematics = PiperForwardKinematics(
            args.piper_urdf,
            frames=args.eef_frames,
        )
        actor_eef = eef_error_summary(
            il_prediction,
            il_target,
            il_valid,
            action_mean=action_mean,
            action_std=action_std,
            kinematics=kinematics,
        )
        reference_eef = eef_error_summary(
            il_reference,
            il_target,
            il_valid,
            action_mean=action_mean,
            action_std=action_std,
            kinematics=kinematics,
        )
        frame_ratios = {}
        regressed_metrics = []
        for frame in kinematics.frames:
            actor_frame = actor_eef["frames"][frame]
            reference_frame = reference_eef["frames"][frame]
            frame_ratios[frame] = {
                "position_rmse_actor_to_reference_ratio": (
                    actor_frame["position_error_mm"]["rmse"]
                    / max(reference_frame["position_error_mm"]["rmse"], 1e-12)
                ),
                "orientation_rmse_actor_to_reference_ratio": (
                    actor_frame["orientation_error_deg"]["rmse"]
                    / max(reference_frame["orientation_error_deg"]["rmse"], 1e-12)
                ),
            }
            for metric, ratio in frame_ratios[frame].items():
                if ratio > args.max_eef_regression_ratio:
                    regressed_metrics.append(
                        {"frame": frame, "metric": metric, "ratio": ratio}
                    )
        eef_evaluation = {
            "urdf": str(kinematics.urdf_path),
            "frames": list(kinematics.frames),
            "primary_frame": kinematics.frames[0],
            "action_normalization": "MEAN_STD",
            "normalizer": str(normalizer_path),
            "actor_vs_gt": actor_eef,
            "reference_vs_gt": reference_eef,
            "actor_to_reference_ratios": frame_ratios,
            "no_regression_verdict": {
                "maximum_ratio": args.max_eef_regression_ratio,
                "passed": not regressed_metrics,
                "regressed_metrics": regressed_metrics,
            },
        }
    report: dict[str, Any] = {
        "actor_checkpoint": str(actor_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "actor_architecture": {
            "z_dim": config.rlt_hidden_dim,
            "hidden_dims": list(config.actor_hidden_dims),
            "activation": config.actor_activation,
            "input_order": config.actor_input_order,
            "reference_dropout": config.ref_dropout,
            "modality_projection_dim": config.modality_projection_dim,
            "modality_projection_activation": config.modality_projection_activation,
            "actor_output_mode": config.actor_output_mode,
            "actor_residual_scale": config.actor_residual_scale,
            "actor_output_semantics": output_semantics,
            "bc_action_weights": action_weights.tolist(),
            "critic_hidden_dims": list(config.critic_hidden_dims),
            "critic_activation": config.critic_activation,
            "critic_layer_norm": config.critic_layer_norm,
        },
        "heldout_il": {
            "episodes": validation_episodes.astype(int).tolist(),
            "samples": int(indices.size),
            "actor_vs_gt": actor_il,
            "reference_vs_gt": reference_il,
            "actor_gt_to_reference_gt_weighted_mse_ratio": (
                actor_il["weighted_mse"] / max(reference_il["weighted_mse"], 1e-12)
            ),
            "reference_zero_ablation_mean_abs_change": float(
                np.mean(np.abs(zero_ref_prediction - il_prediction))
            ),
            "actor_overlap": _overlap_summary(
                il_prediction, indices, il_episode_index
            ),
            "reference_overlap": _overlap_summary(
                il_reference, indices, il_episode_index
            ),
            "gt_overlap": _overlap_summary(il_target, indices, il_episode_index),
            "eef": eef_evaluation,
        },
    }

    if args.replay_root is not None:
        replay_root = args.replay_root.expanduser().resolve()
        replay_paths = discover_replay_bundles(replay_root)
        replay, replay_metadata = load_replay_dataset(
            replay_paths, stage1_checkpoint=stage1_checkpoint
        )
        replay_prediction = _predict(
            actor,
            replay["z_rl"],
            replay["state"],
            replay["ref_actions"],
            action_scale=action_scale,
            output_semantics=output_semantics,
            device=device,
            batch_size=args.batch_size,
        )
        human = replay["is_intervention"].reshape(-1).astype(bool)
        demo = replay["is_demo"].reshape(-1).astype(bool)
        human_or_demo = human | demo
        bc_target = replay["ref_actions"].copy()
        bc_target[human_or_demo] = replay["actions"][human_or_demo]
        source_masks = {
            "online_reference": ~human_or_demo,
            "human_intervention": human & ~demo,
            "il_ground_truth": demo,
        }
        source_metrics = {
            name: {
                "transitions": int(mask.sum()),
                "actor_vs_target": _error_summary(
                    replay_prediction[mask], bc_target[mask], action_weights
                ),
            }
            for name, mask in source_masks.items()
            if mask.any()
        }
        replay_rows = np.arange(replay_prediction.shape[0], dtype=np.int64)
        report["replay"] = {
            "root": str(replay_root),
            "episodes": len(replay_metadata),
            "transitions": int(replay_prediction.shape[0]),
            "success_transitions": int(replay["outcome_success"].sum()),
            "failure_transitions": int(
                replay["outcome_success"].shape[0] - replay["outcome_success"].sum()
            ),
            "actor_vs_bc_target": _error_summary(
                replay_prediction, bc_target, action_weights
            ),
            "actor_vs_reference": _error_summary(
                replay_prediction, replay["ref_actions"], action_weights
            ),
            "by_bc_source": source_metrics,
            "actor_decision_overlap": _overlap_summary(
                replay_prediction,
                replay_rows,
                replay["replay_episode_index"].reshape(-1).astype(np.int64),
            ),
            "reference_decision_overlap": _overlap_summary(
                replay["ref_actions"],
                replay_rows,
                replay["replay_episode_index"].reshape(-1).astype(np.int64),
            ),
        }

    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
