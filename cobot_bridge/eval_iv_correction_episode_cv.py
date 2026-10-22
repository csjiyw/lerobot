"""Episode-held-out CV for intervention-weighted RLT actor correction.

This is an offline supervised diagnostic. It uses cached RLT tokens and replay
actions, holds out every row from each selected intervention episode, freezes
the Stage-1 VLA/RLT representation, and trains only the residual actor. No
critic, simulator, policy server, or robot is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP

from .evaluate_rlt_actor_offline import _mean_std_action_stats
from .rlt_stage2_data import (
    ACTION_CHUNK_TIMEBASE,
    RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
    compose_actor_action,
)
from .rlt_stage2_train import (
    _load_il_anchor_cache,
    discover_replay_bundles,
    load_replay_dataset,
)

LOG = logging.getLogger("cobot_bridge.eval_iv_correction_episode_cv")


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2] / "RL-Token-SmolVLA"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-actor",
        type=Path,
        default=root / "checkpoints/cobot_magic_073rel_rlt_actor_bc_ilgt_ref30_actor10_50k_v1/actor_bc_best.pt",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=root / "replay/cobot_magic_073rel_rlinf_ref30_actor10_recap_eef_v2/replay",
    )
    parser.add_argument(
        "--il-anchor-cache",
        type=Path,
        default=root / "checkpoints/cobot_magic_073rel_rlt_actor_bc_ilgt_ref30_actor10_50k_v1/actor_bc_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "eval/iv_episode_cv_20261010_rlt_actor_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 10.0, 50.0])
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--il-anchor-weight", type=float, default=30.0)
    parser.add_argument("--bc-weight", type=float, default=5.0)
    parser.add_argument("--near-zero-mm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=160)
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _make_episode_folds(
    episode_index: np.ndarray,
    outcome_success: np.ndarray,
    intervention_rows: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[list[int]]:
    intervention_episodes = np.unique(episode_index[intervention_rows]).astype(int)
    if len(intervention_episodes) < n_folds:
        raise ValueError(
            f"need at least {n_folds} intervention episodes, found {len(intervention_episodes)}"
        )
    by_outcome: dict[bool, list[int]] = {False: [], True: []}
    for episode in intervention_episodes:
        rows = episode_index == episode
        labels = np.unique(outcome_success[rows].astype(bool))
        if labels.size != 1:
            raise ValueError(f"episode {episode} has inconsistent outcome labels")
        by_outcome[bool(labels[0])].append(int(episode))
    rng = np.random.default_rng(seed)
    for group in by_outcome.values():
        rng.shuffle(group)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    # Greedy stratification: distribute the rare outcome first, then keep fold
    # sizes balanced. With the current data, one failure episode is expected.
    for success_value in (False, True):
        for episode in by_outcome[success_value]:
            target = min(range(n_folds), key=lambda i: (len(folds[i]), i))
            folds[target].append(episode)
    return folds


def _balanced_episode_batch_schedule(
    *,
    episode_index: np.ndarray,
    outcome_success: np.ndarray,
    train_rows: np.ndarray,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    eligible: dict[bool, tuple[np.ndarray, np.ndarray]] = {}
    for label in (True, False):
        rows = train_rows[outcome_success[train_rows].astype(bool) == label]
        eps, counts = np.unique(episode_index[rows], return_counts=True)
        if eps.size == 0:
            raise ValueError("each training fold needs both success and failure episodes")
        count_by_episode = dict(zip(eps.tolist(), counts.tolist()))
        probs = np.asarray([1.0 / count_by_episode[int(ep)] for ep in episode_index[rows]])
        probs /= probs.sum()
        eligible[label] = rows, probs
    success_count = batch_size // 2
    failure_count = batch_size - success_count
    schedule = []
    for _ in range(steps):
        success_rows, success_probs = eligible[True]
        failure_rows, failure_probs = eligible[False]
        selected = np.concatenate(
            [
                rng.choice(success_rows, size=success_count, replace=True, p=success_probs),
                rng.choice(failure_rows, size=failure_count, replace=True, p=failure_probs),
            ]
        )
        rng.shuffle(selected)
        schedule.append(selected.astype(np.int64, copy=False))
    return schedule


def _target_arrays(
    arrays: dict[str, np.ndarray],
    action_scale: np.ndarray,
    residual_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    reference = np.asarray(arrays["ref_actions"], dtype=np.float32)
    intervention = np.asarray(arrays["is_intervention"]).reshape(-1).astype(bool)
    demo = np.asarray(arrays["is_demo"]).reshape(-1).astype(bool)
    human_step = np.asarray(arrays["action_human_mask"]).astype(bool)
    if human_step.shape != actions.shape[:2]:
        raise ValueError("action_human_mask does not match replay action chunks")
    human_step |= demo[:, None]
    target = np.where(human_step[..., None], actions, reference)
    scale = np.asarray(action_scale, dtype=np.float32).reshape(1, 1, -1)
    low = np.maximum(reference - scale * float(residual_scale), -scale)
    high = np.minimum(reference + scale * float(residual_scale), scale)
    reachable = np.clip(target, low, high)
    clipped = np.any(np.abs(reachable - target) > 1e-7, axis=-1)
    denominator = max(int(human_step.sum()), 1)
    clipped_fraction = float((clipped & human_step).sum() / denominator)
    return reachable.astype(np.float32), human_step, clipped_fraction


def _train_one(
    *,
    initial_state: dict[str, torch.Tensor],
    config: SmolVLARLTConfig,
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    intervention_rows: np.ndarray,
    action_scale_np: np.ndarray,
    action_weights_np: np.ndarray,
    anchor: dict[str, np.ndarray],
    anchor_batches: list[np.ndarray],
    replay_batches: list[np.ndarray],
    iv_weight: float,
    bc_weight: float,
    il_anchor_weight: float,
    device: torch.device,
    log_tag: str,
    log_every: int,
) -> RLActorMLP:
    actor = RLActorMLP(config).to(device)
    actor.load_state_dict(initial_state, strict=True)
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config.actor_lr))
    action_scale = torch.as_tensor(action_scale_np, dtype=torch.float32, device=device).reshape(1, 1, -1)
    action_weights = torch.as_tensor(action_weights_np, dtype=torch.float32, device=device)
    start = time.monotonic()
    for step, (indices, anchor_indices) in enumerate(zip(replay_batches, anchor_batches), start=1):
        z = torch.as_tensor(arrays["z_rl"][indices], dtype=torch.float32, device=device)
        state = torch.as_tensor(arrays["state"][indices], dtype=torch.float32, device=device)
        reference = torch.as_tensor(arrays["ref_actions"][indices], dtype=torch.float32, device=device)
        target = torch.as_tensor(targets[indices], dtype=torch.float32, device=device)
        row_iv = torch.as_tensor(intervention_rows[indices], dtype=torch.bool, device=device)
        sample_weight = torch.ones(len(indices), dtype=torch.float32, device=device)
        sample_weight[row_iv] = float(iv_weight)

        raw, _ = actor(z, state, reference, training=True)
        prediction = compose_actor_action(
            raw,
            reference,
            action_scale,
            output_semantics=RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
            residual_scale=config.actor_residual_scale,
        )
        replay_error = (prediction - target).square()
        replay_weights = sample_weight[:, None, None] * action_weights[None, None, :]
        replay_loss = (replay_error * replay_weights).sum() / (
            sample_weight.sum() * prediction.shape[1] * action_weights.sum()
        ).clamp_min(1.0)

        anchor_z = torch.as_tensor(anchor["z_rl"][anchor_indices], dtype=torch.float32, device=device)
        anchor_state = torch.as_tensor(anchor["state"][anchor_indices], dtype=torch.float32, device=device)
        anchor_reference = torch.as_tensor(anchor["reference"][anchor_indices], dtype=torch.float32, device=device)
        anchor_target = torch.as_tensor(anchor["target"][anchor_indices], dtype=torch.float32, device=device)
        anchor_valid = torch.as_tensor(anchor["valid"][anchor_indices], dtype=torch.bool, device=device)
        anchor_raw, _ = actor(anchor_z, anchor_state, anchor_reference, training=True)
        anchor_prediction = compose_actor_action(
            anchor_raw,
            anchor_reference,
            action_scale,
            output_semantics=RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
            residual_scale=config.actor_residual_scale,
        )
        anchor_weights = anchor_valid.to(anchor_prediction.dtype).unsqueeze(-1) * action_weights.reshape(1, 1, -1)
        anchor_loss = (
            (anchor_prediction - anchor_target).square() * anchor_weights
        ).sum() / anchor_weights.sum().clamp_min(1.0)
        loss = float(bc_weight) * (
            replay_loss + float(il_anchor_weight) * anchor_loss
        ) / (1.0 + float(il_anchor_weight))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == len(replay_batches):
            LOG.info(
                "%s step=%d/%d total=%.6f replay=%.6f il_anchor=%.6f elapsed=%.1fs",
                log_tag, step, len(replay_batches), float(loss.detach()),
                float(replay_loss.detach()), float(anchor_loss.detach()),
                time.monotonic() - start,
            )
    actor.eval()
    return actor


@torch.inference_mode()
def _predict_rows(
    actor: RLActorMLP,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    action_scale: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(rows), batch_size):
        idx = rows[start : start + batch_size]
        z = torch.as_tensor(arrays["z_rl"][idx], dtype=torch.float32, device=device)
        state = torch.as_tensor(arrays["state"][idx], dtype=torch.float32, device=device)
        reference = torch.as_tensor(arrays["ref_actions"][idx], dtype=torch.float32, device=device)
        raw, _ = actor(z, state, reference, training=False)
        chunks.append(
            compose_actor_action(
                raw,
                reference,
                action_scale,
                output_semantics=RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
                residual_scale=actor.config.actor_residual_scale,
            ).float().cpu().numpy()
        )
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,) + arrays["actions"].shape[1:], np.float32)


def _masked_mse(prediction: np.ndarray, target: np.ndarray, row_step_mask: np.ndarray) -> float | None:
    if not row_step_mask.any():
        return None
    error = np.square(prediction - target)
    return float(error[row_step_mask].mean())


def _correction_metrics(
    *,
    model_prediction: np.ndarray,
    bc_prediction: np.ndarray,
    human_target: np.ndarray,
    human_step_mask: np.ndarray,
    intervention_rows: np.ndarray,
    episode_index: np.ndarray,
    selected_episodes: np.ndarray,
    xyz_std: np.ndarray,
    near_zero_m: float,
) -> dict[str, Any]:
    row_mask = intervention_rows & np.isin(episode_index, selected_episodes)
    details: list[dict[str, float | int]] = []
    for row in np.flatnonzero(row_mask):
        valid = human_step_mask[row]
        if not valid.any():
            continue
        desired = (human_target[row, valid, :3] - bc_prediction[row, valid, :3]) * xyz_std
        proposed = (model_prediction[row, valid, :3] - bc_prediction[row, valid, :3]) * xyz_std
        desired_flat = desired.reshape(-1)
        proposed_flat = proposed.reshape(-1)
        desired_norm = float(np.linalg.norm(desired_flat))
        proposed_norm = float(np.linalg.norm(proposed_flat))
        dot = float(np.dot(proposed_flat, desired_flat))
        cosine = dot / max(desired_norm * proposed_norm, 1e-12)
        details.append(
            {
                "episode": int(episode_index[row]),
                "desired_norm_m": desired_norm,
                "proposed_norm_m": proposed_norm,
                "dot": dot,
                "cosine": float(np.clip(cosine, -1.0, 1.0)),
                "projection_k": dot / max(desired_norm**2, 1e-12),
            }
        )
    usable = [item for item in details if item["desired_norm_m"] >= near_zero_m]
    per_episode: dict[int, dict[str, float | int]] = {}
    for episode in np.unique([int(item["episode"]) for item in details]):
        items = [item for item in usable if int(item["episode"]) == int(episode)]
        if not items:
            continue
        per_episode[int(episode)] = {
            "rows": len(items),
            "direction_correct_rate": float(np.mean([float(x["dot"]) > 0 for x in items])),
            "cosine_mean": float(np.mean([float(x["cosine"]) for x in items])),
            "projection_k_mean": float(np.mean([float(x["projection_k"]) for x in items])),
        }
    if not usable:
        return {
            "rows": len(details),
            "near_zero_rows": len(details),
            "near_zero_threshold_mm": near_zero_m * 1000.0,
            "nonzero_rows": 0,
            "per_episode": per_episode,
        }
    episode_values = list(per_episode.values())
    return {
        "rows": len(details),
        "near_zero_rows": len(details) - len(usable),
        "near_zero_threshold_mm": near_zero_m * 1000.0,
        "nonzero_rows": len(usable),
        "direction_correct_rate_pooled": float(np.mean([float(x["dot"]) > 0 for x in usable])),
        "cosine_mean_pooled": float(np.mean([float(x["cosine"]) for x in usable])),
        "cosine_median_pooled": float(np.median([float(x["cosine"]) for x in usable])),
        "projection_k_mean_pooled": float(np.mean([float(x["projection_k"]) for x in usable])),
        "projection_k_median_pooled": float(np.median([float(x["projection_k"]) for x in usable])),
        "direction_correct_rate_episode_macro": float(np.mean([x["direction_correct_rate"] for x in episode_values])),
        "cosine_mean_episode_macro": float(np.mean([x["cosine_mean"] for x in episode_values])),
        "projection_k_mean_episode_macro": float(np.mean([x["projection_k_mean"] for x in episode_values])),
        "per_episode": per_episode,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.folds < 2 or args.steps <= 0 or args.batch_size < 2:
        parser.error("folds >= 2, steps > 0, and batch-size >= 2 are required")
    if sorted(args.weights) != [1.0, 10.0, 50.0]:
        parser.error("this diagnostic expects exactly IV weights 1, 10, and 50")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    initial_path = args.initial_actor.expanduser().resolve()
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if initial.get("actor_output_semantics") != RESIDUAL_ACTOR_OUTPUT_SEMANTICS:
        raise ValueError("initial actor must use the bounded residual action semantics")
    if initial.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
        raise ValueError("initial actor has an incompatible action timebase")
    stage1_path = Path(initial["stage1_checkpoint"]).expanduser().resolve()
    config_payload = dict(initial["rlt_config"])
    config_payload["mode"] = "inference"
    config = SmolVLARLTConfig(**config_payload)
    if config.ref_dropout != 0:
        raise ValueError("paired IV weighting requires ref_dropout=0 in the shared initialization")
    device = torch.device(args.device)
    initial_state = {k: v.detach().cpu().clone() for k, v in initial["actor_state_dict"].items()}
    action_scale_np = np.asarray(initial["actor_action_scale"], dtype=np.float32)
    action_scale_t = torch.as_tensor(action_scale_np, dtype=torch.float32, device=device).reshape(1, 1, -1)
    action_weights_np = np.asarray(config.bc_action_weights or [1.0] * config.action_dim, dtype=np.float32)
    action_mean, action_std, stats_path = _mean_std_action_stats(Path(initial["smolvla_path"]), config.action_dim)

    replay_paths = discover_replay_bundles(args.replay_root.expanduser().resolve())
    arrays, metadata = load_replay_dataset(replay_paths, stage1_checkpoint=stage1_path)
    episode_index = arrays["replay_episode_index"].reshape(-1).astype(np.int64)
    outcome_success = arrays["outcome_success"].reshape(-1).astype(bool)
    intervention_rows = arrays["is_intervention"].reshape(-1).astype(bool) & ~arrays["is_demo"].reshape(-1).astype(bool)
    if int(intervention_rows.sum()) == 0:
        raise ValueError("replay contains no human intervention rows")
    targets, human_step_mask, clipped_fraction = _target_arrays(
        arrays, action_scale_np, config.actor_residual_scale
    )
    episode_folds = _make_episode_folds(
        episode_index, outcome_success, intervention_rows, args.folds, args.seed
    )
    anchor_cache, anchor_train_indices = _load_il_anchor_cache(
        args.il_anchor_cache.expanduser().resolve(), seed=args.seed
    )
    anchor_rng = np.random.default_rng(args.seed + 17)
    fold_runs: list[dict[str, Any]] = []
    weights = sorted(float(w) for w in args.weights)
    predictions = {weight: np.full_like(arrays["actions"], np.nan, dtype=np.float32) for weight in weights}
    episode_name = {
        int(i): str(item.get("path", item.get("name", i)))
        for i, item in enumerate(metadata)
    }
    iv_episodes = np.unique(episode_index[intervention_rows]).astype(int)
    all_episode_summary = []
    for episode in np.unique(episode_index):
        rows = episode_index == episode
        iv_count = int((rows & intervention_rows).sum())
        if iv_count:
            all_episode_summary.append(
                {
                    "episode_index": int(episode),
                    "name": episode_name.get(int(episode), str(episode)),
                    "rows": int(rows.sum()),
                    "intervention_rows": iv_count,
                    "outcome": "success" if bool(outcome_success[rows][0]) else "failure",
                }
            )

    for fold_id, heldout_eps in enumerate(episode_folds):
        heldout_episode_mask = np.isin(episode_index, heldout_eps)
        train_rows = np.flatnonzero(~heldout_episode_mask)
        test_rows = np.flatnonzero(heldout_episode_mask)
        if not np.any(outcome_success[train_rows]) or np.all(outcome_success[train_rows]):
            raise ValueError(f"fold {fold_id} training rows do not contain both outcomes")
        replay_batches = _balanced_episode_batch_schedule(
            episode_index=episode_index,
            outcome_success=outcome_success,
            train_rows=train_rows,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + fold_id,
        )
        anchor_batches = [
            anchor_rng.choice(
                anchor_train_indices,
                size=min(args.batch_size, len(anchor_train_indices)),
                replace=False,
            )
            for _ in range(args.steps)
        ]
        fold_outcome = {
            "success": int(sum(bool(outcome_success[episode_index == ep][0]) for ep in heldout_eps)),
            "failure": int(sum(not bool(outcome_success[episode_index == ep][0]) for ep in heldout_eps)),
        }
        fold_record: dict[str, Any] = {
            "fold": fold_id,
            "heldout_episodes": [
                {
                    "episode_index": int(ep),
                    "name": episode_name.get(int(ep), str(ep)),
                    "outcome": "success" if bool(outcome_success[episode_index == ep][0]) else "failure",
                }
                for ep in heldout_eps
            ],
            "heldout_episode_outcomes": fold_outcome,
            "train_rows": int(len(train_rows)),
            "heldout_rows_all_sources": int(len(test_rows)),
            "heldout_intervention_rows": int((heldout_episode_mask & intervention_rows).sum()),
            "models": {},
        }
        for weight in weights:
            tag = f"fold{fold_id}_iv{int(weight)}"
            torch.manual_seed(args.seed + fold_id)
            actor = _train_one(
                initial_state=initial_state,
                config=config,
                arrays=arrays,
                targets=targets,
                intervention_rows=intervention_rows,
                action_scale_np=action_scale_np,
                action_weights_np=action_weights_np,
                anchor=anchor_cache,
                anchor_batches=anchor_batches,
                replay_batches=replay_batches,
                iv_weight=weight,
                bc_weight=args.bc_weight,
                il_anchor_weight=args.il_anchor_weight,
                device=device,
                log_tag=tag,
                log_every=args.log_every,
            )
            predictions[weight][test_rows] = _predict_rows(
                actor,
                arrays,
                test_rows,
                action_scale_t,
                device,
                max(args.batch_size, 512),
            )
            model_fold_metrics: dict[str, Any] = {
                "heldout_human_action_mse": _masked_mse(
                    predictions[weight][test_rows],
                    arrays["actions"][test_rows],
                    human_step_mask[test_rows]
                    & intervention_rows[test_rows, None],
                ),
                "heldout_normal_reference_mse": _masked_mse(
                    predictions[weight][test_rows],
                    arrays["ref_actions"][test_rows],
                    np.broadcast_to(
                        (~intervention_rows[test_rows])[:, None],
                        human_step_mask[test_rows].shape,
                    ),
                ),
            }
            if weight != 1.0:
                model_fold_metrics["correction_vs_iv1"] = _correction_metrics(
                    model_prediction=predictions[weight],
                    bc_prediction=predictions[1.0],
                    human_target=arrays["actions"],
                    human_step_mask=human_step_mask,
                    intervention_rows=intervention_rows,
                    episode_index=episode_index,
                    selected_episodes=np.asarray(heldout_eps, dtype=np.int64),
                    xyz_std=np.asarray(action_std[:3], dtype=np.float32),
                    near_zero_m=args.near_zero_mm / 1000.0,
                )
            fold_record["models"][str(int(weight))] = model_fold_metrics
            del actor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_runs.append(fold_record)

    heldout_iv_rows = intervention_rows & np.isin(episode_index, iv_episodes)
    heldout_normal_rows = (
        ~intervention_rows
        & ~arrays["is_demo"].reshape(-1).astype(bool)
        & np.isin(episode_index, iv_episodes)
    )
    heldout_normal_steps = np.broadcast_to(
        heldout_normal_rows[:, None],
        human_step_mask.shape,
    )
    human_eval_steps = human_step_mask & heldout_iv_rows[:, None]
    bc_weight = 1.0
    if not np.isfinite(predictions[bc_weight][np.isin(episode_index, iv_episodes)]).all():
        raise RuntimeError("OOF predictions are missing for a held-out intervention episode")
    bc_prediction = predictions[bc_weight]
    model_summaries: dict[str, Any] = {}
    for weight in weights:
        pred = predictions[weight]
        intervention_mse = _masked_mse(pred, arrays["actions"], human_eval_steps)
        normal_mse = _masked_mse(pred, arrays["ref_actions"], heldout_normal_steps)
        episode_human_mse = []
        episode_normal_mse = []
        for episode in iv_episodes:
            row_ep = episode_index == episode
            mask_h = human_step_mask & (row_ep & intervention_rows)[:, None]
            mask_n = np.broadcast_to((row_ep & ~intervention_rows)[:, None], human_step_mask.shape)
            value_h = _masked_mse(pred, arrays["actions"], mask_h)
            value_n = _masked_mse(pred, arrays["ref_actions"], mask_n)
            if value_h is not None:
                episode_human_mse.append(value_h)
            if value_n is not None:
                episode_normal_mse.append(value_n)
        summary: dict[str, Any] = {
            "iv_weight": weight,
            "heldout_intervention_rows": int(heldout_iv_rows.sum()),
            "heldout_human_action_mse_pooled": intervention_mse,
            "heldout_human_action_mse_episode_macro": float(np.mean(episode_human_mse)),
            "heldout_normal_reference_mse_pooled": normal_mse,
            "heldout_normal_reference_mse_episode_macro": (
                float(np.mean(episode_normal_mse)) if episode_normal_mse else None
            ),
        }
        if weight != bc_weight:
            summary["correction_vs_iv1"] = _correction_metrics(
                model_prediction=pred,
                bc_prediction=bc_prediction,
                human_target=arrays["actions"],
                human_step_mask=human_step_mask,
                intervention_rows=intervention_rows,
                episode_index=episode_index,
                selected_episodes=iv_episodes,
                xyz_std=np.asarray(action_std[:3], dtype=np.float32),
                near_zero_m=args.near_zero_mm / 1000.0,
            )
        model_summaries[str(int(weight))] = summary

    report = {
        "experiment": "RLT residual actor supervised episode-held-out CV; no Q/online/simulation",
        "scope": "frozen Stage-1 representation (cached z_rl), residual actor fine-tuning only; not end-to-end VLA vision fine-tuning",
        "initial_actor": str(initial_path),
        "stage1_checkpoint": str(stage1_path),
        "replay_root": str(args.replay_root.expanduser().resolve()),
        "action_stats": str(stats_path),
        "training": {
            "folds": args.folds,
            "optimizer_updates_per_fold_and_weight": args.steps,
            "batch_size": args.batch_size,
            "actor_lr": config.actor_lr,
            "optimizer": "Adam",
            "q_updates": 0,
            "bc_weight": args.bc_weight,
            "il_anchor_weight": args.il_anchor_weight,
            "intervention_weights": weights,
            "same_initialization_per_fold": True,
            "same_replay_and_il_anchor_batches_across_weights": True,
            "replay_sampling": "balanced outcomes; uniform episode then uniform row",
            "all_rows_of_heldout_episodes_excluded_from_training": True,
            "non_intervention_episodes_kept_in_training": True,
            "reachable_target_projection": True,
            "intervention_targets_clipped_fraction": clipped_fraction,
            "near_zero_correction_threshold_mm": args.near_zero_mm,
            "direction_metric": "first 10 action steps, xyz only, denormalized to physical units",
        },
        "dataset": {
            "episodes": int(np.unique(episode_index).size),
            "transitions": int(len(episode_index)),
            "intervention_episodes": int(len(iv_episodes)),
            "intervention_rows": int(intervention_rows.sum()),
            "intervention_row_fraction": float(intervention_rows.mean()),
            "intervention_episodes_detail": all_episode_summary,
            "fold_assignment": [
                {
                    "fold": i,
                    "episodes": [int(ep) for ep in fold],
                    "rows": int(np.isin(episode_index, fold).sum()),
                    "intervention_rows": int((np.isin(episode_index, fold) & intervention_rows).sum()),
                }
                for i, fold in enumerate(episode_folds)
            ],
            "warning": "the only failure-outcome intervention episode is assigned to one fold; report its result separately",
        },
        "out_of_fold_metrics": {
            "models": model_summaries,
            "folds": fold_runs,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    prediction_path = args.output_dir / "oof_predictions.npz"
    np.savez_compressed(
        prediction_path,
        episode_index=episode_index,
        intervention_rows=intervention_rows,
        human_step_mask=human_step_mask,
        heldout_episode_indices=iv_episodes,
        iv1=predictions[1.0],
        iv10=predictions[10.0],
        iv50=predictions[50.0],
    )
    print(json.dumps(_jsonable(model_summaries), indent=2))
    print(f"report={report_path}")
    print(f"oof_predictions={prediction_path}")


if __name__ == "__main__":
    main()
