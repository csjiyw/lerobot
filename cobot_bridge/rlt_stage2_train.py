"""Train or resume the lightweight Cobot Magic RLT Stage 2 actor/critic.

This command consumes immutable replay ``.npz`` bundles produced after labeled
robot episodes.  It never connects to the bridge and cannot move the robot.
Checkpoints are published atomically so inference never observes a partial file.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .rlt_replay_provenance import actor_training_mask

from .rlt_stage2_data import (
    ACTION_CHUNK_FRAME_STRIDE,
    ACTION_CHUNK_TIMEBASE,
    ACTOR_OUTPUT_SEMANTICS,
    DEFAULT_ACTOR_EXPLORATION_STD,
    DIRECT_ACTOR_OUTPUT_SEMANTICS,
    RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
    calibrated_actor_exploration_std,
    apply_eef_relative_exploration_profile,
    compose_actor_action,
    load_replay_bundle,
)


LOG = logging.getLogger("cobot_bridge.rlt_stage2_train")
STAGE2_FORMAT = 3
DEFAULT_STAGE2_DISCOUNT = 0.96
DEFAULT_CRITIC_PER_ACTOR = 4
DEFAULT_UPDATES_PER_EPISODE = 32
DEFAULT_MAX_HELDOUT_BC_RATIO = 1.05
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_REF_ANCHOR_WEIGHT = 0.5
# Fallback weights for direct trainer runs without explicit
# --q-weight/--bc-weight flags.  The web/pipeline paths always pass explicit
# values (sourced from RL-Token-SmolVLA/rlt_actor_weights.json).
RLT_ACTOR_WEIGHTS = {
    "actor": {"q_weight": 1.0, "bc_weight": 5.0},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--actor-init-checkpoint",
        type=Path,
        help=(
            "Actor checkpoint used only when latest.pt does not exist; accepts "
            "IL BC or compatible offline STEAM actor checkpoints."
        ),
    )
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument(
        "--updates-per-episode", type=int, default=DEFAULT_UPDATES_PER_EPISODE
    )
    parser.add_argument(
        "--actor-validation-cache",
        type=Path,
        help="Actor-BC feature cache used to report held-out regressions.",
    )
    parser.add_argument(
        "--max-heldout-bc-ratio", type=float, default=DEFAULT_MAX_HELDOUT_BC_RATIO
    )
    parser.add_argument(
        "--il-anchor-weight",
        type=float,
        default=0.0,
        help="Weight of an additional full-IL training-split BC batch per actor update.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--discount", type=float, default=DEFAULT_STAGE2_DISCOUNT)
    parser.add_argument(
        "--critic-target-mode",
        choices=("td", "mc", "hybrid"),
        default="td",
        help="Critic supervision: TD bootstrap, exact replay MC return, or a hybrid.",
    )
    parser.add_argument(
        "--critic-mc-weight",
        type=float,
        default=1.0,
        help="MC regression weight used by --critic-target-mode=hybrid.",
    )
    parser.add_argument(
        "--critic-per-actor", type=int, default=DEFAULT_CRITIC_PER_ACTOR
    )
    parser.add_argument(
        "--q-target-clip",
        type=float,
        default=1.0,
        help="Clamp bootstrapped target Q to [0, clip] (0 disables). "
        "Matches remote tuned recipe q_target_clip=1.0.",
    )
    parser.add_argument(
        "--actor-q-gate",
        choices=("synthetic", "real-rows"),
        default="synthetic",
        help="synthetic: Q-gradient only on is_synthetic rows (v4). "
        "real-rows: Q-max on every row of the frame-gated actor batch, "
        "matching the tuned offline recipe (gated real rows).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=DEFAULT_GRAD_CLIP_NORM,
        help="Maximum Actor/Critic gradient norm. 0 disables clipping.",
    )
    parser.add_argument(
        "--q-weight",
        type=float,
        default=None,
        help="Override the actor Q loss weight; 0 runs the actor-BC diagnostic.",
    )
    parser.add_argument(
        "--bc-weight",
        type=float,
        default=None,
        help="Override the actor BC loss weight.",
    )
    parser.add_argument(
        "--intervention-bc-weight",
        type=float,
        default=1.0,
        help=(
            "Extra BC multiplier on human-intervention transitions "
            "(is_intervention=1). 1.0 (default) keeps uniform BC; a larger "
            "value (e.g. 50.0) makes the actor learn the operator's "
            "correction actions at failure states."
        ),
    )
    parser.add_argument(
        "--replay-human-bc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clone executed actions for human-intervention rows in replay. "
            "Disable when a separate phase-specific BC-only cache supplies "
            "the corrective Actor supervision."
        ),
    )
    parser.add_argument(
        "--syn-bc-mask",
        type=int,
        default=0,
        help=(
            "v6 channel separation: when replay bundles carry an "
            "is_synthetic flag, drop the actor BC loss on synthetic rows "
            "(the Q-gradient, which is already restricted to synthetic "
            "rows, then owns them exclusively). BC keeps supervising real "
            "rows only. 0 (default) keeps legacy uniform BC."
        ),
    )
    parser.add_argument(
        "--ref-anchor-weight",
        type=float,
        default=DEFAULT_REF_ANCHOR_WEIGHT,
        help=(
            "Extra BC weight pulling the actor output toward the reference VLA "
            "action on every transition. The default 0.5 matches the tuned "
            "always-on reference regularizer."
        ),
    )
    parser.add_argument(
        "--ref-anchor-mask",
        choices=("none", "outside_box"),
        default="none",
        help=(
            "Which transitions the ref anchor applies to. 'none' (default) "
            "keeps the legacy behavior: every bc-supervised transition. "
            "'outside_box' anchors only transitions whose EEF state is "
            "OUTSIDE the hole gate box, pinning the actor to the VLA prior "
            "during approach while leaving the in-box region free to be "
            "corrected. Requires --ref-anchor-weight > 0 and --action-stats."
        ),
    )
    parser.add_argument(
        "--ref-anchor-box",
        default="0.31,0.39,0.30,0.35,0.06,0.13",
        help=(
            "Physical hole gate box 'x0,x1,y0,y1,z0,z1' in meters (same box "
            "the operator uses for intervention gating). Only used with "
            "--ref-anchor-mask outside_box."
        ),
    )
    parser.add_argument(
        "--action-stats",
        type=Path,
        default=None,
        help=(
            "Dataset meta/stats.json path whose 'action' mean/std normalize "
            "the replay states. Required for --ref-anchor-mask outside_box."
        ),
    )
    parser.add_argument(
        "--actor-output-semantics",
        choices=("residual", "direct"),
        default="residual",
        help=(
            "Actor output interpretation: 'residual' (default) composes a "
            "bounded correction around the reference VLA action; 'direct' "
            "makes the actor output the full bounded action, replacing the "
            "reference. Requires a matching-actor_output_semantics "
            "initialization checkpoint."
        ),
    )
    parser.add_argument(
        "--freeze-actor",
        action="store_true",
        help="Train only the critic and keep the initialized actor frozen.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--training-phase", choices=("actor",), default="actor")
    parser.add_argument(
        "--actor-exploration-std", type=float, default=DEFAULT_ACTOR_EXPLORATION_STD
    )
    parser.add_argument(
        "--actor-exploration-std-per-dim",
        type=float,
        nargs=7,
        default=None,
        metavar="STD",
        help=(
            "Raw-space exploration noise override per action dimension "
            "(x y z rx ry rz gripper). Overrides the automatic calibration and "
            "the EEF-relative profile shaping when set."
        ),
    )
    parser.add_argument(
        "--actor-action-scale",
        type=float,
        nargs=7,
        default=None,
        metavar="SCALE",
        help=(
            "Fixed normalized Actor output bound per action dimension. Use this "
            "for controlled fresh runs: deriving a new bound from each replay's "
            "maximum action silently changes the deployed policy class."
        ),
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Atomically updated JSON progress for the dashboard.",
    )
    parser.add_argument(
        "--balanced-outcomes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample successful and failed transitions 50:50 (default: enabled).",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=None,
        help="Override the actor learning rate from the Stage 1 config.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Override the critic learning rate from the Stage 1 config.",
    )
    parser.add_argument(
        "--ref-dropout",
        type=float,
        default=None,
        help="Override the reference dropout probability from the Stage 1 config.",
    )
    parser.add_argument(
        "--actor-residual-scale",
        type=float,
        default=None,
        help="Override the residual composition scale (max action correction) from the Stage 1 config.",
    )
    parser.add_argument(
        "--allow-clipped-reference",
        action="store_true",
        help=(
            "Skip the pre-training check that the actor action scale covers the "
            "replay reference range. Only for reproducing historical runs: a "
            "narrower scale truncates the VLA reference at zero residual."
        ),
    )
    parser.add_argument(
        "--no-bc-target-projection",
        action="store_true",
        help="Use raw replay BC targets without residual reachability projection.",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help=(
            "Override the RL action-chunk length (n_action_steps_rl) from the "
            "Stage 1 config. Use with replay bundles whose action chunks were "
            "truncated to a shorter horizon (e.g. 10 for the remote chunk10 "
            "recipe on a chunk30 VLA: the reference is the first 10 steps of "
            "the VLA plan)."
        ),
    )
    parser.add_argument(
        "--utd-ratio",
        type=float,
        default=None,
        help=(
            "Update-to-data ratio from the remote recipe. When set and "
            "--updates is not given, the total number of gradient updates is "
            "utd_ratio x buffer_transitions (remote: 5 updates per new "
            "transition)."
        ),
    )
    parser.add_argument(
        "--warm-up-steps",
        type=int,
        default=None,
        help=(
            "Remote warm_up_steps. With --total-steps, the first "
            "warm_up_steps/total_steps fraction of the run trains the critic "
            "only (actor frozen at VLA passthrough): the offline analogue of "
            "the online warm-up where the actor is not yet in the loop."
        ),
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Remote total_steps (only used with --warm-up-steps).",
    )
    parser.add_argument(
        "--replay-success-fraction",
        type=float,
        default=0.0,
        help=(
            "Remote replay_success_fraction. Each batch reserves "
            "round(fraction x batch_size) transitions drawn explicitly from "
            "successful episodes; the remainder is sampled uniformly. "
            "Overrides --balanced-outcomes."
        ),
    )
    parser.add_argument(
        "--actor-start-step",
        type=int,
        default=None,
        help=(
            "Remote actor_start_step: the per-episode step where control "
            "hands over from the VLA to the actor (here: block-placement "
            "start step in 30 Hz frames, derived from the 073 gripper data). "
            "Reference/unknown rows require a verified episode frame at or "
            "after this point. Recorded Actor and human corrections remain "
            "eligible regardless of decision density or omitted prefix. "
            "Also stored in the checkpoint as the "
            "deployment handover point."
        ),
    )
    parser.add_argument(
        "--actor-frame-gate-strict",
        action="store_true",
        help=(
            "Train the Actor only on rows with a verified episode frame at or "
            "after --actor-start-step. Use this when deployment is confined to "
            "a late precision phase and earlier human corrections are out of scope."
        ),
    )
    parser.add_argument(
        "--auto-intervention-chunk-rms",
        type=float,
        default=0.125,
        help="Remote auto_intervention_chunk_rms trust-region threshold.",
    )
    parser.add_argument(
        "--auto-intervention-first-action-rms",
        type=float,
        default=0.125,
        help="Remote auto_intervention_first_action_rms trust-region threshold.",
    )
    parser.add_argument(
        "--auto-intervention-max-abs",
        type=float,
        default=0.4,
        help="Remote auto_intervention_max_abs trust-region threshold.",
    )
    return parser


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep progress JSON standards-compliant when actor updates are delayed."""

    return {name: value for name, value in metrics.items() if math.isfinite(value)}


def _write_failed_progress(path: Path, error: BaseException) -> None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        payload = {}
    payload.update(
        {
            "state": "failed",
            "updated_at_ns": time.time_ns(),
            "finished_at_ns": time.time_ns(),
            "eta_seconds": None,
            "error": f"{type(error).__name__}: {error}",
        }
    )
    _atomic_json(payload, path)


@torch.no_grad()
def _actor_deviation_stats(
    actor: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    action_scale: torch.Tensor,
    output_semantics: str = ACTOR_OUTPUT_SEMANTICS,
) -> dict[str, float]:
    deviations: list[np.ndarray] = []
    actor.eval()
    for start in range(0, int(arrays["z_rl"].shape[0]), 2048):
        stop = min(start + 2048, int(arrays["z_rl"].shape[0]))
        indices = np.arange(start, stop)
        batch = _batch(arrays, indices, device)
        raw_actions, _ = actor(batch["z_rl"], batch["state"], batch["ref_actions"])
        actions = compose_actor_action(
            raw_actions,
            batch["ref_actions"],
            action_scale,
            output_semantics=output_semantics,
            residual_scale=actor.config.actor_residual_scale,
        )
        deviations.append(
            torch.abs(actions - batch["ref_actions"]).cpu().float().numpy().reshape(-1)
        )
    values = np.concatenate(deviations)
    return {
        "mean_abs_error_to_reference": float(values.mean()),
        "p95_abs_error_to_reference": float(np.quantile(values, 0.95)),
        "max_abs_error_to_reference": float(values.max()),
    }


def _sample_actor_action(
    action_mean: torch.Tensor,
    exploration_std: float,
    exploration_std_per_dim: torch.Tensor | None = None,
) -> torch.Tensor:
    if not math.isfinite(exploration_std) or exploration_std < 0:
        raise ValueError("actor_exploration_std must be finite and non-negative")
    if exploration_std == 0:
        return action_mean
    if exploration_std_per_dim is None:
        noise_std = torch.full(
            (action_mean.shape[-1],),
            float(exploration_std),
            dtype=action_mean.dtype,
            device=action_mean.device,
        )
    else:
        noise_std = exploration_std_per_dim.to(action_mean).flatten()
        if noise_std.shape != (action_mean.shape[-1],):
            raise ValueError("exploration_std_per_dim must match the action dimension")
        if not torch.isfinite(noise_std).all() or not bool(torch.all(noise_std >= 0)):
            raise ValueError("exploration_std_per_dim must be finite and non-negative")
    return action_mean + torch.randn_like(action_mean) * noise_std.reshape(1, 1, -1)


def _physical_action_std(policy_path: Path, action_dim: int) -> np.ndarray:
    candidates = sorted(
        policy_path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one policy action normalizer under {policy_path}, found {len(candidates)}"
        )
    stats = load_file(candidates[0])
    values = stats["action.std"].cpu().numpy().astype(np.float32)[:action_dim]
    if values.shape != (action_dim,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("policy physical action std is invalid")
    return values


@torch.no_grad()
def _heldout_actor_gt_bc_loss(
    actor: torch.nn.Module,
    cache_path: Path,
    *,
    device: torch.device,
    action_scale: torch.Tensor,
    action_weights: torch.Tensor | None = None,
    seed: int = 42,
    validation_fraction: float = 0.1,
    output_semantics: str = ACTOR_OUTPUT_SEMANTICS,
) -> tuple[float, list[float]]:
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Actor validation cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
        raise ValueError("Actor validation cache uses an incompatible action timebase")
    with np.load(cache_path, allow_pickle=False) as stored:
        cached = {name: np.asarray(stored[name]) for name in stored.files}
    required = {"z_rl", "state", "reference", "target", "valid", "episode_index"}
    if not required.issubset(cached):
        raise ValueError(
            f"Actor validation cache is missing {sorted(required - set(cached))}"
        )
    episodes = np.unique(cached["episode_index"].astype(np.int64))
    shuffled = np.random.default_rng(seed).permutation(episodes)
    val_count = max(1, min(episodes.size - 1, round(episodes.size * validation_fraction)))
    indices = np.flatnonzero(np.isin(cached["episode_index"], shuffled[:val_count]))
    squared_error = 0.0
    denominator = 0.0
    per_dim_squared_error: np.ndarray | None = None
    per_dim_denominator = 0.0
    actor.eval()
    for start in range(0, indices.size, 2048):
        selected = indices[start : start + 2048]
        z_rl = torch.as_tensor(cached["z_rl"][selected], device=device)
        state = torch.as_tensor(cached["state"][selected], device=device)
        reference = torch.as_tensor(cached["reference"][selected], device=device)
        target = torch.as_tensor(cached["target"][selected], device=device)
        valid = torch.as_tensor(
            cached["valid"][selected], dtype=torch.bool, device=device
        )
        raw_action, _ = actor(z_rl, state, reference, training=False)
        prediction = compose_actor_action(
            raw_action,
            reference,
            action_scale,
            output_semantics=output_semantics,
            residual_scale=actor.config.actor_residual_scale,
        )
        weights = valid.to(prediction).unsqueeze(-1)
        if action_weights is not None:
            weights = weights * action_weights.to(prediction).reshape(1, 1, -1)
        else:
            weights = weights.expand_as(prediction)
        squared_error += float(((prediction - target).square() * weights).sum().cpu())
        denominator += float(weights.sum().cpu())
        valid_weights = valid.to(prediction).unsqueeze(-1)
        batch_per_dim = (
            (prediction - target).square() * valid_weights
        ).sum(dim=(0, 1)).cpu().numpy()
        if per_dim_squared_error is None:
            per_dim_squared_error = batch_per_dim
        else:
            per_dim_squared_error += batch_per_dim
        per_dim_denominator += float(valid.sum().cpu())
    if denominator <= 0:
        raise ValueError("Actor validation split has no valid action labels")
    if per_dim_denominator <= 0 or per_dim_squared_error is None:
        raise ValueError("Actor validation split has no per-dimension labels")
    return (
        squared_error / denominator,
        (per_dim_squared_error / per_dim_denominator).tolist(),
    )


def _load_il_anchor_cache(
    cache_path: Path, *, seed: int, validation_fraction: float = 0.1
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load only the IL training split; held-out episodes remain untouched."""

    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Actor validation cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
        raise ValueError("IL anchor cache uses an incompatible action timebase")
    with np.load(cache_path, allow_pickle=False) as stored:
        cached = {name: np.asarray(stored[name]) for name in stored.files}
    required = {"z_rl", "state", "reference", "target", "valid", "episode_index"}
    if not required.issubset(cached):
        raise ValueError(f"IL anchor cache is missing {sorted(required - set(cached))}")
    episodes = np.unique(cached["episode_index"].astype(np.int64))
    if episodes.size < 2:
        raise ValueError("IL anchor cache needs at least two episodes")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    val_count = max(1, min(episodes.size - 1, round(episodes.size * validation_fraction)))
    training_indices = np.flatnonzero(
        ~np.isin(cached["episode_index"], shuffled[:val_count])
    )
    if training_indices.size == 0:
        raise ValueError("IL anchor training split is empty")
    return cached, training_indices


def _derive_actor_action_scale(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Fit a fixed symmetric bound to the initial replay action space."""

    values = np.concatenate(
        [arrays["actions"], arrays["ref_actions"], arrays["next_ref_actions"]], axis=0
    )
    scale = 1.05 * np.max(np.abs(values), axis=(0, 1))
    scale = np.maximum(scale, 0.1).astype(np.float32)
    if scale.ndim != 1 or not np.isfinite(scale).all():
        raise ValueError("could not derive a finite per-joint actor action scale")
    return scale


REFERENCE_REACHABLE_CHANNELS = ("ref_actions", "next_ref_actions", "actions")


def _assert_reference_reachable(
    arrays: dict[str, np.ndarray],
    scale: np.ndarray,
    *,
    allow_clipped: bool = False,
) -> None:
    """Refuse to train when the composition would truncate the VLA reference.

    ``compose_actor_action`` clamps ``reference + residual`` to ``±scale``, so a
    scale narrower than the replay's own action range truncates the reference
    even at zero residual: the "inherits the VLA" initialisation is a lie, the
    BC target on those rows is unreachable, and the deployed policy emits a
    clipped action unrelated to the task.
    """

    values = np.concatenate(
        [
            np.asarray(arrays[name], dtype=np.float32).reshape(-1, arrays[name].shape[-1])
            for name in REFERENCE_REACHABLE_CHANNELS
        ],
        axis=0,
    )
    worst = np.abs(values).max(axis=0)
    offending = np.flatnonzero(worst > scale)
    if offending.size == 0:
        return
    required = _derive_actor_action_scale(arrays)
    clipped_rows = np.mean(
        np.any(np.abs(values) > scale, axis=1)
    ) * 100.0
    detail = ", ".join(
        f"dim {int(i)}: |action| up to {worst[i]:.3f} > scale {scale[i]:.3f}"
        for i in offending
    )
    message = (
        "actor action scale truncates the replay reference channel "
        f"({clipped_rows:.3f}% of action steps): {detail}. "
        f"Use --actor-action-scale {' '.join(f'{v:.17g}' for v in required)} "
        "or pass --allow-clipped-reference to reproduce a historical run."
    )
    if allow_clipped:
        LOG.warning("%s -- continuing because --allow-clipped-reference was set", message)
        return
    raise ValueError(message)


def _assert_stable_metrics(
    metrics: dict[str, float],
    global_update: int,
    *,
    bc_weight_active: bool = True,
) -> None:
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite Stage 2 metrics at update {global_update}: {metrics}")
    if metrics["critic_loss"] > 1.0e4:
        raise RuntimeError(
            f"critic diverged at update {global_update}: loss={metrics['critic_loss']:.6g}"
        )
    if abs(metrics["q1_mean"]) > 1.0e3 or abs(metrics["q2_mean"]) > 1.0e3:
        raise RuntimeError(f"critic Q magnitude diverged at update {global_update}: {metrics}")
    # With bc_weight == 0 the BC term carries no gradient: it is a diagnostic
    # only, and its magnitude is dominated by the replay-action space offset
    # of the offline buffer, not by actor instability. Enforce the absolute
    # threshold only when BC actually trains the actor.
    if bc_weight_active and "bc_loss" in metrics and metrics["bc_loss"] > 10.0:
        raise RuntimeError(
            f"actor diverged at update {global_update}: bc_loss={metrics['bc_loss']:.6g}"
        )
    # Actor drift guard for the pure-RL case: the residual is bounded by
    # residual_scale * action_scale, so a healthy actor stays close to the
    # reference; a large ref-anchor MSE means the Q-gradient pushed it away.
    if not bc_weight_active and "ref_anchor_loss" in metrics and metrics["ref_anchor_loss"] > 10.0:
        raise RuntimeError(
            f"actor drifted from reference at update {global_update}: "
            f"ref_anchor_loss={metrics['ref_anchor_loss']:.6g}"
        )


def discover_replay_bundles(replay_root: Path) -> list[Path]:
    if not replay_root.exists():
        return []
    return sorted(path.resolve() for path in replay_root.rglob("*.npz") if path.is_file())


def replay_names(paths: list[Path]) -> set[str]:
    """Return relocation-stable replay identities for the flat episode store."""

    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate replay filenames are ambiguous: {duplicates}")
    return set(names)


def load_replay_dataset(
    paths: list[Path], *, stage1_checkpoint: Path, discount: float = DEFAULT_STAGE2_DISCOUNT
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    expected_stage1 = str(stage1_checkpoint.expanduser().resolve())
    parts: dict[str, list[np.ndarray]] = {}
    metadata: list[dict[str, Any]] = []
    expected_dimensions: tuple[int, int, int, int] | None = None
    for path in paths:
        arrays, meta = load_replay_bundle(path)
        actual_stage1 = str(Path(meta["stage1_checkpoint"]).expanduser().resolve())
        if actual_stage1 != expected_stage1:
            raise ValueError(
                f"replay {path} belongs to a different Stage 1 checkpoint: {actual_stage1}"
            )
        if meta.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
            raise ValueError(f"replay {path} uses an incompatible action timebase")
        dimensions = (
            int(meta["z_dim"]),
            int(meta["state_dim"]),
            int(meta["action_steps"]),
            int(meta["action_dim"]),
        )
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError(f"incompatible replay dimensions in {path}: {dimensions}")
        for name, values in arrays.items():
            parts.setdefault(name, []).append(values)
        parts.setdefault("replay_episode_index", []).append(
            np.full((arrays["z_rl"].shape[0], 1), len(metadata), dtype=np.int64)
        )
        outcome_value = float(meta.get("outcome") == "success")
        parts.setdefault("outcome_success", []).append(
            np.full((arrays["z_rl"].shape[0], 1), outcome_value, dtype=np.float32)
        )
        metadata.append({**meta, "path": str(path)})
    if not metadata:
        raise ValueError(f"no replay bundles found under {paths!r}")
    merged = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    # Every offline replay episode has a known complete reward trace.  Keep its
    # exact discounted return so a critic can be calibrated without TD3
    # bootstrapping through actor-generated out-of-support actions.
    rewards = merged["rewards"].reshape(-1)
    dones = merged["dones"].reshape(-1)
    mc_returns = np.empty_like(rewards, dtype=np.float32)
    running_return = 0.0
    for index in range(rewards.size - 1, -1, -1):
        running_return = float(rewards[index]) + (
            0.0 if bool(dones[index]) else discount ** float(merged["discount_steps"][index, 0]) * running_return
        )
        mc_returns[index] = running_return
    merged["mc_returns"] = mc_returns[:, None]
    return merged, metadata


def _replay_root_for_metadata(paths: list[Path]) -> str:
    """Return the shared replay directory recorded in published Stage-2 metadata."""

    parents = [str(path.expanduser().resolve().parent) for path in paths]
    return str(Path(os.path.commonpath(parents)))


def _batch(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    return {
        name: torch.as_tensor(values[indices], dtype=torch.float32, device=device)
        for name, values in arrays.items()
    }


def _sample_balanced_outcome_indices(
    generator: np.random.Generator,
    outcome_success: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    labels = np.asarray(outcome_success).reshape(-1).astype(bool)
    success = np.flatnonzero(labels)
    failure = np.flatnonzero(~labels)
    if success.size == 0 or failure.size == 0:
        raise ValueError(
            "balanced outcome sampling requires at least one success and one failure episode"
        )
    success_count = batch_size // 2
    failure_count = batch_size - success_count
    selected = np.concatenate(
        [
            generator.choice(
                success, size=success_count, replace=success.size < success_count
            ),
            generator.choice(
                failure, size=failure_count, replace=failure.size < failure_count
            ),
        ]
    )
    generator.shuffle(selected)
    return selected


def _sample_balanced_episode_indices(
    generator: np.random.Generator,
    outcome_success: np.ndarray,
    replay_episode_index: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Balance outcomes, then sample episodes uniformly within each outcome."""

    labels = np.asarray(outcome_success).reshape(-1).astype(bool)
    episode_ids = np.asarray(replay_episode_index).reshape(-1).astype(np.int64)
    if labels.shape != episode_ids.shape or labels.size == 0:
        raise ValueError("outcome and episode labels must be non-empty and aligned")

    selected_parts: list[np.ndarray] = []
    for success_value, count in ((True, batch_size // 2), (False, batch_size - batch_size // 2)):
        eligible = np.unique(episode_ids[labels == success_value])
        if eligible.size == 0:
            raise ValueError("episode-balanced sampling requires both outcomes")
        sampled_episodes = generator.choice(eligible, size=count, replace=True)
        sampled_rows = np.empty(count, dtype=np.int64)
        for index, episode_id in enumerate(sampled_episodes):
            rows = np.flatnonzero((episode_ids == episode_id) & (labels == success_value))
            sampled_rows[index] = int(generator.choice(rows))
        selected_parts.append(sampled_rows)
    selected = np.concatenate(selected_parts)
    generator.shuffle(selected)
    return selected


def _sample_success_fraction_indices(
    generator: np.random.Generator,
    outcome_success: np.ndarray,
    total_rows: int,
    batch_size: int,
    success_fraction: float,
) -> np.ndarray:
    """Remote replay_buffer.sample(success_fraction): reserve
    round(batch_size * success_fraction) rows drawn from successful
    transitions, fill the remainder uniformly from the whole buffer."""

    labels = np.asarray(outcome_success).reshape(-1).astype(bool)
    success = np.flatnonzero(labels)
    n_success = min(int(round(batch_size * success_fraction)), success.size)
    selected = np.empty(batch_size, dtype=np.int64)
    if n_success > 0:
        selected[:n_success] = generator.choice(
            success, size=n_success, replace=n_success > success.size
        )
    selected[n_success:] = generator.choice(
        total_rows, size=batch_size - n_success, replace=batch_size - n_success > total_rows
    )
    generator.shuffle(selected)
    return selected


def _actor_bc_target_and_mask(
    batch: dict[str, torch.Tensor],
    *,
    replay_human_bc: bool = True,
    target_mode: str = "rlinf_reference_human",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the RLinf RLT BC target with IL demonstrations added.

    ``rlinf_reference_human`` follows the RLinf TD3+BC recipe: autonomous
    rollout rows clone reference while human/IL rows clone executed action.
    ``executed`` is used by the direct-Actor route: every row clones its true
    recorded action so there is no loss term pulling a direct policy toward
    reference.
    """

    if target_mode == "executed":
        return batch["actions"], torch.ones_like(batch["is_demo"].bool().squeeze(-1))
    if target_mode != "rlinf_reference_human":
        raise ValueError(f"unknown actor BC target mode: {target_mode!r}")
    demo = batch["is_demo"].bool().squeeze(-1)
    human_steps = torch.zeros_like(batch["actions"][:, :, 0], dtype=torch.bool)
    if replay_human_bc:
        if "action_human_mask" in batch:
            human_steps = batch["action_human_mask"].bool()
        else:
            human_steps = batch["is_intervention"].bool().expand_as(human_steps)
    executed_steps = human_steps | demo[:, None]
    target = torch.where(
        executed_steps[:, :, None], batch["actions"], batch["ref_actions"]
    )
    supervised = torch.ones_like(demo)
    return target, supervised


def _reachable_bc_target(
    target: torch.Tensor,
    reference: torch.Tensor,
    action_scale: torch.Tensor,
    *,
    output_semantics: str,
    residual_scale: float,
) -> torch.Tensor:
    """Project BC labels into the Actor's physically representable action box."""

    scale = torch.as_tensor(
        action_scale, dtype=target.dtype, device=target.device
    ).reshape(1, 1, -1)
    if output_semantics == RESIDUAL_ACTOR_OUTPUT_SEMANTICS:
        low = torch.clamp(reference - scale * residual_scale, min=-scale, max=scale)
        high = torch.clamp(reference + scale * residual_scale, min=-scale, max=scale)
    elif output_semantics == DIRECT_ACTOR_OUTPUT_SEMANTICS:
        low, high = -scale, scale
    else:
        raise ValueError(f"unsupported actor output semantics: {output_semantics!r}")
    return torch.clamp(target, min=low, max=high)


def _masked_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    supervised: torch.Tensor,
    action_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted action MSE.

    ``action_weights`` shape (dim,) weights each action dimension;
    ``sample_weights`` shape (batch,) weights each transition (e.g. to
    up-weight human-intervention corrections over autonomous chunks).
    """
    if supervised.any():
        error = (predicted[supervised] - target[supervised]).square()
        weights = None
        if action_weights is not None:
            weights = action_weights.to(error).reshape(1, 1, -1).expand_as(error)
        if sample_weights is not None:
            per_sample = sample_weights.to(error)[supervised].reshape(-1, 1, 1)
            weights = per_sample.expand_as(error) if weights is None else weights * per_sample
        if weights is None:
            return error.mean()
        return (error * weights).sum() / weights.sum().clamp_min(1)
    return predicted.sum() * 0.0


def _valid_critic_mse(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    mask = valid.reshape(-1).bool()
    return F.mse_loss(predicted[mask], target[mask]) if mask.any() else predicted.sum() * 0.0


def _masked_timestep_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    action_weights: torch.Tensor,
) -> torch.Tensor:
    weights = valid.to(predicted).unsqueeze(-1) * action_weights.to(predicted).reshape(
        1, 1, -1
    )
    return ((predicted - target).square() * weights).sum() / weights.sum().clamp_min(1)


def train_stage2(
    *,
    stage1_checkpoint: Path,
    actor_init_checkpoint: Path | None = None,
    replay_paths: list[Path],
    output_dir: Path,
    device: str,
    updates: int | None,
    batch_size: int | None,
    seed: int,
    log_freq: int,
    progress_file: Path | None = None,
    training_phase: str = "actor",
    actor_exploration_std: float = DEFAULT_ACTOR_EXPLORATION_STD,
    actor_exploration_std_per_dim_override: list[float] | None = None,
    actor_action_scale_override: list[float] | None = None,
    allow_clipped_reference: bool = False,
    discount: float = DEFAULT_STAGE2_DISCOUNT,
    critic_target_mode: str = "td",
    critic_mc_weight: float = 1.0,
    q_target_clip: float = 1.0,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    actor_q_gate: str = "synthetic",
    critic_per_actor: int = DEFAULT_CRITIC_PER_ACTOR,
    balanced_outcomes: bool = True,
    updates_per_episode: int = DEFAULT_UPDATES_PER_EPISODE,
    actor_validation_cache: Path | None = None,
    max_heldout_bc_ratio: float = DEFAULT_MAX_HELDOUT_BC_RATIO,
    il_anchor_weight: float = 0.0,
    q_weight: float | None = None,
    bc_weight: float | None = None,
    intervention_bc_weight: float = 1.0,
    replay_human_bc: bool = True,
    syn_bc_mask: int = 0,
    ref_anchor_weight: float = DEFAULT_REF_ANCHOR_WEIGHT,
    ref_anchor_mask: str = "none",
    ref_anchor_box: str = "0.31,0.39,0.30,0.35,0.06,0.13",
    action_stats_path: Path | None = None,
    freeze_actor: bool = False,
    actor_output_semantics: str = "residual",
    actor_lr: float | None = None,
    critic_lr: float | None = None,
    ref_dropout: float | None = None,
    actor_residual_scale: float | None = None,
    project_bc_targets: bool = True,
    action_steps: int | None = None,
    utd_ratio: float | None = None,
    warm_up_steps: int | None = None,
    total_steps: int | None = None,
    replay_success_fraction: float = 0.0,
    actor_start_step: int | None = None,
    actor_frame_gate_strict: bool = False,
    auto_intervention_chunk_rms: float = 0.125,
    auto_intervention_first_action_rms: float = 0.125,
    auto_intervention_max_abs: float = 0.4,
) -> dict[str, Any]:
    if log_freq <= 0:
        raise ValueError("log_freq must be positive")
    if training_phase not in RLT_ACTOR_WEIGHTS:
        raise ValueError(f"invalid training_phase: {training_phase!r}")
    if not math.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("discount must be in (0, 1]")
    if critic_target_mode not in {"td", "mc", "hybrid"}:
        raise ValueError("critic_target_mode must be one of td, mc, hybrid")
    if not math.isfinite(critic_mc_weight) or critic_mc_weight < 0:
        raise ValueError("critic_mc_weight must be finite and non-negative")
    if critic_target_mode == "hybrid" and critic_mc_weight == 0:
        raise ValueError("hybrid critic target requires critic_mc_weight > 0")
    if not math.isfinite(q_target_clip) or q_target_clip < 0:
        raise ValueError("q_target_clip must be finite and non-negative")
    if not math.isfinite(grad_clip_norm) or grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be finite and non-negative")
    if actor_q_gate not in {"synthetic", "real-rows"}:
        raise ValueError("actor_q_gate must be synthetic or real-rows")
    if actor_frame_gate_strict and actor_start_step is None:
        raise ValueError("actor_frame_gate_strict requires actor_start_step")
    if critic_per_actor <= 0:
        raise ValueError("critic_per_actor must be positive")
    if updates_per_episode <= 0:
        raise ValueError("updates_per_episode must be positive")
    if not math.isfinite(max_heldout_bc_ratio) or max_heldout_bc_ratio < 1:
        raise ValueError("max_heldout_bc_ratio must be finite and at least 1")
    if not math.isfinite(il_anchor_weight) or il_anchor_weight < 0:
        raise ValueError("il_anchor_weight must be finite and non-negative")
    default_actor_weights = RLT_ACTOR_WEIGHTS[training_phase]
    effective_q_weight = (
        float(default_actor_weights["q_weight"])
        if q_weight is None
        else float(q_weight)
    )
    effective_bc_weight = (
        float(default_actor_weights["bc_weight"])
        if bc_weight is None
        else float(bc_weight)
    )
    if not math.isfinite(effective_q_weight) or effective_q_weight < 0:
        raise ValueError("q_weight must be finite and non-negative")
    if not math.isfinite(effective_bc_weight) or effective_bc_weight < 0:
        raise ValueError("bc_weight must be finite and non-negative")
    if (
        not math.isfinite(float(intervention_bc_weight))
        or float(intervention_bc_weight) <= 0
    ):
        raise ValueError("intervention_bc_weight must be finite and positive")
    if not math.isfinite(ref_anchor_weight) or ref_anchor_weight < 0:
        raise ValueError("ref_anchor_weight must be finite and non-negative")
    if ref_anchor_mask not in {"none", "outside_box"}:
        raise ValueError("ref_anchor_mask must be 'none' or 'outside_box'")
    ref_anchor_box_np: np.ndarray | None = None
    action_stats_mean: np.ndarray | None = None
    action_stats_std: np.ndarray | None = None
    bx0 = bx1 = by0 = by1 = bz0 = bz1 = 0.0
    if ref_anchor_mask == "outside_box":
        if ref_anchor_weight <= 0:
            raise ValueError(
                "ref_anchor_mask outside_box requires ref_anchor_weight > 0"
            )
        if action_stats_path is None:
            raise ValueError(
                "ref_anchor_mask outside_box requires --action-stats "
                "(dataset meta/stats.json with 7-dim 'action' mean/std)"
            )
        try:
            ref_anchor_box_np = np.asarray(
                [float(v) for v in ref_anchor_box.split(",")], dtype=np.float64
            )
        except ValueError as exc:
            raise ValueError(
                f"ref_anchor_box must be 'x0,x1,y0,y1,z0,z1': {ref_anchor_box!r}"
            ) from exc
        if ref_anchor_box_np.shape != (6,) or not bool(
            np.all(np.isfinite(ref_anchor_box_np))
        ):
            raise ValueError(f"ref_anchor_box is invalid: {ref_anchor_box!r}")
        (bx0, bx1, by0, by1, bz0, bz1) = (float(v) for v in ref_anchor_box_np)
        if not (bx0 < bx1 and by0 < by1 and bz0 < bz1):
            raise ValueError(
                f"ref_anchor_box bounds must be ordered: {ref_anchor_box!r}"
            )
        stats_doc = json.loads(action_stats_path.read_text())
        action_block = stats_doc.get("action", {})
        if "mean" not in action_block or "std" not in action_block:
            raise ValueError(f"action stats missing in {action_stats_path}")
        action_stats_mean = np.asarray(
            action_block["mean"], dtype=np.float64
        )[:7]
        action_stats_std = np.asarray(action_block["std"], dtype=np.float64)[:7]
        if (
            action_stats_mean.shape != (7,)
            or action_stats_std.shape != (7,)
            or not bool(np.all(np.isfinite(action_stats_mean)))
            or not bool(np.all(action_stats_std > 0))
        ):
            raise ValueError(f"action stats are invalid: {action_stats_path}")
    if actor_output_semantics not in {"residual", "direct"}:
        raise ValueError("actor_output_semantics must be 'residual' or 'direct'")
    semantics = (
        RESIDUAL_ACTOR_OUTPUT_SEMANTICS
        if actor_output_semantics == "residual"
        else DIRECT_ACTOR_OUTPUT_SEMANTICS
    )
    # Direct Actor is warm-started and supervised against the recorded IL /
    # human target, not reference. This is intentionally distinct from the
    # RLinf reference-cloning loss used by the residual compatibility route.
    bc_target_mode = (
        "executed"
        if semantics == DIRECT_ACTOR_OUTPUT_SEMANTICS
        else "rlinf_reference_human"
    )
    if il_anchor_weight > 0 and actor_validation_cache is None:
        raise ValueError("il_anchor_weight requires --actor-validation-cache")
    _sample_actor_action(torch.zeros(1), actor_exploration_std)
    stage1_checkpoint = stage1_checkpoint.expanduser().resolve()
    if actor_init_checkpoint is not None:
        actor_init_checkpoint = actor_init_checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    stage1 = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    if stage1.get("stage") != "rlt_stage1_joint" or stage1.get("rlt_architecture") != "rlinf_prefix_v1":
        raise ValueError("Stage 2 requires an RLinf prefix-only joint Stage 1 checkpoint")

    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLCriticMLP, RLActorMLP

    latest = output_dir / "latest.pt"
    resume: dict[str, Any] | None = None
    initialization: dict[str, Any] | None = None
    if latest.exists():
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        checkpoint_for_architecture = resume
    elif actor_init_checkpoint is not None:
        initialization = torch.load(
            actor_init_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_for_architecture = initialization
    else:
        # No Actor-BC initialization: start from a freshly constructed actor.
        # With residual output semantics the output head is zero-initialized,
        # so the actor reproduces the VLA reference exactly at step zero
        # (the README Stage-2 protocol: "actor 初始化为近似 VLA 透传").
        # Force the architecture mode to match the requested semantics: the
        # Stage 1 config defaults to actor_output_mode="direct", which would
        # skip the zero-init and break the passthrough guarantee.
        fresh_rlt_config = dict(stage1["rlt_config"])
        if action_steps is not None:
            # C override (chunk10 recipe on a chunk30 VLA): the fresh actor
            # is built at the shorter horizon and the VLA reference is
            # truncated to the first C steps upstream.
            fresh_rlt_config["n_action_steps_rl"] = int(action_steps)
        fresh_rlt_config["actor_output_mode"] = (
            "residual" if semantics == RESIDUAL_ACTOR_OUTPUT_SEMANTICS else "direct"
        )
        LOG.info(
            "fresh actor: forced actor_output_mode=%s (Stage 1 config said %s)",
            fresh_rlt_config["actor_output_mode"],
            stage1["rlt_config"].get("actor_output_mode"),
        )
        checkpoint_for_architecture = {
            "stage": "rlt_stage2_fresh_passthrough",
            "actor_output_semantics": semantics,
            "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
            "stage1_checkpoint": str(stage1_checkpoint),
            "rlt_config": fresh_rlt_config,
        }
        LOG.info(
            "no --actor-init-checkpoint: starting from fresh zero-residual "
            "(VLA passthrough) actor"
        )
    if checkpoint_for_architecture.get("actor_output_semantics") != semantics:
        raise ValueError(
            "actor output semantics mismatch: checkpoint uses "
            f"{checkpoint_for_architecture.get('actor_output_semantics')!r} but the "
            f"run requested {semantics!r}"
        )
    if checkpoint_for_architecture.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
        raise ValueError("Stage 2 checkpoint uses an incompatible action timebase")
    if str(Path(checkpoint_for_architecture["stage1_checkpoint"]).resolve()) != str(
        stage1_checkpoint
    ):
        raise ValueError("Stage 2 actor checkpoint belongs to another Stage 1 checkpoint")
    if action_steps is not None:
        if int(action_steps) <= 0 or int(action_steps) > int(stage1["rlt_config"]["n_action_steps_rl"]):
            raise ValueError(
                "--action-steps must be in (0, stage1 n_action_steps_rl]"
            )
        config_payload_action_steps = int(action_steps)
    else:
        config_payload_action_steps = int(stage1["rlt_config"]["n_action_steps_rl"])
    checkpoint_config = checkpoint_for_architecture.get("rlt_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Stage 2 actor checkpoint has no architecture config")
    config_payload = dict(checkpoint_config)
    stage1_config = dict(stage1["rlt_config"])
    for field in ("rlt_hidden_dim", "state_dim", "action_dim"):
        if config_payload.get(field) != stage1_config.get(field):
            raise ValueError(f"Stage 2 actor {field} differs from Stage 1")
    if config_payload.get("n_action_steps_rl") != config_payload_action_steps:
        raise ValueError(
            "Stage 2 actor n_action_steps_rl differs from the resolved "
            f"action_steps={config_payload_action_steps} "
            f"(checkpoint={config_payload.get('n_action_steps_rl')!r})"
        )
    config_payload["n_action_steps_rl"] = config_payload_action_steps
    config_payload["mode"] = "online_rl"
    config_payload["discount"] = float(discount)
    config_payload["critic_per_actor"] = int(critic_per_actor)
    config_payload["action_stride"] = ACTION_CHUNK_FRAME_STRIDE
    if actor_lr is not None:
        config_payload["actor_lr"] = float(actor_lr)
    if critic_lr is not None:
        config_payload["critic_lr"] = float(critic_lr)
    if ref_dropout is not None:
        config_payload["ref_dropout"] = float(ref_dropout)
    if actor_residual_scale is not None:
        if not 0.0 < float(actor_residual_scale) <= 1.0:
            raise ValueError("actor_residual_scale must be in (0, 1]")
        config_payload["actor_residual_scale"] = float(actor_residual_scale)
    config = SmolVLARLTConfig(**config_payload)
    arrays, replay_metadata = load_replay_dataset(
        replay_paths, stage1_checkpoint=stage1_checkpoint, discount=discount
    )
    actor_eligible = actor_training_mask(arrays, actor_start_step)
    if actor_frame_gate_strict:
        frames = arrays["episode_frame_index"].reshape(-1)
        actor_eligible &= (frames >= 0) & (frames >= int(actor_start_step))
    replay_root_metadata = _replay_root_for_metadata(replay_paths)
    invalid_count = int((arrays["td_valid"] == 0).sum())
    replay_integrity = {
        "provenance_version": "source_and_clock_v1",
        "critic_valid_transitions": int(arrays["td_valid"].sum()),
        "bc_only_transitions": invalid_count,
        "actor_eligible_transitions": int(actor_eligible.sum()),
        "valid_positive_reward_transitions": int(((arrays["td_valid"] > 0) & (arrays["rewards"] > 0)).sum()),
        "unknown_source_transitions": int((arrays["action_source"] == 3).sum()),
    }
    if invalid_count:
        LOG.warning(
            "%d/%d replay rows have unaligned action/next-observation timing; "
            "retaining their BC targets but excluding Critic and Actor Q gradients. "
            "Reconstruct aligned transitions before relying on TD learning from human completions.",
            invalid_count, len(arrays["z_rl"]),
        )
    for meta in replay_metadata:
        if meta.get("provenance_warning"):
            LOG.warning("%s: %s", meta["path"], meta["provenance_warning"])
    expected = (
        config.rlt_hidden_dim,
        config.state_dim,
        config.n_action_steps_rl,
        config.action_dim,
    )
    observed = (
        arrays["z_rl"].shape[1],
        arrays["state"].shape[1],
        arrays["actions"].shape[1],
        arrays["actions"].shape[2],
    )
    if observed != expected:
        raise ValueError(f"replay dimensions {observed} do not match RLT config {expected}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    target_device = torch.device(device)
    actor = RLActorMLP(config).to(target_device)
    critic = RLCriticMLP(config).to(target_device)
    critic_target = RLCriticMLP(config).to(target_device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    total_updates = 0
    trained_replay_names: set[str] = set()
    history: list[dict[str, float | int]] = []
    ready_for_online = False
    actor_action_scale_np = _derive_actor_action_scale(arrays)
    if resume is not None:
        actor.load_state_dict(resume["actor_state_dict"], strict=True)
        critic.load_state_dict(resume["critic_state_dict"], strict=True)
        critic_target.load_state_dict(resume["critic_target_state_dict"], strict=True)
        actor_optimizer.load_state_dict(resume["actor_optimizer_state_dict"])
        critic_optimizer.load_state_dict(resume["critic_optimizer_state_dict"])
        total_updates = int(resume.get("total_updates", 0))
        saved_names = resume.get("trained_replay_names")
        if saved_names is None:
            # v3 checkpoints used absolute paths.  Migrate by basename so a
            # replay store can be copied to its canonical dashboard location
            # without retraining every historical episode.
            saved_names = [
                Path(value).name for value in resume.get("trained_replay_paths", [])
            ]
        trained_replay_names = set(saved_names)
        history = list(resume.get("history", []))
        ready_for_online = bool(resume.get("ready_for_online", False))
        if actor_init_checkpoint is None and resume.get("actor_init_checkpoint"):
            actor_init_checkpoint = Path(resume["actor_init_checkpoint"]).resolve()
        actor_action_scale_np = np.asarray(
            resume.get("actor_action_scale", []), dtype=np.float32
        )
        if actor_action_scale_np.shape != (config.action_dim,):
            raise ValueError("existing Stage 2 checkpoint has incompatible action scale")
        LOG.info("resumed Stage 2 checkpoint updates=%d", total_updates)
    elif initialization is not None:
        if initialization.get("stage") not in {
            "rlt_actor_bc",
            "rlt_stage2_steam_offline",
        }:
            raise ValueError(
                "actor initialization must be an rlt_actor_bc or compatible "
                "rlt_stage2_steam_offline checkpoint"
            )
        actor.load_state_dict(initialization["actor_state_dict"], strict=True)
        actor_action_scale_np = np.asarray(
            initialization["actor_action_scale"], dtype=np.float32
        )
        if actor_action_scale_np.shape != (config.action_dim,):
            raise ValueError("actor initialization has incompatible action scale")
        LOG.info(
            "initialized actor from %s checkpoint %s",
            initialization.get("stage"),
            actor_init_checkpoint,
        )
    if actor_action_scale_override is not None:
        requested_scale = np.asarray(actor_action_scale_override, dtype=np.float32)
        if (
            requested_scale.shape != (config.action_dim,)
            or not np.isfinite(requested_scale).all()
            or np.any(requested_scale <= 0)
        ):
            raise ValueError(
                f"--actor-action-scale must contain {config.action_dim} positive finite values"
            )
        if resume is not None and not np.allclose(
            actor_action_scale_np, requested_scale, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(
                "cannot change --actor-action-scale when resuming an existing checkpoint"
            )
        actor_action_scale_np = requested_scale
        LOG.info("using fixed actor_action_scale=%s", requested_scale.tolist())
    _assert_reference_reachable(
        arrays, actor_action_scale_np, allow_clipped=allow_clipped_reference
    )
    current_names = replay_names(replay_paths)
    new_names = current_names - trained_replay_names
    new_episodes = len(new_names)
    new_transitions = sum(
        int(meta["transition_count"])
        for meta in replay_metadata
        if Path(meta["path"]).name in new_names
    )
    # Remote recipe: UTD (update-to-data) ratio maps total env steps onto
    # gradient updates.  Offline the buffer is the data source, so each
    # buffer transition plays the role of one collected transition and
    # requested_updates = utd_ratio * buffer_transitions reproduces the
    # remote update budget (remote: 5 updates x 10k post-warmup transitions).
    if utd_ratio is not None:
        if not math.isfinite(utd_ratio) or utd_ratio <= 0:
            raise ValueError("utd_ratio must be positive and finite")
        requested_updates = int(round(float(utd_ratio) * arrays["z_rl"].shape[0]))
    else:
        requested_updates = (
            int(updates) if updates is not None else int(updates_per_episode) * new_episodes
        )
    if requested_updates < 0:
        raise ValueError("updates must be non-negative")
    # Offline analogue of warm_up_steps: the buffer already holds the VLA
    # rollouts, so the critic warms up on them (actor frozen at passthrough)
    # for the warm_up/total fraction of the run.
    critic_only_updates = 0
    if warm_up_steps is not None or total_steps is not None:
        if warm_up_steps is None or total_steps is None:
            raise ValueError("--warm-up-steps and --total-steps must be set together")
        if not 0 <= warm_up_steps < total_steps:
            raise ValueError(
                "require 0 <= warm_up_steps < total_steps, got "
                f"{warm_up_steps} / {total_steps}"
            )
        critic_only_updates = int(round(requested_updates * warm_up_steps / total_steps))
    effective_batch = min(
        int(batch_size or config.rl_batch_size), int(arrays["z_rl"].shape[0])
    )
    actor_action_scale = torch.as_tensor(
        actor_action_scale_np, dtype=torch.float32, device=target_device
    ).reshape(1, 1, -1)
    if actor_exploration_std_per_dim_override is not None:
        if len(actor_exploration_std_per_dim_override) != int(config.action_dim):
            raise ValueError(
                "--actor-exploration-std-per-dim needs one value per action "
                f"dimension ({config.action_dim})"
            )
        actor_exploration_std_per_dim = torch.as_tensor(
            actor_exploration_std_per_dim_override,
            dtype=torch.float32,
            device=target_device,
        )
        LOG.info(
            "actor exploration per-dim override active: %s",
            list(actor_exploration_std_per_dim_override),
        )
    elif actor_exploration_std > 0 and config.action_dim == 7:
        policy_path = Path(stage1["smolvla_path"]).expanduser().resolve()
        action_physical_std_np = _physical_action_std(policy_path, config.action_dim)
        action_physical_std = torch.as_tensor(
            action_physical_std_np, dtype=torch.float32, device=target_device
        )
        actor_exploration_std_per_dim = calibrated_actor_exploration_std(
            actor_exploration_std,
            actor_action_scale,
            action_physical_std,
            # In direct mode the raw action maps through
            # scale*tanh(raw/scale) with unit slope around zero, so the
            # exploration gain is 1.0 instead of the residual_scale.
            residual_scale=(
                1.0
                if semantics == DIRECT_ACTOR_OUTPUT_SEMANTICS
                else config.actor_residual_scale
            ),
        )
        try:
            smolvla_config = json.loads((policy_path / "config.json").read_text())
        except (OSError, ValueError):
            smolvla_config = {}
        if bool(smolvla_config.get("use_relative_eef_actions", False)):
            actor_exploration_std_per_dim = apply_eef_relative_exploration_profile(
                actor_exploration_std_per_dim
            )
            LOG.info(
                "relative-EEF exploration profile applied "
                "(x, y, rx, ry, rz full; z halved; gripper deterministic)"
            )
    else:
        actor_exploration_std_per_dim = torch.full(
            (config.action_dim,),
            float(actor_exploration_std),
            dtype=torch.float32,
            device=target_device,
        )
    bc_action_weights_np = np.asarray(
        config.bc_action_weights or [1.0] * config.action_dim,
        dtype=np.float32,
    )
    bc_action_weights = torch.as_tensor(
        bc_action_weights_np, dtype=torch.float32, device=target_device
    )
    if effective_batch <= 0:
        raise ValueError("replay buffer is empty")
    outcome_success_transitions = int(arrays["outcome_success"].sum())
    outcome_failure_transitions = int(
        arrays["outcome_success"].shape[0] - outcome_success_transitions
    )
    if balanced_outcomes and (
        outcome_success_transitions == 0 or outcome_failure_transitions == 0
    ):
        raise ValueError(
            "balanced outcome sampling requires both success and failure replay"
        )
    il_anchor_cache: dict[str, np.ndarray] | None = None
    il_anchor_indices: np.ndarray | None = None
    if il_anchor_weight > 0:
        il_anchor_cache, il_anchor_indices = _load_il_anchor_cache(
            actor_validation_cache.expanduser().resolve(), seed=seed
        )
        LOG.info(
            "IL actor anchor samples=%d weight=%.3f",
            il_anchor_indices.size,
            il_anchor_weight,
        )

    LOG.info(
        "Stage 2 replay=%d transitions bundles=%d new=%d updates=%d batch=%d device=%s",
        arrays["z_rl"].shape[0],
        len(replay_paths),
        new_transitions,
        requested_updates,
        effective_batch,
        target_device,
    )
    if progress_file is not None:
        progress_file = progress_file.expanduser().resolve()
    run_started_ns = time.time_ns()
    run_started = time.monotonic()
    progress_base = {
        "stage": "rlt_stage2",
        "state": "running",
        "stage1_checkpoint": str(stage1_checkpoint),
        "actor_init_checkpoint": str(actor_init_checkpoint) if actor_init_checkpoint else None,
        "device": str(target_device),
        "started_at_ns": run_started_ns,
        "global_updates_before": total_updates,
        "total_updates": requested_updates,
        "new_transitions": new_transitions,
        "new_episodes": new_episodes,
        "updates_per_episode": int(updates_per_episode),
        "buffer_transitions": int(arrays["z_rl"].shape[0]),
        "replay_root": replay_root_metadata,
        "replay_integrity": replay_integrity,
        "batch_size": effective_batch,
        "training_phase": training_phase,
        "q_weight": effective_q_weight,
        "bc_weight": effective_bc_weight,
        "intervention_bc_weight": float(intervention_bc_weight),
        "replay_human_bc": bool(replay_human_bc),
        "ref_anchor_weight": float(ref_anchor_weight),
        "ref_anchor_mask": ref_anchor_mask,
        "ref_anchor_box": ref_anchor_box,
        "actor_exploration_std": float(actor_exploration_std),
        "actor_exploration_std_per_dim": (
            actor_exploration_std_per_dim.detach().cpu().tolist()
        ),
        "discount": float(config.discount),
        "critic_target_mode": critic_target_mode,
        "q_target_clip": q_target_clip,
        "grad_clip_norm": float(grad_clip_norm),
        "actor_q_gate": actor_q_gate,
        "critic_mc_weight": float(critic_mc_weight),
        "critic_per_actor": int(config.critic_per_actor),
        "actor_action_scale": actor_action_scale_np.tolist(),
        "bc_action_weights": bc_action_weights_np.tolist(),
        "balanced_outcomes": bool(balanced_outcomes),
        "il_anchor_weight": float(il_anchor_weight),
        "il_anchor_samples": int(il_anchor_indices.size) if il_anchor_indices is not None else 0,
        "freeze_actor": bool(freeze_actor),
        "success_transitions": outcome_success_transitions,
        "failure_transitions": outcome_failure_transitions,
        "actor_output_semantics": semantics,
        "utd_ratio": float(utd_ratio) if utd_ratio is not None else None,
        "warm_up_steps": int(warm_up_steps) if warm_up_steps is not None else None,
        "total_steps": int(total_steps) if total_steps is not None else None,
        "critic_only_updates": int(critic_only_updates),
        "replay_success_fraction": float(replay_success_fraction),
        "actor_start_step": int(actor_start_step) if actor_start_step is not None else None,
        "actor_frame_gate_strict": bool(actor_frame_gate_strict),
        "auto_intervention_chunk_rms": float(auto_intervention_chunk_rms),
        "auto_intervention_first_action_rms": float(auto_intervention_first_action_rms),
        "auto_intervention_max_abs": float(auto_intervention_max_abs),
        "n_action_steps_rl": int(config.n_action_steps_rl),
    }

    def publish_progress(completed: int, *, force: bool = False) -> None:
        nonlocal last_progress_write
        if progress_file is None:
            return
        now = time.monotonic()
        if not force and completed != 1 and now - last_progress_write < 0.5:
            return
        elapsed = max(0.0, now - run_started)
        rate = completed / elapsed if completed > 0 and elapsed > 0 else None
        remaining = max(0, requested_updates - completed)
        eta = remaining / rate if rate and rate > 0 else None
        percent = 100.0 if requested_updates == 0 else 100.0 * completed / requested_updates
        _atomic_json(
            {
                **progress_base,
                "updated_at_ns": time.time_ns(),
                "current_update": completed,
                "global_update": total_updates + completed,
                "percent": min(100.0, max(0.0, percent)),
                "elapsed_seconds": elapsed,
                "updates_per_second": rate,
                "eta_seconds": eta,
                "last_metrics": _finite_metrics(last_metrics),
            },
            progress_file,
        )
        last_progress_write = now

    generator = np.random.default_rng(seed + total_updates)
    if actor_start_step is not None:
        LOG.info(
            "actor_start_step=%d strict=%s: %d/%d Actor-eligible transitions",
            actor_start_step, actor_frame_gate_strict,
            int(actor_eligible.sum()), actor_eligible.size,
        )
    last_metrics: dict[str, float] = {}
    last_actor_metrics: dict[str, float] = {}
    actor_updates = 0
    last_progress_write = 0.0
    publish_progress(0, force=True)
    for update_index in range(requested_updates):
        if balanced_outcomes and replay_success_fraction <= 0.0:
            indices = _sample_balanced_episode_indices(
                generator,
                arrays["outcome_success"],
                arrays["replay_episode_index"],
                effective_batch,
            )
        elif replay_success_fraction > 0.0:
            # Remote replay_success_fraction: reserve the reserved rows from
            # successful transitions, fill the rest uniformly.
            indices = _sample_success_fraction_indices(
                generator,
                arrays["outcome_success"],
                arrays["z_rl"].shape[0],
                effective_batch,
                replay_success_fraction,
            )
        else:
            indices = generator.choice(
                arrays["z_rl"].shape[0], size=effective_batch, replace=False
            )
        batch = _batch(arrays, indices, target_device)
        with torch.no_grad():
            if critic_target_mode in {"td", "hybrid"}:
                next_action_raw, _ = actor(
                    batch["next_z_rl"], batch["next_state"], batch["next_ref_actions"]
                )
                next_action_raw = _sample_actor_action(
                    next_action_raw,
                    actor_exploration_std,
                    actor_exploration_std_per_dim,
                )
                next_actions = compose_actor_action(
                    next_action_raw,
                    batch["next_ref_actions"],
                    actor_action_scale,
                    output_semantics=semantics,
                    residual_scale=config.actor_residual_scale,
                )
                target_q = critic_target.q_min(
                    batch["next_z_rl"], batch["next_state"], next_actions
                )
                if q_target_clip > 0:
                    target_q = torch.clamp(target_q, 0.0, q_target_clip)
                td_target = batch["rewards"] + config.discount ** batch["discount_steps"] * (1.0 - batch["dones"]) * target_q
        q1, q2 = critic(batch["z_rl"], batch["state"], batch["actions"])
        critic_valid = batch["td_valid"].bool()
        mc_loss = _valid_critic_mse(q1, batch["mc_returns"], critic_valid) + _valid_critic_mse(q2, batch["mc_returns"], critic_valid)
        if critic_target_mode == "mc":
            td_loss = mc_loss.detach() * 0.0
            critic_loss = mc_loss
        else:
            td_loss = _valid_critic_mse(q1, td_target, critic_valid) + _valid_critic_mse(q2, td_target, critic_valid)
            critic_loss = (
                td_loss
                if critic_target_mode == "td"
                else (td_loss + float(critic_mc_weight) * mc_loss)
                / (1.0 + float(critic_mc_weight))
            )
        critic_optimizer.zero_grad(set_to_none=True)
        if critic_valid.any():
            critic_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), float(grad_clip_norm)
                )
            critic_optimizer.step()

        global_update = total_updates + update_index + 1
        # Warm-up analogue of the remote warm_up_steps: during the first
        # warm_up_steps/total_steps fraction of the run the critic learns on
        # the pre-filled buffer while the actor stays at VLA passthrough.
        actor_step_allowed = (
            not freeze_actor
            and update_index >= critic_only_updates
            and global_update % int(config.critic_per_actor) == 0
        )
        if actor_step_allowed and actor_start_step is not None:
            start_rows = np.flatnonzero(actor_eligible[indices])
            if start_rows.size == 0:
                actor_step_allowed = False
            elif start_rows.size < batch["z_rl"].shape[0]:
                sel = torch.as_tensor(start_rows, device=target_device)
                batch = {k: v.index_select(0, sel) for k, v in batch.items()}
        if actor_step_allowed:
            action_raw, _ = actor(
                batch["z_rl"], batch["state"], batch["ref_actions"], training=True
            )
            action_raw = _sample_actor_action(
                action_raw,
                actor_exploration_std,
                actor_exploration_std_per_dim,
            )
            sampled_action = compose_actor_action(
                action_raw,
                batch["ref_actions"],
                actor_action_scale,
                output_semantics=semantics,
                residual_scale=config.actor_residual_scale,
            )
            # Offline mirror of the remote auto-intervention monitor: with
            # the remote thresholds, log the fraction of batch rows on which
            # the composed action would trip the trust-region handover
            # (chunk_rms / first_action_rms / max_abs, normalized units).
            with torch.no_grad():
                ai_dev = (sampled_action - batch["ref_actions"]).abs()
                ai_chunk_rms = ai_dev.pow(2).mean(dim=(1, 2)).sqrt()
                ai_first_rms = ai_dev[:, 0, :].pow(2).mean(dim=1).sqrt()
                ai_max_abs = ai_dev.amax(dim=(1, 2))
                last_actor_metrics["auto_intervention_would_rate"] = (
                    (
                        (ai_chunk_rms > auto_intervention_chunk_rms)
                        | (ai_first_rms > auto_intervention_first_action_rms)
                        | (ai_max_abs > auto_intervention_max_abs)
                    )
                    .float()
                    .mean()
                    .item()
                )
                last_actor_metrics["auto_intervention_chunk_rms_p95"] = (
                    ai_chunk_rms.quantile(0.95).item()
                )
            q_value = critic.q_min(batch["z_rl"].detach(), batch["state"], sampled_action)
            # v4: restrict the Q-gradient to synthetic offset rows.  The
            # critic's action landscape is uncalibrated on the real (clean)
            # rows; applying the Q-gradient there is what drifts the actor
            # away from the reference on clean frames.  On synthetic rows the
            # corr/bad reward contrast teaches the corrective pullback.
            syn_mask = batch.get("is_synthetic")
            q_mask = batch["td_valid"].reshape(-1).bool()
            if actor_q_gate == "synthetic" and syn_mask is not None:
                q_mask = q_mask & syn_mask.reshape(-1).bool()
            q_loss = -q_value.reshape(-1)[q_mask].mean() if q_mask.any() else q_value.sum() * 0.0
            bc_target, bc_supervised = _actor_bc_target_and_mask(
                batch,
                replay_human_bc=replay_human_bc,
                target_mode=bc_target_mode,
            )
            # v6 channel separation: synthetic rows are supervised by the
            # Q-gradient alone; BC (cloning toward ref/human) applies to
            # real rows only.
            if syn_bc_mask and "is_synthetic" in batch:
                bc_supervised = bc_supervised & ~batch["is_synthetic"].reshape(-1).bool()
            bc_sample_weights = None
            if abs(float(intervention_bc_weight) - 1.0) > 1e-12:
                bc_sample_weights = torch.ones_like(bc_supervised, dtype=torch.float32)
                intervention_mask = (
                    bc_supervised
                    & batch["is_intervention"].bool().squeeze(-1)
                    & bool(replay_human_bc)
                )
                bc_sample_weights[intervention_mask] = float(intervention_bc_weight)
            reachable_target = (
                _reachable_bc_target(
                    bc_target,
                    batch["ref_actions"],
                    actor_action_scale,
                    output_semantics=semantics,
                    residual_scale=config.actor_residual_scale,
                )
                if project_bc_targets
                else bc_target
            )
            replay_bc_loss = _masked_action_mse(
                sampled_action,
                reachable_target,
                bc_supervised,
                bc_action_weights,
                sample_weights=bc_sample_weights,
            )
            # Keep the discarded, unreachable residual as a diagnostic.
            with torch.no_grad():
                bc_floor = _masked_action_mse(
                    reachable_target,
                    bc_target,
                    bc_supervised,
                    bc_action_weights,
                    sample_weights=bc_sample_weights,
                )
                reachable_changed = (
                    (reachable_target - bc_target).abs().amax(dim=-1) > 1e-7
                )
                bc_target_clipped_fraction = float(
                    reachable_changed.float().mean().cpu()
                )
            il_anchor_loss = sampled_action.sum() * 0.0
            if il_anchor_cache is not None and il_anchor_indices is not None:
                anchor_rows = generator.choice(
                    il_anchor_indices,
                    size=min(effective_batch, il_anchor_indices.size),
                    replace=False,
                )
                anchor_z = torch.as_tensor(
                    il_anchor_cache["z_rl"][anchor_rows], device=target_device
                )
                anchor_state = torch.as_tensor(
                    il_anchor_cache["state"][anchor_rows], device=target_device
                )
                anchor_reference = torch.as_tensor(
                    il_anchor_cache["reference"][anchor_rows], device=target_device
                )
                anchor_target = torch.as_tensor(
                    il_anchor_cache["target"][anchor_rows], device=target_device
                )
                anchor_valid = torch.as_tensor(
                    il_anchor_cache["valid"][anchor_rows],
                    dtype=torch.bool,
                    device=target_device,
                )
                anchor_raw, _ = actor(
                    anchor_z, anchor_state, anchor_reference, training=True
                )
                anchor_prediction = compose_actor_action(
                    anchor_raw,
                    anchor_reference,
                    actor_action_scale,
                    output_semantics=semantics,
                    residual_scale=config.actor_residual_scale,
                )
                il_anchor_loss = _masked_timestep_action_mse(
                    anchor_prediction,
                    anchor_target,
                    anchor_valid,
                    bc_action_weights,
                )
                bc_loss = (
                    replay_bc_loss + float(il_anchor_weight) * il_anchor_loss
                ) / (1.0 + float(il_anchor_weight))
            else:
                bc_loss = replay_bc_loss
            ref_anchor_loss = sampled_action.sum() * 0.0
            ref_anchor_fraction = 1.0
            if ref_anchor_weight > 0:
                anchor_valid = bc_supervised.unsqueeze(1).expand_as(
                    sampled_action[:, :, 0]
                )
                if ref_anchor_box_np is not None:
                    # Pin the output to the VLA prior only OUTSIDE the hole
                    # gate box; inside the box the actor stays free to learn
                    # the correction.  States are stored normalized with the
                    # dataset 'action' stats (state and action share the EEF
                    # pose space), so denormalize with the same stats.
                    state_phys = batch["state"] * torch.as_tensor(
                        action_stats_std,
                        dtype=batch["state"].dtype,
                        device=batch["state"].device,
                    ) + torch.as_tensor(
                        action_stats_mean,
                        dtype=batch["state"].dtype,
                        device=batch["state"].device,
                    )
                    in_box = (
                        (state_phys[:, 0] >= bx0)
                        & (state_phys[:, 0] <= bx1)
                        & (state_phys[:, 1] >= by0)
                        & (state_phys[:, 1] <= by1)
                        & (state_phys[:, 2] >= bz0)
                        & (state_phys[:, 2] <= bz1)
                    )
                    anchor_valid = anchor_valid * (
                        ~in_box
                    ).to(anchor_valid.dtype).unsqueeze(1)
                ref_anchor_fraction = float(anchor_valid.float().mean())
                ref_anchor_loss = _masked_timestep_action_mse(
                    sampled_action,
                    batch["ref_actions"],
                    anchor_valid,
                    bc_action_weights,
                )
            actor_loss = (
                effective_q_weight * q_loss
                + effective_bc_weight * bc_loss
                + float(ref_anchor_weight) * ref_anchor_loss
            )
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), float(grad_clip_norm)
                )
            actor_optimizer.step()
            actor_updates += 1
            last_actor_metrics = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "q_loss": float(q_loss.detach().cpu()),
                "q_supervised_fraction": float(q_mask.float().mean().detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "replay_bc_loss": float(replay_bc_loss.detach().cpu()),
                "replay_bc_reachability_floor": float(bc_floor.cpu()),
                "replay_bc_target_clipped_fraction": (
                    bc_target_clipped_fraction if project_bc_targets else 0.0
                ),
                "bc_target_projection_enabled": bool(project_bc_targets),
                "il_anchor_loss": float(il_anchor_loss.detach().cpu()),
                "bc_supervised_fraction": float(
                    bc_supervised.float().mean().detach().cpu()
                ),
                "ref_anchor_loss": float(ref_anchor_loss.detach().cpu()),
                "ref_anchor_fraction": ref_anchor_fraction,
            }
            if actor_updates <= 3:
                LOG.info(
                    "actor-start update=%d actor_updates=%d %s",
                    global_update, actor_updates, last_actor_metrics,
                )

        tau = float(config.target_tau)
        for parameter, target in zip(critic.parameters(), critic_target.parameters()):
            target.data.mul_(1.0 - tau).add_(parameter.data, alpha=tau)
        last_metrics = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_td_loss": float(td_loss.detach().cpu()),
            "critic_mc_loss": float(mc_loss.detach().cpu()),
            "critic_valid_fraction": float(critic_valid.float().mean().detach().cpu()),
            "q1_mean": float(q1.mean().detach().cpu()),
            "q2_mean": float(q2.mean().detach().cpu()),
            **last_actor_metrics,
        }
        _assert_stable_metrics(
            last_metrics, global_update, bc_weight_active=effective_bc_weight > 0.0
        )
        if global_update == 1 or global_update % log_freq == 0:
            LOG.info("update=%d metrics=%s", global_update, last_metrics)
        publish_progress(update_index + 1)

    actor_validation = _actor_deviation_stats(
        actor, arrays, target_device, actor_action_scale,
        output_semantics=semantics,
    )
    actor_warnings: list[str] = []
    if not all(math.isfinite(value) for value in actor_validation.values()):
        raise RuntimeError(f"actor validation is non-finite: {actor_validation}")
    if actor_validation["mean_abs_error_to_reference"] > 0.5:
        raise RuntimeError(
            "Stage 2 actor is too far from the reference policy: "
            f"{actor_validation}"
        )
    validation_cache = actor_validation_cache
    if validation_cache is None:
        candidates = [output_dir / "actor_bc_features.npz"]
        if actor_init_checkpoint is not None:
            candidates.insert(0, actor_init_checkpoint.parent / "actor_bc_features.npz")
        validation_cache = next((path for path in candidates if path.is_file()), None)
    heldout_validation: dict[str, float | str] | None = None
    if validation_cache is not None:
        validation_cache = validation_cache.expanduser().resolve()
        initialization: dict[str, Any] = {}
        baseline_per_dim_mse = None
        if actor_init_checkpoint is None:
            # A fresh residual Actor has a zero-initialized correction head, so
            # it is an exact reference-VLA passthrough.  This is a valid and
            # common RLT initialization; evaluate that immutable baseline on
            # the same held-out BC cache instead of incorrectly requiring an
            # unrelated Actor-BC checkpoint.
            if semantics != RESIDUAL_ACTOR_OUTPUT_SEMANTICS:
                raise ValueError(
                    "held-out validation without Actor-BC initialization "
                    "requires residual actor semantics"
                )
            baseline_actor = actor.__class__(config).to(target_device)
            baseline_actor.eval().requires_grad_(False)
            baseline_result = _heldout_actor_gt_bc_loss(
                baseline_actor,
                validation_cache,
                device=target_device,
                action_scale=actor_action_scale,
                action_weights=bc_action_weights,
                output_semantics=semantics,
            )
            if isinstance(baseline_result, tuple):
                baseline, baseline_per_dim_mse = baseline_result
            else:
                baseline = float(baseline_result)
            baseline_checkpoint = "fresh_zero_residual_reference"
        else:
            initialization = torch.load(
                actor_init_checkpoint, map_location="cpu", weights_only=False
            )
            baseline = float(initialization.get("best_val_loss", float("nan")))
            baseline_checkpoint = actor_init_checkpoint
            if not math.isfinite(baseline):
                # Offline STEAM actor checkpoints keep the original IL-BC
                # initialization path and validation metric under ``steam``.
                # Use that immutable BC baseline for the same no-regression gate.
                steam_metrics = initialization.get("steam", {}).get("metrics", {})
                baseline = float(
                    steam_metrics.get("baseline_il", {}).get("target_mse", float("nan"))
                )
                initialization_path = initialization.get("initialization")
                if initialization_path:
                    candidate = Path(initialization_path).expanduser()
                    if candidate.is_file():
                        baseline_checkpoint = candidate.resolve()
                        original_initialization = torch.load(
                            baseline_checkpoint, map_location="cpu", weights_only=False
                        )
                        baseline = float(
                            original_initialization.get("best_val_loss", baseline)
                        )
                        if initialization.get("best_val_per_dim_mse") is None:
                            initialization["best_val_per_dim_mse"] = (
                                original_initialization.get("best_val_per_dim_mse")
                            )
            baseline_per_dim_mse = initialization.get("best_val_per_dim_mse")
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                "actor initialization has no finite IL-BC validation baseline"
            )
        heldout_result = _heldout_actor_gt_bc_loss(
            actor,
            validation_cache,
            device=target_device,
            action_scale=actor_action_scale,
            action_weights=bc_action_weights,
            output_semantics=semantics,
        )
        if isinstance(heldout_result, tuple):
            heldout_loss, heldout_per_dim_mse = heldout_result
        else:  # Backward-compatible with scalar validation hooks/checkpoints.
            heldout_loss = float(heldout_result)
            heldout_per_dim_mse = []
        ratio = heldout_loss / baseline
        per_dim_ratios = None
        regressed_dims: list[int] = []
        if baseline_per_dim_mse is not None and heldout_per_dim_mse:
            if len(baseline_per_dim_mse) != len(heldout_per_dim_mse):
                raise ValueError("Actor-BC per-dimension validation shape is incompatible")
            per_dim_ratios = [
                current / max(float(seed_value), 1e-12)
                for current, seed_value in zip(
                    heldout_per_dim_mse, baseline_per_dim_mse, strict=True
                )
            ]
            regressed_dims = [
                index
                for index, dim_ratio in enumerate(per_dim_ratios)
                if dim_ratio > max_heldout_bc_ratio
            ]
        heldout_validation = {
            "cache": str(validation_cache),
            "baseline_checkpoint": str(baseline_checkpoint),
            "gt_bc_loss": heldout_loss,
            "actor_bc_baseline": baseline,
            "ratio_to_actor_bc": ratio,
            "per_dim_mse": heldout_per_dim_mse,
            "actor_bc_per_dim_mse": baseline_per_dim_mse,
            "per_dim_ratio_to_actor_bc": per_dim_ratios,
            "regressed_dims": regressed_dims,
            "max_ratio": float(max_heldout_bc_ratio),
        }
        heldout_validation["diagnostic_only"] = True

    total_updates += requested_updates
    # RLT does not require a distinct robot warm-up collection phase.  A
    # successfully validated actor-critic update over the available replay is
    # sufficient to publish the actor for subsequent online episodes.
    # Offline one-step MSE is retained as an experiment diagnostic only. It
    # cannot decide whether a 10-step receding-horizon policy is safe or
    # successful on the robot, so it must not gate Actor publication.
    ready_for_online = total_updates > 0 and not freeze_actor
    history.append(
        {
            "time_ns": time.time_ns(),
            "updates": requested_updates,
            "total_updates": total_updates,
            "new_transitions": new_transitions,
            "new_episodes": new_episodes,
            "updates_per_episode": int(updates_per_episode),
            "buffer_transitions": int(arrays["z_rl"].shape[0]),
            "replay_root": replay_root_metadata,
            "training_phase": training_phase,
            "q_weight": effective_q_weight,
            "bc_weight": effective_bc_weight,
            "critic_target_mode": critic_target_mode,
            "q_target_clip": q_target_clip,
            "grad_clip_norm": float(grad_clip_norm),
            "actor_q_gate": actor_q_gate,
            "critic_mc_weight": float(critic_mc_weight),
            **last_metrics,
        }
    )
    checkpoint = {
        "format_version": STAGE2_FORMAT,
        "stage": "rlt_stage2",
                "actor_output_semantics": semantics,
                "bc_target_mode": bc_target_mode,
        "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
        "stage1_checkpoint": str(stage1_checkpoint),
        "actor_init_checkpoint": str(actor_init_checkpoint) if actor_init_checkpoint else None,
        "actor_validation_cache": (
            str(actor_validation_cache.expanduser().resolve())
            if actor_validation_cache is not None
            else None
        ),
        "smolvla_path": stage1["smolvla_path"],
        "rlt_config": config_payload,
        "encoder_state_dict": stage1["encoder_state_dict"],
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "critic_target_state_dict": critic_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "total_updates": total_updates,
        "actor_updates_this_run": actor_updates,
        "buffer_transitions": int(arrays["z_rl"].shape[0]),
        "replay_root": replay_root_metadata,
        "trained_replay_names": sorted(current_names),
        "replay_integrity": replay_integrity,
        "history": history[-200:],
        "last_metrics": last_metrics,
        "saved_at_ns": time.time_ns(),
        "actor_exploration_std": float(actor_exploration_std),
        "actor_exploration_std_per_dim": (
            actor_exploration_std_per_dim.detach().cpu().tolist()
        ),
        "actor_action_scale": actor_action_scale_np.tolist(),
        "bc_action_weights": bc_action_weights_np.tolist(),
        "actor_validation": actor_validation,
        "heldout_validation": heldout_validation,
        "actor_warnings": actor_warnings,
        "training_phase": training_phase,
        "ready_for_online": ready_for_online,
        "q_weight": effective_q_weight,
        "bc_weight": effective_bc_weight,
        "intervention_bc_weight": float(intervention_bc_weight),
        "ref_anchor_weight": float(ref_anchor_weight),
        "ref_anchor_mask": ref_anchor_mask,
        "ref_anchor_box": ref_anchor_box,
        "critic_target_mode": critic_target_mode,
        "q_target_clip": q_target_clip,
        "grad_clip_norm": float(grad_clip_norm),
        "actor_q_gate": actor_q_gate,
        "critic_mc_weight": float(critic_mc_weight),
        "balanced_outcomes": bool(balanced_outcomes),
        "il_anchor_weight": float(il_anchor_weight),
        "ref_anchor_weight": float(ref_anchor_weight),
        "ref_anchor_mask": ref_anchor_mask,
        "ref_anchor_box": ref_anchor_box,
        "freeze_actor": bool(freeze_actor),
        "sampling_strategy": (
            "success_fraction_uniform"
            if replay_success_fraction > 0.0
            else ("balanced_outcome_uniform_episode" if balanced_outcomes else "uniform_transition")
        ),
        "success_transitions": outcome_success_transitions,
        "failure_transitions": outcome_failure_transitions,
        "utd_ratio": float(utd_ratio) if utd_ratio is not None else None,
        "warm_up_steps": int(warm_up_steps) if warm_up_steps is not None else None,
        "total_steps": int(total_steps) if total_steps is not None else None,
        "critic_only_updates": int(critic_only_updates),
        "replay_success_fraction": float(replay_success_fraction),
        "actor_start_step": int(actor_start_step) if actor_start_step is not None else None,
        "actor_frame_gate_strict": bool(actor_frame_gate_strict),
        "auto_intervention_chunk_rms": float(auto_intervention_chunk_rms),
        "auto_intervention_first_action_rms": float(auto_intervention_first_action_rms),
        "auto_intervention_max_abs": float(auto_intervention_max_abs),
        "n_action_steps_rl": int(config.n_action_steps_rl),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    versioned = output_dir / f"checkpoint_{total_updates:08d}.pt"
    _atomic_torch_save(checkpoint, versioned)
    _atomic_torch_save(checkpoint, latest)
    summary = {
        "stage": "rlt_stage2",
        "replay_integrity": replay_integrity,
        "actor_output_semantics": semantics,
        "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
        "checkpoint": str(versioned),
        "latest": str(latest),
        "stage1_checkpoint": str(stage1_checkpoint),
        "actor_init_checkpoint": str(actor_init_checkpoint) if actor_init_checkpoint else None,
        "buffer_transitions": checkpoint["buffer_transitions"],
        "replay_root": replay_root_metadata,
        "new_transitions": new_transitions,
        "new_episodes": new_episodes,
        "updates_per_episode": int(updates_per_episode),
        "updates_this_run": requested_updates,
        "total_updates": total_updates,
        "actor_updates_this_run": actor_updates,
        "last_metrics": last_metrics,
        "actor_validation": actor_validation,
        "heldout_validation": heldout_validation,
        "actor_warnings": actor_warnings,
        "training_phase": training_phase,
        "ready_for_online": ready_for_online,
        "actor_exploration_std": float(actor_exploration_std),
        "actor_exploration_std_per_dim": (
            actor_exploration_std_per_dim.detach().cpu().tolist()
        ),
        "actor_action_scale": actor_action_scale_np.tolist(),
        "bc_action_weights": bc_action_weights_np.tolist(),
        "critic_target_mode": critic_target_mode,
        "q_target_clip": q_target_clip,
        "grad_clip_norm": float(grad_clip_norm),
        "actor_q_gate": actor_q_gate,
        "critic_mc_weight": float(critic_mc_weight),
        "balanced_outcomes": bool(balanced_outcomes),
        "il_anchor_weight": float(il_anchor_weight),
        "ref_anchor_weight": float(ref_anchor_weight),
        "ref_anchor_mask": ref_anchor_mask,
        "ref_anchor_box": ref_anchor_box,
        "freeze_actor": bool(freeze_actor),
        "success_transitions": outcome_success_transitions,
        "failure_transitions": outcome_failure_transitions,
        "utd_ratio": float(utd_ratio) if utd_ratio is not None else None,
        "warm_up_steps": int(warm_up_steps) if warm_up_steps is not None else None,
        "total_steps": int(total_steps) if total_steps is not None else None,
        "critic_only_updates": int(critic_only_updates),
        "replay_success_fraction": float(replay_success_fraction),
        "actor_start_step": int(actor_start_step) if actor_start_step is not None else None,
        "actor_frame_gate_strict": bool(actor_frame_gate_strict),
        "auto_intervention_chunk_rms": float(auto_intervention_chunk_rms),
        "auto_intervention_first_action_rms": float(auto_intervention_first_action_rms),
        "auto_intervention_max_abs": float(auto_intervention_max_abs),
        "n_action_steps_rl": int(config.n_action_steps_rl),
    }
    _atomic_json(summary, output_dir / "status.json")
    if progress_file is not None:
        elapsed = max(0.0, time.monotonic() - run_started)
        rate = requested_updates / elapsed if requested_updates > 0 and elapsed > 0 else None
        _atomic_json(
            {
                **progress_base,
                "state": "completed",
                "updated_at_ns": time.time_ns(),
                "finished_at_ns": time.time_ns(),
                "current_update": requested_updates,
                "global_update": total_updates,
                "percent": 100.0,
                "elapsed_seconds": elapsed,
                "updates_per_second": rate,
                "eta_seconds": 0.0,
                "last_metrics": _finite_metrics(last_metrics),
                "actor_validation": actor_validation,
                "actor_warnings": actor_warnings,
                "checkpoint": str(versioned),
                "latest": str(latest),
            },
            progress_file,
        )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = discover_replay_bundles(args.replay_root.expanduser().resolve())
    if not paths:
        raise RuntimeError(f"no replay bundles under {args.replay_root}")
    try:
        summary = train_stage2(
            stage1_checkpoint=args.stage1_checkpoint,
            actor_init_checkpoint=args.actor_init_checkpoint,
            replay_paths=paths,
            output_dir=args.output_dir,
            device=args.device,
            updates=args.updates,
            batch_size=args.batch_size,
            seed=args.seed,
            log_freq=args.log_freq,
            progress_file=args.progress_file,
            training_phase=args.training_phase,
            actor_exploration_std=args.actor_exploration_std,
            actor_exploration_std_per_dim_override=args.actor_exploration_std_per_dim,
            actor_action_scale_override=args.actor_action_scale,
            allow_clipped_reference=args.allow_clipped_reference,
            discount=args.discount,
            critic_target_mode=args.critic_target_mode,
            critic_mc_weight=args.critic_mc_weight,
            q_target_clip=args.q_target_clip,
            grad_clip_norm=args.grad_clip_norm,
            actor_q_gate=args.actor_q_gate,
            critic_per_actor=args.critic_per_actor,
            balanced_outcomes=args.balanced_outcomes,
            updates_per_episode=args.updates_per_episode,
            actor_validation_cache=args.actor_validation_cache,
            max_heldout_bc_ratio=args.max_heldout_bc_ratio,
            il_anchor_weight=args.il_anchor_weight,
            q_weight=args.q_weight,
            bc_weight=args.bc_weight,
            intervention_bc_weight=args.intervention_bc_weight,
            replay_human_bc=args.replay_human_bc,
            syn_bc_mask=args.syn_bc_mask,
            ref_anchor_weight=args.ref_anchor_weight,
            ref_anchor_mask=args.ref_anchor_mask,
            ref_anchor_box=args.ref_anchor_box,
            action_stats_path=args.action_stats,
            freeze_actor=args.freeze_actor,
            actor_output_semantics=args.actor_output_semantics,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            ref_dropout=args.ref_dropout,
            actor_residual_scale=args.actor_residual_scale,
            project_bc_targets=not args.no_bc_target_projection,
            action_steps=args.action_steps,
            utd_ratio=args.utd_ratio,
            warm_up_steps=args.warm_up_steps,
            total_steps=args.total_steps,
            replay_success_fraction=args.replay_success_fraction,
            actor_start_step=args.actor_start_step,
            actor_frame_gate_strict=args.actor_frame_gate_strict,
            auto_intervention_chunk_rms=args.auto_intervention_chunk_rms,
            auto_intervention_first_action_rms=args.auto_intervention_first_action_rms,
            auto_intervention_max_abs=args.auto_intervention_max_abs,
        )
    except BaseException as exc:
        if args.progress_file is not None:
            _write_failed_progress(args.progress_file.expanduser().resolve(), exc)
        raise
    LOG.info("Stage 2 training complete: %s", summary)


if __name__ == "__main__":
    main()
