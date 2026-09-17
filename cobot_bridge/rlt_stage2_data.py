"""Persistent decision logs and replay bundles for Cobot Magic RLT Stage 2.

The live raw recorder stores camera/state evidence at the 30 Hz policy clock.
This module stores the complementary per-inference tensors needed by the
actor/critic: normalized state, z_rl, and complete reference/policy/executed
action chunks.  Keeping the two streams separate avoids reconstructing a
proposed action chunk from the single action eventually sent to the robot.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


DECISION_FORMAT = "cobot_magic_rlt_decisions_v1"
REPLAY_FORMAT = "cobot_magic_rlt_replay_v1"
ACTION_SOURCE_NAMES = {"reference": 0, "actor": 1, "human": 2}
OUTCOMES = {"failure": 0, "success": 1, "aborted": 2}
COLLECTION_PHASES = ("warmup", "actor", "reference")
DEFAULT_WARMUP_EPISODES = 5
DEFAULT_WARMUP_POST_COLLECT_UPDATES = 30_000
DEFAULT_ACTOR_EXPLORATION_STD = 0.002
# Desired physical joint-noise ratios for Piper.  J6 is the anchor so the UI
# exploration scalar retains its historical raw-space meaning at J6 while the
# other arm dimensions are compensated for normalization/action-scale gains.
PIPER_ARM_EXPLORATION_PHYSICAL_PROFILE = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ACTION_CHUNK_TIMEBASE = "contiguous_30hz_v1"
ACTION_CHUNK_FRAME_STRIDE = 1
DIRECT_ACTOR_OUTPUT_SEMANTICS = "direct_bounded_normalized_action_contiguous_30hz_v3"
RESIDUAL_ACTOR_OUTPUT_SEMANTICS = "residual_bounded_normalized_action_contiguous_30hz_v4"
ACTOR_OUTPUT_SEMANTICS = RESIDUAL_ACTOR_OUTPUT_SEMANTICS
STEAM_OFFLINE_ACTOR_STAGE = "rlt_stage2_steam_offline"
SUPPORTED_ACTOR_CHECKPOINT_STAGES = frozenset({"rlt_stage2", STEAM_OFFLINE_ACTOR_STAGE})
SUPPORTED_ACTOR_OUTPUT_SEMANTICS = frozenset(
    {DIRECT_ACTOR_OUTPUT_SEMANTICS, RESIDUAL_ACTOR_OUTPUT_SEMANTICS}
)


def contiguous_action_chunk(
    actions: torch.Tensor, *, action_steps: int, action_dim: int
) -> torch.Tensor:
    """Select consecutive policy ticks without compressing the time axis."""

    if actions.ndim != 3:
        raise ValueError(f"actions must be rank 3, got {tuple(actions.shape)}")
    if action_steps <= 0 or action_dim <= 0:
        raise ValueError("action_steps and action_dim must be positive")
    if actions.shape[1] < action_steps or actions.shape[2] < action_dim:
        raise ValueError(
            f"action tensor {tuple(actions.shape)} cannot provide "
            f"{action_steps}x{action_dim} contiguous actions"
        )
    return actions[:, :action_steps, :action_dim]


def squash_actor_action(raw_action: torch.Tensor, action_scale: torch.Tensor) -> torch.Tensor:
    """Bound a direct actor action while preserving mean/std-normalized units.

    RLinf's RLT actor applies tanh because its action space is normalized to
    [-1, 1].  SmolVLA here uses mean/std normalization and valid values exceed
    that interval, so use a per-joint symmetric scale.  Dividing before tanh
    preserves unit slope around zero and multiplying restores the original
    normalized units.
    """

    scale = torch.as_tensor(action_scale, dtype=raw_action.dtype, device=raw_action.device)
    if scale.ndim == 1:
        scale = scale.reshape(1, 1, -1)
    if scale.shape[-1] != raw_action.shape[-1] or not torch.isfinite(scale).all():
        raise ValueError("actor action scale is invalid")
    if not bool(torch.all(scale > 0)):
        raise ValueError("actor action scale must be positive")
    return scale * torch.tanh(raw_action / scale)


def compose_actor_action(
    raw_action: torch.Tensor,
    reference_action: torch.Tensor,
    action_scale: torch.Tensor,
    *,
    output_semantics: str = ACTOR_OUTPUT_SEMANTICS,
    residual_scale: float = 0.25,
) -> torch.Tensor:
    """Convert an actor network output into an absolute normalized chunk.

    Legacy actors predict a directly bounded absolute action. Residual actors
    predict a bounded correction around the frozen VLA reference. Exploration
    noise belongs on ``raw_action`` before this function is called.
    """

    if output_semantics == DIRECT_ACTOR_OUTPUT_SEMANTICS:
        return squash_actor_action(raw_action, action_scale)
    if output_semantics != RESIDUAL_ACTOR_OUTPUT_SEMANTICS:
        raise ValueError(f"unsupported actor output semantics: {output_semantics!r}")
    if raw_action.shape != reference_action.shape:
        raise ValueError(
            f"raw/reference action shapes differ: {tuple(raw_action.shape)} != "
            f"{tuple(reference_action.shape)}"
        )
    if not 0 < residual_scale <= 1:
        raise ValueError("residual_scale must be in (0, 1]")
    scale = torch.as_tensor(action_scale, dtype=raw_action.dtype, device=raw_action.device)
    if scale.ndim == 1:
        scale = scale.reshape(1, 1, -1)
    if scale.shape[-1] != raw_action.shape[-1] or not torch.isfinite(scale).all():
        raise ValueError("actor action scale is invalid")
    if not bool(torch.all(scale > 0)):
        raise ValueError("actor action scale must be positive")
    residual_limit = scale * float(residual_scale)
    residual = residual_limit * torch.tanh(raw_action)
    return torch.clamp(reference_action + residual, min=-scale, max=scale)


def calibrated_actor_exploration_std(
    base_std: float,
    action_scale: torch.Tensor,
    action_physical_std: torch.Tensor,
    *,
    residual_scale: float,
) -> torch.Tensor:
    """Return raw Actor noise whose physical Piper arm noise grows J1 -> J6.

    Around the residual Actor's zero correction, raw noise is converted to a
    physical joint perturbation by ``residual_scale * action_scale *
    action_physical_std``.  Compensating that gain is necessary: a monotonic
    raw-space vector does not imply monotonic physical joint exploration.

    Non-Piper action spaces retain scalar exploration for compatibility.
    The gripper keeps ``base_std`` in raw space and is intentionally not mixed
    with the six rotational-joint profile.
    """

    if not np.isfinite(base_std) or base_std < 0:
        raise ValueError("base exploration std must be finite and non-negative")
    scale = torch.as_tensor(action_scale).flatten()
    physical_std = torch.as_tensor(
        action_physical_std, dtype=scale.dtype, device=scale.device
    ).flatten()
    if scale.shape != physical_std.shape or not torch.isfinite(scale).all():
        raise ValueError("exploration action scale/statistics are incompatible")
    if not torch.isfinite(physical_std).all() or not bool(torch.all(physical_std > 0)):
        raise ValueError("exploration physical action std must be finite and positive")
    if not 0 < residual_scale <= 1:
        raise ValueError("residual_scale must be in (0, 1]")
    if scale.numel() != 7:
        return torch.full_like(scale, float(base_std))
    if not bool(torch.all(scale > 0)):
        raise ValueError("exploration action scale must be positive")

    physical_gain = scale * physical_std * float(residual_scale)
    j6_gain = physical_gain[5]
    profile = torch.as_tensor(
        PIPER_ARM_EXPLORATION_PHYSICAL_PROFILE,
        dtype=scale.dtype,
        device=scale.device,
    )
    result = torch.empty_like(scale)
    result[:6] = float(base_std) * profile * j6_gain / physical_gain[:6]
    result[6] = float(base_std)
    return result


# Anchor-relative EEF chains act in [x, y, z, rx, ry, rz, gripper] space, so
# the Piper joint-space J1->J6 profile above does not describe their physical
# exploration.  Shape the calibrated per-dim noise in physical terms instead:
# full lateral/rotational exploration for fixture alignment, halved press-down
# (z) to avoid hammering the fixture, and a deterministic gripper whose release
# timing must not be randomized by exploration noise.
EEF_RELATIVE_EXPLORATION_PROFILE = (1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0)


def apply_eef_relative_exploration_profile(std_per_dim: torch.Tensor) -> torch.Tensor:
    """Shape per-dim Actor exploration for anchor-relative EEF action spaces.

    ``std_per_dim`` is the raw-space noise calibrated to equal physical
    exploration across dimensions; multiplying the EEF profile turns that
    uniform physical exploration into the task-safe anisotropy described by
    ``EEF_RELATIVE_EXPLORATION_PROFILE`` (order: x, y, z, rx, ry, rz, gripper).
    """

    vec = torch.as_tensor(
        EEF_RELATIVE_EXPLORATION_PROFILE,
        dtype=std_per_dim.dtype,
        device=std_per_dim.device,
    )
    if std_per_dim.shape[-1] != vec.numel():
        raise ValueError(
            "eef exploration profile requires a 7-dim action space, got "
            f"{tuple(std_per_dim.shape)}"
        )
    return std_per_dim * vec


@dataclass(frozen=True)
class RLTDimensions:
    z_dim: int
    state_dim: int
    action_steps: int
    action_dim: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _append(dataset: h5py.Dataset, value: Any) -> None:
    index = dataset.shape[0]
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class RLTDecisionWriter:
    """Incrementally write one Stage 2 decision stream.

    All state/action tensors are in the frozen SmolVLA processor's normalized
    space.  Raw joint commands belong in the parallel raw HDF5 episode.
    """

    def __init__(
        self,
        path: Path,
        *,
        dimensions: RLTDimensions,
        stage1_checkpoint: str,
        task: str,
        workflow_mode: int,
        side: str = "right",
        collection_phase: str = "actor",
        flush_decisions: int = 4,
        sent_action_steps: int | None = None,
    ) -> None:
        if workflow_mode not in (1, 2):
            raise ValueError(f"workflow_mode must be 1 or 2, got {workflow_mode}")
        if side not in ("left", "right"):
            raise ValueError(f"invalid side: {side}")
        if collection_phase not in COLLECTION_PHASES:
            raise ValueError(f"invalid collection_phase: {collection_phase!r}")
        if collection_phase in ("warmup", "reference") and workflow_mode != 1:
            raise ValueError(f"{collection_phase} collection requires Mode 1")
        if flush_decisions <= 0:
            raise ValueError("flush_decisions must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.workflow_mode = workflow_mode
        self.flush_decisions = flush_decisions
        self.count = 0
        self.intervention_count = 0
        if sent_action_steps is not None and int(sent_action_steps) <= 0:
            raise ValueError("sent_action_steps must be positive")
        self.sent_action_steps = (
            None if sent_action_steps is None else int(sent_action_steps)
        )
        self.file = h5py.File(self.path, "x", libver="latest")
        self.file.attrs.update(
            {
                "format": DECISION_FORMAT,
                "completed": False,
                "created_at_ns": time.time_ns(),
                "stage1_checkpoint": str(Path(stage1_checkpoint).expanduser().resolve()),
                "task": task,
                "workflow_mode": workflow_mode,
                "collection_phase": collection_phase,
                "side": side,
                "action_space": "smolvla_normalized",
                "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
                "action_chunk_frame_stride": ACTION_CHUNK_FRAME_STRIDE,
                "z_dim": dimensions.z_dim,
                "state_dim": dimensions.state_dim,
                "action_steps": dimensions.action_steps,
                "action_dim": dimensions.action_dim,
                "action_source_names": json.dumps(ACTION_SOURCE_NAMES, sort_keys=True),
                "outcome": "unknown",
                "sent_action_steps": self.sent_action_steps or 0,
                "sent_action_space": "robot_raw_joint_position",
            }
        )
        d = dimensions
        self.datasets = {
            "time_ns": self.file.create_dataset(
                "time_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "bridge_sequence": self.file.create_dataset(
                "bridge_sequence", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "z_rl": self.file.create_dataset(
                "z_rl", shape=(0, d.z_dim), maxshape=(None, d.z_dim), dtype="f4", chunks=True
            ),
            "state": self.file.create_dataset(
                "state",
                shape=(0, d.state_dim),
                maxshape=(None, d.state_dim),
                dtype="f4",
                chunks=True,
            ),
            "reference_actions": self.file.create_dataset(
                "reference_actions",
                shape=(0, d.action_steps, d.action_dim),
                maxshape=(None, d.action_steps, d.action_dim),
                dtype="f4",
                chunks=True,
            ),
            "policy_actions": self.file.create_dataset(
                "policy_actions",
                shape=(0, d.action_steps, d.action_dim),
                maxshape=(None, d.action_steps, d.action_dim),
                dtype="f4",
                chunks=True,
            ),
            "executed_actions": self.file.create_dataset(
                "executed_actions",
                shape=(0, d.action_steps, d.action_dim),
                maxshape=(None, d.action_steps, d.action_dim),
                dtype="f4",
                chunks=True,
            ),
            "action_source": self.file.create_dataset(
                "action_source", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "is_intervention": self.file.create_dataset(
                "is_intervention", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
        }
        if self.sent_action_steps is not None:
            self.datasets["sent_actions_raw"] = self.file.create_dataset(
                "sent_actions_raw",
                shape=(0, self.sent_action_steps, d.action_dim),
                maxshape=(None, self.sent_action_steps, d.action_dim),
                dtype="f4",
                chunks=True,
            )

    def append(
        self,
        *,
        z_rl: Any,
        state: Any,
        reference_actions: Any,
        policy_actions: Any,
        executed_actions: Any,
        sent_actions_raw: Any | None = None,
        action_source: str,
        is_intervention: bool,
        bridge_sequence: int = -1,
        time_ns: int | None = None,
    ) -> None:
        if action_source not in ACTION_SOURCE_NAMES:
            raise ValueError(f"invalid action_source: {action_source!r}")
        if bool(is_intervention) != (action_source == "human"):
            raise ValueError("is_intervention must match action_source='human'")
        if self.workflow_mode == 1 and is_intervention:
            raise ValueError("Mode 1 cannot store human intervention decisions")
        d = self.dimensions
        values = {
            "time_ns": time.time_ns() if time_ns is None else int(time_ns),
            "bridge_sequence": int(bridge_sequence),
            "z_rl": _finite_array(z_rl, (d.z_dim,), "z_rl"),
            "state": _finite_array(state, (d.state_dim,), "state"),
            "reference_actions": _finite_array(
                reference_actions, (d.action_steps, d.action_dim), "reference_actions"
            ),
            "policy_actions": _finite_array(
                policy_actions, (d.action_steps, d.action_dim), "policy_actions"
            ),
            "executed_actions": _finite_array(
                executed_actions, (d.action_steps, d.action_dim), "executed_actions"
            ),
            "action_source": ACTION_SOURCE_NAMES[action_source],
            "is_intervention": int(is_intervention),
        }
        if self.sent_action_steps is not None:
            if sent_actions_raw is None:
                raise ValueError("sent_actions_raw is required for this decision stream")
            values["sent_actions_raw"] = _finite_array(
                sent_actions_raw,
                (self.sent_action_steps, d.action_dim),
                "sent_actions_raw",
            )
        elif sent_actions_raw is not None:
            raise ValueError("writer was not configured for sent_actions_raw")
        for name, value in values.items():
            _append(self.datasets[name], value)
        self.count += 1
        self.intervention_count += int(is_intervention)
        if self.count % self.flush_decisions == 0:
            self.file.flush()

    def close(self, *, outcome: str, completed: bool = True) -> None:
        if not self.file:
            return
        if outcome not in OUTCOMES:
            raise ValueError(f"invalid outcome: {outcome!r}")
        if self.workflow_mode == 1 and self.intervention_count:
            raise ValueError("Mode 1 episode contains intervention decisions")
        self.file.attrs.update(
            {
                "completed": bool(completed),
                "closed_at_ns": time.time_ns(),
                "num_decisions": self.count,
                "intervention_decisions": self.intervention_count,
                "outcome": outcome,
                "outcome_id": OUTCOMES[outcome],
            }
        )
        self.file.flush()
        self.file.close()

    def abort(self) -> None:
        if self.file:
            self.file.close()
        self.path.unlink(missing_ok=True)


def inspect_decision_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        return {name: value.item() if hasattr(value, "item") else value for name, value in source.attrs.items()}


def _nearest_timestamp_indices(
    timestamps_ns: np.ndarray, targets_ns: np.ndarray
) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    targets = np.asarray(targets_ns, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size == 0 or np.any(np.diff(timestamps) < 0):
        raise ValueError("raw timestamps must be a non-empty monotonic vector")
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, timestamps.size - 1)
    left = np.maximum(0, right - 1)
    use_left = np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets)
    return np.where(use_left, left, right).astype(np.int64)


def repair_mode2_human_action_chunks(
    decision_path: Path,
    raw_path: Path,
    *,
    normalize_action_chunk,
    raw_action_to_model_chunk=None,
    action_fps: float,
    max_alignment_error_ms: float = 25.0,
) -> dict[str, Any]:
    """Replace online Teach placeholders with the recorded future command path.

    A human decision is conditioned on the same observation as its reference
    policy prediction, but its BC target must be the command trajectory that
    was subsequently executed at the policy clock.  The raw recorder is the
    authoritative 30 Hz source.  Near Teach release or episode termination,
    the final valid human command is held to complete the fixed-size chunk.
    """

    if not np.isfinite(action_fps) or action_fps <= 0:
        raise ValueError("action_fps must be finite and positive")
    decision_path = Path(decision_path).expanduser().resolve()
    raw_path = Path(raw_path).expanduser().resolve()
    with h5py.File(raw_path, "r") as raw, h5py.File(decision_path, "r+") as decisions:
        if not bool(raw.attrs.get("completed", False)):
            raise ValueError("raw episode is not finalized")
        if not bool(decisions.attrs.get("completed", False)):
            raise ValueError("decision episode is not finalized")
        if int(decisions.attrs.get("workflow_mode", 0)) != 2:
            raise ValueError("human chunk repair requires a Mode 2 decision episode")
        if str(raw.attrs.get("side", "")) != str(decisions.attrs.get("side", "")):
            raise ValueError("raw and decision side do not match")

        source_names = json.loads(str(decisions.attrs["action_source_names"]))
        human_source = int(source_names["human"])
        human_rows = np.flatnonzero(decisions["action_source"][:] == human_source)
        intervention_rows = np.flatnonzero(decisions["is_intervention"][:].astype(bool))
        if not np.array_equal(human_rows, intervention_rows):
            raise ValueError("human action_source and intervention labels disagree")
        if human_rows.size == 0:
            return {
                "human_decisions": 0,
                "repaired_decisions": 0,
                "padded_action_samples": 0,
                "action_fps": float(action_fps),
            }

        timestamps = raw["frames/pc_timestamp_ns"][:].astype(np.int64)
        raw_human = raw["frames/is_human_intervention"][:].astype(bool)
        executed_valid = raw["frames/executed_action_valid"][:].astype(bool)
        executed = raw["frames/executed_action"][:].astype(np.float32)
        if executed.shape != (timestamps.size, int(decisions.attrs["action_dim"])):
            raise ValueError(f"invalid raw executed_action shape: {executed.shape}")
        if not np.isfinite(executed[executed_valid]).all():
            raise ValueError("raw executed actions contain non-finite values")

        decision_times = decisions["time_ns"][human_rows].astype(np.int64)
        alignment_source = "pc_timestamp_ns"
        sequence_error: np.ndarray | None = None
        if "sequence" in raw["frames"] and "bridge_sequence" in decisions:
            raw_sequence = raw["frames/sequence"][:].astype(np.int64)
            decision_sequence = decisions["bridge_sequence"][human_rows].astype(np.int64)
            if (
                raw_sequence.size == timestamps.size
                and np.all(np.diff(raw_sequence) >= 0)
                and np.all(decision_sequence >= 0)
            ):
                starts = _nearest_timestamp_indices(raw_sequence, decision_sequence)
                sequence_error = np.abs(raw_sequence[starts] - decision_sequence)
                alignment_source = "bridge_sequence"
            else:
                starts = _nearest_timestamp_indices(timestamps, decision_times)
        else:
            starts = _nearest_timestamp_indices(timestamps, decision_times)
        alignment_error_ms = np.abs(timestamps[starts] - decision_times) / 1e6
        if alignment_source == "pc_timestamp_ns" and float(alignment_error_ms.max()) > float(
            max_alignment_error_ms
        ):
            raise ValueError(
                "decision/raw alignment exceeds limit: "
                f"{alignment_error_ms.max():.3f} ms > {max_alignment_error_ms:.3f} ms"
            )
        if not raw_human[starts].all():
            bad = human_rows[~raw_human[starts]].tolist()
            raise ValueError(f"human decisions do not align to raw Teach frames: {bad}")

        action_steps = int(decisions.attrs["action_steps"])
        offsets_ns = (
            np.arange(action_steps, dtype=np.float64) * 1e9 / float(action_fps)
        ).round().astype(np.int64)
        normalized_chunks: list[np.ndarray] = []
        padded_samples = 0
        temporal_deltas: list[float] = []
        for row, start in zip(human_rows, starts, strict=True):
            segment_end = int(start)
            while segment_end + 1 < raw_human.size and raw_human[segment_end + 1]:
                segment_end += 1
            indices = _nearest_timestamp_indices(timestamps, timestamps[start] + offsets_ns)
            before_clip = indices.copy()
            indices = np.clip(indices, int(start), segment_end)
            padded_samples += int(np.count_nonzero(indices != before_clip))
            if not raw_human[indices].all() or not executed_valid[indices].all():
                raise ValueError(f"human chunk row {int(row)} contains invalid raw commands")
            raw_chunk = executed[indices]
            model_chunk = (
                raw_chunk
                if raw_action_to_model_chunk is None
                else np.asarray(
                    raw_action_to_model_chunk(raw_chunk, raw["frames/puppet_position"][start]),
                    dtype=np.float32,
                )
            )
            normalized = np.asarray(normalize_action_chunk(model_chunk), dtype=np.float32)
            expected = (action_steps, int(decisions.attrs["action_dim"]))
            if normalized.shape != expected or not np.isfinite(normalized).all():
                raise ValueError(
                    f"normalized human chunk row {int(row)} has shape {normalized.shape}"
                )
            normalized_chunks.append(normalized)
            temporal_deltas.append(
                float(np.max(np.abs(np.diff(raw_chunk, axis=0))))
                if action_steps > 1
                else 0.0
            )

        decisions["executed_actions"][human_rows] = np.stack(normalized_chunks)
        decisions.attrs.update(
            {
                "human_chunk_source": "raw_executed_action_future",
                "human_chunk_action_fps": float(action_fps),
                "human_chunk_repaired_at_ns": time.time_ns(),
                "human_chunk_padded_action_samples": int(padded_samples),
            }
        )
        decisions.flush()
        return {
            "human_decisions": int(human_rows.size),
            "repaired_decisions": int(human_rows.size),
            "padded_action_samples": int(padded_samples),
                "action_fps": float(action_fps),
                "alignment_source": alignment_source,
                "median_alignment_error_ms": float(np.median(alignment_error_ms)),
                "max_alignment_error_ms": float(alignment_error_ms.max()),
                "median_sequence_error": (
                    float(np.median(sequence_error)) if sequence_error is not None else None
                ),
                "max_sequence_error": (
                    int(sequence_error.max()) if sequence_error is not None else None
                ),
                "mean_within_chunk_max_delta_raw": float(np.mean(temporal_deltas)),
            "constant_human_chunks": int(np.count_nonzero(np.asarray(temporal_deltas) == 0)),
        }


def build_replay_bundle(
    decision_path: Path,
    output_path: Path,
    *,
    trim_stationary: bool = True,
    stationary_state_delta: float = 0.002,
    edge_padding_decisions: int = 1,
    recap_values: np.ndarray | None = None,
    recap_discount: float = 0.96,
    terminal_success_reward: float = 1.0,
    terminal_failure_reward: float = -1.0,
) -> dict[str, Any]:
    """Convert a finalized decision stream into one immutable replay bundle.

    ``recap_values`` is an optional fixed state-value trace aligned one-to-one
    with the *untrimmed* decision rows.  When supplied, reward uses potential
    shaping, ``gamma * V(next_obs) - V(obs)``, plus the terminal outcome
    reward.  RECAP remains a state-value teacher: it is never used as an
    action-conditioned critic or as an actor target.
    """

    decision_path = Path(decision_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    with h5py.File(decision_path, "r") as source:
        if source.attrs.get("format") != DECISION_FORMAT:
            raise ValueError(f"unsupported decision format: {source.attrs.get('format')!r}")
        if not bool(source.attrs.get("completed", False)):
            raise ValueError("decision episode is not finalized")
        if source.attrs.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
            raise ValueError("decision episode uses an incompatible action timebase")
        outcome = str(source.attrs.get("outcome", "unknown"))
        if outcome not in ("success", "failure"):
            raise ValueError(f"replay requires success/failure outcome, got {outcome!r}")
        count = int(source["z_rl"].shape[0])
        if count < 2:
            raise ValueError(f"at least two decisions are required, got {count}")
        recap_trace: np.ndarray | None = None
        if recap_values is not None:
            recap_trace = np.asarray(recap_values, dtype=np.float32).reshape(-1)
            if recap_trace.shape != (count,) or not np.isfinite(recap_trace).all():
                raise ValueError(
                    "recap_values must be a finite value for each untrimmed "
                    f"decision: expected {(count,)}, got {recap_trace.shape}"
                )
            if not 0.0 < float(recap_discount) <= 1.0:
                raise ValueError("recap_discount must be in (0, 1]")
            if not np.isfinite(terminal_success_reward) or not np.isfinite(terminal_failure_reward):
                raise ValueError("terminal RECAP rewards must be finite")
        start, end = 0, count
        if trim_stationary:
            if stationary_state_delta <= 0 or edge_padding_decisions < 0:
                raise ValueError("stationary trim parameters are invalid")
            states = source["state"][:].astype(np.float32)
            deltas = np.max(np.abs(np.diff(states, axis=0)), axis=1)
            moving = np.flatnonzero(deltas > float(stationary_state_delta))
            if moving.size:
                start = max(0, int(moving[0]) - edge_padding_decisions)
                end = min(count, int(moving[-1]) + 2 + edge_padding_decisions)
        transition_count = end - start - 1
        if transition_count < 1:
            raise ValueError("stationary trimming left no replay transitions")
        workflow_mode = int(source.attrs["workflow_mode"])
        collection_phase = str(source.attrs.get("collection_phase", "actor"))
        if collection_phase == "reference":
            raise ValueError(
                "reference diagnostic episodes cannot be converted to Stage 2 replay"
            )
        is_intervention = source["is_intervention"][start : end - 1].astype(np.float32)
        if workflow_mode == 1 and bool(is_intervention.any()):
            raise ValueError("Mode 1 replay cannot contain intervention transitions")
        source_ids = source["action_source"][start:end].astype(np.uint8)
        source_counts = {
            name: int(np.count_nonzero(source_ids == source_id))
            for name, source_id in ACTION_SOURCE_NAMES.items()
        }
        if collection_phase == "warmup" and (
            workflow_mode != 1
            or source_counts["actor"] != 0
            or source_counts["human"] != 0
            or source_counts["reference"] != int(source_ids.shape[0])
        ):
            raise ValueError("warmup replay must contain only Mode 1 reference decisions")
        rewards = np.zeros((transition_count, 1), dtype=np.float32)
        dones = np.zeros((transition_count, 1), dtype=np.float32)
        if recap_trace is not None:
            # Potential-based shaping retains the terminal objective while
            # supplying every executed ten-tick transition with visual
            # progress.  ``start:end`` are the source decision indices of the
            # selected replay, so adjacent entries are exactly obs_t/obs_next.
            rewards[:, 0] = (
                float(recap_discount) * recap_trace[start + 1 : end]
                - recap_trace[start : end - 1]
            )
            rewards[-1, 0] += (
                float(terminal_success_reward)
                if outcome == "success"
                else float(terminal_failure_reward)
            )
        else:
            rewards[-1, 0] = float(outcome == "success")
        dones[-1, 0] = 1.0
        arrays = {
            "z_rl": source["z_rl"][start : end - 1].astype(np.float32),
            "state": source["state"][start : end - 1].astype(np.float32),
            "actions": source["executed_actions"][start : end - 1].astype(np.float32),
            "ref_actions": source["reference_actions"][start : end - 1].astype(np.float32),
            "rewards": rewards,
            "next_z_rl": source["z_rl"][start + 1 : end].astype(np.float32),
            "next_state": source["state"][start + 1 : end].astype(np.float32),
            "next_ref_actions": source["reference_actions"][start + 1 : end].astype(np.float32),
            "dones": dones,
            "is_intervention": is_intervention[:, None],
            "is_demo": np.zeros((transition_count, 1), dtype=np.float32),
        }
        metadata = {
            "format": REPLAY_FORMAT,
            "created_at_ns": time.time_ns(),
            "decision_path": str(decision_path),
            "stage1_checkpoint": str(source.attrs["stage1_checkpoint"]),
            "task": str(source.attrs["task"]),
            "workflow_mode": workflow_mode,
            "collection_phase": collection_phase,
            "outcome": outcome,
            "transition_count": transition_count,
            "intervention_transitions": int(is_intervention.sum()),
            "action_source_counts": source_counts,
            "source_decisions": count,
            "selected_decision_start": start,
            "selected_decision_end": end,
            "trimmed_prefix_decisions": start,
            "trimmed_suffix_decisions": count - end,
            "stationary_trim_enabled": bool(trim_stationary),
            "stationary_state_delta": float(stationary_state_delta),
            "action_space": str(source.attrs["action_space"]),
            "action_chunk_timebase": str(source.attrs.get("action_chunk_timebase", "")),
            "action_chunk_frame_stride": int(
                source.attrs.get("action_chunk_frame_stride", 0)
            ),
            "z_dim": int(source.attrs["z_dim"]),
            "state_dim": int(source.attrs["state_dim"]),
            "action_steps": int(source.attrs["action_steps"]),
            "action_dim": int(source.attrs["action_dim"]),
            "reward_source": (
                "recap_potential_v1" if recap_trace is not None else "terminal_binary_v1"
            ),
            "recap_discount": float(recap_discount) if recap_trace is not None else None,
            "terminal_success_reward": (
                float(terminal_success_reward) if recap_trace is not None else 1.0
            ),
            "terminal_failure_reward": (
                float(terminal_failure_reward) if recap_trace is not None else 0.0
            ),
        }
        if recap_trace is not None:
            metadata["recap_value_current_min"] = float(recap_trace[start : end - 1].min())
            metadata["recap_value_current_max"] = float(recap_trace[start : end - 1].max())
            metadata["recap_shaping_reward_mean"] = float(rewards[:, 0].mean())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(output_path.with_suffix(output_path.suffix + ".json"), metadata)
    return metadata


def load_replay_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"].item()))
        if metadata.get("format") != REPLAY_FORMAT:
            raise ValueError(f"unsupported replay format in {path}")
        arrays = {
            name: np.asarray(bundle[name], dtype=np.float32)
            for name in (
                "z_rl",
                "state",
                "actions",
                "ref_actions",
                "rewards",
                "next_z_rl",
                "next_state",
                "next_ref_actions",
                "dones",
                "is_intervention",
            )
        }
        # Replay v1 predates IL demonstration replay.  Preserve compatibility
        # with already collected robot rollouts while making the supervision
        # source explicit for newly generated bundles.
        if "is_demo" in bundle.files:
            arrays["is_demo"] = np.asarray(bundle["is_demo"], dtype=np.float32)
        else:
            arrays["is_demo"] = np.zeros_like(arrays["is_intervention"])
        # v4 augmentation: rows with fabricated offset states.  When present
        # the actor's Q-gradient is restricted to these rows.
        if "is_synthetic" in bundle.files:
            arrays["is_synthetic"] = np.asarray(bundle["is_synthetic"], dtype=np.float32)
        else:
            arrays["is_synthetic"] = np.zeros_like(arrays["is_intervention"])
    count = int(metadata["transition_count"])
    if any(array.shape[0] != count for array in arrays.values()):
        raise ValueError(f"inconsistent replay array lengths in {path}")
    return arrays, metadata


def load_replay_metadata(path: Path) -> dict[str, Any]:
    """Read only the small metadata member without inflating replay tensors."""

    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"].item()))
    if metadata.get("format") != REPLAY_FORMAT:
        raise ValueError(f"unsupported replay format in {path}")
    return metadata


def inspect_warmup_replays(
    replay_root: Path,
    *,
    stage1_checkpoint: Path,
    target: int = DEFAULT_WARMUP_EPISODES,
) -> dict[str, Any]:
    """Count only verified warm-up bundles belonging to one exact Stage 1 artifact."""

    replay_root = Path(replay_root).expanduser().resolve()
    expected_stage1 = str(Path(stage1_checkpoint).expanduser().resolve())
    valid: list[str] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(replay_root.glob("*.npz")) if replay_root.exists() else []:
        try:
            metadata = load_replay_metadata(path)
            counts = dict(metadata.get("action_source_counts") or {})
            reasons = []
            if str(Path(metadata.get("stage1_checkpoint", "")).expanduser().resolve()) != expected_stage1:
                reasons.append("stage1_mismatch")
            if metadata.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
                reasons.append("action_timebase_mismatch")
            if metadata.get("collection_phase") != "warmup":
                reasons.append("not_warmup")
            if int(metadata.get("workflow_mode", 0)) != 1:
                reasons.append("not_mode1")
            if int(metadata.get("intervention_transitions", 0)) != 0:
                reasons.append("intervention")
            if int(counts.get("actor", -1)) != 0 or int(counts.get("human", -1)) != 0:
                reasons.append("non_reference_action")
            if int(counts.get("reference", 0)) != int(metadata.get("source_decisions", -1)):
                reasons.append("incomplete_reference_actions")
            if reasons:
                rejected.append({"path": str(path), "reason": ",".join(reasons)})
            else:
                valid.append(str(path))
        except Exception as exc:
            rejected.append({"path": str(path), "reason": f"invalid:{exc}"})
    return {
        "count": len(valid),
        "target": int(target),
        "complete": len(valid) >= int(target),
        "valid_replays": valid,
        "rejected": rejected,
        "stage1_checkpoint": expected_stage1,
        "replay_root": str(replay_root),
    }
