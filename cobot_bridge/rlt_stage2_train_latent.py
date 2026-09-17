"""Train the RLT-Latent Stage 2 actor (iv50 recipe, latent-residual route).

This is the NEW test route, deliberately separate from the deployed
action-residual route (``rlt_stage2_train.py``).  Architecture:

    z_rl (frozen RLT encoder) -> latent actor (z, s, ref) -> r_lat (B, C, 7)
    r_lat --W (zero-init passthrough, xavier)--> suffix offset (B, C, H)
    frozen SmolVLA: c'_t = c_t + W r_lat inside the 10-step flow denoise
    decoded 30-step chunk -> first 10 steps = a_exec (normalized relative EEF)

Faithful iv50 mirrors:
  * critic: UNCHANGED, Q(z, s, a_exec_replay) on the replay's executed
    actions, MC target mode, q_target_clip 1.0, discount 0.96.
  * actor: ALL loss terms in DECODE space through the frozen VLA
    (prefix KV under no_grad; gradients only through the 10 suffix denoise
    steps), mirroring the iv50 composition:
      q      : -Q(z, s, a_exec)                       (weight 0.01)
      bc     : target = where(is_intervention|is_demo, actions, ref_actions);
               intervention rows get sample weight 50     (weight 0.1)
      ref    : mse(a_exec, ref) on outside-box rows      (weight 5)
      il     : mse(r_lat, 0) on held-out IL anchor rows  (weight 30,
               latent-space adaptation: no images exist for the 50k IL
               cache, so the anchor keeps the decode at the VLA prior)
  * balanced outcomes, 256 updates / 64 actor updates (critic_per_actor 4).

Row -> observation mapping: bundle row i of bundle b = decision row
``selected_decision_start + i`` of the bundle's decisions h5; the decision's
``time_ns`` indexes the raw archive's ``frames/pc_timestamp_ns`` exactly
(verified: decoded z_rl and state match the bundle to 0.0 max-diff).
The model state is the physical gripper-center EEF state
(joints -> EEF via PiperForwardKinematics, as in the reconstruction
pipeline); images are the decision's start frame.

Noise: fixed per buffer row (hash) so the same state always decodes with the
same flow noise across updates (deterministic training signal).
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
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .rlt_latent import (
    LATENT_ACTOR_OUTPUT_SEMANTICS,
    LatentRuntime,
)
from .rlt_policy_server import _decode_jpeg
from .rlt_stage2_data import ACTION_CHUNK_FRAME_STRIDE, ACTION_CHUNK_TIMEBASE
from .rlt_stage2_train import (
    DEFAULT_CRITIC_PER_ACTOR,
    DEFAULT_STAGE2_DISCOUNT,
    _atomic_json,
    _atomic_torch_save,
    _derive_actor_action_scale,
    _finite_metrics,
    _load_il_anchor_cache,
    _sample_balanced_episode_indices,
    discover_replay_bundles,
    load_replay_dataset,
    replay_names,
)

LOG = logging.getLogger("cobot_bridge.rlt_stage2_train_latent")
STAGE2_LATENT_FORMAT = 3
DEFAULT_MAX_HELDOUT_RATIO = 1.05
DEFAULT_ACTOR_DECODE_ROWS = 256
DEFAULT_NOISE_SEED_MIXIN = 2654435761


def _noise_seed_for_row(row: int) -> int:
    """Deterministic per-row flow noise seed."""
    return int((int(row) * DEFAULT_NOISE_SEED_MIXIN) % (1 << 31))


# ─────────────────────────────────────────────────────────────────────────────
# Row -> raw-frame mapping
# ─────────────────────────────────────────────────────────────────────────────
class RowFrameMap:
    """Maps every buffer row to (raw_path, decision_start_frame).

    Built once at startup: for each bundle, decision rows
    ``selected_decision_start .. selected_decision_start + n - 1`` resolve to
    raw frames through the decisions h5 ``time_ns`` and the raw archive's
    monotonic ``frames/pc_timestamp_ns``.
    """

    #: Warn / abort thresholds for |decision_ts - nearest raw_ts|.
    #: A 30 Hz frame period is ~33 ms: beyond one period the nearest frame is
    #: ambiguous, beyond two periods the mapping is broken.  Observed real
    #: data has clock-rate drift (30 Hz grid vs 30.7 Hz raw) producing deltas
    #: up to ~17 ms, plus isolated dropped-frame gaps (~138 ms in one bundle).
    WARN_DELTA_NS = 34_000_000
    #: Beyond ~9 frame periods the mapping is clearly broken (not just a
    #: dropped frame); observed worst case is one 138 ms gap in one bundle.
    ABORT_DELTA_NS = 300_000_000

    def __init__(
        self,
        replay_paths: list[Path],
        replay_metadata: list[dict[str, Any]],
    ) -> None:
        total = sum(int(meta["transition_count"]) for meta in replay_metadata)
        self.raw_path_by_row = np.empty(total, dtype=object)
        self.frame_by_row = np.empty(total, dtype=np.int64)
        self.task_by_row = np.empty(total, dtype=object)
        self.n_warn = 0
        offset = 0
        n_fallback = 0
        for meta in replay_metadata:
            with np.load(meta["path"], allow_pickle=False) as bundle:
                bundle_meta = json.loads(str(bundle["metadata"].item()))
                n_rows = int(bundle_meta["transition_count"])
            dec_path = Path(bundle_meta["decision_path"])
            with h5py.File(dec_path, "r") as dec:
                raw_path = str(dec.attrs["raw_episode_path"])
                time_ns = np.asarray(dec["time_ns"], dtype=np.int64)
                task = str(dec.attrs["task"])
            with h5py.File(raw_path, "r") as raw:
                ts = np.asarray(raw["frames/pc_timestamp_ns"], dtype=np.int64)
            start = int(bundle_meta["selected_decision_start"])
            if start + n_rows > time_ns.shape[0]:
                raise ValueError(
                    f"{dec_path.name}: bundle needs {start + n_rows} decisions "
                    f"but only {time_ns.shape[0]} recorded"
                )
            dec_time = time_ns[start : start + n_rows]
            # A decision ts may sit a few ms before/after a raw ts (clock
            # rounding); a 30 Hz period is ~33 ms, so the NEAREST raw frame
            # is unambiguous within 10 ms. Compare both neighbours.
            right = np.clip(
                np.searchsorted(ts, dec_time, side="left"), 0, ts.shape[0] - 1
            )
            left = np.clip(right - 1, 0, ts.shape[0] - 1)
            d_right = np.abs(ts[right] - dec_time)
            d_left = np.abs(ts[left] - dec_time)
            frames = np.where(d_left < d_right, left, right)
            delta = np.minimum(d_left, d_right)
            warn = delta > self.WARN_DELTA_NS
            abort = delta > self.ABORT_DELTA_NS
            if bool(abort.any()):
                raise ValueError(
                    f"{dec_path.name}: {int(abort.sum())}/{n_rows} decision timestamps "
                    f"are farther than {self.ABORT_DELTA_NS / 1e6:.0f} ms from "
                    "the nearest raw frame (mapping is broken)"
                )
            if bool(warn.any()):
                LOG.warning(
                    "%s: %d/%d rows farther than %.0f ms from the nearest raw "
                    "frame (max %.1f ms); using the nearest frame",
                    dec_path.name, int(warn.sum()), n_rows,
                    self.WARN_DELTA_NS / 1e6, float(delta.max()) / 1e6,
                )
                self.n_warn += int(warn.sum())
            n_fallback += int((delta > 0).sum())
            self.raw_path_by_row[offset : offset + n_rows] = raw_path
            self.frame_by_row[offset : offset + n_rows] = frames
            self.task_by_row[offset : offset + n_rows] = task
            offset += n_rows
        if offset != total:
            raise RuntimeError("row-frame map size mismatch")
        LOG.info(
            "row->frame map: %d rows, %d rows with nonzero ts delta, %d rows "
            "beyond the %.0f ms warn threshold",
            total, n_fallback, self.n_warn, self.WARN_DELTA_NS / 1e6,
        )


class RawReader:
    """LRU h5py reader for raw archives (JPEG frames + puppet joints)."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max_open
        self._lru: "OrderedDict[str, h5py.File]" = OrderedDict()

    def _get(self, raw_path: str) -> h5py.File:
        handle = self._lru.get(raw_path)
        if handle is not None:
            self._lru.move_to_end(raw_path)
            return handle
        if len(self._lru) >= self.max_open:
            old_path, old_handle = self._lru.popitem(last=False)
            old_handle.close()
        handle = h5py.File(raw_path, "r")
        self._lru[raw_path] = handle
        return handle

    def read_row(
        self, raw_path: str, frame: int, cameras: tuple[str, ...]
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        handle = self._get(raw_path)
        images = {
            name: _decode_jpeg(
                bytes(np.asarray(handle[f"images/{name}/jpeg"][frame]))
            )
            for name in cameras
        }
        state = np.asarray(
            handle["frames/puppet_position"][frame], dtype=np.float32
        )
        return images, state

    def close(self) -> None:
        for handle in self._lru.values():
            handle.close()
        self._lru.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def _latent_actor_step(
    *,
    runtime: LatentRuntime,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    indices: np.ndarray,
    row_map: RowFrameMap,
    reader: RawReader,
    device: torch.device,
    q_weight: float,
    bc_weight: float,
    intervention_bc_weight: float,
    ref_anchor_weight: float,
    ref_anchor_box_np: np.ndarray | None,
    action_stats_mean: np.ndarray,
    action_stats_std: np.ndarray,
    il_anchor_weight: float,
    il_anchor_cache: dict[str, np.ndarray] | None,
    il_anchor_indices: np.ndarray | None,
    generator: np.random.Generator,
    exploration_std: float,
    train_vla: bool = False,
) -> dict[str, float]:
    """One actor update: decode-space losses through the reference VLA.

    Backwards row-by-row (each row's graph is freed after its backward) to
    bound memory; per-row losses are scaled by 1/B so the accumulated
    gradient equals the batch-mean gradient.

    When ``train_vla`` is set the reference VLA is fine-tuned too, but ONLY
    through the 10-step execution window (a_exec = the first ``n_action_steps_rl``
    steps of the chunk): every loss term (Q, BC, ref-anchor) is computed on
    a_exec, so no gradient reaches the VLA from the unexecuted 10..29 tail.
    """
    from lerobot.utils.constants import OBS_STATE  # noqa: F401  (kept for parity)

    z = batch["z_rl"]
    s = batch["state"]
    ref = batch["ref_actions"]
    actions = batch["actions"]
    b_size = int(z.shape[0])

    # NOTE: the actor forward is run per row inside the decode loop so each
    # row's graph is freed by its own backward (a single batch-level graph
    # would be freed by the first backward and break the next row).
    # BC target / weights (iv50 mirror): intervention|demo -> executed action
    # with weight intervention_bc_weight, else the VLA reference with weight 1.
    iv_mask = batch["is_intervention"].bool().squeeze(-1)
    demo_mask = (
        batch["is_demo"].bool().squeeze(-1)
        if "is_demo" in batch
        else torch.zeros(b_size, dtype=torch.bool, device=device)
    )
    supervised = iv_mask | demo_mask
    target = torch.where(
        supervised.unsqueeze(1).unsqueeze(2).expand_as(actions),
        actions,
        ref,
    )
    sample_w = torch.where(supervised, torch.tensor(float(intervention_bc_weight), device=device), torch.tensor(1.0, device=device))

    # outside-box ref anchor mask (physical EEF xyz via the shared stats)
    if ref_anchor_box_np is not None:
        std_t = torch.as_tensor(action_stats_std, dtype=torch.float32, device=device)
        mean_t = torch.as_tensor(action_stats_mean, dtype=torch.float32, device=device)
        state_phys = s * std_t + mean_t
        bx0, bx1, by0, by1, bz0, bz1 = (float(v) for v in ref_anchor_box_np)
        in_box = (
            (state_phys[:, 0] >= bx0)
            & (state_phys[:, 0] <= bx1)
            & (state_phys[:, 1] >= by0)
            & (state_phys[:, 1] <= by1)
            & (state_phys[:, 2] >= bz0)
            & (state_phys[:, 2] <= bz1)
        )
        ref_valid = ~in_box
        ref_fraction = float(ref_valid.float().mean())
    else:
        ref_valid = torch.ones(b_size, dtype=torch.bool, device=device)
        ref_fraction = 1.0

    il_scale = float(il_anchor_weight) / (1.0 + float(il_anchor_weight))
    bc_scale = (1.0 / (1.0 + float(il_anchor_weight)))

    actor_optimizer.zero_grad(set_to_none=True)
    metrics = {
        "q_loss": 0.0,
        "replay_bc_loss": 0.0,
        "ref_anchor_loss": 0.0,
        "r_lat_absmax": 0.0,
        "decode_latency_ms": 0.0,
    }
    q_sum = bc_sum = ref_sum = 0.0
    decode_rows = 0
    for j in range(b_size):
        row = int(indices[j])
        raw_path = str(row_map.raw_path_by_row[row])
        frame = int(row_map.frame_by_row[row])
        task = str(row_map.task_by_row[row])
        images, puppet = reader.read_row(raw_path, frame, runtime.cameras)
        state_eef = runtime.eef_state(puppet)
        r_lat_raw_j, _ = runtime.actor(z[j : j + 1], s[j : j + 1], ref[j : j + 1], training=True)
        if exploration_std > 0.0:
            r_lat_j = r_lat_raw_j + float(exploration_std) * torch.randn_like(r_lat_raw_j)
        else:
            r_lat_j = r_lat_raw_j
        t0 = time.perf_counter()
        out = runtime.decode(
            state_eef,
            images,
            task,
            r_lat=r_lat_j,
            seed=_noise_seed_for_row(row),
            keep_graph=True,
        )
        a_exec = out["_reference"]  # (1, C, 7), grad-connected
        metrics["decode_latency_ms"] += time.perf_counter() - t0

        q1, q2 = critic(z[j : j + 1].detach(), s[j : j + 1], a_exec)
        q_row = -torch.minimum(q1, q2).mean()
        bc_row = F.mse_loss(a_exec, target[j : j + 1]) * float(sample_w[j])
        if ref_valid[j]:
            ref_row = F.mse_loss(a_exec, ref[j : j + 1])
        else:
            ref_row = a_exec.sum() * 0.0

        loss_row = (
            float(q_weight) * q_row
            + float(bc_weight) * bc_scale * bc_row
            + float(ref_anchor_weight) * ref_row
        )
        (loss_row / b_size).backward()
        q_sum += float(q_row.detach())
        bc_sum += float(bc_row.detach()) / float(int(supervised.sum()) or 1)
        ref_sum += float(ref_row.detach()) / max(1, int(ref_valid.sum()))
        decode_rows += 1
        metrics["r_lat_absmax"] = max(
            metrics["r_lat_absmax"], float(r_lat_raw_j.abs().max().detach())
        )

    metrics["q_loss"] = q_sum / max(1, decode_rows)
    metrics["replay_bc_loss"] = bc_sum / max(1, decode_rows)
    metrics["ref_anchor_loss"] = ref_sum / max(1, decode_rows)
    metrics["decode_latency_ms"] = 1000.0 * metrics["decode_latency_ms"] / max(1, decode_rows)

    # Latent-space IL anchor: keep r_lat at 0 on IL-distribution rows.
    il_anchor_loss = 0.0
    if il_anchor_cache is not None and il_anchor_indices is not None:
        anchor_rows = generator.choice(
            il_anchor_indices,
            size=min(b_size, int(il_anchor_indices.size)),
            replace=False,
        )
        anchor_z = torch.as_tensor(
            il_anchor_cache["z_rl"][anchor_rows], dtype=torch.float32, device=device
        )
        anchor_state = torch.as_tensor(
            il_anchor_cache["state"][anchor_rows], dtype=torch.float32, device=device
        )
        anchor_reference = torch.as_tensor(
            il_anchor_cache["reference"][anchor_rows], dtype=torch.float32, device=device
        )
        anchor_raw, _ = runtime.actor(
            anchor_z, anchor_state, anchor_reference, training=True
        )
        il_anchor_loss = F.mse_loss(anchor_raw, torch.zeros_like(anchor_raw))
        (float(bc_weight) * il_scale * il_anchor_loss).backward()
        metrics["il_anchor_loss"] = float(il_anchor_loss.detach())

    clip_params: list = (
        list(runtime.actor.parameters()) + list(runtime.projector.parameters())
    )
    if train_vla:
        clip_params = clip_params + list(runtime.policy.parameters())
    torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
    if train_vla:
        vla_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(runtime.policy.parameters()), 10.0
        )
        metrics["vla_grad_norm"] = float(vla_grad_norm)
    actor_optimizer.step()
    metrics["bc_supervised_fraction"] = float(supervised.float().mean())
    metrics["ref_anchor_fraction"] = ref_fraction
    return metrics


def _latent_heldout_gate(
    *,
    runtime: LatentRuntime,
    arrays: dict[str, np.ndarray],
    row_map: RowFrameMap,
    reader: RawReader,
    device: torch.device,
    seed: int,
    max_gate_rows: int = 1500,
) -> dict[str, Any]:
    """Decoded-space held-out gate on replay intervention rows.

    Baseline = the stored VLA reference (r_lat=0 prior) vs the executed
    (human-corrected) action; actor = the decoded a_exec with the trained
    latent actor (no exploration, fixed per-row noise).  Per-dim
    actor_mse / baseline_mse must stay <= the threshold on every dim.
    """
    episodes = np.unique(arrays["replay_episode_index"].reshape(-1))
    if episodes.size < 4:
        raise ValueError("held-out gate needs at least 4 replay episodes")
    val_count = max(1, min(episodes.size - 1, round(episodes.size * 0.10)))
    val_eps = np.sort(np.random.default_rng(seed + 777).choice(episodes, size=val_count, replace=False))
    episode_index = arrays["replay_episode_index"].reshape(-1)
    is_intervention = arrays["is_intervention"].reshape(-1).astype(bool)
    val_rows = np.isin(episode_index, val_eps)
    gate_rows = np.flatnonzero(val_rows & is_intervention)
    if gate_rows.size < 32:
        LOG.warning(
            "held-out gate: only %d intervention rows in the val split; "
            "falling back to all val rows", gate_rows.size,
        )
        gate_rows = np.flatnonzero(val_rows)
    gate_rows = gate_rows[: max_gate_rows]
    if gate_rows.size == 0:
        raise ValueError("held-out gate has no rows")

    per_dim_actor = np.zeros(7, dtype=np.float64)
    per_dim_baseline = np.zeros(7, dtype=np.float64)
    with torch.no_grad():
        for row in gate_rows:
            raw_path = str(row_map.raw_path_by_row[int(row)])
            frame = int(row_map.frame_by_row[int(row)])
            task = str(row_map.task_by_row[int(row)])
            images, puppet = reader.read_row(raw_path, frame, runtime.cameras)
            state_eef = runtime.eef_state(puppet)
            z = torch.as_tensor(
                arrays["z_rl"][int(row)], dtype=torch.float32, device=device
            ).unsqueeze(0)
            s = torch.as_tensor(
                arrays["state"][int(row)], dtype=torch.float32, device=device
            ).unsqueeze(0)
            ref = torch.as_tensor(
                arrays["ref_actions"][int(row)], dtype=torch.float32, device=device
            ).unsqueeze(0)
            r_lat, _ = runtime.actor(z, s, ref, training=False)
            out = runtime.decode(
                state_eef,
                images,
                task,
                r_lat=r_lat,
                seed=_noise_seed_for_row(int(row)),
                keep_graph=False,
            )
            decoded = np.asarray(out["reference_actions"], dtype=np.float64)
            target = arrays["actions"][int(row)].astype(np.float64)
            baseline = arrays["ref_actions"][int(row)].astype(np.float64)
            per_dim_actor += ((decoded - target) ** 2).mean(axis=0)
            per_dim_baseline += ((baseline - target) ** 2).mean(axis=0)
    per_dim_actor /= gate_rows.size
    per_dim_baseline /= gate_rows.size
    ratios = np.ones(7, dtype=np.float64)
    for d in range(7):
        if per_dim_baseline[d] > 1.0e-8:
            ratios[d] = per_dim_actor[d] / per_dim_baseline[d]
        else:
            ratios[d] = 1.0 if per_dim_actor[d] < 1.0e-8 else 10.0
    return {
        "method": "decoded_space_heldout_iv_rows",
        "n_val_episodes": int(val_count),
        "n_gate_rows": int(gate_rows.size),
        "per_dim_mse": [float(v) for v in per_dim_actor],
        "per_dim_baseline_mse": [float(v) for v in per_dim_baseline],
        "per_dim_ratio": [float(v) for v in ratios],
        "max_ratio": float(ratios.max()),
        "regressed_dims": [int(d) for d in np.flatnonzero(ratios > DEFAULT_MAX_HELDOUT_RATIO)],
    }


def _latent_deviation_stats(
    runtime: LatentRuntime,
    arrays: dict[str, np.ndarray],
    row_map: RowFrameMap,
    reader: RawReader,
    device: torch.device,
    sample_rows: int = 160,
    seed: int = 42,
) -> dict[str, float]:
    """Mean |a_exec - ref| on a sample of rows (decode, no exploration)."""
    n = int(arrays["z_rl"].shape[0])
    rows = np.random.default_rng(seed).choice(n, size=min(sample_rows, n), replace=False)
    dev = 0.0
    with torch.no_grad():
        for row in rows:
            raw_path = str(row_map.raw_path_by_row[int(row)])
            frame = int(row_map.frame_by_row[int(row)])
            task = str(row_map.task_by_row[int(row)])
            images, puppet = reader.read_row(raw_path, frame, runtime.cameras)
            state_eef = runtime.eef_state(puppet)
            z = torch.as_tensor(
                arrays["z_rl"][int(row)], dtype=torch.float32, device=device
            ).unsqueeze(0)
            s = torch.as_tensor(
                arrays["state"][int(row)], dtype=torch.float32, device=device
            ).unsqueeze(0)
            ref = torch.as_tensor(
                arrays["ref_actions"][int(row)], dtype=torch.float32, device=device
            ).unsqueeze(0)
            r_lat, _ = runtime.actor(z, s, ref, training=False)
            out = runtime.decode(
                state_eef, images, task, r_lat=r_lat,
                seed=_noise_seed_for_row(int(row)), keep_graph=False,
            )
            decoded = np.asarray(out["reference_actions"], dtype=np.float64)
            dev += float(np.abs(decoded - ref.squeeze(0).cpu().numpy()).mean())
    return {"mean_abs_error_to_reference": dev / len(rows)}


def train_stage2_latent(
    *,
    stage1_checkpoint: Path,
    replay_paths: list[Path],
    output_dir: Path,
    device: str,
    updates: int,
    batch_size: int,
    seed: int,
    log_freq: int,
    progress_file: Path | None,
    discount: float,
    critic_target_mode: str,
    q_weight: float,
    bc_weight: float,
    intervention_bc_weight: float,
    ref_anchor_weight: float,
    ref_anchor_mask: str,
    ref_anchor_box: str,
    il_anchor_weight: float,
    il_anchor_cache: Path | None,
    critic_per_actor: int,
    q_target_clip: float,
    actor_exploration_std: float,
    actor_lr: float | None,
    critic_lr: float | None,
    balanced_outcomes: bool,
    gate_rows: int,
    max_heldout_ratio: float,
    train_vla: bool = False,
    vla_lr: float = 1.0e-5,
    vla_weight_decay: float = 0.0,
) -> dict[str, Any]:
    if critic_target_mode != "mc":
        raise ValueError("latent route currently supports only --critic-target-mode mc")
    if ref_anchor_mask != "outside_box" and ref_anchor_weight > 0:
        raise ValueError("this run uses ref_anchor_mask=outside_box")
    ref_anchor_box_np: np.ndarray | None = None
    if ref_anchor_weight > 0:
        ref_anchor_box_np = np.asarray(
            [float(v) for v in ref_anchor_box.split(",")], dtype=np.float64
        )
        if ref_anchor_box_np.shape != (6,):
            raise ValueError("ref_anchor_box must be 'x0,x1,y0,y1,z0,z1'")
    stage1_checkpoint = stage1_checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    if (
        stage1.get("stage") != "rlt_stage1_joint"
        or stage1.get("rlt_architecture") != "rlinf_prefix_v1"
    ):
        raise ValueError("Stage 2 latent requires an RLinf prefix Stage 1 checkpoint")

    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLCriticMLP

    arrays, replay_metadata = load_replay_dataset(
        replay_paths, stage1_checkpoint=stage1_checkpoint
    )
    if discount != DEFAULT_STAGE2_DISCOUNT:
        # recompute MC returns at the requested discount
        rewards = arrays["rewards"].reshape(-1)
        dones = arrays["dones"].reshape(-1)
        mc_returns = np.empty_like(rewards, dtype=np.float32)
        running_return = 0.0
        for index in range(rewards.size - 1, -1, -1):
            running_return = float(rewards[index]) + (
                0.0 if bool(dones[index]) else float(discount) * running_return
            )
            mc_returns[index] = running_return
        arrays["mc_returns"] = mc_returns[:, None]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    target_device = torch.device(device)

    # Frozen VLA + encoder + (trainable) latent actor + projector.
    # With train_vla the reference VLA is fine-tuned too: gradients flow only
    # through the 30 action-suffix tokens (the prefix KV stays no_grad), i.e.
    # the reference's decode/output path is trainable while the image-encoder
    # path remains constant per row.
    runtime = LatentRuntime(stage1_checkpoint=stage1_checkpoint, device=device)
    config = runtime.config
    runtime.actor.train()
    runtime.actor.requires_grad_(True)
    runtime.projector.requires_grad_(True)
    if train_vla:
        runtime.policy.requires_grad_(True)
        n_vla = sum(int(p.numel()) for p in runtime.policy.parameters())
        LOG.info(
            "VLA fine-tuning enabled: %d params, lr=%.2e wd=%.2e",
            n_vla, float(vla_lr), float(vla_weight_decay),
        )
    else:
        runtime.policy.requires_grad_(False)

    if actor_lr is not None:
        config_payload_lr = float(actor_lr)
    else:
        config_payload_lr = float(getattr(config, "actor_lr", 3.0e-4))
    critic_lr_value = float(critic_lr) if critic_lr is not None else float(
        getattr(config, "critic_lr", 3.0e-4)
    )
    critic = RLCriticMLP(config).to(target_device)
    critic_target = RLCriticMLP(config).to(target_device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.requires_grad_(False)
    actor_param_groups: list[dict] = [
        {
            "params": list(runtime.actor.parameters())
            + list(runtime.projector.parameters()),
            "lr": config_payload_lr,
            "weight_decay": 0.0,
        }
    ]
    if train_vla:
        actor_param_groups.append(
            {
                "params": list(runtime.policy.parameters()),
                "lr": float(vla_lr),
                "weight_decay": float(vla_weight_decay),
            }
        )
    actor_optimizer = torch.optim.Adam(actor_param_groups)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr_value)

    # outside-box stats: the shared EEF mean/std live in the normalizer under
    # 'observation.state' (state and action share the EEF pose space).
    state_stats = runtime.normalizer._tensor_stats.get("observation.state", {})
    if ref_anchor_box_np is not None:
        action_stats_mean = np.asarray(state_stats["mean"], dtype=np.float64)[:7]
        action_stats_std = np.asarray(state_stats["std"], dtype=np.float64)[:7]
        if action_stats_mean.shape != (7,) or not bool(np.all(action_stats_std > 0)):
            raise ValueError("state stats are invalid for the outside-box mask")
    else:
        action_stats_mean = np.zeros(7)
        action_stats_std = np.ones(7)

    row_map = RowFrameMap(replay_paths, replay_metadata)
    reader = RawReader(max_open=8)

    il_anchor_cache_arr: dict[str, np.ndarray] | None = None
    il_anchor_indices: np.ndarray | None = None
    if il_anchor_weight > 0:
        if il_anchor_cache is None:
            raise ValueError("il_anchor_weight requires --il-anchor-cache")
        il_anchor_cache_arr, il_anchor_indices = _load_il_anchor_cache(
            il_anchor_cache.expanduser().resolve(), seed=seed
        )
        LOG.info(
            "IL latent anchor rows=%d weight=%.3f",
            il_anchor_indices.size,
            il_anchor_weight,
        )

    total_rows = int(arrays["z_rl"].shape[0])
    effective_batch = min(batch_size, total_rows)
    n_actor_updates = int(math.ceil(updates / critic_per_actor))
    LOG.info(
        "latent Stage 2: rows=%d bundles=%d updates=%d batch=%d actor_updates=%d "
        "device=%s q=%.3f bc=%.2f iv=%.1f ref=%.1f il=%.1f",
        total_rows, len(replay_paths), updates, effective_batch,
        n_actor_updates, target_device, q_weight, bc_weight,
        intervention_bc_weight, ref_anchor_weight, il_anchor_weight,
    )

    run_started = time.monotonic()
    last_progress_write = 0.0
    progress_base = {
        "stage": "rlt_stage2_latent",
        "state": "running",
        "route": "latent_residual",
        "stage1_checkpoint": str(stage1_checkpoint),
        "device": str(target_device),
        "started_at_ns": time.time_ns(),
        "total_updates": updates,
        "buffer_transitions": total_rows,
        "replay_root": str(replay_paths[0].resolve().parent),
        "batch_size": effective_batch,
        "q_weight": float(q_weight),
        "bc_weight": float(bc_weight),
        "intervention_bc_weight": float(intervention_bc_weight),
        "ref_anchor_weight": float(ref_anchor_weight),
        "ref_anchor_mask": ref_anchor_mask,
        "ref_anchor_box": ref_anchor_box,
        "il_anchor_weight": float(il_anchor_weight),
        "discount": float(discount),
        "critic_target_mode": critic_target_mode,
        "q_target_clip": float(q_target_clip),
        "critic_per_actor": int(critic_per_actor),
        "actor_exploration_std": float(actor_exploration_std),
        "actor_output_semantics": LATENT_ACTOR_OUTPUT_SEMANTICS,
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
        remaining = max(0, updates - completed)
        eta = remaining / rate if rate and rate > 0 else None
        percent = 100.0 if updates == 0 else 100.0 * completed / updates
        _atomic_json(
            {
                **progress_base,
                "updated_at_ns": time.time_ns(),
                "current_update": completed,
                "percent": min(100.0, max(0.0, percent)),
                "elapsed_seconds": elapsed,
                "updates_per_second": rate,
                "eta_seconds": eta,
                "last_metrics": _finite_metrics(last_metrics),
            },
            progress_file,
        )
        last_progress_write = now

    generator = np.random.default_rng(seed)
    last_metrics: dict[str, float] = {}
    last_actor_metrics: dict[str, float] = {}
    actor_updates_done = 0
    publish_progress(0, force=True)
    for update_index in range(updates):
        if balanced_outcomes:
            indices = _sample_balanced_episode_indices(
                generator,
                arrays["outcome_success"],
                arrays["replay_episode_index"],
                effective_batch,
            )
        else:
            indices = generator.choice(total_rows, size=effective_batch, replace=False)
        batch = {
            name: torch.as_tensor(
                values[indices], dtype=torch.float32, device=target_device
            )
            for name, values in arrays.items()
        }
        # ── critic step (iv50 mirror: MC target on the executed actions) ──
        with torch.no_grad():
            pass
        q1, q2 = critic(batch["z_rl"], batch["state"], batch["actions"])
        mc_loss = (
            F.mse_loss(q1, batch["mc_returns"]) + F.mse_loss(q2, batch["mc_returns"])
        )
        critic_optimizer.zero_grad(set_to_none=True)
        mc_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        critic_optimizer.step()

        global_update = update_index + 1
        if global_update % critic_per_actor == 0:
            actor_metrics = _latent_actor_step(
                runtime=runtime,
                critic=critic,  # iv50 mirror: actor Q-gradient on the online critic
                actor_optimizer=actor_optimizer,
                batch=batch,
                indices=indices,
                row_map=row_map,
                reader=reader,
                device=target_device,
                q_weight=q_weight,
                bc_weight=bc_weight,
                intervention_bc_weight=intervention_bc_weight,
                ref_anchor_weight=ref_anchor_weight,
                ref_anchor_box_np=ref_anchor_box_np,
                action_stats_mean=action_stats_mean,
                action_stats_std=action_stats_std,
                il_anchor_weight=il_anchor_weight,
                il_anchor_cache=il_anchor_cache_arr,
                il_anchor_indices=il_anchor_indices,
                generator=generator,
                exploration_std=actor_exploration_std,
                train_vla=train_vla,
            )
            last_actor_metrics = actor_metrics
            actor_updates_done += 1
            if actor_updates_done <= 3:
                LOG.info(
                    "actor-start update=%d actor_updates=%d %s",
                    global_update, actor_updates_done, actor_metrics,
                )

        # target critic EMA
        tau = 0.005
        for parameter, target in zip(critic.parameters(), critic_target.parameters()):
            target.data.mul_(1.0 - tau).add_(parameter.data, alpha=tau)

        last_metrics = {
            "critic_loss": float(mc_loss.detach().cpu()),
            "q1_mean": float(q1.mean().detach().cpu()),
            "q2_mean": float(q2.mean().detach().cpu()),
            **last_actor_metrics,
        }
        if not all(math.isfinite(v) for v in last_metrics.values()):
            raise RuntimeError(
                f"non-finite latent Stage 2 metrics at update {global_update}: "
                f"{last_metrics}"
            )
        if last_metrics["critic_loss"] > 1.0e4 or abs(last_metrics["q1_mean"]) > 1.0e3:
            raise RuntimeError(
                f"critic diverged at update {global_update}: {last_metrics}"
            )
        if global_update == 1 or global_update % log_freq == 0:
            LOG.info("update=%d metrics=%s", global_update, last_metrics)
        publish_progress(global_update)

    # ── final validation + gate ──
    LOG.info("computing latent actor deviation stats (160 rows)...")
    actor_validation = _latent_deviation_stats(
        runtime, arrays, row_map, reader, target_device, seed=seed
    )
    LOG.info("deviation stats: %s", actor_validation)
    if not all(math.isfinite(v) for v in actor_validation.values()):
        raise RuntimeError(f"latent actor validation is non-finite: {actor_validation}")
    if actor_validation["mean_abs_error_to_reference"] > 0.5:
        raise RuntimeError(
            "latent actor is too far from the reference decode: "
            f"{actor_validation}"
        )
    LOG.info("computing decoded-space held-out gate...")
    heldout = _latent_heldout_gate(
        runtime=runtime,
        arrays=arrays,
        row_map=row_map,
        reader=reader,
        device=target_device,
        seed=seed,
        max_gate_rows=gate_rows,
    )
    ready_for_online = bool(
        heldout["max_ratio"] <= max_heldout_ratio
        and actor_validation["mean_abs_error_to_reference"] <= 0.25
    )
    heldout["max_ratio_threshold"] = float(max_heldout_ratio)
    heldout["ready_for_online"] = ready_for_online
    LOG.info(
        "gate: max_ratio=%.4f (threshold %.2f) regressed_dims=%s ready=%s",
        heldout["max_ratio"], max_heldout_ratio,
        heldout["regressed_dims"], ready_for_online,
    )

    config_payload = dict(runtime._config_payload)
    config_payload["expert_hidden_size"] = int(runtime.expert_hidden)
    config_payload["n_action_steps_rl"] = int(runtime.n_action_steps_rl)
    config_payload["chunk_size"] = int(runtime.chunk_size)
    config_payload["actor_output_mode"] = "residual"
    payload = {
        "format_version": STAGE2_LATENT_FORMAT,
        "stage": "rlt_stage2_latent",
        "route": "latent_residual",
        "actor_output_semantics": LATENT_ACTOR_OUTPUT_SEMANTICS,
        "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
        "action_chunk_frame_stride": ACTION_CHUNK_FRAME_STRIDE,
        "stage1_checkpoint": str(stage1_checkpoint),
        "smolvla_path": stage1.get("smolvla_path"),
        "base_smolvla_path": stage1.get("base_smolvla_path"),
        "rlt_config": config_payload,
        "actor_state_dict": runtime.actor.state_dict(),
        "projector_state_dict": runtime.projector.state_dict(),
        "policy_state_dict": runtime.policy.state_dict() if train_vla else None,
        "vla_trained": bool(train_vla),
        "critic_state_dict": critic.state_dict(),
        "critic_target_state_dict": critic_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "total_updates": updates,
        "actor_updates": actor_updates_done,
        "buffer_transitions": total_rows,
        "replay_root": str(replay_paths[0].resolve().parent),
        "replay_paths": [str(p.resolve()) for p in replay_paths],
        "trained_replay_names": sorted(replay_names(replay_paths)),
        "weights": {
            "q_weight": float(q_weight),
            "bc_weight": float(bc_weight),
            "intervention_bc_weight": float(intervention_bc_weight),
            "ref_anchor_weight": float(ref_anchor_weight),
            "ref_anchor_mask": ref_anchor_mask,
            "ref_anchor_box": ref_anchor_box,
            "il_anchor_weight": float(il_anchor_weight),
            "critic_target_mode": critic_target_mode,
            "discount": float(discount),
            "q_target_clip": float(q_target_clip),
            "critic_per_actor": int(critic_per_actor),
            "actor_exploration_std": float(actor_exploration_std),
            "actor_lr": config_payload_lr,
            "critic_lr": critic_lr_value,
            "train_vla": bool(train_vla),
            "vla_lr": float(vla_lr),
            "vla_weight_decay": float(vla_weight_decay),
        },
        "last_metrics": _finite_metrics(last_metrics),
        "actor_validation": actor_validation,
        "heldout_validation": heldout,
        "ready_for_online": ready_for_online,
        "expert_hidden_size": int(runtime.expert_hidden),
    }
    checkpoint_path = output_dir / f"checkpoint_{updates:08d}.pt"
    _atomic_torch_save(payload, checkpoint_path)
    _atomic_torch_save(payload, output_dir / "latest.pt")
    status = {
        key: payload[key]
        for key in (
            "stage",
            "route",
            "actor_output_semantics",
            "action_chunk_timebase",
            "stage1_checkpoint",
            "total_updates",
            "actor_updates",
            "buffer_transitions",
            "replay_root",
            "weights",
            "last_metrics",
            "actor_validation",
            "heldout_validation",
            "ready_for_online",
            "vla_trained",
        )
        if key in payload
    }
    status["checkpoint"] = str(checkpoint_path)
    status["latest"] = str(output_dir / "latest.pt")
    status["output_dir"] = str(output_dir)
    _atomic_json(status, output_dir / "status.json")
    if progress_file is not None:
        progress_base["state"] = "complete"
        progress_base["finished_at_ns"] = time.time_ns()
        _atomic_json(
            {
                **progress_base,
                "current_update": updates,
                "percent": 100.0,
                "elapsed_seconds": time.monotonic() - run_started,
                "last_metrics": _finite_metrics(last_metrics),
                "ready_for_online": ready_for_online,
            },
            progress_file,
        )
    reader.close()
    LOG.info(
        "latent Stage 2 complete: %d updates / %d actor updates, ready=%s",
        updates, actor_updates_done, ready_for_online,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=16)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--discount", type=float, default=DEFAULT_STAGE2_DISCOUNT)
    parser.add_argument(
        "--critic-target-mode", choices=("mc",), default="mc"
    )
    parser.add_argument("--q-weight", type=float, default=0.01)
    parser.add_argument("--bc-weight", type=float, default=0.1)
    parser.add_argument("--intervention-bc-weight", type=float, default=50.0)
    parser.add_argument("--ref-anchor-weight", type=float, default=5.0)
    parser.add_argument(
        "--ref-anchor-mask", choices=("outside_box", "none"), default="outside_box"
    )
    parser.add_argument(
        "--ref-anchor-box", default="0.31,0.39,0.30,0.35,0.06,0.13"
    )
    parser.add_argument("--il-anchor-weight", type=float, default=30.0)
    parser.add_argument("--il-anchor-cache", type=Path, default=None)
    parser.add_argument("--critic-per-actor", type=int, default=DEFAULT_CRITIC_PER_ACTOR)
    parser.add_argument("--q-target-clip", type=float, default=1.0)
    parser.add_argument("--actor-exploration-std", type=float, default=0.05)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument(
        "--balanced-outcomes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gate-rows", type=int, default=1500)
    parser.add_argument("--max-heldout-ratio", type=float, default=DEFAULT_MAX_HELDOUT_RATIO)
    parser.add_argument(
        "--train-vla",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fine-tune the reference VLA too (gradients through the 10-step "
             "execution window only; prefix KV stays no-grad)",
    )
    parser.add_argument("--vla-lr", type=float, default=1.0e-5)
    parser.add_argument("--vla-weight-decay", type=float, default=0.0)
    return parser


def _setup_logging() -> None:
    """Guarantee an INFO-level console handler on the root logger."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _setup_logging()
    args = build_parser().parse_args()
    replay_paths = discover_replay_bundles(args.replay_root)
    if not replay_paths:
        raise FileNotFoundError(f"no replay bundles under {args.replay_root}")
    train_stage2_latent(
        stage1_checkpoint=args.stage1_checkpoint,
        replay_paths=replay_paths,
        output_dir=args.output_dir,
        device=args.device,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed,
        log_freq=args.log_freq,
        progress_file=args.progress_file,
        discount=args.discount,
        critic_target_mode=args.critic_target_mode,
        q_weight=args.q_weight,
        bc_weight=args.bc_weight,
        intervention_bc_weight=args.intervention_bc_weight,
        ref_anchor_weight=args.ref_anchor_weight,
        ref_anchor_mask=args.ref_anchor_mask,
        ref_anchor_box=args.ref_anchor_box,
        il_anchor_weight=args.il_anchor_weight,
        il_anchor_cache=args.il_anchor_cache,
        critic_per_actor=args.critic_per_actor,
        q_target_clip=args.q_target_clip,
        actor_exploration_std=args.actor_exploration_std,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        balanced_outcomes=args.balanced_outcomes,
        gate_rows=args.gate_rows,
        max_heldout_ratio=args.max_heldout_ratio,
        train_vla=args.train_vla,
        vla_lr=args.vla_lr,
        vla_weight_decay=args.vla_weight_decay,
    )


if __name__ == "__main__":
    main()
