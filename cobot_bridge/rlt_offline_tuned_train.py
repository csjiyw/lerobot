"""Offline Stage-2 RLT training — port of the tuned LIBERO TD3 recipe.

Source of truth for the method: the tuned LIBERO RLT repo
(/workspace/vla-rlt-main on host 172.16.128.24): scripts/train_libero.py +
configs/tasks/libero_pick_place_chunk10_exec10_tuned.yaml, README.md —
run runs/rlt_libero_spatial_task0_tuned_start40_warmup10k_chunk10_exec10_
20260828 (VLA 58% -> RLT 62% over 50 paired episodes, 2026-09-01).
Hyperparameters:

  CHUNK=10, GAMMA=0.99 (bootstrap gamma**10 per 10-frame transition),
  BATCH=64, UTD=5, CRITIC_PER_ACTOR=2, BC_W=0.5,
  MAX_CORR=0.03 (per-dim correction cap, see ACTION_SCALE/RESIDUAL_SCALE),
  Q_TARGET_CLIP=1.0, ACTOR_LR=5e-6, CRITIC_LR=5e-5, TAU=0.005,
  WARMUP_ITERS=10000 (critic-only), SUC_FRAC=0.10 (pool mix),
  ref_dropout=0.0 (tuned yaml), actor/critic MLP [256,256] silu.

Actor composition = the repo's residual form: action = ref + tanh(c)*cap
with a zero-initialized correction head (passthrough of the VLA at init).
It is computed with rlt_stage2_data.compose_actor_action (RESIDUAL
semantics) so training and the online rlt_policy_server compose
IDENTICALLY (train==deploy by construction). Per-dim cap =
ACTION_SCALE[i] * RESIDUAL_SCALE (~0.03 on the low-scale dims, the tuned
max_action_correction; proportionally larger on the high-scale joints).

Offline adaptation for the real-robot cobot_magic_rlt dataset (172 episodes):
  * No live environment: buffer = all robot-episode replay bundles (static).
  * Success pool (10% of each batch) = success-episode rows + IL demo rows.
  * Total iterations = UTD * n_buffer_rows: every transition gets its UTD-5
    update block (the online UTD budget, applied once per stored transition).
  * Exploration noise and auto-intervention: not applicable (no execution).
  * z = 960-dim Stage-1 encoder output (rlinf960_do02), identical to the
    76% A run — same data contract, same frozen VLA prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cobot_bridge.rlt_stage2_data import RESIDUAL_ACTOR_OUTPUT_SEMANTICS

CHUNK = 10
GAMMA = 0.99
BATCH = 64
UTD = 5
CRITIC_PER_ACTOR = 2
BC_W = 0.5
MAX_CORR = 0.03
Q_TARGET_CLIP = 1.0
ACTOR_LR = 5e-6
CRITIC_LR = 5e-5
TAU = 0.005
WARMUP_ITERS = 10000
SUC_FRAC = 0.10
REF_DROPOUT = 0.0  # tuned yaml value (default 0.5)
SEED = 42
EVAL_SLICE = 512
# Auto-intervention thresholds (tuned yaml) — offline takeover diagnostic.
INTERV_CHUNK_RMS = 0.125
INTERV_FIRST_RMS = 0.125
INTERV_MAX_ABS = 0.4
# Deployment contract (rlt_policy_server / rlt_stage2_data).
ACTION_CHUNK_TIMEBASE = "contiguous_30hz_v1"
# Per-dim action scale = clamp bound for the RESIDUAL compose.
# Must be >= per-dim max |ref| (data: [2.0,1.8,2.0,9.4,3.1,6.6,1.5]) or the
# VLA reference itself gets clipped. 1.15x margin over measured ref max.
ACTION_SCALE = [2.10, 2.02, 2.27, 7.58, 3.74, 5.17, 1.66]
# Scalar residual scale: per-dim correction cap = scale_i * residual_scale.
# 0.015 -> cap ~0.03 on the low-scale dims (tuned max_action_correction),
# ~0.11 normalized on the high-scale joints (~1.4 deg physical, benign).
RESIDUAL_SCALE = 0.015


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-root", type=Path, required=True)
    p.add_argument("--stage1-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--utd", type=int, default=UTD)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--bc-weight", type=float, default=BC_W)
    p.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE,
                   help="per-dim correction cap = ACTION_SCALE[i]*residual_scale")
    p.add_argument("--suf-frac", type=float, default=SUC_FRAC)
    p.add_argument("--ref-dropout", type=float, default=REF_DROPOUT)
    p.add_argument("--actor-lr", type=float, default=ACTOR_LR)
    p.add_argument("--critic-lr", type=float, default=CRITIC_LR)
    p.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    p.add_argument("--exec-steps", type=int, default=CHUNK,
                   help="executed frames per transition; bootstrap = gamma**exec_steps")
    p.add_argument("--q-target-clip", type=float, default=Q_TARGET_CLIP)
    p.add_argument("--iters", type=int,
                   help="total td3_update blocks; default = utd * n_buffer_rows")
    p.add_argument("--max-iters", type=int, default=400000)
    p.add_argument("--log-freq", type=int, default=500)
    p.add_argument("--ckpt-freq", type=int, default=5000)
    p.add_argument("--smoke", action="store_true",
                   help="short run (1000 iters) for sanity checking only")
    # ── Placement-phase gate (real-robot analog of actor_start_step) ──
    p.add_argument("--no-gate", action="store_true",
                   help="disable the placement gate: actor Q-max on all rows "
                        "(reproduces the un-gated 51k recipe)")
    p.add_argument("--gate-box-min", type=float, nargs=3, default=(0.31, 0.30, 0.060),
                   metavar=("X", "Y", "Z"),
                   help="insertion box min, right Piper base frame (m); "
                        "default = rlt_eef_gate.EEFWorkspace default")
    p.add_argument("--gate-box-max", type=float, nargs=3, default=(0.39, 0.35, 0.130),
                   metavar=("X", "Y", "Z"),
                   help="insertion box max, right Piper base frame (m)")
    p.add_argument("--gate-latch-hits", type=int, default=2,
                   help="consecutive in-box rows that latch the placement "
                        "phase (deployment EEF gate: 2)")
    p.add_argument("--gate-fallback-tail", type=int, default=4,
                   help="if the EEF never enters the box, treat the last N "
                        "chunks of the episode as the placement attempt")
    p.add_argument("--actor-init-checkpoint", type=Path, default=None,
                   help="path to the actor-BC init checkpoint; recorded in the "
                        "payload so the dashboard finds the IL validation cache")
    return p


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────

class StaticBuffer:
    """Rows: (z, s, ref, a, r, dn, zn, sn, rfn, gate), contiguous arrays.

    gate[row] = True when the row belongs to the episode's placement
    phase (latched once the EEF enters the insertion box).  Only gated
    rows contribute to the actor Q-max term; BC-to-reference applies to
    all rows (this is the real-robot analog of the tuned LIBERO sim's
    ``actor_start_step`` rollout gate, applied per-row in training).
    """

    def __init__(self, z, s, ref, a, r, dn, zn, sn, rfn, success_row, gate):
        self.z, self.s, self.ref, self.a = z, s, ref, a
        self.r, self.dn = r, dn
        self.zn, self.sn, self.rfn = zn, sn, rfn
        self.success_row = success_row.astype(bool)
        self.gate = gate.astype(bool)
        self.n = int(z.shape[0])
        assert all(x.shape[0] == self.n for x in
                   (s, ref, a, r, dn, zn, sn, rfn, self.success_row, self.gate))


def load_replay(root: Path):
    """Load all bundles; robot rows -> buffer, success rows + demos -> pool."""
    from cobot_bridge.rlt_stage2_data import load_replay_bundle
    root = root.expanduser().resolve()
    bundles = sorted(root.glob("*.npz"))
    if not bundles:
        raise FileNotFoundError(f"no .npz bundles in {root}")
    z_blocks, s_blocks, ref_blocks, a_blocks = [], [], [], []
    r_blocks, dn_blocks, zn_blocks, sn_blocks, rfn_blocks = [], [], [], [], []
    succ_blocks, name_blocks = [], []
    n_robot_bundles = n_demo_bundles = 0
    for p in bundles:
        arrays, meta = load_replay_bundle(p)
        is_demo = bool(meta.get("is_demo", 0)) or "il_demo" in p.name
        n = int(arrays["rewards"].shape[0])
        if is_demo:
            n_demo_bundles += 1
        else:
            n_robot_bundles += 1
        z_blocks.append(arrays["z_rl"])
        s_blocks.append(arrays["state"])
        ref_blocks.append(arrays["ref_actions"].reshape(n, -1))
        a_blocks.append(arrays["actions"].reshape(n, -1))
        r_blocks.append(arrays["rewards"].reshape(-1))
        dn_blocks.append(arrays["dones"].reshape(-1))
        zn_blocks.append(arrays["next_z_rl"])
        sn_blocks.append(arrays["next_state"])
        rfn_blocks.append(arrays["next_ref_actions"].reshape(n, -1))
        succ = np.zeros(n, dtype=np.float32)
        if not is_demo and meta.get("outcome") == "success":
            succ[:] = 1.0
        succ_blocks.append(succ)
        name_blocks.extend([p.name] * n)

    def cat(blocks):
        return np.concatenate(blocks, axis=0).astype(np.float32)

    data = {
        "z": cat(z_blocks), "s": cat(s_blocks),
        "ref": cat(ref_blocks), "a": cat(a_blocks),
        "r": cat(r_blocks), "dn": cat(dn_blocks),
        "zn": cat(zn_blocks), "sn": cat(sn_blocks),
        "rfn": cat(rfn_blocks),
        "success_row": cat(succ_blocks),
    }
    names = np.array(name_blocks)
    n = data["z"].shape[0]
    print(f"[data] {len(bundles)} bundles ({n_robot_bundles} robot + "
          f"{n_demo_bundles} demo) -> {n} robot rows, "
          f"success-row share = {data['success_row'].mean():.3f}", flush=True)
    return data, names


# ─────────────────────────────────────────────────────────────────────
# Placement-phase gate (real-robot analog of the sim's actor_start_step)
# ─────────────────────────────────────────────────────────────────────
# Tuned LIBERO sim recipe: rollout gate ``actor_active = total_env_steps
# >= warm_up_steps and episode_steps >= actor_start_step(40)`` — pure VLA
# for the first 40 steps of every episode, actor residual from step 40 to
# episode end (latched).  That gate shaped the DATA: pre-start rows carried
# executed == VLA reference (zero residual), post-start rows carried the
# actor's correction.
#
# Real robot (measured on this replay set, FK over normalized joint state):
# the EEF enters the insertion box at median ~chunk 30 (~step 300),
# p25=chunk 14 / p75=chunk 47, i.e. 46-73% of episode length — far too
# variable for a fixed step index.  The spatial analog is: a row is
# "placement" from the first `latch_hits` consecutive EEF-in-box hits of
# its episode until the episode end (same box + latch semantics as the
# deployment EEF gate, rlt_eef_gate.py EEFWorkspace defaults).  Episodes
# whose EEF never enters the box (failed attempts) fall back to the last
# `fallback_tail` chunks as the placement attempt.
#
# Effect in training: pre-placement rows lose the actor Q-max term and are
# trained by BC-to-reference only -> the actor learns a zero correction
# there (passthrough VLA), exactly matching the sim's pre-start data.

def compute_placement_gate(data, names, stage1_ckpt: Path, box_min, box_max,
                           latch_hits: int = 2, fallback_tail: int = 4):
    from safetensors import safe_open
    from cobot_bridge.piper_eef_metrics import (
        DEFAULT_PIPER_URDF, PiperForwardKinematics)

    # Frozen SmolVLA processor normalizer stats (state in bundles is
    # normalized; FK needs raw joint angles).
    s1 = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    pol = Path(s1["smolvla_path"]).expanduser().resolve()
    norm = sorted(pol.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(norm) != 1:
        raise RuntimeError(f"expected one normalizer file under {pol}, found {len(norm)}")
    with safe_open(norm[0], framework="pt") as src:
        st_mean = src.get_tensor("observation.state.mean").numpy()
        st_std = src.get_tensor("observation.state.std").numpy()

    raw = np.asarray(data["s"], dtype=np.float64) * st_std + st_mean
    fk = PiperForwardKinematics(DEFAULT_PIPER_URDF, frames=("gripper_center",))
    pos = fk.forward(raw[:, :6])["gripper_center"][0]
    lo = np.asarray(box_min, dtype=np.float64)
    hi = np.asarray(box_max, dtype=np.float64)
    inside = np.all((pos >= lo) & (pos <= hi), axis=1)

    # Episode segments = runs of identical bundle names (bundles are
    # concatenated in sorted order, one episode per bundle).
    _, starts = np.unique(np.asarray(names), return_index=True)
    n_rows = int(data["s"].shape[0])
    starts = sorted(starts.tolist()) + [n_rows]

    def first_latch(seg: np.ndarray) -> int | None:
        if len(seg) < latch_hits or not seg.any():
            return None
        for i in range(len(seg) - latch_hits + 1):
            if seg[i:i + latch_hits].all():
                return int(i)
        return None

    gate = np.zeros(n_rows, dtype=bool)
    n_latched = n_fallback = 0
    for k in range(len(starts) - 1):
        lo_i, hi_i = starts[k], starts[k + 1]
        start = first_latch(inside[lo_i:hi_i])
        if start is None:
            tail = max(fallback_tail, 1)
            gate[hi_i - tail:hi_i] = True
            n_fallback += 1
        else:
            gate[lo_i + start:hi_i] = True
            n_latched += 1

    n_episodes = len(starts) - 1
    stats = {
        "n_episodes": n_episodes, "n_rows": n_rows,
        "latched_eps": n_latched, "fallback_eps": n_fallback,
        "gate_row_share": round(float(gate.mean()), 4),
        "inside_row_share": round(float(inside.mean()), 4),
        "box_min": list(map(float, box_min)), "box_max": list(map(float, box_max)),
        "latch_hits": int(latch_hits), "fallback_tail_chunks": int(fallback_tail),
    }
    return gate, stats


# ─────────────────────────────────────────────────────────────────────
# Model + recipe (verbatim port of /tmp/rlt_stage12_libero_v2.py)
# ─────────────────────────────────────────────────────────────────────

def build_rlt_cfg(stage1_ckpt: Path, ref_dropout: float,
                  residual_scale: float = RESIDUAL_SCALE):
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import (
        SmolVLARLTConfig)
    ck = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    payload = dict(ck["rlt_config"])
    payload.pop("expert_hidden_dim", None)
    payload.pop("rlt_max_reconstruction_tokens", None)
    # Tuned-recipe overrides (vla-rlt-main tuned yaml), on top of the
    # real-robot embodiment dimensions (z=960, state=7, action=7, chunk=10).
    payload.update(
        mode="online_rl",
        vlm_hidden_dim=int(payload.get("vlm_hidden_dim", 960)),
        rlt_hidden_dim=int(payload.get("rlt_hidden_dim", 960)),
        rlt_num_heads=int(payload.get("rlt_num_heads", 8)),
        state_dim=int(payload.get("state_dim", 7)),
        action_dim=int(payload.get("action_dim", 7)),
        n_action_steps_rl=int(payload.get("n_action_steps_rl", CHUNK)),
        actor_output_mode="residual",
        actor_residual_scale=float(residual_scale),
        actor_input_order="z_state_ref",
        modality_projection_dim=0,
        ref_dropout=ref_dropout,
        bc_weight=BC_W,
        discount=GAMMA,
        target_tau=TAU,
        actor_hidden_dims=[256, 256],
        actor_activation="silu",
        critic_hidden_dims=[256, 256],
        critic_activation="silu",
        critic_layer_norm=False,
    )
    return SmolVLARLTConfig(**payload), ck


def compose_action(ref, c, residual_scale):
    """Executed action — literal RESIDUAL branch of compose_actor_action.

    Uses the exact deployment function so training and the online
    rlt_policy_server compose identically (train==deploy by construction).
    ref + (ACTION_SCALE * residual_scale) * tanh(c), clamped to ±ACTION_SCALE.
    """
    from cobot_bridge.rlt_stage2_data import (
        RESIDUAL_ACTOR_OUTPUT_SEMANTICS, compose_actor_action)
    return compose_actor_action(
        c, ref, ACTION_SCALE,
        output_semantics=RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
        residual_scale=residual_scale)


def mix_batch(buf: StaticBuffer, pool: StaticBuffer, batch_size: int,
              suf_frac: float, rng):
    n_succ = int(round(batch_size * suf_frac))
    n_fail = batch_size - n_succ
    if buf.n >= 1:
        idx_f = rng.integers(0, buf.n, n_fail)
        o = {k: getattr(buf, k)[idx_f] for k in
             ("z", "s", "ref", "a", "r", "dn", "zn", "sn", "rfn", "gate")}
    else:
        o = None
    idx_s = rng.integers(0, pool.n, n_succ)
    p = {k: getattr(pool, k)[idx_s] for k in
         ("z", "s", "ref", "a", "r", "dn", "zn", "sn", "rfn", "gate")}
    if o is None:
        return p
    out = {k: np.concatenate([o[k], p[k]]) for k in o}
    perm = rng.permutation(batch_size)
    return {k: v[perm] for k, v in out.items()}


def t_(x, device):
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def td3_update(rlt, buf, pool, opt_a, opt_c, rng, do_actor, args, device):
    m = dict(lc=0.0, la=0.0, q=0.0)
    for u in range(args.utd):
        b = mix_batch(buf, pool, args.batch_size, args.suf_frac, rng)
        z, s = t_(b["z"], device), t_(b["s"], device)
        ref = t_(b["ref"], device).reshape(-1, CHUNK, 7)
        a = t_(b["a"], device).reshape(-1, CHUNK, 7)
        r = t_(b["r"], device).unsqueeze(-1)
        dn = t_(b["dn"], device).unsqueeze(-1)
        zn, sn = t_(b["zn"], device), t_(b["sn"], device)
        rfn = t_(b["rfn"], device).reshape(-1, CHUNK, 7)
        with torch.no_grad():
            a2, _ = rlt.actor(zn, sn, rfn, training=False)
            qt = rlt.critic_target.q_min(
                zn, sn, compose_action(rfn, a2, args.residual_scale))
            if args.q_target_clip > 0:
                qt = torch.clamp(qt, 0.0, args.q_target_clip)
            # Each row is one fully-executed 10-frame chunk, so bootstrap
            # with gamma ** executed_steps (repo: bootstrap_discount =
            # gamma ** executed_steps), not per-row gamma.
            y = r + (args.gamma ** args.exec_steps) * (1.0 - dn) * qt
        q1, q2 = rlt.critic(z, s, a)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_c.zero_grad()
        lc.backward()
        opt_c.step()
        if do_actor and u % CRITIC_PER_ACTOR == 0:
            am, _ = rlt.actor(z, s, ref, training=True)
            af = compose_action(ref, am, args.residual_scale)
            # Placement-phase gate: only rows inside the episode's
            # placement phase (EEF latched in the insertion box) may
            # receive Q-max-driven deviation; all rows keep the
            # BC-to-reference pull (zero correction outside the gate).
            g = t_(b["gate"], device).float()
            q_min = rlt.critic.q_min(z.detach(), s, af)
            la = (-(q_min * g).sum() / g.sum().clamp(min=1.0)
                  + args.bc_weight * F.mse_loss(af, ref))
            opt_a.zero_grad()
            la.backward()
            opt_a.step()
            m["la"] += float(la)
        m["lc"] += float(lc)
        m["q"] += float(y.mean())
    for p, tp in zip(rlt.critic.parameters(), rlt.critic_target.parameters()):
        tp.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)
    return {k: v / args.utd for k, v in m.items()}


@torch.no_grad()
def eval_metrics(rlt, buf, idx, device, args):
    z = t_(buf.z[idx], device)
    s = t_(buf.s[idx], device)
    ref = t_(buf.ref[idx], device).reshape(-1, CHUNK, 7)
    am, _ = rlt.actor(z, s, ref, training=False)
    af = compose_action(ref, am, args.residual_scale)
    corr = (af - ref).abs()
    q_a = rlt.critic.q_min(z, s, af)
    q_r = rlt.critic.q_min(z, s, ref)
    # Offline proxy of the tuned recipe's auto-intervention gate
    # (chunk rms / first-action rms / max-abs thresholds; episode-level
    # latching cannot be applied to disjoint offline rows, so this is the
    # per-row takeover probability).
    dev = af - ref
    chunk_rms = dev.pow(2).mean(dim=(1, 2)).sqrt()
    first_rms = dev[:, 0, :].pow(2).mean(dim=1).sqrt()
    max_abs = dev.abs().flatten(1).max(dim=1).values
    intervene = (
        (chunk_rms > INTERV_CHUNK_RMS)
        | (first_rms > INTERV_FIRST_RMS)
        | (max_abs > INTERV_MAX_ABS)
    )
    # Placement-phase split (gate = True rows): the retrained actor must
    # stay at ~zero correction before the episode's placement start and
    # keep its learned correction during placement.
    gm = t_(buf.gate[idx], device)
    pre = ~gm
    post = gm
    corr_pre = corr[pre]
    corr_post = corr[post]
    return {
        "corr_rms": float(corr.pow(2).mean().sqrt()),
        "corr_max": float(corr.max()),
        "corr_rms_pre": float(corr_pre.pow(2).mean().sqrt()) if pre.any() else 0.0,
        "corr_rms_post": float(corr_post.pow(2).mean().sqrt()) if post.any() else 0.0,
        "q_actor": float(q_a.mean()),
        "q_ref": float(q_r.mean()),
        "q_margin": float((q_a - q_r).mean()),
        "intervention_rate": float(intervene.float().mean()),
    }


def rank_auc(scores_pos, scores_neg):
    """One-vs-rest rank AUC from per-row Q scores."""
    pos = np.asarray(scores_pos, dtype=np.float64)
    neg = np.asarray(scores_neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    r_pos = ranks[: pos.size]
    return float((r_pos.sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parser().parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    cfg, stage1_ckpt = build_rlt_cfg(args.stage1_checkpoint, args.ref_dropout,
                                args.residual_scale)
    print(f"[cfg] z={cfg.rlt_hidden_dim} s={cfg.state_dim} a={cfg.action_dim} "
          f"chunk={cfg.n_action_steps_rl} lr_a={args.actor_lr} "
          f"lr_c={args.critic_lr} gamma={args.gamma} bc={args.bc_weight} "
          f"res_scale={args.residual_scale} ref_do={cfg.ref_dropout}", flush=True)

    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (
        SmolVLARLTPolicy)
    rlt = SmolVLARLTPolicy(cfg).to(device)
    rlt.actor.train()
    rlt.critic.train()
    opt_a = torch.optim.Adam(rlt.actor.parameters(), lr=args.actor_lr)
    opt_c = torch.optim.Adam(rlt.critic.parameters(), lr=args.critic_lr)

    data, names = load_replay(args.replay_root)
    if not np.isfinite(data["z"]).all() or not np.isfinite(data["a"]).all():
        raise ValueError("non-finite values in replay data")
    n = data["z"].shape[0]
    # Placement-phase gate: the real-robot analog of the sim's
    # actor_start_step rollout gate (vla-rlt-main trainer.py: pure VLA
    # before the per-episode start step, actor residual from the start to
    # episode end).  On the real robot the start is the per-episode
    # placement onset (EEF latch-in-box; measured median ~step 300,
    # p25=140 / p75=470, too variable for a fixed step index).
    if args.no_gate:
        data["gate"] = np.ones(n, dtype=bool)
        gate_stats = {"disabled": True}
    else:
        data["gate"], gate_stats = compute_placement_gate(
            data, names, args.stage1_checkpoint,
            args.gate_box_min, args.gate_box_max,
            args.gate_latch_hits, args.gate_fallback_tail)
        print(f"[gate] {json.dumps(gate_stats)}", flush=True)
    # Buffer: all robot rows. Pool: success rows + demo rows.
    demo_mask = np.char.startswith(names, "il_demo")
    succ_mask = data["success_row"] > 0.5
    buf_mask = ~demo_mask
    pool_mask = (~demo_mask & succ_mask) | demo_mask
    print(f"[data] buffer rows={int(buf_mask.sum())} pool rows={int(pool_mask.sum())}"
          f" (success-robot={int((buf_mask & succ_mask).sum())}, "
          f"demo={int(demo_mask.sum())})", flush=True)

    def sub(mask):
        return StaticBuffer(
            data["z"][mask], data["s"][mask], data["ref"][mask], data["a"][mask],
            data["r"][mask], data["dn"][mask], data["zn"][mask],
            data["sn"][mask], data["rfn"][mask], data["success_row"][mask],
            data["gate"][mask])

    buf = sub(buf_mask)
    pool = sub(pool_mask)
    if buf.n == 0 or pool.n == 0:
        raise ValueError("empty buffer or pool")

    n_iters = args.iters or args.utd * buf.n
    if args.smoke:
        n_iters = min(n_iters, 1000)
    n_iters = min(n_iters, args.max_iters)
    print(f"[train] iters={n_iters} utd={args.utd} batch={args.batch_size} "
          f"warmup={args.warmup_iters} -> {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed eval slice from the buffer.
    eval_idx = rng.choice(buf.n, size=min(EVAL_SLICE, buf.n), replace=False)
    order = np.argsort(eval_idx)
    eval_idx = np.sort(eval_idx)

    # Row -> buffer index mapping for AUC (success vs failure rows).
    buf_row_success = data["success_row"][buf_mask]

    history = []
    t0 = time.time()
    ckpt_every = 0 if args.smoke else args.ckpt_freq
    for it in range(n_iters):
        do_actor = it >= args.warmup_iters
        m = td3_update(rlt, buf, pool, opt_a, opt_c, rng, do_actor, args, device)
        if (it + 1) % args.log_freq == 0 or it == 0:
            em = eval_metrics(rlt, buf, eval_idx, device, args)
            row = {
                "iter": it + 1,
                "lc": round(m["lc"], 5),
                "la": round(m["la"], 5) if m["la"] else None,
                "q": round(m["q"], 5),
                **{k: round(v, 5) for k, v in em.items()},
                "t_s": round(time.time() - t0, 1),
            }
            history.append(row)
            with (args.output_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{it+1}/{n_iters}] lc={m['lc']:.4f} "
                  f"la={m['la'] if m['la'] is None else format(m['la'], '.4f')} "
                  f"q={m['q']:.4f} corr_rms={em['corr_rms']:.5f} "
                  f"q_margin={em['q_margin']:.5f} "
                  f"({row['t_s']}s)", flush=True)
        if ckpt_every and (it + 1) % ckpt_every == 0:
            save_checkpoint(args, rlt, opt_a, opt_c, it + 1, history, buf, pool,
                            name=f"iter_{it+1:07d}")
    # Final evaluation + checkpoint.
    with torch.no_grad():
        z = t_(buf.z[eval_idx], device)
        s = t_(buf.s[eval_idx], device)
        ref = t_(buf.ref[eval_idx], device).reshape(-1, CHUNK, 7)
        am, _ = rlt.actor(z, s, ref, training=False)
        af = compose_action(ref, am, args.residual_scale)
        q_scores = rlt.critic.q_min(z, s, ref)
    q_ref_rows = q_scores.flatten().cpu().numpy()
    # AUC on a fixed eval slice (success rows vs failure rows within buffer).
    eval_success = buf_row_success[eval_idx] > 0.5
    auc = rank_auc(q_ref_rows[eval_success], q_ref_rows[~eval_success])
    em = eval_metrics(rlt, buf, eval_idx, device, args)
    final = {
        "iter": n_iters,
        "buffer_rows": buf.n,
        "pool_rows": pool.n,
        "success_rows": int(buf_row_success.sum()),
        "final": em,
        "critic_auc_success_vs_failure": round(auc, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "n_iters": n_iters,
    }
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2) + "\n")
    print(f"[final] {json.dumps(final)}", flush=True)
    save_checkpoint(args, rlt, opt_a, opt_c, n_iters, history, buf, pool,
                    name="final")


def save_checkpoint(args, rlt, opt_a, opt_c, iteration, history, buf, pool,
                    name="latest"):
    payload = {
        "format_version": 1,
        "method": "offline_tuned_td3_v2_placement_gated",
        "method_source": ("172.16.128.24:/workspace/vla-rlt-main "
                          "(scripts/train_libero.py + "
                          "configs/tasks/libero_pick_place_chunk10_exec10_tuned.yaml, "
                          "README.md); actor_start_step ported as the per-row "
                          "placement-phase gate"),
        "iteration": int(iteration),
        "hyperparams": {
            "batch_size": args.batch_size, "utd": args.utd,
            "gamma": args.gamma, "tau": args.tau, "bc_weight": args.bc_weight,
            "residual_scale": args.residual_scale,
            "action_scale": list(ACTION_SCALE), "suf_frac": args.suf_frac,
            "ref_dropout": args.ref_dropout, "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr, "warmup_iters": args.warmup_iters,
            "q_target_clip": args.q_target_clip, "exec_steps": args.exec_steps,
            "seed": args.seed, "chunk": CHUNK,
            "placement_gate": (
                {"disabled": True} if getattr(args, "no_gate", False) else {
                    "box_min": [float(x) for x in args.gate_box_min],
                    "box_max": [float(x) for x in args.gate_box_max],
                    "latch_hits": int(args.gate_latch_hits),
                    "fallback_tail_chunks": int(args.gate_fallback_tail),
                    "gated_row_share": round(float(buf.gate.mean()), 4),
                }),
        },
        "dataset": {
            "replay_root": str(args.replay_root.resolve()),
            "buffer_rows": buf.n, "pool_rows": pool.n,
            "success_rows": int(buf.success_row.sum()),
            "stage1_checkpoint": str(args.stage1_checkpoint),
        },
        # ── Deployment contract (rlt_policy_server._load_stage2) ──
        # stage / actor_output_semantics / action_chunk_timebase /
        # stage1_checkpoint / rlt_config / actor_state_dict / critic_state_dict
        # / total_updates / actor_action_scale are the keys the online
        # runtime validates before loading this actor.
        "stage": "rlt_stage2",
        "actor_output_semantics": RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
        "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
        "stage1_checkpoint": str(
            args.stage1_checkpoint.expanduser().resolve()),
        "total_updates": int(iteration),
        "actor_action_scale": [float(x) for x in ACTION_SCALE],
        "actor_residual_scale": float(args.residual_scale),
        "ready_for_online": False,  # no online no-regression gate run yet
        # Dashboard rollout-start gate (web_dashboard.py): a non-read-only
        # actor payload without actor_init_checkpoint -> 409 "no IL validation
        # cache".  Point at the actor-BC init of this lineage so the
        # post-episode pipeline's IL validation cache is found.
        "actor_init_checkpoint": (
            str(args.actor_init_checkpoint.expanduser().resolve())
            if getattr(args, "actor_init_checkpoint", None) else None),
        "rlt_config": {k: v for k, v in vars(rlt.config).items()
                       if not k.startswith("_")
                       and k not in ("input_features", "output_features")},
        "config": {k: v for k, v in vars(rlt.config).items()
                   if not k.startswith("_")
                   and k not in ("input_features", "output_features")},
        "actor_state_dict": rlt.actor.state_dict(),
        "critic_state_dict": rlt.critic.state_dict(),
        "critic_target_state_dict": rlt.critic_target.state_dict(),
        "actor_optimizer_state_dict": opt_a.state_dict(),
        "critic_optimizer_state_dict": opt_c.state_dict(),
        "metrics_tail": history[-50:],
    }
    path = args.output_dir / f"checkpoint_{name}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[ckpt] {path.name} (iter {iteration})", flush=True)


if __name__ == "__main__":
    main()
