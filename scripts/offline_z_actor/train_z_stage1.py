#!/usr/bin/env python3
"""Stage-1 replacement: task-relevant ``z`` from a frozen SmolVLA prefix.

Design (2026-10-21):
  * the frozen VLA (default: the 60k phase-1 checkpoint) produces prefix
    embeddings; a small attention encoder pools them into ``z`` (low-dim,
    LayerNorm);
  * ``z`` is trained with auxiliary heads that matter for the residual actor:
      - future EEF delta at t+15 and t+30 (xyz + rotvec)
      - gripper command at t
    (no prefix reconstruction: it would waste capacity on task-irrelevant
    variance);
  * the data loader mixes one or more LeRobot datasets with fixed ratios
    (e.g. the 5 current-environment episodes up-weighted against the 498
    historical episodes).

Output: ``z_encoder.pt`` with the encoder state dict and a config describing
the z contract for stage-2.
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("train_z_stage1")


class ZEncoder(nn.Module):
    """Attention-pooled z from frozen VLA prefix embeddings."""

    def __init__(self, vlm_dim: int, z_dim: int = 192, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_dim, d_model)
        self.type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.z_head = nn.Linear(d_model, z_dim)
        self.z_dim = z_dim
        self.d_model = d_model

    def forward(self, vlm_emb: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.norm_in(self.vlm_proj(vlm_emb.float())) + self.type_emb[0]
        out = self.encoder(tokens)
        if pad_mask is not None:
            m = pad_mask.float().unsqueeze(-1)
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = out.mean(dim=1)
        z = self.z_head(self.norm(pooled))
        return F.layer_norm(z, (self.z_dim,))


class ZHeads(nn.Module):
    def __init__(self, z_dim: int = 192, rot_dim: int = 3):
        super().__init__()
        self.d15 = nn.Linear(z_dim, 3 + rot_dim)
        self.d30 = nn.Linear(z_dim, 3 + rot_dim)
        self.grip = nn.Linear(z_dim, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"d15": self.d15(z), "d30": self.d30(z), "grip": self.grip(z)}


def _make_dataset(root: Path, repo_id: str, fps: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Only the action chunk is time-stretched: the preprocessor's relative-EEF
    # step expects a single (B, 7) state, so future states are derived from the
    # raw absolute action chunk instead.
    deltas = {"action": [i / fps for i in range(30)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=deltas, video_backend="pyav")


def _rotvec_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation difference in rotvec space (approximation, small angles)."""
    return (b - a).astype(np.float32)


def _egocentric_delta(state_t: torch.Tensor, state_k: torch.Tensor) -> torch.Tensor:
    """EEF delta in the frame of state_t: xyz rotated by R(state_t)^T plus rotvec delta."""
    from lerobot.policies.smolvla.relative_eef import rotvec_to_matrix, matrix_to_rotvec

    out = torch.zeros((state_k.shape[0], 6), dtype=torch.float32)
    for i in range(state_k.shape[0]):
        st = state_t[i].detach().cpu().numpy().astype(np.float64)
        sk = state_k[i].detach().cpu().numpy().astype(np.float64)
        R = rotvec_to_matrix(st[3:6])
        out[i, :3] = torch.from_numpy((R.T @ (sk[:3] - st[:3])).astype(np.float32))
        dR = R.T @ rotvec_to_matrix(sk[3:6])
        out[i, 3:] = torch.from_numpy(matrix_to_rotvec(dR).astype(np.float32))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smolvla-path", required=True)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="triples: <repo_id>:<root>:<batch_fraction>")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--z-dim", type=int, default=192)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eef-weight", type=float, default=1.0)
    ap.add_argument("--grip-weight", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true", help="one tiny batch, no video decode")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── frozen VLA (60k) + its inherited processor ──────────────────────
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    LOG.info("loading frozen SmolVLA from %s", args.smolvla_path)
    policy = SmolVLAPolicy.from_pretrained(args.smolvla_path, local_files_only=True).to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    vla = policy.model
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.smolvla_path
    )

    # ── datasets ───────────────────────────────────────────────────────
    specs = []
    for item in args.datasets:
        repo, root, frac = item.rsplit(":", 2)
        specs.append((repo, Path(root), float(frac)))
    total_frac = sum(f for _, _, f in specs)
    fps = 30
    datasets, loaders = [], []
    for repo, root, frac in specs:
        ds = _make_dataset(root, repo, fps)
        bs = max(1, round(args.batch_size * frac / total_frac))
        loaders.append([iter(DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)), ds, bs])
        datasets.append(ds)
        LOG.info("dataset %s: %d frames, %d episodes (batch %d)", repo, len(ds), ds.num_episodes, bs)

    def next_batch() -> dict:
        merged: dict[str, list] = {}
        for i, (it, ds, bs) in enumerate(loaders):
            try:
                b = next(it)
            except StopIteration:
                it = iter(DataLoader(ds, batch_size=bs, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True, drop_last=True))
                loaders[i] = [it, ds, bs]
                b = next(it)
            for k, v in b.items():
                merged.setdefault(k, [])
                merged[k].append(v)
        out = {}
        for k, vs in merged.items():
            if isinstance(vs[0], list):
                out[k] = [x for v in vs for x in v]
            else:
                out[k] = torch.cat(vs, dim=0)
        return out

    # ── model ──────────────────────────────────────────────────────────
    vlm_dim = int(vla.vlm_with_expert.config.text_config.hidden_size)
    LOG.info("vlm_dim=%d", vlm_dim)
    enc = ZEncoder(vlm_dim, z_dim=args.z_dim, d_model=args.d_model).to(device)
    heads = ZHeads(z_dim=args.z_dim).to(device)
    params = list(enc.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    def step_loss(batch):
        # raw absolute action chunk / state must be captured before the
        # processor turns actions into chunk-relative ones.
        raw_action = batch[ACTION].float().clone()
        raw_state = batch["observation.state"].float().clone()
        batch = preprocessor(batch)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)                 # (B, max_state_dim)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        with torch.no_grad():
            prefix_out, prefix_mask = vla.extract_rlt_prefix(
                images, img_masks, lang_tokens, lang_masks, state
            )
        z = enc(prefix_out, prefix_mask)
        pred = heads(z)

        st = raw_state[:, :7].to(device)
        target15 = _egocentric_delta(st, raw_action[:, 15, :7].to(device)).to(device)
        target30 = _egocentric_delta(st, raw_action[:, 29, :7].to(device)).to(device)
        grip = raw_action[:, 0, 6].to(device).unsqueeze(-1)
        loss_d15 = F.mse_loss(pred["d15"], target15)
        loss_d30 = F.mse_loss(pred["d30"], target30)
        loss_grip = F.mse_loss(pred["grip"], grip)
        loss = args.eef_weight * (loss_d15 + loss_d30) + args.grip_weight * loss_grip
        return loss, {"d15": loss_d15.item(), "d30": loss_d30.item(), "grip": loss_grip.item()}, z

    if args.dry_run:
        b = next_batch()
        loss, parts, z = step_loss(b)
        loss.backward()
        LOG.info("dry-run ok: z=%s loss=%.4f parts=%s", tuple(z.shape), float(loss), parts)
        return

    LOG.info("training z for %d steps", args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        loss, parts, _ = step_loss(next_batch())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            LOG.info("step %d/%d loss %.4f (d15 %.4f d30 %.4f grip %.4f) %.1f it/s",
                     step, args.steps, float(loss), parts["d15"], parts["d30"], parts["grip"],
                     step / (time.time() - t0))
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"encoder": enc.state_dict(), "heads": heads.state_dict()}, save_dir / "z_encoder.pt")
            (save_dir / "config.json").write_text(json.dumps({
                "vlm_dim": vlm_dim, "z_dim": args.z_dim,
                "d_model": args.d_model, "step": step, "smolvla_path": args.smolvla_path,
                "heads": ["d15", "d30", "grip"],
            }, indent=2))
    LOG.info("done")


if __name__ == "__main__":
    main()
