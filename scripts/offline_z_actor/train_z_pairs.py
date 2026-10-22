#!/usr/bin/env python3
"""Stage-1 z with temporal smoothness on consecutive-frame pairs.

Loss = EEF-delta + gripper aux on frame a  +  lambda * ||z(a+1) - z(a)||^2.
Only the current-environment dataset is used (it provides consecutive frames).
"""
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.relative_eef import rotvec_to_matrix, matrix_to_rotvec
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_z_stage1 import ZEncoder, ZHeads


def egocentric_delta(state_t, state_k):
    out = torch.zeros((state_k.shape[0], 6), dtype=torch.float32)
    for i in range(state_k.shape[0]):
        st = state_t[i].detach().cpu().numpy().astype(np.float64)
        sk = state_k[i].detach().cpu().numpy().astype(np.float64)
        R = rotvec_to_matrix(st[3:6])
        out[i, :3] = torch.from_numpy((R.T @ (sk[:3] - st[:3])).astype(np.float32))
        dR = R.T @ rotvec_to_matrix(sk[3:6])
        out[i, 3:] = torch.from_numpy(matrix_to_rotvec(dR).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smolvla-path", required=True)
    ap.add_argument("--dataset", required=True, help="repo_id:root")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--smooth-weight", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=5, help="pair start stride")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()
    dev = torch.device(args.device)
    repo, root = args.dataset.rsplit(":", 1)
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)

    pol = SmolVLAPolicy.from_pretrained(args.smolvla_path, local_files_only=True).to(dev).eval()
    for p in pol.parameters(): p.requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=args.smolvla_path)
    vla = pol.model
    vlm_dim = int(vla.vlm_with_expert.config.text_config.hidden_size)

    ds = LeRobotDataset(repo, root=root, delta_timestamps={"action": [i / 30 for i in range(30)]},
                        video_backend="pyav")
    starts = np.arange(0, len(ds) - 1, args.stride)
    rng = np.random.default_rng(0)
    order = rng.permutation(starts)
    A = Subset(ds, order.tolist()); B = Subset(ds, (order + 1).tolist())
    la = DataLoader(A, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    lb = DataLoader(B, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    enc = ZEncoder(vlm_dim, z_dim=args.z_dim, d_model=args.d_model).to(dev)
    heads = ZHeads(z_dim=args.z_dim).to(dev)
    params = list(enc.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    def encode(batch):
        batch = pre(batch)
        imgs, masks = pol.prepare_images(batch)
        state = pol.prepare_state(batch)
        prefix, pmask = vla.extract_rlt_prefix(imgs, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state)
        return enc(prefix, pmask), batch

    it_a = iter(la); it_b = iter(lb)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            ba = next(it_a); bb = next(it_b)
        except StopIteration:
            it_a = iter(la); it_b = iter(lb); ba = next(it_a); bb = next(it_b)
        raw_a = ba[ACTION].float().clone(); raw_st = ba["observation.state"].float().clone()
        with torch.no_grad():
            za, _ = encode(ba)
        zb, _ = encode(bb)
        pred = heads(za)
        target15 = egocentric_delta(raw_st[:, :7], raw_a[:, 15, :7]).to(dev)
        target30 = egocentric_delta(raw_st[:, :7], raw_a[:, 29, :7]).to(dev)
        grip = raw_a[:, 0, 6].to(dev).unsqueeze(-1)
        loss = (F.mse_loss(pred["d15"], target15) + F.mse_loss(pred["d30"], target30)
                + 0.5 * F.mse_loss(pred["grip"], grip)
                + args.smooth_weight * F.mse_loss(zb, za.detach()))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if step % args.log_every == 0:
            print(f"step {step}/{args.steps} loss {float(loss):.4f} smooth {float(F.mse_loss(zb, za.detach())):.4f} {step/(time.time()-t0):.1f} it/s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"encoder": enc.state_dict(), "heads": heads.state_dict()}, save / "z_encoder.pt")
            (save / "config.json").write_text(json.dumps({"vlm_dim": vlm_dim, "z_dim": args.z_dim, "d_model": args.d_model,
                                                          "step": step, "smooth_weight": args.smooth_weight,
                                                          "smolvla_path": args.smolvla_path}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
