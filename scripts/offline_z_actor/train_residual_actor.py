#!/usr/bin/env python3
"""Residual actor: action = ref + cap * tanh(MLP(z, state, ref)).

BC targets: policy rows -> ref (zero correction), human rows -> human action.
No critic (q_weight = 0 per the current RLT evidence).
"""
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--human-weight", type=float, default=5.0)
    ap.add_argument("--cap-frac", type=float, default=0.5, help="cap = cap_frac * action std")
    ap.add_argument("--val-episode", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--drop-z", action="store_true", help="ablation: actor input = state + ref only")
    ap.add_argument("--standardize", action="store_true", help="per-dim standardize z/state/ref using train statistics")
    args = ap.parse_args()

    dev = torch.device(args.device)
    d = np.load(args.data)
    z, st, ref, act, human, ep = d["z"], d["state"], d["ref"], d["action"], d["human"], d["episode"]
    std = act.reshape(-1, 7).std(axis=0).astype(np.float32)
    cap = torch.tensor(args.cap_frac * std, device=dev)
    val = ep == args.val_episode
    tr = ~val
    print(f"samples train={tr.sum()} val={val.sum()} human_frac={human.mean():.2f} cap={np.round(args.cap_frac*std,3)}")

    class Actor(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.SiLU(),
                                     nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, out_dim))
        def forward(self, x):
            return self.net(x)

    def feats(idx):
        parts = [st[idx], ref[idx].reshape(len(idx), -1)]
        if not args.drop_z:
            parts.insert(0, z[idx])
        return np.concatenate(parts, 1)
    if args.standardize:
        tr_rows = np.where(ep != args.val_episode)[0]
        mu = X_all_train = np.concatenate([feats(tr_rows)], 1).mean(0)
        sd = np.concatenate([feats(tr_rows)], 1).std(0) + 1e-6
    else:
        mu = sd = None
    def prep(idx):
        x = feats(idx)
        if mu is not None:
            x = (x - mu) / sd
        return (torch.tensor(x, dtype=torch.float32, device=dev),
                torch.tensor(ref[idx], dtype=torch.float32, device=dev),
                torch.tensor(act[idx], dtype=torch.float32, device=dev),
                torch.tensor(human[idx], dtype=torch.float32, device=dev))
    tr_idx = np.where(tr)[0]; val_idx = np.where(val)[0]
    in_dim = (0 if args.drop_z else int(z.shape[1])) + 7 + int(ref.shape[1] * ref.shape[2])
    print(f'actor input dim {in_dim}', flush=True)
    model = Actor(in_dim, 210).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(0)

    def loss_fn(x, r, a, h):
        corr = (cap * torch.tanh(model(x).reshape(-1, 30, 7)))
        pred = r + corr
        target = torch.where(h[:, None, None] > 0.5, a, r)
        w = torch.where(h > 0.5, args.human_weight, 1.0)[:, None, None]
        return (w * (pred - target).pow(2)).mean(), pred, corr

    for step in range(1, args.steps + 1):
        idx = rng.choice(tr_idx, args.batch, replace=True)
        x, r, a, h = prep(idx)
        loss, _, _ = loss_fn(x, r, a, h)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {float(loss):.6f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            with torch.no_grad():
                xv, rv, av, hv = prep(val_idx)
                _, pv, cv = loss_fn(xv, rv, av, hv)
                zero = (rv - torch.where(hv[:, None, None] > 0.5, av, rv)).pow(2)
                model_err = (pv - torch.where(hv[:, None, None] > 0.5, av, rv)).pow(2)
                mh = hv > 0.5
                print(f"VAL step {step}: human xyz err zero={float(zero[mh][:, :, :3].mean()**0.5):.4f} "
                      f"model={float(model_err[mh][:, :, :3].mean()**0.5):.4f} | "
                      f"policy corr_abs={float(cv[~mh].abs().mean()):.4f}", flush=True)
            Path(args.out).mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cap": (args.cap_frac * std).tolist(),
                        "step": step, "in_dim": (0 if args.drop_z else int(z.shape[1])) + 7 + 210}, Path(args.out) / "actor.pt")
            cfg_out = {"z_dim": int(z.shape[1]), "state_dim": 7, "chunk": 30,
                       "in_dim": (0 if args.drop_z else int(z.shape[1])) + 7 + 210,
                       "cap": (args.cap_frac * std).tolist(),
                       "human_weight": args.human_weight, "step": step}
            if mu is not None:
                cfg_out["standardize"] = True
                cfg_out["mu"] = [float(v) for v in mu]
                cfg_out["sd"] = [float(v) for v in sd]
            (Path(args.out) / "config.json").write_text(json.dumps(cfg_out, indent=2))

if __name__ == "__main__":
    main()
