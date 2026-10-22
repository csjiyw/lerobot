#!/usr/bin/env python3
"""Offline evaluation of the deployed composed policy (actor_v1) on the 5 episodes."""
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"
import json, sys
import numpy as np, torch, torch.nn as nn

DATA = sys.argv[2] if len(sys.argv) > 2 else str(OUTPUTS / "actor_data/workpiece3_full.npz")
D = np.load(DATA)
z, st, ref, act, human, ep = D["z"], D["state"], D["ref"], D["action"], D["human"], D["episode"]
ACTOR = sys.argv[1] if len(sys.argv) > 1 else str(OUTPUTS / "actor_v1")
cfg = json.load(open(f"{ACTOR}/config.json"))
cap = torch.tensor(np.asarray(cfg["cap"], dtype=np.float32))
ck = torch.load(f"{ACTOR}/actor.pt", map_location="cpu")

class Actor(nn.Module):
    def __init__(self, in_dim=409):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 210))
    def forward(self, x): return self.net(x)

m = Actor(int(ck.get("in_dim", cfg.get("in_dim", 409)))); m.load_state_dict(ck["model"]); m.eval()
Xnp = np.concatenate([z, st, ref.reshape(len(z), -1)], 1).astype(np.float32)
if cfg.get("standardize"):
    Xnp = (Xnp - np.asarray(cfg["mu"], dtype=np.float32)) / np.asarray(cfg["sd"], dtype=np.float32)
X = torch.tensor(Xnp)
with torch.no_grad():
    corr = cap * torch.tanh(m(X).reshape(-1, 30, 7))
pred = torch.tensor(ref) + corr
act_t = torch.tensor(act)
hmask = torch.tensor(human > 0.5)

print(f"{'ep':>3} {'n_h':>4} {'ref_xyz':>8} {'new_xyz':>8} {'chg':>7} {'pol_corr':>9} {'sat%':>6}")
tot = []
for e in sorted(np.unique(ep)):
    sel = torch.tensor(ep == e)
    h = sel & hmask; p = sel & ~hmask
    r = (torch.tensor(ref)[h] - act_t[h])[..., :3]
    n = (pred[h] - act_t[h])[..., :3]
    a = r.pow(2).mean().sqrt().item(); b = n.pow(2).mean().sqrt().item()
    pc = corr[p].abs().mean().item() if p.any() else float("nan")
    sat = (corr[sel].abs() >= 0.999 * cap).float().mean().item()
    print(f"{int(e):3d} {int(h.sum()):4d} {a:8.4f} {b:8.4f} {100*(b/a-1):+6.1f}% {pc:9.4f} {sat*100:5.1f}")
    tot.append((a, b, int(h.sum())))

# truncation: error over the first N executed steps (human rows)
print("\ntruncation (human rows, RMS xyz):")
for n in (5, 10, 15, 20, 30):
    r = ((torch.tensor(ref)[hmask] - act_t[hmask])[:, :n, :3]).pow(2).mean().sqrt().item()
    q = ((pred[hmask] - act_t[hmask])[:, :n, :3]).pow(2).mean().sqrt().item()
    print(f"  first {n:2d} steps: ref {r:.4f} -> composed {q:.4f} ({100*(q/r-1):+.1f}%)")
