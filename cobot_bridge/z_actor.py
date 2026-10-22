"""Stage-1 z encoder + residual actor used by the composed policy server."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZEncoder(nn.Module):
    def __init__(self, vlm_dim: int, z_dim: int = 192, d_model: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_dim, d_model)
        self.type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                           batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.z_head = nn.Linear(d_model, z_dim)
        self.z_dim = z_dim

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


class ResidualActor(nn.Module):
    def __init__(self, in_dim: int = 409, out_dim: int = 210):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                                 nn.Linear(512, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_z_encoder(ckpt: str, config: str) -> ZEncoder:
    cfg = json.loads(Path(config).read_text())
    enc = ZEncoder(cfg["vlm_dim"], z_dim=cfg["z_dim"], d_model=cfg["d_model"])
    enc.load_state_dict(torch.load(ckpt, map_location="cpu")["encoder"])
    enc.eval()
    return enc


def load_actor(actor_pt: str, config: str) -> tuple[ResidualActor, np.ndarray, dict]:
    cfg = json.loads(Path(config).read_text())
    payload = torch.load(actor_pt, map_location="cpu")
    in_dim = int(payload.get("in_dim", cfg.get("in_dim", 192 + 7 + 210)))
    actor = ResidualActor(in_dim, 210)
    actor.load_state_dict(payload["model"])
    actor.eval()
    stats = {}
    if cfg.get("standardize"):
        stats = {"mu": np.asarray(cfg["mu"], dtype=np.float32), "sd": np.asarray(cfg["sd"], dtype=np.float32)}
    return actor, np.asarray(cfg["cap"], dtype=np.float32), stats
