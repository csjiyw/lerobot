"""Verify an offline-trained RLT actor checkpoint against the deployment contract.

Simulates rlt_policy_server's stage-2 loading path (validation keys, strict
state_dict loads, train==deploy action-composition parity) without touching
the robot or the bridge.

Usage:
  python -m cobot_bridge.rlt_verify_deploy_load \
      --checkpoint .../checkpoint_final.pt \
      --replay-root .../replay/<root> \
      [--device cuda:0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cobot_bridge.rlt_stage2_data import (  # noqa: E402
    ACTION_CHUNK_TIMEBASE,
    RESIDUAL_ACTOR_OUTPUT_SEMANTICS,
    SUPPORTED_ACTOR_CHECKPOINT_STAGES,
    SUPPORTED_ACTOR_OUTPUT_SEMANTICS,
    compose_actor_action,
    load_replay_bundle,
)

from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import (  # noqa: E402
    SmolVLARLTConfig)
from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (  # noqa: E402
    RLCriticMLP, RLActorMLP)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--replay-root", type=Path, required=True,
                   help="any bundle root with at least one .npz (for a real (z,s,ref) row)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print(f"[load] {args.checkpoint.name}")
    # ── contract keys (exact checks done by rlt_policy_server) ──
    check("stage supported", ck.get("stage") in SUPPORTED_ACTOR_CHECKPOINT_STAGES,
          f"stage={ck.get('stage')!r}")
    check("actor_output_semantics supported",
          ck.get("actor_output_semantics") in SUPPORTED_ACTOR_OUTPUT_SEMANTICS,
          f"semantics={ck.get('actor_output_semantics')!r}")
    check("action_chunk_timebase",
          ck.get("action_chunk_timebase") == ACTION_CHUNK_TIMEBASE,
          f"got={ck.get('action_chunk_timebase')!r}")
    s1 = ck.get("stage1_checkpoint")
    s1_resolved = Path(s1).expanduser().resolve() if s1 else None
    check("stage1_checkpoint exists", bool(s1_resolved and s1_resolved.is_file()),
          str(s1))
    check("rlt_config is dict", isinstance(ck.get("rlt_config"), dict))
    check("total_updates int", isinstance(ck.get("total_updates"), int),
          f"total_updates={ck.get('total_updates')}")
    scale = torch.as_tensor(ck.get("actor_action_scale", []), dtype=torch.float32)
    check("actor_action_scale (7,) finite > 0",
          scale.shape == (7,) and torch.isfinite(scale).all() and bool(torch.all(scale > 0)),
          f"scale={scale.tolist()}")

    # ── rebuild networks exactly like the server, strict loads ──
    payload = dict(ck["rlt_config"])
    payload["mode"] = "inference"
    actor_config = SmolVLARLTConfig(**payload)
    actor = RLActorMLP(actor_config).to(args.device)
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    check("actor strict load", True)
    critic = RLCriticMLP(actor_config).to(args.device)
    critic.load_state_dict(ck["critic_state_dict"], strict=True)
    critic.eval().requires_grad_(False)
    check("critic strict load", True)

    # ── parity: server-side compose on a real data row ──
    bundles = sorted(args.replay_root.glob("*.npz"))
    arrays, meta = load_replay_bundle(bundles[0])
    n = arrays["rewards"].shape[0]
    z = torch.tensor(arrays["z_rl"][0], dtype=torch.float32, device=args.device).unsqueeze(0)
    s = torch.tensor(arrays["state"][0], dtype=torch.float32, device=args.device).unsqueeze(0)
    action_steps = int(actor_config.n_action_steps_rl)
    ref = torch.tensor(arrays["ref_actions"][0], dtype=torch.float32,
                       device=args.device).view(1, action_steps, 7)
    residual_scale = float(actor_config.actor_residual_scale)
    with torch.no_grad():
        c, _ = actor(z, s, ref, training=False)
        # server path: compose_actor_action with the checkpoint's scale/residual_scale
        server_action = compose_actor_action(
            c, ref, scale.to(args.device),
            output_semantics=ck["actor_output_semantics"],
            residual_scale=residual_scale)
        # Independent expansion of the residual-v4 checkpoint contract.
        shaped_scale = scale.to(args.device).reshape(1, 1, -1)
        manual_action = torch.clamp(
            ref + shaped_scale * residual_scale * torch.tanh(c),
            min=-shaped_scale,
            max=shaped_scale,
        )
    diff = (server_action - manual_action).abs().max().item()
    check("compose contract parity", diff == 0.0, f"maxdiff={diff:.3e}")

    # ── sane output: bounded correction, ref passthrough shape ──
    corr = (server_action - ref).abs()
    bound = scale.to(args.device).reshape(1, 1, -1) * residual_scale + 1e-6
    check("correction bounded by scale*residual_scale",
          bool((corr <= bound).all()),
          f"corr_max={corr.max().item():.5f}")
    q = critic.q_min(z, s, server_action)
    check("critic output finite", bool(torch.isfinite(q).all()),
          f"q={q.mean().item():.5f}")
    print(f"  [info] sample row from {bundles[0].name}: corr_rms="
          f"{corr.pow(2).mean().sqrt().item():.5f} q_min={q.mean().item():.5f}")

    if not all(ok):
        print("DEPLOY CHECK FAILED")
        sys.exit(1)
    print(f"DEPLOY CHECK OK ({len(ok)}/{len(ok)})")


if __name__ == "__main__":
    main()
