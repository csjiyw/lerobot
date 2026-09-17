"""Validate a local RLT Stage 1 checkpoint without loading or moving the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlt-checkpoint", type=Path, required=True)
    return parser


def validate(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "rlt_stage1_joint" or checkpoint.get("rlt_architecture") != "rlinf_prefix_v1":
        raise RuntimeError("expected an RLinf prefix-only joint Stage 1 checkpoint")

    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (
        RLTokenDecoder,
        RLTokenEncoder,
    )

    config_payload = checkpoint.get("rlt_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("checkpoint is missing rlt_config")
    config = SmolVLARLTConfig(**config_payload)
    encoder = RLTokenEncoder(config)
    decoder = RLTokenDecoder(config)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)

    smolvla_path = Path(str(checkpoint.get("smolvla_path", ""))).expanduser()
    required_policy_files = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    )
    missing_policy_files = [
        name for name in required_policy_files if not (smolvla_path / name).is_file()
    ]
    if missing_policy_files:
        raise RuntimeError(
            f"base SmolVLA checkpoint is incomplete at {smolvla_path}: {missing_policy_files}"
        )

    vlm_length = 177
    with torch.inference_mode():
        prefix = torch.zeros(1, vlm_length, config.vlm_hidden_dim)
        prefix_mask = torch.ones(1, vlm_length, dtype=torch.bool)
        z_rl = encoder(prefix, prefix_mask)
        reconstruction = decoder(z_rl, prefix, prefix_mask)
    if reconstruction.shape != prefix.shape:
        raise RuntimeError("RLT reconstruction shape check failed")
    if not all(torch.isfinite(value).all() for value in (z_rl, reconstruction)):
        raise RuntimeError("RLT checkpoint produced non-finite values")

    return {
        "ready": True,
        "motion": False,
        "stage": checkpoint["stage"],
        "rlt_architecture": checkpoint["rlt_architecture"],
        "step": int(checkpoint.get("step", 0)),
        "loss": float(checkpoint.get("loss", float("nan"))),
        "smolvla_path": str(smolvla_path.resolve()),
        "dataset_root": str(checkpoint.get("dataset_root", "")),
        "prefix_embedding": [vlm_length, int(config.vlm_hidden_dim)],
        "z_rl": int(config.rlt_hidden_dim),
        "state_dim": int(config.state_dim),
        "action_dim": int(config.action_dim),
        "strict_load": True,
        "finite_forward": True,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(validate(args.rlt_checkpoint), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
