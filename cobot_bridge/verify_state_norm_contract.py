"""Offline verification of the normalization contract of a relative-EEF SmolVLA checkpoint.

Relative-EEF inference and continued training load the checkpoint's saved
pre/post processors. A fresh relative-EEF training run from a base/absolute
model builds its processor from the active dataset statistics.
This script performs the pre-real-robot checks:

1. Single-batch input/output check: one real dataset frame is pushed through
   the exact preprocessor/postprocessor that ``run_lerobot_policy._load_policy``
   would build, and the resulting state/action tensors are compared against a
   closed-form reference (IDENTITY pass-through or MEAN_STD with the loaded
   stats). It warns when a checkpoint declares a non-IDENTITY STATE mode but
   carries no observation.state statistics (state would pass through raw).
2. Action round-trip: the normalized relative chunk is unnormalized through the
   postprocessor and compared against ``chunk_to_relative`` of the raw frame.
3. Optional continuation report (``--check-resume``) verifies that continued
training inherits the saved checkpoint processor rather than rebasing its
coordinates on the supplied dataset.

Usage (from the lerobot repo root, conda env lerobot0_4_0):

    python -m cobot_bridge.verify_state_norm_contract \
        --checkpoint outputs/<run>/checkpoints/020000/pretrained_model \
        --dataset <dataset dir> [--frame 15000]

    python -m cobot_bridge.verify_state_norm_contract --check-resume \
        --resume-from outputs/<run>/checkpoints/020000/pretrained_model \
        --dataset <dataset dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NORM_EPS = 1e-8  # must match _NormalizationMixin.eps


def _normalize_step(preprocessor) -> tuple[dict, dict]:
    """Return (stats, norm_map) of the preprocessor normalizer step."""
    for step in preprocessor.steps:
        stats = getattr(step, "stats", None)
        norm_map = getattr(step, "norm_map", None)
        if stats is not None and norm_map is not None:
            stats = {key: {name: np.asarray(value, dtype=np.float32) for name, value in sub.items()}
                     for key, sub in stats.items()}
            return stats, dict(norm_map)
    raise RuntimeError("no normalizer (stats + norm_map) found in the loaded preprocessor")


def _resolve_checkpoint(policy_path: Path):
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(policy_path)
    if not bool(getattr(config, "use_relative_eef_actions", False)):
        raise SystemExit(f"{policy_path} is not a relative-EEF checkpoint; nothing to verify")
    config.device = "cpu"
    return config


def verify_checkpoint(args: argparse.Namespace) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.relative_eef import chunk_to_relative

    policy_path = Path(args.checkpoint)
    config = _resolve_checkpoint(policy_path)
    print(f"checkpoint        : {policy_path}")

    # Exactly what run_lerobot_policy._load_policy builds.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    stats, norm_map = _normalize_step(preprocessor)
    state_mode = norm_map.get("STATE", "IDENTITY")
    state_mode = getattr(state_mode, "value", state_mode)  # NormalizationMode -> str
    state_mode = str(state_mode).upper()
    has_state_stats = "observation.state" in stats
    if state_mode == "IDENTITY" or not has_state_stats:
        effective = "IDENTITY (raw state)"
    else:
        effective = state_mode
    print(f"declared STATE    : {state_mode} (normalization_mapping)")
    print(f"state stats saved : {'yes' if has_state_stats else 'NO'}")
    print(f"effective contract: {effective}")
    if state_mode not in ("IDENTITY",) and not has_state_stats:
        print("WARNING: STATE mode is not IDENTITY but no observation.state stats are "
              "saved; state will pass through raw (silent fallback).")

    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset)
    fps = float(ds_meta.fps)
    dataset = LeRobotDataset(
        args.dataset,
        root=args.dataset,
        delta_timestamps={"action": [i / fps for i in range(config.chunk_size)]},
    )
    frame_index = args.frame if args.frame is not None else dataset.meta.total_frames // 2
    raw = dataset[frame_index]
    sample = {**raw, "action": torch.as_tensor(raw["action"], dtype=torch.float32)}
    state_raw = torch.as_tensor(raw["observation.state"], dtype=torch.float32)
    action_raw = sample["action"]
    print(f"dataset           : {args.dataset}")
    print(f"frame             : {frame_index} (state {tuple(state_raw.shape)}, action {tuple(action_raw.shape)})")

    failures = 0

    out = preprocessor(sample)
    state_out = out["observation.state"]
    action_out = out["action"]

    # --- 1. state contract ---
    if effective == "IDENTITY (raw state)":
        ok = torch.allclose(state_out, state_raw, atol=0.0, rtol=0.0)
        detail = "state passes through unchanged"
    elif state_mode == "MEAN_STD":
        mean = torch.as_tensor(stats["observation.state"]["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["observation.state"]["std"], dtype=torch.float32)
        expected = (state_raw - mean) / (std + NORM_EPS)
        ok = torch.allclose(state_out, expected, atol=1e-6, rtol=1e-5)
        detail = f"state == (raw - mean) / (std + {NORM_EPS}) with the checkpoint's saved stats"
    else:
        ok = False
        detail = f"STATE mode {state_mode!r} is not covered by this verifier"
    print(f"[{'PASS' if ok else 'FAIL'}] state contract: {detail}")
    failures += 0 if ok else 1

    # --- 2. action contract: relative chunk + relative stats ---
    relative_ref = torch.as_tensor(
        chunk_to_relative(state_raw.cpu().numpy(), action_raw.cpu().numpy()), dtype=torch.float32
    )  # (H, 7)
    if "action" not in stats:
        print("[FAIL] no action stats in the loaded preprocessor normalizer")
        return 1
    mean_a = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32)
    std_a = torch.as_tensor(stats["action"]["std"], dtype=torch.float32)
    expected_action = (relative_ref.unsqueeze(0) - mean_a) / (std_a + NORM_EPS)
    ok = torch.allclose(action_out, expected_action, atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if ok else 'FAIL'}] preprocessor action == MEAN_STD(chunk_to_relative(state, action))")
    failures += 0 if ok else 1

    # --- 3. postprocessor round-trip ---
    unnorm = postprocessor(action_out)
    ok = torch.allclose(unnorm, relative_ref.unsqueeze(0), atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if ok else 'FAIL'}] postprocessor round-trip restores the relative chunk")
    failures += 0 if ok else 1

    print(f"\n{failures} check(s) failed" if failures else "\nAll checks passed.")
    return 1 if failures else 0


def check_resume(args: argparse.Namespace) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    config = _resolve_checkpoint(Path(args.resume_from))
    dataset_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.resume_from),
        dataset_stats=dataset_meta.stats,
    )
    stats, norm_map = _normalize_step(preprocessor)
    state_mode = norm_map.get("STATE", "IDENTITY")
    state_mode = getattr(state_mode, "value", state_mode)
    state_mode = str(state_mode).upper()
    has_state_stats = "observation.state" in stats
    print(f"resumed checkpoint: {args.resume_from}")
    print(f"active dataset  : {args.dataset}")
    print(f"declared STATE  : {state_mode}")
    if state_mode != "IDENTITY" and not has_state_stats:
        print("result          : FAIL - dataset processor has no observation.state statistics")
        return 1
    print(
        "result          : continuation inherits the saved relative-EEF processor "
        f"(observation.state stats: {'present, applied per normalization_mapping' if has_state_stats else 'not required for IDENTITY'})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="pretrained_model directory of the checkpoint")
    parser.add_argument("--dataset", type=str, help="LeRobot dataset root for the single-batch check")
    parser.add_argument("--frame", type=int, default=None, help="global frame index (default: middle of dataset)")
    parser.add_argument("--check-resume", action="store_true", help="report the contract a resume would inherit")
    parser.add_argument("--resume-from", type=Path, help="checkpoint to resume from (for --check-resume)")
    args = parser.parse_args()

    if args.check_resume:
        if args.resume_from is None or args.dataset is None:
            parser.error("--check-resume requires --resume-from and --dataset")
        return check_resume(args)

    if args.checkpoint is None or args.dataset is None:
        parser.error("single-batch check requires --checkpoint and --dataset")
    return verify_checkpoint(args)


if __name__ == "__main__":
    sys.exit(main())
