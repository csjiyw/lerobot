"""Run the Cobot Magic RLT representation path in read-only shadow mode.

The process reads bridge observations and evaluates the frozen SmolVLA plus
the Stage 1 RLT encoder. It never requests a control mode, enables output, or
sends an action to the robot.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .client import CobotBridgeClient
from .remote_robot import CobotMagicRemoteRobot
from .run_lerobot_policy import _load_policy, _policy_observation


LOG = logging.getLogger("cobot_bridge.rlt_shadow")
STOP_REQUESTED = False


def _request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--rlt-checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Diagnostic duration in seconds; 0 runs until interrupted.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cameras", nargs="+", choices=("front", "left", "right"), default=("front", "right"))
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    return parser


def load_models(args: argparse.Namespace):
    checkpoint = torch.load(args.rlt_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "rlt_stage1_joint" or checkpoint.get("rlt_architecture") != "rlinf_prefix_v1":
        raise RuntimeError("RLT shadow requires an RLinf prefix-only joint Stage 1 checkpoint")
    smolvla_path = Path(str(checkpoint.get("smolvla_path", ""))).expanduser().resolve()
    policy_args = argparse.Namespace(
        policy_path=str(smolvla_path),
        device=args.device,
        vlm_model_path=None,
        n_action_steps=None,
    )
    policy, preprocessor, postprocessor = _load_policy(policy_args)

    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLTokenEncoder

    rlt_config = SmolVLARLTConfig(**checkpoint["rlt_config"])
    encoder = RLTokenEncoder(rlt_config).to(args.device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.eval().requires_grad_(False)
    return policy, preprocessor, postprocessor, encoder, rlt_config, smolvla_path


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.fps <= 0 or args.duration < 0:
        raise ValueError("fps must be positive and duration must be non-negative")
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    cameras = tuple(args.cameras)
    device = torch.device(args.device)
    policy, preprocessor, postprocessor, encoder, rlt_config, smolvla_path = load_models(args)

    from lerobot.policies.utils import prepare_observation_for_inference

    output = args.diagnostic_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.fps
    frames = 0
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    LOG.info(
        "RLT shadow ready device=%s base=%s stage1=%s output=%s action_publisher=False",
        device,
        smolvla_path,
        args.rlt_checkpoint,
        output,
    )
    with output.open("a", encoding="utf-8") as diagnostics:
        with CobotBridgeClient(args.endpoint, timeout_s=5.0) as client:
            robot = CobotMagicRemoteRobot(client, side=args.side, images=cameras)
            while not STOP_REQUESTED and (deadline is None or time.monotonic() < deadline):
                started = time.perf_counter()
                observation = robot.observe()
                if not observation.ready:
                    LOG.warning("observation not ready: %s", observation.raw.get("missing_or_stale"))
                    time.sleep(max(0.0, period - (time.perf_counter() - started)))
                    continue
                raw_observation = _policy_observation(observation, cameras)
                batch = prepare_observation_for_inference(
                    raw_observation,
                    device,
                    args.task,
                    "cobot_magic_piper",
                )
                batch = preprocessor(batch)
                amp = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" and bool(getattr(policy.config, "use_amp", False))
                    else nullcontext()
                )
                with torch.inference_mode(), amp:
                    normalized_actions, prefix_embedding, prefix_mask = (
                        policy.predict_action_chunk_with_rlt_prefix(batch)
                    )
                    if prefix_embedding.shape[-1] != rlt_config.vlm_hidden_dim:
                        raise RuntimeError(
                            f"VLM embedding mismatch: {prefix_embedding.shape[-1]} != "
                            f"{rlt_config.vlm_hidden_dim}"
                        )
                    z_rl = encoder(prefix_embedding.float(), prefix_mask)
                    processed = [
                        postprocessor(normalized_actions[:, index, :])
                        for index in range(normalized_actions.shape[1])
                    ]
                    reference_actions = torch.stack(processed, dim=1).squeeze(0)
                if not torch.isfinite(z_rl).all() or not torch.isfinite(reference_actions).all():
                    raise RuntimeError("RLT shadow produced non-finite values")
                frames += 1
                record = {
                    "time_ns": time.time_ns(),
                    "frame": frames,
                    "bridge_sequence": observation.raw.get("sequence"),
                    "teach_active": observation.teach_active,
                    "effective_mode": observation.effective_mode,
                    "puppet_state": observation.puppet_state.astype(np.float32).tolist(),
                    "reference_action_0": reference_actions[0].detach().cpu().float().tolist(),
                    "reference_chunk_size": int(reference_actions.shape[0]),
                    "z_rl_norm": float(torch.linalg.vector_norm(z_rl).detach().cpu()),
                    "z_rl_mean": float(z_rl.mean().detach().cpu()),
                    "z_rl_std": float(z_rl.std().detach().cpu()),
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "action_sent": False,
                }
                diagnostics.write(json.dumps(record, separators=(",", ":")) + "\n")
                diagnostics.flush()
                if frames == 1 or frames % 10 == 0:
                    LOG.info(
                        "shadow frame=%d latency_ms=%.1f z_norm=%.3f chunk=%d action_sent=False",
                        frames,
                        record["latency_ms"],
                        record["z_rl_norm"],
                        record["reference_chunk_size"],
                    )
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
    LOG.info(
        "RLT shadow complete frames=%d stopped_by_user=%s action_sent=0 output=%s",
        frames,
        STOP_REQUESTED,
        output,
    )


if __name__ == "__main__":
    main()
