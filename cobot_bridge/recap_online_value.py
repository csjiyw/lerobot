"""Score one completed raw robot episode with the frozen RECAP value model.

This is deliberately separate from the global RECAP feature cache: after an
online rollout it computes the visual/language value trace directly from that
raw H5 archive, then writes ``{sequence, values}`` for dense replay building.
It never connects to the bridge or sends robot commands.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from .build_recap_dense_replay import ValueHead
from .convert_lerobot_joint_to_eef import _joint_to_eef
from .piper_eef_metrics import PiperForwardKinematics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--recap-checkpoint", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def _decode_jpeg(value) -> torch.Tensor:
    data = bytes(value) if isinstance(value, (bytes, bytearray)) else bytes(value.tobytes())
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def _normalizer_stats(policy_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    candidates = sorted(policy_path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one state normalizer under {policy_path}, found {len(candidates)}")
    with safe_open(candidates[0], framework="pt") as source:
        return (
            source.get_tensor("observation.state.mean").float().to(device),
            source.get_tensor("observation.state.std").float().to(device),
        )


def score_raw_episode(
    *, stage1_checkpoint: Path, recap_checkpoint: Path, raw_path: Path, task: str,
    output: Path, device: str, batch_size: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector, resize_with_pad

    target = torch.device(device)
    stage1 = torch.load(stage1_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    policy_path = Path(stage1["smolvla_path"]).expanduser().resolve()
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.device = str(target)
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=policy_config, local_files_only=True).to(target)
    policy.eval().requires_grad_(False)
    state_mean, state_std = _normalizer_stats(policy_path, target)

    recap_payload = torch.load(recap_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    value_head = ValueHead(
        int(recap_payload["feature_dim"]), int(recap_payload["hidden"]), int(recap_payload["num_bins"])
    ).to(target)
    value_head.load_state_dict(recap_payload["state_dict"], strict=True)
    value_head.eval().requires_grad_(False)

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    language = tokenizer(
        [f"{task}\n"], padding="max_length", max_length=48, truncation=True, return_tensors="pt"
    )
    language_tokens = language["input_ids"].to(target)
    language_mask = language["attention_mask"].to(target).bool()
    height, width = tuple(policy.config.resize_imgs_with_padding)

    def prepare(image: torch.Tensor) -> torch.Tensor:
        value = resize_with_pad(image.unsqueeze(0).to(target, dtype=torch.float32), height, width, pad_value=0)
        return (value * 2.0 - 1.0).squeeze(0)

    raw_path = raw_path.expanduser().resolve()
    with h5py.File(raw_path, "r") as raw, torch.inference_mode():
        sequence = np.asarray(raw["frames/sequence"], dtype=np.int64)
        # Raw archives preserve physical Piper joint feedback.  The current
        # relative-EEF SmolVLA was trained with gripper-center EEF state; using
        # joint radians against its EEF mean/std silently feeds the frozen VLM
        # an out-of-distribution state.  Mirror the rollout/replay contract.
        joint_states = np.asarray(raw["frames/puppet_position"], dtype=np.float32)
        states = _joint_to_eef(
            joint_states,
            np.ones(joint_states.shape[0], dtype=bool),
            PiperForwardKinematics(),
        )
        front = raw["images/front/jpeg"]
        right = raw["images/right/jpeg"]
        if states.shape != (len(sequence), 7):
            raise ValueError(f"raw puppet state shape is invalid: {states.shape}")
        values = np.empty(len(sequence), dtype=np.float32)
        for start in range(0, len(sequence), batch_size):
            stop = min(start + batch_size, len(sequence))
            count = stop - start
            images = [
                torch.stack([prepare(_decode_jpeg(front[index])) for index in range(start, stop)]),
                torch.stack([prepare(_decode_jpeg(right[index])) for index in range(start, stop)]),
            ]
            image_masks = [torch.ones(count, dtype=torch.bool, device=target) for _ in images]
            state = torch.from_numpy(states[start:stop]).to(target)
            state = pad_vector((state - state_mean) / state_std, policy.config.max_state_dim)
            with torch.autocast(device_type=target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
                prefix, padding = policy.model.extract_rlt_prefix(
                    images, image_masks, language_tokens.expand(count, -1), language_mask.expand(count, -1), state
                )
                valid = padding.bool()
                last_valid = valid.shape[1] - 1 - valid.to(torch.int8).flip(1).argmax(1)
                read = valid.clone()
                read[torch.arange(count, device=target), last_valid] = False
                features = (prefix.float() * read.unsqueeze(-1)).sum(1) / read.sum(1, keepdim=True).clamp_min(1)
                values[start:stop] = value_head.value(features).float().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("RECAP value trace contains non-finite values")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sequence=sequence, values=values)


def main() -> None:
    args = _parser().parse_args()
    score_raw_episode(
        stage1_checkpoint=args.stage1_checkpoint,
        recap_checkpoint=args.recap_checkpoint,
        raw_path=args.raw,
        task=args.task,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
