#!/usr/bin/env python
"""Measure sustained Cobot Magic video loading plus policy preprocessing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


DEFAULT_ROOT = Path(
    "/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/"
    "datasets/cobot_magic_il_cross_073_clean_arm_state_t3_cmd_gripper_v3_eef"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("pyav", "ffmpeg", "torchcodec"), required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-aug", action="store_true", help="Enable the formal photometric augmentation set.")
    parser.add_argument("--max-image-transforms", type=int, default=2, help="Random transforms per image.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    image_transforms = None
    if args.image_aug:
        image_transforms = ImageTransforms(
            ImageTransformsConfig(
                enable=True,
                max_num_transforms=args.max_image_transforms,
                tfs={
                    "brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.8, 1.2)}),
                    "contrast": ImageTransformConfig(type="ColorJitter", kwargs={"contrast": (0.8, 1.2)}),
                    "saturation": ImageTransformConfig(type="ColorJitter", kwargs={"saturation": (0.7, 1.3)}),
                    "hue": ImageTransformConfig(type="ColorJitter", kwargs={"hue": (-0.05, 0.05)}),
                    "sharpness": ImageTransformConfig(type="SharpnessJitter", kwargs={"sharpness": (0.7, 1.3)}),
                },
            )
        )
    dataset = LeRobotDataset(
        "local/cobot_magic_il_cross_073_clean_arm_state_t3_cmd_gripper_v3_eef",
        root=root,
        delta_timestamps={"action": [index / 30 for index in range(30)]},
        video_backend=args.backend,
        image_transforms=image_transforms,
    )
    features = dataset_to_policy_features(dataset.meta.features)
    config = SmolVLAConfig(
        input_features={key: value for key, value in features.items() if key.startswith("observation.state") or key.startswith("observation.images.camera_")},
        output_features={"action": features["action"]},
        use_relative_eef_actions=True,
        relative_eef_action_stats_path=str(root / "meta" / "relative_eef_action_stats.json"),
        device="cpu",
        pad_language_to="max_length",
    )
    preprocessor, _ = make_pre_post_processors(
        config, pretrained_path="relative-eef-benchmark", dataset_stats=dataset.meta.stats
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=True,
        pin_memory=False,
    )
    iterator = iter(loader)
    fetch_times: list[float] = []
    preprocess_times: list[float] = []
    started = time.perf_counter()
    for step in range(args.batches + args.warmup):
        start = time.perf_counter()
        batch = next(iterator)
        fetch_s = time.perf_counter() - start
        start = time.perf_counter()
        processed = preprocessor(batch)
        preprocess_s = time.perf_counter() - start
        assert processed["action"].shape == (args.batch_size, 30, 7)
        if step >= args.warmup:
            fetch_times.append(fetch_s)
            preprocess_times.append(preprocess_s)
        if (step + 1) % 25 == 0:
            print(f"progress backend={args.backend} batch={step + 1}/{args.batches + args.warmup}", flush=True)
    elapsed = time.perf_counter() - started
    fetch = torch.tensor(fetch_times)
    preprocess = torch.tensor(preprocess_times)
    print(
        "RESULT"
        f" backend={args.backend} batches={args.batches} batch_size={args.batch_size}"
        f" fetch_mean_s={fetch.mean().item():.4f} fetch_p95_s={torch.quantile(fetch, .95).item():.4f}"
        f" preprocess_mean_s={preprocess.mean().item():.4f}"
        f" total_samples_per_s={args.batches * args.batch_size / elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
