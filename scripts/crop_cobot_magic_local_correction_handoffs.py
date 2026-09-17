#!/usr/bin/env python
"""Crop non-canonical return-to-ready tails from seven local corrections.

The source recordings remain immutable. Each output ends after the successful
correction plus its initial safe retreat, before the operator-specific return
motion starts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


END_EXCLUSIVE = {
    "episode_20260916_171950_018111.raw.h5": 376,
    "episode_20260916_172325_366232.raw.h5": 310,
    "episode_20260916_172402_766054.raw.h5": 333,
    "episode_20260916_172432_932049.raw.h5": 314,
    "episode_20260916_172500_518341.raw.h5": 245,
    "episode_20260916_172534_936466.raw.h5": 465,
    "episode_20260916_172614_856574.raw.h5": 475,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    report = []
    for filename, end in END_EXCLUSIVE.items():
        source = args.source / filename
        output = args.output / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output)

        with h5py.File(output, "r+") as episode:
            old_frames = int(episode.attrs["num_frames"])
            if not 1 < end < old_frames:
                raise ValueError(f"Invalid crop {filename}: {end=} {old_frames=}")

            datasets: list[h5py.Dataset] = []
            episode.visititems(
                lambda _name, obj: datasets.append(obj) if isinstance(obj, h5py.Dataset) else None
            )
            for dataset in datasets:
                if dataset.ndim > 0 and dataset.shape[0] == old_frames:
                    dataset.resize((end, *dataset.shape[1:]))

            # Preserve a well-formed successful terminal sample at the new end.
            episode["frames/done"][-1] = 1
            episode["frames/success"][-1] = 1
            episode["frames/phase"][-1] = 5
            episode["frames/reward"][-1] = 1.0
            episode["frames/reward_valid"][-1] = 1
            episode.attrs["num_frames"] = end
            episode.attrs["closed_at_ns"] = int(episode["frames/pc_timestamp_ns"][-1])
            episode.attrs["stop_reason"] = "offline_crop_after_correction_handoff"
            episode.attrs["offline_crop_source_frames"] = old_frames
            episode.attrs["offline_crop_end_exclusive"] = end
            episode.flush()

            lengths = []
            episode.visititems(
                lambda _name, obj: lengths.append(obj.shape[0])
                if isinstance(obj, h5py.Dataset) and obj.ndim > 0
                else None
            )
            if set(lengths) != {end}:
                raise RuntimeError(f"Dataset lengths diverged in {filename}: {set(lengths)}")

        source_result = source.with_suffix(source.suffix + ".result")
        if source_result.is_file():
            shutil.copy2(source_result, output.with_suffix(output.suffix + ".result"))
        report.append({"file": filename, "source_frames": old_frames, "cropped_frames": end})

    (args.output / "crop_manifest.json").write_text(
        json.dumps({"episodes": report}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
