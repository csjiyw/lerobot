#!/usr/bin/env python3
"""Merge ordered, cache-only Actor-BC feature shards without changing samples.

``train_rlt_actor_bc.py`` can encode disjoint contiguous portions of a
LeRobot dataset on separate GPUs.  This helper verifies that those portions
form exactly one dataset order, then concatenates them into the cache format
the trainer normally writes itself.  It never recomputes, shuffles, or
normalizes features.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _metadata(path: Path) -> dict:
    meta_path = path.with_suffix(path.suffix + ".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing cache metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("format_version") != 2:
        raise ValueError(f"{path}: unsupported cache format {metadata.get('format_version')}")
    return metadata


def merge(inputs: list[Path], output: Path) -> dict:
    if len(inputs) < 2:
        raise ValueError("at least two cache shards are required")
    rows = sorted(((_metadata(path), path) for path in inputs), key=lambda item: item[0]["shard_start"])
    baseline, _ = rows[0]
    required_same = (
        "stage1_checkpoint", "action_chunk_timebase", "action_chunk_frame_stride",
        "action_steps", "full_samples", "dataset_repo_id", "dataset_root", "has_is_success",
    )
    for metadata, path in rows:
        if any(metadata.get(key) != baseline.get(key) for key in required_same):
            raise ValueError(f"{path}: cache metadata does not match the first shard")
    expected_start = 0
    arrays_by_name: dict[str, list[np.ndarray]] = {}
    for metadata, path in rows:
        start, stop = int(metadata["shard_start"]), int(metadata["shard_stop"])
        if start != expected_start or stop <= start:
            raise ValueError(f"{path}: non-contiguous shard [{start}, {stop}), expected {expected_start}")
        with np.load(path, allow_pickle=False) as source:
            names = set(source.files)
            if arrays_by_name and names != set(arrays_by_name):
                raise ValueError(f"{path}: array fields differ across shards")
            for name in sorted(names):
                value = np.asarray(source[name])
                if value.shape[0] != stop - start:
                    raise ValueError(f"{path}: {name} has {value.shape[0]} rows, expected {stop - start}")
                arrays_by_name.setdefault(name, []).append(value)
        expected_start = stop
    if expected_start != int(baseline["full_samples"]):
        raise ValueError(f"shards cover [0,{expected_start}), not all {baseline['full_samples']} samples")
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays_by_name.items()}
    if any(value.shape[0] != expected_start for value in merged.values()):
        raise ValueError("merged arrays have inconsistent sample counts")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **merged)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = dict(baseline)
    metadata.update({"samples": expected_start, "cache_num_shards": 1, "cache_shard_id": 0, "shard_start": 0, "shard_stop": expected_start})
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    args = _parser().parse_args()
    metadata = merge([path.expanduser().resolve() for path in args.inputs], args.output)
    print(f"merged Actor-BC cache samples={metadata['samples']} output={args.output}")


if __name__ == "__main__":
    main()
