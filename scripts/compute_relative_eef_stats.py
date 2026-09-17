#!/usr/bin/env python3
"""Compute anchor-relative EEF chunk statistics for a *_eef LeRobot dataset.

Mirrors the training-time transform (contract "groot_chunk_relative_eef"):
    relative[t, h] = chunk_to_relative(state[t], action[t + h])  # h in 0..29
with action indices clamped to the last frame of the episode, exactly like the
dataset chunk loader does at the episode tail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.policies.smolvla.relative_eef import chunk_to_relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="path to the *_eef dataset root")
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output JSON path; defaults to <root>/meta/relative_eef_action_stats.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {root}/data")

    rows: list[np.ndarray] = []
    samples = 0
    for path in parquet_files:
        table = pq.read_table(path, columns=["episode_index", "observation.state", "action"])
        ep_col = table.column("episode_index").to_numpy()
        st_col = np.stack(table.column("observation.state").to_numpy()).astype(np.float64)
        ac_col = np.stack(table.column("action").to_numpy()).astype(np.float64)
        # group rows by episode, preserving on-disk order (already time-sorted)
        bounds = np.flatnonzero(np.diff(ep_col) != 0) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(ep_col)]])
        for lo, hi in zip(starts, ends):
            n = hi - lo
            states = st_col[lo:hi]
            targets = ac_col[lo:hi]
            idx = np.clip(np.arange(n)[:, None] + np.arange(args.chunk_size)[None, :], 0, n - 1)
            for t in range(n):
                rows.append(chunk_to_relative(states[t], targets[idx[t]]).astype(np.float32))
            samples += n

    data = np.concatenate(rows, axis=0)
    print(f"samples={samples} action_rows={data.shape[0]}")
    stats = {
        "contract": {
            "type": "groot_chunk_relative_eef",
            "reference": "observation.state[t]",
            "formula": "inverse(T_state_t) @ T_action_t_plus_h",
            "chunk_size": int(args.chunk_size),
            "frame": "gripper_center",
            "rotation": "axis_angle_rotation_vector_rad",
            "gripper": "absolute_executed_command",
        },
        "samples": int(samples),
        "action_rows": int(data.shape[0]),
        "action": {
            "min": [float(v) for v in data.min(axis=0)],
            "max": [float(v) for v in data.max(axis=0)],
            "mean": [float(v) for v in data.mean(axis=0)],
            "std": [float(v) for v in data.std(axis=0)],
            "count": [int(data.shape[0])],
            "q01": [float(v) for v in np.quantile(data, 0.01, axis=0)],
            "q10": [float(v) for v in np.quantile(data, 0.10, axis=0)],
            "q50": [float(v) for v in np.quantile(data, 0.50, axis=0)],
            "q90": [float(v) for v in np.quantile(data, 0.90, axis=0)],
            "q99": [float(v) for v in np.quantile(data, 0.99, axis=0)],
        },
    }
    out = args.output or (root / "meta" / "relative_eef_action_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
