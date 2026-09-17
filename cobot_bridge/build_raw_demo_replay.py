"""Encode one labeled raw cross-task episode as a Stage 2 IL demo replay."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .rlt_policy_server import RLTPolicyRuntime, _decode_jpeg
from .rlt_stage2_data import (
    ACTION_CHUNK_FRAME_STRIDE,
    ACTION_CHUNK_TIMEBASE,
    REPLAY_FORMAT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decision-stride-frames", type=int, default=30)
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help="Action-chunk horizon override (e.g. 10 for the remote chunk10 "
        "recipe on a chunk30 VLA). Must be <= the runtime chunk length.",
    )
    return parser


def complete_demo_rows(
    learning_mask: np.ndarray,
    executed_valid: np.ndarray,
    *,
    action_steps: int,
    action_stride: int,
    decision_stride_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    learning = np.asarray(learning_mask).reshape(-1).astype(bool)
    valid = np.asarray(executed_valid).reshape(-1).astype(bool)
    if learning.shape != valid.shape or learning.size == 0:
        raise ValueError("learning and executed-valid masks must be non-empty and aligned")
    if min(action_steps, action_stride, decision_stride_frames) <= 0:
        raise ValueError("action and decision strides must be positive")
    offsets = np.arange(action_steps, dtype=np.int64) * action_stride
    candidates = np.flatnonzero(learning)
    candidates = candidates[candidates + offsets[-1] < learning.size]
    candidates = np.asarray(
        [row for row in candidates if valid[row + offsets].all()], dtype=np.int64
    )
    if candidates.size < 2:
        raise ValueError("raw demo has fewer than two complete action chunks")
    selected = candidates[::decision_stride_frames]
    if selected.size < 2:
        selected = candidates[[0, -1]]
    return selected, selected[:, None] + offsets[None, :]


def _jpeg(dataset: h5py.Dataset, index: int) -> np.ndarray:
    return _decode_jpeg(bytes(np.asarray(dataset[index], dtype=np.uint8)))


def _atomic_bundle(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary_json = sidecar.with_name(sidecar.name + f".{uuid.uuid4().hex}.tmp")
    temporary_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, sidecar)


def main() -> None:
    args = build_parser().parse_args()
    raw_path = args.raw.expanduser().resolve()
    stage1 = args.stage1_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    runtime = RLTPolicyRuntime(
        stage1_checkpoint=stage1,
        stage2_checkpoint=output.parent / ".no_stage2_checkpoint.pt",
        device=args.device,
        cameras=("front", "right"),
    )
    runtime.begin_episode(workflow_mode=2, allow_actor=False, actor_exploration_std=0.0)
    try:
        with h5py.File(raw_path, "r") as raw:
            if not bool(raw.attrs.get("completed", False)):
                raise ValueError("raw demo is incomplete")
            if str(raw.attrs.get("outcome", "")) != "success":
                raise ValueError("raw IL demo must be labeled success")
            cameras = tuple(json.loads(str(raw.attrs["cameras"])))
            if cameras != runtime.cameras:
                raise ValueError(f"camera mismatch: {cameras} versus {runtime.cameras}")
            dimensions = runtime.dimensions
            native_action_steps = int(dimensions["action_steps"])
            if args.action_steps is not None:
                if not 1 <= args.action_steps <= native_action_steps:
                    raise ValueError(
                        f"action-steps must be in [1, {native_action_steps}] "
                        f"(runtime chunk length), got {args.action_steps}"
                    )
                action_steps = int(args.action_steps)
            else:
                action_steps = native_action_steps
            selected, action_rows = complete_demo_rows(
                raw["frames/learning_mask"][:],
                raw["frames/executed_action_valid"][:],
                action_steps=action_steps,
                action_stride=int(runtime.config.action_stride),
                decision_stride_frames=args.decision_stride_frames,
            )
            state_raw = raw["frames/puppet_position"][:]
            executed_raw = raw["frames/executed_action"][:]
            intervention_raw = raw["frames/is_human_intervention"][:].astype(bool)
            image_datasets = {name: raw[f"images/{name}/jpeg"] for name in cameras}
            z_rl: list[np.ndarray] = []
            states: list[np.ndarray] = []
            references: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            interventions: list[bool] = []
            for number, (observation_row, chunk_rows) in enumerate(
                zip(selected, action_rows, strict=True), 1
            ):
                inference = runtime.predict(
                    state=np.asarray(state_raw[observation_row], dtype=np.float32),
                    images={
                        name: _jpeg(dataset, int(observation_row))
                        for name, dataset in image_datasets.items()
                    },
                    task=str(raw.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                z_rl.append(np.asarray(inference["z_rl"], dtype=np.float32))
                states.append(np.asarray(inference["state"], dtype=np.float32))
                references.append(
                    np.asarray(inference["reference_actions"], dtype=np.float32)[
                        :action_steps
                    ]
                )
                # normalize_actions requires the full native chunk (n_action_steps_rl,
                # action_dim); normalize the native 30-step chunk and truncate to the
                # requested C (normalization is row-wise affine, so truncating after
                # normalizing is equivalent to normalizing the truncated rows).
                # Rows past the end of the episode are clamped; only the first
                # action_steps rows (guaranteed in-range) are kept.
                action_stride = int(runtime.config.action_stride)
                full_rows = int(observation_row) + np.arange(
                    native_action_steps, dtype=np.int64
                ) * action_stride
                full_rows = np.minimum(full_rows, len(executed_raw) - 1)
                normalized_full = runtime.normalize_actions(
                    np.asarray(executed_raw[full_rows], dtype=np.float32)
                )
                actions.append(normalized_full[:action_steps])
                interventions.append(bool(intervention_raw[chunk_rows].any()))
                if number == 1 or number % 20 == 0:
                    print(f"decision={number}/{len(selected)}", flush=True)
            z = np.stack(z_rl)
            state = np.stack(states)
            reference = np.stack(references)
            action = np.stack(actions)
            intervention = np.asarray(interventions, dtype=np.float32)[:, None]
            transition_count = len(selected) - 1
            rewards = np.zeros((transition_count, 1), dtype=np.float32)
            dones = np.zeros((transition_count, 1), dtype=np.float32)
            rewards[-1, 0] = 1.0
            dones[-1, 0] = 1.0
            arrays = {
                "z_rl": z[:-1],
                "state": state[:-1],
                "actions": action[:-1],
                "ref_actions": reference[:-1],
                "rewards": rewards,
                "next_z_rl": z[1:],
                "next_state": state[1:],
                "next_ref_actions": reference[1:],
                "dones": dones,
                "is_intervention": intervention[:-1],
                "is_demo": np.ones((transition_count, 1), dtype=np.float32),
            }
            metadata = {
                "format": REPLAY_FORMAT,
                "created_at_ns": time.time_ns(),
                "decision_path": None,
                "raw_path": str(raw_path),
                "stage1_checkpoint": str(stage1),
                "task": str(raw.attrs["task"]),
                "workflow_mode": 2,
                "collection_phase": "il_demo",
                "outcome": "success",
                "transition_count": transition_count,
                "intervention_transitions": int(intervention[:-1].sum()),
                "demo_transitions": transition_count,
                "source_decisions": len(selected),
                "decision_stride_frames": args.decision_stride_frames,
                "action_space": "smolvla_normalized",
                "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
                "action_chunk_frame_stride": ACTION_CHUNK_FRAME_STRIDE,
                "z_dim": int(z.shape[1]),
                "state_dim": int(state.shape[1]),
                "action_steps": int(action.shape[1]),
                "action_dim": int(action.shape[2]),
            }
            _atomic_bundle(output, metadata, arrays)
            print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        runtime.end_episode()


if __name__ == "__main__":
    main()
