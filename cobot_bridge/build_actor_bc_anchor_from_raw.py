"""Build phase-specific Actor BC anchors from raw robot recordings.

Unlike a Stage-2 replay bundle, an anchor row has no reward or next state:
it is used only in the actor BC loss.  This keeps short corrective clips
(including a single 10-step window) out of critic/Q training while retaining
their supervised action signal.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np

from .convert_lerobot_joint_to_eef import _joint_to_eef
from .piper_eef_metrics import PiperForwardKinematics
from .reconstruct_rlt_10tick_from_raw import _jpeg
from .rlt_policy_server import RLTPolicyRuntime
from .rlt_stage2_data import ACTION_CHUNK_TIMEBASE
from lerobot.policies.smolvla.relative_eef import chunk_to_relative


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--decision-stride-frames", type=int, default=10)
    parser.add_argument("--pattern", default="*.raw.h5")
    return parser


def bc_window_starts(frame_count: int, action_steps: int, stride_frames: int) -> np.ndarray:
    """Return every complete supervised action window; no next observation needed."""

    if min(frame_count, action_steps, stride_frames) <= 0:
        raise ValueError("frame_count, action_steps, and stride_frames must be positive")
    return np.arange(0, frame_count - action_steps + 1, stride_frames, dtype=np.int64)


def _atomic_save(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _parser().parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    raw_paths = sorted(args.raw_root.expanduser().resolve().glob(args.pattern))
    if not raw_paths:
        raise FileNotFoundError(args.raw_root)
    runtime = RLTPolicyRuntime(
        stage1_checkpoint=args.stage1_checkpoint.expanduser().resolve(),
        stage2_checkpoint=output.parent / ".no_stage2_checkpoint.pt",
        device=args.device,
        cameras=("front", "right"),
        action_steps=args.action_steps,
    )
    dimensions = runtime.dimensions
    if int(dimensions["action_steps"]) != int(args.action_steps):
        raise RuntimeError("runtime did not adopt requested action horizon")
    kinematics = PiperForwardKinematics(frames=("gripper_center",))
    z_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    episode_rows: list[int] = []
    items: list[dict] = []
    try:
        for episode_index, raw_path in enumerate(raw_paths):
            with h5py.File(raw_path, "r") as raw:
                if not bool(raw.attrs.get("completed", False)):
                    continue
                cameras = tuple(json.loads(str(raw.attrs["cameras"])))
                if cameras != runtime.cameras:
                    raise ValueError(f"{raw_path.name}: camera mismatch {cameras}")
                joints = np.asarray(raw["frames/puppet_position"], dtype=np.float32)
                executed = np.asarray(raw["frames/executed_action"], dtype=np.float32)
                action_valid = np.asarray(raw["frames/executed_action_valid"], dtype=bool)
                if joints.shape != executed.shape or joints.shape[1:] != (7,):
                    raise ValueError(f"{raw_path.name}: expected matching (N,7) state/action")
                eef_state = _joint_to_eef(joints, np.ones(len(joints), dtype=bool), kinematics)
                eef_executed = _joint_to_eef(executed, action_valid, kinematics)
                starts = bc_window_starts(
                    len(joints), int(args.action_steps), int(args.decision_stride_frames)
                )
                images = {name: raw[f"images/{name}/jpeg"] for name in cameras}
                runtime.begin_episode(
                    workflow_mode=2, allow_actor=False, actor_exploration_std=0.0
                )
                written = 0
                try:
                    for start in starts:
                        window = slice(int(start), int(start) + int(args.action_steps))
                        if not action_valid[window].all():
                            continue
                        inference = runtime.predict(
                            state=eef_state[int(start)],
                            images={name: _jpeg(dataset, int(start)) for name, dataset in images.items()},
                            task=str(raw.attrs["task"]),
                            robot_type="cobot_magic_piper",
                        )
                        target = runtime.normalize_actions(
                            chunk_to_relative(eef_state[int(start)], eef_executed[window])
                        )
                        z_rows.append(np.asarray(inference["z_rl"], dtype=np.float32))
                        state_rows.append(np.asarray(inference["state"], dtype=np.float32))
                        reference_rows.append(np.asarray(inference["reference_actions"], dtype=np.float32))
                        target_rows.append(np.asarray(target, dtype=np.float32))
                        valid_rows.append(np.ones(int(args.action_steps), dtype=bool))
                        episode_rows.append(episode_index)
                        written += 1
                finally:
                    runtime.end_episode()
                items.append({"raw": str(raw_path), "episode_index": episode_index, "rows": written})
    finally:
        if runtime.active_episode:
            runtime.end_episode()
    if not z_rows:
        raise ValueError("no complete BC-only windows were found")
    arrays = {
        "z_rl": np.stack(z_rows),
        "state": np.stack(state_rows),
        "reference": np.stack(reference_rows),
        "target": np.stack(target_rows),
        "valid": np.stack(valid_rows),
        "episode_index": np.asarray(episode_rows, dtype=np.int64),
    }
    metadata = {
        "format": "cobot_magic_actor_bc_anchor_v1",
        "stage1_checkpoint": str(runtime.stage1_path),
        "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
        "action_steps": int(args.action_steps),
        "decision_stride_frames": int(args.decision_stride_frames),
        "bc_only": True,
        "rows": int(arrays["z_rl"].shape[0]),
        "episodes": int(np.unique(arrays["episode_index"]).size),
        "items": items,
    }
    _atomic_save(output, arrays, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
