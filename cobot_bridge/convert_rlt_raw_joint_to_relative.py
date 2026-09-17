#!/usr/bin/env python3
"""Rebuild the behavior channel of BC-runner raw archives in policy space.

Episodes recorded through the Standard/joint rollout pipeline store
``frames/executed_action`` as the post-IK ABSOLUTE joint targets (rad) that
the bridge actually commanded, while ``build_rlt_replay_from_raw`` expects
the policy-space relative-EEF deltas used by the RLT runtime (the do02
byte-exact contract).  Normalizing joint angles with the relative-EEF
normalizer explodes them to >80 sigma and silently corrupts every Q/BC
target (observed as ``actor diverged ... bc_loss=372`` on the first actor
update).

This tool converts joint-target raws into relative-EEF raws WITHOUT touching
anything else: for every 30-frame decision window (stride = --action-steps,
must match the stage1 n_action_steps_rl), the new behavior value at frame
``start+h`` is ``eef_pose_delta(FK(puppet[start]), FK(joint_target[start+h]))``
using the exact helpers from convert_lerobot_joint_to_eef (gripper_center FK,
base-frame translation, spatial relative rotation, absolute gripper).  Frames
outside decision windows and every other channel are copied untouched.

Usage:
    python -m cobot_bridge.convert_rlt_raw_joint_to_relative \
        --raw-root ../raw_datasets/masterpiece \
        --output-root ../raw_datasets/masterpiece_relative \
        --action-steps 30
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from .convert_lerobot_joint_to_eef import _joint_to_eef
from .piper_eef_metrics import DEFAULT_PIPER_URDF, PiperForwardKinematics
from .piper_eef_transform import eef_pose_delta


def _convert_episode(
    raw_path: Path,
    out_path: Path,
    kinematics: PiperForwardKinematics,
    action_steps: int,
) -> dict[str, int]:
    shutil.copyfile(raw_path, out_path)
    sidecar = raw_path.with_name(raw_path.name + ".result")
    if sidecar.is_file():
        shutil.copyfile(sidecar, out_path.with_name(out_path.name + ".result"))
    with h5py.File(out_path, "a") as raw:
        executed = np.asarray(raw["frames/executed_action"], dtype=np.float64)
        puppet = np.asarray(raw["frames/puppet_position"], dtype=np.float64)
        if executed.shape != puppet.shape or executed.shape[1] != 7:
            raise ValueError(f"{raw_path.name}: expected (N,7) executed/puppet")
        n = executed.shape[0]
        state_pose = _joint_to_eef(puppet, np.ones(n, dtype=bool), kinematics)
        target_pose = _joint_to_eef(executed, np.ones(n, dtype=bool), kinematics)
        delta = np.zeros_like(executed, dtype=np.float32)
        covered = 0
        for start in range(0, max(0, n - action_steps), action_steps):
            window = slice(start, start + action_steps)
            anchor = state_pose[start]
            # Same anchor for the whole window: policy chunks are relative to
            # the state at the decision frame, not per-frame deltas.
            for offset in range(action_steps):
                index = start + offset
                delta[index] = eef_pose_delta(anchor, target_pose[index])
            covered += action_steps
        raw["frames/executed_action"][:] = delta
        raw.attrs["behavior_channel_converted"] = (
            f"joint_targets_to_relative_eef_v1 action_steps={action_steps}"
        )
        return {"frames": n, "covered": covered, "tail": n - covered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, default=30)
    parser.add_argument("--pattern", default="*.raw.h5")
    args = parser.parse_args()
    kinematics = PiperForwardKinematics(
        urdf_path=DEFAULT_PIPER_URDF, frames=("gripper_center",)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_root.rglob(args.pattern))
    for path in files:
        stats = _convert_episode(
            path, args.output_root / path.name, kinematics, args.action_steps
        )
        print(f"{path.name}: {stats}")
    print(f"converted {len(files)} episodes -> {args.output_root}")


if __name__ == "__main__":
    main()
