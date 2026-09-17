#!/usr/bin/env python3
"""Rebuild RLinf-compatible 10-tick decision streams directly from raw archives.

Old dashboard logs were produced by 30-tick open-loop rollout.  Their adjacent
decisions are therefore 30 policy ticks apart even when the stored action
chunk happens to have shape ``(10, 7)``.  This offline tool ignores those
decision logs and samples authoritative 30 Hz raw records at t, t+10, ... .
Thus each stored transition has exactly the Stage-2 action horizon between
``obs`` and ``next_obs``.

It writes a new decision directory only; use ``build_recap_dense_replay`` on
the result to attach fixed RECAP potential rewards.  No bridge is opened.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import h5py
import numpy as np

from .convert_lerobot_joint_to_eef import _joint_to_eef
from .piper_eef_metrics import PiperForwardKinematics
from .rlt_policy_server import RLTPolicyRuntime, _decode_jpeg
from .rlt_stage2_data import RLTDecisionWriter, RLTDimensions
from lerobot.policies.smolvla.relative_eef import chunk_to_relative


LOG = logging.getLogger("cobot_bridge.reconstruct_rlt_10tick_from_raw")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--action-steps",
        type=int,
        default=10,
        help=(
            "Contiguous Stage-2 execution horizon. It may be shorter than the "
            "frozen Stage-1 reference horizon; default 10 implements ref30/actor10."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pattern", default="*.raw.h5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def ten_tick_starts(frame_count: int, action_steps: int) -> np.ndarray:
    """Starts whose action window and following observation both exist."""

    if action_steps <= 0:
        raise ValueError("action_steps must be positive")
    # Need a next decision/observation after the executed action window.
    return np.arange(0, max(0, frame_count - action_steps), action_steps, dtype=np.int64)


def _jpeg(dataset: h5py.Dataset, index: int) -> np.ndarray:
    return _decode_jpeg(bytes(np.asarray(dataset[index], dtype=np.uint8)))


def _outcome(raw: h5py.File, path: Path) -> str:
    value = str(raw.attrs.get("outcome", "")).lower()
    if value in ("success", "failure"):
        return value
    sidecar = path.with_name(path.name + ".result")
    if sidecar.is_file():
        value = sidecar.read_text().strip().lower()
    if value not in ("success", "failure"):
        raise ValueError(f"{path.name}: missing success/failure label")
    return value


def reconstruct_one(
    raw_path: Path,
    output_root: Path,
    runtime: RLTPolicyRuntime,
    kinematics: PiperForwardKinematics,
) -> dict:
    stem = raw_path.name.removesuffix(".raw.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    if decision_path.exists():
        raise FileExistsError(decision_path)
    dimensions = RLTDimensions(**runtime.dimensions)
    writer: RLTDecisionWriter | None = None
    runtime.begin_episode(workflow_mode=2, allow_actor=False, actor_exploration_std=0.0)
    try:
        with h5py.File(raw_path, "r") as raw:
            if not bool(raw.attrs.get("completed", False)):
                raise ValueError(f"{raw_path.name}: raw archive is incomplete")
            cameras = tuple(json.loads(str(raw.attrs["cameras"])))
            if cameras != runtime.cameras:
                raise ValueError(f"{raw_path.name}: camera mismatch {cameras} != {runtime.cameras}")
            executed = np.asarray(raw["frames/executed_action"], dtype=np.float32)
            valid = np.asarray(raw["frames/executed_action_valid"], dtype=bool)
            state = np.asarray(raw["frames/puppet_position"], dtype=np.float32)
            sequence = np.asarray(raw["frames/sequence"], dtype=np.int64)
            timestamps = np.asarray(raw["frames/pc_timestamp_ns"], dtype=np.int64)
            human = np.asarray(raw["frames/is_human_intervention"], dtype=bool)
            if not (len(executed) == len(valid) == len(state) == len(sequence) == len(timestamps) == len(human)):
                raise ValueError(f"{raw_path.name}: inconsistent raw frame arrays")
            starts = ten_tick_starts(len(state), dimensions.action_steps)
            if len(starts) < 2:
                raise ValueError(f"{raw_path.name}: fewer than two complete 10-tick decisions")
            image_sets = {camera: raw[f"images/{camera}/jpeg"] for camera in cameras}
            # The raw archive records Piper joint feedback and sent joint
            # commands.  The current 073 policy was trained on absolute TCP
            # EEF state and chunk-anchor-relative EEF actions.  Project both
            # streams before invoking the saved processor; normalizing raw
            # joint radians with EEF statistics is a silent but catastrophic
            # space mismatch (e.g. normalized values around 50 instead of 1).
            valid_state = np.ones(len(state), dtype=bool)
            eef_state = _joint_to_eef(state, valid_state, kinematics)
            eef_executed = _joint_to_eef(executed, valid, kinematics)
            writer = RLTDecisionWriter(
                decision_path,
                dimensions=dimensions,
                stage1_checkpoint=str(runtime.stage1_path),
                task=str(raw.attrs["task"]),
                workflow_mode=2,
                side=str(raw.attrs.get("side", "right")),
                collection_phase="actor",
            )
            writer.file.attrs.update(
                {
                    "raw_episode_path": str(raw_path.resolve()),
                    "reconstruction": "raw_contiguous_10tick_v1",
                    "raw_state_action_space": "piper_joint_radians",
                    "model_state_action_space": "gripper_center_eef_anchor_relative",
                    "source_fps": 30.0,
                }
            )
            for number, start in enumerate(starts, 1):
                window = slice(int(start), int(start + dimensions.action_steps))
                if not valid[window].all():
                    continue
                is_human = bool(human[window].any())
                inference = runtime.predict(
                    state=eef_state[start],
                    images={camera: _jpeg(images, int(start)) for camera, images in image_sets.items()},
                    task=str(raw.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                behavior_eef = chunk_to_relative(eef_state[start], eef_executed[window])
                behavior = runtime.normalize_actions(behavior_eef)
                # The replay action is the action actually executed.  Keep the
                # frozen VLA proposal separately as the actor BC fallback.
                writer.append(
                    z_rl=inference["z_rl"],
                    state=inference["state"],
                    reference_actions=inference["reference_actions"],
                    policy_actions=behavior,
                    executed_actions=behavior,
                    action_source="human" if is_human else "actor",
                    is_intervention=is_human,
                    bridge_sequence=int(sequence[start]),
                    time_ns=int(timestamps[start]),
                )
                if number == 1 or number % 50 == 0:
                    LOG.info("%s decision=%d/%d", stem, number, len(starts))
            if writer.count < 2:
                raise ValueError(f"{raw_path.name}: fewer than two valid 10-tick windows")
            outcome = _outcome(raw, raw_path)
            writer.close(outcome=outcome)
            writer = None
        return {"raw": str(raw_path), "decision": str(decision_path), "outcome": outcome}
    except BaseException:
        if writer is not None:
            writer.abort()
        decision_path.unlink(missing_ok=True)
        raise
    finally:
        runtime.end_episode()


def main() -> None:
    args = _parser().parse_args()
    # This utility is deliberately offline-only: all weights and processors
    # come from the supplied Stage-1 checkpoint plus the local HF cache.
    # Prevent optional Hub metadata checks from making reconstruction depend
    # on network availability.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(raw_root.glob(args.pattern))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no raw archives matched {args.pattern!r}")
    runtime = RLTPolicyRuntime(
        stage1_checkpoint=args.stage1_checkpoint.expanduser().resolve(),
        stage2_checkpoint=output_root / ".no_actor.pt",
        device=args.device,
        cameras=("front", "right"),
        action_steps=args.action_steps,
    )
    if not bool(getattr(runtime.policy.config, "use_relative_eef_actions", False)):
        raise ValueError(
            "raw joint reconstruction currently requires a relative-EEF policy checkpoint"
        )
    kinematics = PiperForwardKinematics()
    complete, failed = [], []
    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        try:
            row = reconstruct_one(path, output_root, runtime, kinematics)
            complete.append(row)
            LOG.info("episode=%d/%d DONE %s", index, len(paths), path.name)
        except Exception as error:
            LOG.exception("episode=%d/%d FAIL %s", index, len(paths), path.name)
            failed.append({"raw": str(path), "error": str(error)})
            if args.fail_fast:
                raise
        (output_root / "manifest.json").write_text(
            json.dumps(
                {
                    "stage1_checkpoint": str(runtime.stage1_path),
                    "contract": "raw_contiguous_10tick_v1",
                    "target": len(paths),
                    "done": len(complete),
                    "failed": failed,
                    "elapsed_seconds": time.monotonic() - started,
                    "items": complete,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
