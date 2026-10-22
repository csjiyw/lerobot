#!/usr/bin/env python3
"""Rebuild deployment-time-aligned decision streams from raw archives.

At rollout speed 0.5, a 30-waypoint model chunk is phase-resampled to 60 robot
ticks.  Training therefore needs obs(t), a 30-waypoint inverse-resampling of
the actual 60-tick command path, and next_obs(t+60).  Merely pairing a 30-tick
path with next_obs(t+30) teaches a different transition from deployment.

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
    parser.add_argument(
        "--raw-root",
        type=Path,
        action="append",
        required=True,
        help="Raw episode directory; repeat to reconstruct several collection batches in one manifest.",
    )
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
    parser.add_argument(
        "--rollout-speed",
        type=float,
        default=1.0,
        help="Deployment retiming factor; 0.5 maps a 30-waypoint model chunk to 60 raw frames.",
    )
    parser.add_argument("--pattern", default="*.raw.h5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def execution_frame_span(action_steps: int, rollout_speed: float) -> int:
    """Mirror ``rlt_online.resample_action_chunk`` without importing robot code."""

    if action_steps <= 0:
        raise ValueError("action_steps must be positive")
    if not np.isfinite(rollout_speed) or not 0.0 < rollout_speed <= 1.0:
        raise ValueError("rollout_speed must be in (0, 1]")
    return max(2, int(round(action_steps / float(rollout_speed))))


def aligned_starts(
    frame_count: int, action_steps: int, rollout_speed: float = 1.0
) -> np.ndarray:
    """Starts whose physical execution span and following observation exist."""

    span = execution_frame_span(action_steps, rollout_speed)
    # Need a next decision/observation after the executed action window.
    return np.arange(0, max(0, frame_count - span), span, dtype=np.int64)


def phase_resample(values: np.ndarray, output_steps: int) -> np.ndarray:
    """Least-squares inverse of deployment's piecewise-linear retiming.

    Directly interpolating 60 recorded samples back to 30 smooths every
    original knot because most knot phases do not coincide with a 60-point
    grid.  Solving the known deployment interpolation matrix instead exactly
    recovers model chunks that were retimed by ``resample_action_chunk`` and
    gives the closest representable model chunk for a human path.
    """

    array = np.asarray(values)
    if array.ndim != 2 or len(array) < 2 or output_steps < 2:
        raise ValueError("phase_resample expects a (T,D) path and at least two outputs")
    model_phase = np.linspace(0.0, 1.0, output_steps, dtype=np.float64)
    physical_phase = np.linspace(0.0, 1.0, len(array), dtype=np.float64)
    interpolation = np.stack(
        [
            np.interp(physical_phase, model_phase, np.eye(output_steps)[column])
            for column in range(output_steps)
        ],
        axis=1,
    )
    recovered = np.linalg.lstsq(interpolation, array.astype(np.float64), rcond=None)[0]
    # The online retimer preserves endpoints exactly; retain that contract for
    # arbitrary human paths as well as actor-generated paths.
    recovered[0] = array[0]
    recovered[-1] = array[-1]
    return recovered.astype(array.dtype)


def phase_resample_mask(mask: np.ndarray, output_steps: int) -> np.ndarray:
    """Nearest-phase ownership mask for the corresponding resampled actions."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    indices = np.rint(np.linspace(0, len(values) - 1, output_steps)).astype(np.int64)
    return values[indices]


# Compatibility alias used by older callers/tests.
def ten_tick_starts(frame_count: int, action_steps: int) -> np.ndarray:
    return aligned_starts(frame_count, action_steps, 1.0)


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
    rollout_speed: float = 1.0,
) -> dict:
    stem = raw_path.name.removesuffix(".raw.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    if decision_path.exists():
        raise FileExistsError(decision_path)
    dimensions = RLTDimensions(**runtime.dimensions)
    writer: RLTDecisionWriter | None = None
    runtime.begin_episode(workflow_mode=2, allow_actor=False, actor_exploration_std=0.0)
    # A normal reference-only online episode deliberately restores the full
    # Stage-1 horizon.  Offline reconstruction is different: ``action_steps``
    # defines the future Stage-2 actor horizon, so both its reference slice
    # and action normalizer must retain the explicit shorter request.
    runtime.execution_action_steps = int(runtime._requested_action_steps)
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
            frame_span = execution_frame_span(dimensions.action_steps, rollout_speed)
            starts = aligned_starts(len(state), dimensions.action_steps, rollout_speed)
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
                    "action_source_verified": True,
                    "rollout_speed_scale": float(rollout_speed),
                    "execution_frame_span": int(frame_span),
                    "behavior_chunk_source": "raw_executed_deployment_span_phase_resampled_v1",
                }
            )
            for number, start in enumerate(starts, 1):
                window = slice(int(start), int(start + frame_span))
                if not valid[window].all():
                    continue
                human_mask = phase_resample_mask(human[window], dimensions.action_steps)
                is_human = bool(human_mask.any())
                inference = runtime.predict(
                    state=eef_state[start],
                    images={camera: _jpeg(images, int(start)) for camera, images in image_sets.items()},
                    task=str(raw.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                behavior_path = phase_resample(
                    eef_executed[window], dimensions.action_steps
                )
                behavior_eef = chunk_to_relative(eef_state[start], behavior_path)
                behavior = runtime.normalize_actions(behavior_eef)
                # The replay action is the action actually executed.  Keep the
                # frozen VLA proposal separately as the actor BC fallback.
                writer.append(
                    z_rl=inference["z_rl"],
                    state=inference["state"],
                    reference_actions=inference["reference_actions"],
                    policy_actions=behavior,
                    executed_actions=behavior,
                    action_source="human" if is_human else "unknown",
                    is_intervention=is_human,
                    action_human_mask=human_mask,
                    bridge_sequence=int(sequence[start]),
                    time_ns=int(timestamps[start]),
                    episode_frame_index=int(start),
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
    raw_roots = [path.expanduser().resolve() for path in args.raw_root]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for root in raw_roots for path in root.glob(args.pattern))
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate raw episode names across --raw-root values: {duplicates}")
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
            row = reconstruct_one(
                path, output_root, runtime, kinematics, rollout_speed=args.rollout_speed
            )
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
                    "raw_roots": [str(path) for path in raw_roots],
                    "contract": "deployment_span_phase_resampled_v1",
                    "rollout_speed": float(args.rollout_speed),
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
