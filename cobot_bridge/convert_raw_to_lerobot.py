"""Convert one finalized PC-side Cobot Magic raw HDF5 episode to LeRobot."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from .record_raw import (
    COLLECTION_TYPE_NAMES,
    CONTROL_HUMAN,
    CONTROL_POLICY,
    OUTCOME_NAMES,
    OUTCOME_UNKNOWN,
    PHASE_TERMINAL,
    SUPPORTED_FORMATS,
    classify_control_source,
    default_phase,
)


LOG = logging.getLogger("cobot_bridge.convert_raw_to_lerobot")
JOINT_NAMES = [f"joint{index}" for index in range(7)]
HYBRID_STATE_T3_ACTION_SOURCE = "puppet_state_t3_arm_cmd_gripper"
HYBRID_STATE_HORIZON = 3


def _decode(jpeg: np.ndarray) -> np.ndarray:
    bgr = cv2.imdecode(np.asarray(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("invalid JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _create_dataset(
    args: argparse.Namespace, raw: h5py.File, cameras: list[str], target_fps: int
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image_dtype = "video" if args.use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "observation.master_state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "rlt.reference_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "rlt.policy_action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "rlt.human_action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "rlt.executed_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "observation.robot_timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["timestamp_ns"],
        },
        "observation.teach_active": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["teach_active"],
        },
        "observation.is_policy": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_policy"],
        },
        "observation.is_human_intervention": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_human_intervention"],
        },
        "observation.control_source": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["control_source"],
        },
        "observation.intervention_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["intervention_id"],
        },
        "observation.teach_event": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["teach_event"],
        },
        "observation.effective_mode": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["effective_mode"],
        },
        "observation.control_epoch": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["control_epoch"],
        },
        "rlt.reference_action_valid": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["reference_action_valid"],
        },
        "rlt.policy_action_valid": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["policy_action_valid"],
        },
        "rlt.human_action_valid": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["human_action_valid"],
        },
        "rlt.executed_action_valid": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["executed_action_valid"],
        },
        "rlt.phase": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
        "rlt.cycle_id": {"dtype": "int64", "shape": (1,), "names": ["cycle_id"]},
        "rlt.learning_mask": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["learning_mask"],
        },
        "rlt.intervention_event": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["intervention_event"],
        },
        "rlt.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "rlt.reward_valid": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["reward_valid"],
        },
        "rlt.done": {"dtype": "int64", "shape": (1,), "names": ["done"]},
        "rlt.success": {"dtype": "int64", "shape": (1,), "names": ["success"]},
        # RLinf STEAM reads one representative row per rollout episode and
        # requires this episode-level label to be present on every frame.
        "is_success": {"dtype": "int64", "shape": (1,), "names": ["is_success"]},
        "rlt.collection_type": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["collection_type"],
        },
    }
    for camera in cameras:
        image = _decode(raw[f"images/{camera}/jpeg"][0])
        key = f"observation.images.camera_{camera[0] if camera != 'front' else 'f'}"
        features[key] = {
            "dtype": image_dtype,
            "shape": tuple(image.shape),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=target_fps,
        root=args.root,
        robot_type="cobot_magic_piper",
        features=features,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )
    if not hasattr(dataset, "img_format"):
        dataset.img_format = "png"
    return dataset


def derive_control_labels(
    teach_active: np.ndarray, effective_mode: np.ndarray
) -> dict[str, np.ndarray]:
    teach = np.asarray(teach_active, dtype=bool)
    modes = np.asarray(effective_mode, dtype=np.int64)
    if teach.shape != modes.shape or teach.ndim != 1:
        raise ValueError("teach_active and effective_mode must be same-length vectors")
    sources = np.asarray(
        [
            classify_control_source(teach_active=bool(active), effective_mode=int(mode))
            for active, mode in zip(teach, modes, strict=True)
        ],
        dtype=np.int64,
    )
    events = np.zeros(len(teach), dtype=np.int64)
    if len(teach):
        events[0] = int(teach[0])
        events[1:] = teach[1:].astype(np.int64) - teach[:-1].astype(np.int64)
    human = sources == CONTROL_HUMAN
    intervention_events = np.zeros(len(teach), dtype=np.int64)
    if len(human):
        intervention_events[0] = int(human[0])
        intervention_events[1:] = human[1:].astype(np.int64) - human[:-1].astype(np.int64)
    intervention_ids = np.zeros(len(teach), dtype=np.int64)
    current_id = 0
    for index, event in enumerate(intervention_events):
        if event == 1:
            current_id += 1
        if human[index] or event == -1:
            intervention_ids[index] = current_id
    return {
        "control_source": sources,
        "is_policy": (sources == CONTROL_POLICY).astype(np.float32),
        "is_human_intervention": (sources == CONTROL_HUMAN).astype(np.float32),
        "intervention_id": intervention_ids,
        "teach_event": events,
        "intervention_event": intervention_events,
    }


def _read_rlt_fields(
    raw: h5py.File, labels: dict[str, np.ndarray], count: int
) -> dict[str, np.ndarray]:
    frames = raw["frames"]
    zeros_action = np.zeros((count, 7), dtype=np.float32)
    zeros = np.zeros(count, dtype=np.uint8)
    executed = np.asarray(frames["puppet_command"][:], dtype=np.float32)
    executed_valid = np.asarray(frames["puppet_command_valid"][:], dtype=np.uint8)
    executed = np.where(executed_valid[:, None].astype(bool), executed, 0.0).astype(np.float32)
    source = np.asarray(labels["control_source"], dtype=np.int64)
    fallback = {
        "reference_action": zeros_action.copy(),
        "reference_action_valid": zeros.copy(),
        "policy_action": np.where((source == CONTROL_POLICY)[:, None], executed, 0.0),
        "policy_action_valid": ((source == CONTROL_POLICY) & executed_valid.astype(bool)).astype(np.uint8),
        "human_action": np.where(
            (source == CONTROL_HUMAN)[:, None],
            np.asarray(frames["master_position"][:], dtype=np.float32),
            0.0,
        ),
        "human_action_valid": (source == CONTROL_HUMAN).astype(np.uint8),
        "executed_action": executed,
        "executed_action_valid": executed_valid,
        "phase": np.asarray([default_phase(int(value)) for value in source], dtype=np.int8),
        "cycle_id": np.full(count, -1, dtype=np.int32),
        "learning_mask": np.isin(source, (CONTROL_POLICY, CONTROL_HUMAN)).astype(np.uint8),
        "intervention_event": np.asarray(labels["intervention_event"], dtype=np.int8),
        "reward": np.zeros(count, dtype=np.float32),
        "reward_valid": zeros.copy(),
        "done": zeros.copy(),
        "success": np.full(count, OUTCOME_UNKNOWN, dtype=np.int8),
    }
    for name in tuple(fallback):
        if name in frames:
            fallback[name] = np.asarray(frames[name][:])
    if count and "done" not in frames:
        fallback["done"][-1] = 1
    return fallback


def trim_episode_bounds(
    timestamps_ns: np.ndarray,
    trajectory: np.ndarray,
    teach_active: np.ndarray,
    *,
    collection_type: str,
    joint_tolerance: float,
    gripper_tolerance: float,
    settle_seconds: float,
    keep_final_steady: bool = False,
    head_tolerance: float = 0.02,
) -> tuple[int, int, dict[str, Any]]:
    """Return [start, end) after removing IL wait time and stationary holds.

    With ``keep_final_steady=True`` only the head (initial wait + initial
    steady) is removed; the tail — including the return-to-ready motion and
    any final steady hold — is kept verbatim (去头不去尾).
    """

    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    teach_active = np.asarray(teach_active, dtype=bool)
    if timestamps_ns.ndim != 1 or trajectory.shape != (len(timestamps_ns), 7):
        raise ValueError("timestamps and trajectory must have shapes (N,) and (N, 7)")
    if teach_active.shape != timestamps_ns.shape:
        raise ValueError("teach_active must have shape (N,)")
    if len(timestamps_ns) < 2 or np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("episode timestamps must be strictly increasing")
    if joint_tolerance <= 0 or gripper_tolerance <= 0 or settle_seconds < 0:
        raise ValueError("trim tolerances must be positive and settle seconds non-negative")

    base_start = 0
    base_end = len(timestamps_ns)
    if collection_type.upper() == "IL":
        teach_indices = np.flatnonzero(teach_active)
        if not len(teach_indices):
            raise ValueError("IL episode has no Teach-active demonstration frames")
        base_start = int(teach_indices[0])
        base_end = int(teach_indices[-1]) + 1

    if keep_final_steady:
        # Head-only trim (去头不去尾): never cut the tail.  The tail still
        # provides termination supervision (return-to-ready + final steady).
        # Head departure uses a cumulative L2 distance with its own tolerance
        # (default 2 cm) so slow sub-1 cm drift at the start is still removed,
        # matching the v3 spec in NOTAIL_REBUILD_RECORD.md §4.3.
        segment = trajectory[base_start:base_end]
        initial_pose = np.median(segment[: min(3, len(segment))], axis=0)
        departed_initial = np.flatnonzero(
            np.linalg.norm(segment - initial_pose, axis=1) > head_tolerance
        )
        start = base_start
        if base_end - base_start >= 3 and len(departed_initial):
            first_motion = base_start + int(departed_initial[0])
            settle_ns = int(round(settle_seconds * 1e9))
            start_time = max(int(timestamps_ns[base_start]), int(timestamps_ns[first_motion]) - settle_ns)
            start = max(base_start, int(np.searchsorted(timestamps_ns, start_time, side="left")))
            start = min(start, base_end - 2)
        seconds = timestamps_ns.astype(np.float64) / 1e9
        info: dict[str, Any] = {
            "original_frames": len(timestamps_ns),
            "trimmed_frames": base_end - start,
            "start_index": start,
            "end_index_exclusive": base_end,
            "initial_wait_removed_s": float(seconds[base_start] - seconds[0]),
            "initial_steady_removed_s": float(seconds[start] - seconds[base_start]),
            "final_steady_removed_s": 0.0,
            "post_teach_removed_s": 0.0,
        }
        return start, base_end, info

    segment = trajectory[base_start:base_end]
    tolerance = np.asarray([joint_tolerance] * 6 + [gripper_tolerance])
    reference_count = min(3, len(segment))
    initial_pose = np.median(segment[:reference_count], axis=0)
    final_pose = np.median(segment[-reference_count:], axis=0)
    departed_initial = np.flatnonzero(np.any(np.abs(segment - initial_pose) > tolerance, axis=1))
    outside_final = np.flatnonzero(np.any(np.abs(segment - final_pose) > tolerance, axis=1))

    start = base_start
    end = base_end
    if len(departed_initial) and len(outside_final):
        first_motion = base_start + int(departed_initial[0])
        last_motion = base_start + min(int(outside_final[-1]) + 1, len(segment) - 1)
        settle_ns = int(round(settle_seconds * 1e9))
        start_time = max(int(timestamps_ns[base_start]), int(timestamps_ns[first_motion]) - settle_ns)
        end_time = min(int(timestamps_ns[base_end - 1]), int(timestamps_ns[last_motion]) + settle_ns)
        start = max(base_start, int(np.searchsorted(timestamps_ns, start_time, side="left")))
        end = min(base_end, int(np.searchsorted(timestamps_ns, end_time, side="right")))
    if end - start < 2:
        raise ValueError(f"stationary trim left too few frames: start={start} end={end}")

    seconds = timestamps_ns.astype(np.float64) / 1e9
    info: dict[str, Any] = {
        "original_frames": len(timestamps_ns),
        "trimmed_frames": end - start,
        "start_index": start,
        "end_index_exclusive": end,
        "initial_wait_removed_s": float(seconds[base_start] - seconds[0]),
        "initial_steady_removed_s": float(seconds[start] - seconds[base_start]),
        "final_steady_removed_s": float(seconds[base_end - 1] - seconds[end - 1]),
        "post_teach_removed_s": float(seconds[-1] - seconds[base_end - 1]),
    }
    return start, end, info


def sample_frame_indices(
    timestamps_ns: np.ndarray,
    *,
    start: int,
    end: int,
    raw_fps: int,
    target_fps: int,
) -> np.ndarray:
    if target_fps == raw_fps:
        return np.arange(start, end, dtype=np.int64)
    clipped_time = (np.asarray(timestamps_ns[start:end], dtype=np.int64) - timestamps_ns[start]) / 1e9
    targets = np.arange(0.0, clipped_time[-1] + 0.5 / target_fps, 1.0 / target_fps)
    right = np.searchsorted(clipped_time, targets, side="left")
    right = np.clip(right, 0, len(clipped_time) - 1)
    left = np.maximum(0, right - 1)
    choose_left = np.abs(clipped_time[left] - targets) <= np.abs(clipped_time[right] - targets)
    local = np.unique(np.where(choose_left, left, right))
    return local.astype(np.int64) + start


def hybrid_state_t3_action(
    puppet_position: np.ndarray,
    executed_action: np.ndarray,
    executed_action_valid: np.ndarray,
    *,
    index: int,
    future_index: int,
) -> np.ndarray:
    """Use achieved future arm pose while preserving the gripper command."""

    puppet = np.asarray(puppet_position, dtype=np.float32)
    executed = np.asarray(executed_action, dtype=np.float32)
    valid = np.asarray(executed_action_valid, dtype=bool)
    if puppet.ndim != 2 or puppet.shape[1] != 7 or executed.shape != puppet.shape:
        raise ValueError("puppet_position and executed_action must both have shape (N, 7)")
    if valid.shape != (len(puppet),):
        raise ValueError("executed_action_valid must have shape (N,)")
    if not 0 <= index < future_index < len(puppet):
        raise ValueError(f"invalid t+3 indices: index={index} future_index={future_index}")
    if not valid[index]:
        raise ValueError(f"executed gripper command is invalid at raw frame {index}")
    action = executed[index].copy()
    action[:6] = puppet[future_index, :6]
    if not np.isfinite(action).all():
        raise ValueError(f"hybrid action contains non-finite values at raw frame {index}")
    return action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--task",
        dest="task_override",
        help="canonical task text to write for every converted frame; raw task text is preserved in the manifest",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="offline output FPS; defaults to the raw recording FPS",
    )
    parser.add_argument(
        "--action-source",
        choices=(
            "master_state",
            "puppet_command",
            "reference_action",
            "policy_action",
            "human_action",
            "executed_action",
            HYBRID_STATE_T3_ACTION_SOURCE,
        ),
        default="executed_action",
    )
    parser.add_argument("--use-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--trim-stationary-holds", action="store_true")
    parser.add_argument(
        "--keep-final-steady",
        action="store_true",
        help="head-only trim (去头不去尾): keep the tail (return-to-ready + final steady) instead of trimming it",
    )
    parser.add_argument("--trim-joint-tolerance", type=float, default=0.01)
    parser.add_argument("--trim-gripper-tolerance", type=float, default=0.002)
    parser.add_argument("--trim-settle-seconds", type=float, default=0.5)
    parser.add_argument(
        "--trim-head-tolerance",
        type=float,
        default=0.02,
        help="L2 (cumulative) distance for head-only trim departure detection (default 0.02 = 2 cm)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # When this script is launched through an environment Python's absolute
    # path, its packages are active but the environment's ``bin`` directory is
    # not necessarily on PATH. LeRobot's video encoder invokes ``ffmpeg`` as a
    # subprocess, so make the executable colocated with this Python visible.
    environment_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    if args.use_videos and shutil.which("ffmpeg") is None:
        raise FileNotFoundError(f"ffmpeg not found on PATH, including {environment_bin}")
    if args.root.exists():
        raise FileExistsError(args.root)
    dataset = None
    expected_cameras: list[str] | None = None
    expected_fps: int | None = None
    total_raw_frames = 0
    total_output_frames = 0
    manifest: dict[str, Any] = {
        "repo_id": args.repo_id,
        "task_override": args.task_override,
        "action_source": args.action_source,
        "action_rule": (
            {
                "observation": "puppet_position[t]",
                "arm_action": "puppet_position[t+3, 0:6]",
                "gripper_action": "executed_action[t, 6]",
                "horizon_frames": HYBRID_STATE_HORIZON,
                "clock_hz": 30,
                "tail_policy": "drop_last_3_frames_per_episode",
            }
            if args.action_source == HYBRID_STATE_T3_ACTION_SOURCE
            else None
        ),
        "trim_stationary_holds": args.trim_stationary_holds,
        "keep_final_steady": args.keep_final_steady,
        "trim_joint_tolerance": args.trim_joint_tolerance,
        "trim_gripper_tolerance": args.trim_gripper_tolerance,
        "trim_settle_seconds": args.trim_settle_seconds,
        "trim_head_tolerance": args.trim_head_tolerance,
        "episodes": [],
    }
    try:
        for episode_number, raw_path in enumerate(args.raw):
            with h5py.File(raw_path, "r") as raw:
                if raw.attrs.get("format") not in SUPPORTED_FORMATS:
                    raise ValueError(
                        f"unsupported raw format in {raw_path}: {raw.attrs.get('format')!r}"
                    )
                if not bool(raw.attrs.get("completed", False)):
                    raise ValueError(f"raw episode is not finalized: {raw_path}")
                count = int(raw.attrs["num_frames"])
                if count <= 0:
                    raise ValueError(f"raw episode is empty: {raw_path}")
                cameras = list(json.loads(raw.attrs["cameras"]))
                raw_task = str(raw.attrs["task"])
                task = raw_task if args.task_override is None else args.task_override.strip()
                if not task:
                    raise ValueError("task text must not be empty")
                raw_fps = int(raw.attrs["fps"])
                target_fps = raw_fps if args.fps is None else args.fps
                if target_fps <= 0 or target_fps > raw_fps:
                    raise ValueError(f"target fps must be in [1, {raw_fps}], got {target_fps}")
                if dataset is None:
                    dataset = _create_dataset(args, raw, cameras, target_fps)
                    expected_cameras = cameras
                    expected_fps = target_fps
                elif cameras != expected_cameras or target_fps != expected_fps:
                    raise ValueError(
                        f"episode schema mismatch in {raw_path}: cameras={cameras}, fps={target_fps}"
                    )

                source_time = np.asarray(raw["frames/pc_timestamp_ns"][:], dtype=np.int64)
                teach_active = np.asarray(raw["frames/teach_active"][:], dtype=np.uint8)
                effective_mode = np.asarray(raw["frames/effective_mode"][:], dtype=np.int64)
                raw_labels = derive_control_labels(teach_active, effective_mode)
                for name in (
                    "control_source",
                    "is_policy",
                    "is_human_intervention",
                    "intervention_id",
                    "teach_event",
                    "intervention_event",
                ):
                    path = f"frames/{name}"
                    if path in raw:
                        raw_labels[name] = np.asarray(raw[path][:])
                rlt = _read_rlt_fields(raw, raw_labels, count)
                puppet_position = np.asarray(
                    raw["frames/puppet_position"][:], dtype=np.float32
                )

                if args.trim_stationary_holds:
                    start, end, trim_info = trim_episode_bounds(
                        source_time,
                        np.asarray(raw["frames/master_position"][:], dtype=np.float64),
                        teach_active,
                        collection_type=str(raw.attrs.get("collection_type", "")),
                        joint_tolerance=args.trim_joint_tolerance,
                        gripper_tolerance=args.trim_gripper_tolerance,
                        settle_seconds=args.trim_settle_seconds,
                        keep_final_steady=args.keep_final_steady,
                        head_tolerance=args.trim_head_tolerance,
                    )
                else:
                    start, end = 0, count
                    trim_info = {
                        "original_frames": count,
                        "trimmed_frames": count,
                        "start_index": 0,
                        "end_index_exclusive": count,
                        "initial_wait_removed_s": 0.0,
                        "initial_steady_removed_s": 0.0,
                        "final_steady_removed_s": 0.0,
                        "post_teach_removed_s": 0.0,
                    }
                frame_indices = sample_frame_indices(
                    source_time,
                    start=start,
                    end=end,
                    raw_fps=raw_fps,
                    target_fps=target_fps,
                )
                sampled_frame_indices = frame_indices
                horizon_dropped_frames = 0
                if args.action_source == HYBRID_STATE_T3_ACTION_SOURCE:
                    if str(raw.attrs.get("collection_type", "")).upper() != "IL":
                        raise ValueError(
                            f"{HYBRID_STATE_T3_ACTION_SOURCE} is only valid for IL episodes"
                        )
                    if target_fps != 30:
                        raise ValueError(
                            f"{HYBRID_STATE_T3_ACTION_SOURCE} requires 30 FPS, got {target_fps}"
                        )
                    if len(sampled_frame_indices) <= HYBRID_STATE_HORIZON:
                        raise ValueError(
                            f"too few frames for t+3 labels in {raw_path}: "
                            f"{len(sampled_frame_indices)}"
                        )
                    frame_indices = sampled_frame_indices[:-HYBRID_STATE_HORIZON]
                    horizon_dropped_frames = HYBRID_STATE_HORIZON
                labels = derive_control_labels(
                    teach_active[frame_indices], effective_mode[frame_indices]
                )
                outcome = str(raw.attrs.get("outcome", "unknown")).lower()
                outcome_id = OUTCOME_NAMES.get(outcome, OUTCOME_UNKNOWN)
                episode_manifest = {
                    "episode_index": episode_number,
                    "raw": str(raw_path),
                    "raw_task": raw_task,
                    "task": task,
                    "outcome": outcome,
                    "raw_frames": count,
                    "output_frames": len(frame_indices),
                    "action_horizon_dropped_frames": horizon_dropped_frames,
                    **trim_info,
                }
                manifest["episodes"].append(episode_manifest)
                LOG.info(
                    "episode %d/%d: raw=%s frames=%d->%d start_removed=%.2fs end_removed=%.2fs",
                    episode_number + 1,
                    len(args.raw),
                    raw_path.name,
                    count,
                    len(frame_indices),
                    trim_info["initial_wait_removed_s"]
                    + trim_info["initial_steady_removed_s"],
                    trim_info["final_steady_removed_s"] + trim_info["post_teach_removed_s"],
                )

                for output_index, index in enumerate(frame_indices):
                    terminal = output_index == len(frame_indices) - 1
                    if args.action_source == HYBRID_STATE_T3_ACTION_SOURCE:
                        future_index = int(
                            sampled_frame_indices[output_index + HYBRID_STATE_HORIZON]
                        )
                        action = hybrid_state_t3_action(
                            puppet_position,
                            rlt["executed_action"],
                            rlt["executed_action_valid"],
                            index=int(index),
                            future_index=future_index,
                        )
                    elif args.action_source == "master_state":
                        action = np.asarray(
                            raw["frames/master_position"][index], dtype=np.float32
                        )
                    elif args.action_source == "puppet_command":
                        if bool(raw["frames/puppet_command_valid"][index]):
                            action = np.asarray(
                                raw["frames/puppet_command"][index], dtype=np.float32
                            )
                        else:
                            action = np.asarray(
                                raw["frames/puppet_position"][index], dtype=np.float32
                            )
                    else:
                        valid = bool(rlt[f"{args.action_source}_valid"][index])
                        if valid:
                            action = np.asarray(
                                rlt[args.action_source][index], dtype=np.float32
                            )
                        else:
                            # SAFE/ALIGN frames have no policy target. Use a neutral hold
                            # action while preserving the explicit validity and learning mask.
                            action = np.asarray(
                                raw["frames/puppet_position"][index], dtype=np.float32
                            )
                    frame: dict[str, Any] = {
                        "observation.state": np.asarray(
                            puppet_position[index], dtype=np.float32
                        ),
                        "observation.master_state": np.asarray(
                            raw["frames/master_position"][index], dtype=np.float32
                        ),
                        "action": action,
                        "rlt.reference_action": np.asarray(
                            rlt["reference_action"][index], dtype=np.float32
                        ),
                        "rlt.policy_action": np.asarray(
                            rlt["policy_action"][index], dtype=np.float32
                        ),
                        "rlt.human_action": np.asarray(
                            rlt["human_action"][index], dtype=np.float32
                        ),
                        "rlt.executed_action": np.asarray(
                            rlt["executed_action"][index], dtype=np.float32
                        ),
                        "observation.robot_timestamp_ns": np.asarray(
                            [raw["frames/bridge_time_ns"][index]], dtype=np.int64
                        ),
                        "observation.teach_active": np.asarray(
                            [teach_active[index]], dtype=np.float32
                        ),
                        "observation.is_policy": np.asarray(
                            [labels["is_policy"][output_index]], dtype=np.float32
                        ),
                        "observation.is_human_intervention": np.asarray(
                            [labels["is_human_intervention"][output_index]], dtype=np.float32
                        ),
                        "observation.control_source": np.asarray(
                            [labels["control_source"][output_index]], dtype=np.int64
                        ),
                        "observation.intervention_id": np.asarray(
                            [labels["intervention_id"][output_index]], dtype=np.int64
                        ),
                        "observation.teach_event": np.asarray(
                            [labels["teach_event"][output_index]], dtype=np.int64
                        ),
                        "observation.effective_mode": np.asarray(
                            [effective_mode[index]], dtype=np.int64
                        ),
                        "observation.control_epoch": np.asarray(
                            [raw["frames/control_epoch"][index]], dtype=np.int64
                        ),
                        "rlt.reference_action_valid": np.asarray(
                            [rlt["reference_action_valid"][index]], dtype=np.int64
                        ),
                        "rlt.policy_action_valid": np.asarray(
                            [rlt["policy_action_valid"][index]], dtype=np.int64
                        ),
                        "rlt.human_action_valid": np.asarray(
                            [rlt["human_action_valid"][index]], dtype=np.int64
                        ),
                        "rlt.executed_action_valid": np.asarray(
                            [rlt["executed_action_valid"][index]], dtype=np.int64
                        ),
                        "rlt.phase": np.asarray(
                            [PHASE_TERMINAL if terminal else rlt["phase"][index]],
                            dtype=np.int64,
                        ),
                        "rlt.cycle_id": np.asarray(
                            [rlt["cycle_id"][index]], dtype=np.int64
                        ),
                        "rlt.learning_mask": np.asarray(
                            [0 if terminal else rlt["learning_mask"][index]], dtype=np.int64
                        ),
                        "rlt.intervention_event": np.asarray(
                            [labels["intervention_event"][output_index]], dtype=np.int64
                        ),
                        "rlt.reward": np.asarray(
                            [
                                float(outcome == "success")
                                if terminal
                                else rlt["reward"][index]
                            ],
                            dtype=np.float32,
                        ),
                        "rlt.reward_valid": np.asarray(
                            [
                                int(outcome in ("success", "failure"))
                                if terminal
                                else rlt["reward_valid"][index]
                            ],
                            dtype=np.int64,
                        ),
                        "rlt.done": np.asarray(
                            [1 if terminal else rlt["done"][index]], dtype=np.int64
                        ),
                        "rlt.success": np.asarray(
                            [outcome_id if terminal else rlt["success"][index]],
                            dtype=np.int64,
                        ),
                        "is_success": np.asarray(
                            [int(outcome == "success")], dtype=np.int64
                        ),
                        "rlt.collection_type": np.asarray(
                            [
                                COLLECTION_TYPE_NAMES.get(
                                    str(raw.attrs.get("collection_type", "")).upper(), -1
                                )
                            ],
                            dtype=np.int64,
                        ),
                        "task": task,
                    }
                    for camera in cameras:
                        key = (
                            f"observation.images.camera_{camera[0] if camera != 'front' else 'f'}"
                        )
                        frame[key] = _decode(raw[f"images/{camera}/jpeg"][index])
                    dataset.add_frame(frame)
                dataset.save_episode()
                total_raw_frames += count
                total_output_frames += len(frame_indices)
    finally:
        if dataset is not None:
            dataset.finalize()
    if dataset is None:
        raise ValueError("no raw episodes were provided")
    manifest["total_raw_frames"] = total_raw_frames
    manifest["total_output_frames"] = total_output_frames
    manifest["fps"] = expected_fps
    manifest_path = args.root / "meta" / "cobot_magic_conversion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    LOG.info(
        "conversion complete: episodes=%d raw_frames=%d output_frames=%d fps=%d root=%s",
        len(args.raw),
        total_raw_frames,
        total_output_frames,
        expected_fps,
        args.root,
    )


if __name__ == "__main__":
    main()
