"""Record a loss-resistant raw Cobot Magic episode on the PC.

The recorder writes one HDF5 file incrementally. Camera payloads stay as the
JPEG bytes produced by the robot cameras; no RGB decoding, LeRobot conversion,
or video encoding occurs in the live path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .client import CobotBridgeClient
from .remote_robot import CobotMagicRemoteRobot, RemoteObservation


LOG = logging.getLogger("cobot_bridge.record_raw")
STOP = False
FORMAT_V1 = "cobot_magic_raw_v1"
FORMAT = "cobot_magic_raw_v2"
SUPPORTED_FORMATS = (FORMAT_V1, FORMAT)
CONTROL_OTHER = 0
CONTROL_POLICY = 1
CONTROL_HUMAN = 2
CONTROL_ALIGNMENT = 3
CONTROL_SOURCE_NAMES = {
    "other": CONTROL_OTHER,
    "policy": CONTROL_POLICY,
    "human": CONTROL_HUMAN,
    "alignment": CONTROL_ALIGNMENT,
}

PHASE_OTHER = 0
PHASE_ALIGNMENT = 1
PHASE_POLICY = 2
PHASE_LOOP_TRANSITION = 3
PHASE_HUMAN = 4
PHASE_TERMINAL = 5
PHASE_NAMES = {
    "other": PHASE_OTHER,
    "alignment": PHASE_ALIGNMENT,
    "policy": PHASE_POLICY,
    "loop_transition": PHASE_LOOP_TRANSITION,
    "human": PHASE_HUMAN,
    "terminal": PHASE_TERMINAL,
}

OUTCOME_UNKNOWN = -1
OUTCOME_FAILURE = 0
OUTCOME_SUCCESS = 1
OUTCOME_ABORTED = 2
OUTCOME_NAMES = {
    "unknown": OUTCOME_UNKNOWN,
    "failure": OUTCOME_FAILURE,
    "success": OUTCOME_SUCCESS,
    "aborted": OUTCOME_ABORTED,
}
COLLECTION_TYPE_NAMES = {"IL": 0, "RL": 1, "REPLAY": 2}


def classify_control_source(*, teach_active: bool, effective_mode: int) -> int:
    if teach_active or effective_mode == 3:
        return CONTROL_HUMAN
    if effective_mode in (1, 2):
        return CONTROL_POLICY
    if effective_mode == 5:
        return CONTROL_ALIGNMENT
    return CONTROL_OTHER


def default_phase(control_source: int) -> int:
    if control_source == CONTROL_ALIGNMENT:
        return PHASE_ALIGNMENT
    if control_source == CONTROL_POLICY:
        return PHASE_POLICY
    if control_source == CONTROL_HUMAN:
        return PHASE_HUMAN
    return PHASE_OTHER


def _optional_action(value: Any) -> tuple[np.ndarray, bool]:
    if value is None:
        return np.zeros(7, dtype=np.float32), False
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"action must contain 7 finite values, got shape={action.shape}")
    return action, True


def _stop(_signal: int, _frame: Any) -> None:
    global STOP
    STOP = True


def read_result_outcome(path: Path) -> str:
    outcome = path.read_text().strip().lower()
    if outcome not in ("success", "failure", "aborted", "discard"):
        raise ValueError(f"invalid outcome in result file {path}: {outcome!r}")
    return outcome


def _append(dataset: h5py.Dataset, value: Any) -> None:
    index = dataset.shape[0]
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


class RawEpisodeWriter:
    def __init__(
        self,
        path: Path,
        *,
        side: str,
        cameras: tuple[str, ...],
        task: str,
        fps: int,
        flush_frames: int,
        episode_id: int = 0,
        policy_id: str = "",
        collection_type: str = "IL",
    ) -> None:
        if collection_type not in COLLECTION_TYPE_NAMES:
            raise ValueError(f"invalid collection type: {collection_type!r}")
        self.path = path
        self.side = side
        self.cameras = cameras
        self.flush_frames = flush_frames
        self.file = h5py.File(path, "x", libver="latest")
        self.count = 0
        self.skipped = 0
        self.previous_teach_active = False
        self.previous_human_active = False
        self.intervention_count = 0

        self.file.attrs.update(
            {
                "format": FORMAT,
                "completed": False,
                "created_at_ns": time.time_ns(),
                "side": side,
                "cameras": json.dumps(cameras),
                "task": task,
                "fps": fps,
                "episode_id": int(episode_id),
                "policy_id": policy_id,
                "collection_type": collection_type,
                "collection_type_id": COLLECTION_TYPE_NAMES[collection_type],
                "collection_type_names": json.dumps(COLLECTION_TYPE_NAMES, sort_keys=True),
                "control_source_names": json.dumps(CONTROL_SOURCE_NAMES, sort_keys=True),
                "phase_names": json.dumps(PHASE_NAMES, sort_keys=True),
                "outcome_names": json.dumps(OUTCOME_NAMES, sort_keys=True),
                "outcome": "unknown",
                "outcome_id": OUTCOME_UNKNOWN,
            }
        )
        frames = self.file.create_group("frames")
        self.numeric = {
            "pc_timestamp_ns": frames.create_dataset(
                "pc_timestamp_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "bridge_time_ns": frames.create_dataset(
                "bridge_time_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "sequence": frames.create_dataset(
                "sequence", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "master_position": frames.create_dataset(
                "master_position", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "puppet_position": frames.create_dataset(
                "puppet_position", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "puppet_command": frames.create_dataset(
                "puppet_command", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "puppet_command_valid": frames.create_dataset(
                "puppet_command_valid", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "master_stamp_ns": frames.create_dataset(
                "master_stamp_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "puppet_stamp_ns": frames.create_dataset(
                "puppet_stamp_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "teach_active": frames.create_dataset(
                "teach_active", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "control_epoch": frames.create_dataset(
                "control_epoch", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "effective_mode": frames.create_dataset(
                "effective_mode", shape=(0,), maxshape=(None,), dtype="i1", chunks=True
            ),
            "control_source": frames.create_dataset(
                "control_source", shape=(0,), maxshape=(None,), dtype="i1", chunks=True
            ),
            "is_policy": frames.create_dataset(
                "is_policy", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "is_human_intervention": frames.create_dataset(
                "is_human_intervention", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "intervention_id": frames.create_dataset(
                "intervention_id", shape=(0,), maxshape=(None,), dtype="i4", chunks=True
            ),
            "teach_event": frames.create_dataset(
                "teach_event", shape=(0,), maxshape=(None,), dtype="i1", chunks=True
            ),
            "intervention_event": frames.create_dataset(
                "intervention_event", shape=(0,), maxshape=(None,), dtype="i1", chunks=True
            ),
            "episode_id": frames.create_dataset(
                "episode_id", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            ),
            "cycle_id": frames.create_dataset(
                "cycle_id", shape=(0,), maxshape=(None,), dtype="i4", chunks=True
            ),
            "phase": frames.create_dataset(
                "phase", shape=(0,), maxshape=(None,), dtype="i1", chunks=True
            ),
            "learning_mask": frames.create_dataset(
                "learning_mask", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "reference_action": frames.create_dataset(
                "reference_action", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "reference_action_valid": frames.create_dataset(
                "reference_action_valid", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "policy_action": frames.create_dataset(
                "policy_action", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "policy_action_valid": frames.create_dataset(
                "policy_action_valid", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "human_action": frames.create_dataset(
                "human_action", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "human_action_valid": frames.create_dataset(
                "human_action_valid", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "executed_action": frames.create_dataset(
                "executed_action", shape=(0, 7), maxshape=(None, 7), dtype="f4", chunks=True
            ),
            "executed_action_valid": frames.create_dataset(
                "executed_action_valid", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "reward": frames.create_dataset(
                "reward", shape=(0,), maxshape=(None,), dtype="f4", chunks=True
            ),
            "reward_valid": frames.create_dataset(
                "reward_valid", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "done": frames.create_dataset(
                "done", shape=(0,), maxshape=(None,), dtype="u1", chunks=True
            ),
            "success": frames.create_dataset(
                "success", shape=(0,), maxshape=(None,), dtype="i1", chunks=True
            ),
        }
        self.control_json = frames.create_dataset(
            "control_json",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        image_group = self.file.create_group("images")
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        self.images: dict[str, dict[str, h5py.Dataset]] = {}
        for camera in cameras:
            group = image_group.create_group(camera)
            self.images[camera] = {
                "jpeg": group.create_dataset(
                    "jpeg", shape=(0,), maxshape=(None,), dtype=jpeg_dtype, chunks=True
                ),
                "stamp_ns": group.create_dataset(
                    "stamp_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
                ),
            }

    def append(
        self,
        observation: RemoteObservation,
        *,
        reference_action: Any = None,
        policy_action: Any = None,
        phase: int | None = None,
        cycle_id: int = -1,
    ) -> None:
        raw = observation.raw
        master = raw["master"][self.side]
        puppet = raw["puppet"][self.side]
        command = observation.executed_puppet_action()
        command_valid = command is not None
        if command is None:
            command = np.full(7, np.nan, dtype=np.float32)

        teach_active = bool(observation.teach_active)
        effective_mode = int(observation.control.get("effective_mode_id", -1))
        teach_event = 0
        if teach_active and not self.previous_teach_active:
            teach_event = 1
        elif not teach_active and self.previous_teach_active:
            teach_event = -1
        control_source = classify_control_source(
            teach_active=teach_active, effective_mode=effective_mode
        )
        human_active = control_source == CONTROL_HUMAN
        intervention_event = 0
        if human_active and not self.previous_human_active:
            self.intervention_count += 1
            intervention_event = 1
        elif not human_active and self.previous_human_active:
            intervention_event = -1
        intervention_id = (
            self.intervention_count if human_active or intervention_event == -1 else 0
        )
        phase = default_phase(control_source) if phase is None else int(phase)
        if phase not in PHASE_NAMES.values():
            raise ValueError(f"invalid phase id: {phase}")

        reference, reference_valid = _optional_action(reference_action)
        policy, policy_valid = _optional_action(policy_action)
        executed, executed_valid = _optional_action(command if command_valid else None)
        human, human_valid = _optional_action(
            observation.master_state if control_source == CONTROL_HUMAN else None
        )
        # A plain policy/replay recorder can recover the accepted policy command
        # from robot feedback. Reference actions remain explicitly invalid unless
        # the policy runner supplies them.
        if not policy_valid and control_source == CONTROL_POLICY and executed_valid:
            policy = executed.copy()
            policy_valid = True
        learning_mask = int(phase in (PHASE_POLICY, PHASE_HUMAN))
        values = {
            "pc_timestamp_ns": time.time_ns(),
            "bridge_time_ns": int(raw.get("bridge_time_ns", 0)),
            "sequence": int(raw.get("sequence", 0)),
            "master_position": observation.master_state,
            "puppet_position": observation.puppet_state,
            "puppet_command": command,
            "puppet_command_valid": int(command_valid),
            "master_stamp_ns": int(master.get("stamp_ns", 0)),
            "puppet_stamp_ns": int(puppet.get("stamp_ns", 0)),
            "teach_active": int(teach_active),
            "control_epoch": int(observation.control.get("epoch", -1)),
            "effective_mode": effective_mode,
            "control_source": control_source,
            "is_policy": int(control_source == CONTROL_POLICY),
            "is_human_intervention": int(control_source == CONTROL_HUMAN),
            "intervention_id": intervention_id,
            "teach_event": teach_event,
            "intervention_event": intervention_event,
            "episode_id": int(self.file.attrs["episode_id"]),
            "cycle_id": int(cycle_id),
            "phase": phase,
            "learning_mask": learning_mask,
            "reference_action": reference,
            "reference_action_valid": int(reference_valid),
            "policy_action": policy,
            "policy_action_valid": int(policy_valid),
            "human_action": human,
            "human_action_valid": int(human_valid),
            "executed_action": executed,
            "executed_action_valid": int(executed_valid),
            "reward": 0.0,
            "reward_valid": 0,
            "done": 0,
            "success": OUTCOME_UNKNOWN,
        }
        for key, value in values.items():
            _append(self.numeric[key], value)
        _append(
            self.control_json,
            json.dumps(observation.control, separators=(",", ":"), sort_keys=True),
        )
        image_meta = raw.get("images", {})
        for camera in self.cameras:
            _append(
                self.images[camera]["jpeg"],
                np.frombuffer(observation.image_bytes[camera], dtype=np.uint8),
            )
            metadata = image_meta.get(camera) or {}
            _append(self.images[camera]["stamp_ns"], int(metadata.get("stamp_ns", 0)))

        self.count += 1
        self.previous_teach_active = teach_active
        self.previous_human_active = human_active
        if self.count % self.flush_frames == 0:
            self.file.flush()

    def close(
        self,
        *,
        completed: bool,
        stop_reason: str = "unknown",
        outcome: str = "unknown",
    ) -> None:
        if not self.file:
            return
        if outcome not in OUTCOME_NAMES:
            raise ValueError(f"invalid outcome: {outcome!r}")
        if outcome == "unknown" and self.intervention_count > 0:
            outcome = "success"
        outcome_id = OUTCOME_NAMES[outcome]
        if self.count:
            self.numeric["phase"][-1] = PHASE_TERMINAL
            self.numeric["learning_mask"][-1] = 0
            self.numeric["done"][-1] = 1
            self.numeric["success"][-1] = outcome_id
            if outcome in ("success", "failure"):
                self.numeric["reward"][-1] = float(outcome == "success")
                self.numeric["reward_valid"][-1] = 1
        self.file.attrs["completed"] = bool(completed)
        self.file.attrs["num_frames"] = self.count
        self.file.attrs["skipped_observations"] = self.skipped
        self.file.attrs["intervention_count"] = self.intervention_count
        self.file.attrs["stop_reason"] = stop_reason
        self.file.attrs["outcome"] = outcome
        self.file.attrs["outcome_id"] = outcome_id
        self.file.attrs["closed_at_ns"] = time.time_ns()
        self.file.flush()
        self.file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help=(
            "Atomically written after the first valid observation is stored. "
            "A policy runner can wait for this before using its first input."
        ),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--policy-id", default="")
    parser.add_argument(
        "--collection-type",
        choices=tuple(COLLECTION_TYPE_NAMES),
        default="IL",
        help="explicit episode-level collection class",
    )
    parser.add_argument(
        "--outcome",
        choices=tuple(OUTCOME_NAMES),
        default="unknown",
        help="explicit episode outcome; unknown becomes success when a Human/Teach intervention occurred",
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("front", "left", "right"),
        default=None,
        help="defaults to front plus the camera on the selected arm side",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flush-frames", type=int, default=20)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--stop-on-teach-release",
        action="store_true",
        help="record policy then one Teach intervention; stop after the release frame",
    )
    parser.add_argument("--convert-root", type=Path)
    parser.add_argument("--convert-repo-id")
    parser.add_argument("--convert-fps", type=int)
    parser.add_argument(
        "--convert-action-source",
        choices=(
            "master_state",
            "puppet_command",
            "reference_action",
            "policy_action",
            "human_action",
            "executed_action",
        ),
        default="executed_action",
    )
    parser.add_argument(
        "--convert-use-videos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--convert-vcodec", default="h264")
    return parser


def _convert_in_thread(args: argparse.Namespace, errors: list[BaseException]) -> None:
    command = [
        sys.executable,
        "-m",
        "cobot_bridge.convert_raw_to_lerobot",
        "--raw",
        str(args.output),
        "--repo-id",
        args.convert_repo_id,
        "--root",
        str(args.convert_root),
        "--action-source",
        args.convert_action_source,
        "--vcodec",
        args.convert_vcodec,
        "--use-videos" if args.convert_use_videos else "--no-use-videos",
    ]
    if args.convert_fps is not None:
        command.extend(("--fps", str(args.convert_fps)))
    try:
        subprocess.run(command, check=True)
    except BaseException as exc:
        errors.append(exc)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.cameras is None:
        args.cameras = ["front", args.side]
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    if args.stop_file is not None and args.stop_file.exists():
        raise FileExistsError(f"stop file already exists: {args.stop_file}")
    if args.ready_file is not None and args.ready_file.exists():
        raise FileExistsError(f"ready file already exists: {args.ready_file}")
    if args.fps <= 0 or args.flush_frames <= 0 or args.max_seconds <= 0:
        raise ValueError("fps, flush_frames and max_seconds must be positive")
    if (args.convert_root is None) != (args.convert_repo_id is None):
        raise ValueError("convert-root and convert-repo-id must be provided together")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    writer = RawEpisodeWriter(
        partial,
        side=args.side,
        cameras=tuple(args.cameras),
        task=args.task,
        fps=args.fps,
        flush_frames=args.flush_frames,
        episode_id=args.episode_id,
        policy_id=args.policy_id,
        collection_type=args.collection_type,
    )
    completed = False
    stop_reason = "max_seconds"
    episode_outcome = args.outcome
    discard_requested = False
    teach_seen = False
    period = 1.0 / args.fps
    started_at = time.monotonic()
    recorder_ready = False
    try:
        with CobotBridgeClient(args.endpoint, timeout_s=args.timeout) as client:
            robot = CobotMagicRemoteRobot(client, side=args.side, images=tuple(args.cameras))
            LOG.info("raw recording started: %s", partial)
            while not STOP and time.monotonic() - started_at < args.max_seconds:
                if args.stop_file is not None and args.stop_file.exists():
                    result = read_result_outcome(args.stop_file)
                    if result == "discard":
                        discard_requested = True
                        episode_outcome = "aborted"
                        stop_reason = "pc_abort"
                    else:
                        episode_outcome = result
                        stop_reason = "pc_save"
                    LOG.info(
                        "PC result received: result=%s file=%s",
                        result,
                        args.stop_file,
                    )
                    break
                tick = time.perf_counter()
                observation = robot.observe()
                if observation.ready and all(
                    camera in observation.image_bytes for camera in args.cameras
                ):
                    writer.append(observation)
                    if not recorder_ready and args.ready_file is not None:
                        # The marker is published only after the observation
                        # is durably appended, not merely after the recorder
                        # process has connected to the bridge.
                        writer.file.flush()
                        temporary = args.ready_file.with_name(
                            args.ready_file.name + f".{uuid.uuid4().hex}.tmp"
                        )
                        temporary.write_text(
                            json.dumps(
                                {
                                    "raw_frame_index": writer.count - 1,
                                    "pc_timestamp_ns": int(writer.numeric["pc_timestamp_ns"][-1]),
                                    "bridge_sequence": int(writer.numeric["sequence"][-1]),
                                },
                                sort_keys=True,
                            )
                        )
                        os.replace(temporary, args.ready_file)
                        recorder_ready = True
                        LOG.info("raw recorder ready after frame=%d", writer.count - 1)
                    if args.stop_on_teach_release:
                        if observation.teach_active:
                            teach_seen = True
                        elif teach_seen:
                            stop_reason = "teach_released"
                            LOG.info("Teach release recorded; closing intervention episode")
                            break
                else:
                    writer.skipped += 1
                time.sleep(max(0.0, period - (time.perf_counter() - tick)))
        if not discard_requested and writer.count < args.min_frames:
            raise RuntimeError(f"too few raw frames: {writer.count} < {args.min_frames}")
        completed = not discard_requested
    finally:
        if STOP:
            stop_reason = "signal"
        try:
            writer.close(completed=completed, stop_reason=stop_reason, outcome=episode_outcome)
        finally:
            if discard_requested:
                partial.unlink(missing_ok=True)
                if args.ready_file is not None:
                    args.ready_file.unlink(missing_ok=True)
                if args.stop_file is not None:
                    args.stop_file.unlink(missing_ok=True)

    if discard_requested:
        LOG.info("raw episode aborted and discarded: frames=%d", writer.count)
        return

    os.replace(partial, args.output)
    LOG.info("raw episode finalized: frames=%d output=%s", writer.count, args.output)
    if args.convert_root is not None:
        errors: list[BaseException] = []
        conversion = threading.Thread(
            target=_convert_in_thread,
            args=(args, errors),
            name=f"convert-{args.output.stem}",
            daemon=False,
        )
        conversion.start()
        LOG.info("LeRobot conversion thread started: root=%s", args.convert_root)
        conversion.join()
        if errors:
            raise RuntimeError("LeRobot conversion thread failed") from errors[0]
        LOG.info("LeRobot conversion thread completed: root=%s", args.convert_root)


if __name__ == "__main__":
    main()
