"""Collect demonstrations directly on the PC into a LeRobot dataset.

The AGX bridge keeps no dataset or raw recording. By default, episode lifetime
is controlled by the PC process and is independent of the robot's teach mode.
Teach-edge triggering remains available as an explicit compatibility option.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from .client import CobotBridgeClient
from .remote_robot import CobotMagicRemoteRobot, RemoteObservation


LOG = logging.getLogger("cobot_bridge.collect_lerobot")
STOP = False
JOINT_NAMES = [f"joint{index}" for index in range(7)]


def _stop(_signal: int, _frame: Any) -> None:
    global STOP
    STOP = True


def _action_from(observation: RemoteObservation, source: str) -> np.ndarray | None:
    if source == "puppet_command":
        return observation.executed_puppet_action()
    if source == "master_state":
        return observation.master_state.copy()
    raise ValueError(source)


def _create_dataset(args: argparse.Namespace, first: RemoteObservation):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    images = {name: first.decode_image(name) for name in args.cameras}
    image_dtype = "video" if args.use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "observation.master_state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
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
    }
    for name, image in images.items():
        features[f"observation.images.camera_{name[0] if name != 'front' else 'f'}"] = {
            "dtype": image_dtype,
            "shape": tuple(image.shape),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="cobot_magic_piper",
        features=features,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )
    # Compatibility with the local LeRobot branch, whose custom image path
    # implementation reads ``img_format`` but does not initialize it in
    # ``LeRobotDataset.create``.
    if not hasattr(dataset, "img_format"):
        dataset.img_format = "png"
    return dataset


def _frame(
    observation: RemoteObservation,
    action: np.ndarray,
    task: str,
    cameras: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "observation.state": observation.puppet_state.astype(np.float32),
        "observation.master_state": observation.master_state.astype(np.float32),
        "action": action.astype(np.float32),
        "observation.robot_timestamp_ns": np.asarray(
            [int(observation.raw.get("bridge_time_ns", 0))], dtype=np.int64
        ),
        "observation.teach_active": np.asarray(
            [float(observation.teach_active)], dtype=np.float32
        ),
        "task": task,
    }
    for name in cameras:
        key = f"observation.images.camera_{name[0] if name != 'front' else 'f'}"
        result[key] = observation.decode_image(name)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--cameras", nargs="+", choices=("front", "left", "right"), default=["front", "left", "right"]
    )
    parser.add_argument(
        "--action-source",
        choices=("puppet_command", "master_state"),
        default="puppet_command",
    )
    parser.add_argument(
        "--episode-trigger",
        choices=("pc", "teach"),
        default="pc",
        help="PC starts/stops episodes by duration; teach uses hardware teach edges",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
        help="episode duration when --episode-trigger=pc",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="PC-side delay before each timed episode starts",
    )
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--use-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.root.exists():
        raise FileExistsError(f"dataset root already exists: {args.root}")
    if args.num_episodes <= 0 or args.fps <= 0:
        raise ValueError("num_episodes and fps must be positive")
    if args.episode_seconds <= 0:
        raise ValueError("episode_seconds must be positive")
    if args.start_delay < 0:
        raise ValueError("start_delay cannot be negative")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    dataset = None
    period = 1.0 / args.fps
    with CobotBridgeClient(args.endpoint, timeout_s=args.timeout) as client:
        robot = CobotMagicRemoteRobot(
            client, side=args.side, images=tuple(args.cameras)
        )
        saved = 0
        try:
            while saved < args.num_episodes and not STOP:
                observation = robot.observe()
                if args.episode_trigger == "teach":
                    LOG.info(
                        "waiting for teach press: episode %d/%d",
                        saved + 1,
                        args.num_episodes,
                    )
                    while not STOP and not observation.teach_active:
                        time.sleep(min(0.05, period))
                        observation = robot.observe()
                else:
                    if args.start_delay:
                        LOG.info(
                            "PC episode %d/%d starts in %.1f seconds",
                            saved + 1,
                            args.num_episodes,
                            args.start_delay,
                        )
                        deadline = time.monotonic() + args.start_delay
                        while not STOP and time.monotonic() < deadline:
                            time.sleep(min(0.05, deadline - time.monotonic()))
                    observation = robot.observe()
                if STOP:
                    break
                if not observation.ready:
                    LOG.warning(
                        "episode start requested but observation is not ready: %s",
                        observation.raw.get("missing_or_stale"),
                    )
                    time.sleep(min(0.1, period))
                    continue
                if dataset is None:
                    dataset = _create_dataset(args, observation)

                frame_count = 0
                stop_at = (
                    time.monotonic() + args.episode_seconds
                    if args.episode_trigger == "pc"
                    else None
                )
                LOG.info(
                    "recording episode %d (%s controlled%s)",
                    saved + 1,
                    args.episode_trigger,
                    f", {args.episode_seconds:.1f}s" if stop_at is not None else "",
                )
                while not STOP:
                    if args.episode_trigger == "teach" and not observation.teach_active:
                        break
                    if stop_at is not None and time.monotonic() >= stop_at:
                        break
                    started = time.perf_counter()
                    if observation.ready:
                        action = _action_from(observation, args.action_source)
                        if action is not None:
                            dataset.add_frame(
                                _frame(observation, action, args.task, args.cameras)
                            )
                            frame_count += 1
                    time.sleep(max(0.0, period - (time.perf_counter() - started)))
                    observation = robot.observe()

                if frame_count >= args.min_frames:
                    dataset.save_episode()
                    saved += 1
                    LOG.info("saved episode %d with %d frames", saved, frame_count)
                else:
                    dataset.clear_episode_buffer()
                    LOG.warning("discarded short episode with %d frames", frame_count)
        finally:
            if dataset is not None:
                dataset.finalize()
        LOG.info("collection complete: %d episode(s), root=%s", saved, args.root)


if __name__ == "__main__":
    main()
