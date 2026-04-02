#!/usr/bin/env python3
"""
双臂移动操作机器人数据转换脚本 - 转换为 LeRobot 格式

当前 H5 结构:
    State:
        - arm/position: 12
        - effector/position: 2
        - robot/angular: 3
        - robot/velocity: 3

    Action:
        - arm/position: 12
        - effector/position: 2
        - robot/angular: 3
        - robot/velocity: 3

    最终:
        observation.state: 20
        action: 20
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
from datasets import disable_progress_bar
from tqdm import tqdm

disable_progress_bar()

os.environ["LEROBOT_W"] = str(640)
os.environ["LEROBOT_H"] = str(480)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROBOT_CONFIG = {
    "robot_type": "cobotmagic",
    "target_image_size": (640, 480),  # (width, height)
}


def to_numpy(x):
    if len(x) == 1:
        return x[0]
    return x


def detect_img_ext(arr):
    x = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
    if x.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if x.startswith(b"\xff\xd8\xff"):
        return "jpg"
    raise ValueError("unknown image format")


def build_features(metadata, h, w):
    joint_info = metadata["data_info"]["joint_info"]
    camera_info = metadata["data_info"]["camera_info"]

    fps = int(round(metadata["time_align_info"]["fps"]))
    use_cameras = list(camera_info.keys())

    state_names = (
        joint_info["state/arm"]["joint_names"]
        + joint_info["state/effector"]["joint_names"]
        + ["robot_angular_x", "robot_angular_y", "robot_angular_z"]
        + ["robot_vel_x", "robot_vel_y", "robot_vel_z"]
    )

    action_names = (
        joint_info["action/arm"]["joint_names"]
        + joint_info["action/effector"]["joint_names"]
        + ["robot_angular_x", "robot_angular_y", "robot_angular_z"]
        + ["robot_vel_x", "robot_vel_y", "robot_vel_z"]
    )

    features = {}

    for cam in use_cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": {"motors": state_names},
        "fps": fps,
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": {"motors": action_names},
        "fps": fps,
    }

    return features, use_cameras


def load_h5(h5_path: Path) -> Dict[str, Any]:
    data = {}

    with h5py.File(h5_path, "r") as f:
        metadata = json.loads(f["metadata.json"][()])
        data["task"] = metadata.get("task_name", "manipulation_task")
        n = metadata["time_align_info"]["frame_count"]
        data["frames"] = n

        _timestamps = f["timestamp"][:n]

        state_arm = f["joints/state/arm/position"][:n]
        state_effector = f["joints/state/effector/position"][:n]
        state_robot_angular = f["joints/state/robot/angular"][:n]
        state_robot_velocity = f["joints/state/robot/velocity"][:n]

        action_arm = f["joints/action/arm/position"][:n]
        action_effector = f["joints/action/effector/position"][:n]
        action_robot_angular = f["joints/action/robot/angular"][:n]
        action_robot_velocity = f["joints/action/robot/velocity"][:n]

        data["state"] = np.concatenate(
            [
                state_arm,
                state_effector,
                state_robot_angular,
                state_robot_velocity,
            ],
            axis=1,
        ).astype(np.float32)

        data["action"] = np.concatenate(
            [
                action_arm,
                action_effector,
                action_robot_angular,
                action_robot_velocity,
            ],
            axis=1,
        ).astype(np.float32)

        images = {}
        image_shapes = {}

        target_width, target_height = ROBOT_CONFIG["target_image_size"]
        data["features"], use_cameras = build_features(metadata, target_height, target_width)

        for cam_id in use_cameras:
            cam_key = f"cameras/{cam_id}/color/data"
            if cam_key in f:
                # img_bytes_array = f[cam_key][:n]
                # images[cam_id] = img_bytes_array
                images[cam_id]=cam_key
                image_shapes[cam_id] = (1, target_height, target_width, 3)
            else:
                raise ValueError(f"Camera key not found: {cam_key}")

        data["images"] = images
        data["image_shapes"] = image_shapes

    return data


def convert_episode(
    # episode_data: Dict[str, Any],
    h5_path,
    output_dir: Path,
    repo_id: str,
    episode_index: int,
    fps: int,
    task_override: str = None,
    dataset=None,
):
    
    episode_data = {}

    with h5py.File(h5_path, "r") as f:
        metadata = json.loads(f["metadata.json"][()])
        episode_data["task"] = metadata.get("task_name", "manipulation_task")
        n = metadata["time_align_info"]["frame_count"]
        episode_data["frames"] = n

        _timestamps = f["timestamp"][:n]

        state_arm = f["joints/state/arm/position"][:n]
        state_effector = f["joints/state/effector/position"][:n]
        state_robot_angular = f["joints/state/robot/angular"][:n]
        state_robot_velocity = f["joints/state/robot/velocity"][:n]

        action_arm = f["joints/action/arm/position"][:n]
        action_effector = f["joints/action/effector/position"][:n]
        action_robot_angular = f["joints/action/robot/angular"][:n]
        action_robot_velocity = f["joints/action/robot/velocity"][:n]

        episode_data["state"] = np.concatenate(
            [
                state_arm,
                state_effector,
                state_robot_angular,
                state_robot_velocity,
            ],
            axis=1,
        ).astype(np.float32)

        episode_data["action"] = np.concatenate(
            [
                action_arm,
                action_effector,
                action_robot_angular,
                action_robot_velocity,
            ],
            axis=1,
        ).astype(np.float32)

        images = {}
        image_shapes = {}

        target_width, target_height = ROBOT_CONFIG["target_image_size"]
        episode_data["features"], use_cameras = build_features(metadata, target_height, target_width)

        for cam_id in use_cameras:
            cam_key = f"cameras/{cam_id}/color/data"
            if cam_key in f:
                # img_bytes_array = f[cam_key][:n]
                # images[cam_id] = img_bytes_array
                images[cam_id]=cam_key
                image_shapes[cam_id] = (1, target_height, target_width, 3)
            else:
                raise ValueError(f"Camera key not found: {cam_key}")

        episode_data["images"] = images
        episode_data["image_shapes"] = image_shapes

        num_frames = episode_data["frames"]
        task = task_override if task_override else episode_data.get("task", "manipulation_task")

        if dataset is None:
            episode_dir = output_dir / f"episode_{episode_index:04d}"
            if episode_dir.exists():
                shutil.rmtree(episode_dir)

            dataset = LeRobotDataset.create(
                repo_id=f"{repo_id}/episode_{episode_index:04d}",
                root=episode_dir,
                robot_type=ROBOT_CONFIG["robot_type"],
                fps=fps,
                features=episode_data["features"],
                use_videos=True,
                image_writer_threads=3,
                image_writer_processes=0,
            )

        for i in range(num_frames):
            frame_dict = {
                "observation.state": episode_data["state"][i],
                "action": episode_data["action"][i],
                "task": task,
            }

            for cam_id in episode_data["image_shapes"].keys():
                cam_key=episode_data["images"][cam_id]
                img_arr = to_numpy(f[cam_key][i])
                if i == 0:
                    dataset.img_format = detect_img_ext(img_arr)
                frame_dict[f"observation.images.{cam_id}"] = img_arr

            dataset.add_frame(frame_dict)

            # Periodically wait for image writer to drain queue - reduces memory buildup
            if i % 500 == 0 and i > 0:
                dataset.image_writer.wait_until_done()

        dataset.save_episode(parallel_encoding=True, logger=logger)
    return dataset


def convert_episode_wrapper(args: tuple) -> dict:
    h5_paths, output_dir, repo_id, episode_index, fps, task_override = args

    result = {
        "episode_index": episode_index,
        "success": False,
        "frames": 0,
        "error": None,
        "dataset_path": None,
        "episode_cnt": len(h5_paths),
    }

    try:
        dataset = None
        for h5_path in h5_paths:
            # episode_data = load_h5(h5_path)
            dataset = convert_episode(
                h5_path,
                # episode_data=episode_data,
                output_dir=output_dir,
                repo_id=repo_id,
                episode_index=episode_index,
                fps=fps,
                task_override=task_override,
                dataset=dataset,
            )

        result["success"] = True
        result["frames"] = 1
        result["dataset_path"] = str(dataset.root)

    except Exception as e:
        print(traceback.format_exc())
        return {
            "episode_index": episode_index,
            "success": False,
            "frames": 0,
            "error": f"{e}\n{traceback.format_exc()}",
            "dataset_path": None,
            "episode_cnt": len(h5_paths),
        }

    del dataset
    gc.collect()
    return result


def find_episodes(data_dir: Path) -> List[Path]:
    episodes = []

    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            h5_files = list(subdir.glob("*.h5"))
            if h5_files:
                episodes.append(h5_files[0])

    for h5_file in sorted(data_dir.glob("*.h5")):
        if h5_file not in episodes:
            episodes.append(h5_file)

    return episodes


def main():
    parser = argparse.ArgumentParser(description="机器人数据转换为 LeRobot 格式")
    parser.add_argument("--input", type=Path, required=True, help="输入目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--repo_id", type=str, default=None, help="repo id")
    parser.add_argument("--task", type=str, nargs="+", default=["manipulation_task"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if isinstance(args.task, list):
        args.task = " ".join(args.task)

    if args.repo_id is None:
        args.repo_id = args.output.name

    episodes = find_episodes(args.input)
    if not episodes:
        print(f"No .h5 files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(episodes)} episodes")
    print(f"Using {args.workers} workers")

    output_root = args.output
    if output_root.exists():
        shutil.rmtree(output_root)

    separate_dir = output_root.parent / f"{output_root.name}_separate_episodes"
    separate_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    batch_size = 2
    for i in range(0, len(episodes), batch_size):
        batch_eps = episodes[i:i + batch_size]
        tasks.append(
            (
                batch_eps,
                separate_dir,
                args.repo_id,
                i // batch_size,
                args.fps,
                args.task,
            )
        )

    success_datasets = []
    failed_episodes = []

    start_time = time.time()
    pbar = tqdm(total=len(episodes), desc="Converting Episodes")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_episode_wrapper, t): t for t in tasks}

        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                success_datasets.append(Path(res["dataset_path"]))
            else:
                failed_episodes.append(f'Episode {res["episode_index"]}: {res["error"]}')
            pbar.update(res["episode_cnt"])

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"Conversion completed in {elapsed:.1f}s")
    print(f"Success: {len(success_datasets)} / {len(tasks)}")
    print(f"Failed: {len(failed_episodes)}")
    print(f"{'=' * 70}")

    if failed_episodes:
        print("\nFailed episodes:")
        for f in failed_episodes[:5]:
            print(f"  - {f[:200]}...")

    if not success_datasets:
        print("No valid datasets. Exiting.")
        sys.exit(1)

    from lerobot.datasets.dataset_tools import merge_datasets

    print("\nMerging into final dataset...")
    datasets_to_merge = []
    for dpath in sorted(success_datasets):
        ds = LeRobotDataset(root=dpath, repo_id=dpath.name)
        datasets_to_merge.append(ds)

    if datasets_to_merge:
        merged = merge_datasets(
            datasets=datasets_to_merge,
            output_dir=output_root,
            output_repo_id=args.repo_id,
        )

        print("\nMerge Complete!")
        print(f"Output: {output_root}")
        print(f"Total Episodes: {merged.meta.total_episodes}")
        print(f"Total Frames: {merged.meta.total_frames}")

        if separate_dir.exists():
            shutil.rmtree(separate_dir)
            print("Cleaned up temporary episode datasets.")


if __name__ == "__main__":
    main()