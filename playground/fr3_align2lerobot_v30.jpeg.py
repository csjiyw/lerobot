#!/usr/bin/env python3
"""
Franka FR3 双臂数据转换脚本 - 转换为 LeRobot 格式

State / Action 结构:
    left_arm(7) + left_gripper(1) +
    right_arm(7) + right_gripper(1) +
    left_eef_6d(6) + right_eef_6d(6)

总维度:
    observation.state: 28
    action: 28

说明:
- H5 中没有 action/end，因此 action 的 eef 6D 使用 next-state 的末端位姿
- 最后一帧的 action eef 使用当前帧自身的 eef
"""

import argparse
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
    "robot_type": "fr3",
    "target_image_size": (640, 480),  # (width, height)
}


STATE_NAMES = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_joint_7",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
    "right_gripper",
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_eef_rx",
    "left_eef_ry",
    "left_eef_rz",
    "right_eef_x",
    "right_eef_y",
    "right_eef_z",
    "right_eef_rx",
    "right_eef_ry",
    "right_eef_rz",
]
ACTION_NAMES = STATE_NAMES.copy()


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


def pose_to_6d(pose: np.ndarray) -> np.ndarray:
    """
    pose: [x, y, z, qx, qy, qz, qw] -> [x, y, z, roll, pitch, yaw]
    """
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape[0] < 7:
        out = np.zeros(6, dtype=np.float32)
        out[: min(3, pose.shape[0])] = pose[: min(3, pose.shape[0])]
        return out

    x, y, z, qx, qy, qz, qw = pose[:7]

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)


def build_fr3_features(metadata, h, w):
    camera_info = metadata["data_info"]["camera_info"]
    fps = int(round(metadata["time_align_info"]["fps"]))
    use_cameras = list(camera_info.keys())

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
        "shape": (len(STATE_NAMES),),
        "names": {"motors": STATE_NAMES},
        "fps": fps,
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(ACTION_NAMES),),
        "names": {"motors": ACTION_NAMES},
        "fps": fps,
    }

    return features, use_cameras


def combine_fr3_state(
    state_arm: np.ndarray,
    state_effector: np.ndarray,
    state_end_position: np.ndarray,
    state_end_orientation: np.ndarray,
    index: int,
) -> np.ndarray:
    """
    输入:
        state_arm: (N, 14)
        state_effector: (N, 2)
        state_end_position: (N, 2, 3)
        state_end_orientation: (N, 2, 4)
    输出:
        (28,)
    """
    left_arm = state_arm[index, :7]
    right_arm = state_arm[index, 7:14]

    left_gripper = state_effector[index, 0:1]
    right_gripper = state_effector[index, 1:2]

    left_pose7 = np.concatenate(
        [state_end_position[index, 0], state_end_orientation[index, 0]], axis=0
    )
    right_pose7 = np.concatenate(
        [state_end_position[index, 1], state_end_orientation[index, 1]], axis=0
    )

    left_eef_6d = pose_to_6d(left_pose7)
    right_eef_6d = pose_to_6d(right_pose7)

    return np.concatenate(
        [
            left_arm,
            left_gripper,
            right_arm,
            right_gripper,
            left_eef_6d,
            right_eef_6d,
        ],
        axis=0,
    ).astype(np.float32)


def combine_fr3_action(
    action_arm: np.ndarray,
    action_effector: np.ndarray,
    state_end_position: np.ndarray,
    state_end_orientation: np.ndarray,
    index: int,
) -> np.ndarray:
    """
    action 的 eef 6D 使用 next-state 的末端位姿
    最后一帧回退到当前帧
    """
    next_index = min(index + 1, action_arm.shape[0] - 1)

    left_arm = action_arm[index, :7]
    right_arm = action_arm[index, 7:14]

    left_gripper = action_effector[index, 0:1]
    right_gripper = action_effector[index, 1:2]

    left_pose7 = np.concatenate(
        [state_end_position[next_index, 0], state_end_orientation[next_index, 0]], axis=0
    )
    right_pose7 = np.concatenate(
        [state_end_position[next_index, 1], state_end_orientation[next_index, 1]], axis=0
    )

    left_eef_6d = pose_to_6d(left_pose7)
    right_eef_6d = pose_to_6d(right_pose7)

    return np.concatenate(
        [
            left_arm,
            left_gripper,
            right_arm,
            right_gripper,
            left_eef_6d,
            right_eef_6d,
        ],
        axis=0,
    ).astype(np.float32)


def load_fr3_h5(h5_path: Path) -> Dict[str, Any]:
    episode_data = {}
    episode_data["h5_path"]=h5_path

    with h5py.File(h5_path, "r") as f:
        metadata = json.loads(f["metadata.json"][()])
        episode_data["task"] = metadata.get("task_name", "manipulation_task")
        n = metadata["time_align_info"]["frame_count"]
        episode_data["frames"] = n

        _timestamps = f["timestamp"][:n]

        state_arm = f["joints/state/arm/position"][:n]               # (N, 14)
        state_effector = f["joints/state/effector/position"][:n]     # (N, 2)
        state_end_position = f["joints/state/end/position"][:n]      # (N, 2, 3)
        state_end_orientation = f["joints/state/end/orientation"][:n]# (N, 2, 4)

        action_arm = f["joints/action/arm/position"][:n]             # (N, 14)
        action_effector = f["joints/action/effector/position"][:n]   # (N, 2)

        states = np.stack(
            [
                combine_fr3_state(
                    state_arm,
                    state_effector,
                    state_end_position,
                    state_end_orientation,
                    i,
                )
                for i in range(n)
            ],
            axis=0,
        )

        actions = np.stack(
            [
                combine_fr3_action(
                    action_arm,
                    action_effector,
                    state_end_position,
                    state_end_orientation,
                    i,
                )
                for i in range(n)
            ],
            axis=0,
        )

        episode_data["state"] = states.astype(np.float32)
        episode_data["action"] = actions.astype(np.float32)

        images = {}
        image_shapes = {}

        target_width, target_height = ROBOT_CONFIG["target_image_size"]
        episode_data["features"], use_cameras = build_fr3_features(
            metadata, target_height, target_width
        )

        for cam_id in use_cameras:
            cam_key = f"cameras/{cam_id}/color/data"
            if cam_key in f:
                # img_bytes_array = f[cam_key][:n]
                images[cam_id] = cam_key
                image_shapes[cam_id] = (1, target_height, target_width, 3)
            else:
                raise ValueError(f"Camera key not found: {cam_key}")

        episode_data["images"] = images
        episode_data["image_shapes"] = image_shapes

    return data


def convert_episode(
    h5_path,
    output_dir: Path,
    repo_id: str,
    episode_index: int,
    fps: int,
    vcodec: str = "libsvtav1",
    crf: int = 30,
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

        state_arm = f["joints/state/arm/position"][:n]               # (N, 14)
        state_effector = f["joints/state/effector/position"][:n]     # (N, 2)
        state_end_position = f["joints/state/end/position"][:n]      # (N, 2, 3)
        state_end_orientation = f["joints/state/end/orientation"][:n]# (N, 2, 4)

        action_arm = f["joints/action/arm/position"][:n]             # (N, 14)
        action_effector = f["joints/action/effector/position"][:n]   # (N, 2)

        states = np.stack(
            [
                combine_fr3_state(
                    state_arm,
                    state_effector,
                    state_end_position,
                    state_end_orientation,
                    i,
                )
                for i in range(n)
            ],
            axis=0,
        )

        actions = np.stack(
            [
                combine_fr3_action(
                    action_arm,
                    action_effector,
                    state_end_position,
                    state_end_orientation,
                    i,
                )
                for i in range(n)
            ],
            axis=0,
        )

        episode_data["state"] = states.astype(np.float32)
        episode_data["action"] = actions.astype(np.float32)

        images = {}
        image_shapes = {}

        target_width, target_height = ROBOT_CONFIG["target_image_size"]
        episode_data["features"], use_cameras = build_fr3_features(
            metadata, target_height, target_width
        )

        for cam_id in use_cameras:
            cam_key = f"cameras/{cam_id}/color/data"
            if cam_key in f:
                # img_bytes_array = f[cam_key][:n]
                images[cam_id] = cam_key
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
                cam_key = episode_data["images"][cam_id]
                img_arr = to_numpy(f[cam_key][i])
                if i == 0:
                    dataset.img_format = detect_img_ext(img_arr)
                frame_dict[f"observation.images.{cam_id}"] = img_arr


            dataset.add_frame(frame_dict)

            if i % 500 == 0 and i > 0:
                dataset.image_writer.wait_until_done()

    dataset.save_episode(parallel_encoding=True, logger=logger)
    return dataset


def convert_episode_wrapper(args: tuple) -> dict:
    h5_paths, output_dir, repo_id, episode_index, fps, vcodec, crf, task_override = args

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
            # episode_data = load_fr3_h5(h5_path)
            dataset = convert_episode(
                h5_path,
                output_dir=output_dir,
                repo_id=repo_id,
                episode_index=episode_index,
                fps=fps,
                vcodec=vcodec,
                crf=crf,
                task_override=task_override,
                dataset=dataset,
            )

        result["success"] = True
        result["frames"] = 1
        result["dataset_path"] = str(dataset.root)
        return result

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
    parser = argparse.ArgumentParser(
        description="FR3 双臂机器人数据转换为 LeRobot 格式"
    )

    parser.add_argument("--input", type=Path, required=True, help="输入数据目录 (包含 H5 文件)")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace 仓库 ID")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=["manipulation_task"],
        help="任务描述",
    )
    parser.add_argument("--fps", type=int, default=30, help="数据集帧率")
    parser.add_argument("--workers", type=int, default=4, help="并行工作进程数")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="视频编码器")
    parser.add_argument("--crf", type=int, default=30, help="视频质量 CRF")

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
    batch_size = 1
    for i in range(0, len(episodes), batch_size):
        batch_eps = episodes[i:i + batch_size]
        tasks.append(
            (
                batch_eps,
                separate_dir,
                args.repo_id,
                i // batch_size,
                args.fps,
                args.vcodec,
                args.crf,
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