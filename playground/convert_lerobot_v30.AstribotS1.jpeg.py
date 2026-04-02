#!/usr/bin/env python3
"""
智元机器人 A2D / S1 数据转换脚本 - 转换为 LeRobot 格式

数据结构:
    State:
        - arm
        - effector
        - head
        - torso (optional)
        - chassis (optional)

    Action:
        - arm
        - effector
        - head
        - torso (optional)
        - chassis (optional; if missing, fallback to state[t-1]/chassis)

相机:
    - 默认只保留 head 和 hand_* 视角
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

try:
    import av  # noqa: F401
    USE_PYAV = True
except ImportError:
    USE_PYAV = False
    print("Warning: PyAV not found, falling back to FFmpeg subprocess")

from lerobot.datasets.lerobot_dataset import LeRobotDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 配置常量
# =============================================================================

ROBOT_CONFIG = {
    "robot_type": "AstribotS1",
    "target_image_size": (640, 480),  # (width, height)
}


# =============================================================================
# 工具函数
# =============================================================================

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


# =============================================================================
# 数据读取工具
# =============================================================================

def build_full_dof_features_chw(metadata, h, w):
    joint_info = metadata["data_info"]["joint_info"]
    camera_info = metadata["data_info"]["camera_info"]

    fps = 30  # int(round(metadata["time_align_info"]["fps"]))

    # 只保留 head 和 hand_* 视角
    all_cameras = list(camera_info.keys())
    use_cameras = [cam for cam in all_cameras if cam == "head" or cam.startswith("hand_") or cam=="torso"]
    logger.info(f"Using cameras: {use_cameras} (filtered from {all_cameras})")

    state_names = (
        joint_info["state/arm"]["joint_names"]
        + joint_info["state/effector"]["joint_names"]
        + joint_info["state/head"]["joint_names"]
        + joint_info.get("state/torso", {}).get("joint_names", [])
        + joint_info.get("state/chassis", {}).get("joint_names", [])
    )

    action_names = (
        joint_info["action/arm"]["joint_names"]
        + joint_info["action/effector"]["joint_names"]
        + joint_info["action/head"]["joint_names"]
        + joint_info.get("action/torso", {}).get("joint_names", [])
        + joint_info.get(
            "action/chassis", {}).get(
            "joint_names",
            joint_info.get("state/chassis", {}).get("joint_names", []),
        )
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

        _timestamps = f["timestamp"][:n]

        # =========================
        # state
        # =========================
        state_arm = f["joints/state/arm/position"][:n]
        state_effector = f["joints/state/effector/position"][:n]
        state_head = f["joints/state/head/position"][:n]
        state_torso = f["joints/state/torso/position"][:n]
        state_chassis = f["joints/state/chassis/position"][:n]

        # =========================
        # action
        # =========================
        action_arm = f["joints/action/arm/position"][:n]
        action_effector = f["joints/action/effector/position"][:n]
        action_head = f["joints/action/head/position"][:n]
        action_torso = f["joints/action/torso/position"][:n]

        use_prev_chassis_state = "joints/action/chassis/position" not in f

        if use_prev_chassis_state:
            logger.warning(
                f"{h5_path}: missing joints/action/chassis/position, "
                f"fallback to previous state/chassis/position and reduce length by 1"
            )
            action_chassis = state_chassis[:-1]

            state_arm = state_arm[1:]
            state_effector = state_effector[1:]
            state_head = state_head[1:]
            state_torso = state_torso[1:]
            state_chassis = state_chassis[1:]

            action_arm = action_arm[1:]
            action_effector = action_effector[1:]
            action_head = action_head[1:]
            action_torso = action_torso[1:]
        else:
            action_chassis = f["joints/action/chassis/position"][:n]

        # 对齐最终长度
        final_len = min(
            len(state_arm),
            len(state_effector),
            len(state_head),
            len(state_torso),
            len(state_chassis),
            len(action_arm),
            len(action_effector),
            len(action_head),
            len(action_torso),
            len(action_chassis),
        )

        state_arm = state_arm[:final_len]
        state_effector = state_effector[:final_len]
        state_head = state_head[:final_len]
        state_torso = state_torso[:final_len]
        state_chassis = state_chassis[:final_len]

        action_arm = action_arm[:final_len]
        action_effector = action_effector[:final_len]
        action_head = action_head[:final_len]
        action_torso = action_torso[:final_len]
        action_chassis = action_chassis[:final_len]

        data["frames"] = final_len

        # =========================
        # concatenate
        # =========================
        data["state"] = np.concatenate(
            [state_arm, state_effector, state_head, state_torso, state_chassis],
            axis=1,
        ).astype(np.float32)

        data["action"] = np.concatenate(
            [action_arm, action_effector, action_head, action_torso, action_chassis],
            axis=1,
        ).astype(np.float32)

        # =========================
        # images
        # =========================
        images = {}
        image_shapes = {}

        target_width, target_height = ROBOT_CONFIG["target_image_size"]
        data["features"], use_cameras = build_full_dof_features_chw(
            metadata, target_height, target_width
        )

        for cam_id in use_cameras:
            cam_key = f"cameras/{cam_id}/color/data"
            if cam_key in f:
                img_bytes_array = f[cam_key][:n]
                if use_prev_chassis_state:
                    img_bytes_array = img_bytes_array[1:]
                img_bytes_array = img_bytes_array[:final_len]

                images[cam_id] = img_bytes_array
                image_shapes[cam_id] = (1, target_height, target_width, 3)
            else:
                raise ValueError(f"Camera key is invalid: {cam_key}")

        data["images"] = images
        data["image_shapes"] = image_shapes

    return data


# =============================================================================
# Episode 转换
# =============================================================================

def convert_episode(
    episode_data: Dict[str, Any],
    output_dir: Path,
    repo_id: str,
    episode_index: int,
    fps: int,
    vcodec: str = "libsvtav1",
    crf: int = 30,
    task_override: str = None,
    dataset=None,
):
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
            image_writer_threads=4,
            image_writer_processes=2,
        )

    for i in range(num_frames):
        frame_dict = {
            "observation.state": episode_data["state"][i],
            "action": episode_data["action"][i],
            "task": task,
        }

        for cam_id in episode_data["image_shapes"].keys():
            img_arr = to_numpy(episode_data["images"][cam_id][i])
            if i == 0:
                dataset.img_format = detect_img_ext(img_arr)
            frame_dict[f"observation.images.{cam_id}"] = img_arr

        dataset.add_frame(frame_dict)

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
            episode_data = load_h5(h5_path)
            dataset = convert_episode(
                episode_data=episode_data,
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


# =============================================================================
# 主入口
# =============================================================================

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
        description="智元 A2D / S1 机器人数据转换为 LeRobot 格式"
    )

    parser.add_argument("--input", type=Path, required=True, help="数据目录 (包含 H5 文件)")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace 仓库 ID")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=["manipulation_task"],
        help="任务描述 (默认: manipulation_task，可以不用引号，多个词会自动拼接)",
    )
    parser.add_argument("--fps", type=int, default=30, help="数据集帧率 (默认: 30)")
    parser.add_argument("--workers", type=int, default=4, help="并行工作进程数 (默认: 4)")
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

    print(f"Found {len(episodes)} episodes:")
    print(f"Using {args.workers} workers")

    output_root = args.output
    if output_root.exists():
        shutil.rmtree(output_root)

    separate_dir = output_root.parent / f"{output_root.name}_separate_episodes"
    separate_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    batch_size = 4
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