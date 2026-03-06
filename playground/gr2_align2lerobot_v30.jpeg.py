#!/usr/bin/env python3
"""
智元机器人 A2D 数据转换脚本 - 转换为 LeRobot 格式

数据结构:
    State/Action:
        - arm: 14维 (left_arm 7 + right_arm 7)
        - effector: 2维 (left_gripper + right_gripper), 已归一化 [0,1]
    
    相机:
        - head: 720x1280 RGB
        - hand_left: 480x848 RGB
        - hand_right: 480x848 RGB

Usage:
    python a2d_to_lerobot.py \
        --data-dir ./raw_data \
        --output-dir ./lerobot_dataset \
        --repo-id a2d/task_name \
        --fps 30 --workers 4
"""

import argparse
import json
import logging
import os

from datasets import disable_progress_bar

from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
disable_progress_bar()


os.environ["LEROBOT_W"] = str(640)
os.environ["LEROBOT_H"] = str(480)

import shutil
import signal
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm, trange

# Video encoding
try:
    import av
    USE_PYAV = True
except ImportError:
    USE_PYAV = False
    import subprocess
    print("Warning: PyAV not found, falling back to FFmpeg subprocess")


# 屏蔽 stderr
# stderr_fd = sys.stderr.fileno()
# saved_stderr = os.dup(stderr_fd)
# devnull = os.open(os.devnull, os.O_WRONLY)
# os.dup2(devnull, stderr_fd)

# LeRobot imports


from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.datasets.compute_stats import compute_episode_stats


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 配置常量
# =============================================================================

ROBOT_CONFIG = {
    "robot_type": "GR2",
    "cameras": [ "head_left", "head_right"],
    "target_image_size": (640, 480),  # (width, height) - 统一resize目标尺寸：宽640、高480
}




# =============================================================================
# 视频编码工具
# =============================================================================



import torch
import torchvision.io as io
import torch.nn.functional as F
import torchvision.transforms.functional as F

def to_numpy(x):
    if len(x)==1:
        return x[0]
    return x



# =============================================================================
# 数据读取工具
# =============================================================================
import numpy as np

def build_full_dof_features_chw(metadata, h, w):
    joint_info = metadata["data_info"]["joint_info"]
    camera_info = metadata["data_info"]["camera_info"]

    fps = int(round(metadata["time_align_info"]["fps"]))

    use_cameras = list(camera_info.keys())

    state_names = (
        joint_info["state/arm"]["joint_names"]
        + joint_info["state/effector"]["joint_names"]
        + joint_info["state/head"]["joint_names"]
        + joint_info["state/leg"]["joint_names"]
        + joint_info["state/waist"]["joint_names"]
    )

    action_names = (
        joint_info["action/arm"]["joint_names"]
        + joint_info["action/effector"]["joint_names"]
        + joint_info["action/head"]["joint_names"]
        + joint_info["action/leg"]["joint_names"]
        + joint_info["action/waist"]["joint_names"]
    )

    features = {}

    for cam in use_cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [
                h,
                w,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
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
        "shape": (len(state_names),),   # 41
        "names": {
            "motors": state_names
        },
        "fps":fps
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(action_names),),  # 41
        "names": {
            "motors": action_names
        },
        "fps":fps
    }

    return features


def load_a2d_h5(h5_path: Path) -> Dict[str, Any]:
    """
    从 A2D H5 文件加载数据
    
    Returns:
        {
            'frames': int,
            'state': np.ndarray,  # (N, 16) = arm(14) + effector(2)
            'action': np.ndarray, # (N, 16) = arm(14) + effector(2)
            'images': {
                'camera_id': List[np.ndarray],
                ...
            },
            'task': str,
            'image_shapes': dict,
        }
    """
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        # 读取 metadata
        metadata = json.loads(f['metadata.json'][()])

        data['task'] = metadata.get('task_name', 'manipulation_task')
        n=metadata["time_align_info"]["frame_count"]
        data['frames'] = n

        # 时间戳
        timestamps = f['timestamp'][:n]
        
        
        # joints: remove end
        state_arm = f["joints/state/arm/position"][()]
        state_effector = f["joints/state/effector/position"][()]
        state_head = f["joints/state/head/position"][()]
        state_leg = f["joints/state/leg/position"][()]
        state_waist = f["joints/state/waist/position"][()]

        action_arm = f["joints/action/arm/position"][()]
        action_effector = f["joints/action/effector/position"][()]
        action_head = f["joints/action/head/position"][()]
        action_leg = f["joints/action/leg/position"][()]
        action_waist = f["joints/action/waist/position"][()]

        # concatenate first
        data['state'] = np.concatenate(
            [state_arm, state_effector, state_head, state_leg, state_waist],
            axis=1,
        ).astype(np.float32)   # (T, 41)

        data['action'] = np.concatenate(
            [action_arm, action_effector, action_head, action_leg, action_waist],
            axis=1,
        ).astype(np.float32)   # (T, 41)
        
        # 图像
        images = {}
        image_shapes = {}
        
        target_width, target_height = ROBOT_CONFIG["target_image_size"]

        data["features"]=build_full_dof_features_chw(metadata,target_height,target_width)

 
        
        for cam_id in ROBOT_CONFIG["cameras"]:
            cam_key = f'cameras/{cam_id}/color/data'
            if cam_key in f:
                img_bytes_array = f[cam_key][:n]

                images[cam_id] = img_bytes_array  # (N, C, H, W)
                # print(img_tensor.dtype,img_tensor.shape)
                image_shapes[cam_id] = (1,target_height,target_width,3)
            
            
            else:
                raise ValueError("cam_key is invalid")
        
        data['images'] = images
        data['image_shapes'] = image_shapes

        # print("image_shapes",image_shapes)
        # for k in data.keys():
        #     # data[k]=np.array(data[k])
        #     try:
        #         print(k,data[k].shape)
        #     except Exception as e:
        #         print(k)
        # exit()

    
    return data

def detect_img_ext(arr):
    x = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
    if x.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if x.startswith(b"\xff\xd8\xff"):
        return "jpg"
    raise ValueError("unknown image format")

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
) -> dict:
    """将单个 episode 数据转换为 LeRobot 数据集格式"""

    
    if True:
        # print("------------------")
        # print(episode_data)
        # exit()

        num_frames = episode_data['frames']
        # 优先使用命令行指定的 task，否则使用 H5 文件中的 task
        task = task_override if task_override else episode_data.get('task', 'manipulation_task')
        
        
        # 创建临时目录用于视频编码

        video_paths = {}
        
        
        # 创建数据集
        
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
                image_writer_processes=2
            )

            
        
        # 添加帧
        # logger.info(f"Adding {num_frames} frames...")


        
        for i in range(num_frames):
            frame_dict = {
                "observation.state": episode_data['state'][i],
                "action": episode_data['action'][i],
                "task": task,
                
            }
            
            
            for cam_id, shape in episode_data['image_shapes'].items():
                # h, w, c = shape
                # if i==0:
                #     print(episode_data['images'][cam_id][i].shape)
                #     jpg=np.array(episode_data['images'][cam_id][i])
                #     print(jpg[:10])

                #     DEFAULT_IMAGE_PATH # "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.jpg"
                #     os._exit(0)
                img_arr=to_numpy(episode_data['images'][cam_id][i])
                if i==0:
                    dataset.img_format=detect_img_ext(img_arr)
                frame_dict[f"observation.images.{cam_id}"] = to_numpy(episode_data['images'][cam_id][i])
                    
            dataset.add_frame(frame_dict)
        
        # logger.info(f"Adding {num_frames} frames... finish")

        
        dataset.save_episode(parallel_encoding=True,logger=logger)
       
        
        # 清理临时目录
        
    # except Exception as e:
    #     result['error'] = f"{e}\n{traceback.format_exc()}"
    #     logger.error(f"Failed processing episode {episode_index}: {e}")
    
    return dataset


def convert_episode_wrapper(args: tuple) -> dict:
    """进程池包装器"""

    h5_paths, output_dir, repo_id, episode_index, fps, vcodec, crf, task_override = args

    result = {
            'episode_index': episode_index,
            'success': False,
            'frames': 0,
            'error': None,
            'dataset_path': None,
            "episode_cnt":len(h5_paths)

        }
    
    try:
        dataset=None
        for h5_path in h5_paths:
            # logger.info(f"load h5 {h5_path} ...")
            episode_data = load_a2d_h5(h5_path)
            # logger.info(f"cvt task ...")
            dataset= convert_episode(
                episode_data=episode_data,
                output_dir=output_dir,
                repo_id=repo_id,
                episode_index=episode_index,
                fps=fps,
                vcodec=vcodec,
                crf=crf,
                task_override=task_override,
            )

        # logger.info(f"finish")
        
        
 
        
        result['success'] = True
        result['frames'] = 1
        result['dataset_path'] = str(dataset.root)
        
        # print("finish")
        return result
    except Exception as e:
        print(traceback.format_exc())
        return {
            'episode_index': episode_index,
            'success': False,
            'frames': 0,
            'error': f"{e}\n{traceback.format_exc()}",
            'dataset_path': None,
        }


# =============================================================================
# 主入口
# =============================================================================

def find_episodes(data_dir: Path) -> List[Path]:
    """查找所有可转换的 episode H5 文件"""
    episodes = []
    
    # 格式1: 子目录中的 h5 文件 (序列号/序列号.h5)
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            h5_files = list(subdir.glob("*.h5"))
            if h5_files:
                episodes.append(h5_files[0])
    
    # 格式2: 直接在目录下的 .h5 文件
    for h5_file in sorted(data_dir.glob("*.h5")):
        if h5_file not in episodes:
            episodes.append(h5_file)
    
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="智元 A2D 机器人数据转换为 LeRobot 格式"
    )
    
    parser.add_argument("--input", type=Path, required=True,
                        help="数据目录 (包含 H5 文件)")
    parser.add_argument("--output", type=Path, required=True,
                        help="输出目录")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace 仓库 ID")
    parser.add_argument("--task", type=str, nargs='+', default=["manipulation_task"],
                        help="任务描述 (默认: manipulation_task，可以不用引号，多个词会自动拼接)")
    parser.add_argument("--fps", type=int, default=30,
                        help="数据集帧率 (默认: 30)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行工作进程数 (默认: 4)")
    parser.add_argument("--vcodec", type=str, default="libsvtav1",
                        help="视频编码器 (默认: libsvtav1)")
    parser.add_argument("--crf", type=int, default=30,
                        help="视频质量 CRF (默认: 30)")
    
    args = parser.parse_args()
    
    # 处理 task 参数：如果是列表，拼接成字符串
    if isinstance(args.task, list):
        args.task = ' '.join(args.task)

    if args.repo_id is None:
        args.repo_id = args.output.name  # 使用输出目录名作为默认 repo_id

    # 查找 episodes
    episodes = find_episodes(args.input)
    # episodes=episodes[:30]
    if not episodes:
        print(f"No .h5 files found in {args.input}")
        sys.exit(1)
    
    print(f"Found {len(episodes)} episodes:")
    # for path in episodes:
    #     print(f"  - {path.name}")
    
    print(f"Using {args.workers} workers")
    
    # 设置输出目录
    output_root = args.output
    if output_root.exists():
        shutil.rmtree(output_root)
    separate_dir = output_root.parent / f"{output_root.name}_separate_episodes"
    separate_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建任务
    tasks = []

    batch_size = 4
    for i in range(0, len(episodes), batch_size):
        batch_eps = episodes[i:i + batch_size]

        tasks.append((
            batch_eps,                # 变成 list
            separate_dir,
            args.repo_id,
            i // batch_size,          # batch id
            args.fps,
            args.vcodec,
            args.crf,
            args.task
        ))
    
    # 并行处理
    success_datasets = []
    failed_episodes = []
    
    start_time = time.time()

    pbar=tqdm(total=len(episodes), desc="Converting Episodes")

    # os.setpgrp()  # 让当前进程成为进程组 leader（Linux）
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
    # with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_episode_wrapper, t): t for t in tasks}
        

        for future in as_completed(futures):
            res = future.result()
            # print("res",res['success'])
            if res['success']:
                success_datasets.append(Path(res['dataset_path']))
                # print(f"✓ Episode {res['episode_index']}: {res['frames']} frames")
            else:
                failed_episodes.append(f"Episode {res['episode_index']}: {res['error']}")
                # print(f"✗ Episode {res['episode_index']}: FAILED")
            pbar.update(res["episode_cnt"])
        
    
    
    elapsed = time.time() - start_time
    
    # 总结
    print(f"\n{'='*70}")
    print(f"Conversion completed in {elapsed:.1f}s")
    print(f"Success: {len(success_datasets)} / {len(tasks)}")
    print(f"Failed: {len(failed_episodes)}")
    print(f"{'='*70}")
    
    if failed_episodes:
        print("\nFailed episodes:")
        for f in failed_episodes[:5]:
            print(f"  - {f[:200]}...")
    
    if not success_datasets:
        print("No valid datasets. Exiting.")
        sys.exit(1)
    
    # 合并数据集
   
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
        
        print(f"\nMerge Complete!")
        print(f"Output: {output_root}")
        print(f"Total Episodes: {merged.meta.total_episodes}")
        print(f"Total Frames: {merged.meta.total_frames}")
        
        # 清理临时目录
        if separate_dir.exists():
            shutil.rmtree(separate_dir)
            print("Cleaned up temporary episode datasets.")


if __name__ == "__main__":
    main()
