#!/usr/bin/env python3
"""
Base LeRobot Dataset Converter - Optimized Version

Usage:
    python base_converter.py --robot cobotmagic --input ./data --output ./output --workers 4

Optimizations:
    - HDF5 file handle caching to reduce open/close overhead
    - Preallocated numpy arrays for state/action loading
    - Configurable image writer threads (default: 2 for memory efficiency)
    - Early H5 structure validation
    - Incremental merge to reduce memory usage
    - Optional per-episode progress bar
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
from typing import Any, Dict, List, Optional, Set

import h5py
import numpy as np
from datasets import disable_progress_bar
from tqdm import tqdm, trange

disable_progress_bar()

os.environ["LEROBOT_W"] = str(640)  # target image W
os.environ["LEROBOT_H"] = str(480)  # target image H

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Robot Configurations - Load from JSON
# =============================================================================

CONFIG_DIR = Path(__file__).parent / "configs"


def get_robot_config(robot_name: str) -> dict:
    """Get robot configuration from JSON file."""
    config_path = CONFIG_DIR / f"{robot_name}.json"

    if not config_path.exists():
        available = [p.stem for p in CONFIG_DIR.glob("*.json")]
        raise ValueError(f"Unknown robot: {robot_name}. Available: {available}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


# =============================================================================
# Utility Functions
# =============================================================================

def to_numpy(x):
    """Convert to numpy, squeeze if length is 1."""
    if len(x) == 1:
        return x[0]
    return x


def detect_img_ext(arr) -> str:
    """Detect image format from bytes."""
    if arr.ndim==3:
        return "png"
    
    x = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
    if x.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if x.startswith(b"\xff\xd8\xff"):
        return "jpg"
    raise ValueError("Unknown image format")


def validate_h5_structure(h5_path: Path, robot_config: dict) -> Optional[str]:
    """
    Early validation of H5 file structure (Optimization #4).
    Returns error message if invalid, None if valid.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Check metadata
            if "metadata.json" not in f:
                return "Missing metadata.json"
            
            metadata = json.loads(f["metadata.json"][()])
            n = metadata.get("time_align_info", {}).get("frame_count", 0)
            if n == 0:
                return "Invalid frame_count in metadata"
            
            # Check camera info in metadata (not H5 keys)
            camera_info = metadata.get("data_info", {}).get("camera_info", {})
            if not camera_info:
                return "No camera info in metadata"
            
            # Check required state joints (skip optional)
            state_joints = robot_config.get("state_joints", {})
            optional_state = set(state_joints.get("optional", []))
            
            for joint_key in state_joints.get("order", []):
                if joint_key in optional_state:
                    continue
                path = state_joints["paths"].get(joint_key)
                if path and path not in f:
                    return f"Missing required state joint: {path}"
            
            return None
            
    except Exception as e:
        return f"H5 validation error: {str(e)}"


def build_features(metadata: dict, h: int, w: int, robot_config: dict) -> tuple:
    """Build LeRobot features dict from metadata and JSON config."""
    joint_info = metadata["data_info"]["joint_info"]
    # print(json.dumps(joint_info, indent=2))
    camera_info = metadata["data_info"]["camera_info"]

    fps = int(round(metadata["time_align_info"]["fps"]))

    # Get all cameras
    all_cameras = list(camera_info.keys())

    # Filter cameras if configured
    use_cameras = all_cameras
    if "camera_filter" in robot_config:
        filter_cfg = robot_config["camera_filter"]
        include_patterns = filter_cfg.get("include_patterns", [])
        if include_patterns:
            use_cameras = [
                cam for cam in all_cameras
                if cam in include_patterns or any(cam.startswith(p) for p in include_patterns)
            ]
            logger.info(f"Using cameras: {use_cameras} (filtered from {all_cameras})")

    # Build state names from config joint order
    # Support both flat keys (state/arm) and nested keys (state -> arm)
    state_names = []
    state_joints = robot_config["state_joints"]
    hardcoded_names =robot_config.get("hardcoded_names",[])


    for joint_key in state_joints["order"]:
        if joint_key in hardcoded_names:
            state_names.extend(hardcoded_names[joint_key])
        elif f"state/{joint_key}" in joint_info:
            state_names.extend(joint_info[f"state/{joint_key}"]["joint_names"])
        else:
            raise ValueError(f"state name not found: {joint_key}")



    # Build action names from config joint order
    action_names = []
    action_joints = robot_config["action_joints"]


    for joint_key in action_joints["order"]:
        if joint_key in hardcoded_names:
            action_names.extend(hardcoded_names[joint_key])
        elif f"action/{joint_key}" in joint_info:
            action_names.extend(joint_info[f"action/{joint_key}"]["joint_names"])
        # fall back to state
        elif f"state/{joint_key}" in joint_info:
            action_names.extend(joint_info[f"state/{joint_key}"]["joint_names"])
        else:
            raise ValueError(f"action name not found: {joint_key}")

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


# =============================================================================
# Data Loading
# =============================================================================
import cv2
def imgbyte2rgb_hwc(arr,w,h):
    # arr = np.frombuffer(jpeg_bytes, np.uint8)
    hwc = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    hwc = cv2.resize(hwc,(w,h))
    hwc = cv2.cvtColor(hwc, cv2.COLOR_BGR2RGB)
    # chw = np.transpose(hwc, (2, 0, 1))
    # return chw
    return hwc 

# =============================================================================
# Episode Conversion
# =============================================================================

def convert_episode(
    h5_path: Path,
    output_dir: Path,
    repo_id: str,
    episode_index: int,
    fps: int,
    robot_config: dict,
    task_override: str = None,
    dataset:LeRobotDataset=None,
) -> tuple:
    """Convert single episode to LeRobot format."""

    # Early validation (Optimization #4)
    error = validate_h5_structure(h5_path, robot_config)
    if error:
        raise ValueError(f"H5 validation failed: {error}")

    # Open H5 file directly (no cache)
    with h5py.File(h5_path, "r") as f:
        # Load metadata
        metadata = json.loads(f["metadata.json"][()])

        # Build features
        target_width, target_height = robot_config["target_image_size"]
        features, use_cameras = build_features(metadata, target_height, target_width, robot_config)
        print(json.dumps(features, indent=2))
        # Create dataset if first episode
        if dataset is None:
            episode_dir = output_dir / f"episode_{episode_index:04d}"
            if episode_dir.exists():
                shutil.rmtree(episode_dir)

            dataset = LeRobotDataset.create(
                repo_id=f"{repo_id}/episode_{episode_index:04d}",
                root=episode_dir,
                robot_type=robot_config["robot_type"],
                fps=fps,
                features=features,
                use_videos=True,
                image_writer_threads=2,
                image_writer_processes=0,
            )


        
        task = task_override if task_override else metadata.get("task_name", "manipulation_task")
        n = metadata["time_align_info"]["frame_count"]

        # Load state and action data inline
        state_joints = robot_config["state_joints"]
        action_joints = robot_config["action_joints"]

        end_index=n

        # Load state data
        state_data = {}
        state_dim = 0
        for joint_key in state_joints["order"]:
            path = state_joints["paths"][joint_key]
            if path in f:
                data = np.array(f[path][:n]).astype(np.float32).reshape(n, -1)
                state_data[joint_key] = data
                state_dim += data.shape[1] 
                # print(joint_key,data.shape[1] )
            else:
                raise KeyError(f"Required state joint '{joint_key}' not found: {path}")

        # Load action data
        action_data = {}
        action_dim = 0

        for joint_key in action_joints["order"]:
            path = action_joints["paths"][joint_key]
            if path in f:
                data = np.array(f[path][:n]).astype(np.float32).reshape(n, -1)
                action_data[joint_key] = data
                action_dim += data.shape[1]
            else:
                action_data[joint_key] = state_data[joint_key][1:n]
                action_dim += action_data[joint_key].shape[1]
                end_index = n-1
                

        # print(state_dim)
        # Preallocate and copy state
        state = np.zeros((n, state_dim), dtype=np.float32)
        col = 0
        for joint_key in state_joints["order"]:
            if joint_key in state_data:
                
                data = state_data[joint_key]
                # print(joint_key,data.shape,data.shape[1])
                state[:end_index, col:col + data.shape[1]] = data[:end_index]
                col += data.shape[1]

        # Preallocate and copy action
        action = np.zeros((n, action_dim), dtype=np.float32)
        col = 0
        for joint_key in action_joints["order"]:
            if joint_key in action_data:
                data = action_data[joint_key]
                action[:end_index, col:col + data.shape[1]] = data[:end_index]
                col += data.shape[1]

   

        
        
        print(f"State shape  : {state.shape}")
        print(f"Action shape : {action.shape}")
        print(f"Task         : {task}")
 

        # Add frames
        for i in trange(0,end_index):
            frame_dict = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }

            for cam_id in use_cameras:
                cam_key = f"cameras/{cam_id}/color/data"
                img_arr = to_numpy(f[cam_key][i])

                # disable to accelerate img write speed which need modify lerobot code
                img_arr=imgbyte2rgb_hwc(img_arr,target_width,target_height)

                if i == 0:
                    dataset.img_format = detect_img_ext(img_arr)

                

                frame_dict[f"observation.images.{cam_id}"] = img_arr

            dataset.add_frame(frame_dict)

            # Periodically drain image writer queue to reduce memory
            if i % 500 == 0 and i > 0:
                dataset.image_writer.wait_until_done()

        dataset.save_episode(parallel_encoding=True)

    return dataset


def convert_episode_wrapper(args: tuple) -> dict:
    """ProcessPoolExecutor wrapper."""
    h5_paths, output_dir, repo_id, episode_index, fps, robot_config, task_override = args

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
            dataset = convert_episode(
                h5_path,
                output_dir=output_dir,
                repo_id=repo_id,
                episode_index=episode_index,
                fps=fps,
                robot_config=robot_config,
                task_override=task_override,
                dataset=dataset,
            )

        result["success"] = True
        result["frames"] = 1
        result["dataset_path"] = str(dataset.root)

    except Exception as e:
        print(traceback.format_exc())
        result["error"] = f"{e}\n{traceback.format_exc()}"
        return result

    
    del dataset
    gc.collect()
    return result


# =============================================================================
# Episode Discovery
# =============================================================================

def find_episodes(data_dir: Path) -> List[Path]:
    """Find all H5 episode files."""
    episodes = []
    
    # Format 1: Subdirectories with H5 files
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            h5_files = list(subdir.glob("*.h5"))
            if h5_files:
                episodes.append(h5_files[0])
    
    # Format 2: H5 files directly in directory
    for h5_file in sorted(data_dir.glob("*.h5")):
        if h5_file not in episodes:
            episodes.append(h5_file)
    
    return episodes


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert robot data to LeRobot format")
    parser.add_argument("--robot", type=str, required=True,
                        help="Robot type")
    parser.add_argument("--input", type=Path, required=True, help="Input data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace repo ID")
    parser.add_argument("--task", type=str, nargs="+", default=["manipulation_task"],
                        help="Task description")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Episodes per batch (reduce for memory issues)")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip merge step (keep separate episode datasets)")

    args = parser.parse_args()

    if isinstance(args.task, list):
        args.task = " ".join(args.task)

    if args.repo_id is None:
        args.repo_id = args.output.name

    # Get robot config
    robot_config = get_robot_config(args.robot)

    # Find episodes
    episodes = find_episodes(args.input)[:1]
    if not episodes:
        print(f"No .h5 files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(episodes)} episodes")
    print(f"Using {args.workers} workers, batch_size={args.batch_size}")
    print(f"Robot type: {robot_config['robot_type']}")

    # Setup output
    output_root = args.output
    if output_root.exists():
        shutil.rmtree(output_root)

    separate_dir = output_root.parent / f"{output_root.name}_separate_episodes"
    separate_dir.mkdir(parents=True, exist_ok=True)

    # Build tasks
    tasks = []
    for i in range(0, len(episodes), args.batch_size):
        batch_eps = episodes[i:i + args.batch_size]
        tasks.append(
            (
                batch_eps,
                separate_dir,
                args.repo_id,
                i // args.batch_size,
                args.fps,
                robot_config,
                args.task,
            )
        )

    # Process
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

    # Merge step (Optimization #5: incremental merge)
    if not args.skip_merge:
        from lerobot.datasets.dataset_tools import merge_datasets

        print("\nMerging into final dataset...")
        
        # Incremental merge: merge in batches to reduce memory
        if len(success_datasets) > 10:
            logger.info(f"Merging {len(success_datasets)} datasets incrementally...")
            merged_so_far = None
            merge_batch_size = 10
            
            for i in range(0, len(success_datasets), merge_batch_size):
                batch = success_datasets[i:i + merge_batch_size]
                batch_datasets = [LeRobotDataset(root=dpath, repo_id=dpath.name) for dpath in batch]
                
                if merged_so_far is None:
                    merged_so_far = merge_datasets(
                        datasets=batch_datasets,
                        output_dir=output_root,
                        output_repo_id=args.repo_id,
                    )
                else:
                    # Merge with existing
                    batch_datasets.insert(0, merged_so_far)
                    merged_so_far = merge_datasets(
                        datasets=batch_datasets,
                        output_dir=output_root,
                        output_repo_id=args.repo_id,
                    )
                
                logger.info(f"Merged batch {i // merge_batch_size + 1}, total episodes: {merged_so_far.meta.total_episodes}")
                
                # Clean up intermediate datasets
                for ds in batch_datasets[1:]:
                    del ds
                gc.collect()
            
            merged = merged_so_far
        else:
            # Direct merge for small number of datasets
            datasets_to_merge = []
            for dpath in sorted(success_datasets):
                ds = LeRobotDataset(root=dpath, repo_id=dpath.name)
                datasets_to_merge.append(ds)
            
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
    else:
        print("\nMerge skipped. Separate episode datasets available at:")
        print(f"  {separate_dir}")


if __name__ == "__main__":
    main()
