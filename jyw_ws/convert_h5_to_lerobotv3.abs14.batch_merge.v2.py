#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from dataclasses import dataclass,field

import parser

@dataclass(slots=True)
class Config:
    # Width of the image (default: 640)
    W: int = 640
    
    # Height of the image (default: 480)
    H: int = 480
    
    # Raw FPS value (default: 30)
    raw_fps: int = 30
    
    # Step size for FPS calculation (default: 2)
    # It represents how often the frames are sampled from the original raw FPS.
    # For example, if raw_fps is 30 and step is 2, the effective FPS will be 15.
    step: int = 1
    
    # Robot type (default: "agx")
    # This could be the type of robot you're using for training or simulation.
    robot: str = "agx"
    
    # Task description (default: "fold the blue cloth towel neatly")
    # This is the description of the task the robot is performing.
    task_text: str = "fold the blue cloth towel neatly"
    
    # Directory where the data is stored (default: "/workspace/data/downloads/537e9a6f62034a19aad056fe21b53fbb")
    # This is the directory path for your dataset or related files.
    data_dir: str = "/workspace/data/downloads/537e9a6f62034a19aad056fe21b53fbb"
    
    # Number of worker threads for data processing (default: 16)
    # This controls how many threads to use when processing the dataset (e.g., loading, transforming).
    workers: int = 16
    
    # How many episodes per dataset repository (default: 1)
    # This controls the number of episodes that will be grouped into a dataset during training.
    episodes_per_dataset: int = 5
    
    # Size of batch encoding (default: 0)
    # This determines how large each batch of data will be when encoding or processing the data.
    batch_encoding_size: int = 0
    
    # Number of threads to use for image writing (default: 0)
    # This allows you to control how many threads are used when saving image data.
    image_writer_threads: int = 0
    
    # Number of processes for image writing (default: 0)
    # This allows you to control how many processes to spawn for saving image data.
    image_writer_processes: int = 0
    
    # Flag to enable parallel encoding (default: False)
    # This flag, if set to True, will allow parallel processing during data encoding.
    parallel_encoding: bool = False
    
    # Number of frames to keep from the head of the sequence (default: 5)
    # This defines how many initial frames will be kept when processing each episode.
    head_keep: int = 5
    
    # Number of frames to keep from the tail of the sequence (default: 1)
    # This defines how many final frames will be kept when processing each episode.
    tail_keep: int = 1
    
    # Epsilon value used for thresholding (default: 1e-2)
    # This value could be used for filtering or processing data that is below a certain threshold.
    eps: float = 1e-2
    
    # Flag to force the start frame to be zero (default: False)
    # If set to True, forces the data to start at frame 0.
    force_start0: bool = False
    
    # A custom note or tag for the configuration (default: "test1")
    # This can be used to add metadata or a label to identify different runs or configurations.
    note: str = "test1"
    
    # The number of expert data samples to use (default: 0)
    # If set to 0, it means using all available expert data. Otherwise, use the first N samples.
    expert_data_num: int = 0

    fps: int = field(init=False)  # 声明为字段，但不从外部传入

    repo_id: str = field(init=False)  # 声明为字段，但不从外部传入

    def __post_init__(self):
        # This method is executed after the class is initialized to calculate any derived properties.
        
        # Calculate the FPS based on raw FPS and step value.
        # The FPS is reduced by the step factor.
        # For example, if raw_fps=30 and step=2, then fps will be 15.
        self.fps = self.raw_fps // self.step
        
        # Create a unique repository ID for the configuration that includes robot type, task, FPS, width, height, and note.
        self.repo_id = f"jyw/{self.robot}_{self.task_text.lower().replace('.', '').replace(' ', '_')}_fps{self.fps}_w{self.W}h{self.H}_{self.note}"



@parser.wrap()
def init(cfg:Config)->Config:
    return cfg

args:Config = init()

print(args)

os.environ["LEROBOT_W"] = str(args.W)
os.environ["LEROBOT_H"] = str(args.H)

# Parse command-line arguments


from pathlib import Path
from typing import Optional, Tuple, List

import h5py
import numpy as np

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

import subprocess
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import multiprocessing as mp
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()



# 原始相机键 -> 目标相机键
CAM_KEYMAP = {
    "/cameras/head/color/data": "observation.images.head",
    "cameras/hand_left/color/data": "observation.images.hand_left",
    "/cameras/hand_right/color/data": "observation.images.hand_right",
}

# —— 关节命名（右臂6关节 + 夹爪1）——
DEFAULT_MOTORS = (
    [f"J_arm_l_0{i}" if i < 10 else f"J_arm_l_{i}" for i in range(1, 7)] + ["J_gripper_l"]
    + [f"J_arm_r_0{i}" if i < 10 else f"J_arm_r_{i}" for i in range(1, 7)] + ["J_gripper_r"]
)

def load_episode_h5(path_h5: str, step: int = 1):
    assert step >= 1

    with h5py.File(path_h5, "r") as root:
        T = root["/joints/state/arm/position"].shape[0]
        idx = np.arange(0, T, step)

        # ===== arms =====
        arm_action  = root["/joints/action/arm/position"][idx]    # (T',12)
        arm_state   = root["/joints/state/arm/position"][idx]     # (T',12)

        ef_action   = root["/joints/action/effector/position"][idx]  # (T',2)
        ef_state    = root["/joints/state/effector/position"][idx]   # (T',2)

        left_arm_action      = arm_action[:, :6]     # (T', 6)
        left_gripper_action  = ef_action[:, :1]       # (T', 1)

        right_arm_action      = arm_action[:, 6:12]     # (T', 6)
        right_gripper_action  = ef_action[:, 1:2]       # (T', 1)
        action = np.concatenate([left_arm_action,left_gripper_action, right_arm_action, right_gripper_action], axis=1).astype(np.float32)  # (T',7)


        left_arm_state       = arm_state[:, :6]      # (T', 6)
        left_gripper_state   = ef_state[:, :1]        # (T', 1)

        right_arm_state       = arm_state[:, 6:12]      # (T', 6)
        right_gripper_state   = ef_state[:, 1:2]        # (T', 1)
        state  = np.concatenate([left_arm_state,left_gripper_state,right_arm_state, right_gripper_state], axis=1).astype(np.float32)   # (T',7)

        # ===== images =====
        images = {}
        for cam in CAM_KEYMAP.keys():
            images[cam] = root[cam][idx]  # 期望为 (T',3,H,W) 或 (T',H,W,3) 或已是最终格式
    return action, state, images

def build_features_dict(cams, state_dim, motors_names, fps):
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names,
            "fps": fps,
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names,
            "fps": fps,
        },
    }

    vid_info = {
        "video.height": args.H,
        "video.width": args.W,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }

    for cam in cams:
        feats[cam] = {
            "dtype": "video",
            "shape": (3, args.H, args.W),
            "names": ["channels", "height", "width"],
            "info": vid_info,
        }
    return feats

def find_motion_range(states: np.ndarray, eps: float = 1e-4) -> Tuple[Optional[int], Optional[int]]:
    T = states.shape[0]
    if T < 2:
        return None, None
    diffs = np.max(np.abs(states[1:] - states[:-1]), axis=1)  # (T-1,)
    idx = np.where(diffs > eps)[0]
    if len(idx) == 0:
        return None, None
    return int(idx[0] + 1), int(idx[-1] + 1)

def compute_trim_range_by_state(
    states: np.ndarray,
    head_keep: int = 10,
    tail_keep: int = 20,
    eps: float = 1e-4,
) -> Tuple[int, int]:
    T = states.shape[0]
    first_motion, last_motion = find_motion_range(states, eps=eps)

    if first_motion is None:
        start = 0
        end = min(head_keep + tail_keep, T)
        return start, end

    start = max(0, first_motion - head_keep)
    end = min(T, last_motion + tail_keep + 1)

    if start >= end:
        start = 0
        end = T
    return start, end

def get_episode_path(data_home: Path) -> List[Path]:
    h5_files: List[Path] = []
    for p in sorted(data_home.iterdir()):
        if not p.is_dir():
            continue
        hid = p.name
        fp = p / f"{hid}.h5"
        if fp.exists():
            h5_files.append(fp)
    return h5_files

def _normalize_image_frame(x: np.ndarray) -> np.ndarray:
    """
    让输入变成 (3,H,W) uint8 或 float32（按你数据实际来）
    你当前代码直接 frame[cam]=seq[t]，这里做一个很轻量的兜底:
    - 若是 HWC -> CHW
    - 若是 CHW 但 dtype 不是 uint8/float32 也不强转（避免多余拷贝）
    注意：不做 resize（你数据里如果不是224，需要你在上游处理）
    """
    if x.ndim == 3 and x.shape[0] == 3:
        return x
    if x.ndim == 3 and x.shape[-1] == 3:
        return np.transpose(x, (2, 0, 1))
    return x

def process_one_group(job):
    """
    一组 h5 -> 一个 LeRobotDataset repo（repo 内包含多个 episode）
    返回: (group_idx, success, repo_id, msg)
    """
    h5_paths     = job["h5_paths"]
    group_idx    = job["group_idx"]
    base_repo_id = job["base_repo_id"]
    fps          = job["fps"]
    step         = job["step"]
    task_text    = job["task_text"]
    head_keep    = job["head_keep"]
    tail_keep    = job["tail_keep"]
    eps          = job["eps"]
    force_start0 = job["force_start0"]
    parallel_encoding = job["parallel_encoding"]

    repo_id = f"{base_repo_id}_g{group_idx:05d}"
    out_root = Path(HF_LEROBOT_HOME) / repo_id
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    std_cam_list = list(CAM_KEYMAP.values())
    state_dim = len(DEFAULT_MOTORS)
    motor_names = DEFAULT_MOTORS

    feats = build_features_dict(std_cam_list, state_dim, motor_names, fps)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=args.robot,
        fps=fps,
        features=feats,
        # 你已经验证 batch_encoding_size 越大越快：可在 CLI 里控制
        # batch_encoding_size=job["batch_encoding_size"],
        # 外层已经并行，这里不建议再开很多进程（除非你确认不会撑爆资源）
        # image_writer_threads=job["image_writer_threads"],
        # image_writer_processes=job["image_writer_processes"],
    )

    ok_cnt = 0
    fail_msgs = []

    for h5_path in h5_paths:
        try:
            a, s, images = load_episode_h5(h5_path, step=step)
            T = a.shape[0]
            if T < 2:
                fail_msgs.append(f"{Path(h5_path).name}: too short T={T}")
                continue

            start, end = compute_trim_range_by_state(
                a,
                head_keep=head_keep,
                tail_keep=tail_keep,
                eps=eps,
            )
            # if force_start0:
            start = 0

            for t in range(start, end):
                frame = {
                    "observation.state": s[t].astype(np.float32, copy=False),
                    "action": a[t].astype(np.float32, copy=False),
                    "task": task_text,
                }
                for raw_cam, seq in images.items():
                    std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
                    if std_cam not in std_cam_list:
                        continue
                    frame[std_cam] = _normalize_image_frame(seq[t])

                dataset.add_frame(frame=frame)

            dataset.save_episode(parallel_encoding=False)
            ok_cnt += 1

        except Exception as e:
            fail_msgs.append(f"{Path(h5_path).name}: {repr(e)}")

    success = ok_cnt > 0
    msg = f"ok_episodes={ok_cnt}/{len(h5_paths)}"
    if fail_msgs:
        msg += " | " + "; ".join(fail_msgs[:3])

    return (group_idx, success, repo_id, msg)

def chunk_list(lst: List[Path], chunk_size: int) -> List[List[Path]]:
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def main():
    


    

    all_h5 = get_episode_path(Path(args.data_dir))
    if not all_h5:
        raise FileNotFoundError(f"No *.h5 found under {args.data_dir}")

    if args.expert_data_num == 0:
        target_h5 = all_h5
    else:
        target_h5 = all_h5[:args.expert_data_num]

    groups = chunk_list(target_h5, max(1, args.episodes_per_dataset))

    jobs = []
    for group_idx, h5_list in enumerate(groups):
        jobs.append({
            "group_idx": group_idx,
            "h5_paths": [str(p) for p in h5_list],
            "base_repo_id": args.repo_id,
            "fps": args.fps,
            "step": args.step,
            "task_text": args.task_text,
            "head_keep": args.head_keep,
            "tail_keep": args.tail_keep,
            "eps": args.eps,
            "force_start0": args.force_start0,
            "batch_encoding_size": args.batch_encoding_size,
            "image_writer_threads": args.image_writer_threads,
            "image_writer_processes": args.image_writer_processes,
            "parallel_encoding": args.parallel_encoding,
        })

    print(f"[INFO] total episodes: {len(target_h5)}")
    print(f"[INFO] episodes_per_dataset: {args.episodes_per_dataset} => groups: {len(groups)}")
    print(f"[INFO] base repo_id: {args.repo_id}")
    print(f"[INFO] workers: {args.workers}")
    print(f"[INFO] batch_encoding_size: {args.batch_encoding_size}")
    print(f"[INFO] image_writer_threads: {args.image_writer_threads}, processes: {args.image_writer_processes}")
    print(f"[INFO] parallel_encoding: {args.parallel_encoding}")

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                            #  mp_context=ctx
                             ) as ex:
    # with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_group, job) for job in jobs]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]

    print(f"\n[DONE] groups ok={len(ok)} bad={len(bad)}")
    if bad:
        print("[FAILED GROUPS]")
        for group_idx, _, repo_id, msg in bad[:50]:
            print(f"  - g{group_idx:05d} ({repo_id}): {msg}")

    # merge：按 group_idx 排序
    ok_sorted = sorted((group_idx, repo_id) for group_idx, success, repo_id, _ in results if success)
    repo_ids_to_merge = [repo_id for _, repo_id in ok_sorted]

    merged_repo_id = f"{args.repo_id}_merged"

    repo_path = Path(HF_LEROBOT_HOME) / merged_repo_id

    if repo_path.exists():
        import shutil
    
        shutil.rmtree(repo_path)
    
    cmd = [
        "lerobot-edit-dataset",
        "--repo_id", merged_repo_id,
        "--operation.type", "merge",
        "--operation.repo_ids", json.dumps(repo_ids_to_merge),
    ]

    print("\n[MERGE CMD]")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"[DONE] merged into: {merged_repo_id}")

    # ===== cleanup intermediate group repos =====
    print("[CLEANUP] removing intermediate dataset repos...")

    for repo_id in repo_ids_to_merge:
        repo_path = Path(HF_LEROBOT_HOME) / repo_id
        if not repo_path.exists():
            print(f"  [SKIP] not found: {repo_id}")
            continue

        # 双保险：只删当前任务前缀的 group repo
        if not repo_id.startswith(f"{args.repo_id}_g"):
            print(f"  [SKIP] unexpected repo name: {repo_id}")
            continue

        import shutil
        shutil.rmtree(repo_path)
        print(f"  [DEL] {repo_id}")

    print("[CLEANUP DONE]")

if __name__ == "__main__":
    main()
