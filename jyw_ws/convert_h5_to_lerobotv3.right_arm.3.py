#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
import re

import h5py
import numpy as np
import cv2

import lerobot

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

H=224
W=224

# 原始相机键 -> 目标相机键
CAM_KEYMAP = {
    "/cameras/head/color/data": "observation.images.head",
    # "/cameras/hand_right/color/data": "observation.images.hand_right",
}

# —— 关节命名（每臂6关节 + 夹爪1）——
DEFAULT_MOTORS = (
    [f"J_arm_r_0{i}" if i < 10 else f"J_arm_r_{i}" for i in range(1, 7)] + ["J_gripper_r"]
)

def load_episode_h5(
    path_h5: str,
    step: int = 1
):
    assert step >= 1

    with h5py.File(path_h5, "r") as root:

        # 读取原始长度
        T = root["/joints/state/arm/position"].shape[0]

        idx = np.arange(0, T, step)

        # ===== arms =====
        arm_action  = root["/joints/action/arm/position"][idx]   # (T',12)
        arm_state   = root["/joints/state/arm/position"][idx]    # (T',12)

        ef_action = root["/joints/action/effector/position"][idx]  # (T',2)
        ef_state  = root["/joints/state/effector/position"][idx]   # (T',2)

        right_arm_action = arm_action[:, 6:12]     # (T', 6)
        right_gripper_action = ef_action[:, 1:2]   # (T', 1)

        right_action = np.concatenate(
            [right_arm_action, right_gripper_action],
            axis=1
        )  # (T', 7)

        right_arm_state = arm_state[:, 6:12]      # (T', 6)
        right_gripper_state = ef_state[:, 1:2]    # (T', 1)

        right_state = np.concatenate(
            [right_arm_state, right_gripper_state],
            axis=1
        )  # (T', 7)



        # ===== images =====
        images = {}
        for cam in CAM_KEYMAP.keys():
            images[cam] = root[cam][idx]   # (T',) bytes / encoded frames

    return (
        right_action.astype(np.float32),
        right_state.astype(np.float32),
        images,
    )

def build_features_dict(cams, state_dim, motors_names, fps):
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names,
            "fps":fps
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names,
            "fps":fps
        }
    }
    vid_info = {
        "video.height": H,
        "video.width": W,
        # "video.codec": "raw",
        # "video.pix_fmt": "rgb24",
        "video.codec": "h264",     # <- metadata (optional)
        "video.pix_fmt": "yuv420p",# <- metadata (optional)
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False
    }
    for cam in cams:
        feats[cam] = {
            "dtype": "video",
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
            "info": vid_info
        }
    return feats

from typing import Optional, Tuple

def find_motion_range(
    states: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[Optional[int], Optional[int]]:
    """
    states: (T, S)

    返回:
        first_motion_idx : 第一次明显变化发生在 t（t 相对 t-1）
        last_motion_idx  : 最后一次明显变化发生在 t（t 相对 t-1）
    """
    T = states.shape[0]
    if T < 2:
        return None, None

    diffs = np.max(np.abs(states[1:] - states[:-1]), axis=1)  # (T-1,)
    idx = np.where(diffs > eps)[0]

    if len(idx) == 0:
        return None, None

    first_motion = int(idx[0] + 1)
    last_motion  = int(idx[-1] + 1)
    return first_motion, last_motion

def compute_trim_range_by_state(
    states: np.ndarray,
    head_keep: int = 10,
    tail_keep: int = 20,
    eps: float = 1e-4,
) -> Tuple[int, int]:
    """
    根据 state 自动裁剪 episode

    返回:
        (start, end) → states[start:end]

    规则:
    - start = max(0, first_motion - head_keep)
    - end   = min(T, last_motion + tail_keep + 1)
    """
    T = states.shape[0]

    first_motion, last_motion = find_motion_range(states, eps=eps)

    # 完全静止（例如 reset 出问题的数据）
    if first_motion is None:
        start = 0
        end = min(head_keep + tail_keep, T)
        return start, end

    start = max(0, first_motion - head_keep)
    end   = min(T, last_motion + tail_keep + 1)

    if start >= end:  # 兜底
        start = 0
        end = T

    return start, end


from PIL import Image

import os

def save_chw_rgb(seq_t, path):
    """
    seq_t: torch.Tensor or np.ndarray, shape (3,H,W), RGB
    """
    if hasattr(seq_t, "cpu"):
        img = seq_t.cpu().numpy()
    else:
        img = seq_t

    # (3,H,W) -> (H,W,3)
    img = np.transpose(img, (1, 2, 0))

    # 如果是 float，转 uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img, mode="RGB").save(path)

def get_episode_path(data_home):
    h5_files: list[Path] = []
    for p in sorted(data_home.iterdir()):
        if not p.is_dir():
            continue
        hid = p.name
        fp = p / f"{hid}.h5"
        if fp.exists():
            h5_files.append(fp)
    
    return h5_files

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import torch

def process_one_h5(job):
    """
    单个 h5 -> 单独一个 LeRobotDataset repo（只写一个 episode）
    job: dict
    """
    # 避免每个进程把 CPU 线程拉满
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    h5_path      = job["h5_path"]
    epi_idx      = job["epi_idx"]          # 用于排序/命名
    base_repo_id = job["base_repo_id"]
    fps          = job["fps"]
    step         = job["step"]
    task_text    = job["task_text"]

    # 每个 episode 一个 repo
    repo_id = f"{base_repo_id}_ep{epi_idx:06d}"
    out_root = Path(HF_LEROBOT_HOME) / repo_id
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    # 相机列表（你这里是固定的）
    std_cam_list = list(CAM_KEYMAP.values())

    state_dim   = len(DEFAULT_MOTORS)  # 7
    motor_names = DEFAULT_MOTORS

    feats = build_features_dict(
        cams=std_cam_list,
        state_dim=state_dim,
        motors_names=motor_names,
        fps=fps
    )

    # 这里非常关键：外层已经并行了，内部别再开进程
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="agelix",
        fps=fps,
        features=feats
    )

    # load
    a, s, images = load_episode_h5(str(h5_path), step=step)

    T = a.shape[0]
    if T < 2:
        return (epi_idx, False, repo_id, f"too short T={T}")

    start, end = compute_trim_range_by_state(
        a,
        head_keep=5,
        tail_keep=1,
        eps=1e-2,
    )
    start = 0  # 你原来强制 start=0，这里保持一致

    # 写帧
    for t in range(start, end):
        frame = {
            "observation.state": s[t].astype(np.float32),
            "action": a[t].astype(np.float32),
            "task": task_text,
        }

        for raw_cam, seq in images.items():
            std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
            if std_cam not in std_cam_list:
                continue
            frame[std_cam] = seq[t]   # 你这里 seq[t] 已经是 (3,H,W) RGB

        dataset.add_frame(frame=frame)

    dataset.save_episode(parallel_encoding=False)
    return (epi_idx, True, repo_id, "ok")


def main():
    task_name="agx_pick_silver_metal_battery_to_white_plate_right_arm"
    config="fps30_cam2_img224_abs7_trim_tail_test"

    parser = argparse.ArgumentParser(description="Convert episodes to LeRobot v3 (per-episode repo + parallel + merge).")
    parser.add_argument("--expert_data_num", type=int, default=0, help="0=all, else first N")
    parser.add_argument("--repo_id", type=str, default=f"jyw/task_{task_name}_cfg_{config}")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    # ===================== 用户配置 =====================
    data_dir = Path("/workspace/data/rclone-v1.68.2-linux-amd64/data/9f22b6b23ed64ced8fc8074d454e5672")

    raw_fps = 30
    step    = 1
    fps     = int(raw_fps / step)

    # 发现所有 h5
    all_h5 = get_episode_path(data_dir)
    if not all_h5:
        raise FileNotFoundError(f"No *.h5 found under {data_dir}")

    # 截断数量
    if args.expert_data_num == 0:
        target_h5 = all_h5
    else:
        target_h5 = all_h5[:args.expert_data_num]

    # 给每个 h5 一个稳定的序号 epi_idx（用于排序 & 命名）
    # 这里用“排序后的列表索引”最稳，不依赖 stem 是否纯数字
    jobs = []
    task_text = "Pick up the silver metal battery and place it into the white plate"
    for epi_idx, h5_path in enumerate(target_h5):
        jobs.append({
            "epi_idx": epi_idx,
            "h5_path": h5_path,
            "base_repo_id": args.repo_id,
            "fps": fps,
            "step": step,
            "task_text": task_text,
        })

    print(f"[INFO] total episodes to convert: {len(jobs)}")
    print(f"[INFO] base repo_id: {args.repo_id}")
    print(f"[INFO] workers: {args.workers}")

    # 并行（spawn 更稳）
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [ex.submit(process_one_h5, job) for job in jobs]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]

    print(f"\n[DONE] ok={len(ok)} bad={len(bad)}")
    if bad:
        print("[FAILED]")
        for epi_idx, _, repo_id, msg in bad[:50]:
            print(f"  - ep{epi_idx:06d} ({repo_id}): {msg}")

    # ========= merge: 按 epi_idx 排序 =========
    ok_sorted = sorted((epi_idx, repo_id) for epi_idx, success, repo_id, _ in results if success)
    repo_ids_to_merge = [repo_id for _, repo_id in ok_sorted]

    merged_repo_id = f"{args.repo_id}_merged"

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


if __name__ == "__main__":
    main()
