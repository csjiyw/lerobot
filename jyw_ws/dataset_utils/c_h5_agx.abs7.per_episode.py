#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple
import multiprocessing as mp

import h5py
import numpy as np

import torch
import torch.nn.functional as F

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

H = 224
W = 224

CAM_KEYMAP = {
    "/camera/color/front": "observation.images.head",
    "/camera/color/right": "observation.images.hand_right",
}

DEFAULT_MOTORS_BOTH14 = (
    [f"J_arm_r_0{i}" if i < 10 else f"J_arm_r_{i}" for i in range(1, 7)] + ["J_gripper_r"]
)

def bgr_bhwc_to_rgb_chw_224(x):
    """
    x: np.ndarray or torch.Tensor, shape (B,H,W,C), BGR uint8
    return: torch.Tensor, shape (B,3,224,224), float/uint8(取决于输入)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # BGR -> RGB
    x = x[..., [2, 1, 0]]

    # BHWC -> BCHW
    x = x.permute(0, 3, 1, 2).contiguous()

    # resize
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x

def load_episode_h5(path_h5: str, step: int = 1):
    assert step >= 1
    with h5py.File(path_h5, "r") as root:
        T = root["/arm/jointStatePosition/puppetLeft"].shape[0]
        idx = np.arange(0, T, step)

        left_arm_action  = root["/arm/jointStatePosition/masterLeft"][idx]
        left_arm_state   = root["/arm/jointStatePosition/puppetLeft"][idx]

        right_arm_action = root["/arm/jointStatePosition/masterRight"][idx]
        right_arm_state  = root["/arm/jointStatePosition/puppetRight"][idx]

        images = {}
        for cam in CAM_KEYMAP.keys():
            images[cam] = root[cam][idx]  # (T',H,W,C) or (T',) bytes -> 取决于你数据实际存法
    return (
        left_arm_action.astype(np.float32),
        left_arm_state.astype(np.float32),
        right_arm_action.astype(np.float32),
        right_arm_state.astype(np.float32),
        images,
    )

def find_motion_range(states: np.ndarray, eps: float = 1e-4) -> Tuple[Optional[int], Optional[int]]:
    T = states.shape[0]
    if T < 2:
        return None, None
    diffs = np.max(np.abs(states[1:] - states[:-1]), axis=1)
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
    end   = min(T, last_motion + tail_keep + 1)
    if start >= end:
        start, end = 0, T
    return start, end

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
        "video.height": H,
        "video.width": W,
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
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
            "info": vid_info,
        }
    return feats

def _discover_episode_ids(data_dir: str) -> list[int]:
    ids = []
    pat = re.compile(r"episode(\d+)\.hdf5$")
    for p in Path(data_dir).glob("episode*.hdf5"):
        m = pat.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)

def process_one_episode(args_tuple):
    """
    单进程 worker：只处理一个 episode，并写到独立 repo_id
    """
    (
        epi,
        src_root,
        base_repo_id,
        task_text,
        raw_fps,
        step,
        head_keep,
        tail_keep,
        eps,
    ) = args_tuple

    # 避免 CPU 线程爆炸（非常重要：多进程+torch 很容易过载）
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    fps = int(raw_fps / step)

    h5_path = os.path.join(src_root, f"episode{epi}.hdf5")
    if not os.path.exists(h5_path):
        return (epi, False, f"missing file: {h5_path}")

    # 先 probe 相机 keys（每个 episode 自己做也行；更稳）
    _, _, _, _, images0 = load_episode_h5(h5_path, step=step)
    std_cam_list = sorted([CAM_KEYMAP.get(k, k) for k in images0.keys()])

    # 你现在 state_dim=7（6+grip），但你代码里 state/action 实际拼的是 r_s[t] / r_a[t]
    # 确保 r_s / r_a 的维度确实是 7，否则这里要对齐
    state_dim = 7
    motor_names = DEFAULT_MOTORS_BOTH14  # 这里其实是 7 个名字（你定义的是右臂6+grip），OK

    feats = build_features_dict(
        cams=std_cam_list,
        state_dim=state_dim,
        motors_names=motor_names,
        fps=fps
    )

    # 每个 episode 独立 repo_id（目录不共享 → 可以并行）
    repo_id = f"{base_repo_id}_ep{epi:06d}"
    out_root = Path(HF_LEROBOT_HOME) / repo_id
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="agelix",
        fps=fps,
        features=feats,
        # image_writer_threads=4,     # 每进程少一点，避免总线程数爆炸
        # image_writer_processes=8,   # 强烈建议：外层已经多进程了，这里别再开进程
    )

    l_a, l_s, r_a, r_s, images = load_episode_h5(h5_path, step=step)

    T = r_a.shape[0]
    if T < 2:
        return (epi, False, f"too short T={T}")

    start, end = compute_trim_range_by_state(
        r_a,
        head_keep=head_keep,
        tail_keep=tail_keep,
        eps=eps,
    )

    # 一次性把图像转成 torch (T,3,224,224)
    for raw_cam, seq in images.items():
        std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
        if std_cam not in std_cam_list:
            continue
        images[raw_cam] = bgr_bhwc_to_rgb_chw_224(seq)  # torch tensor

    # 写帧
    for t in range(start, end):
        state_t = np.concatenate([r_s[t]], axis=0)
        action_t = np.concatenate([r_a[t]], axis=0)

        frame = {
            "observation.state": state_t.astype(np.float32),
            "action": action_t.astype(np.float32),
            "task": task_text,
        }

        for raw_cam, seq in images.items():
            std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
            if std_cam not in std_cam_list:
                continue
            # (3,224,224) torch -> numpy uint8/float
            frame[std_cam] = seq[t].numpy()

        dataset.add_frame(frame=frame)

    dataset.save_episode(parallel_encoding=False)
    return (epi, True, str(out_root))

def main():
    task_name="agx_pick_place_yellow_capsule_to_white_plate_right_arm"
    config="fps30_cam2_img224_abs7_trim_head_tail_test"

    parser = argparse.ArgumentParser(description="Convert episodes to LeRobot v3 (one dataset per episode, parallel).")
    parser.add_argument("--expert_data_num", type=int, default=0, help="0=all, else first N episodes")
    parser.add_argument("--repo_id", type=str, default=f"jyw/task_{task_name}_cfg_{config}")
    parser.add_argument("--root", type=str, default="/workspace/data/data_no_index", help="source root")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers")
    args = parser.parse_args()

    raw_fps = 30
    step = 1

    data_dir = args.root
    all_ids = _discover_episode_ids(data_dir)
    if not all_ids:
        raise FileNotFoundError(f"No episode*.hdf5 found under {data_dir}")

    if args.expert_data_num == 0:
        target_ids = all_ids
    else:
        target_ids = all_ids[:args.expert_data_num]

    task_text = "Pick up the small yellow capsule-shaped container and place it into the white plate"

    # 并行任务参数
    jobs = []
    for epi in target_ids:

        jobs.append((
            epi,
            data_dir,
            args.repo_id,   # base_repo_id
            task_text,
            raw_fps,
            step,
            5,      # head_keep
            1,      # tail_keep
            1e-2,   # eps
        ))

    # spawn 更稳（尤其你后面可能在某些环境里用到 torch）
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_episode, jobs), total=len(jobs)))

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]

    print(f"\n[DONE] ok={len(ok)} bad={len(bad)}")
    if bad:
        print("[FAILED EPISODES]")
        for epi, _, msg in bad[:50]:
            print(f"  - episode{epi}: {msg}")
    

    import subprocess
    import json

    # ok 结果里你可以返回 repo_id（建议），或者这里按 epi 推导 repo_id


    ok_epis = sorted(epi for epi, success, _ in results if success)

    repo_ids_to_merge = [f"{args.repo_id}_ep{epi:06d}" for epi in ok_epis]

    # 目标合并后的 repo_id（你想叫啥都行）
    merged_repo_id = f"{args.repo_id}_merged"

    cmd = [
        "lerobot-edit-dataset",
        "--repo_id", merged_repo_id,
        "--operation.type", "merge",
        "--operation.repo_ids", json.dumps(repo_ids_to_merge),
    ]

    print("[MERGE CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"[DONE] merged into: {merged_repo_id}")

if __name__ == "__main__":
    main()
