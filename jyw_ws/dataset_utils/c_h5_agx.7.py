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
    "/camera/color/front": "observation.images.head",
    "/camera/color/right": "observation.images.hand_right",
}

# —— 关节命名（每臂6关节 + 夹爪1）——
DEFAULT_MOTORS_BOTH14 = (
    [f"J_arm_r_0{i}" if i < 10 else f"J_arm_r_{i}" for i in range(1, 7)] + ["J_gripper_r"]
)

def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name:50s} dtype={obj.dtype} shape={obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"[GROUP]   {name}")

import torch
import torch.nn.functional as F

def bgr_bhwc_to_rgb_chw_224(x):
    """
    seq: np.ndarray or torch.Tensor, shape (B,H,W,C), RGB
    return: torch.Tensor, shape (B,3,224,224)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    # BGR -> RGB（通道反转，几乎零成本）
    x = x[..., [2, 1, 0]]

    # BHWC -> BCHW
    x = x.permute(0, 3, 1, 2).contiguous()

    # resize
    x = F.interpolate(
        x,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )
    return x

def load_episode_h5(
    path_h5: str,
    step: int = 1
):
    assert step >= 1

    with h5py.File(path_h5, "r") as root:

        # 读取原始长度
        T = root["/arm/jointStatePosition/puppetLeft"].shape[0]

        idx = np.arange(0, T, step)

        # ===== arms =====
        left_arm_action  = root["/arm/jointStatePosition/masterLeft"][idx]   # (T',)
        left_arm_state   = root["/arm/jointStatePosition/puppetLeft"][idx]    # (T',6)

        right_arm_action = root["/arm/jointStatePosition/masterRight"][idx]  # (T',)
        right_arm_state  = root["/arm/jointStatePosition/puppetRight"][idx]   # (T',6)

        # ===== images =====
        images = {}
        for cam in CAM_KEYMAP.keys():
            images[cam] = root[cam][idx]   # (T',) bytes / encoded frames

    return (
        left_arm_action.astype(np.float32),
        left_arm_state.astype(np.float32),
        right_arm_action.astype(np.float32),
        right_arm_state.astype(np.float32),
        images,
    )

def jpgbytes_to_chw_rgb(img_bgr):




    # arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    # img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed")
    
    # cv2.imwrite("test_color.jpg", img_bgr)

    # exit()

    # img_bgr = cv2.resize(img_bgr, (W,H))
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_bgr[..., ::-1]   # BGR → RGB

    chw = np.transpose(img_rgb, (2, 0, 1))#.astype(np.uint8)      #(H,W,C) (3,H,W)
    return chw

def build_features_dict(cams, state_dim, motors_names, fps):
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names,
            "fps":30
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names,
            "fps":30
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


def main():
    task_name="agx_pick_place_yellow_capsule_to_white_plate_right_arm"
    config="fps30_cam2_img224_state6dof_delta_action_gripper_abs_trim_head_tail_test2"

    parser = argparse.ArgumentParser(description="Convert episodes to LeRobot v3.")
    parser.add_argument("--expert_data_num", type=int,default=0, help="number of episodes to process")
    parser.add_argument(
        "--repo_id", type=str, 
        default=f"jyw/task_{task_name}_cfg_{config}")
    parser.add_argument("--root", type=str, default="/workspace/data/data_no_index", help="source root")
    args = parser.parse_args()

    raw_fps=30
    step=1
    fps=int(raw_fps/step)

    src_root = os.path.join(args.root)
    episodes_to_read = args.expert_data_num

    out_root = Path(HF_LEROBOT_HOME) / args.repo_id
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    # 探测相机
    probe_h5 = os.path.join(src_root, f"episode0.hdf5")
    _, _, _, _, images0 = load_episode_h5(probe_h5)
    std_cam_list = sorted([CAM_KEYMAP.get(k, k) for k in images0.keys()])

    # 14 维： (6+1) * 2
    state_dim   = 7
    motor_names = DEFAULT_MOTORS_BOTH14

    feats = build_features_dict(
        cams=std_cam_list,
        state_dim=state_dim,
        motors_names=motor_names,
        fps=fps
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="agelix",
        fps=fps,
        features=feats,
        image_writer_threads=4,
        image_writer_processes=8,
        # batch_encoding_size=2
    )

    print(f"[INFO] HF_LEROBOT_HOME = {HF_LEROBOT_HOME}")
    print(f"[INFO] Repo path        = {out_root}")
    print(f"[INFO] Cameras          = {std_cam_list}")
    print(f"[INFO] State dim        = {state_dim} (6+grip)")

    from tqdm import trange

    


    def _discover_episode_ids(data_dir: str) -> list[int]:
        ids = []
        pat = re.compile(r"episode(\d+)\.hdf5$")
        for p in Path(data_dir).glob("episode*.hdf5"):
            m = pat.search(p.name)
            if m:
                ids.append(int(m.group(1)))
        return sorted(ids)

    data_dir = os.path.join(src_root)
    all_ids = _discover_episode_ids(data_dir)

    if not all_ids:
        raise FileNotFoundError(f"No episode*.hdf5 found under {data_dir}")

    # episodes_to_read: 0 = all; >0 = take first N after sorting
    if episodes_to_read == 0:
        target_ids = all_ids
    else:
        target_ids = all_ids[:episodes_to_read]

    # target_ids=[367,1316]

    # for epi in tqdm(range(len(target_ids))):
    for epi in tqdm(target_ids):
        h5_path = os.path.join(src_root, f"episode{epi}.hdf5")

   
        l_a, l_s, r_a, r_s, images = load_episode_h5(h5_path,step)

        start,end = compute_trim_range_by_state(
            r_a,
            head_keep=5,
            tail_keep=1,          # 你可以改 30
            eps=1e-2,              # 视你动作量级可调
        )

        

        T = l_a.shape[0]
        if T < 2:
            print(f"[WARN] episode{epi} too short (T={T}), skip")
            continue
        
        print(f"epi:{epi} origin T:{T} s:{start} e:{T-end} keep:{end-start}")





        # import matplotlib
        # matplotlib.use("Agg")
        # import matplotlib.pyplot as plt

        # save_dir = "plots_state_action"
        # os.makedirs(save_dir, exist_ok=True)

        # states  = np.stack(r_s[start:end])   # (T, 7)
        # actions = np.stack(r_a[start:end])  # (T, 7) = state[t+1]

        # s_off=0
        # a_off=0
        # T = min(300, len(states) - s_off,len(states) - a_off)

        # for joint_id in range(states.shape[1]):
        #     plt.figure(figsize=(10, 4))
            
        #     # plt.plot(states[:T, joint_id], label="state[t]")
        #     # plt.plot(actions[a_off:T+a_off, joint_id], label=f"action[t+{a_off}]")
        #     plt.plot(actions[:T, joint_id]-states[:T, joint_id], label=f"at-st")
        #     # plt.plot(states[s_off:T+s_off, joint_id], "--", label=f"state[t+{s_off}] (gt)")

        #     plt.xlabel("t")
        #     plt.ylabel("joint value")
        #     plt.title(f"Joint {joint_id}")
        #     plt.legend()
        #     plt.tight_layout()

        #     out_path = os.path.join(save_dir, f"joint_{joint_id:02d}.png")
        #     plt.savefig(out_path, dpi=150)
        #     plt.close()   # 非常重要（防内存泄漏）
        
        # exit(0)

        # very fast image transfer
        for raw_cam, seq in images.items():
            std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
            if std_cam not in std_cam_list:
                continue

            images[raw_cam] = bgr_bhwc_to_rgb_chw_224(seq)
        
        # os.makedirs("debug_frames", exist_ok=True)


    
        for t in range(start,end):
            state_t = np.concatenate([
                r_s[t],
            ], axis=0)

            action_t = np.concatenate([
                r_a[t]
            ], axis=0)

            frame = {
                "observation.state": state_t,
                "action": action_t,
            }
            for raw_cam, seq in images.items():
                std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
                if std_cam not in std_cam_list:
                    continue
                frame[std_cam] = seq[t].numpy()

                # save_chw_rgb(
                #     seq[t],
                #     f"debug_frames/{std_cam}_t{t}.jpg"
                # )


            frame["task"]="Pick up the small yellow capsule-shaped container and place it into the white plate"
            dataset.add_frame(frame=frame)
        
        # exit()

        # print(instructions["instructions"])

        dataset.save_episode()

    print(f"[DONE] Converted {len(target_ids)} episode(s) to {out_root}")

if __name__ == "__main__":
    main()
