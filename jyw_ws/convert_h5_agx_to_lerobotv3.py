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

H=480
W=640

# 原始相机键 -> 目标相机键
CAM_KEYMAP = {
    "/camera/color/front": "observation.images.head",
    "/camera/color/left": "observation.images.hand_left",
    "/camera/color/right": "observation.images.hand_right",
}

# —— 关节命名（每臂6关节 + 夹爪1）——
DEFAULT_MOTORS_BOTH14 = (
    [f"J_arm_l_0{i}" if i < 10 else f"J_arm_l_{i}" for i in range(1, 7)] + ["J_gripper_l"] +
    [f"J_arm_r_0{i}" if i < 10 else f"J_arm_r_{i}" for i in range(1, 7)] + ["J_gripper_r"]
)

def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name:50s} dtype={obj.dtype} shape={obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"[GROUP]   {name}")


def load_episode_h5(path_h5):
    with h5py.File(path_h5, "r") as root:

        # root.visititems(visit)
        
        left_arm_action = root["/arm/jointStatePosition/masterLeft"][()]    # (T,)
        left_arm_state    = root["/arm/jointStatePosition/puppetLeft"][()]        # (T,6)  <-- 6 维
        right_arm_action= root["/arm/jointStatePosition/masterRight"][()]   # (T,)
        right_arm_state    = root["/arm/jointStateVelocity/puppetRight"][()]       # (T,6)

        images = {}
        for cam in CAM_KEYMAP.keys():
            images[cam] = root[cam][()]    # (T,) bytes
    return left_arm_action, left_arm_state, right_arm_action, right_arm_state, images

def jpgbytes_to_chw_rgb(img_bgr):
    # arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    # img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed")

    img_bgr = cv2.resize(img_bgr, (W,H))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.uint8)      # (3,H,W)
    return chw

def build_features_dict(cams, state_dim, motors_names, fps):
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": motors_names
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

def main():
    parser = argparse.ArgumentParser(description="Convert episodes to LeRobot v2 (both14).")

    parser.add_argument("--expert_data_num", type=int,default=2, help="number of episodes to process")
    parser.add_argument("--repo_id", type=str, default="jyw/pick_yellow_box")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--root", type=str, default="/workspace/data/data_no_index", help="source root")
    args = parser.parse_args()

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
    state_dim   = 14
    motor_names = DEFAULT_MOTORS_BOTH14

    feats = build_features_dict(
        cams=std_cam_list,
        state_dim=state_dim,
        motors_names=motor_names,
        fps=args.fps
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="qinglong_table",
        fps=args.fps,
        features=feats,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    print(f"[INFO] HF_LEROBOT_HOME = {HF_LEROBOT_HOME}")
    print(f"[INFO] Repo path        = {out_root}")
    print(f"[INFO] Cameras          = {std_cam_list}")
    print(f"[INFO] State dim        = {state_dim} (6+grip)*2")

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

   
        l_a, l_s, r_a, r_s, images = load_episode_h5(h5_path)

        T = l_a.shape[0]
        if T < 2:
            print(f"[WARN] episode{epi} too short (T={T}), skip")
            continue

        for t in range(T):
            # 左臂6 + 左夹爪1 + 右臂6 + 右夹爪1  → 14维
            state_t = np.concatenate([
                l_s[t].astype(np.float32),
                r_s[t].astype(np.float32),
            ], axis=0)

            action_t = np.concatenate([
                l_a[t].astype(np.float32),  
                r_a[t].astype(np.float32), 
            ], axis=0)

            frame = {
                "observation.state": state_t,
                "action": action_t,
            }
            for raw_cam, seq in images.items():
                std_cam = CAM_KEYMAP.get(raw_cam, raw_cam)
                if std_cam not in std_cam_list:
                    continue
                chw = jpgbytes_to_chw_rgb(seq[t])
                frame[std_cam] = chw

            frame["task"]="Lift the bottle headup from the table"
            dataset.add_frame(frame=frame)

        # print(instructions["instructions"])

        dataset.save_episode()

    print(f"[DONE] Converted {len(target_ids)} episode(s) to {out_root}")

if __name__ == "__main__":
    main()
