#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import io
import shutil
import h5py
from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
from PIL import Image
from tqdm import tqdm


from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ===================== 用户配置 =====================
data_home = Path("data/rclone-v1.68.2-linux-amd64/data/9f22b6b23ed64ced8fc8074d454e5672")

REPO_ID = "jyw/place_metallic_object_fast"   # 输出到: HF_LEROBOT_HOME/REPO_ID
ROBOT_TYPE = "generic"
FPS = 30
TASK_NAME = "place metallic object"
WIPE_OUTPUT = True
START_FRAME = 0

USE_HEAD_CAM = True
USE_HAND_LEFT_CAM = True
USE_HAND_RIGHT_CAM = True

# 是否把 state 也包含 velocity / angular（你现在的 H5 都有，这里默认全用）
USE_ARM_VEL = False
USE_EFFECTOR_VEL = False
USE_BASE_VEL_ANG = False
# ====================================================


# --------- H5 keys（根据你贴出来的结构固定） ----------
K = {
    "head_img": "cameras/head/color/data",
    "hand_left_img": "cameras/hand_left/color/data",
    "hand_right_img": "cameras/hand_right/color/data",

    "ts": "timestamp",

    "s_arm_p": "joints/state/arm/position",
    "s_arm_v": "joints/state/arm/velocity",
    "s_eff_p": "joints/state/effector/position",
    "s_eff_v": "joints/state/effector/velocity",
    "s_base_v": "joints/state/robot/velocity",
    "s_base_w": "joints/state/robot/angular",

    "a_arm_p": "joints/action/arm/position",
    "a_arm_v": "joints/action/arm/velocity",
    "a_eff_p": "joints/action/effector/position",
    "a_eff_v": "joints/action/effector/velocity",
    "a_base_v": "joints/action/robot/velocity",
    "a_base_w": "joints/action/robot/angular",
}
# ------------------------------------------------------





def infer_T_min(*dsets: h5py.Dataset) -> int:
    Ts = [int(d.shape[0]) for d in dsets if d is not None]
    return min(Ts) if Ts else 0


def build_state_action(f: h5py.File, t: int):
    """拼接 state/action 向量"""
    # state
    parts_s = [np.asarray(f[K["s_arm_p"]][t], dtype=np.float32).reshape(-1)]
    if USE_ARM_VEL:
        parts_s.append(np.asarray(f[K["s_arm_v"]][t], dtype=np.float32).reshape(-1))

    parts_s.append(np.asarray(f[K["s_eff_p"]][t], dtype=np.float32).reshape(-1))
    if USE_EFFECTOR_VEL:
        parts_s.append(np.asarray(f[K["s_eff_v"]][t], dtype=np.float32).reshape(-1))

    if USE_BASE_VEL_ANG:
        parts_s.append(np.asarray(f[K["s_base_v"]][t], dtype=np.float32).reshape(-1))
        parts_s.append(np.asarray(f[K["s_base_w"]][t], dtype=np.float32).reshape(-1))

    state = np.concatenate(parts_s, axis=0)

    # action
    parts_a = [np.asarray(f[K["a_arm_p"]][t], dtype=np.float32).reshape(-1)]
    # 你可以选择只用 position 作为 action；但既然 H5 里有 velocity，这里默认也拼进去
    # parts_a.append(np.asarray(f[K["a_arm_v"]][t], dtype=np.float32).reshape(-1))

    parts_a.append(np.asarray(f[K["a_eff_p"]][t], dtype=np.float32).reshape(-1))
    # parts_a.append(np.asarray(f[K["a_eff_v"]][t], dtype=np.float32).reshape(-1))

    parts_a.append(np.asarray(f[K["a_base_v"]][t], dtype=np.float32).reshape(-1))
    parts_a.append(np.asarray(f[K["a_base_w"]][t], dtype=np.float32).reshape(-1))

    action = np.concatenate(parts_a, axis=0)

    return state, action


def build_feature_spec(sample_h5: Path):
    """用 sample h5 推断图像 H/W，state/action dim"""
    with h5py.File(sample_h5, "r") as f:
        # 图像尺寸：解码第 0 帧（更可靠，因为压缩 bytes 不含 shape）
        head_hw = None
        hl_hw = None
        hr_hw = None
        H,W=480,640


        if USE_HEAD_CAM:
            
            head_hw = (H,W)  # (H,W)
        if USE_HAND_LEFT_CAM:
            
            hl_hw = (H,W)
        if USE_HAND_RIGHT_CAM:
            
            hr_hw = (H,W)

        # state/action dim：拼一次即可
        s0, a0 = build_state_action(f, 0)
        state_dim = 14#int(s0.shape[0])
        action_dim = 14#int(a0.shape[0])

    feats = {}
    if USE_HEAD_CAM and head_hw is not None:
        H, W = head_hw
        feats["observation.images.head"] = {
            "dtype": "video",
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
            "info": {"video.height": H, "video.width": W,
                      "video.codec": "av1",
                      "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                     "video.fps": FPS,
                       "video.channels": 3,
                         "has_audio": False}
        }
    if USE_HAND_LEFT_CAM and hl_hw is not None:
        H, W = hl_hw
        feats["observation.images.hand_left"] = {
            "dtype": "video",
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
            "info":  {"video.height": H, "video.width": W,
                      "video.codec": "av1",
                      "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                     "video.fps": FPS,
                       "video.channels": 3,
                         "has_audio": False}
        }
    if USE_HAND_RIGHT_CAM and hr_hw is not None:
        H, W = hr_hw
        feats["observation.images.hand_right"] = {
            "dtype": "video",
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
            "info":  {"video.height": H, "video.width": W,
                      "video.codec": "av1",
                      "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                     "video.fps": FPS,
                       "video.channels": 3,
                         "has_audio": False}
        }

    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"s_{i}" for i in range(state_dim)],
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"a_{i}" for i in range(action_dim)],
    }

    return feats


def generate_frames_from_h5_no_prev(
    f,
    T: int,
    task_name: str,
    collection_id: str,
    use_base: bool = False,   # 是否把底盘(velocity+angular)拼进 state/action
):
    d_head = f["cameras/head/color/data"] if "cameras/head/color/data" in f else None
    d_hl   = f["cameras/hand_left/color/data"] if "cameras/hand_left/color/data" in f else None
    d_hr   = f["cameras/hand_right/color/data"] if "cameras/hand_right/color/data" in f else None

    s_arm_p = f["joints/state/arm/position"]          # (T,12)
    s_eff_p = f["joints/state/effector/position"]     # (T,2)
    s_base_v= f["joints/state/robot/velocity"]        # (T,3)
    s_base_w= f["joints/state/robot/angular"]         # (T,3)

    a_arm_p = f["joints/action/arm/position"]         # (T,12)
    a_eff_p = f["joints/action/effector/position"]    # (T,2)
    a_base_v= f["joints/action/robot/velocity"]       # (T,3)
    a_base_w= f["joints/action/robot/angular"]        # (T,3)

    has_ts = "timestamp" in f
    d_ts = f["timestamp"] if has_ts else None

    for t in range(T):
        frame = {
            "task": task_name,

        }
        # if has_ts:
        #     frame["timestamp"] = np.array([int(d_ts[t])], dtype=np.int64)

        if d_head is not None:
            frame["observation.images.head"] = (d_head[t])
        if d_hl is not None:
            frame["observation.images.hand_left"] = (d_hl[t])
        if d_hr is not None:
            frame["observation.images.hand_right"] = (d_hr[t])

        # observation.state: 直接用 state 的当前帧
        obs = [
            np.asarray(s_arm_p[t], dtype=np.float32).reshape(-1),
            np.asarray(s_eff_p[t], dtype=np.float32).reshape(-1),
        ]
        act = [
            np.asarray(a_arm_p[t], dtype=np.float32).reshape(-1),
            np.asarray(a_eff_p[t], dtype=np.float32).reshape(-1),
        ]

        if use_base:
            obs += [
                np.asarray(s_base_v[t], dtype=np.float32).reshape(-1),
                np.asarray(s_base_w[t], dtype=np.float32).reshape(-1),
            ]
            act += [
                np.asarray(a_base_v[t], dtype=np.float32).reshape(-1),
                np.asarray(a_base_w[t], dtype=np.float32).reshape(-1),
            ]

        frame["observation.state"] = np.concatenate(obs, axis=0).astype(np.float32)
        frame["action"] = np.concatenate(act, axis=0).astype(np.float32)

        yield frame


def main():
    assert data_home.exists(), f"data_home not found: {data_home}"

    out_root = Path(HF_LEROBOT_HOME) / REPO_ID
    if WIPE_OUTPUT and out_root.exists():
        shutil.rmtree(out_root)

    # 找到所有 <id>/<id>.h5
    h5_files: list[Path] = []
    for p in sorted(data_home.iterdir()):
        if not p.is_dir():
            continue
        hid = p.name
        fp = p / f"{hid}.h5"
        if fp.exists():
            h5_files.append(fp)

    # 只用前 10 个测试
    # h5_files = h5_files[:10]

    if not h5_files:
        raise RuntimeError(f"No <id>/<id>.h5 found under {data_home}")

    # 推断 features
    features = build_feature_spec(h5_files[0])

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    for h5_path in tqdm(h5_files, desc="H5 → LeRobot episodes"):
        collection_id = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            T    = int(f["timestamp"].shape[0])

            for frame in generate_frames_from_h5_no_prev(
                f=f, T=T,
                task_name=TASK_NAME,
                collection_id=collection_id,
                use_base=False,
            ):
                ds.add_frame(frame=frame)
        ds.save_episode()

    print(f"Done. Dataset root: {out_root}")


if __name__ == "__main__":
    main()
