#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import io
import shutil
import h5py
import time
from contextlib import contextmanager

from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ===================== 时间打点工具 =====================
@contextmanager
def log_time(tag: str):
    t0 = time.perf_counter()
    print(f"[START] {tag}", flush=True)
    yield
    dt = time.perf_counter() - t0
    print(f"[END]   {tag}: {dt:.3f}s", flush=True)
# ======================================================


# ===================== 用户配置 =====================
data_home = Path("data/rclone-v1.68.2-linux-amd64/data/9f22b6b23ed64ced8fc8074d454e5672")

REPO_ID = "jyw/your_dataset_name"
ROBOT_TYPE = "generic"
FPS = 30
TASK_NAME = "your_task_name"
WIPE_OUTPUT = True

USE_HEAD_CAM = True
USE_HAND_LEFT_CAM = True
USE_HAND_RIGHT_CAM = True

USE_ARM_VEL = False
USE_EFFECTOR_VEL = False
USE_BASE_VEL_ANG = True
# ====================================================


# --------- H5 keys ----------
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
    "a_eff_p": "joints/action/effector/position",
    "a_base_v": "joints/action/robot/velocity",
    "a_base_w": "joints/action/robot/angular",
}
# --------------------------------


def build_state_action(f: h5py.File, t: int):
    parts_s = [np.asarray(f[K["s_arm_p"]][t], dtype=np.float32).reshape(-1)]
    parts_s.append(np.asarray(f[K["s_eff_p"]][t], dtype=np.float32).reshape(-1))

    if USE_BASE_VEL_ANG:
        parts_s.append(np.asarray(f[K["s_base_v"]][t], dtype=np.float32).reshape(-1))
        parts_s.append(np.asarray(f[K["s_base_w"]][t], dtype=np.float32).reshape(-1))

    state = np.concatenate(parts_s, axis=0)

    parts_a = [
        np.asarray(f[K["a_arm_p"]][t], dtype=np.float32).reshape(-1),
        np.asarray(f[K["a_eff_p"]][t], dtype=np.float32).reshape(-1),
        np.asarray(f[K["a_base_v"]][t], dtype=np.float32).reshape(-1),
        np.asarray(f[K["a_base_w"]][t], dtype=np.float32).reshape(-1),
    ]
    action = np.concatenate(parts_a, axis=0)

    return state, action


def build_feature_spec(sample_h5: Path):
    with log_time("build_feature_spec"):
        with h5py.File(sample_h5, "r") as f:
            s0, a0 = build_state_action(f, 0)
            state_dim = int(s0.shape[0])
            action_dim = int(a0.shape[0])

        feats = {}

        if USE_HEAD_CAM:
            feats["observation.images.head"] = {
                "dtype": "video",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }

        if USE_HAND_LEFT_CAM:
            feats["observation.images.hand_left"] = feats["observation.images.head"]

        if USE_HAND_RIGHT_CAM:
            feats["observation.images.hand_right"] = feats["observation.images.head"]

        feats["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
        }
        feats["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
        }

        return feats


def generate_frames_from_h5_no_prev(f, T, task_name: str, use_base=True):
    d_head = f.get(K["head_img"])
    d_hl   = f.get(K["hand_left_img"])
    d_hr   = f.get(K["hand_right_img"])

    for t in range(T):
        frame = {"task": task_name}

        if d_head is not None:
            frame["observation.images.head"] = d_head[t]
        if d_hl is not None:
            frame["observation.images.hand_left"] = d_hl[t]
        if d_hr is not None:
            frame["observation.images.hand_right"] = d_hr[t]

        obs = [
            np.asarray(f[K["s_arm_p"]][t], dtype=np.float32).reshape(-1),
            np.asarray(f[K["s_eff_p"]][t], dtype=np.float32).reshape(-1),
        ]
        act = [
            np.asarray(f[K["a_arm_p"]][t], dtype=np.float32).reshape(-1),
            np.asarray(f[K["a_eff_p"]][t], dtype=np.float32).reshape(-1),
        ]

        if use_base:
            obs += [
                np.asarray(f[K["s_base_v"]][t], dtype=np.float32).reshape(-1),
                np.asarray(f[K["s_base_w"]][t], dtype=np.float32).reshape(-1),
            ]
            act += [
                np.asarray(f[K["a_base_v"]][t], dtype=np.float32).reshape(-1),
                np.asarray(f[K["a_base_w"]][t], dtype=np.float32).reshape(-1),
            ]

        frame["observation.state"] = np.concatenate(obs).astype(np.float32)
        frame["action"] = np.concatenate(act).astype(np.float32)

        yield frame


def main():
    assert data_home.exists(), f"data_home not found: {data_home}"

    out_root = Path(HF_LEROBOT_HOME) / REPO_ID
    if WIPE_OUTPUT and out_root.exists():
        shutil.rmtree(out_root)

    h5_files = []
    for p in sorted(data_home.iterdir()):
        fp = p / f"{p.name}.h5"
        if fp.exists():
            h5_files.append(fp)

    h5_files = h5_files[:5]
    assert h5_files, "No h5 files found"

    features = build_feature_spec(h5_files[0])

    with log_time("LeRobotDataset.create"):
        ds = LeRobotDataset.create(
            repo_id=REPO_ID,
            robot_type=ROBOT_TYPE,
            fps=FPS,
            features=features,
            image_writer_threads=8,
            image_writer_processes=4,
        )

    for h5_path in tqdm(h5_files, desc="H5 → LeRobot"):
        with h5py.File(h5_path, "r") as f:
            T = int(f["timestamp"].shape[0])

            first = True
            for frame in generate_frames_from_h5_no_prev(
                f=f, T=T, task_name=TASK_NAME
            ):
                if first:
                    with log_time("FIRST ds.add_frame"):
                        ds.add_frame(frame)
                    first = False
                else:
                    ds.add_frame(frame)

    with log_time("ds.save_episode"):
        ds.save_episode()

    print(f"Done. Dataset root: {out_root}")


if __name__ == "__main__":
    main()
