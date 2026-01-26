import json
import logging
import multiprocessing
from pathlib import Path
from pprint import pformat
from contextlib import nullcontext
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm


from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.factory import make_dataset






# def eval_main():
   
    
#     dataset = LeRobotDataset(repo_id="jyw/pick_yellow_box_train")

#     # print(dataset.splits)

#     print("num episodes:", dataset.num_episodes)
#     print("num samples:", len(dataset))



    
#     dataset_loader = DataLoader(
#             dataset,
#             batch_size=1,#cfg.eval.batch_size,
#             shuffle=False,
#             # num_workers=4,
#             # pin_memory=True,
#         )


     


#     logging.info("End of dataset evaluation")


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 关键：服务器无显示
import matplotlib.pyplot as plt

from collections import defaultdict
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


import numpy as np

def find_last_motion_index(
    actions: np.ndarray,
    eps: float = 1e-3,
    min_active_ratio: float = 0.2,
):
    """
    actions: (T, A)
    eps:      action magnitude threshold
    min_active_ratio:
        if a timestep has > this ratio of dims moving, treat as 'motion'
    """
    T, A = actions.shape
    for t in range(T - 1, -1, -1):
        active_dims = np.abs(actions[t]) > eps
        if active_dims.mean() >= min_active_ratio:
            return t
    return None


def trim_episode_by_action(
    states: np.ndarray,     # (T, S)
    actions: np.ndarray,    # (T, A)
    tail_keep: int = 20,
    **motion_kwargs,
):
    """
    Return trimmed (states, actions)
    """
    assert len(states) == len(actions)
    T = len(actions)

    last_motion = find_last_motion_index(actions, **motion_kwargs)

    if last_motion is None:
        # 整个 episode 都不动，直接丢弃或只保留很短
        end = min(tail_keep, T)
    else:
        end = min(last_motion + tail_keep, T)

    return states[:end], actions[:end]

def extract_state(batch, key="observation.state"):
    x = batch[key]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    return x.astype(np.float32)


def extract_episode_id(batch):
    # LeRobot 常见 episode key
    for k in [
        "episode_index",
        "episode_id",
        "episode",
        "meta.episode_index",
        "meta.episode_id",
    ]:
        if k in batch:
            v = batch[k]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return int(np.asarray(v).reshape(-1)[0])
    return 0


def eval_draw_states_save(
    repo_id="jyw/pick_yellow_box_train",
    state_key="observation.state",
    out_dir="plots_state",
):
    os.makedirs(out_dir, exist_ok=True)

    dataset = LeRobotDataset(repo_id=repo_id)
    print("num episodes:", dataset.num_episodes)
    print("num samples:", len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ep_states = defaultdict(list)
    D = None

    for batch in loader:
        s = extract_state(batch, key=state_key)
        if D is None:
            D = s.shape[0]
        ep_id = extract_episode_id(batch)
        ep_states[ep_id].append(s)

    ep_ids = sorted(ep_states.keys())
    ep_arrays = {eid: np.stack(ep_states[eid], axis=0) for eid in ep_ids}

    print(f"Saving plots to: {out_dir}")
    print(f"State dim = {D}, episodes = {len(ep_ids)}")

    

    # 每个 joint 一张图
    for j in range(D):
        plt.figure(figsize=(14, 6))

        for eid in ep_ids:
            arr = ep_arrays[eid]  # (T, D)
            y = arr[:, j]
            x = np.arange(len(y))
            plt.plot(x, y, linewidth=1, alpha=0.7)

        plt.title(f"{repo_id} | state joint {j} (all episodes)")
        plt.xlabel("t (frame index)")
        plt.ylabel("state value")
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = os.path.join(out_dir, f"joint_{j:02d}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    print("Done saving all joint plots.")


def eval_main():
    eval_draw_states_save(
        repo_id="jyw/pick_yellow_box_train",
        state_key="observation.state",
        out_dir="plots_state",
    )


if __name__ == "__main__":
    eval_main()
