import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def plot_state_timeseries_with_reference(
    states: np.ndarray,      # (T,2,7)
    ref_state: np.ndarray,   # (2,7)
    arm_idx: int,
    out_path: Path,
):
    """
    在时间线上画 dataset state，并叠加一条 ref_state 的水平线
    """
    T = states.shape[0]
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)

    for j in range(7):
        # dataset 时间线
        axes[j].plot(
            states[:, arm_idx, j],
            label="dataset state",
            linewidth=1.2,
        )

        # 你的 state：水平参考线
        axes[j].axhline(
            y=ref_state[arm_idx, j],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="given state" if j == 0 else None,
        )

        axes[j].set_ylabel(f"Dim {j}")
        axes[j].grid(True)

        if j == 0:
            axes[j].legend(loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Arm {arm_idx} | Dataset State Timeline + Given State", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    init_logging()
    logging.info("Loading dataset...")

    # ===== 配置 =====
    dataset_id = "jyw/pick_yellow_box"
    episodes = [0]
    batch_size = 1
    out_dir = Path("plots_state_timeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 你提供的 state (2,7)
    given_state = np.array([
        [ 0.052,  0.052,  0.005,  0.016, -0.028, -0.127,  0.005],
        [ 0.056,  1.048, -0.496,  0.973,  0.178, -0.965,  0.047],
    ], dtype=np.float32)
    # =================

    dataset = LeRobotDataset(repo_id=dataset_id, episodes=episodes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_states = []

    for batch in tqdm(loader, desc="Iter dataset"):
        state = batch["observation.state"].cpu().numpy()

        # 去 batch 维
        if state.shape[0] == 1:
            state = state.squeeze(0)

        # 统一成 (T,14)
        if state.ndim == 1:
            state = state[None, :]

        all_states.append(state)

    all_states = np.concatenate(all_states, axis=0)  # (T,14)
    T = all_states.shape[0]
    logging.info(f"Total timesteps: {T}")

    # reshape -> (T,2,7)
    states = all_states.reshape(T, 2, 7)

    # 画时间线（arm0 / arm1）
    plot_state_timeseries_with_reference(
        states,
        given_state,
        arm_idx=0,
        out_path=out_dir / "arm0_state_timeline_with_given.png",
    )

    plot_state_timeseries_with_reference(
        states,
        given_state,
        arm_idx=1,
        out_path=out_dir / "arm1_state_timeline_with_given.png",
    )

    logging.info("Done.")
    print("Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
