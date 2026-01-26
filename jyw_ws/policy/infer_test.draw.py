import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contextlib import nullcontext

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig

from matplotlib.ticker import MultipleLocator


def plot_action_vs_action0(action_gt, action0, arm_idx, out_path):
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    for j in range(7):
        axes[j].plot(action_gt[:, arm_idx, j], linewidth=1.2, label="GT action")
        axes[j].plot(action0[:, arm_idx, j], linestyle="--", alpha=0.9, label="action0")
        axes[j].set_ylabel(f"Dim {j}")
        axes[j].grid(True)
        axes[j].yaxis.set_major_locator(MultipleLocator(0.5))
        if j == 0:
            axes[j].legend(loc="upper right")
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Arm {arm_idx} | action vs action0", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    init_logging()
    device = get_safe_torch_device("cuda", log=True)
    set_seed(cfg.seed)

    dataset_repo_id = "jyw/pick_yellow_box_train"
    dataset = LeRobotDataset(repo_id=dataset_repo_id, episodes=[0])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ACTPolicy.from_pretrained(cfg.policy.pretrained_path)
    preprocess, postprocess = make_pre_post_processors(model.config, cfg.policy.pretrained_path)

    all_action_gt = []
    all_action0 = []

    with torch.inference_mode(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for batch in tqdm(loader):
            obs = preprocess(batch)
            pred = model.predict_action_chunk(obs)
            action = postprocess(pred)
            action0 = action[0, 0, :]
            action_gt = batch["action"].squeeze(0)
            all_action_gt.append(action_gt.cpu().numpy())
            all_action0.append(action0.cpu().numpy())

    all_action_gt = np.stack(all_action_gt, axis=0)
    all_action0 = np.stack(all_action0, axis=0)

    T = all_action_gt.shape[0]
    action_gt_ts = all_action_gt.reshape(T, 2, 7)
    action0_ts = all_action0.reshape(T, 2, 7)

    out_dir = Path("plots_action_vs_action0")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_action_vs_action0(action_gt_ts, action0_ts, 0, out_dir / "arm0_action_vs_action0.png")
    plot_action_vs_action0(action_gt_ts, action0_ts, 1, out_dir / "arm1_action_vs_action0.png")

    print("saved to:", out_dir.resolve())


if __name__ == "__main__":
    eval_main()
