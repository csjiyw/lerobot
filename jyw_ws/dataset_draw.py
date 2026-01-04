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

from lerobot.policies.factory import make_policy2
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.factory import make_dataset

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action


def eval_main():


    

    logging.info("Loading dataset for evaluation.")

    dataset_id="jyw/pick_yellow_box"
    
    dataset = LeRobotDataset(repo_id=dataset_id,episodes=[0])
    
    # dataset_loader = DataLoader(
    #         dataset,
    #         batch_size=1,#cfg.eval.batch_size,
    #         shuffle=False,
    #         # num_workers=4,
    #         # pin_memory=True,
    #     )

    # test 

    


    def img_float_to_int8(x):
        y = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
        return y

    # subset = Subset(dataset, list(range(10)))

    dataset_loader = DataLoader(
        dataset,
        batch_size=1,#cfg.eval.batch_size,
        shuffle=False,
        # num_workers=4,
        # pin_memory=True,
    )


   


    # warm up 
    i=0



    sum_diff = 0.0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    num_elems = 0
    num_batches = 0


    all_states = []
    all_actions = []



    
    for batch in dataset_loader:

        observation_frame= {
            'observation.images.head': batch['observation.images.head'],
            'observation.images.hand_left':batch['observation.images.hand_left'],
            'observation.images.hand_right':batch['observation.images.hand_right'],
            'observation.state': batch['observation.state']}
        

        # from PIL import Image
        # print(batch['observation.images.head'].shape)

        # img =  batch['observation.images.head'][0].numpy().transpose(1, 2, 0)   # RGB
        # Image.fromarray((img*255).astype(np.uint8), mode="RGB").save("data/cam_front.jpg")

        # img = batch['observation.images.hand_left'][0].numpy().transpose(1, 2, 0)   # RGB
        # Image.fromarray((img*255).astype(np.uint8), mode="RGB").save("data/cam_left.jpg")

        # img = batch['observation.images.hand_right'][0].numpy().transpose(1, 2, 0)   # RGB
        # Image.fromarray((img*255).astype(np.uint8), mode="RGB").save("data/cam_right.jpg")

        # print(batch['observation.state'])
        # print("###",batch['observation.state'].cpu().numpy().reshape((2,7)))
        # observation_tensor = {k: v.to(device, non_blocking=True) for k, v in observation.items()}
        # observation_tensor["task"]="place green ball to blue plate"

        state = batch['observation.state'].cpu().numpy().squeeze(0)
        action = batch['action'].cpu().numpy().squeeze(0)

        # 统一成 (T, 14)
        if state.ndim == 1:
            state = state[None, :]
            action = action[None, :]

        all_states.append(state)
        all_actions.append(action)

    all_states = np.concatenate(all_states, axis=0)    # (Total_T, 14)
    all_actions = np.concatenate(all_actions, axis=0)  # (Total_T, 14)

    print("Total timesteps:", all_states.shape[0])

    T = all_states.shape[0]

    states = all_states.reshape(T, 2, 7)
    actions = all_actions.reshape(T, 2, 7)

    def plot_one_arm(states, actions, arm_idx, out_path):
        """
        states, actions: (T, 2, 7)
        arm_idx: 0 or 1
        """
        fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)

        for j in range(7):
            axes[j].plot(states[:, arm_idx, j], label="state", linewidth=1)
            axes[j].plot(actions[:, arm_idx, j], label="action", alpha=0.7)
            axes[j].set_ylabel(f"Dim {j}")
            axes[j].grid(True)

            if j == 0:
                axes[j].legend(loc="upper right")

        axes[-1].set_xlabel("Timestep")
        fig.suptitle(f"Arm {arm_idx}  |  State vs Action", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_path, dpi=400)
        plt.close()


    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_one_arm(
        states,
        actions,
        arm_idx=0,
        out_path=out_dir / "arm0_state_action_timeseries.png"
    )

    plot_one_arm(
        states,
        actions,
        arm_idx=1,
        out_path=out_dir / "arm1_state_action_timeseries.png"
    )

           
            
        




    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
