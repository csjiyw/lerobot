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

from network_ipc_v2 import NetworkIPC




def eval_main():
   
    
    dataset = LeRobotDataset(repo_id="jyw/pick_yellow_box_train")

    # print(dataset.splits)

    print("num episodes:", dataset.num_episodes)
    print("num samples:", len(dataset))



    
    dataset_loader = DataLoader(
            dataset,
            batch_size=1,#cfg.eval.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
        )


     


    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
