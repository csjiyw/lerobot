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

from network_ipc_v2 import NetworkIPC

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy



def eval_main():

    logging.info("Loading dataset for evaluation.")

    
    dataset = LeRobotDataset(repo_id="jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14",episodes=[0])
    
    dataset_loader = DataLoader(
            dataset,
            batch_size=1,#cfg.eval.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
        )

    # test 

    

   

    # warm up 
    warm_up_count=60
    i=0


    for batch in tqdm(dataset_loader):
        # print("server wait")
        
        # print("server turn")

        from PIL import Image


        cam_front = batch['observation.images.head']*255


        img = cam_front[0].numpy().astype(np.uint8)
        
        img=img.transpose(1, 2, 0)   # RGB

        import cv2

        Image.fromarray(img, mode="RGB").save(f"outputs/data/cam_front_gt_rgb_{i}.jpg")

        # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        # Image.fromarray(img, mode="RGB").save("cam_front_gt_bgr.jpg")
        

        # print("pred",pred.shape)
        state=batch['observation.state']

        
        i+=1


        # print("server turn finish")


    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
