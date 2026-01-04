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

from network_ipc_v2 import NetworkIPC

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

# 设置打印选项，仅影响打印输出

# 设置打印精度和小数点格式
np.set_printoptions(precision=3, suppress=True)

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(cfg))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    

    logging.info("Loading dataset for evaluation.")

    config = {
                "state":    ((1, 14), np.float32),
                "action":    ((100, 14), np.float32),
                "cam_front": ((1,3, 480, 640), np.uint8),
                "cam_left": ((1,3, 480, 640), np.uint8),
                "cam_right": ((1,3, 480, 640), np.uint8),
            }

    
    dataset = LeRobotDataset(repo_id="jyw/pick_yellow_box_test",episodes=[0])
    
    dataset_loader = DataLoader(
            dataset,
            batch_size=1,#cfg.eval.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
        )
   

    logging.info("Making policy.")

    model_id=cfg.policy.pretrained_path
    model = ACTPolicy.from_pretrained(model_id)
    preprocess, postprocess = make_pre_post_processors(model.config, model_id )

    import cv2

    HOST = "0.0.0.0"  # set to server machine IP for cross-PC
    PORT = 7890

    ipc = NetworkIPC("test", config, is_server=True,host=HOST,port=PORT)


    # warm up 
    warm_up_count=60
    i=0

    print("device",device)

    # init_act=np.array([[ 0.1768, -0.1400, -0.1749, -1.6732, -0.0633,  1.5029,  0.7946,  0.0400]])

    from tqdm import tqdm

    with torch.inference_mode(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        
        for batch in tqdm(dataset_loader):
            # print("server wait")
            ipc.talk()

            act=ipc.get("action")
            if act[0,0]==-999:
                model.reset()
                ipc.set("action",act*0)
                i=0
                print("reset")
                continue

            # print("server turn")


            # print("obs",observation['observation.state'].device)
 
            obs = preprocess(batch)

            # use chunk
            action = model.predict_action_chunk(obs)
            # action= pred[:,0,:]
            action = postprocess(action)
            ipc.set("action",action.reshape((100,14)))

            last_action=action[0,99,:].reshape((2,7))
            print("last-action",last_action)

            # use single action
            # action = model.select_action(obs)
            # action = postprocess(action)
            # ipc.set("action",action.reshape((1,14)))
            
            i+=1

            # print("<-action",pred)

            # print("server turn finish")

    ipc.close()


    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
