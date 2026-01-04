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
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.factory import make_dataset

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    # logging.info(pformat(cfg))

    device = get_safe_torch_device("cuda", log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    

    logging.info("Loading dataset for evaluation.")

    dataset_repo_id="jyw/pick_yellow_box_test"
    dataset_repo_id="jyw/pick_yellow_box_train"

 

    
    dataset = LeRobotDataset(repo_id=dataset_repo_id,episodes=[0])
    # dataset = make_dataset(cfg)
    
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

    
    model_id=cfg.policy.pretrained_path
    model = ACTPolicy.from_pretrained(model_id)


    preprocess, postprocess = make_pre_post_processors(model.config,
                                                       model_id
                                                        # dataset_stats=dataset_metadata.stats
                                                        )
    
    # for p in preprocess.steps:
    #     print(p)

    # exit(0)

    # for i, step in enumerate(preprocess.steps):
    #     print(f"[preprocess step {i}]")
    #     print("  class :", step.__class__.__name__)
    #     print("  repr  :", step)

    # for i, step in enumerate(postprocess.steps):
    #     print(f"[postprocess step {i}]")
    #     print("  class :", step.__class__.__name__)
    #     print("  repr  :", step)


   


    # warm up 
    i=0

    print("device",device)

    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    num_elems = 0
    num_batches = 0


    

    with torch.inference_mode(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        
        for batch in dataset_loader:

            # observation_frame= {
            #     'observation.images.head': batch['observation.images.head'],
            #     'observation.images.hand_left':batch['observation.images.hand_left'],
            #     'observation.images.hand_right':batch['observation.images.hand_right'],
            #     'observation.state': batch['observation.state']}

            observation_frame=batch
            

            # from PIL import Image
            # print(batch['observation.images.head'].shape)

            # img =  batch['observation.images.head'][0].numpy().transpose(1, 2, 0)   # RGB
            # Image.fromarray((img*255).astype(np.uint8), mode="RGB").save("data/cam_front.jpg")

            # img = batch['observation.images.hand_left'][0].numpy().transpose(1, 2, 0)   # RGB
            # Image.fromarray((img*255).astype(np.uint8), mode="RGB").save("data/cam_left.jpg")

            # img = batch['observation.images.hand_right'][0].numpy().transpose(1, 2, 0)   # RGB
            # Image.fromarray((img*255).astype(np.uint8), mode="RGB").save("data/cam_right.jpg")

            # print(batch['observation.state'])
            # print("state",batch['observation.state'].cpu().numpy().reshape((2,7)))
            # observation_tensor = {k: v.to(device, non_blocking=True) for k, v in observation.items()}
            # observation_tensor["task"]="place green ball to blue plate"
            # exit()

            # print("observation_frame",observation_frame['observation.state'].device)

            # print(observation_frame['observation.images.head'])



            obs = preprocess(observation_frame)

            # print(obs['observation.images.head'])

            # print("obs",obs['observation.state'].device)

            # exit(0)

    
            pred = model.predict_action_chunk(obs)#.cpu()  # shape (B, D)


            

            # print("pred",pred.shape)

                
            # action= pred[:,0,:]

            # action = policy.select_action(obs)
            action = postprocess(pred)

            # last_action=action[0,99,:].reshape((2,7))
            # print("last-action",last_action)


            action0=action[0,0,:]

            # action = make_robot_action(action, dataset_metadata.features)  # add name for each value, return a dict



            action_gt=batch['action']
            # diff= action_gt+action

            # print("diff",torch.mean(diff))
            # print("gt",action_gt)
            # print("pred",action)
            # print("client turn finish")
            
            i+=1

            diff = action0 - action_gt        # ⚠️ 通常是 pred - gt（不是加）

            sum_abs_diff += diff.abs().sum().item()
            sum_sq_diff += (diff ** 2).sum().item()

            num_elems += diff.numel()
            num_batches += 1

    # === 全局统计 ===
    mae = sum_abs_diff / num_elems
    rmse = (sum_sq_diff / num_elems) ** 0.5

    print("====== Dataset statistics ======")
    print(f"num_batches: {num_batches}")
    print(f"MAE:         {mae:.6f}")
    print(f"RMSE:        {rmse:.6f}")



    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
