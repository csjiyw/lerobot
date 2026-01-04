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

    
    dataset = LeRobotDataset(repo_id="jyw/place_metallic_object_fast",episodes=[0])
    
    dataset_loader = DataLoader(
            dataset,
            batch_size=1,#cfg.eval.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
        )

    # test 

    

    def writer_process(config):

        def img_float_to_int8(x):
            y = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
            return y

        subset = Subset(dataset, list(range(10)))

        dataset_loader = DataLoader(
            subset,
            batch_size=1,#cfg.eval.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
        )
        time.sleep(1)
  
        ipc1 = NetworkIPC("test", config, is_server=False)
        for batch in dataset_loader:
            print("client wait")
            print("client turn")
            
            # print(f"[Reader] Received state: {state}")

            x=batch['observation.images.head']
            print(type(x), x.shape, x.dtype, x.device)

            

            ipc1.set("cam_front", img_float_to_int8(batch['observation.images.head']))
            ipc1.set("cam_left", img_float_to_int8(batch['observation.images.hand_left']))
            ipc1.set("cam_right", img_float_to_int8(batch['observation.images.hand_right']))
            ipc1.set("state", batch['observation.state'])

            ipc1.talk()

            action = ipc1.get("action")

            action_gt=batch['action']
            diff= action_gt-action
            print("diff",torch.mean(diff))
            print("gt",action_gt)
            print("pred",action)
            print("client turn finish")

            # time.sleep(0.5)
        ipc1.close()
    
    writer = multiprocessing.Process(target=writer_process, args=(config,))
    # writer.start()


    ds_meta = dataset.meta
   

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

    with torch.inference_mode(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        
        while True:
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
            
            cam_front = ipc.get("cam_front")
            cam_left = ipc.get("cam_left")
            cam_right=ipc.get("cam_right")
            state=ipc.get("state")


          


            obs= {'observation.images.head': cam_front.astype(np.float32) / 255.0,
                        'observation.images.hand_left':cam_left.astype(np.float32) / 255.0,
                        'observation.images.hand_right':cam_right.astype(np.float32) / 255.0,
                        'observation.state': state
                        }
            
            obs = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in obs.items()}
            # observation_tensor["task"]="place green ball to blue plate"
            

            # from PIL import Image

            # img = cam_front[0].transpose(1, 2, 0)   # RGB
            # Image.fromarray(img, mode="RGB").save("data/cam_front.jpg")

            # img = cam_left[0].transpose(1, 2, 0)   # RGB
            # Image.fromarray(img, mode="RGB").save("data/cam_left.jpg")

            # img = cam_right[0].transpose(1, 2, 0)   # RGB
            # Image.fromarray(img, mode="RGB").save("data/cam_right.jpg")

            print("state",state.reshape((2,7)))
            # exit(0)


            # print("obs",observation['observation.state'].device)
 
            obs = preprocess(obs)

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
