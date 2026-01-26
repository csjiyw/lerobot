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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# 设置打印选项，仅影响打印输出

# 设置打印精度和小数点格式
np.set_printoptions(precision=3, suppress=True)

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(cfg))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    H=224
    W=224
    action_chunk=1

    

    logging.info("Loading dataset for evaluation.")

    config = {
                "state":    ((1, 7), np.float32),
                "action":    ((action_chunk, 7), np.float32),
                "cam_front": ((1,3, H, W), np.uint8),
                "cam_left": ((1,3, H, W), np.uint8),
                "cam_right": ((1,3, H, W), np.uint8),
            }

    


    # test 

    

    def writer_process(config):

        dataset = LeRobotDataset(repo_id="jyw/pick_yellow_box224_right2",episodes=[0])
    
        dataset_loader = DataLoader(
                dataset,
                batch_size=1,#cfg.eval.batch_size,
                shuffle=False,
                # num_workers=4,
                # pin_memory=True,
            )



        def img_float_to_int8(x):
            y = (x * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
            return y

        # subset = Subset(dataset, list(range(10)))

        # dataset_loader = DataLoader(
        #     subset,
        #     batch_size=1,#cfg.eval.batch_size,
        #     shuffle=False,
        #     # num_workers=4,
        #     # pin_memory=True,
        # )
        time.sleep(1)
  
        ipc1 = NetworkIPC("test", config, is_server=False)

        from tqdm import tqdm
        for batch in tqdm(dataset_loader):
            # print("client wait")
            # print("client turn")
            
            # print(f"[Reader] Received state: {state}")

            # x=batch['observation.images.head']
            # print(type(x), x.shape, x.dtype, x.device)

            

            ipc1.set("cam_front", img_float_to_int8(batch['observation.images.head']))
            ipc1.set("cam_right", img_float_to_int8(batch['observation.images.hand_right']))
            ipc1.set("state", batch['observation.state'].numpy())
            ipc1.set("action", batch['action'].numpy())

            ipc1.talk()

            action = ipc1.get("action")

            # action_gt=batch['action']
            # diff= action_gt-action
            # print("diff",torch.mean(diff))
            # print("gt",action_gt)
            # print("pred",action)
            # print("client turn finish")

            # time.sleep(0.5)
        ipc1.close()
    
    writer = multiprocessing.Process(target=writer_process, args=(config,))
    writer.start()
   

    logging.info("Making policy.")

    model_id=cfg.policy.pretrained_path
    model = ACTPolicy.from_pretrained(model_id)

    # dataset_id = "jyw/pick_yellow_box224"
    # # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    # dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    # preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

    preprocess, postprocess = make_pre_post_processors(model.config, model_id )

    # for p in preprocess.steps:
    #     print(p)

    # exit(0)

    print("preprocess")
    for p in preprocess.steps:
        print(p)

    print("postprocess")
    for p in postprocess.steps:
        print(p)




    import cv2

    # HOST = "0.0.0.0"  # set to server machine IP for cross-PC
    # PORT = 7890

    # ipc = NetworkIPC("test", config, is_server=True,host=HOST,port=PORT)
    ipc = NetworkIPC("test", config, is_server=True)


    # warm up 
    warm_up_count=60
    i=0

    print("device",device)

    # init_act=np.array([[ 0.1768, -0.1400, -0.1749, -1.6732, -0.0633,  1.5029,  0.7946,  0.0400]])

    gt_states=[]
    pred_states=[]
    actions=[]
    last_action=None


    try:
    
        while True:
            # print("server wait")
            ipc.talk()

            act=ipc.get("action")

            # print("server turn")
            
            cam_front = ipc.get("cam_front")
            cam_left = ipc.get("cam_left")
            cam_right=ipc.get("cam_right")
            state=ipc.get("state")

            k=0.0

            if last_action is None:
                obs_state=state
            else:
                obs_state=(state*(1-k)+k*last_action)

            obs= {
                'observation.images.head': cam_front.astype(np.float32) / 255.0,
                'observation.images.hand_right':cam_right.astype(np.float32) / 255.0,
                'observation.state': obs_state,
                'action':act
            }
            
            obs = {k: torch.from_numpy(v) for k, v in obs.items()}


            # print("state",state.reshape((2,7)))
            # exit(0)


            # print("obs",observation['observation.state'].device)

            # print("batch raw")
            # for k,v in obs.items():
            #     if "images" in k: 
            #         continue
            #     print(k,v)



            batch_norm = preprocess(obs)


            # print("batch normalized")
            # for k,v in batch_norm.items():
            #     if "images" in k: 
            #         continue
            #     print(k,v)

            

            # use chunk
            pred = model.predict_action_chunk(batch_norm)

            # print("pred raw",pred)
            # for k,v in pred.items():
            #     if "images" in k: 
            #         continue
            #     print(k,v)

            # action= pred[:,0,:]
            action_trunk = postprocess(pred)


            # print("pred normalized",action_trunk)
            # for k,v in action_trunk.items():
            #     if "images" in k: 
            #         continue
            #     print(k,v)
            
            # exit()


            # ipc.set("action",action_trunk.reshape((action_chunk,7)))

            last_action=action_trunk[0,0,:].numpy().reshape((1,7))



            # print("last-action",last_action)

            # use single action
            # action = model.select_action(obs)
            # action = postprocess(action)
            # ipc.set("action",action.reshape((1,14)))

            # last_action=last_action+state

            gt_states.append(state.reshape((1,7)))
            pred_states.append(act.reshape((1,7)))
            actions.append(last_action.reshape((1,7)))
            
            i+=1

            # print("<-action",pred)

            # print("server turn finish")
    except KeyboardInterrupt:
        print("Ctrl+C pressed, exiting cleanly")

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from pathlib import Path

        gt_states = np.stack(gt_states, axis=0)    # (T,2,7)
        pred_states = np.stack(pred_states, axis=0)    # (T,2,7)
        actions = np.stack(actions, axis=0)  # (T,2,7)


        T = gt_states.shape[0]
        print("Collected timesteps:", T)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = Path(f"plots_state_action_relation/serverts{timestamp}")
        out_dir.mkdir(parents=True,exist_ok=True)

        def plot_state_action(states,pred_states, actions, arm_idx, name):
            fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)

            for j in range(7):
                ax = axes[j]
                ax.plot(states[:, arm_idx, j], label="gt_state", linewidth=1.2)
                ax.plot(pred_states[:, arm_idx, j], label="gt_action", linewidth=1.2)
                ax.plot(actions[:, arm_idx, j], label="pred_action", linestyle="--", alpha=0.9)
                ax.set_ylabel(f"Dim {j}")
                ax.grid(True)
                # ax.yaxis.set_major_locator(MultipleLocator(0.3))
                if j == 0:
                    ax.legend(loc="upper right")

            axes[-1].set_xlabel("Timestep")
            fig.suptitle(f"{name} | Arm {arm_idx} | state vs action", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(out_dir / f"{name}_arm{arm_idx}.png", dpi=300)
            plt.close(fig)

        plot_state_action(gt_states,pred_states, actions, arm_idx=0, name="relation")

        print("plots saved to:", out_dir.resolve())
    finally:

        ipc.close()


    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
