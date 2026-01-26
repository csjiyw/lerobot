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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# 设置打印选项，仅影响打印输出

# 设置打印精度和小数点格式
np.set_printoptions(precision=3, suppress=True)

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(cfg))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    set_seed(cfg.seed)

    H=224
    W=224
    action_chunk=1

    

    logging.info("Loading dataset for evaluation.")



    


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
            y = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
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
            ipc1.set("state", batch['observation.state'])
            ipc1.set("action", batch['action'])

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
    
    # writer = multiprocessing.Process(target=writer_process, args=(config,))
    # writer.start()
   

    logging.info("Making policy.")

    model_id=cfg.policy.pretrained_path
    model = ACTPolicy.from_pretrained(model_id)

    # dataset_id = "jyw/pick_yellow_box224"
    # # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    # dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    # preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

    preprocess, postprocess = make_pre_post_processors(model.config, model_id )

    print("preprocess")
    for p in preprocess.steps:
        print(p)

    print("postprocess")
    for p in postprocess.steps:
        print(p)


    # exit(0)

    import cv2


    i=0

    print("device",device)


    states=[]
    pred_actions=[]
    actions=[]


    dataset = LeRobotDataset(repo_id="jyw/task_pick_place_yellow_capsule_to_white_plate_right_arm_cfg_cam2_img224_state7dof_delta_action_trim_head_tail",episodes=[0])
    
    dataset_loader = DataLoader(
            dataset,
            batch_size=1,#cfg.eval.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
        )


    last_action=None

    for batch in tqdm(dataset_loader):
        
        
        # obs = {k: torch.from_numpy(v) for k, v in obs.items()}


        # print("state",state.reshape((2,7)))
        # exit(0)
        state=batch['observation.state'].numpy()
        action=batch['action'].numpy()

        if last_action is not None:
            batch['observation.state'][0,:]=torch.from_numpy(last_action)

        # del batch['action']

        # print("obs",observation['observation.state'].device)

        # print("batch raw")
        # for k,v in batch.items():
        #     if "images" in k: 
        #         continue
        #     print(k,v)



        batch_norm = preprocess(batch)


        # print("batch normalized")
        # for k,v in batch_norm.items():
        #     if "images" in k: 
        #         continue
        #     print(k,v)

        

        # use chunk
        pred = model.predict_action_chunk(batch_norm)
        # action= pred[:,0,:]
        action_trunk = postprocess(pred)


        # print("pred raw",pred)
        # print("pred normalized",action_trunk)

        # exit()


        # ipc.set("action",action_trunk.reshape((action_chunk,7)))

        last_action=action_trunk[0,0,:].numpy()
  
        states.append(state.reshape((1,7)))
        pred_actions.append(last_action.reshape((1,7)))
        actions.append(action.reshape((1,7)))
        

   

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from pathlib import Path

    states = np.stack(states, axis=0)    # (T,2,7)
    pred_actions = np.stack(pred_actions, axis=0)    # (T,2,7)
    actions = np.stack(actions, axis=0)  # (T,2,7)


    T = states.shape[0]
    print("Collected timesteps:", T)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(f"plots_state_action_relation/testts{timestamp}")
    out_dir.mkdir(parents=True,exist_ok=True)

    def plot_state_action(states,preds, actions, arm_idx, name):
        fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)

        for j in range(7):
            ax = axes[j]
            # ax.plot(states[:, arm_idx, j], label="state", linewidth=1.2)
            ax.plot(preds[:, arm_idx, j], label="pred", linewidth=1.2)
            ax.plot(actions[:, arm_idx, j], label="gt_action", linestyle="--", alpha=0.9)
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

    plot_state_action(states,pred_actions, actions, arm_idx=0, name="relation")

    print("plots saved to:", out_dir.resolve())



    logging.info("End of dataset evaluation")


if __name__ == "__main__":
    init_logging()

    eval_main()
