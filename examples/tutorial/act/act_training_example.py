"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

torch.set_printoptions(
    edgeitems=2,     # 每个维度只打印前后各 2 个
    threshold=20,    # 超过 50 个元素就省略
    linewidth=120,
)

from torch.amp import autocast, GradScaler

scaler = GradScaler(enabled=False)  # bf16 不需要 scaler


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_id = "jyw/pick_yellow_box224_right2"
    train_folder="outputs/robot_learning_tutorial4"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    # input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # cfg = ACTConfig(input_features=input_features, output_features=output_features)
    # cfg.chunk_size=20
    # cfg.n_action_steps=20
    # policy = ACTPolicy(cfg)

    model_id="outputs/robot_learning_tutorial2/act_20000"
    policy = ACTPolicy.from_pretrained(model_id)
    cfg=policy.config

    for key in dataset_metadata.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset_metadata.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)



    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    print("dataset_metadata.stats",dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # Instantiate the dataset
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Number of training steps and logging frequency
    training_steps = 100000
    log_freq = 100

    save_freq=20000

    # Run training loop
    step = 0
    done = False

    from torch.nn.attention import sdpa_kernel




    from tqdm import tqdm
    while not done:

        with autocast("cuda", dtype=torch.bfloat16):

            # print(next(policy.parameters()).dtype)

            for batch in tqdm(dataloader):

                # print("batch raw")
                # for k,v in batch.items():
                #     if "images" in k: 
                #         continue
                #     print(k,v)



                batch_norm = preprocessor(batch)


                # print("batch normalized")
                # for k,v in batch_norm.items():
                #     if "images" in k: 
                #         continue
                #     print(k,v)


                # exit()

                from torch.nn.attention import (
                    SDPBackend,
                    sdpa_kernel
                )

                


                # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                #     loss, info = policy(batch_norm)

                loss, _ = policy.forward(batch_norm)

                # print("loss dtype:", loss.dtype)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if step % log_freq == 0:
                    print(f"step: {step} loss: {loss.item():.3f}")
                
                if step>0 and step % save_freq==0:
                    output_directory = Path(f"{train_folder}/act_{step}")
                    output_directory.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(output_directory)
                    preprocessor.save_pretrained(output_directory)
                    postprocessor.save_pretrained(output_directory)


                step += 1
                if step >= training_steps:
                    done = True
                    break

    # Save the policy checkpoint, alongside the pre/post processors
    output_directory = Path(f"{train_folder}/act_final")
    output_directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    # Save all assets to the Hub
    # policy.push_to_hub("<user>/robot_learning_tutorial_act")
    # preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    # postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")


if __name__ == "__main__":
    main()
