# system env
## change container time zone
apt install -y tzdata

## install
pip install pysocks



ssh -R 0.0.0.0:17890:127.0.0.1:7890 -p 2222 root@172.16.128.4

ssh -R 0.0.0.0:17897:127.0.0.1:7897 -p 2222 root@172.16.128.4

export ALL_PROXY=socks5h://127.0.0.1:17897


export http_proxy=http://192.168.25.160:7897;
export https_proxy=http://192.168.25.160:7897

uv pip install "torch==2.7" "torchvision<0.23.0" "torchcodec<0.6.0"  
 --index-url https://download.pytorch.org/whl/cu126


uv pip install -e ".[pi]"   --index-url https://download.pytorch.org/whl/cu126

***too low version will cause error***

# wandb use help
```shell
WANDB_API_BASE=http://172.16.128.4:8280/api \
  WANDB_BASE_URL=http://172.16.128.4:8280  \
  WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
  TORCH_FORCE_SDPA=1 TORCH_CUDNN_SDPA_ENABLED=1 \
  HF_HUB_OFFLINE=1  \
  accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14 \
  --policy.push_to_hub=false  \
  --policy.path=lerobot/smolvla_base \
  --batch_size=8 \
  --job_name=smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14 \
  --output_dir=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14 \
  --wandb.enable=true \
  --wandb.project=jyw_act_project  \
  --wandb.entity=bigmodel  \
  --wandb.disable_artifact=true
```

# act
## train




```
lerobot-train \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w224h224_abs14_trim_tail_test_merged \
  --policy.type=act \
  --output_dir=outputs/train/act_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w224h224_abs14_trim_tail_test_merged \
  --wandb.enable=false \
  --policy.repo_id=jyw/act_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w224h224_abs14_trim_tail_test_merged



lerobot-train \
  --dataset.repo_id=jyw/place_metallic_object_fast \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_mobile_place_metallic_object_fast_test \
  --job_name=act_aloha_mobile_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jyw/place_metallic_object_fast_act

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/your_dataset_name  \
  --policy.type=act \
  --policy.push_to_hub=false  \
  --output_dir=outputs/train/act_multi_gpu \
  --job_name=act_multi_gpu \
  --wandb.enable=false  \
  --policy.repo_id=jyw/your_dataset_name_act_multi_gpu


python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"

lerobot-train \
  --dataset.repo_id=jyw/pick_yellow_box_train \
  --policy.type=act \
  --policy.push_to_hub=false  \
  --output_dir=outputs/train/act_aloha_mobile_pick_yellow_box_train_split \
  --job_name=act_aloha_mobile_pick_yellow_box \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jyw/act_aloha_mobile_pick_yellow_box


tmux capture-pane -t lerobot -p -S - > lerobot_train.log


lerobot-train \
  --dataset.repo_id=jyw/place_metallic_object \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_mobile_place_metallic_object_fast_test \
  --job_name=act_aloha_mobile_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jyw/place_metallic_object_fast_act


lerobot-train \
  --config_path="outputs/train/act_aloha_mobile_place_metallic_object_fast_test/checkpoints/last/pretrained_model/train_config.json" \
  --wandb.enable=false \
  --resume true \
  --batch_size=48  \
  --steps=500000 



lerobot-train \
  --dataset.repo_id=jyw/pick_yellow_box \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_mobile_pick_yellow_box \
  --job_name=act_aloha_mobile_pick_yellow_box \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jyw/act_aloha_mobile_pick_yellow_box  \
  --steps=200000


WANDB_API_BASE=http://172.16.128.4:8280/api \
WANDB_BASE_URL=http://172.16.128.4:8280  \
WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
lerobot-train \
    --dataset.repo_id=jyw/pick_yellow_box224 \
    --policy.type=act \
    --job_name=act_aloha_mobile_pick_yellow_box \
    --output_dir=outputs/train/act_aloha_mobile_pick_yellow_box_train \
    --wandb.enable=true \
    --wandb.project=jyw_act_project  \
    --wandb.entity=bigmodel  \
    --wandb.disable_artifact=true  \
    --batch_size=4 \
    --steps=1000000


CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --dataset.repo_id=jyw/pick_yellow_box224 \
  --policy.type=act \
  --job_name=act_aloha_mobile_pick_yellow_box \
  --output_dir=outputs/train/act_aloha_mobile_pick_yellow_box_train2 \
  --wandb.enable=false  \
  --dataset.remapping={}

```


## infer

python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_place_metallic_object_fast_test/checkpoints/last/pretrained_model"

python jyw_ws/infer_server.py \
    --policy.path="/workspace/lerobot/outputs/train/act_multi_gpu/checkpoints/last/pretrained_model"


python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"

CUDA_VISIBLE_DEVICES=1 python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_pick_yellow_box_train/checkpoints/last/pretrained_model"

python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_pick_yellow_box_train/checkpoints/0020000/pretrained_model"

CUDA_VISIBLE_DEVICES=1 python jyw_ws/infer_server.py \
  --policy.path="outputs/robot_learning_tutorial/act_60000"

CUDA_VISIBLE_DEVICES=0 python jyw_ws/infer_server.py \
  --policy.path="outputs/robot_learning_tutorial/act_final"

CUDA_VISIBLE_DEVICES=1 python jyw_ws/infer_server.py \
    --policy.path="outputs/robot_learning_tutorial/act"

CUDA_VISIBLE_DEVICES=1 python examples/tutorial/act/act_training_example.copy.py
CUDA_VISIBLE_DEVICES=1 python examples/tutorial/act/act_training_example.py

CUDA_VISIBLE_DEVICES=0 python jyw_ws/infer_server.py \
  --policy.path="outputs/robot_learning_tutorial2/act_40000"

CUDA_VISIBLE_DEVICES=0 python jyw_ws/infer_server.delta7.py \
  --policy.path="outputs/robot_learning_tutorial2_delta_action/act_80000"

## test

CUDA_VISIBLE_DEVICES=1 python jyw_ws/infer_test.py \
  --policy.path="outputs/train/act_aloha_mobile_pick_yellow_box_train/checkpoints/last/pretrained_model"

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_test.draw2.py \
  --policy.path=/workspace/lerobot/outputs/robot_learning_tutorial2/act_20000

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_test.draw2.py \
  --policy.path=outputs/train/act_aloha_mobile_place_metallic_object_fast_test/checkpoints/020000/pretrained_model

python jyw_ws/infer_server.test_data_rollout.py \
    --policy.path="/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"


python jyw_ws/infer_test.draw.py \
    --policy.path="/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"

python jyw_ws/infer_test.py \
  --policy.path="/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"  

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_test.py \
  --policy.path=outputs/train/act_aloha_mobile_place_metallic_object_fast_test/checkpoints/last/pretrained_model



CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.draw.py \
  --policy.path=/workspace/lerobot/outputs/robot_learning_tutorial2/act_20000

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.py \
  --policy.path=/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box_train2/checkpoints/last/pretrained_model


CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.py \
  --policy.path=/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box_train2/checkpoints/last/pretrained_model

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.act.abs7.py \
  --policy.path=outputs/train/act_chunk30_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300/checkpoints/last/pretrained_model

HF_HUB_OFFLINE=1  CUDA_VISIBLE_DEVICES=3  \
  python jyw_ws/infer_server.smolvla.abs7.py \
  --policy.path=outputs/train/smolvla_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300/checkpoints/last/pretrained_model


python jyw_ws/infer_test.act.action_state_delta.draw.py \
  --policy.path=outputs/train/act_chunk20_agx_task_pick_place_yellow_capsule_to_white_plate_right_arm_cfg_cam2_img224_state7dof_delta_action_trim_head_tail/checkpoints/last/pretrained_model


CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.py   --policy.path=/workspace/lerobot/outputs/robot_learning_tutorial3/act_final


# smolval
## switch smolval version
```bash
ln -sf /root/workspace/lerobot/src/lerobot/policies/smolvla.new  \
    /root/workspace/lerobot/src/lerobot/policies/smolvla

ln -sf /root/workspace/lerobot/src/lerobot/policies/smolvla.old  \
  /root/workspace/lerobot/src/lerobot/policies/smolvla
```

## train
```
HF_ENDPOINT=https://hf-mirror.com 

HF_HUB_OFFLINE=1  \
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false  \
  --dataset.repo_id=jyw/pick_yellow_box \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training  \
  --wandb.enable=false




HF_HUB_OFFLINE=1  \
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false  \
  --dataset.repo_id=jyw/pick_yellow_box224_right2 \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla2 \
  --job_name=my_smolvla_training  \
  --wandb.enable=false 

HF_HUB_OFFLINE=1  \
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false  \
  --dataset.repo_id=jyw/pick_yellow_box224_right2 \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla2 \
  --job_name=my_smolvla_training  \
  --wandb.enable=false

CUDA_VISIBLE_DEVICES=1  \
WANDB_API_BASE=http://172.16.128.4:8280/api \
WANDB_BASE_URL=http://172.16.128.4:8280  \
WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
lerobot-train \
    --dataset.repo_id=jyw/task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --policy.push_to_hub=false  \
    --policy.type=act \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --batch_size=32 \
    --steps=200000 \
    --job_name=act_chunk30_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --output_dir=outputs/train/act_chunk30_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --wandb.enable=true \
    --wandb.project=jyw_act_project  \
    --wandb.entity=bigmodel  \
    --wandb.disable_artifact=true


lerobot-train \
  --policy.type=act \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --batch_size=1 \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_test_merged \
  --policy.push_to_hub=false  \
  --job_name=act_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped_merged_test \
  --output_dir=outputs/train/act_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped_merged_test \
  --wandb.enable=false



--dynamo_mode=max-autotune
  --dynamo_backend=inductor \
accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=inductor \
  $(which lerobot-train) \
    --dataset.repo_id=jyw/task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --policy.push_to_hub=false  \
    --policy.type=act \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --job_name=act_chunk30_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --output_dir=outputs/train/act_test \
    --wandb.enable=false

HF_HUB_OFFLINE=1  \
CUDA_VISIBLE_DEVICES=1  \
WANDB_API_BASE=http://172.16.128.4:8280/api \
WANDB_BASE_URL=http://172.16.128.4:8280  \
WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
lerobot-train \
    --dataset.repo_id=jyw/task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --policy.push_to_hub=false  \
    --policy.path=lerobot/smolvla_base \
    --batch_size=64 \
    --steps=100000 \
    --job_name=smolvla_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --output_dir=outputs/train/smolvla_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
    --wandb.enable=true \
    --wandb.project=jyw_act_project  \
    --wandb.entity=bigmodel  \
    --wandb.disable_artifact=true


CUDA_VISIBLE_DEVICES=1  \

TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1  TORCH_FORCE_SDPA=1 TORCH_CUDNN_SDPA_ENABLED=1 \
  HF_HUB_OFFLINE=1  \
  accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
  --policy.push_to_hub=false  \
  --policy.path=lerobot/smolvla_base \
  --batch_size=8 \
  --job_name=smolvla_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
  --output_dir=outputs/train/smolvla_test \
  --wandb.enable=false


TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1  TORCH_FORCE_SDPA=1 TORCH_CUDNN_SDPA_ENABLED=1 \
  HF_HUB_OFFLINE=1  \
  accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
  --policy.push_to_hub=false  \
  --policy.path=lerobot/smolvla_base \
  --policy.num_vlm_layers=8  \
  --policy.tokenizer_max_length=16  \
  --batch_size=8 \
  --job_name=smolvla_task_agx_pick_silver_metal_battery_to_white_plate_right_arm_cfg_fps30_cam2_img224_abs7_trim_tail_300 \
  --output_dir=outputs/train/smolvla_test \
  --wandb.enable=false




WANDB_API_BASE=http://172.16.128.4:8280/api \
  WANDB_BASE_URL=http://172.16.128.4:8280  \
  WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
  TORCH_FORCE_SDPA=1 TORCH_CUDNN_SDPA_ENABLED=1 \
  HF_HUB_OFFLINE=1  \
  accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped \
  --policy.push_to_hub=false  \
  --policy.path=lerobot/smolvla_base \
  --batch_size=8 \
  --job_name=smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail \
  --output_dir=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail \
  --wandb.enable=true \
  --wandb.project=jyw_act_project  \
  --wandb.entity=bigmodel  \
  --wandb.disable_artifact=true

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  WANDB_API_BASE=http://172.16.128.4:8280/api \
  WANDB_BASE_URL=http://172.16.128.4:8280  \
  WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
  TORCH_FORCE_SDPA=1 TORCH_CUDNN_SDPA_ENABLED=1 \
  HF_HUB_OFFLINE=1  \
  accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14 \
  --policy.push_to_hub=false  \
  --policy.path=lerobot/smolvla_base \
  --batch_size=16 \
  --job_name=smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14 \
  --output_dir=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14 \
  --wandb.enable=false

  WANDB_API_BASE=http://172.16.128.4:8280/api \
  WANDB_BASE_URL=http://172.16.128.4:8280  \
  WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11  \
  TORCH_FORCE_SDPA=1 TORCH_CUDNN_SDPA_ENABLED=1 \
  HF_HUB_OFFLINE=1  \
  accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped \
  --policy.push_to_hub=false  \
  --policy.path=lerobot/smolvla_base \
  --batch_size=1 \
  --job_name=smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped_merged \
  --output_dir=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped_merged \
  --wandb.enable=false
```
## test
```bash

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_test.smolvla.draw.py \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.smolvla.14.py \
  --policy.path=outputs/train/my_smolvla2/checkpoints/last/pretrained_model

# old version
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.smolvla.14.py \
  --policy.path=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail/checkpoints/last/pretrained_model

# new version
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.smolvla.14.py \
  --policy.path=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14/checkpoints/last/pretrained_model  \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14  \
  --eval_dataset=True


HF_HUB_OFFLINE=1  \
CUDA_VISIBLE_DEVICES=3  \
accelerate launch  \
--num_processes=1  \
--mixed_precision=bf16  \
jyw_ws/infer_server.smolvla.14.py \
  --policy.path=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14/checkpoints/last/pretrained_model  \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14  \
  --eval_dataset=True

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.smolvla.14.py \
  --policy.path=outputs/train/smolvla_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14/checkpoints/last/pretrained_model
```

# pi0.5
## train
```
python src/lerobot/scripts/lerobot_train.py\
  --dataset.repo_id=jyw/your_dataset_name \
  --policy.type=pi05 \
  --output_dir=./outputs/pi05_training \
  --job_name=pi05_training \
  --policy.repo_id=jyw/place_metallic_object_pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --wandb.enable=false \
  --policy.dtype=bfloat16 \
  --steps=3000 \
  --policy.device=cuda \
  --batch_size=16

# Using a multi-GPU setup
  HF_HUB_OFFLINE=1  \
accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --output_dir=outputs/train/groot \
  --policy.push_to_hub=false \
  --policy.type=groot \
  --batch_size=1 \
  --policy.batch_size=1 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.repo_id=jyw/groot_task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=jyw/task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w640h480_abs14_trim_tail_grouped \
  --wandb.enable=false
```
