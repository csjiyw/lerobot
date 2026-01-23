# install
pip install pysocks

ssh -R 0.0.0.0:17890:127.0.0.1:7890 -p 2222 root@172.16.128.4

export ALL_PROXY=socks5h://127.0.0.1:17897


export http_proxy=http://192.168.25.160:7897;
export https_proxy=http://192.168.25.160:7897

uv pip install "torch==2.7" "torchvision<0.23.0" "torchcodec<0.6.0"  
 --index-url https://download.pytorch.org/whl/cu126


uv pip install -e ".[pi]"   --index-url https://download.pytorch.org/whl/cu126

# too low version will cause error


# train


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
    --batch_size=1



# infer

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

# test

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

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_test.smolvla.draw.py \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.smolvla.14.py \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.draw.py \
  --policy.path=/workspace/lerobot/outputs/robot_learning_tutorial2/act_20000

CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.py \
  --policy.path=/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box_train2/checkpoints/last/pretrained_model


python jyw_ws/infer_test.py \
  --policy.path="/workspace/lerobot_jpeg/outputs/train/act_aloha_mobile_pick_yellow_box_train_split/checkpoints/last/pretrained_model"  


CUDA_VISIBLE_DEVICES=3 python jyw_ws/infer_server.py   --policy.path=/workspace/lerobot/outputs/robot_learning_tutorial3/act_final


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
  --wandb.enable=false \
  --rename_map='{"observation.images.hand_left": "observation.images.camera1", "observation.images.head": "observation.images.camera2","observation.images.hand_right":"observation.images.camera3"}'