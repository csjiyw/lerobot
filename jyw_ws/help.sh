# install


export http_proxy=http://192.168.25.160:15678;
export https_proxy=http://192.168.25.160:15678

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




# infer

python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_place_metallic_object_fast_test/checkpoints/last/pretrained_model"

python jyw_ws/infer_server.py \
    --policy.path="/workspace/lerobot/outputs/train/act_multi_gpu/checkpoints/last/pretrained_model"


python jyw_ws/infer_server.py \
    --policy.path="outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"


# test

python jyw_ws/infer_server.test_data_rollout.py \
    --policy.path="/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"


python jyw_ws/infer_test.draw.py \
    --policy.path="/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"

python jyw_ws/infer_test.py \
  --policy.path="/workspace/lerobot/outputs/train/act_aloha_mobile_pick_yellow_box/checkpoints/last/pretrained_model"  


python jyw_ws/infer_test.py \
  --policy.path="/workspace/lerobot_jpeg/outputs/train/act_aloha_mobile_pick_yellow_box_train_split/checkpoints/last/pretrained_model"  