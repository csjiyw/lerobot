export http_proxy=http://192.168.8.102:15678;
export https_proxy=http://192.168.8.102:15678

uv pip install "torch==2.7" "torchvision<0.23.0" "torchcodec<0.6.0"  
 --index-url https://download.pytorch.org/whl/cu126

# too low version will cause error

lerobot-train \
  --dataset.repo_id=jyw/your_dataset_name \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_mobile_test \
  --job_name=act_aloha_mobile_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jyw/your_dataset_name_act

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