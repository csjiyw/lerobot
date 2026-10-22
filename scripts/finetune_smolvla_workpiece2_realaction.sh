#!/usr/bin/env bash
set -euo pipefail

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="${COBOT_DATASET_ROOT:-/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_workpiece2_realaction_headtrim_v3}"
DATASET_REPO_ID="${COBOT_DATASET_REPO_ID:-local/cobot_magic_il_workpiece2_realaction_headtrim_v3}"
INITIAL_MODEL="${COBOT_INITIAL_MODEL:-${LEROBOT_ROOT}/outputs/cobot_magic_smolvla_5482_grippercmd_abs_joint_chunk30_100k_bs32_gpu1_256_photoaug/checkpoints/100000/pretrained_model}"
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_workpiece2_realaction_abs_joint_chunk30_from100k_ft1000_bs32_gpu1_lr1e-5}"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.9,1.1]}}}'
for required in "${DATASET_ROOT}/meta/info.json" "${INITIAL_MODEL}/model.safetensors" "${INITIAL_MODEL}/policy_preprocessor.json"; do
    [[ -f "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${COBOT_CUDA_DEVICE:-1}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${LEROBOT_ROOT}"
exec accelerate launch \
    --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=false \
    --dataset.image_transforms.enable=true --dataset.image_transforms.use_gpu=true \
    --dataset.image_transforms.pre_resize='[256,256]' \
    --dataset.image_transforms.max_num_transforms=2 --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS_TFS}" \
    --policy.path="${INITIAL_MODEL}" --inherit_pretrained_processors=true \
    --policy.chunk_size=30 --policy.n_action_steps=30 \
    --policy.use_relative_eef_actions=false --policy.use_delta_joint_actions_aloha=false \
    --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true \
    --policy.state_dropout=0 --policy.optimizer_lr="${COBOT_LR:-1e-5}" \
    --policy.scheduler_warmup_steps=0 --policy.scheduler_decay_steps="${COBOT_TRAIN_STEPS:-1000}" \
    --policy.scheduler_decay_lr=1e-6 --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" --job_name="${RUN_NAME}" \
    --batch_size=32 --steps="${COBOT_TRAIN_STEPS:-1000}" --num_workers=8 \
    --seed=20261013 --eval_freq=0 --log_freq=25 --save_checkpoint=true --save_freq="${COBOT_SAVE_FREQ:-100}" --wandb.enable=false
