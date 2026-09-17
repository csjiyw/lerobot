#!/usr/bin/env bash
set -euo pipefail

# Start a new 300k run initialized only from the 20k model weights.
# This is intentionally not --resume: optimizer, scheduler, RNG and step
# numbering all start fresh, while relative-EEF preprocessing uses the active
# dataset statistics.

ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
SOURCE_RUN="${ROOT}/outputs/cobot_magic_smolvla_cross_073_anchor_relative_eef_chunk30_bc20k_globalbs64_512x512_dualgpu_photoaug"
MODEL_INIT="${SOURCE_RUN}/checkpoints/020000/pretrained_model"
CONFIG_PATH="${MODEL_INIT}/train_config.json"
RUN_NAME="cobot_magic_smolvla_cross_073_anchor_relative_eef_chunk30_from20k_retrain300k_globalbs64_512x512_dualgpu_lr1e-5_cosine"
OUTPUT_DIR="${ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${ROOT}/outputs/${RUN_NAME}.log"

test -f "${CONFIG_PATH}"
test -f "${MODEL_INIT}/model.safetensors"
test ! -e "${OUTPUT_DIR}"

export PATH="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${ROOT}"
exec accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --config_path="${CONFIG_PATH}" \
    --policy.path="${MODEL_INIT}" \
    --resume=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --steps=300000 \
    --policy.optimizer_lr=1e-5 \
    --policy.scheduler_warmup_steps=0 \
    --policy.scheduler_decay_steps=300000 \
    --policy.scheduler_decay_lr=2.5e-7 \
    --save_freq=5000 \
    --log_freq=100 \
    2>&1 | tee "${LOG_PATH}"
