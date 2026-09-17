#!/usr/bin/env bash
set -euo pipefail

# Controlled comparison with the original IL run:
# - same dataset, base model, seed/default scheduler, feature schema and global batch 64
# - action chunk shortened from 50 to 30
# - training capped at 20k optimizer steps

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_001"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'

RUN_NAME="cobot_magic_smolvla_il_001_chunk30_20k_bs64"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${LEROBOT_ROOT}/outputs/${RUN_NAME}.log"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"
exec accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/cobot_magic_il_001 \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --policy.type=smolvla \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.load_vlm_weights=true \
    --policy.prefix_length=0 \
    --policy.pad_language_to=max_length \
    --policy.num_expert_layers=0 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size=32 \
    --steps=20000 \
    --num_workers=8 \
    --seed=1000 \
    --eval_freq=0 \
    --log_freq=100 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --wandb.enable=false \
    2>&1 | tee "${LOG_PATH}"
