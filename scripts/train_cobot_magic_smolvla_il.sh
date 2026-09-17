#!/usr/bin/env bash
set -euo pipefail

# Task-specific SmolVLA behavior cloning before RLT Stage 1.
# BATCH_SIZE is per GPU; the default effective batch is 32 x 2 = 64.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_001"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'

RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_il_001_200k_bs64}"
OUTPUT_DIR="${COBOT_OUTPUT_DIR:-${LEROBOT_ROOT}/outputs/${RUN_NAME}}"
STEPS="${COBOT_STEPS:-200000}"
BATCH_SIZE="${COBOT_BATCH_SIZE:-32}"
NUM_WORKERS="${COBOT_NUM_WORKERS:-8}"
SAVE_FREQ="${COBOT_SAVE_FREQ:-10000}"
LOG_FREQ="${COBOT_LOG_FREQ:-100}"
GPU_IDS="${COBOT_GPU_IDS:-0,1}"

if [[ ! -x "${CONDA_BIN}/accelerate" ]]; then
    echo "accelerate is missing from ${CONDA_BIN}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "LeRobot dataset is missing: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
    echo "SmolVLA base weights are missing: ${BASE_MODEL}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
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
    --policy.load_vlm_weights=true \
    --policy.prefix_length=0 \
    --policy.pad_language_to=max_length \
    --policy.num_expert_layers=0 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --num_workers="${NUM_WORKERS}" \
    --eval_freq=0 \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable=false
