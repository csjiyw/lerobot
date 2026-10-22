#!/usr/bin/env bash
# LoRA fine-tune from the 60k base on the merged dataset with the IV pool up-weighted.
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CHECKPOINT="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model"
DATASET_ROOT="${TASK_ROOT}/../datasets/cobot_magic_metal_v4_ivmix"

export COBOT_BASE_MODEL="${CHECKPOINT}"
export COBOT_RUN_NAME="cobot_magic_smolvla_metal_v4_ivlora_iv50_nomask_ft1000_from60k_lr1e-4"
export COBOT_DATASET_ROOT="${DATASET_ROOT}"
export COBOT_DATASET_REPO_ID="local/cobot_magic_metal_v4_ivmix"
export COBOT_GROUP_WEIGHTS_JSON="${DATASET_ROOT}/meta/sampler_weights_iv50.json"
export COBOT_STATE_MASK_INDICES='[]'
export COBOT_STATE_DROPOUT=0.3
export COBOT_STATE_NOISE=0.05
export COBOT_CAMERA_DROPOUT=0.1
export COBOT_LR=1e-4
export COBOT_WARMUP=100
export COBOT_DECAY_STEPS=1000
export COBOT_DECAY_LR=1e-5
export COBOT_TRAIN_STEPS=1000
export COBOT_BATCH_SIZE=32
export COBOT_CUDA_DEVICE=1
export COBOT_SAVE_FREQ=250
export COBOT_LOG_FREQ=20
export COBOT_NUM_WORKERS=4
export COBOT_PEFT=1
export COBOT_PEFT_R=16
[[ -f "${CHECKPOINT}/model.safetensors" ]]
[[ -f "${COBOT_GROUP_WEIGHTS_JSON}" ]]
[[ ! -e "${TASK_ROOT}/outputs/${COBOT_RUN_NAME}" ]] || { echo "Refusing to overwrite" >&2; exit 1; }
cd "${TASK_ROOT}"
exec bash scripts/train_metal_v4.sh
