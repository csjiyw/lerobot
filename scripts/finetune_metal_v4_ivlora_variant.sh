#!/usr/bin/env bash
# Parameterised LoRA variant. Env: TAG, LORA_LR, LORA_STEPS, POOL_TAG, PEFT_TARGETS (optional regex)
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CHECKPOINT="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model"
DATASET_ROOT="${TASK_ROOT}/../datasets/cobot_magic_metal_v4_ivmix"
TAG="${TAG:?set TAG}"
export COBOT_BASE_MODEL="${CHECKPOINT}"
export COBOT_RUN_NAME="cobot_magic_smolvla_metal_v4_ivlora_${TAG}"
export COBOT_DATASET_ROOT="${DATASET_ROOT}"
export COBOT_DATASET_REPO_ID="local/cobot_magic_metal_v4_ivmix"
export COBOT_GROUP_WEIGHTS_JSON="${DATASET_ROOT}/meta/sampler_weights_${POOL_TAG:-iv50}.json"
export COBOT_STATE_MASK_INDICES='[]'
export COBOT_STATE_DROPOUT=0.3
export COBOT_STATE_NOISE=0.05
export COBOT_CAMERA_DROPOUT=0.1
export COBOT_LR="${LORA_LR:-5e-5}"
export COBOT_WARMUP=50
export COBOT_DECAY_STEPS="${LORA_STEPS:-500}"
export COBOT_DECAY_LR=1e-6
export COBOT_TRAIN_STEPS="${LORA_STEPS:-500}"
export COBOT_BATCH_SIZE=32
export COBOT_CUDA_DEVICE=1
export COBOT_SAVE_FREQ="${LORA_SAVE:-250}"
export COBOT_LOG_FREQ=20
export COBOT_NUM_WORKERS=4
export COBOT_PEFT=1
export COBOT_PEFT_R="${LORA_R:-16}"
if [[ -n "${PEFT_TARGETS:-}" ]]; then export COBOT_PEFT_TARGETS="${PEFT_TARGETS}"; fi
[[ -f "${CHECKPOINT}/model.safetensors" ]]
[[ -f "${COBOT_GROUP_WEIGHTS_JSON}" ]]
[[ ! -e "${TASK_ROOT}/outputs/${COBOT_RUN_NAME}" ]] || { echo "Refusing to overwrite" >&2; exit 1; }
cd "${TASK_ROOT}"
exec bash scripts/train_metal_v4.sh
