#!/usr/bin/env bash
# Joint fine-tune from env39_ft20k on full-flow data + IV episodes, IV up-weighted.
# Data: datasets/cobot_magic_metal_v4_ivmix (498 original + 5 IV episodes).
# Weights: original sampling (newonly) total 1.0, group "iv5" weight ${IV_WEIGHT:-10}.
set -euo pipefail

TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CHECKPOINT="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_xyz_gripper_env39_ft20k_frommasked5k_bs32_gpu1_lr2e-6/checkpoints/020000/pretrained_model"
DATASET_ROOT="${TASK_ROOT}/../datasets/cobot_magic_metal_v4_ivmix"
IV_WEIGHT="${IV_WEIGHT:-10}"
GROUP_WEIGHTS="${DATASET_ROOT}/meta/sampler_weights_iv${IV_WEIGHT}.json"

export COBOT_BASE_MODEL="${CHECKPOINT}"
export COBOT_RUN_NAME="cobot_magic_smolvla_metal_v4_xyz_gripper_ivmix_w${IV_WEIGHT}_nomask_ft3000_fromenv39_lr2e-6"
export COBOT_DATASET_ROOT="${DATASET_ROOT}"
export COBOT_DATASET_REPO_ID="local/cobot_magic_metal_v4_ivmix"
export COBOT_GROUP_WEIGHTS_JSON="${GROUP_WEIGHTS}"
export COBOT_STATE_MASK_INDICES='[]'
export COBOT_STATE_DROPOUT=0
export COBOT_STATE_NOISE=0
export COBOT_CAMERA_DROPOUT=0.1
export COBOT_LR=2e-6
export COBOT_WARMUP=200
export COBOT_DECAY_STEPS=3000
export COBOT_DECAY_LR=5e-7
export COBOT_TRAIN_STEPS=3000
export COBOT_BATCH_SIZE=32
export COBOT_CUDA_DEVICE=1
export COBOT_SAVE_FREQ=1000
export COBOT_LOG_FREQ=25
export COBOT_NUM_WORKERS=4

[[ -f "${CHECKPOINT}/model.safetensors" ]]
[[ -f "${GROUP_WEIGHTS}" ]]
[[ ! -e "${TASK_ROOT}/outputs/${COBOT_RUN_NAME}" ]] || {
    echo "Refusing to overwrite ${TASK_ROOT}/outputs/${COBOT_RUN_NAME}" >&2
    exit 1
}

cd "${TASK_ROOT}"
exec bash scripts/train_metal_v4.sh
