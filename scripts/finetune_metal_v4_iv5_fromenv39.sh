#!/usr/bin/env bash
# Fine-tune env39_ft20k on the 5 human-intervention (IV) segments only.
# Data: datasets/cobot_magic_metal_v4_iv5 (5 episodes, 3,215 frames, ~107 s).
set -euo pipefail

TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CHECKPOINT="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_xyz_gripper_env39_ft20k_frommasked5k_bs32_gpu1_lr2e-6/checkpoints/020000/pretrained_model"
DATASET_ROOT="${TASK_ROOT}/../datasets/cobot_magic_metal_v4_iv5"
GROUP_WEIGHTS="${DATASET_ROOT}/meta/sampler_weights.json"

export COBOT_BASE_MODEL="${CHECKPOINT}"
export COBOT_RUN_NAME="cobot_magic_smolvla_metal_v4_xyz_gripper_iv5_ft1500_fromenv39_lr5e-6"
export COBOT_DATASET_ROOT="${DATASET_ROOT}"
export COBOT_DATASET_REPO_ID="local/cobot_magic_metal_v4_iv5"
export COBOT_GROUP_WEIGHTS_JSON="${GROUP_WEIGHTS}"
export COBOT_STATE_MASK_INDICES='[3,4,5]'
export COBOT_STATE_DROPOUT=0
export COBOT_STATE_NOISE=0
export COBOT_CAMERA_DROPOUT=0.1
export COBOT_LR=5e-6
export COBOT_WARMUP=100
export COBOT_DECAY_STEPS=1500
export COBOT_DECAY_LR=5e-7
export COBOT_TRAIN_STEPS=1500
export COBOT_BATCH_SIZE=32
export COBOT_CUDA_DEVICE=1
export COBOT_SAVE_FREQ=500
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
