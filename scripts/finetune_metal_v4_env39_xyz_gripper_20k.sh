#!/usr/bin/env bash
# Continue from the 5k XYZ+gripper-masked model using only current-scene data
# (32 current episodes + 7 correction episodes; negligible probability for other groups).
set -euo pipefail

TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CHECKPOINT="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_xyz_gripper_state_mask_ft5k_bs32_gpu1_lr1e-5/checkpoints/005000/pretrained_model"
GROUP_WEIGHTS="${TASK_ROOT}/../datasets/cobot_magic_metal_v4/meta/sampler_weights_newonly.json"

export COBOT_BASE_MODEL="${CHECKPOINT}"
export COBOT_RUN_NAME="cobot_magic_smolvla_metal_v4_xyz_gripper_env39_ft20k_frommasked5k_bs32_gpu1_lr2e-6"
export COBOT_GROUP_WEIGHTS_JSON="${GROUP_WEIGHTS}"
export COBOT_STATE_MASK_INDICES='[3,4,5]'
export COBOT_STATE_DROPOUT=0
export COBOT_STATE_NOISE=0
export COBOT_CAMERA_DROPOUT=0.1
export COBOT_LR=2e-6
export COBOT_WARMUP=200
export COBOT_DECAY_STEPS=20000
export COBOT_DECAY_LR=5e-7
export COBOT_TRAIN_STEPS=20000
export COBOT_BATCH_SIZE=32
export COBOT_CUDA_DEVICE=1
export COBOT_SAVE_FREQ=5000
export COBOT_LOG_FREQ=25

[[ -f "${CHECKPOINT}/model.safetensors" ]]
[[ -f "${GROUP_WEIGHTS}" ]]
[[ ! -e "${TASK_ROOT}/outputs/${COBOT_RUN_NAME}" ]] || {
    echo "Refusing to overwrite ${TASK_ROOT}/outputs/${COBOT_RUN_NAME}" >&2
    exit 1
}

cd "${TASK_ROOT}"
exec bash scripts/train_metal_v4.sh
