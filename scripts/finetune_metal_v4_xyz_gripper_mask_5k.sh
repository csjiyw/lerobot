#!/usr/bin/env bash
# 5k controlled fine-tune: keep XYZ + gripper state, mask EEF orientation.
set -euo pipefail

TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
INITIAL_MODEL="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model"

export COBOT_BASE_MODEL="${INITIAL_MODEL}"
export COBOT_RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_metal_v4_xyz_gripper_state_mask_ft5k_bs32_gpu1_lr1e-5}"
export COBOT_STATE_MASK_INDICES='[3,4,5]'
# Keep only the requested XYZ + gripper channels consistently available.
export COBOT_STATE_DROPOUT=0
export COBOT_STATE_NOISE=0
export COBOT_CAMERA_DROPOUT=0.1
export COBOT_LR="${COBOT_LR:-1e-5}"
export COBOT_WARMUP="${COBOT_WARMUP:-100}"
export COBOT_DECAY_STEPS=5000
export COBOT_DECAY_LR=1e-6
export COBOT_TRAIN_STEPS=5000
export COBOT_BATCH_SIZE=32
export COBOT_CUDA_DEVICE=1
export COBOT_SAVE_FREQ=1000
export COBOT_LOG_FREQ=25

cd "${TASK_ROOT}"
exec bash scripts/train_metal_v4.sh
