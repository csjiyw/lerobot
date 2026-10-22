#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
export COBOT_DATASET_ROOT="${TASK_ROOT}/../datasets/cobot_magic_workpiece_correction5_realaction_v3"
export COBOT_DATASET_REPO_ID=local/cobot_magic_workpiece_correction5_realaction_v3
export COBOT_INITIAL_MODEL="${TASK_ROOT}/outputs/cobot_magic_smolvla_workpiece2_realaction_abs_joint_chunk30_from100k_ft1000_bs32_gpu1_lr1e-5/checkpoints/001000/pretrained_model"
export COBOT_RUN_NAME="cobot_magic_smolvla_workpiece_correction5_abs_joint_chunk30_fromft1000_ft2000_bs32_gpu1_lr2e-6"
export COBOT_TRAIN_STEPS=2000
export COBOT_SAVE_FREQ=500
export COBOT_LR=2e-6
export COBOT_CUDA_DEVICE=1
exec bash "${TASK_ROOT}/scripts/finetune_smolvla_workpiece2_realaction.sh"
