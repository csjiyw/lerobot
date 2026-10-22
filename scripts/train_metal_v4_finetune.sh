#!/usr/bin/env bash
# v4 current-env fine-tune (plan step 8): saturate current env, keep anti-shortcut knobs, small LR.
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
: "${V4_FT_FROM:?set V4_FT_FROM=<phase2 ckpt>/pretrained_model}"
export COBOT_BASE_MODEL="${V4_FT_FROM}"
export COBOT_RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_metal_v4_ft_current_lr2e5}"
export COBOT_STATE_DROPOUT="${COBOT_STATE_DROPOUT:-0.15}"
export COBOT_STATE_NOISE="${COBOT_STATE_NOISE:-0.03}"
export COBOT_CAMERA_DROPOUT="${COBOT_CAMERA_DROPOUT:-0.05}"
export COBOT_LR="${COBOT_LR:-2e-5}" COBOT_WARMUP="${COBOT_WARMUP:-200}"
export COBOT_DECAY_STEPS="${COBOT_DECAY_STEPS:-10000}" COBOT_DECAY_LR="${COBOT_DECAY_LR:-1e-6}"
export COBOT_TRAIN_STEPS="${COBOT_TRAIN_STEPS:-10000}"
# current-heavy sampler weights ship next to the dataset meta
sed 's#meta/sampler_weights.json#meta/sampler_weights_ft_current.json#' "${TASK_ROOT}/scripts/train_metal_v4.sh" | exec bash /dev/stdin
