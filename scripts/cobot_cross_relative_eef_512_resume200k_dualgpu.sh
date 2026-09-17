#!/usr/bin/env bash
set -euo pipefail

# Resume the completed 20k 512x512 anchor-relative-EEF run.
# The checkpoint train_config supplies the dataset, processor, model and
# optimizer configuration; CLI overrides only extend the terminal step.

ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CHECKPOINT_CONFIG="${ROOT}/outputs/cobot_magic_smolvla_cross_073_anchor_relative_eef_chunk30_bc20k_globalbs64_512x512_dualgpu_photoaug/checkpoints/020000/pretrained_model/train_config.json"
LOG_PATH="${ROOT}/outputs/cobot_magic_smolvla_cross_073_anchor_relative_eef_chunk30_bc200k_globalbs64_512x512_dualgpu_resume.log"

test -f "${CHECKPOINT_CONFIG}"
test -f "${ROOT}/outputs/cobot_magic_smolvla_cross_073_anchor_relative_eef_chunk30_bc20k_globalbs64_512x512_dualgpu_photoaug/checkpoints/020000/training_state/training_step.json"

export PATH="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${ROOT}"
exec accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --config_path="${CHECKPOINT_CONFIG}" \
    --resume=true \
    --steps=200000 \
    --save_freq=5000 \
    --log_freq=100 \
    2>&1 | tee "${LOG_PATH}"
