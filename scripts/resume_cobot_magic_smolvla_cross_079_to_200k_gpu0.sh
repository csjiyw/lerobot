#!/usr/bin/env bash
set -euo pipefail

# Resume the 20k SmolVLA IL checkpoint on GPU 0 and train to 200k total steps.
# The original 20k run auto-scaled its cosine schedule to end at step 20k,
# where the LR reached 2.5e-6. Keeping decay_steps=20k for this continuation
# preserves that LR instead of introducing a discontinuous LR jump on resume.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
SOURCE_CHECKPOINT="${LEROBOT_ROOT}/outputs/cobot_magic_smolvla_cross_079_t3_chunk30_20k_bs64_meanstd_photoaug/checkpoints/020000"
SOURCE_CONFIG="${SOURCE_CHECKPOINT}/pretrained_model/train_config.json"
RUN_NAME="cobot_magic_smolvla_cross_079_t3_chunk30_200k_bs64_meanstd_photoaug_gpu0_resume"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${LEROBOT_ROOT}/outputs/${RUN_NAME}.log"

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "Source checkpoint config is missing: ${SOURCE_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}/training_state/training_step.json" ]]; then
    echo "Source training state is missing: ${SOURCE_CHECKPOINT}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"
exec "${CONDA_BIN}/accelerate" launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --config_path="${SOURCE_CONFIG}" \
    --resume=true \
    --steps=200000 \
    --batch_size=64 \
    --num_workers=8 \
    --scheduler.num_decay_steps=20000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --log_freq=100 \
    --save_checkpoint=true \
    --save_freq=20000 \
    --wandb.enable=false \
    2>&1 | tee "${LOG_PATH}"
