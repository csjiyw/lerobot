#!/usr/bin/env bash
set -euo pipefail

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
SOURCE_CHECKPOINT="${LEROBOT_ROOT}/outputs/cobot_magic_smolvla_workpiece2_realaction_abs_joint_chunk30_from100k_ft1000_bs32_gpu1_lr1e-5/checkpoints/001000"
RUN_NAME="cobot_magic_smolvla_workpiece2_realaction_abs_joint_chunk30_from100k_ft5000_resume1000_bs32_gpu1_lr1e-5"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
for required in "${SOURCE_CHECKPOINT}/pretrained_model/train_config.json" "${SOURCE_CHECKPOINT}/training_state/training_step.json" "${SOURCE_CHECKPOINT}/training_state/optimizer_state.safetensors"; do
    [[ -f "${required}" ]] || { echo "Missing resume state: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${LEROBOT_ROOT}"

# Restore the 1000-update weights, Adam moments, RNG and step counter.
# The previous 1000-update cosine schedule has ended; start a new no-warmup
# cosine at 1e-5 for the remaining 4000 updates, ending at 1e-6.
exec accelerate launch \
    --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --config_path="${SOURCE_CHECKPOINT}/pretrained_model/train_config.json" \
    --resume=true --resume_scheduler=false --inherit_pretrained_processors=true \
    --optimizer.lr=1e-5 --scheduler.peak_lr=1e-5 --scheduler.decay_lr=1e-6 \
    --scheduler.num_warmup_steps=0 --scheduler.num_decay_steps=4000 \
    --policy.optimizer_lr=1e-5 --policy.scheduler_warmup_steps=0 \
    --policy.scheduler_decay_steps=4000 --policy.scheduler_decay_lr=1e-6 \
    --output_dir="${OUTPUT_DIR}" --job_name="${RUN_NAME}" \
    --steps=5000 --save_freq=500 --log_freq=50 --eval_freq=0 --wandb.enable=false
