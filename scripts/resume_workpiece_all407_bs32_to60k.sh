#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
SOURCE_CHECKPOINT="${TASK_ROOT}/outputs/cobot_magic_smolvla_workpiece_all407_anchor_relative_eef_chunk30_base_20k_bs16_gpu1_256_photoaug_cs8/checkpoints/020000"
RUN_NAME=cobot_magic_smolvla_workpiece_all407_anchor_relative_eef_chunk30_resume20k_to60k_bs32_gpu1_256_photoaug_cs8
OUTPUT_DIR="${TASK_ROOT}/outputs/${RUN_NAME}"
for required in "${SOURCE_CHECKPOINT}/pretrained_model/train_config.json" "${SOURCE_CHECKPOINT}/training_state/training_step.json" "${SOURCE_CHECKPOINT}/training_state/optimizer_state.safetensors"; do
    [[ -f "${required}" ]] || { echo "Missing resume state: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
export PATH="${CONDA_BIN}:${PATH}" CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${TASK_ROOT}"
exec accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --config_path="${SOURCE_CHECKPOINT}/pretrained_model/train_config.json" \
    --resume=true --resume_scheduler=false --inherit_pretrained_processors=true \
    --batch_size=32 --num_workers=8 --consecutive_frames_per_sample=8 \
    --optimizer.lr=1e-5 --scheduler.peak_lr=1e-5 --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=0 --scheduler.num_decay_steps=40000 \
    --policy.optimizer_lr=1e-5 --policy.scheduler_warmup_steps=0 --policy.scheduler_decay_steps=40000 --policy.scheduler_decay_lr=2.5e-6 \
    --output_dir="${OUTPUT_DIR}" --job_name="${RUN_NAME}" --steps=60000 \
    --save_checkpoint=true --save_freq=5000 --log_freq=100 --eval_freq=0 --wandb.enable=false
