#!/usr/bin/env bash
# v4 metal-task training (2026-10-14): group-weighted sampling + state dropout/noise + camera dropout.
# Phase 1: vision frozen, base from smolvla_base snapshot. See DATASET_V4_PLAN.md §4/§6.
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="${COBOT_DATASET_ROOT:-${TASK_ROOT}/../datasets/cobot_magic_metal_v4}"
BASE_MODEL="${COBOT_BASE_MODEL:-/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_metal_v4_anchor_eef_chunk30_bs32_2gpu_256_photoaug_gw}"
OUTPUT_DIR="${TASK_ROOT}/outputs/${RUN_NAME}"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.9,1.1]}}}'
for required in "${DATASET_ROOT}/meta/info.json" "${DATASET_ROOT}/meta/relative_eef_action_stats.json" "${DATASET_ROOT}/meta/sampler_weights.json" "${BASE_MODEL}/model.safetensors"; do
    [[ -f "${required}" ]] || { echo "Missing ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${COBOT_CUDA_DEVICE:-1}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH="${CONDA_BIN}:${PATH}"
cd "${TASK_ROOT}"
NP="${COBOT_NUM_PROCESSES:-1}"
LAUNCH_ARGS=(--num_processes="${NP}" --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no)
if [[ "${NP}" -gt 1 ]]; then LAUNCH_ARGS+=(--multi_gpu); fi
PEFT_ARGS=()
if [[ "${COBOT_PEFT:-0}" == "1" ]]; then
    PEFT_ARGS+=(--peft.method_type=LORA --peft.r="${COBOT_PEFT_R:-16}")
    if [[ -n "${COBOT_PEFT_TARGETS:-}" ]]; then PEFT_ARGS+=(--peft.target_modules="${COBOT_PEFT_TARGETS}"); fi
fi
exec accelerate launch "${LAUNCH_ARGS[@]}" \
    -m lerobot.scripts.lerobot_train \
    "${PEFT_ARGS[@]}" \
    --dataset.repo_id="${COBOT_DATASET_REPO_ID:-local/cobot_magic_metal_v4}" --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav --dataset.use_imagenet_stats=false \
    --dataset.group_weights_json="${COBOT_GROUP_WEIGHTS_JSON:-${DATASET_ROOT}/meta/sampler_weights.json}" \
    --dataset.image_transforms.enable=true --dataset.image_transforms.use_gpu=true \
    --dataset.image_transforms.pre_resize='[256,256]' --dataset.image_transforms.max_num_transforms=2 --dataset.image_transforms.tfs="${TFS}" \
    --policy.type=smolvla --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" --policy.output_features='{"action":{"type":"ACTION","shape":[7]}}' \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --policy.resize_imgs_with_padding='[256,256]' --policy.chunk_size=30 --policy.n_action_steps=30 \
    --policy.use_relative_eef_actions=true --policy.relative_eef_action_stats_path="${DATASET_ROOT}/meta/relative_eef_action_stats.json" \
    --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true \
    --policy.state_mask_indices="${COBOT_STATE_MASK_INDICES:-[]}" \
    --policy.state_dropout="${COBOT_STATE_DROPOUT:-0.3}" --policy.state_noise_std="${COBOT_STATE_NOISE:-0.05}" --policy.camera_dropout="${COBOT_CAMERA_DROPOUT:-0.1}" \
    --policy.unfreeze_vision_last_blocks="${COBOT_UNFREEZE_VISION:-0}" \
    --policy.load_vlm_weights=true --policy.prefix_length=0 --policy.pad_language_to=max_length --policy.num_expert_layers=0 \
    --policy.optimizer_lr="${COBOT_LR:-1e-4}" --policy.scheduler_warmup_steps="${COBOT_WARMUP:-500}" \
    --policy.scheduler_decay_steps="${COBOT_DECAY_STEPS:-60000}" --policy.scheduler_decay_lr="${COBOT_DECAY_LR:-2.5e-6}" \
    --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" --job_name="${RUN_NAME}" \
    --batch_size="${COBOT_BATCH_SIZE:-32}" --steps="${COBOT_TRAIN_STEPS:-60000}" --num_workers="${COBOT_NUM_WORKERS:-8}" \
    --consecutive_frames_per_sample="${COBOT_CS:-1}" \
    --seed=20261014 --eval_freq=0 --log_freq="${COBOT_LOG_FREQ:-50}" \
    --save_checkpoint=true --save_freq="${COBOT_SAVE_FREQ:-5000}" --wandb.enable=false
