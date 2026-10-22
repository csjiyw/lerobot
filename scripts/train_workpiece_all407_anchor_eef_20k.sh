#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="${TASK_ROOT}/../datasets/cobot_magic_workpiece_all407_anchor_eef_v3"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_workpiece_all407_anchor_relative_eef_chunk30_base_20k_bs16_gpu1_256_photoaug_cs8}"
OUTPUT_DIR="${TASK_ROOT}/outputs/${RUN_NAME}"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.9,1.1]}}}'
for required in "${DATASET_ROOT}/meta/info.json" "${DATASET_ROOT}/meta/relative_eef_action_stats.json" "${BASE_MODEL}/model.safetensors"; do
    [[ -f "${required}" ]] || { echo "Missing ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
export PATH="${CONDA_BIN}:${PATH}" CUDA_VISIBLE_DEVICES="${COBOT_CUDA_DEVICE:-1}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${TASK_ROOT}"
LAUNCH_ARGS=(--num_processes="${COBOT_NUM_PROCESSES:-1}" --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no)
if [[ "${COBOT_NUM_PROCESSES:-1}" -gt 1 ]]; then LAUNCH_ARGS+=(--multi_gpu); fi
exec accelerate launch "${LAUNCH_ARGS[@]}" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/cobot_magic_workpiece_all407_anchor_eef_v3 --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav --dataset.use_imagenet_stats=false \
    --dataset.image_transforms.enable=true --dataset.image_transforms.use_gpu=true \
    --dataset.image_transforms.pre_resize='[256,256]' --dataset.image_transforms.max_num_transforms=2 --dataset.image_transforms.tfs="${TFS}" \
    --policy.type=smolvla --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" --policy.output_features='{"action":{"type":"ACTION","shape":[7]}}' \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --policy.resize_imgs_with_padding='[256,256]' --policy.chunk_size=30 --policy.n_action_steps=30 \
    --policy.use_relative_eef_actions=true --policy.relative_eef_action_stats_path="${DATASET_ROOT}/meta/relative_eef_action_stats.json" \
    --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true --policy.state_dropout=0 \
    --policy.load_vlm_weights=true --policy.prefix_length=0 --policy.pad_language_to=max_length --policy.num_expert_layers=0 \
    --policy.optimizer_lr=1e-4 --policy.scheduler_warmup_steps=500 --policy.scheduler_decay_steps=20000 --policy.scheduler_decay_lr=2.5e-6 \
    --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" --job_name="${RUN_NAME}" --batch_size="${COBOT_BATCH_SIZE:-16}" --steps="${COBOT_TRAIN_STEPS:-20000}" --num_workers="${COBOT_NUM_WORKERS:-8}" \
    --consecutive_frames_per_sample=8 \
    --seed=20261014 --eval_freq=0 --log_freq="${COBOT_LOG_FREQ:-100}" --save_checkpoint="${COBOT_SAVE_CHECKPOINT:-true}" --save_freq=5000 --wandb.enable=false
