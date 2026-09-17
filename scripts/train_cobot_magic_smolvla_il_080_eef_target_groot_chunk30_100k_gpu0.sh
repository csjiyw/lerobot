#!/usr/bin/env bash
set -euo pipefail

# Groot-style EEF chunk BC training.
#
# Contract:
#   observation.state = absolute EEF [x,y,z,rx,ry,rz,gripper]
#   action[t]         = absolute EEF target[t+3]
#   policy output     = 30 absolute EEF targets in the Piper base frame
#
# The action was verified against state[t+3] on all 80 episodes with zero
# error for the first six dimensions.  This is deliberately separate from the
# older per-frame eef_delta checkpoint, whose action steps are state-relative.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="${COBOT_DATASET_ROOT:-/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_080_eef_target_groot_chunk30_corrected_v30}"
DATASET_REPO_ID="${COBOT_DATASET_REPO_ID:-local/cobot_magic_il_080_eef_target_groot_chunk30_corrected_v30}"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.7,1.3]}}}'

BATCH_SIZE="${COBOT_BATCH_SIZE:-32}"
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_il_080_eef_target_groot_chunk30_corrected_stats_100k_bs${BATCH_SIZE}_gpu0_photoaug}"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${LEROBOT_ROOT}/outputs/${RUN_NAME}.log"
STEPS="${COBOT_TRAIN_STEPS:-100000}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"
exec accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS_TFS}" \
    --policy.type=smolvla \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.load_vlm_weights=true \
    --policy.prefix_length=0 \
    --policy.pad_language_to=max_length \
    --policy.num_expert_layers=0 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --num_workers=8 \
    --seed=1000 \
    --eval_freq=0 \
    --log_freq=100 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --wandb.enable=false \
    2>&1 | tee "${LOG_PATH}"
