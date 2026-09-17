#!/usr/bin/env bash
set -euo pipefail

# Single-episode convergence sanity test for the anchor-relative-EEF SmolVLA.
# This intentionally uses one episode, batch size 1, one GPU and no image
# augmentation. It tests whether the model/training/data path can overfit a
# small fixed distribution before running a full-dataset experiment.

ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_cross_073_clean_arm_state_t3_cmd_gripper_v3_eef"
DATASET_REPO_ID="local/cobot_magic_il_cross_073_clean_arm_state_t3_cmd_gripper_v3_eef"
RELATIVE_STATS="${DATASET_ROOT}/meta/relative_eef_action_stats.json"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"

EPISODE="${COBOT_OVERFIT_EPISODE:-0}"
BATCH_SIZE="${COBOT_BATCH_SIZE:-1}"
STEPS="${COBOT_TRAIN_STEPS:-2000}"
NUM_WORKERS="${COBOT_NUM_WORKERS:-0}"
IMAGE_SIZE="${COBOT_IMAGE_SIZE:-512}"
CUDA_DEVICE="${COBOT_CUDA_DEVICE:-0}"
SAVE_FREQ="${COBOT_SAVE_FREQ:-500}"
LOG_FREQ="${COBOT_LOG_FREQ:-20}"
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_cross_073_anchor_relative_eef_overfit_ep${EPISODE}_bs1_${IMAGE_SIZE}_naug_2k}"
OUTPUT_DIR="${ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${ROOT}/outputs/${RUN_NAME}.log"

INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
NORMALIZATION_MAPPING='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'

for required in "${DATASET_ROOT}/meta/info.json" "${RELATIVE_STATS}" "${BASE_MODEL}/model.safetensors"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required input is missing: ${required}" >&2
        exit 1
    fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${ROOT}"
exec accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="[${EPISODE}]" \
    --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=false \
    --policy.type=smolvla \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.normalization_mapping="${NORMALIZATION_MAPPING}" \
    --policy.resize_imgs_with_padding="[${IMAGE_SIZE},${IMAGE_SIZE}]" \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_relative_eef_actions=true \
    --policy.relative_eef_action_stats_path="${RELATIVE_STATS}" \
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
    --num_workers="${NUM_WORKERS}" \
    --seed=1000 \
    --eval_freq=0 \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable=false \
    2>&1 | tee "${LOG_PATH}"
