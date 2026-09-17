#!/usr/bin/env bash
set -euo pipefail

# Fresh SmolVLA BC run for the clean, single-instruction 66-episode dataset.
# Source data: cobot_magic_il_2_eef, with original episodes 0..3 excluded.
# batch_size is per process in this LeRobot version; one process makes 32 the
# effective/global batch size.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="${COBOT_DATASET_ROOT:-/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_2_eef_unified_metal_fixture_v1_drop_ep0_3}"
DATASET_REPO_ID="${COBOT_DATASET_REPO_ID:-local/cobot_magic_il_2_eef_unified_metal_fixture_v1_drop_ep0_3}"
RELATIVE_STATS="${DATASET_ROOT}/meta/relative_eef_action_stats.json"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
NORMALIZATION_MAPPING='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.7,1.3]}}}'

BATCH_SIZE=32
NUM_WORKERS="${COBOT_NUM_WORKERS:-8}"
STEPS="${COBOT_TRAIN_STEPS:-100000}"
SAVE_FREQ="${COBOT_SAVE_FREQ:-5000}"
LOG_FREQ="${COBOT_LOG_FREQ:-100}"
IMAGE_HEIGHT=256
IMAGE_WIDTH=256
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_il2_eef_unified_fixture_66ep_relative_eef_chunk30_100k_globalbs32_gpu0_256_photoaug}"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${LEROBOT_ROOT}/outputs/${RUN_NAME}.log"

for required in "${DATASET_ROOT}/meta/info.json" "${RELATIVE_STATS}" "${BASE_MODEL}/model.safetensors"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required input is missing: ${required}" >&2
        exit 1
    fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"
exec accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.use_gpu=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS_TFS}" \
    --policy.type=smolvla \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.normalization_mapping="${NORMALIZATION_MAPPING}" \
    --policy.resize_imgs_with_padding="[${IMAGE_HEIGHT},${IMAGE_WIDTH}]" \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_relative_eef_actions=true \
    --policy.relative_eef_action_stats_path="${RELATIVE_STATS}" \
    --policy.load_vlm_weights=true \
    --policy.prefix_length=0 \
    --policy.pad_language_to=max_length \
    --policy.num_expert_layers=0 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=30000 \
    --policy.scheduler_decay_lr=2.5e-6 \
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
