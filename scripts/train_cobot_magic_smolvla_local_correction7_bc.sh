#!/usr/bin/env bash
set -euo pipefail

# Targeted BC refinement from a compatible relative-EEF checkpoint.
# This is a targeted continuation, so it inherits the source checkpoint's
# complete processor/statistics contract. The mixed dataset supplies samples
# only; rebasing state/action normalization here would change the coordinates
# seen by the existing weights. Vision stays frozen: the probe confirms that
# the current encoder preserves the close-up visual difference, while the
# action expert/state projection learn the human correction behavior.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="${COBOT_DATASET_ROOT:-/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il2_base66_localcorrection7x8_chunk10_eef}"
DATASET_REPO_ID="${COBOT_DATASET_REPO_ID:-local/$(basename "${DATASET_ROOT}")}"
DATASET_EPISODES="${COBOT_DATASET_EPISODES:-}"
INITIAL_MODEL="${COBOT_INITIAL_MODEL:-${LEROBOT_ROOT}/outputs/cobot_magic_smolvla_il2_trimmed_metal_relative_eef_chunk10_100000steps_bs32_256x256_photoaug_frozen_lr1e-4/checkpoints/100000/pretrained_model}"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
NORMALIZATION_MAPPING='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.9,1.1]}}}'

BATCH_SIZE="${COBOT_BATCH_SIZE:-32}"
STEPS="${COBOT_TRAIN_STEPS:-10000}"
NUM_WORKERS="${COBOT_NUM_WORKERS:-8}"
SAVE_FREQ="${COBOT_SAVE_FREQ:-1000}"
LOG_FREQ="${COBOT_LOG_FREQ:-50}"
GPU_ID="${COBOT_CUDA_DEVICE:-1}"
LR="${COBOT_LR:-1e-5}"
STATE_DROPOUT="${COBOT_STATE_DROPOUT:-0.5}"
ACTION_CHUNK_SIZE="${COBOT_ACTION_CHUNK_SIZE:-10}"
SOURCE_LABEL="${COBOT_SOURCE_LABEL:-100k_frozen}"
python - "${STATE_DROPOUT}" <<'PY'
import sys

p = float(sys.argv[1])
if not 0.0 <= p <= 1.0:
    raise SystemExit("COBOT_STATE_DROPOUT must be in [0, 1]")
PY
[[ "${ACTION_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "COBOT_ACTION_CHUNK_SIZE must be a positive integer" >&2; exit 2; }
RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_il2_localcorrection7x8_bc_${STEPS}steps_bs${BATCH_SIZE}_chunk${ACTION_CHUNK_SIZE}_from${SOURCE_LABEL}_lr${LR}_statedrop${STATE_DROPOUT}}"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${LEROBOT_ROOT}/outputs/${RUN_NAME}.log"

DATASET_EPISODE_ARGS=()
if [[ -n "${DATASET_EPISODES}" ]]; then
    DATASET_EPISODE_ARGS+=(--dataset.episodes="${DATASET_EPISODES}")
fi

for required in "${DATASET_ROOT}/meta/info.json" "${INITIAL_MODEL}/config.json" "${INITIAL_MODEL}/policy_preprocessor.json" "${BASE_MODEL}/model.safetensors"; do
    [[ -f "${required}" ]] || { echo "Required input is missing: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2; exit 1; }

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
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
    "${DATASET_EPISODE_ARGS[@]}" \
    --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.use_gpu=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS_TFS}" \
    --policy.type=smolvla \
    --policy.pretrained_path="${INITIAL_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.normalization_mapping="${NORMALIZATION_MAPPING}" \
    --policy.resize_imgs_with_padding="[256,256]" \
    --policy.chunk_size="${ACTION_CHUNK_SIZE}" \
    --policy.n_action_steps="${ACTION_CHUNK_SIZE}" \
    --policy.use_relative_eef_actions=true \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.state_dropout="${STATE_DROPOUT}" \
    --policy.prefix_length=0 \
    --policy.pad_language_to=max_length \
    --policy.num_expert_layers=0 \
    --policy.optimizer_lr="${LR}" \
    --policy.scheduler_warmup_steps=0 \
    --policy.scheduler_decay_steps="${STEPS}" \
    --policy.scheduler_decay_lr=1e-6 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.use_amp=true \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --num_workers="${NUM_WORKERS}" \
    --seed=20260917 \
    --eval_freq=0 \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable=false \
    2>&1 | tee "${LOG_PATH}"
