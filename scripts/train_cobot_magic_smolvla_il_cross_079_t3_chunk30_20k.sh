#!/usr/bin/env bash
set -euo pipefail

# SmolVLA IL training for the 79 cross-placement demonstrations converted as:
#   observation.state[t] = puppet state[t]
#   arm action[t]         = puppet state[t + 3, 0:6]
#   gripper action[t]     = executed gripper command[t, 6]
# Both data and policy actions use the 30 Hz clock.
# Photometric augmentation is enabled; spatial transforms stay disabled because
# they would move the object in image space without changing the action label.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
CONDA_BIN="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin"
DATASET_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_cross_079_arm_state_t3_cmd_gripper"
BASE_MODEL="/home/fuxin/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
NORMALIZATION_MAPPING='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.7,1.3]}}}'

RUN_NAME="${COBOT_RUN_NAME:-cobot_magic_smolvla_cross_079_t3_chunk30_20k_bs64_meanstd_photoaug}"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/${RUN_NAME}"
LOG_PATH="${LEROBOT_ROOT}/outputs/${RUN_NAME}.log"
STEPS="${COBOT_TRAIN_STEPS:-20000}"
SAVE_FREQ="${COBOT_SAVE_FREQ:-5000}"
LOG_FREQ="${COBOT_LOG_FREQ:-100}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Converted dataset is not ready: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "Base model is missing: ${BASE_MODEL}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"
exec accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/cobot_magic_il_cross_079_arm_state_t3_cmd_gripper \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS_TFS}" \
    --policy.type=smolvla \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.normalization_mapping="${NORMALIZATION_MAPPING}" \
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
    --batch_size=32 \
    --steps="${STEPS}" \
    --num_workers=8 \
    --seed=1000 \
    --eval_freq=0 \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable=false \
    2>&1 | tee "${LOG_PATH}"
