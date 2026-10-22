#!/usr/bin/env bash
set -euo pipefail

# v3 retrain: metal-fixture workpiece task, scene A only.
# Dataset: cobot_magic_metal_fixture_v3_headonly_success_human_eef
#   - 90 episodes / 48381 frames: 66 BC + 7 corrections + 17 RLT human segments
#   - success episodes only, human control only (control_source=2), head-only trim (去头不去尾)
# Recipe: smolvla_base from scratch, single 96GB GPU, 64 per GPU (global 64),
#   no gradient accumulation, 200k steps, checkpoint every 10k, state_dropout 0.2,
#   chunk30 relative-eef actions (fresh stats on this dataset), 256px photoaug.
# NOTE: earlier 2x256 dual-GPU variant hit the container cgroup 128GiB memory cap
#   (dataloader workers) and PFS dataloader stalls; single-GPU 64bs per user decision.

LEROBOT_ROOT="${LEROBOT_ROOT:-/jyw_ws/vla_v3/lerobot}"
PYBIN="${PYBIN:-/jyw_ws/vla_v3/.venv/bin}"
# Local-disk dataset copy: PFS vs local disk made no difference to data_s (smoke
# tested 2026-09-18); keep the copy as the default (836M fits easily in the 256G
# overlay) and allow PFS fallback via env.
DATASET_ROOT="${DATASET_ROOT:-/root/datasets_v3/cobot_magic_metal_fixture_v3_headonly_success_human_eef}"
DATASET_REPO_ID="local/cobot_magic_metal_fixture_v3_headonly_success_human_eef"
RELATIVE_STATS="${DATASET_ROOT}/meta/relative_eef_action_stats.json"
BASE_MODEL="${BASE_MODEL:-/jyw_ws/vla_v3/hf_cache/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
# Optional resume: RESUME_FROM=<ckpt dir with pretrained_model/ + training_state/>
#   -> model weights from <RESUME_FROM>/pretrained_model,
#      step/optimizer/scheduler/rng restored from <RESUME_FROM> via --resume.
RESUME_FROM="${RESUME_FROM:-}"
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera_f":{"type":"VISUAL","shape":[3,480,640]},"observation.images.camera_r":{"type":"VISUAL","shape":[3,480,640]}}'
NORMALIZATION_MAPPING='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.7,1.3]}}}'

# 64 per GPU, single GPU -> global batch 64, no gradient accumulation.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
STEPS="${STEPS:-200000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
# 16 workers measured fastest on pro6000 (64-CPU quota): 8w data_s 0.24s, 16w 0.13s,
# 24w 0.18s (oversubscription). See 2026-09-18 smokes on GPU1.
NUM_WORKERS="${NUM_WORKERS:-16}"
LOG_FREQ=100
IMAGE_HEIGHT=256
IMAGE_WIDTH=256
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
RUN_NAME="${RUN_NAME:-cobot_magic_smolvla_v3_headonly_eef_90ep_success_human_200k_globalbs64_singlegpu64_photoaug_stdrop0.2}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEROBOT_ROOT}/outputs/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}.log}"

if [[ -n "${RESUME_FROM}" ]]; then
    PRETRAINED_PATH="${RESUME_FROM}/pretrained_model"
else
    PRETRAINED_PATH="${BASE_MODEL}"
fi

for required in "${DATASET_ROOT}/meta/info.json" "${RELATIVE_STATS}" "${PRETRAINED_PATH}/model.safetensors"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required input is missing: ${required}" >&2
        exit 1
    fi
done
if [[ -e "${OUTPUT_DIR}" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

export PATH="${PYBIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-/jyw_ws/vla_v3/hf_cache}"
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"

COMMON_ARGS=(
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.use_gpu=true \
    --dataset.image_transforms.pre_resize="[${IMAGE_HEIGHT},${IMAGE_WIDTH}]" \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="${IMAGE_TRANSFORMS_TFS}" \
    --policy.type=smolvla \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.normalization_mapping="${NORMALIZATION_MAPPING}" \
    --policy.resize_imgs_with_padding="[${IMAGE_HEIGHT},${IMAGE_WIDTH}]" \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_relative_eef_actions=true \
    --policy.relative_eef_action_stats_path="${RELATIVE_STATS}" \
    --policy.eef_relative_in_dataset=true \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.state_dropout="${STATE_DROPOUT}" \
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
    --batch_size="${BATCH_SIZE_PER_GPU}" \
    --steps="${STEPS}" \
    --num_workers="${NUM_WORKERS}" \
    --seed=1000 \
    --eval_freq=0 \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable=false
)

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    # Fork resume contract (configs/train.py validate()):
    #   --config_path=<ckpt>/pretrained_model/train_config.json loads the whole
    #   checkpoint train config (CLI args still override it), sets
    #   policy.pretrained_path=<ckpt>/pretrained_model and
    #   checkpoint_path=<ckpt> (for training_state/). No --policy.path.
    for required_ckpt in "${RESUME_FROM}/pretrained_model/train_config.json" "${RESUME_FROM}/training_state/training_step.json"; do
        if [[ ! -f "${required_ckpt}" ]]; then
            echo "RESUME_FROM incomplete (missing ${required_ckpt})" >&2
            exit 1
        fi
    done
    RESUME_ARGS+=(--resume=true --config_path="${RESUME_FROM}/pretrained_model/train_config.json" --resume_scheduler=true)
    echo "Resuming from ${RESUME_FROM}" >&2
fi

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    exec accelerate launch \
        --multi_gpu \
        --num_processes="${NUM_PROCESSES}" \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        -m lerobot.scripts.lerobot_train \
        "${COMMON_ARGS[@]}" "${RESUME_ARGS[@]}"
else
    exec "${PYBIN}/python" -m lerobot.scripts.lerobot_train "${COMMON_ARGS[@]}" "${RESUME_ARGS[@]}"
fi
