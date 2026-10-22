#!/usr/bin/env bash
# One-click offline pipeline: z stage-1 -> actor data -> anchors -> merge -> actor -> eval.
# Artifacts land under lerobot/outputs/{z_stage1,actor_data,actor}.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"
SMOLVLA="${TASK_ROOT}/outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model"
CUR_DS="${TASK_ROOT}/../datasets/workpiece3_full"
OLD_DS="${TASK_ROOT}/../datasets/cobot_magic_metal_v4"
Z_DIR="${Z_DIR:-${TASK_ROOT}/outputs/z_stage1_pipeline}"
DATA_DIR="${DATA_DIR:-${TASK_ROOT}/outputs/actor_data}"
ACTOR_DIR="${ACTOR_DIR:-${TASK_ROOT}/outputs/actor_pipeline}"
STEPS_Z="${STEPS_Z:-3000}"
STEPS_A="${STEPS_A:-2000}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${TASK_ROOT}/.."

echo "[1/6] z stage-1 (128 dims, EEF delta + gripper heads)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "${PY}" -u scripts/train_z_stage1.py \
  --smolvla-path "${SMOLVLA}" \
  --datasets "local/workpiece3_full:${CUR_DS}:0.45" "local/cobot_magic_metal_v4:${OLD_DS}:0.55" \
  --steps "${STEPS_Z}" --batch-size 16 --lr 3e-4 --z-dim 128 --eef-weight 1.0 --grip-weight 0.5 \
  --save-dir "${Z_DIR}" --device cuda --save-every 1000

echo "[2/6] actor data from current-env episodes"
STRIDE="${CUR_STRIDE:-2}" Z_PATH="${Z_DIR}/z_encoder.pt" Z_CONFIG="${Z_DIR}/config.json" \
  OUT_NPZ="${DATA_DIR}/pipeline_cur.npz" \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "${PY}" -u scripts/build_actor_data.py

echo "[3/6] historical zero-correction anchors"
STRIDE="${OLD_STRIDE:-200}" Z_PATH="${Z_DIR}/z_encoder.pt" Z_CONFIG="${Z_DIR}/config.json" \
  OUT_NPZ="${DATA_DIR}/pipeline_old.npz" \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "${PY}" -u scripts/build_actor_data_old.py

echo "[4/6] merge"
"${PY}" scripts/merge_actor_data.py --cur "${DATA_DIR}/pipeline_cur.npz" --old "${DATA_DIR}/pipeline_old.npz" --out "${DATA_DIR}/pipeline_merged.npz"

echo "[5/6] residual actor (cap=1.0 std, human weight 1)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "${PY}" -u scripts/train_residual_actor.py \
  --data "${DATA_DIR}/pipeline_merged.npz" --out "${ACTOR_DIR}" \
  --steps "${STEPS_A}" --batch 64 --lr 1e-4 --human-weight 1.0 --cap-frac 1.0 --val-episode 4

echo "[6/6] offline evaluation"
"${PY}" -u scripts/eval_composed_offline.py "${ACTOR_DIR}" "${DATA_DIR}/pipeline_cur.npz"
echo "done: z=${Z_DIR} actor=${ACTOR_DIR}"
