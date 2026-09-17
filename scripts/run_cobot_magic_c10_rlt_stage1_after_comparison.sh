#!/usr/bin/env bash
set -euo pipefail

# Start the C10-compatible RLT Stage 1 only after the offline 30-vs-10
# comparison has produced selection.json.  This is training-only and never
# starts the robot, ROS, bridge, rollout client, or policy server.

ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
RLT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/RL-Token-SmolVLA"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"
SELECTION="${ROOT}/web_runs/cobot_magic_c10_vs_chunk30/selection.json"
FROZEN="${ROOT}/outputs/cobot_magic_smolvla_il2_trimmed_metal_relative_eef_chunk10_100000steps_bs32_256x256_photoaug_frozen_lr1e-4/checkpoints/100000/pretrained_model"
UNFROZEN="${ROOT}/outputs/cobot_magic_smolvla_il2_trimmed_metal_relative_eef_chunk10_100000steps_bs32_256x256_photoaug_unfrozen_lr1e-5/checkpoints/100000/pretrained_model"
STAGE1_SCRIPT="${RLT_ROOT}/scripts/train_cobot_magic_rlt_stage1_il2trimrel_rlinf960_ref30_actor10_5k.sh"
LOG="${ROOT}/outputs/c10_rlt_stage1_after_comparison.log"
WAIT_SECONDS="${COBOT_RLT_WAIT_SECONDS:-60}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

while [[ ! -s "${SELECTION}" ]]; do
    log "waiting for offline comparison: ${SELECTION}"
    sleep "${WAIT_SECONDS}"
done

variant="$(${PYTHON} - "${SELECTION}" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text()).get("recommended_variant")
if value not in {"frozen", "unfrozen"}:
    raise SystemExit(f"invalid recommended_variant: {value!r}")
print(value)
PY
)"

case "${variant}" in
    frozen)
        smolvla_path="${FROZEN}"
        ;;
    unfrozen)
        smolvla_path="${UNFROZEN}"
        ;;
esac
if [[ ! -f "${smolvla_path}/model.safetensors" ]]; then
    log "selected C10 checkpoint is missing: ${smolvla_path}"
    exit 2
fi

run_name="cobot_magic_smolvla_il2_trimmed_metal_c10_${variant}_rlt_stage1_chunk10_actor10_5k_bs64"
output_dir="${RLT_ROOT}/checkpoints/${run_name}"
if [[ -e "${output_dir}" ]]; then
    log "RLT output already exists; refusing to overwrite: ${output_dir}"
    exit 3
fi

log "selected ${variant} C10 checkpoint from offline comparison"
log "launching RLT Stage 1 with reference chunk10 and actor horizon10"
export CUDA_VISIBLE_DEVICES="${COBOT_RLT_CUDA_DEVICE:-0}"
export SMOLVLA_PATH="${smolvla_path}"
export RUN_NAME="${run_name}"
exec bash "${STAGE1_SCRIPT}"
