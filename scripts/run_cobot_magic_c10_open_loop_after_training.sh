#!/usr/bin/env bash
set -euo pipefail

# Wait for the paired C10 IL runs, then compare both final C10 policies with
# the existing 30-step policy.  This script is offline-only: it never starts
# ROS, the bridge, a policy server, or a robot controller.

ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"
TRAIN_SCRIPT="${ROOT}/scripts/train_cobot_magic_smolvla_il2_relative_eef_chunk10_pair.sh"
DATASET_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3_eef"
DATASET_REPO_ID="local/cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3_eef"
CHUNK30="${ROOT}/outputs/cobot_magic_smolvla_il2_eef_trimmed_fixture_v2_66ep_relative_eef_chunk30_100k_globalbs32_gpu0_256_photoaug/checkpoints/100000/pretrained_model"
FROZEN_RUN="${ROOT}/outputs/cobot_magic_smolvla_il2_trimmed_metal_relative_eef_chunk10_100000steps_bs32_256x256_photoaug_frozen_lr1e-4"
UNFROZEN_RUN="${ROOT}/outputs/cobot_magic_smolvla_il2_trimmed_metal_relative_eef_chunk10_100000steps_bs32_256x256_photoaug_unfrozen_lr1e-5"
FROZEN="${FROZEN_RUN}/checkpoints/100000/pretrained_model"
UNFROZEN="${UNFROZEN_RUN}/checkpoints/100000/pretrained_model"
REPORT_ROOT="${ROOT}/web_runs/cobot_magic_c10_vs_chunk30"
WAIT_SECONDS="${COBOT_OPEN_LOOP_WAIT_SECONDS:-60}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [[ ! -f "${CHUNK30}/model.safetensors" ]]; then
    log "missing chunk-30 comparator: ${CHUNK30}"
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    log "missing dataset: ${DATASET_ROOT}"
    exit 1
fi

while :; do
    frozen_done=0
    unfrozen_done=0
    [[ -f "${FROZEN}/model.safetensors" && -f "${FROZEN}/config.json" ]] && frozen_done=1
    [[ -f "${UNFROZEN}/model.safetensors" && -f "${UNFROZEN}/config.json" ]] && unfrozen_done=1
    frozen_live=0
    unfrozen_live=0
    # The training sessions invoke the script with a relative path
    # (``./scripts/...``), so match the stable script basename rather than the
    # absolute path stored in TRAIN_SCRIPT.
    pgrep -f "train_cobot_magic_smolvla_il2_relative_eef_chunk10_pair.sh frozen" >/dev/null 2>&1 && frozen_live=1 || true
    pgrep -f "train_cobot_magic_smolvla_il2_relative_eef_chunk10_pair.sh unfrozen" >/dev/null 2>&1 && unfrozen_live=1 || true
    log "waiting: frozen=${frozen_done}/1 live=${frozen_live}; unfrozen=${unfrozen_done}/1 live=${unfrozen_live}"
    if (( frozen_done && unfrozen_done && !frozen_live && !unfrozen_live )); then
        break
    fi
    if (( !frozen_live && !frozen_done )) || (( !unfrozen_live && !unfrozen_done )); then
        log "a training process ended without its final checkpoint"
        exit 2
    fi
    sleep "${WAIT_SECONDS}"
done

mkdir -p "${REPORT_ROOT}"
log "both final C10 checkpoints are complete; starting offline comparison"

run_eval() {
    local gpu="$1"
    local candidate="$2"
    local output="$3"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    args=(
        "${PYTHON}" "${ROOT}/scripts/compare_smolvla_open_loop_chunks.py"
        --dataset-root "${DATASET_ROOT}"
        --dataset-repo-id "${DATASET_REPO_ID}"
        --chunk30 "${CHUNK30}"
        --chunk10 "${candidate}"
        --device cuda:0
        --output "${output}"
    )
    if [[ -n "${COBOT_OPEN_LOOP_MAX_SAMPLES:-}" ]]; then
        args+=(--max-samples "${COBOT_OPEN_LOOP_MAX_SAMPLES}")
    fi
    "${args[@]}"
}

run_eval 0 "${FROZEN}" "${REPORT_ROOT}/frozen_vs_chunk30.json" >"${REPORT_ROOT}/frozen.log" 2>&1 &
frozen_pid=$!
run_eval 1 "${UNFROZEN}" "${REPORT_ROOT}/unfrozen_vs_chunk30.json" >"${REPORT_ROOT}/unfrozen.log" 2>&1 &
unfrozen_pid=$!
wait "${frozen_pid}"
wait "${unfrozen_pid}"

"${PYTHON}" - "${REPORT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = {
    "frozen": json.loads((root / "frozen_vs_chunk30.json").read_text()),
    "unfrozen": json.loads((root / "unfrozen_vs_chunk30.json").read_text()),
}

def metrics(report):
    model = report["models"][1]
    error = model["per_sample_error"]
    boundary = model["boundary_gap_mm"]
    return {
        "checkpoint": model["checkpoint"],
        "chunk_size": model["chunk_size"],
        "samples": model["samples"],
        "absolute_position_rmse": error["absolute_position_rmse"],
        "absolute_rotation_rmse": error["absolute_rotation_rmse"],
        "relative_position_rmse": error["position_rmse"],
        "relative_rotation_rmse": error["rotation_rmse"],
        "boundary_gap_mm_mean": boundary["mean"],
        "boundary_gap_mm_p95": boundary["p95"],
    }

summary = {name: metrics(report) for name, report in reports.items()}
ranking = sorted(
    summary,
    key=lambda name: (
        summary[name]["absolute_position_rmse"],
        summary[name]["absolute_rotation_rmse"],
        summary[name]["boundary_gap_mm_mean"],
    ),
)
payload = {
    "comparison": "same recorded observations; chunk30 baseline vs each C10 model",
    "selection_rule": "absolute EEF position RMSE, then rotation RMSE, then boundary gap",
    "ranking": ranking,
    "recommended_variant": ranking[0],
    "metrics": summary,
    "reports": {name: str(root / f"{name}_vs_chunk30.json") for name in reports},
}
(root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
log "comparison complete; see ${REPORT_ROOT}/selection.json"
