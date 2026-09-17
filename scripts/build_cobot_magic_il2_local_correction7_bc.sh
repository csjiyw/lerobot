#!/usr/bin/env bash
set -euo pipefail

# Build the targeted local-correction BC dataset.
#
# The source is the established 66-episode metal-fixture IL corpus plus the
# seven valid, close-up human correction demonstrations.  The local episodes
# are repeated so their corrective behavior is not drowned out by the long
# approach/grasp portions of the base corpus.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
DATASETS_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"
SOURCE_MANIFEST="${DATASETS_ROOT}/cobot_magic_il_2_eef/meta/cobot_magic_conversion.json"
RAW_LOCAL="${COBOT_LOCAL_RAW_ROOT:-/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/raw_datasets/cobot_magic_il-fake}"
LOCAL_REPEATS="${COBOT_LOCAL_CORRECTION_REPEATS:-8}"
JOINT_NAME="${COBOT_DATASET_NAME:-cobot_magic_il2_base66_localcorrection7x${LOCAL_REPEATS}_chunk10}"
EEF_NAME="${JOINT_NAME}_eef"
JOINT_ROOT="${DATASETS_ROOT}/${JOINT_NAME}"
EEF_ROOT="${DATASETS_ROOT}/${EEF_NAME}"
TASK="Pick up the metal workpiece, align it with the matching fixture, and place it securely."

for required in "${SOURCE_MANIFEST}" "${RAW_LOCAL}" "${LEROBOT_ROOT}/pink_ik/robot/robot_descriptions/manipulator/AgileX/piper_description/urdf/piper.urdf"; do
    [[ -e "${required}" ]] || { echo "Required input is missing: ${required}" >&2; exit 1; }
done
[[ ! -e "${JOINT_ROOT}" && ! -e "${EEF_ROOT}" ]] || {
    echo "Refusing to overwrite existing output: ${JOINT_ROOT} or ${EEF_ROOT}" >&2
    exit 1
}
[[ "${LOCAL_REPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "COBOT_LOCAL_CORRECTION_REPEATS must be positive" >&2; exit 2; }

mapfile -t BASE_RAW < <("${PYTHON}" - "${SOURCE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for episode in manifest["episodes"][4:]:
    print(episode["raw"])
PY
)
mapfile -t LOCAL_RAW_ABS < <(find "${RAW_LOCAL}" -maxdepth 1 -type f -name '*.raw.h5' -printf '%f\n' | sort)

[[ "${#BASE_RAW[@]}" -eq 66 ]] || { echo "Expected 66 base episodes, got ${#BASE_RAW[@]}" >&2; exit 1; }
[[ "${#LOCAL_RAW_ABS[@]}" -eq 7 ]] || { echo "Expected 7 valid local correction episodes, got ${#LOCAL_RAW_ABS[@]}" >&2; exit 1; }

RAW_FILES=("${BASE_RAW[@]}")
for ((repeat = 0; repeat < LOCAL_REPEATS; repeat++)); do
    for filename in "${LOCAL_RAW_ABS[@]}"; do
        RAW_FILES+=("${RAW_LOCAL}/${filename}")
    done
done
EXPECTED=$((66 + 7 * LOCAL_REPEATS))
[[ "${#RAW_FILES[@]}" -eq "${EXPECTED}" ]] || { echo "unexpected input count" >&2; exit 1; }

export PATH="$(dirname "${PYTHON}"):${PATH}"
cd "${LEROBOT_ROOT}"

"${PYTHON}" -m cobot_bridge.convert_raw_to_lerobot \
    --raw "${RAW_FILES[@]}" \
    --repo-id "local/${JOINT_NAME}" \
    --root "${JOINT_ROOT}" \
    --task "${TASK}" \
    --fps 30 \
    --action-source executed_action \
    --trim-stationary-holds \
    --trim-joint-tolerance 0.01 \
    --trim-gripper-tolerance 0.002 \
    --trim-settle-seconds 0 \
    --use-videos \
    --vcodec h264 \
    --image-writer-threads 4

"${PYTHON}" -m cobot_bridge.convert_lerobot_joint_to_eef \
    --datasets-root "${DATASETS_ROOT}" \
    --source "${JOINT_ROOT}" \
    --mode absolute

"${PYTHON}" scripts/compute_relative_eef_stats.py \
    --root "${EEF_ROOT}" \
    --chunk-size 10 \
    --output "${EEF_ROOT}/meta/relative_eef_action_stats_chunk10.json"

echo "Built ${EEF_ROOT} with ${EXPECTED} episodes (66 base + 7 local x ${LOCAL_REPEATS})."
