#!/usr/bin/env bash
set -euo pipefail

# Rebuild the clean 66-episode metal-fixture IL dataset directly from raw H5.
# It intentionally excludes source episodes 0..3, preserves the one canonical
# task instruction, and uses the same start/end stationary-hold trim proven on
# cobot_magic_il_cross_073_clean_arm_state_t3_cmd_gripper_v3.

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
DATASETS_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"
SOURCE_MANIFEST="${DATASETS_ROOT}/cobot_magic_il_2_eef/meta/cobot_magic_conversion.json"
JOINT_NAME="cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3"
EEF_NAME="${JOINT_NAME}_eef"
JOINT_ROOT="${DATASETS_ROOT}/${JOINT_NAME}"
EEF_ROOT="${DATASETS_ROOT}/${EEF_NAME}"
TASK="Pick up the metal workpiece, align it with the matching fixture, and place it securely."

for required in "${SOURCE_MANIFEST}" "${LEROBOT_ROOT}/pink_ik/robot/robot_descriptions/manipulator/AgileX/piper_description/urdf/piper.urdf"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required input is missing: ${required}" >&2
        exit 1
    fi
done
if [[ -e "${JOINT_ROOT}" || -e "${EEF_ROOT}" ]]; then
    echo "Refusing to overwrite existing output: ${JOINT_ROOT} or ${EEF_ROOT}" >&2
    exit 1
fi

mapfile -t RAW_FILES < <("${PYTHON}" - "${SOURCE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for episode in manifest["episodes"][4:]:
    print(episode["raw"])
PY
)
if [[ "${#RAW_FILES[@]}" -ne 66 ]]; then
    echo "Expected 66 raw episodes after excluding source 0..3; got ${#RAW_FILES[@]}" >&2
    exit 1
fi

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

"${PYTHON}" scripts/compute_relative_eef_stats.py --root "${EEF_ROOT}" --chunk-size 30
