#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
RAW_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/raw_datasets/cobot_magic_rlt"
OUTPUT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets/cobot_magic_steam_rollout_155_v1"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"

cd "${WORKSPACE}"
mapfile -t RAW_EPISODES < <(find "${RAW_ROOT}" -maxdepth 1 -type f -name '*.raw.h5' -print | sort)
if [[ "${#RAW_EPISODES[@]}" -ne 155 ]]; then
  echo "Expected 155 finalized raw episodes, found ${#RAW_EPISODES[@]}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_ROOT}" >&2
  exit 3
fi

exec "${PYTHON}" -m cobot_bridge.convert_raw_to_lerobot \
  --raw "${RAW_EPISODES[@]}" \
  --repo-id local/cobot_magic_steam_rollout_155_v1 \
  --root "${OUTPUT_ROOT}" \
  --task 'pick and place the cross object' \
  --action-source executed_action \
  --use-videos \
  --vcodec h264 \
  --image-writer-threads 12
