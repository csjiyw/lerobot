#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
DATASETS="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/datasets"
MANIFESTS="${DATASETS}/cobot_magic_steam_manifests_v1"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"

cd "${WORKSPACE}"

convert_split() {
  local split="$1"
  local expected="$2"
  local type="$3"
  local output="${DATASETS}/cobot_magic_steam_${split}_v1"
  local manifest="${MANIFESTS}/${split}.txt"
  mapfile -t episodes < "${manifest}"
  if [[ "${#episodes[@]}" -ne "${expected}" ]]; then
    echo "${split}: expected ${expected} episodes, found ${#episodes[@]}" >&2
    return 2
  fi
  if [[ -e "${output}" ]]; then
    echo "${split}: refusing to overwrite ${output}" >&2
    return 3
  fi
  echo "CONVERT_START ${split} episodes=${#episodes[@]}"
  "${PYTHON}" -m cobot_bridge.convert_raw_to_lerobot \
    --raw "${episodes[@]}" \
    --repo-id "local/cobot_magic_steam_${split}_v1" \
    --root "${output}" \
    --task 'pick and place the cross object' \
    --action-source executed_action \
    --use-videos \
    --vcodec h264 \
    --image-writer-threads 12
  echo "CONVERT_DONE ${split} type=${type}"
}

convert_split il_train 64 sft
convert_split il_eval 15 sft
convert_split rollout_train 125 rollout
convert_split rollout_eval 30 rollout
