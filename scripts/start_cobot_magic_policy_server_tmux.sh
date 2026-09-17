#!/usr/bin/env bash
set -Eeuo pipefail

LEROBOT_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
PYTHON="/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python"
POLICY_PATH="${LEROBOT_ROOT}/outputs/cobot_magic_smolvla_il_001_200k_bs64_zellij/checkpoints/150000/pretrained_model"
SESSION="${COBOT_POLICY_SESSION:-cobot_policy_150k}"
BIND="${COBOT_POLICY_BIND:-tcp://127.0.0.1:5560}"
LOG_PATH="${LEROBOT_ROOT}/outputs/cobot_policy_150k_server.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Policy server tmux session already exists: ${SESSION}"
    exit 0
fi

command=(
    "${PYTHON}" -m cobot_bridge.policy_server
    --bind "${BIND}"
    --policy-path "${POLICY_PATH}"
    --device cuda:0
    --cameras front right
)
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" \
    "cd '${LEROBOT_ROOT}' && exec bash -lc '${quoted}2>&1 | tee -a \"${LOG_PATH}\"'"

echo "Started ${SESSION}; attach with: tmux attach -t ${SESSION}"
echo "Policy endpoint: ${BIND}"
echo "Log: ${LOG_PATH}"
