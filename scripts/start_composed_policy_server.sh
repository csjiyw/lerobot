#!/usr/bin/env bash
# Start the composed policy server: 60k ref + z encoder + residual actor.
set -euo pipefail
TASK_ROOT="/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 COBOT_PYTHON=/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python
cd "${TASK_ROOT}"
exec /home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python -m cobot_bridge.policy_server \
  --bind tcp://127.0.0.1:5560 --cameras front right --device cuda:0 \
  --policy-path outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model \
  --residual-actor outputs/actor_v1 --z-encoder outputs/z_stage1_v1
