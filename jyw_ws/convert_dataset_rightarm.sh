#!/usr/bin/env bash
set -euo pipefail

PY="python"
SCRIPT="jyw_ws/convert_h5_to_lerobotv3.abs7right.batch_merge.v2.py"

# 统一参数（你按需改）
W=224
H=224
RAW_FPS=30
STEP=1
ROBOT="agx"
WORKERS=16
EPISODES_PER_DATASET=5
NOTE="cam2_abs7right_trim_tail"

BASE_DATA_DIR="/workspace/data/downloads"

# uuid -> English instruction
declare -A TASK_TEXT=(
#   ["c9a6987478e94899942253615d297400"]="Place the white plastic box directly in front of you, with its shorter side facing you."
#   ["dd6908730ae14bf9ab77cc5994386197"]="Put the biscuits into the basket first, then open the second drawer and put the chips into the basket."
#   ["4cd1553ccf134ce6b779a93202a89cbf"]="Remove the lid of the large plastic box and place it on the left."
  # ["09c0699ffc394c89928dba055a1a7846"]="Move the metal tray from the left to the right."
#   ["e54b314b4bdd4dda9bf8effa55756dee"]="Place the plastic box within the red boundary lines."
 ["9f22b6b23ed64ced8fc8074d454e5672"]="Put the metal battery into the white box."
)

for uuid in "${!TASK_TEXT[@]}"; do
  DATA_DIR="${BASE_DATA_DIR}/${uuid}"
  TASK="${TASK_TEXT[$uuid]}"

  # repo_id：建议带上 uuid，防止同名覆盖
  # 你脚本里如果自己会拼 repo_id，也可以不传 --repo_id（看你实现）
#   REPO_ID="${HF_USER}/task_${ROBOT}_${uuid}"

  echo "=== Converting: ${uuid}"
  echo "DATA_DIR=${DATA_DIR}"
  echo "TASK=${TASK}"

  $PY "$SCRIPT" \
    --W="$W" \
    --H="$H" \
    --raw_fps="$RAW_FPS" \
    --step="$STEP" \
    --robot="$ROBOT" \
    --task_text="$TASK" \
    --data_dir="$DATA_DIR" \
    --workers="$WORKERS" \
    --episodes_per_dataset="$EPISODES_PER_DATASET" \
    --note="$NOTE" 
done

echo "ALL DONE."
