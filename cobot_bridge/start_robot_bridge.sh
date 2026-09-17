#!/usr/bin/env bash
set -Eeo pipefail

ROBOT_ROOT="/home/agilex/lerobot"
ROS_SETUP="/opt/ros/humble/setup.bash"
PIPER_SETUP="/home/agilex/piper_ros/install/setup.bash"
BRIDGE_PYTHON="/home/agilex/miniforge3/envs/robot_bridge/bin/python"
BRIDGE_BIND="${COBOT_BRIDGE_BIND:-tcp://192.168.250.2:5555}"
# 2026-09: 只读门已从 server.py 移除, --allow-control 仅为兼容保留(空操作)。
# 默认改为 true, 使启动脚本显式传参, 行为不再依赖该开关。
ALLOW_CONTROL="${COBOT_BRIDGE_ALLOW_CONTROL:-true}"

source "${ROS_SETUP}"
source "${PIPER_SETUP}"
set -u
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1

args=(--bind "${BRIDGE_BIND}")
if [[ "${ALLOW_CONTROL}" == "true" ]]; then
    args+=(--allow-control)
fi
if [[ -n "${COBOT_BRIDGE_TOKEN:-}" ]]; then
    args+=(--token "${COBOT_BRIDGE_TOKEN}")
fi

cd "${ROBOT_ROOT}"
exec "${BRIDGE_PYTHON}" -m cobot_bridge.server "${args[@]}"
