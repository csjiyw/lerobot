#!/usr/bin/env bash
set -Eeuo pipefail

# ROS-only master/puppet stack. The default is dry-run and cannot publish arm
# commands. Set COBOT_LIVE_OUTPUT=true only for an explicitly supervised test.

LEROBOT_ROOT="/home/agilex/lerobot"
PIPER_ROOT="/home/agilex/piper_ros"
ROS_SETUP="/opt/ros/humble/setup.bash"
PIPER_SETUP="${PIPER_ROOT}/install/setup.bash"
CAN_CONFIG="${LEROBOT_ROOT}/scripts/configure_cobot_magic_can.sh"
LAUNCH_FILE="${LEROBOT_ROOT}/scripts/cobot_magic_master_puppet.launch.py"
LIVE_OUTPUT="${COBOT_LIVE_OUTPUT:-false}"

source "${ROS_SETUP}"
source "${PIPER_SETUP}"
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1

sudo -v
CONFIGURE_BASE_CAN=0 bash "${CAN_CONFIG}"

exec ros2 launch "${LAUNCH_FILE}" live_output:="${LIVE_OUTPUT}"
