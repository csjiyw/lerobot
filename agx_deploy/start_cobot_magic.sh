#!/usr/bin/env bash
set -Eeuo pipefail

# Cobot Magic startup stack: four arms, three cameras, and a synthetic zero
# base heartbeat for the single-machine collection middleware.  The heartbeat
# never commands the arms and must be excluded from training conversion.

SESSION="${COBOT_MAGIC_SESSION:-cobot_magic}"
LEROBOT_ROOT="/home/agilex/lerobot"
PIPER_ROOT="/home/agilex/piper_ros"
CAMERA_ROOT="/home/agilex/camera_ros"
ROS_SETUP="/opt/ros/humble/setup.bash"
ROS2_WS_SETUP="/home/agilex/ros2_ws/install/setup.bash"
PIPER_SETUP="${PIPER_ROOT}/install/setup.bash"
CAN_CONFIG="${LEROBOT_ROOT}/scripts/configure_cobot_magic_can.sh"
ARM_LAUNCH="${LEROBOT_ROOT}/scripts/cobot_magic_master_puppet.launch.py"
FAKE_BASE="${LEROBOT_ROOT}/scripts/fake_base_heartbeat.py"
LIVE_OUTPUT="${COBOT_LIVE_OUTPUT:-false}"

unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1

die() {
    echo "Error: $*" >&2
    exit 1
}

command -v tmux >/dev/null 2>&1 || die "tmux is not installed."
[[ -f "${ROS_SETUP}" ]] || die "Missing ROS 2 setup: ${ROS_SETUP}"
[[ -f "${PIPER_SETUP}" ]] || die "Missing Piper ROS 2 setup: ${PIPER_SETUP}"
[[ -f "${ROS2_WS_SETUP}" ]] || die "Missing ROS 2 workspace: ${ROS2_WS_SETUP}"
[[ -f "${CAN_CONFIG}" ]] || die "Missing CAN setup script: ${CAN_CONFIG}"
[[ -f "${ARM_LAUNCH}" ]] || die "Missing four-arm launch file: ${ARM_LAUNCH}"
[[ -f "${FAKE_BASE}" ]] || die "Missing fake base heartbeat: ${FAKE_BASE}"
[[ -f "${CAMERA_ROOT}/scripts/start_orbbec_camera.sh" ]] || \
    die "Missing camera startup script."

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Tmux session '${SESSION}' already exists; attaching."
    exec tmux attach-session -t "${SESSION}"
fi

if pgrep -af '[c]obot_magic_(four_arm|master_puppet)\.launch\.py|[c]obot_magic_robot_control\.py|[s]tart_two_piper\.launch\.py|[s]tart_multi_piper\.sh|[p]iper_single_ctrl|[f]ake_base_heartbeat\.py' \
    | grep -vE '^[[:space:]]*[0-9]+ tmux new-session ' \
    | grep -q .; then
    die "Existing arm or fake-base ROS processes detected; stop them before startup."
fi

sudo -v || die "sudo authentication failed; no ROS process was started."

ARM_CMD="source ${ROS_SETUP}
source ${PIPER_SETUP}
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1
cd ${LEROBOT_ROOT}
CONFIGURE_BASE_CAN=0 bash ${CAN_CONFIG}
exec ros2 launch ${ARM_LAUNCH} live_output:=${LIVE_OUTPUT}"

CAMERA_CMD="source ${ROS_SETUP}
source ${ROS2_WS_SETUP}
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1
cd ${CAMERA_ROOT}/scripts
exec bash start_orbbec_camera.sh"

FAKE_BASE_CMD="source ${ROS_SETUP}
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1
echo 'SYNTHETIC ZERO BASE: /cmd_vel + /odom at 50 Hz'
exec python3 ${FAKE_BASE} --rate 50"

printf -v ARM_CMD_QUOTED '%q' "${ARM_CMD}"
printf -v CAMERA_CMD_QUOTED '%q' "${CAMERA_CMD}"
printf -v FAKE_BASE_CMD_QUOTED '%q' "${FAKE_BASE_CMD}"

echo "Creating Cobot Magic tmux session: ${SESSION}"
echo "Starting four arms, three cameras, and synthetic zero base heartbeat."
echo "Synthetic base topics: /cmd_vel and /odom; excluded from training semantics."
echo "Live output capability: ${LIVE_OUTPUT}; /control/output_enabled is still required."

tmux new-session -d -s "${SESSION}" -n arm \
    "bash -lc ${ARM_CMD_QUOTED}"
tmux split-window -v -t "${SESSION}:0.0" \
    "bash -lc ${CAMERA_CMD_QUOTED}"
tmux split-window -h -t "${SESSION}:0.0" \
    "bash -lc ${FAKE_BASE_CMD_QUOTED}"
tmux select-layout -t "${SESSION}:0" tiled

exec tmux attach-session -t "${SESSION}"
