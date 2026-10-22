#!/usr/bin/env bash
set -Eeuo pipefail

source /opt/ros/humble/setup.bash
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1

exec python3 /home/agilex/lerobot/scripts/fake_base_heartbeat.py --rate 50
