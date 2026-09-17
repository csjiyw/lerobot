#!/usr/bin/env bash
# Configure the five Cobot Magic CAN adapters by their current USB topology.
set -euo pipefail

PIPER_CAN_ACTIVATE="/home/agilex/piper_ros/can_activate.sh"

if [[ ! -x "${PIPER_CAN_ACTIVATE}" ]]; then
    echo "Missing executable: ${PIPER_CAN_ACTIVATE}" >&2
    exit 1
fi

# Four arms at 1 Mbps. This mapping was re-verified after moving all four CAN
# adapters to the new USB hub on 2026-08-07. The Candlelight serial numbers and
# the PCAN disconnect/reconnect order preserve the original arm roles.
bash "${PIPER_CAN_ACTIVATE}" can_left_m 1000000 "1-1.1.4:1.0"
bash "${PIPER_CAN_ACTIVATE}" can_left_s 1000000 "1-1.1.3:1.0"
bash "${PIPER_CAN_ACTIVATE}" can_right_m 1000000 "1-1.1.1:1.0"
bash "${PIPER_CAN_ACTIVATE}" can_right_s 1000000 "1-1.1.2:1.0"

# Chassis at 500 kbps. It remains untouched unless explicitly requested.
if [[ "${CONFIGURE_BASE_CAN:-0}" == "1" ]]; then
    bash "${PIPER_CAN_ACTIVATE}" can_base 500000 "1-4:1.0"
else
    echo "Base CAN remains disabled (set CONFIGURE_BASE_CAN=1 to configure it)."
fi

echo "Current CAN interfaces:"
ip -br link show type can
