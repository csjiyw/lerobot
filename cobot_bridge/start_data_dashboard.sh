#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${COBOT_PYTHON:-/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python}"
WEB_HOST="${COBOT_WEB_HOST:-0.0.0.0}"
WEB_PORT="${COBOT_WEB_PORT:-8088}"

cd "${WORKSPACE}"
exec "${PYTHON_BIN}" -m cobot_bridge.web_dashboard --host "${WEB_HOST}" --port "${WEB_PORT}"
