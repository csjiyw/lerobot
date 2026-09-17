from argparse import Namespace
from pathlib import PurePosixPath

import pytest

from cobot_bridge.remote_rollout import _tmux_command


def _args(**overrides):
    values = {
        "ssh_host": "agilex@192.168.250.2",
        "remote_workspace": PurePosixPath("/home/agilex/smolvla_runtime"),
        "remote_python": PurePosixPath("/env/bin/python"),
        "session": "cobot_rollout_test123",
        "endpoint": "tcp://192.168.250.2:5555",
        "policy_server_endpoint": "tcp://127.0.0.1:5560",
        "task": "pick and place the object",
        "side": "right",
        "fps": 15.0,
        "duration": 60.0,
        "mode": 2,
        "runner": "standard",
        "n_action_steps": 5,
        "cameras": ["front", "right"],
        "send_actions": True,
        "startup_init_hold": True,
        "diagnostic_output": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_remote_rollout_uses_robot_local_policy_endpoint_and_exact_tmux_session():
    start, attach = _tmux_command(_args())
    assert "cobot_rollout_test123" in start
    assert "tcp://127.0.0.1:5560" in start
    assert "--send-actions" in start
    assert "--startup-init-hold" in start
    assert "attach-session" in attach


def test_remote_rollout_omits_action_limit_to_consume_complete_model_chunk():
    start, _ = _tmux_command(_args(n_action_steps=None))

    assert "--n-action-steps" not in start


def test_remote_rollout_can_select_smooth_runner():
    start, _ = _tmux_command(
        _args(
            runner="smooth_v2",
            diagnostic_output=PurePosixPath("/runtime/diagnostics/test.jsonl"),
        )
    )

    assert "cobot_bridge.run_lerobot_policy_smooth" in start
    assert "--diagnostic-output /runtime/diagnostics/test.jsonl" in start


def test_remote_rollout_can_select_smooth_v3_runner():
    start, _ = _tmux_command(_args(runner="smooth_v3"))

    assert "cobot_bridge.run_lerobot_policy_smooth_v3" in start


def test_remote_rollout_can_select_edge_v4_runner():
    start, _ = _tmux_command(_args(runner="edge_v4"))

    assert "cobot_bridge.run_lerobot_policy_edge_v4" in start


def test_remote_rollout_rejects_unsafe_session_name():
    with pytest.raises(ValueError):
        _tmux_command(_args(session="bad session; rm"))
