import numpy as np

from cobot_bridge.rlt_online import (
    build_parser,
    resample_action_chunk,
    resample_chunk_to_output_hz,
)


def test_half_speed_resamples_ten_actions_to_twenty_and_preserves_endpoints():
    actions = np.linspace(np.zeros(7), np.arange(7, dtype=np.float32), 10)

    slowed = resample_action_chunk(actions, 0.5)

    assert slowed.shape == (20, 7)
    np.testing.assert_allclose(slowed[0], actions[0])
    np.testing.assert_allclose(slowed[-1], actions[-1])
    assert np.all(np.diff(slowed[:, 5]) >= 0)


def test_full_speed_keeps_original_chunk():
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)
    np.testing.assert_array_equal(resample_action_chunk(actions, 1.0), actions)


def test_policy_chunk_is_expanded_to_uniform_200_hz_grid():
    actions = np.linspace(np.zeros(7), np.ones(7), 10)

    expanded = resample_chunk_to_output_hz(
        actions, source_hz=30.0, output_hz=200.0
    )

    assert expanded.shape == (61, 7)
    np.testing.assert_allclose(expanded[0], actions[0])
    np.testing.assert_allclose(expanded[-1], actions[-1])


def test_legacy_replanner_switch_is_not_used_by_rlt():
    parser = build_parser()
    action = parser._option_string_actions["--replan-actions"]
    assert action.default is False


def test_rlt_exposes_gate_and_smooth_profile_controls():
    parser = build_parser()
    smooth = parser._option_string_actions["--smooth-profile"]
    assert set(smooth.choices) == {"edge_v4", "raw", "smooth_v3", "smooth_v2"}
    assert smooth.default == "edge_v4"
    gate = parser._option_string_actions["--actor-eef-gate"]
    assert gate.default is True
    assert parser.parse_args(["--stage1-checkpoint", "stage1", "--task", "x", "--workflow-mode", "1", "--rollout-phase", "actor", "--raw-output", "raw", "--decision-output", "decisions", "--replay-output", "replay", "--result-file", "result"]).actor_eef_min == (0.34, 0.05, 0.08)
