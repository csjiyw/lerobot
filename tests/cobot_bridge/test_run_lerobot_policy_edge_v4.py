import numpy as np
import pytest

from cobot_bridge.run_lerobot_policy_edge_v4 import (
    adaptive_edge_blend_steps,
    edge_blend_action,
    prefetch_trigger_step,
    smoothstep,
    time_aligned_start_step,
)
from cobot_bridge.synchronized_chunk_replanner import PolicyRateCommandLimiter


def test_smoothstep_has_zero_slope_endpoints():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == 0.5


def test_edge_blending_modifies_only_first_three_actions():
    previous = np.zeros(7)
    edge_start = np.ones(7)
    current = np.ones(7)

    first, first_alpha = edge_blend_action(
        previous, edge_start, current, step=0, blend_steps=3
    )
    third, third_alpha = edge_blend_action(
        previous, edge_start, current, step=2, blend_steps=3
    )
    fourth, fourth_alpha = edge_blend_action(
        previous, edge_start, current, step=3, blend_steps=3
    )

    assert first_alpha == 0.0
    np.testing.assert_allclose(first[:6], previous[:6])
    assert first[6] == current[6]
    assert third_alpha == 1.0
    np.testing.assert_allclose(third, current)
    assert fourth_alpha == 1.0
    np.testing.assert_allclose(fourth, current)


def test_prefetch_starts_seven_steps_before_thirty_step_edge():
    assert prefetch_trigger_step(30, 7) == 23
    assert prefetch_trigger_step(3, 5) == 0


def test_prefetched_chunk_starts_at_its_current_time_phase():
    observed_at = 10.0
    playback_at = observed_at + 7.0 / 30.0
    assert (
        time_aligned_start_step(
            observed_at, playback_at, fps=30.0, chunk_length=30
        )
        == 7
    )


def test_expired_prefetched_chunk_is_rejected():
    with pytest.raises(RuntimeError, match="expired"):
        time_aligned_start_step(10.0, 11.0, fps=30.0, chunk_length=30)


def test_edge_window_grows_with_aligned_arm_gap():
    previous = np.zeros(7)
    small = np.full(7, 0.01)
    large = np.full(7, 0.35)

    small_steps, small_gap = adaptive_edge_blend_steps(
        previous, small, min_steps=3, max_steps=6, max_joint_step=0.08
    )
    large_steps, large_gap = adaptive_edge_blend_steps(
        previous, large, min_steps=3, max_steps=6, max_joint_step=0.08
    )

    assert (small_steps, small_gap) == pytest.approx((3, 0.01))
    assert (large_steps, large_gap) == pytest.approx((6, 0.35))


def test_edge_window_can_use_policy_rate_kinematic_limits():
    previous = np.zeros(7)
    edge = np.full(7, 0.10)
    steps, _ = adaptive_edge_blend_steps(
        previous,
        edge,
        min_steps=3,
        max_steps=30,
        max_joint_step=0.08,
        action_hz=30.0,
    )

    assert steps > 3


def test_policy_rate_limiter_keeps_30_hz_command_differences_bounded():
    limiter = PolicyRateCommandLimiter(action_hz=30.0)
    limiter.reset(np.zeros(7))
    target = np.asarray([0.35] * 6 + [0.05])
    outputs = [limiter.step(target).astype(np.float64) for _ in range(100)]
    commands = np.vstack([np.zeros((1, 7)), outputs])
    velocity = np.diff(commands, axis=0) * 30.0
    acceleration = np.diff(velocity, axis=0) * 30.0

    assert np.max(np.abs(velocity) / limiter.velocity_limits) <= 1.00001
    assert np.max(np.abs(acceleration) / limiter.acceleration_limits) <= 1.00001
