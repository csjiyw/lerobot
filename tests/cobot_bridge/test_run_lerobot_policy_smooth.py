import math

import numpy as np
import pytest

from cobot_bridge.run_lerobot_policy_smooth import (
    JerkLimitedTargetFilter,
    TargetLowPass,
    TemporalEnsembleBuffer,
    TimedActionChunk,
    smooth_action_chunk,
)


def _chunk(sequence: int, observed_at: float, actions: np.ndarray) -> TimedActionChunk:
    return TimedActionChunk(
        sequence=sequence,
        observed_at=observed_at,
        ready_at=observed_at + 0.1,
        actions=actions,
    )


def test_temporal_ensemble_interpolates_at_wall_clock_phase():
    buffer = TemporalEnsembleBuffer(action_hz=10.0, decay=2.0)
    actions = np.repeat(np.arange(4, dtype=np.float64)[:, None], 7, axis=1)
    buffer.add(_chunk(0, 10.0, actions))

    action, diagnostics = buffer.action(10.15)

    np.testing.assert_allclose(action, np.full(7, 1.5))
    assert diagnostics["active_chunks"] == 1
    assert diagnostics["sequences"] == [0]
    assert diagnostics["proposals"][0]["phase"] == pytest.approx(1.5)
    assert diagnostics["proposals"][0]["weight"] == 1.0


def test_temporal_ensemble_blends_overlapping_predictions():
    buffer = TemporalEnsembleBuffer(action_hz=10.0, decay=0.0)
    buffer.add(_chunk(0, 10.0, np.zeros((5, 7))))
    buffer.add(_chunk(1, 10.1, np.full((5, 7), 2.0)))

    action, diagnostics = buffer.action(10.2)

    np.testing.assert_allclose(action, np.ones(7))
    assert diagnostics["active_chunks"] == 2
    assert diagnostics["sequences"] == [0, 1]
    np.testing.assert_allclose(
        [proposal["weight"] for proposal in diagnostics["proposals"]], [0.5, 0.5]
    )


def test_temporal_ensemble_discards_expired_chunks():
    buffer = TemporalEnsembleBuffer(action_hz=10.0, decay=0.0)
    buffer.add(_chunk(0, 10.0, np.zeros((3, 7))))

    action, diagnostics = buffer.action(10.31)

    assert action is None
    assert diagnostics["active_chunks"] == 0


def test_target_low_pass_starts_from_measured_position():
    low_pass = TargetLowPass(np.zeros(7), tau=1.0, now=10.0)

    filtered = low_pass.update(np.ones(7), 11.0)

    np.testing.assert_allclose(filtered, np.full(7, 1.0 - math.exp(-1.0)))


def test_chunk_smoothing_reduces_arm_noise_without_changing_gripper():
    actions = np.zeros((3, 7), dtype=np.float64)
    actions[:, 0] = [0.0, 1.0, 0.0]
    actions[:, 6] = [0.0, 1.0, 0.0]

    smoothed = smooth_action_chunk(actions, passes=1)

    np.testing.assert_allclose(smoothed[:, 0], [0.25, 0.5, 0.25])
    np.testing.assert_allclose(smoothed[:, 6], actions[:, 6])


def test_consistency_weight_rejects_an_outlier_chunk():
    buffer = TemporalEnsembleBuffer(
        action_hz=10.0, decay=0.0, consistency_scale=0.1
    )
    buffer.add(_chunk(0, 10.0, np.zeros((4, 7))))
    buffer.add(_chunk(1, 10.0, np.full((4, 7), 0.05)))
    buffer.add(_chunk(2, 10.0, np.full((4, 7), 2.0)))

    action, diagnostics = buffer.action(10.1)

    assert action[0] < 0.1
    assert diagnostics["proposals"][2]["weight"] < 0.01


def test_jerk_limited_target_filter_respects_all_three_limits():
    limiter = JerkLimitedTargetFilter(
        np.zeros(7),
        max_velocity=np.full(7, 0.4),
        max_acceleration=np.full(7, 0.8),
        max_jerk=np.full(7, 4.0),
        now=0.0,
    )
    accelerations = []
    velocities = []
    for step in range(1, 101):
        limiter.update(np.ones(7), step * 0.01)
        accelerations.append(limiter.acceleration.copy())
        velocities.append(limiter.velocity.copy())

    accelerations = np.asarray(accelerations)
    velocities = np.asarray(velocities)
    assert np.max(np.abs(velocities)) <= 0.4 + 1e-12
    assert np.max(np.abs(accelerations)) <= 0.8 + 1e-12
    assert np.max(np.abs(np.diff(accelerations, axis=0) / 0.01)) <= 4.0 + 1e-9
