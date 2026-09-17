from __future__ import annotations

import numpy as np

from cobot_bridge.eef_trajectory_limiter import (
    CartesianLimits,
    CartesianTrajectoryLimiter,
    blend_eef_chunk_edge,
)


def test_cartesian_limiter_binds_linear_velocity_and_converges():
    limits = CartesianLimits(control_hz=200.0, max_linear_velocity=0.2, max_linear_acceleration=1.0)
    limiter = CartesianTrajectoryLimiter(limits)
    limiter.reset(np.zeros(7))
    target = np.asarray([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08])
    path = np.asarray(limiter.advance(target, 200))
    velocity = np.diff(path[:, 0]) * limits.control_hz
    assert np.max(np.abs(velocity)) <= limits.max_linear_velocity + 1e-5
    assert path[-1, 0] > 0.09
    assert path[-1, 0] <= target[0] + 1e-6


def test_cartesian_limiter_uses_short_se3_rotation_path():
    limiter = CartesianTrajectoryLimiter(CartesianLimits(max_angular_velocity=1.0))
    limiter.reset(np.zeros(7))
    target = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0])
    sample = limiter.advance(target, 1)[0]
    assert 0.0 < sample[5] < target[5]


def test_chunk_edge_blends_in_se3_and_uses_dynamic_length():
    previous = np.zeros(7, dtype=np.float32)
    chunk = np.tile(np.asarray([0.04, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.08], dtype=np.float32), (30, 1))
    blended, steps = blend_eef_chunk_edge(previous, chunk, policy_hz=30.0)
    assert steps > 1
    assert 0.0 < blended[0, 0] < chunk[0, 0]
    assert 0.0 < blended[0, 5] < chunk[0, 5]
    np.testing.assert_allclose(blended[-1], chunk[-1])
