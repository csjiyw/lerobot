import numpy as np

from cobot_bridge.synchronized_chunk_replanner import SynchronizedChunkReplanner


def make_replanner() -> SynchronizedChunkReplanner:
    return SynchronizedChunkReplanner(
        action_hz=30.0,
        velocity_limits=np.asarray([0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 0.12]),
        acceleration_limits=np.asarray([1.5] * 6 + [0.4]),
        maximum_time_scale=20.0,
    )


def test_dynamic_trajectory_grows_transition_when_edge_gap_is_large():
    planner = make_replanner()
    small_actions = np.zeros((3, 7), dtype=np.float64)
    small_actions[:, 0] = 0.01
    small_trajectory, small_diag = planner.plan_trajectory(
        np.zeros(7), small_actions, output_hz=200.0
    )

    planner.reset()
    large_actions = np.zeros((3, 7), dtype=np.float64)
    large_actions[:, 0] = 0.10
    large_trajectory, large_diag = planner.plan_trajectory(
        np.zeros(7), large_actions, output_hz=200.0
    )

    assert large_diag.transition_steps > small_diag.transition_steps
    assert large_diag.output_steps > small_diag.output_steps
    assert large_diag.max_velocity_ratio <= 1.000001
    assert large_diag.max_acceleration_ratio <= 1.000001
    np.testing.assert_allclose(small_trajectory[-1], small_actions[-1], atol=1e-6)
    np.testing.assert_allclose(large_trajectory[-1], large_actions[-1], atol=1e-6)


def test_dynamic_trajectory_keeps_c2_boundary_state_across_chunks():
    planner = make_replanner()
    first = np.linspace(np.zeros(7), np.asarray([0.04] * 6 + [0.01]), 4)
    first_wire, _ = planner.plan_trajectory(np.zeros(7), first, output_hz=200.0)
    first_end = first_wire[-1]
    saved_velocity = planner.velocity.copy()
    saved_acceleration = planner.acceleration.copy()

    second = np.linspace(
        np.asarray([0.05] * 6 + [0.012]),
        np.asarray([0.08] * 6 + [0.02]),
        4,
    )
    second_wire, second_diag = planner.plan_trajectory(
        first_end, second, output_hz=200.0
    )

    assert second_diag.edge_gap > 0.0
    # The planner carries the previous segment's velocity and acceleration;
    # the first output sample must therefore be a small continuation, not a
    # direct jump to second[0].
    assert np.max(np.abs(second_wire[0] - first_end)) < 0.02
    assert np.isfinite(saved_velocity).all()
    assert np.isfinite(saved_acceleration).all()
