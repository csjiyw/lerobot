import numpy as np

from cobot_bridge.reconstruct_rlt_10tick_from_raw import (
    aligned_starts,
    execution_frame_span,
    phase_resample,
    phase_resample_mask,
    ten_tick_starts,
)
from cobot_bridge.rlt_online import resample_action_chunk


def test_ten_tick_starts_has_exact_action_horizon_before_next_observation():
    np.testing.assert_array_equal(ten_tick_starts(31, 10), [0, 10, 20])
    np.testing.assert_array_equal(ten_tick_starts(20, 10), [0])
    np.testing.assert_array_equal(ten_tick_starts(10, 10), [])


def test_half_speed_uses_sixty_raw_frames_for_thirty_model_actions():
    assert execution_frame_span(30, 0.5) == 60
    np.testing.assert_array_equal(aligned_starts(181, 30, 0.5), [0, 60, 120])
    np.testing.assert_array_equal(aligned_starts(120, 30, 0.5), [0])


def test_half_speed_short_actor_replans_every_twenty_raw_frames():
    assert execution_frame_span(10, 0.5) == 20
    np.testing.assert_array_equal(aligned_starts(61, 10, 0.5), [0, 20, 40])


def test_phase_resample_approximately_inverts_deployment_retiming():
    rng = np.random.default_rng(4)
    model_chunk = np.cumsum(rng.normal(0, 0.01, (30, 7)), axis=0).astype(np.float32)
    physical_chunk = resample_action_chunk(model_chunk, 0.5)

    recovered = phase_resample(physical_chunk, 30)

    np.testing.assert_allclose(recovered, model_chunk, atol=2e-4)


def test_human_ownership_is_resampled_per_timestep():
    raw_mask = np.zeros(60, dtype=bool)
    raw_mask[30:] = True
    model_mask = phase_resample_mask(raw_mask, 30)
    assert model_mask[:15].sum() == 0
    assert model_mask[15:].all()
