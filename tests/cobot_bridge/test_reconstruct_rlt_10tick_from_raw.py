import numpy as np

from cobot_bridge.reconstruct_rlt_10tick_from_raw import ten_tick_starts


def test_ten_tick_starts_has_exact_action_horizon_before_next_observation():
    np.testing.assert_array_equal(ten_tick_starts(31, 10), [0, 10, 20])
    np.testing.assert_array_equal(ten_tick_starts(20, 10), [0])
    np.testing.assert_array_equal(ten_tick_starts(10, 10), [])
