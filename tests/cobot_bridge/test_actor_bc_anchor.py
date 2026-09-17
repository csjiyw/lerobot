from cobot_bridge.build_actor_bc_anchor_from_raw import bc_window_starts


def test_bc_window_starts_keeps_a_single_complete_ten_step_clip():
    assert bc_window_starts(15, 10, 10).tolist() == [0]
    assert bc_window_starts(20, 10, 10).tolist() == [0, 10]
