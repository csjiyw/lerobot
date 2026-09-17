import numpy as np

from cobot_bridge.build_il_demo_replay import decision_rows, split_episode_ids


def test_split_episode_ids_matches_actor_bc_holdout() -> None:
    train, validation = split_episode_ids(
        np.repeat(np.arange(79), 2), seed=42, val_fraction=0.1
    )
    assert validation.size == 8
    assert train.size == 71
    assert set(train).isdisjoint(set(validation))
    np.testing.assert_array_equal(validation, [33, 50, 40, 28, 60, 25, 0, 59])


def test_decision_rows_keeps_episode_terminal_state() -> None:
    selected = decision_rows(np.arange(95), stride=30)
    np.testing.assert_array_equal(selected, [0, 30, 60, 90, 94])
