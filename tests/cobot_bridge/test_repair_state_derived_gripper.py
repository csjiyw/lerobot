import numpy as np

from cobot_bridge.repair_state_derived_gripper import infer_contact_command


def test_contact_hold_is_closed_without_changing_source_or_release():
    source = np.r_[np.full(20, .08), np.full(60, .0548),
                   np.full(20, .075), np.zeros(20)].astype(np.float32)
    original = source.copy()
    result, mask = infer_contact_command(source, contact_upper=.0566)
    np.testing.assert_array_equal(source, original)
    assert mask.sum() == 60
    assert np.all(result[20:80] == 0)
    np.testing.assert_array_equal(result[~mask], source[~mask])


def test_short_transits_and_partial_openings_are_not_changed():
    source = np.r_[np.full(20, .08), np.full(8, .0548),
                   np.full(40, .065), np.zeros(20)].astype(np.float32)
    result, mask = infer_contact_command(source, contact_upper=.0566)
    assert not mask.any()
    np.testing.assert_array_equal(result, source)


def test_initial_ambiguous_hold_and_opening_pause_are_not_closed():
    source = np.r_[np.full(25, .0548), np.zeros(20),
                   np.full(25, .0548), np.full(20, .08)].astype(np.float32)
    result, mask = infer_contact_command(source, contact_upper=.0566)
    assert not mask.any()
    np.testing.assert_array_equal(result, source)


def test_slow_sweep_is_not_mistaken_for_contact_hold():
    source = np.r_[np.full(20, .08), np.linspace(.056, .021, 100),
                   np.zeros(20)].astype(np.float32)
    result, mask = infer_contact_command(source, contact_upper=.0566)
    assert not mask.any()
    np.testing.assert_array_equal(result, source)
