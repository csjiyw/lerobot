from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cobot_bridge.replay_raw import (
    TeachEpisodeComplete,
    establish_sync_mode,
    longest_valid_run,
    queue_sync_chunks,
    trim_stationary_holds,
)


def test_longest_valid_run_discards_safe_wait_and_stale_prefix():
    valid = np.asarray([1, 1, 0, 0, 1, 1, 1, 1, 0], dtype=bool)

    selected = longest_valid_run(valid)

    assert selected == slice(4, 8)


def test_longest_valid_run_rejects_missing_command_stream():
    with pytest.raises(ValueError, match="no valid puppet_command"):
        longest_valid_run(np.zeros(10, dtype=bool))


def test_trim_stationary_holds_preserves_settle_around_motion():
    timestamps = np.arange(0.0, 2.1, 0.1)
    trajectory = np.zeros((len(timestamps), 7), dtype=np.float64)
    trajectory[6:11, 0] = np.linspace(0.0, 1.0, 5)
    trajectory[11:, 0] = 1.0

    clipped_time, clipped, info = trim_stationary_holds(
        timestamps,
        trajectory,
        threshold=1e-4,
        settle_seconds=0.2,
    )

    assert info["trimmed"] is True
    assert info["start_hold_removed_s"] == 0.4
    assert np.isclose(info["end_hold_removed_s"], 0.8)
    assert clipped.shape == (9, 7)
    assert clipped_time[0] == 0.0
    assert np.isclose(clipped_time[-1], 0.8)
    assert clipped[0, 0] == 0.0
    assert clipped[-1, 0] == 1.0


def test_trim_stationary_holds_leaves_stationary_episode_unchanged():
    timestamps = np.arange(0.0, 1.1, 0.1)
    trajectory = np.zeros((len(timestamps), 7), dtype=np.float64)

    clipped_time, clipped, info = trim_stationary_holds(
        timestamps,
        trajectory,
        threshold=1e-4,
        settle_seconds=0.2,
    )

    assert info["trimmed"] is False
    np.testing.assert_array_equal(clipped_time, timestamps)
    np.testing.assert_array_equal(clipped, trajectory)


def test_teach_release_terminates_policy_without_resync():
    observations = iter(
        (
            SimpleNamespace(teach_active=True, effective_mode=3, control={"epoch": 8}),
            SimpleNamespace(teach_active=False, effective_mode=4, control={"epoch": 9}),
        )
    )

    class Robot:
        def observe(self):
            return next(observations)

    class Client:
        def resync(self):
            raise AssertionError("Teach release must not call resync")

    with pytest.raises(TeachEpisodeComplete, match="Teach released"):
        establish_sync_mode(
            Client(),
            Robot(),
            np.zeros(7),
            alignment_timeout=1.0,
            initial_request=False,
        )


def test_sync_alignment_fills_missing_side_with_stationary_master_hold(monkeypatch):
    monkeypatch.setattr("cobot_bridge.replay_raw.time.sleep", lambda _seconds: None)
    left_master = np.arange(7, dtype=np.float32) * 0.01

    class Robot:
        side = "right"

        def __init__(self):
            self.mode = 5
            self.output_enabled = False
            self.targets = []

        def request_mode(self, mode):
            assert mode == 2
            return {"accepted": True}

        def set_output(self, enabled):
            self.output_enabled = enabled
            return {"accepted": True}

        def observe(self):
            return SimpleNamespace(
                teach_active=False,
                effective_mode=self.mode,
                control={"output_enabled": self.output_enabled},
                position=lambda role, side: left_master.copy(),
            )

        def send_alignment_target(self, action, *, observation):
            self.targets.append(("right", list(action)))
            return {
                "accepted": True,
                "reason": "sync_first_target_buffered",
                "waiting_sides": ["left"],
            }

        def send_alignment_target_for_side(self, side, action, *, observation):
            self.targets.append((side, list(action)))
            self.mode = 2
            return {
                "accepted": True,
                "reason": "individual_alignment_started",
                "duration": 1.0,
                "maximum_delta": 0.2,
                "epoch": 4,
            }

    robot = Robot()
    current, alignment, holds = establish_sync_mode(
        object(),
        robot,
        np.zeros(7),
        alignment_timeout=1.0,
        initial_request=True,
    )

    assert current.effective_mode == 2
    assert alignment["reason"] == "individual_alignment_started"
    assert robot.targets == [("right", [0.0] * 7), ("left", left_master.tolist())]
    np.testing.assert_array_equal(holds["left"], left_master)


def test_sync_chunks_keep_auxiliary_side_stationary(monkeypatch):
    sequences = iter((100, 101))
    monkeypatch.setattr("cobot_bridge.replay_raw.time.time_ns", lambda: next(sequences))

    class Client:
        def __init__(self):
            self.calls = []

        def set_action_chunk(self, side, actions, **kwargs):
            self.calls.append((side, np.asarray(actions), kwargs))
            return {"accepted": True, "duration_s": 0.1, "side": side}

    client = Client()
    moving = np.arange(21, dtype=np.float64).reshape(3, 7)
    left_hold = np.arange(7, dtype=np.float64) * 0.01
    selected, holds = queue_sync_chunks(
        client,
        selected_side="right",
        chunk=moving,
        auxiliary_holds={"left": left_hold},
        epoch=8,
        waypoint_hz=20.0,
    )

    assert selected["side"] == "right"
    assert holds["left"]["side"] == "left"
    assert [call[0] for call in client.calls] == ["left", "right"]
    np.testing.assert_array_equal(client.calls[0][1], np.repeat(left_hold[None, :], 3, axis=0))
    np.testing.assert_array_equal(client.calls[1][1], moving)
    assert client.calls[0][2]["sequence"] == 100
    assert client.calls[1][2]["sequence"] == 101
