from __future__ import annotations

import sys

import numpy as np

from cobot_bridge import move_init_pose
from cobot_bridge.move_init_pose import INIT_POSE, smoothstep_chunk


def test_init_pose_matches_training_episode_start_target():
    np.testing.assert_allclose(
        INIT_POSE, [0.0670, 0.0040, -0.3450, -0.1100, 0.5710, -0.1280, 0.0]
    )


def test_smoothstep_chunk_starts_at_feedback_and_ends_at_init():
    start = [0.2, -0.1, 0.3, -0.2, 0.1, 0.4, 0.03]
    chunk = np.asarray(smoothstep_chunk(start, duration=2.0, waypoint_hz=20.0))

    assert chunk.shape == (41, 7)
    np.testing.assert_allclose(chunk[0], start)
    np.testing.assert_allclose(chunk[-1], INIT_POSE)
    assert np.all(np.diff(chunk[:, 0]) <= 0)
    assert np.all(np.diff(chunk[:, 1]) >= 0)


def test_execute_uses_mode_2_targets_both_sides_and_finishes_safe(monkeypatch):
    calls = []

    def observation(mode, *, output, positions):
        value = {
            "ready": True,
            "missing_or_stale": [],
            "control": {
                "teach_active": False,
                "effective_mode_id": mode,
                "output_enabled": output,
                "epoch": 7,
            },
        }
        for role in ("master", "puppet"):
            value[role] = {
                side: {"position": list(positions)} for side in ("left", "right")
            }
        return value

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.observation_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_observation(self, **_kwargs):
            self.observation_count += 1
            if self.observation_count == 1:
                return observation(0, output=False, positions=[0.1] * 7), {}
            if self.observation_count == 2:
                return observation(5, output=True, positions=[0.1] * 7), {}
            return observation(2, output=True, positions=INIT_POSE), {}

        def set_mode(self, mode):
            calls.append(("mode", mode))
            return {"accepted": True}

        def set_output(self, enabled):
            calls.append(("output", enabled))
            return {"accepted": True}

        def set_action(self, side, action, *, epoch, sequence, alignment_speed=None):
            calls.append(("action", side, list(action), epoch, sequence, alignment_speed))
            return {"accepted": True, "duration": 0.1}

    monkeypatch.setattr(move_init_pose, "CobotBridgeClient", FakeClient)
    monkeypatch.setattr(move_init_pose.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(move_init_pose, "STOP", False)
    monkeypatch.setattr(sys, "argv", ["move_init_pose", "--execute"])

    move_init_pose.main()

    assert calls[0:2] == [("mode", 2), ("output", True)]
    action_calls = [call for call in calls if call[0] == "action"]
    assert [call[1] for call in action_calls] == ["left", "right"]
    assert all(
        call[2] == INIT_POSE.tolist() and call[3:] == (7, 1, 0.48)
        for call in action_calls
    )
    assert calls[-2:] == [("output", False), ("mode", 0)]
