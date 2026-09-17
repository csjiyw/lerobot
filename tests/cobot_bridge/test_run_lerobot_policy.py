from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cobot_bridge.run_lerobot_policy import (
    CumulativeControlTickScheduler,
    INIT_POSE,
    _hold_startup_init,
    eef_target_from_delta,
    resolve_action_steps,
)
from cobot_bridge.remote_robot import ALIGN_SYNC, POLICY_PUPPET, RLT_SYNC


class FakeRobot:
    def __init__(
        self,
        state: np.ndarray,
        *,
        mode: int = POLICY_PUPPET,
        master_state: np.ndarray | None = None,
    ):
        self.state = np.asarray(state, dtype=np.float32)
        self.master_state = np.asarray(
            state if master_state is None else master_state, dtype=np.float32
        )
        self.mode = mode
        self.side = "right"
        self.actions: list[list[float]] = []
        self.side_actions: list[tuple[str, list[float]]] = []
        self.alignment_targets: list[tuple[str, list[float]]] = []
        self.reject_first_action_as_stale = False
    def observe(self):
        states = {
            "left": self.master_state.copy(),
            "right": self.master_state.copy(),
        }
        return SimpleNamespace(
            puppet_state=self.state.copy(),
            master_state=self.master_state.copy(),
            effective_mode=self.mode,
            teach_active=False,
            position=lambda role, side: states[side].copy(),
        )

    def send_action(self, action, *, observation):
        self.actions.append(list(action))
        if self.reject_first_action_as_stale:
            self.reject_first_action_as_stale = False
            return {"accepted": False, "reason": "stale_epoch"}
        self.state = np.asarray(action, dtype=np.float32)
        return {"accepted": True}

    def send_action_for_side(self, side, action, *, observation):
        self.side_actions.append((side, list(action)))
        return {"accepted": True}

    def send_alignment_target(self, action, *, observation):
        self.alignment_targets.append((self.side, list(action)))
        self.state = np.asarray(action, dtype=np.float32)
        self.master_state = np.asarray(action, dtype=np.float32)
        self.mode = RLT_SYNC
        return {"accepted": True}

    def send_alignment_target_for_side(self, side, action, *, observation):
        self.alignment_targets.append((side, list(action)))
        return {"accepted": True}


def test_action_steps_default_to_complete_model_chunk():
    assert resolve_action_steps(None, 30) == 30


def test_action_steps_are_clamped_to_complete_model_chunk():
    assert resolve_action_steps(50, 30) == 30


def test_action_steps_preserve_shorter_closed_loop_horizon():
    assert resolve_action_steps(5, 30) == 5


def test_action_steps_reject_empty_model_chunk():
    with pytest.raises(RuntimeError, match="empty action chunk"):
        resolve_action_steps(None, 0)


def test_eef_delta_is_applied_to_measured_pose_but_gripper_stays_absolute():
    state = np.asarray([0.3, 0.1, 0.12, 0.1, -0.2, 0.3, 0.08], dtype=np.float32)
    delta = np.asarray([0.001, -0.002, 0.003, 0.01, 0.02, -0.03, 0.0], dtype=np.float32)
    target = eef_target_from_delta(state, delta)
    np.testing.assert_allclose(target[:3], state[:3] + delta[:3])
    assert target[6] == 0.0


def test_eef_relative_rotation_delta_round_trip():
    from cobot_bridge.piper_eef_transform import eef_pose_delta

    state = np.asarray([0.3, 0.1, 0.12, 0.7, -0.4, 0.2, 0.08], dtype=np.float32)
    target = np.asarray([0.31, 0.08, 0.15, -0.2, 0.6, 0.5, 0.01], dtype=np.float32)
    delta = eef_pose_delta(state, target)
    reconstructed = eef_target_from_delta(state, delta)
    np.testing.assert_allclose(reconstructed, target, atol=1e-6)


def test_pink_tick_scheduler_preserves_long_run_200hz_timing():
    scheduler = CumulativeControlTickScheduler(policy_hz=30.0)
    ticks = [scheduler.next() for _ in range(30)]

    assert set(ticks) <= {6, 7}
    assert sum(ticks) == 200


def test_startup_init_hold_sends_open_gripper_target():
    robot = FakeRobot(INIT_POSE.copy())
    robot.state[6] = 0.0

    _hold_startup_init(robot)

    assert robot.actions == [INIT_POSE.tolist()]
    assert robot.alignment_targets == []


def test_startup_init_hold_refreshes_once_after_stale_epoch():
    robot = FakeRobot(INIT_POSE.copy())
    robot.reject_first_action_as_stale = True

    _hold_startup_init(robot)

    assert robot.actions == [INIT_POSE.tolist(), INIT_POSE.tolist()]


def test_mode_2_startup_uses_alignment_target_then_fresh_hold_action():
    puppet_state = INIT_POSE.copy()
    puppet_state[6] = 0.0
    master_state = INIT_POSE.copy()
    master_state[0] = 0.03
    robot = FakeRobot(
        puppet_state,
        mode=ALIGN_SYNC,
        master_state=master_state,
    )

    _hold_startup_init(robot)

    assert robot.alignment_targets == [
        ("right", INIT_POSE.tolist()),
        ("left", master_state.tolist()),
    ]
    assert robot.actions == [INIT_POSE.tolist()]
    assert robot.side_actions == [("left", master_state.tolist())]
    assert robot.mode == RLT_SYNC


def test_startup_init_hold_refuses_large_arm_jump():
    state = INIT_POSE.copy()
    state[1] = 0.2
    robot = FakeRobot(state)

    with pytest.raises(RuntimeError, match="refused"):
        _hold_startup_init(robot)

    assert robot.actions == []
