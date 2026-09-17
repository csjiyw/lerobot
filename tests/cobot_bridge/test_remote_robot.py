from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from cobot_bridge.piper_eef_transform import PiperEEFFrameTransform
from cobot_bridge.remote_robot import CobotMagicRemoteRobot, ControlNotReady


def _observation(mode=2, epoch=7, teach=False):
    joint = {
        "position": [0.0] * 7,
        "fresh": True,
    }
    return {
        "ready": True,
        "control": {
            "effective_mode_id": mode,
            "epoch": epoch,
            "teach_active": teach,
        },
        "master": {"left": joint},
        "puppet": {"left": joint},
        "commands": {"puppet": {"left": joint}},
        "master_eef": {
            "left": {"position": [0.056, 0.0, 0.210, 0.0, 1.5, 0.0, 0.08], "fresh": True}
        },
        "puppet_eef": {
            "left": {"position": [0.056, 0.0, 0.210, 0.0, 1.5, 0.0, 0.08], "fresh": True}
        },
        "images": {},
    }


class FakeClient:
    def __init__(self, observation):
        self.observation = observation
        self.actions = []
        self.chunks = []

    def get_observation(self, **_kwargs):
        return self.observation, {}

    def set_action(self, side, action, *, epoch, sequence, action_space="joint"):
        self.actions.append((side, action, epoch, sequence, action_space))
        return {"accepted": False, "reason": "dry_run"}

    def set_action_chunk(self, side, actions, *, epoch, sequence, waypoint_hz):
        self.chunks.append((side, actions, epoch, sequence, waypoint_hz))
        return {"accepted": False, "reason": "dry_run"}

    def set_mode(self, mode):
        return {"accepted": True, "mode": mode}

    def set_output(self, enabled):
        return {"accepted": True, "enabled": enabled}

    def resync(self):
        return {"accepted": True}


def test_remote_robot_carries_current_epoch():
    client = FakeClient(_observation())
    robot = CobotMagicRemoteRobot(client)
    observation = robot.observe()
    result = robot.send_action([0.0] * 7, observation=observation)
    assert result["reason"] == "dry_run"
    assert client.actions[0][2:4] == (7, 1)


def test_remote_robot_converts_numpy_action_to_wire_list():
    client = FakeClient(_observation())
    robot = CobotMagicRemoteRobot(client)

    robot.send_action(np.zeros(7, dtype=np.float32), observation=robot.observe())

    assert isinstance(client.actions[0][1], list)
    assert client.actions[0][1] == [0.0] * 7


def test_remote_robot_converts_model_eef_target_to_sdk_link6():
    client = FakeClient(_observation(mode=1, epoch=4))
    robot = CobotMagicRemoteRobot(client)
    observation = robot.observe()
    model_state = robot.model_eef_state(observation=observation)
    transform = PiperEEFFrameTransform()
    expected_sdk = transform.gripper_center_to_link6(model_state)

    result = robot.send_model_eef_target(model_state, observation=observation)

    assert result["reason"] == "dry_run"
    np.testing.assert_allclose(client.actions[0][1], expected_sdk, atol=1e-6)
    assert client.actions[0][4] == "eef_target"


def test_remote_robot_sends_alignment_target_in_mode_5():
    client = FakeClient(_observation(mode=5, epoch=9))
    robot = CobotMagicRemoteRobot(client)
    result = robot.send_alignment_target([0.0] * 7, observation=robot.observe())
    assert result["reason"] == "dry_run"
    assert client.actions[0][2:4] == (9, 1)


def test_remote_robot_sends_complete_chunk_with_current_epoch():
    client = FakeClient(_observation(mode=1, epoch=11))
    robot = CobotMagicRemoteRobot(client)
    chunk = np.zeros((10, 7), dtype=np.float32)

    result = robot.send_action_chunk(chunk, waypoint_hz=30, observation=robot.observe())

    assert result["reason"] == "dry_run"
    assert client.chunks[0][0] == "left"
    assert isinstance(client.chunks[0][1], list)
    assert all(isinstance(action, list) for action in client.chunks[0][1])
    np.testing.assert_allclose(client.chunks[0][1], chunk)
    assert client.chunks[0][2] == 11
    assert client.chunks[0][3] > 0
    assert client.chunks[0][4] == 30.0


class _FakeContinuousPinkIK:
    def __init__(self):
        self.pin = SimpleNamespace(
            exp3=lambda _rotvec: np.eye(3),
            SE3=lambda rotation, translation: (rotation, translation),
        )
        self.reset_calls = []
        self.posture_calls = []
        self.step_measured = []

    @staticmethod
    def coerce_measured_joints(joints):
        return np.asarray(joints).copy()

    def reset(self, joints):
        self.reset_calls.append(np.asarray(joints).copy())

    def set_posture_reference(self, joints):
        self.posture_calls.append(np.asarray(joints).copy())

    def step(self, _target, *, measured_joints=None):
        self.step_measured.append(
            None if measured_joints is None else np.asarray(measured_joints).copy()
        )
        return SimpleNamespace(
            joints=np.zeros(6),
            diagnostics=SimpleNamespace(as_dict=lambda: {"finite": True}),
        )


def test_pink_ik_preserves_controller_state_within_control_epoch(monkeypatch):
    robot = CobotMagicRemoteRobot(FakeClient(_observation(mode=1, epoch=4)))
    fake_ik = _FakeContinuousPinkIK()
    monkeypatch.setattr(robot, "_get_pink_ik", lambda: fake_ik)
    target = np.zeros(7)

    robot._pink_joint_chunk_from_target(target, observation=robot.observe(), ticks=2)
    robot._pink_joint_chunk_from_target(target, observation=robot.observe(), ticks=2)

    assert len(fake_ik.reset_calls) == 1
    assert len(fake_ik.posture_calls) == 1
    assert fake_ik.step_measured[0] is None
    assert fake_ik.step_measured[1] is None
    assert fake_ik.step_measured[2] is None
    assert fake_ik.step_measured[3] is None


def test_pink_ik_resets_after_control_epoch_change(monkeypatch):
    client = FakeClient(_observation(mode=1, epoch=4))
    robot = CobotMagicRemoteRobot(client)
    fake_ik = _FakeContinuousPinkIK()
    monkeypatch.setattr(robot, "_get_pink_ik", lambda: fake_ik)

    robot._pink_joint_chunk_from_target(np.zeros(7), observation=robot.observe(), ticks=1)
    client.observation["control"]["epoch"] = 5
    robot._pink_joint_chunk_from_target(np.zeros(7), observation=robot.observe(), ticks=1)

    assert len(fake_ik.reset_calls) == 2
    assert len(fake_ik.posture_calls) == 2
    assert fake_ik.step_measured == [None, None]


def test_remote_robot_chunk_sequence_survives_new_client_process(monkeypatch):
    stamps = iter((1_000_000, 2_000_000))
    monkeypatch.setattr("cobot_bridge.remote_robot.time.time_ns", lambda: next(stamps))
    first_client = FakeClient(_observation(mode=1, epoch=11))
    second_client = FakeClient(_observation(mode=1, epoch=12))

    first_robot = CobotMagicRemoteRobot(first_client)
    first_robot.send_action_chunk(
        np.zeros((2, 7), dtype=np.float32),
        waypoint_hz=30,
        observation=first_robot.observe(),
    )
    second_robot = CobotMagicRemoteRobot(second_client)
    second_robot.send_action_chunk(
        np.zeros((2, 7), dtype=np.float32),
        waypoint_hz=30,
        observation=second_robot.observe(),
    )

    assert second_client.chunks[0][3] > first_client.chunks[0][3]


@pytest.mark.parametrize(
    "mode,teach,reason",
    [(3, True, "teach_active"), (4, False, "resync_required"), (0, False, "policy_not_owner")],
)
def test_remote_robot_blocks_invalid_owner(mode, teach, reason):
    robot = CobotMagicRemoteRobot(FakeClient(_observation(mode=mode, teach=teach)))
    observation = robot.observe()
    with pytest.raises(ControlNotReady, match=reason):
        robot.send_action([0.0] * 7, observation=observation)
