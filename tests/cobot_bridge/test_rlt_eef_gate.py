from pathlib import Path

import numpy as np

from cobot_bridge.rlt_eef_gate import (
    EEFWorkspace,
    PiperEEFInsertionGate,
    PiperGraspedEEFInsertionGate,
)


class FakeKinematics:
    def __init__(self, positions):
        self.positions = iter(positions)

    def forward(self, _joints):
        position = np.asarray(next(self.positions), dtype=np.float64)[None, :]
        return {"gripper_center": (position, np.eye(3)[None, :, :])}


def _gate(positions, confirmations=2):
    gate = PiperEEFInsertionGate.__new__(PiperEEFInsertionGate)
    gate.workspace = EEFWorkspace()
    gate.confirmation_samples = confirmations
    gate.kinematics = FakeKinematics(positions)
    gate.consecutive_hits = 0
    gate.latched = False
    gate.last_position = None
    return gate


def test_gate_requires_consecutive_workspace_hits_and_then_latches():
    gate = _gate(((0.35, 0.32, 0.12), (0.20, 0.10, 0.20),
                  (0.35, 0.32, 0.12), (0.36, 0.31, 0.10),
                  (0.20, 0.10, 0.20)))
    state = np.zeros(7)

    assert gate.update(state) is False
    assert gate.update(state) is False
    assert gate.update(state) is False
    assert gate.update(state) is True
    assert gate.update(state) is True


def test_default_workspace_covers_successful_demo_insertion_region():
    workspace = EEFWorkspace()
    assert workspace.contains((0.35, 0.32, 0.10))
    assert not workspace.contains((0.39, 0.18, 0.10))  # pickup side
    assert not workspace.contains((0.35, 0.32, 0.25))  # high transit


def test_real_fk_gate_constructs_with_official_urdf():
    gate = PiperEEFInsertionGate(confirmation_samples=1)
    assert Path(gate.kinematics.urdf_path).is_file()


def test_grasped_fixture_gate_requires_open_then_closed_before_takeover():
    gate = PiperGraspedEEFInsertionGate.__new__(PiperGraspedEEFInsertionGate)
    gate.workspace = EEFWorkspace((0.34, 0.05, 0.08), (0.42, 0.11, 0.18))
    gate.confirmation_samples = 2
    gate.kinematics = FakeKinematics(((0.38, 0.08, 0.12),) * 5)
    gate.consecutive_hits = 0
    gate.latched = False
    gate.last_position = None
    gate.open_threshold = 0.02
    gate.closed_threshold = 0.04
    gate.seen_open = False
    gate.grasp_latched = False

    opened = np.zeros(7)
    closed = np.zeros(7)
    closed[6] = 0.055

    assert gate.update(opened) is False  # open at reset is not a grasp
    assert gate.update(opened) is False
    assert gate.update(closed) is False  # first fixture confirmation
    assert gate.update(closed) is True   # second fixture confirmation


def test_current_metal_task_gate_latches_after_observed_close_and_fixture_region():
    """Regression for the successful 2026-09-16 rollout calibration.

    Piper's coordinate is near-zero=open, not closed. The observed grasp at
    x=.54 is therefore followed by two fixture hits at x=.381 and x=.393.
    """

    gate = PiperGraspedEEFInsertionGate.__new__(PiperGraspedEEFInsertionGate)
    gate.workspace = EEFWorkspace((0.34, 0.05, 0.08), (0.42, 0.11, 0.18))
    gate.confirmation_samples = 2
    gate.kinematics = FakeKinematics(
        (
            (0.5414, 0.1091, 0.1785),  # grasp closes outside fixture box
            (0.3807, 0.0789, 0.1056),  # first post-grasp fixture hit
            (0.3929, 0.0803, 0.0816),  # second post-grasp fixture hit
        )
    )
    gate.consecutive_hits = 0
    gate.latched = False
    gate.last_position = None
    gate.open_threshold = 0.04
    gate.closed_threshold = 0.02
    gate.seen_open = False
    gate.grasp_latched = False

    opened = np.zeros(7)
    closed = np.zeros(7)
    closed[6] = 0.055

    assert gate.update(opened) is False
    assert gate.update(closed) is False
    assert gate.update(closed) is True
