from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pinocchio")

from cobot_bridge.piper_eef_metrics import (
    DEFAULT_PIPER_URDF,
    PiperForwardKinematics,
    eef_error_summary,
)


def test_official_piper_fk_has_flange_and_gripper_center() -> None:
    assert Path(DEFAULT_PIPER_URDF).is_file()
    kinematics = PiperForwardKinematics()

    assert kinematics.model.nq == 6
    assert kinematics.frames == ("gripper_center", "link6")
    transforms = kinematics.forward(np.zeros((1, 6), dtype=np.float64))
    assert transforms["gripper_center"][0].shape == (1, 3)
    assert transforms["link6"][1].shape == (1, 3, 3)


def test_eef_metric_uses_arm_joints_but_not_gripper_dimension() -> None:
    kinematics = PiperForwardKinematics()
    target = np.zeros((2, 3, 7), dtype=np.float32)
    predicted = target.copy()
    predicted[0, :, 6] = 1.0
    valid = np.ones((2, 3), dtype=bool)
    mean = np.asarray([0.0, 1.2, -1.2, 0.0, 0.3, 0.0, 0.04])
    std = np.asarray([0.3, 0.5, 0.4, 0.2, 0.2, 0.2, 0.03])

    gripper_only = eef_error_summary(
        predicted,
        target,
        valid,
        action_mean=mean,
        action_std=std,
        kinematics=kinematics,
    )
    assert gripper_only["frames"]["gripper_center"]["position_error_mm"]["rmse"] == 0
    assert gripper_only["frames"]["gripper_center"]["orientation_error_deg"]["rmse"] == 0

    predicted[1, :, 0] = 0.2
    arm_error = eef_error_summary(
        predicted,
        target,
        valid,
        action_mean=mean,
        action_std=std,
        kinematics=kinematics,
    )
    assert arm_error["samples"] == 6
    assert arm_error["frames"]["gripper_center"]["position_error_mm"]["rmse"] > 0
    assert arm_error["frames"]["gripper_center"]["orientation_error_deg"]["rmse"] > 0
    assert len(arm_error["frames"]["gripper_center"]["per_horizon"]) == 3


def test_eef_metric_respects_valid_mask() -> None:
    kinematics = PiperForwardKinematics(frames=("gripper_center",))
    target = np.zeros((1, 2, 7), dtype=np.float32)
    predicted = target.copy()
    predicted[0, 1, 0] = 1.0
    valid = np.asarray([[True, False]])
    mean = np.asarray([0.0, 1.2, -1.2, 0.0, 0.3, 0.0, 0.04])
    std = np.ones(7)

    summary = eef_error_summary(
        predicted,
        target,
        valid,
        action_mean=mean,
        action_std=std,
        kinematics=kinematics,
    )

    assert summary["samples"] == 1
    assert summary["frames"]["gripper_center"]["position_error_mm"]["maximum"] == 0
    assert summary["frames"]["gripper_center"]["per_horizon"][1]["samples"] == 0
