from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pink")
pytest.importorskip("pinocchio")
pytest.importorskip("qpsolvers")

from cobot_bridge.piper_pink_ik import (
    PiperIKSettings,
    PiperPinkIK,
    hardware_to_pink_joints,
    pink_to_hardware_joints,
)


PIPER_KINEMATIC_URDF = """<?xml version="1.0"?>
<robot name="piper_kinematic_test">
  <link name="base_link"/><link name="link1"/><link name="link2"/>
  <link name="link3"/><link name="link4"/><link name="link5"/><link name="link6"/>
  <link name="gripper_base"/><link name="link7"/><link name="link8"/>
  <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/><origin xyz="0 0 0.123" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-2.618" upper="2.618" effort="100" velocity="5"/></joint>
  <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0 0 0" rpy="1.5708 -0.1359 -3.1416"/><axis xyz="0 0 1"/><limit lower="0" upper="3.14" effort="100" velocity="5"/></joint>
  <joint name="joint3" type="revolute"><parent link="link2"/><child link="link3"/><origin xyz="0.28503 0 0" rpy="0 0 -1.7939"/><axis xyz="0 0 1"/><limit lower="-2.967" upper="0" effort="100" velocity="5"/></joint>
  <joint name="joint4" type="revolute"><parent link="link3"/><child link="link4"/><origin xyz="-0.021984 -0.25075 0" rpy="1.5708 0 0"/><axis xyz="0 0 1"/><limit lower="-1.745" upper="1.745" effort="100" velocity="5"/></joint>
  <joint name="joint5" type="revolute"><parent link="link4"/><child link="link5"/><origin xyz="0 0 0" rpy="-1.5708 0 0"/><axis xyz="0 0 1"/><limit lower="-1.22" upper="1.22" effort="100" velocity="5"/></joint>
  <joint name="joint6" type="revolute"><parent link="link5"/><child link="link6"/><origin xyz="0.000088259 -0.091 0" rpy="1.5708 0 0"/><axis xyz="0 0 1"/><limit lower="-2.0944" upper="2.0944" effort="100" velocity="3"/></joint>
  <joint name="joint6_to_gripper_base" type="fixed"><parent link="link6"/><child link="gripper_base"/></joint>
  <joint name="joint7" type="prismatic"><parent link="gripper_base"/><child link="link7"/><origin xyz="0 0 0.1358" rpy="1.5708 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="0.035" effort="10" velocity="1"/></joint>
  <joint name="joint8" type="prismatic"><parent link="gripper_base"/><child link="link8"/><origin xyz="0 0 0.1358" rpy="1.5708 0 -3.1416"/><axis xyz="0 0 -1"/><limit lower="-0.035" upper="0" effort="10" velocity="1"/></joint>
</robot>
"""


def test_piper_hardware_and_urdf_joint_conventions_are_explicit():
    hardware = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    pink = hardware_to_pink_joints(hardware)
    np.testing.assert_allclose(pink, hardware)
    np.testing.assert_allclose(pink_to_hardware_joints(pink), hardware)


@pytest.fixture()
def piper_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "piper.urdf"
    path.write_text(PIPER_KINEMATIC_URDF)
    return path


def test_model_reduces_gripper_and_preserves_official_limits(piper_urdf: Path):
    ik = PiperPinkIK(piper_urdf)

    assert ik.model.nq == 6
    np.testing.assert_allclose(
        ik.lower_limits,
        [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944],
    )
    np.testing.assert_allclose(
        ik.upper_limits,
        [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944],
    )
    np.testing.assert_allclose(ik.model.velocityLimit, [0.45] * 6)


def test_measured_joint_noise_is_clipped_only_within_tolerance(piper_urdf: Path):
    ik = PiperPinkIK(piper_urdf)

    measured = np.asarray([0.0, -0.006, 0.005, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(ik.coerce_measured_joints(measured), np.zeros(6))
    with pytest.raises(ValueError, match="beyond feedback tolerance"):
        ik.coerce_measured_joints([0.0, -0.03, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exceed URDF limits"):
        ik.reset(measured)


def test_measured_joint5_uses_the_configured_physical_end_stop_margin(piper_urdf: Path):
    ik = PiperPinkIK(piper_urdf)

    # J5 may report within 0.10 rad beyond the URDF's 1.22-rad planner limit.
    measured = np.asarray([0.0, 0.0, 0.0, 0.0, 1.30, 0.0])
    clipped = ik.coerce_measured_joints(measured)
    assert clipped[4] == pytest.approx(1.22)
    with pytest.raises(ValueError, match="beyond feedback tolerance"):
        ik.coerce_measured_joints([0.0, 0.0, 0.0, 0.0, 1.33, 0.0])


def test_fk_to_ik_reconstruction_converges_from_nearby_posture(piper_urdf: Path):
    ik = PiperPinkIK(piper_urdf)
    target_joints = np.asarray([0.35, 1.35, -1.55, 0.15, 0.45, -0.30])
    current_joints = target_joints + np.asarray([-0.04, 0.03, -0.03, 0.02, -0.02, 0.03])
    target_pose = ik.forward_kinematics(target_joints)

    step, iterations, converged = ik.solve_pose(
        target_pose,
        current_joints=current_joints,
        maximum_iterations=1_500,
    )

    assert converged, step.diagnostics
    assert iterations < 1_500
    assert step.diagnostics.position_error_m <= 5e-4
    assert step.diagnostics.orientation_error_rad <= 5e-3
    assert step.diagnostics.finite


def test_one_tick_obeys_velocity_and_acceleration_bounds(piper_urdf: Path):
    settings = PiperIKSettings(max_joint_velocity=0.30, max_joint_acceleration=0.60)
    ik = PiperPinkIK(piper_urdf, settings=settings)
    current = np.asarray([0.35, 1.35, -1.55, 0.15, 0.45, -0.30])
    target = current + np.asarray([0.20, 0.10, -0.10, 0.10, -0.10, 0.10])
    ik.reset(current)

    step = ik.step(ik.forward_kinematics(target))

    assert np.max(np.abs(step.velocity)) <= settings.max_joint_acceleration * ik.dt + 1e-5
    assert step.diagnostics.maximum_joint_step_rad <= (
        settings.max_joint_acceleration * ik.dt * ik.dt + 1e-6
    )
    assert step.diagnostics.maximum_joint_velocity_rad_s <= settings.max_joint_velocity + 1e-6
