from pathlib import Path
import math
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from cobot_magic_control_core import (
    ControlMode,
    ControlStateMachine,
    OneShotTeachAlignmentGate,
    VelocityAccelerationLimiter,
    euler_xyz_to_rotvec,
    rotvec_to_euler_xyz,
    validate_eef_target,
    validate_joint_action,
)


def live_machine(mode: ControlMode) -> ControlStateMachine:
    machine = ControlStateMachine(dry_run=False)
    assert machine.request_mode(mode).accepted
    assert machine.set_output_enabled(True).accepted
    return machine


def test_safe_by_default():
    machine = ControlStateMachine()
    assert machine.effective_mode == ControlMode.SAFE
    assert machine.policy_targets() == ()
    assert machine.authorize_action(epoch=machine.epoch).reason == "owner_safe"


def test_mode_1_targets_only_puppet():
    machine = live_machine(ControlMode.POLICY_PUPPET)
    assert machine.policy_targets() == ("puppet",)
    assert machine.authorize_action(epoch=machine.epoch).accepted


def test_mode_2_targets_master_and_puppet():
    machine = live_machine(ControlMode.RLT_SYNC)
    assert machine.effective_mode == ControlMode.ALIGN_SYNC
    assert machine.policy_targets() == ()
    assert machine.authorize_action(epoch=machine.epoch).reason == "align_sync_in_progress"
    assert machine.authorize_alignment_target(epoch=machine.epoch).accepted
    assert machine.mark_alignment_started().accepted
    assert machine.finish_alignment().accepted
    assert machine.effective_mode == ControlMode.RLT_SYNC
    assert machine.policy_targets() == ("master", "puppet")
    assert machine.authorize_action(epoch=machine.epoch).accepted


def test_teach_overrides_policy_and_requires_resync():
    machine = live_machine(ControlMode.RLT_SYNC)
    before_teach = machine.epoch
    assert machine.set_teach_active(True).accepted
    assert machine.effective_mode == ControlMode.TEACH
    assert machine.epoch > before_teach
    assert machine.authorize_action(epoch=machine.epoch).reason == "teach_active"

    assert machine.set_teach_active(False).accepted
    assert machine.effective_mode == ControlMode.RESYNC_REQUIRED
    assert machine.authorize_action(epoch=machine.epoch).reason == "resync_required"

    assert machine.resync().accepted
    assert machine.effective_mode == ControlMode.ALIGN_SYNC
    assert machine.authorize_alignment_target(epoch=machine.epoch).accepted
    assert machine.finish_alignment().accepted
    assert machine.effective_mode == ControlMode.RLT_SYNC
    assert machine.authorize_action(epoch=machine.epoch).accepted


def test_teach_is_not_blocked_by_normal_output_gate():
    machine = ControlStateMachine(dry_run=False)
    assert machine.output_enabled is False
    assert machine.set_teach_active(True).accepted
    assert machine.effective_mode == ControlMode.TEACH
    assert machine.output_enabled is False


def test_teach_overrides_emergency_but_policy_stays_latched_off():
    machine = ControlStateMachine(dry_run=False)
    machine.set_teach_active(True)
    assert machine.effective_mode == ControlMode.TEACH

    decision = machine.emergency_safe()
    assert decision.accepted
    assert machine.emergency_latched is True
    assert machine.effective_mode == ControlMode.TEACH
    assert machine.output_enabled is False

    assert machine.set_teach_active(True).reason == "teach_unchanged"
    assert machine.effective_mode == ControlMode.TEACH
    assert machine.set_output_enabled(True).reason == "emergency_latched"

    assert machine.set_teach_active(False).reason == "teach_released_emergency_latched"
    assert machine.emergency_latched is True
    assert machine.effective_mode == ControlMode.SAFE
    assert machine.request_mode(ControlMode.POLICY_PUPPET).reason == "emergency_latched"

    assert machine.set_teach_active(True).accepted
    assert machine.effective_mode == ControlMode.TEACH
    assert machine.set_teach_active(False).accepted
    assert machine.emergency_latched is True

    assert machine.clear_emergency().accepted
    assert machine.emergency_latched is False
    assert machine.effective_mode == ControlMode.SAFE


def test_old_epoch_is_rejected():
    machine = live_machine(ControlMode.POLICY_PUPPET)
    old_epoch = machine.epoch
    machine.request_mode(ControlMode.RLT_SYNC)
    decision = machine.authorize_alignment_target(epoch=old_epoch)
    assert not decision.accepted
    assert decision.reason == "stale_epoch"


def test_dry_run_is_hard_gate():
    machine = ControlStateMachine(dry_run=True)
    machine.request_mode(ControlMode.RLT_SYNC)
    machine.set_output_enabled(True)
    decision = machine.authorize_alignment_target(epoch=machine.epoch)
    assert not decision.accepted
    assert decision.reason == "dry_run"


def test_joint_action_validation():
    assert validate_joint_action([0] * 7) == [0.0] * 7


def test_eef_target_validation_and_rotation_round_trip():
    target = [0.32, 0.14, 0.11, 0.1, -0.2, 0.3, 0.08]
    assert validate_eef_target(target) == target
    euler = rotvec_to_euler_xyz(target[3:6])
    recovered = euler_xyz_to_rotvec(euler)
    for expected, actual in zip(target[3:6], recovered, strict=True):
        assert abs(expected - actual) < 1e-9


def test_piper_feedback_uses_extrinsic_xyz_rpy():
    # This is the right-puppet feedback captured from the SDK/ROS path.
    sdk_rpy_deg = [-8.696, 85.297, -8.553]
    expected_rotvec = [-0.011498776831613313, 1.4896194275412111, -0.007623619685298663]
    sdk_rpy = [math.radians(value) for value in sdk_rpy_deg]

    recovered_rotvec = euler_xyz_to_rotvec(sdk_rpy)
    for expected, actual in zip(expected_rotvec, recovered_rotvec, strict=True):
        assert abs(expected - actual) < 2e-6

    recovered_rpy = rotvec_to_euler_xyz(expected_rotvec)
    for expected, actual in zip(sdk_rpy, recovered_rpy, strict=True):
        assert abs(expected - actual) < 2e-6


def test_velocity_acceleration_limiter_bounds_motion_and_reversal():
    limiter = VelocityAccelerationLimiter([1.0] * 7, [2.0] * 7, reset_after_s=0.5)
    position, velocity = limiter.step([1.0] * 7, now=0.0, measured_position=[0.0] * 7)
    assert position == [0.0] * 7
    assert velocity == [0.0] * 7

    previous_velocity = velocity
    for tick in range(1, 1001):
        target = [1.0] * 7 if tick < 80 else [-1.0] * 7
        position, velocity = limiter.step(target, now=tick * 0.005)
        assert all(abs(value) <= 1.0 + 1e-12 for value in velocity)
        assert all(
            abs(value - previous) / 0.005 <= 2.0 + 1e-9
            for value, previous in zip(velocity, previous_velocity, strict=True)
        )
        previous_velocity = velocity


def test_velocity_acceleration_limiter_resets_to_feedback_after_gap():
    limiter = VelocityAccelerationLimiter([1.0] * 7, [2.0] * 7, reset_after_s=0.1)
    limiter.step([1.0] * 7, now=0.0, measured_position=[0.0] * 7)
    limiter.step([1.0] * 7, now=0.01)
    position, velocity = limiter.step(
        [2.0] * 7, now=0.5, measured_position=[0.25] * 7
    )
    assert position == [0.25] * 7
    assert velocity == [0.0] * 7


def test_teach_alignment_is_one_shot_until_button_release():
    gate = OneShotTeachAlignmentGate()

    assert gate.needs_alignment(maximum_error=0.5, threshold=0.25)
    gate.finish_alignment()
    assert gate.follow_ready
    assert not gate.needs_alignment(maximum_error=1.0, threshold=0.25)

    gate.reset()
    assert not gate.follow_ready
    assert gate.needs_alignment(maximum_error=0.5, threshold=0.25)


def test_teach_alignment_small_initial_error_enters_direct_follow():
    gate = OneShotTeachAlignmentGate()

    assert not gate.needs_alignment(maximum_error=0.1, threshold=0.25)
    assert gate.follow_ready
    assert not gate.needs_alignment(maximum_error=1.0, threshold=0.25)
