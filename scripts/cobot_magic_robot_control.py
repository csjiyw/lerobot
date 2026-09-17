#!/usr/bin/env python3
"""AGX-side ROS control arbiter for Cobot Magic master/puppet arms.

The node is safe by default: it starts in SAFE, output is disabled, and live
publication is impossible unless the process is launched with ``--live-output``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Empty, String, UInt8, UInt64

from piper_msgs.msg import PiperStatusMsg

from cobot_magic_control_core import (
    ControlMode,
    ControlStateMachine,
    OneShotTeachAlignmentGate,
    VelocityAccelerationLimiter,
    euler_xyz_to_rotvec,
    quaternion_to_rotvec,
    rotvec_to_euler_xyz,
    validate_eef_target,
    validate_joint_action,
)


SIDES = ("left", "right")
JOINT_COUNT = 7


@dataclass
class JointSnapshot:
    name: list[str]
    position: list[float]
    velocity: list[float]
    effort: list[float]
    stamp_ns: int
    received_at: float


@dataclass
class EefSnapshot:
    position: list[float]
    stamp_ns: int
    received_at: float


@dataclass
class StatusSnapshot:
    ctrl_mode: int
    teach_status: int
    arm_status: int
    motion_status: int
    err_code: int
    received_at: float


@dataclass
class ImageSnapshot:
    data: bytes
    fmt: str
    stamp_ns: int
    received_at: float


@dataclass
class CachedAction:
    position: list[float]
    epoch: int
    sequence: Optional[int]
    source: str
    received_at: float
    alignment_speed: Optional[float] = None


@dataclass
class CachedEefAction:
    target: list[float]
    epoch: int
    sequence: Optional[int]
    source: str
    received_at: float


@dataclass
class AlignmentState:
    start: list[float]
    target: list[float]
    started_at: float
    duration: float


@dataclass
class SyncAlignmentPlan:
    starts: dict[str, dict[str, list[float]]]
    targets: dict[str, list[float]]
    started_at: float
    duration: float
    deadline: float
    speed: float
    stable_since: Optional[float] = None


def _stamp_ns(message: Any) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def add_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live-output", action="store_true", help="allow ROS command publication")
    parser.add_argument("--require-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--active-sides", nargs="+", choices=SIDES, default=list(SIDES))
    parser.add_argument("--tick-hz", type=float, default=200.0)
    parser.add_argument("--action-timeout", type=float, default=0.25)
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument(
        "--command-smoothing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-joint-velocity", type=float, default=0.45)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.0)
    parser.add_argument("--max-gripper-velocity", type=float, default=0.08)
    parser.add_argument("--max-gripper-acceleration", type=float, default=0.20)
    parser.add_argument("--smoothing-reset-seconds", type=float, default=0.25)
    parser.add_argument("--teach-stable-seconds", type=float, default=0.10)
    parser.add_argument("--teach-modes", nargs="+", type=lambda value: int(value, 0), default=[0x02, 0x06])
    parser.add_argument("--slow-alignment-threshold", type=float, default=math.radians(15.0))
    parser.add_argument("--slow-alignment-speed", type=float, default=0.12)
    parser.add_argument("--slow-alignment-min-seconds", type=float, default=2.0)
    # Retained for launch compatibility. Teach alignment is now one-shot and
    # no longer restarts while the master is moving.
    parser.add_argument("--slow-retarget-threshold", type=float, default=0.15)
    parser.add_argument("--sync-alignment-speed", type=float, default=0.12)
    parser.add_argument("--max-sync-alignment-speed", type=float, default=0.48)
    parser.add_argument("--sync-alignment-min-seconds", type=float, default=1.0)
    parser.add_argument("--sync-alignment-timeout", type=float, default=30.0)
    parser.add_argument("--sync-position-tolerance", type=float, default=0.03)
    parser.add_argument("--sync-gripper-tolerance", type=float, default=0.01)
    parser.add_argument("--sync-stable-seconds", type=float, default=0.20)
    parser.add_argument("--post-align-action-timeout", type=float, default=1.0)

    parser.add_argument("--master-left-feedback", default="/puppet/joint_rear_left")
    parser.add_argument("--master-right-feedback", default="/puppet/joint_rear_right")
    parser.add_argument("--puppet-left-feedback", default="/puppet/joint_left")
    parser.add_argument("--puppet-right-feedback", default="/puppet/joint_right")
    parser.add_argument("--master-left-eef-feedback", default="/puppet/end_pose_rear_left")
    parser.add_argument("--master-right-eef-feedback", default="/puppet/end_pose_rear_right")
    parser.add_argument("--puppet-left-eef-feedback", default="/puppet/end_pose_left")
    parser.add_argument("--puppet-right-eef-feedback", default="/puppet/end_pose_right")
    parser.add_argument("--master-left-status", default="/rear_left/piper_ctrl/arm_status")
    parser.add_argument("--master-right-status", default="/rear_right/piper_ctrl/arm_status")

    parser.add_argument("--master-left-state", default="/master/left_state")
    parser.add_argument("--master-right-state", default="/master/right_state")
    parser.add_argument("--puppet-left-state", default="/puppet/left_state")
    parser.add_argument("--puppet-right-state", default="/puppet/right_state")
    parser.add_argument("--master-left-command", default="/master/left_cmd")
    parser.add_argument("--master-right-command", default="/master/right_cmd")
    parser.add_argument("--puppet-left-command", default="/puppet/left_cmd")
    parser.add_argument("--puppet-right-command", default="/puppet/right_cmd")
    parser.add_argument("--master-left-eef-command", default="/master/left_eef_cmd")
    parser.add_argument("--master-right-eef-command", default="/master/right_eef_cmd")
    parser.add_argument("--puppet-left-eef-command", default="/puppet/left_eef_cmd")
    parser.add_argument("--puppet-right-eef-command", default="/puppet/right_eef_cmd")
    parser.add_argument("--master-left-enable", default="/rear_left/enable_flag")
    parser.add_argument("--master-right-enable", default="/rear_right/enable_flag")
    parser.add_argument("--puppet-left-enable", default="/front_left/enable_flag")
    parser.add_argument("--puppet-right-enable", default="/front_right/enable_flag")

    parser.add_argument("--camera-left", default="/camera_l/color/image_raw/compressed")
    parser.add_argument("--camera-front", default="/camera_f/color/image_raw/compressed")
    parser.add_argument("--camera-right", default="/camera_r/color/image_raw/compressed")
    parser.add_argument("--policy-left-input", default="/policy/left_cmd")
    parser.add_argument("--policy-right-input", default="/policy/right_cmd")
    parser.add_argument("--policy-left-eef-input", default="/policy/left_eef_cmd")
    parser.add_argument("--policy-right-eef-input", default="/policy/right_eef_cmd")
    parser.add_argument("--mode-request-topic", default="/control/request_mode")
    parser.add_argument("--output-request-topic", default="/control/output_enabled")
    parser.add_argument("--resync-topic", default="/control/resync")
    parser.add_argument("--emergency-topic", default="/control/emergency_safe")
    parser.add_argument("--clear-emergency-topic", default="/control/clear_emergency")
    parser.add_argument("--state-topic", default="/control/state")
    parser.add_argument("--teach-active-topic", default="/control/teach_active")
    parser.add_argument("--effective-mode-topic", default="/control/effective_mode")
    parser.add_argument("--epoch-topic", default="/control/epoch")
    parser.add_argument("--owner-topic", default="/control/owner")
    parser.add_argument("--action-result-topic", default="/control/action_result")


class CobotMagicRobotControl(Node):
    """Canonical topic adapter, observation cache and single command owner."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("cobot_magic_robot_control")
        self.args = args
        self.active_sides = tuple(args.active_sides)
        self._lock = threading.RLock()
        self.machine = ControlStateMachine(dry_run=not args.live_output)

        self.master_state: dict[str, Optional[JointSnapshot]] = {side: None for side in SIDES}
        self.puppet_state: dict[str, Optional[JointSnapshot]] = {side: None for side in SIDES}
        self.master_eef_state: dict[str, Optional[EefSnapshot]] = {
            side: None for side in SIDES
        }
        self.puppet_eef_state: dict[str, Optional[EefSnapshot]] = {
            side: None for side in SIDES
        }
        self.master_status: dict[str, Optional[StatusSnapshot]] = {side: None for side in SIDES}
        self.images: dict[str, Optional[ImageSnapshot]] = {
            "left": None,
            "front": None,
            "right": None,
        }
        self.actions: dict[str, Optional[CachedAction]] = {side: None for side in SIDES}
        self.eef_actions: dict[str, Optional[CachedEefAction]] = {
            side: None for side in SIDES
        }
        self.alignments: dict[str, Optional[AlignmentState]] = {side: None for side in SIDES}
        self.teach_alignment_gates = {
            side: OneShotTeachAlignmentGate() for side in SIDES
        }
        self.sync_targets: dict[str, Optional[CachedAction]] = {side: None for side in SIDES}
        self.sync_alignment: Optional[SyncAlignmentPlan] = None
        self.teach_flags: dict[str, bool] = {side: False for side in SIDES}
        self._pending_teach_active = False
        self._pending_teach_since = time.monotonic()
        self._last_enable_publish = 0.0
        self._last_state_publish = 0.0
        self._observation_sequence = 0
        self._applied_sequence = 0
        self.last_applied: dict[str, dict[str, Optional[list[float]]]] = {
            side: {"master": None, "puppet": None} for side in SIDES
        }
        self.last_applied_velocity: dict[str, dict[str, Optional[list[float]]]] = {
            side: {"master": None, "puppet": None} for side in SIDES
        }
        velocity_limits = [args.max_joint_velocity] * 6 + [args.max_gripper_velocity]
        acceleration_limits = [args.max_joint_acceleration] * 6 + [
            args.max_gripper_acceleration
        ]
        self.command_limiters = {
            side: {
                role: VelocityAccelerationLimiter(
                    velocity_limits,
                    acceleration_limits,
                    reset_after_s=args.smoothing_reset_seconds,
                )
                for role in ("master", "puppet")
            }
            for side in SIDES
        }

        master_feedback = {
            "left": args.master_left_feedback,
            "right": args.master_right_feedback,
        }
        puppet_feedback = {
            "left": args.puppet_left_feedback,
            "right": args.puppet_right_feedback,
        }
        master_eef_feedback = {
            "left": args.master_left_eef_feedback,
            "right": args.master_right_eef_feedback,
        }
        puppet_eef_feedback = {
            "left": args.puppet_left_eef_feedback,
            "right": args.puppet_right_eef_feedback,
        }
        master_status = {
            "left": args.master_left_status,
            "right": args.master_right_status,
        }
        canonical_master_state = {
            "left": args.master_left_state,
            "right": args.master_right_state,
        }
        canonical_puppet_state = {
            "left": args.puppet_left_state,
            "right": args.puppet_right_state,
        }
        master_commands = {
            "left": args.master_left_command,
            "right": args.master_right_command,
        }
        puppet_commands = {
            "left": args.puppet_left_command,
            "right": args.puppet_right_command,
        }
        master_eef_commands = {
            "left": args.master_left_eef_command,
            "right": args.master_right_eef_command,
        }
        puppet_eef_commands = {
            "left": args.puppet_left_eef_command,
            "right": args.puppet_right_eef_command,
        }
        master_enable = {
            "left": args.master_left_enable,
            "right": args.master_right_enable,
        }
        puppet_enable = {
            "left": args.puppet_left_enable,
            "right": args.puppet_right_enable,
        }

        self.master_state_publishers = {
            side: self.create_publisher(JointState, canonical_master_state[side], 10)
            for side in SIDES
        }
        self.puppet_state_publishers = {
            side: self.create_publisher(JointState, canonical_puppet_state[side], 10)
            for side in SIDES
        }
        self.master_eef_state_publishers = {
            side: self.create_publisher(PoseStamped, f"/master/{side}_eef_state", 10)
            for side in SIDES
        }
        self.puppet_eef_state_publishers = {
            side: self.create_publisher(PoseStamped, f"/puppet/{side}_eef_state", 10)
            for side in SIDES
        }
        self.master_command_publishers = {
            side: self.create_publisher(JointState, master_commands[side], 10)
            for side in SIDES
        }
        self.puppet_command_publishers = {
            side: self.create_publisher(JointState, puppet_commands[side], 10)
            for side in SIDES
        }
        self.master_eef_command_publishers = {
            side: self.create_publisher(PosCmd, master_eef_commands[side], 10)
            for side in SIDES
        }
        self.puppet_eef_command_publishers = {
            side: self.create_publisher(PosCmd, puppet_eef_commands[side], 10)
            for side in SIDES
        }
        self.master_enable_publishers = {
            side: self.create_publisher(Bool, master_enable[side], 10) for side in SIDES
        }
        self.puppet_enable_publishers = {
            side: self.create_publisher(Bool, puppet_enable[side], 10) for side in SIDES
        }

        for side in SIDES:
            self.create_subscription(
                JointState,
                master_feedback[side],
                lambda message, arm=side: self._joint_callback("master", arm, message),
                10,
            )
            self.create_subscription(
                PoseStamped,
                master_eef_feedback[side],
                lambda message, arm=side: self._eef_callback("master", arm, message),
                10,
            )
            self.create_subscription(
                PoseStamped,
                puppet_eef_feedback[side],
                lambda message, arm=side: self._eef_callback("puppet", arm, message),
                10,
            )
            self.create_subscription(
                JointState,
                puppet_feedback[side],
                lambda message, arm=side: self._joint_callback("puppet", arm, message),
                10,
            )
            self.create_subscription(
                PiperStatusMsg,
                master_status[side],
                lambda message, arm=side: self._status_callback(arm, message),
                10,
            )

        camera_topics = {
            "left": args.camera_left,
            "front": args.camera_front,
            "right": args.camera_right,
        }
        for name, topic in camera_topics.items():
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, camera=name: self._image_callback(camera, message),
                10,
            )

        policy_topics = {
            "left": args.policy_left_input,
            "right": args.policy_right_input,
        }
        for side, topic in policy_topics.items():
            self.create_subscription(
                JointState,
                topic,
                lambda message, arm=side: self._policy_callback(arm, message),
                10,
            )
        policy_eef_topics = {
            "left": args.policy_left_eef_input,
            "right": args.policy_right_eef_input,
        }
        for side, topic in policy_eef_topics.items():
            self.create_subscription(
                PosCmd,
                topic,
                lambda message, arm=side: self._policy_eef_callback(arm, message),
                10,
            )

        self.create_subscription(UInt8, args.mode_request_topic, self._mode_request_callback, 10)
        self.create_subscription(Bool, args.output_request_topic, self._output_request_callback, 10)
        self.create_subscription(Empty, args.resync_topic, self._resync_callback, 10)
        self.create_subscription(Empty, args.emergency_topic, self._emergency_callback, 10)
        self.create_subscription(
            Empty, args.clear_emergency_topic, self._clear_emergency_callback, 10
        )
        self.state_publisher = self.create_publisher(String, args.state_topic, 10)
        self.teach_publisher = self.create_publisher(Bool, args.teach_active_topic, 10)
        self.mode_publisher = self.create_publisher(UInt8, args.effective_mode_topic, 10)
        self.epoch_publisher = self.create_publisher(UInt64, args.epoch_topic, 10)
        self.owner_publisher = self.create_publisher(String, args.owner_topic, 10)
        self.action_result_publisher = self.create_publisher(String, args.action_result_topic, 10)
        self.create_timer(1.0 / max(1.0, float(args.tick_hz)), self._tick)

        safety = "LIVE OUTPUT ALLOWED" if args.live_output else "DRY-RUN: command publication blocked"
        self.get_logger().warning(f"Cobot control started in SAFE; {safety}")

    def _reset_sync_locked(self) -> None:
        self.sync_targets = {side: None for side in SIDES}
        self.sync_alignment = None

    def _joint_callback(self, role: str, side: str, message: JointState) -> None:
        if len(message.position) < JOINT_COUNT:
            self.get_logger().warning(f"drop short {role}/{side} JointState")
            return
        position = [float(value) for value in message.position[:JOINT_COUNT]]
        if not _finite(position):
            self.get_logger().warning(f"drop non-finite {role}/{side} JointState")
            return
        velocity = [float(value) for value in message.velocity[:JOINT_COUNT]]
        velocity += [0.0] * (JOINT_COUNT - len(velocity))
        effort = [float(value) for value in message.effort[:JOINT_COUNT]]
        effort += [0.0] * (JOINT_COUNT - len(effort))
        names = list(message.name[:JOINT_COUNT])
        names += [f"joint{index}" for index in range(len(names), JOINT_COUNT)]
        snapshot = JointSnapshot(
            name=names,
            position=position,
            velocity=velocity,
            effort=effort,
            stamp_ns=_stamp_ns(message),
            received_at=time.monotonic(),
        )
        with self._lock:
            target = self.master_state if role == "master" else self.puppet_state
            target[side] = snapshot
            self._observation_sequence += 1
        publisher = (
            self.master_state_publishers[side]
            if role == "master"
            else self.puppet_state_publishers[side]
        )
        publisher.publish(message)

    def _eef_callback(self, role: str, side: str, message: PoseStamped) -> None:
        pose = message.pose
        try:
            position = [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                *quaternion_to_rotvec(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ),
            ]
        except (TypeError, ValueError):
            self.get_logger().warning(f"drop invalid {role}/{side} EEF feedback")
            return
        if not _finite(position):
            self.get_logger().warning(f"drop non-finite {role}/{side} EEF feedback")
            return
        snapshot = EefSnapshot(
            position=position,
            stamp_ns=_stamp_ns(message),
            received_at=time.monotonic(),
        )
        with self._lock:
            target = self.master_eef_state if role == "master" else self.puppet_eef_state
            target[side] = snapshot
            self._observation_sequence += 1
        publisher = (
            self.master_eef_state_publishers[side]
            if role == "master"
            else self.puppet_eef_state_publishers[side]
        )
        publisher.publish(message)

    def _policy_eef_callback(self, side: str, message: PosCmd) -> None:
        try:
            target = [
                float(message.x),
                float(message.y),
                float(message.z),
                *euler_xyz_to_rotvec(
                    (float(message.roll), float(message.pitch), float(message.yaw))
                ),
                float(message.gripper),
            ]
        except (TypeError, ValueError):
            result = {"accepted": False, "reason": "invalid_eef_command", "side": side}
        else:
            result = self.submit_eef_action(
                side,
                target,
                epoch=int(getattr(message, "mode1", -1)),
                sequence=int(getattr(message, "mode2", 0)),
                source="ros_policy_eef",
            )
        self._publish_result("SET_ACTION", {"action_space": "eef_target", **result})

    def _status_callback(self, side: str, message: PiperStatusMsg) -> None:
        snapshot = StatusSnapshot(
            ctrl_mode=int(message.ctrl_mode),
            teach_status=int(message.teach_status),
            arm_status=int(message.arm_status),
            motion_status=int(message.motion_status),
            err_code=int(message.err_code),
            received_at=time.monotonic(),
        )
        with self._lock:
            self.master_status[side] = snapshot
            self._observation_sequence += 1

    def _image_callback(self, name: str, message: CompressedImage) -> None:
        snapshot = ImageSnapshot(
            data=bytes(message.data),
            fmt=str(message.format or "jpeg"),
            stamp_ns=_stamp_ns(message),
            received_at=time.monotonic(),
        )
        with self._lock:
            self.images[name] = snapshot
            self._observation_sequence += 1

    @staticmethod
    def _action_metadata(
        frame_id: str,
    ) -> tuple[Optional[int], Optional[int], Optional[float]]:
        epoch: Optional[int] = None
        sequence: Optional[int] = None
        alignment_speed: Optional[float] = None
        normalized = frame_id.replace(":", "=").replace(",", ";")
        for item in normalized.split(";"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            try:
                if key == "epoch":
                    epoch = int(value)
                elif key in ("seq", "sequence"):
                    sequence = int(value)
                elif key in ("align_speed", "alignment_speed"):
                    alignment_speed = float(value)
            except ValueError:
                continue
        return epoch, sequence, alignment_speed

    def _publish_result(self, command: str, result: dict[str, Any]) -> None:
        message = String()
        payload = {"command": command, "timestamp_ns": time.time_ns(), **result}
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self.action_result_publisher.publish(message)

    def _policy_callback(self, side: str, message: JointState) -> None:
        epoch, sequence, alignment_speed = self._action_metadata(message.header.frame_id)
        result = self.submit_action(
            side,
            list(message.position),
            epoch=epoch,
            sequence=sequence,
            source="ros_policy",
            alignment_speed=alignment_speed,
        )
        self._publish_result("SET_ACTION", result)

    def _mode_request_callback(self, message: UInt8) -> None:
        result = self.request_mode(int(message.data))
        self._publish_result("SET_MODE", result)

    def _output_request_callback(self, message: Bool) -> None:
        result = self.set_output_enabled(bool(message.data))
        self._publish_result("SET_OUTPUT", result)

    def _resync_callback(self, _message: Empty) -> None:
        result = self.resync()
        self._publish_result("RESYNC", result)

    def _emergency_callback(self, _message: Empty) -> None:
        result = self.emergency_safe()
        self._publish_result("EMERGENCY_SAFE", result)

    def _clear_emergency_callback(self, _message: Empty) -> None:
        result = self.clear_emergency()
        self._publish_result("CLEAR_EMERGENCY", result)

    def _fresh(self, received_at: float, timeout: Optional[float] = None) -> bool:
        limit = self.args.feedback_timeout if timeout is None else timeout
        return time.monotonic() - received_at <= limit

    def _status_fresh_locked(self, side: str) -> bool:
        status = self.master_status[side]
        return status is not None and self._fresh(status.received_at)

    def _joint_fresh_locked(self, role: str, side: str) -> bool:
        state = self.master_state[side] if role == "master" else self.puppet_state[side]
        return state is not None and self._fresh(state.received_at)

    def _eef_fresh_locked(self, role: str, side: str) -> bool:
        state = self.master_eef_state[side] if role == "master" else self.puppet_eef_state[side]
        return state is not None and self._fresh(state.received_at)

    def _eef_position_locked(self, role: str, side: str) -> Optional[list[float]]:
        state = self.master_eef_state[side] if role == "master" else self.puppet_eef_state[side]
        joint = self.master_state[side] if role == "master" else self.puppet_state[side]
        if state is None or joint is None:
            return None
        return [*state.position, float(joint.position[6])]

    def _raw_teach_flags_locked(self) -> dict[str, bool]:
        flags: dict[str, bool] = {}
        for side in SIDES:
            status = self.master_status[side]
            flags[side] = bool(
                status is not None
                and self._fresh(status.received_at)
                and status.ctrl_mode in self.args.teach_modes
                and status.teach_status == 0x01
            )
        return flags

    def _refresh_teach_locked(self) -> None:
        now = time.monotonic()
        flags = self._raw_teach_flags_locked()
        previous_flags = dict(self.teach_flags)
        raw_active = any(flags[side] for side in self.active_sides)
        # Rising edge owns control immediately. Only the release edge is
        # debounced, so policy cannot continue during a button-press window.
        if raw_active:
            self._pending_teach_active = True
            self._pending_teach_since = now
            transition = not self.machine.teach_active
        else:
            if self._pending_teach_active:
                self._pending_teach_active = False
                self._pending_teach_since = now
            transition = bool(
                self.machine.teach_active
                and now - self._pending_teach_since >= self.args.teach_stable_seconds
            )
        if transition:
            self.machine.set_teach_active(raw_active)
            self.actions = {side: None for side in SIDES}
            self.eef_actions = {side: None for side in SIDES}
            self.alignments = {side: None for side in SIDES}
            self._reset_sync_locked()
            self._reset_command_limiters_locked()
        for side in SIDES:
            if flags[side] != previous_flags[side]:
                self.alignments[side] = None
                self.teach_alignment_gates[side].reset()
                self.command_limiters[side]["puppet"].reset()
        self.teach_flags = flags

    def _reset_command_limiters_locked(self) -> None:
        for side in SIDES:
            for role in ("master", "puppet"):
                self.command_limiters[side][role].reset()

    def request_mode(self, mode: ControlMode | int | str) -> dict[str, Any]:
        with self._lock:
            self._refresh_teach_locked()
            decision = self.machine.request_mode(mode)
            self.actions = {side: None for side in SIDES}
            self.eef_actions = {side: None for side in SIDES}
            self.alignments = {side: None for side in SIDES}
            self._reset_sync_locked()
            self._reset_command_limiters_locked()
            return decision.as_dict()

    def set_output_enabled(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            decision = self.machine.set_output_enabled(enabled)
            if not enabled:
                self.actions = {side: None for side in SIDES}
                self.eef_actions = {side: None for side in SIDES}
                self._reset_sync_locked()
                self._reset_command_limiters_locked()
            return decision.as_dict()

    def emergency_safe(self) -> dict[str, Any]:
        with self._lock:
            decision = self.machine.emergency_safe()
            self.actions = {side: None for side in SIDES}
            self.eef_actions = {side: None for side in SIDES}
            self.alignments = {side: None for side in SIDES}
            self._reset_sync_locked()
            self._reset_command_limiters_locked()
            return decision.as_dict()

    def clear_emergency(self) -> dict[str, Any]:
        with self._lock:
            decision = self.machine.clear_emergency()
            self.actions = {side: None for side in SIDES}
            self.eef_actions = {side: None for side in SIDES}
            self.alignments = {side: None for side in SIDES}
            self._reset_sync_locked()
            self._reset_command_limiters_locked()
            return decision.as_dict()

    def resync(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_teach_locked()
            decision = self.machine.resync()
            self.actions = {side: None for side in SIDES}
            self.eef_actions = {side: None for side in SIDES}
            self.alignments = {side: None for side in SIDES}
            self._reset_sync_locked()
            self._reset_command_limiters_locked()
            return decision.as_dict()

    def submit_action(
        self,
        side: str,
        values: list[Any],
        *,
        epoch: Optional[int],
        sequence: Optional[int] = None,
        source: str = "policy",
        alignment_speed: Optional[float] = None,
    ) -> dict[str, Any]:
        if side not in self.active_sides:
            return {"accepted": False, "reason": "inactive_side", "side": side}
        try:
            position = validate_joint_action(values)
        except (TypeError, ValueError) as exc:
            return {"accepted": False, "reason": str(exc), "side": side}
        if alignment_speed is not None and (
            not math.isfinite(alignment_speed)
            or alignment_speed <= 0
            or alignment_speed > self.args.max_sync_alignment_speed
        ):
            return {
                "accepted": False,
                "reason": "alignment_speed_exceeds_limit",
                "side": side,
                "maximum_alignment_speed": self.args.max_sync_alignment_speed,
            }

        with self._lock:
            self._refresh_teach_locked()
            if self.args.require_epoch and epoch is None:
                return {
                    "accepted": False,
                    "reason": "missing_epoch",
                    "side": side,
                    "epoch": self.machine.epoch,
                }
            if self.machine.effective_mode == ControlMode.ALIGN_SYNC:
                decision = self.machine.authorize_alignment_target(epoch=epoch)
                response = decision.as_dict()
                response["side"] = side
                response["phase"] = "ALIGN_SYNC"
                if not decision.accepted:
                    return response
                if not self._status_fresh_locked(side):
                    failure = self._fail_safe_locked("sync_feedback_stale_before_alignment")
                    failure["side"] = side
                    return failure
                if self.sync_alignment is not None:
                    response.update({"accepted": False, "reason": "align_sync_in_progress"})
                    return response
                if not self._joint_fresh_locked("master", side) or not self._joint_fresh_locked(
                    "puppet", side
                ):
                    failure = self._fail_safe_locked("sync_feedback_stale_before_alignment")
                    failure["side"] = side
                    return failure
                self.sync_targets[side] = CachedAction(
                    position=position,
                    epoch=self.machine.epoch,
                    sequence=sequence,
                    source=source,
                    received_at=time.monotonic(),
                    alignment_speed=alignment_speed,
                )
                if all(self.sync_targets[arm] is not None for arm in self.active_sides):
                    started = self._start_sync_alignment_locked()
                    response.update(started)
                else:
                    response.update(
                        {
                            "accepted": True,
                            "reason": "sync_first_target_buffered",
                            "waiting_sides": [
                                arm for arm in self.active_sides if self.sync_targets[arm] is None
                            ],
                        }
                    )
                return response
            if not self._status_fresh_locked(side):
                return {
                    "accepted": False,
                    "reason": "master_status_stale",
                    "side": side,
                    "epoch": self.machine.epoch,
                }
            decision = self.machine.authorize_action(epoch=epoch)
            response = decision.as_dict()
            response["side"] = side
            response["targets"] = list(self.machine.policy_targets())
            if not decision.accepted:
                return response
            self.actions[side] = CachedAction(
                position=position,
                epoch=self.machine.epoch,
                sequence=sequence,
                source=source,
                received_at=time.monotonic(),
            )
            self.eef_actions[side] = None
            response["applied"] = True
            return response

    def submit_eef_action(
        self,
        side: str,
        values: list[Any],
        *,
        epoch: Optional[int],
        sequence: Optional[int] = None,
        source: str = "policy_eef",
    ) -> dict[str, Any]:
        """Accept an absolute EEF target for the SDK PosCmd path.

        The first integration is deliberately Mode 1 only. Mode 2 continues
        to use the existing joint-space individual alignment until an EEF
        alignment target/feedback pair is implemented and tested separately.
        """

        if side not in self.active_sides:
            return {"accepted": False, "reason": "inactive_side", "side": side}
        try:
            target = validate_eef_target(values)
        except (TypeError, ValueError) as exc:
            return {
                "accepted": False,
                "reason": str(exc),
                "side": side,
                "action_space": "eef_target",
            }

        with self._lock:
            self._refresh_teach_locked()
            if self.args.require_epoch and epoch is None:
                return {
                    "accepted": False,
                    "reason": "missing_epoch",
                    "side": side,
                    "epoch": self.machine.epoch,
                    "action_space": "eef_target",
                }
            if self.machine.effective_mode in (ControlMode.ALIGN_SYNC, ControlMode.RLT_SYNC):
                return {
                    "accepted": False,
                    "reason": "eef_mode2_requires_joint_alignment",
                    "side": side,
                    "epoch": self.machine.epoch,
                    "action_space": "eef_target",
                }
            if not self._status_fresh_locked(side):
                return {
                    "accepted": False,
                    "reason": "master_status_stale",
                    "side": side,
                    "epoch": self.machine.epoch,
                    "action_space": "eef_target",
                }
            if not self._eef_fresh_locked("puppet", side):
                return {
                    "accepted": False,
                    "reason": "puppet_eef_feedback_stale",
                    "side": side,
                    "epoch": self.machine.epoch,
                    "action_space": "eef_target",
                }
            decision = self.machine.authorize_action(epoch=epoch)
            response = decision.as_dict()
            response.update(
                {
                    "side": side,
                    "action_space": "eef_target",
                    "targets": list(self.machine.policy_targets()),
                }
            )
            if not decision.accepted:
                return response
            self.eef_actions[side] = CachedEefAction(
                target=target,
                epoch=self.machine.epoch,
                sequence=sequence,
                source=source,
                received_at=time.monotonic(),
            )
            self.actions[side] = None
            response["applied"] = True
            return response

    def _fail_safe_locked(self, reason: str) -> dict[str, Any]:
        decision = self.machine.fail_safe(reason)
        self.actions = {side: None for side in SIDES}
        self.eef_actions = {side: None for side in SIDES}
        self.alignments = {side: None for side in SIDES}
        self._reset_sync_locked()
        self._reset_command_limiters_locked()
        result = decision.as_dict()
        self._publish_result("FAIL_SAFE", result)
        self.get_logger().error(f"control fail-safe: {reason}")
        return result

    def _start_sync_alignment_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        starts: dict[str, dict[str, list[float]]] = {}
        targets: dict[str, list[float]] = {}
        maximum_delta = 0.0
        for side in self.active_sides:
            target_action = self.sync_targets[side]
            master = self.master_state[side]
            puppet = self.puppet_state[side]
            if (
                target_action is None
                or master is None
                or puppet is None
                or not self._status_fresh_locked(side)
                or not self._fresh(master.received_at)
                or not self._fresh(puppet.received_at)
            ):
                return self._fail_safe_locked("sync_feedback_stale_before_alignment")
            starts[side] = {
                "master": list(master.position),
                "puppet": list(puppet.position),
            }
            targets[side] = list(target_action.position)
            for role in ("master", "puppet"):
                maximum_delta = max(
                    maximum_delta,
                    max(
                        abs(starts[side][role][index] - targets[side][index])
                        for index in range(JOINT_COUNT)
                    ),
                )

        alignment_speed = min(
            target.alignment_speed
            if target is not None and target.alignment_speed is not None
            else self.args.sync_alignment_speed
            for target in (self.sync_targets[side] for side in self.active_sides)
        )
        duration = max(
            self.args.sync_alignment_min_seconds,
            maximum_delta / max(0.01, alignment_speed),
        )
        if duration >= self.args.sync_alignment_timeout:
            return self._fail_safe_locked("sync_alignment_duration_exceeds_timeout")
        self.sync_alignment = SyncAlignmentPlan(
            starts=starts,
            targets=targets,
            started_at=now,
            duration=duration,
            deadline=now + self.args.sync_alignment_timeout,
            speed=alignment_speed,
        )
        decision = self.machine.mark_alignment_started()
        result = decision.as_dict()
        result.update(
            {
                "accepted": True,
                "duration": duration,
                "alignment_speed": alignment_speed,
                "maximum_delta": maximum_delta,
                "phase": "ALIGN_SYNC",
            }
        )
        self._publish_result("ALIGN_SYNC_STARTED", result)
        return result

    def _sync_feedback_reached_locked(self, plan: SyncAlignmentPlan) -> bool:
        for side in self.active_sides:
            target = plan.targets[side]
            for state in (self.master_state[side], self.puppet_state[side]):
                if state is None or not self._fresh(state.received_at):
                    return False
                joint_error = max(
                    abs(state.position[index] - target[index]) for index in range(6)
                )
                gripper_error = abs(state.position[6] - target[6])
                if (
                    joint_error > self.args.sync_position_tolerance
                    or gripper_error > self.args.sync_gripper_tolerance
                ):
                    return False
        return True

    def _publish_sync_alignment_locked(self, now: float) -> None:
        plan = self.sync_alignment
        if plan is None:
            return
        for side in self.active_sides:
            if (
                not self._status_fresh_locked(side)
                or not self._joint_fresh_locked("master", side)
                or not self._joint_fresh_locked("puppet", side)
            ):
                self._fail_safe_locked("sync_feedback_stale")
                return
        if now >= plan.deadline:
            self._fail_safe_locked("sync_alignment_timeout")
            return

        alpha = min(1.0, max(0.0, (now - plan.started_at) / plan.duration))
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        for side in self.active_sides:
            target = plan.targets[side]
            for role in ("master", "puppet"):
                start = plan.starts[side][role]
                position = [
                    start[index] + smooth * (target[index] - start[index])
                    for index in range(JOINT_COUNT)
                ]
                self._publish_command_locked(role, side, position)

        if alpha < 1.0:
            plan.stable_since = None
            return
        if not self._sync_feedback_reached_locked(plan):
            plan.stable_since = None
            return
        if plan.stable_since is None:
            plan.stable_since = now
            return
        if now - plan.stable_since < self.args.sync_stable_seconds:
            return

        for side in self.active_sides:
            first_target = self.sync_targets[side]
            if first_target is None:
                self._fail_safe_locked("sync_first_target_missing_at_completion")
                return
            self.actions[side] = CachedAction(
                position=list(first_target.position),
                epoch=self.machine.epoch,
                sequence=first_target.sequence,
                source="align_hold",
                received_at=now,
            )
        decision = self.machine.finish_alignment()
        if not decision.accepted:
            self._fail_safe_locked(decision.reason)
            return
        self.sync_alignment = None
        self.sync_targets = {side: None for side in SIDES}
        result = decision.as_dict()
        result["phase"] = "RLT_SYNC"
        self._publish_result("ALIGN_SYNC_COMPLETE", result)

    def _make_command(
        self,
        position: list[float],
        source: Optional[JointSnapshot] = None,
    ) -> JointState:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(source.name) if source is not None else [f"joint{i}" for i in range(7)]
        message.position = [float(value) for value in position[:JOINT_COUNT]]
        # Piper's ROS driver does not consume this as per-joint velocity. It
        # treats velocity[6] as one global speed percentage, so keep the field
        # zero. When smoothing is enabled, bounded 200 Hz position increments
        # implement the limits instead. Teach direct-follow bypasses them.
        message.velocity = [0.0] * JOINT_COUNT
        message.effort = [0.0] * 6 + [1.0]
        return message

    def _publish_command_locked(
        self,
        role: str,
        side: str,
        position: list[float],
        *,
        apply_smoothing: bool = True,
    ) -> None:
        source = self.master_state[side] if role == "master" else self.puppet_state[side]
        command_position = list(position)
        command_velocity = [0.0] * JOINT_COUNT
        if apply_smoothing and self.args.command_smoothing:
            command_position, command_velocity = self.command_limiters[side][role].step(
                position,
                now=time.monotonic(),
                measured_position=source.position if source is not None else None,
            )
        message = self._make_command(command_position, source)
        publisher = (
            self.master_command_publishers[side]
            if role == "master"
            else self.puppet_command_publishers[side]
        )
        publisher.publish(message)
        self.last_applied[side][role] = list(command_position)
        self.last_applied_velocity[side][role] = list(command_velocity)
        self._applied_sequence += 1

    def _publish_eef_command_locked(
        self, role: str, side: str, target: list[float], *, epoch: int, sequence: int | None
    ) -> None:
        message = PosCmd()
        message.x, message.y, message.z = (float(value) for value in target[:3])
        message.roll, message.pitch, message.yaw = rotvec_to_euler_xyz(target[3:6])
        message.gripper = float(target[6])
        # PosCmd has no Header. The SDK ignores mode1/mode2; the arbiter uses
        # them only as an epoch/sequence carrier on the policy input path.
        message.mode1 = int(epoch)
        message.mode2 = int(sequence or 0) % 2_000_000_000
        publisher = (
            self.master_eef_command_publishers[side]
            if role == "master"
            else self.puppet_eef_command_publishers[side]
        )
        publisher.publish(message)
        self.last_applied[side][role] = list(target)
        self.last_applied_velocity[side][role] = [0.0] * JOINT_COUNT
        self._applied_sequence += 1

    def _publish_policy_locked(self, now: float) -> None:
        targets = self.machine.policy_targets()
        if not targets:
            return
        for side in self.active_sides:
            eef_action = self.eef_actions[side]
            if eef_action is not None:
                if now - eef_action.received_at > self.args.action_timeout:
                    continue
                if eef_action.epoch != self.machine.epoch or not self._status_fresh_locked(side):
                    continue
                if not self._eef_fresh_locked("puppet", side):
                    continue
                for role in targets:
                    self._publish_eef_command_locked(
                        role,
                        side,
                        eef_action.target,
                        epoch=eef_action.epoch,
                        sequence=eef_action.sequence,
                    )
                continue
            action = self.actions[side]
            if action is None:
                continue
            timeout = (
                self.args.post_align_action_timeout
                if action.source == "align_hold"
                else self.args.action_timeout
            )
            if now - action.received_at > timeout:
                if action.source == "align_hold":
                    self._fail_safe_locked("post_alignment_action_timeout")
                    return
                continue
            if action.epoch != self.machine.epoch or not self._status_fresh_locked(side):
                continue
            for role in targets:
                self._publish_command_locked(role, side, action.position)

    def _begin_alignment_locked(self, side: str, master: JointSnapshot, puppet: JointSnapshot) -> None:
        maximum = max(
            abs(master.position[index] - puppet.position[index]) for index in range(6)
        )
        if maximum <= self.args.slow_alignment_threshold:
            self.alignments[side] = None
            return
        duration = max(
            self.args.slow_alignment_min_seconds,
            maximum / max(0.01, self.args.slow_alignment_speed),
        )
        self.alignments[side] = AlignmentState(
            start=list(puppet.position),
            target=list(master.position),
            started_at=time.monotonic(),
            duration=duration,
        )
        self.get_logger().warning(
            f"{side} teach alignment: {maximum:.3f} rad over {duration:.1f}s"
        )

    def _teach_position_locked(
        self, side: str, master: JointSnapshot, puppet: JointSnapshot, now: float
    ) -> list[float]:
        gate = self.teach_alignment_gates[side]
        if gate.follow_ready:
            return list(master.position)

        alignment = self.alignments[side]
        if alignment is None:
            maximum = max(
                abs(master.position[index] - puppet.position[index]) for index in range(6)
            )
            if gate.needs_alignment(maximum, self.args.slow_alignment_threshold):
                self._begin_alignment_locked(side, master, puppet)
                alignment = self.alignments[side]
            else:
                return list(master.position)
        if alignment is None:
            gate.finish_alignment()
            return list(master.position)

        alpha = min(1.0, max(0.0, (now - alignment.started_at) / alignment.duration))
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        result = [
            alignment.start[index]
            + smooth * (alignment.target[index] - alignment.start[index])
            for index in range(JOINT_COUNT)
        ]
        if alpha >= 1.0:
            self.alignments[side] = None
            gate.finish_alignment()
        return result

    def _publish_teach_locked(self, now: float) -> None:
        for side in self.active_sides:
            if not self.teach_flags[side]:
                continue
            master = self.master_state[side]
            puppet = self.puppet_state[side]
            if master is None or puppet is None:
                continue
            if not self._fresh(master.received_at) or not self._fresh(puppet.received_at):
                continue
            position = self._teach_position_locked(side, master, puppet, now)
            # Physical Teach is already human-rate feedback at 200 Hz. Send
            # the selected Teach trajectory directly so the generic policy
            # velocity/acceleration limiter cannot add follower lag.
            self._publish_command_locked(
                "puppet", side, position, apply_smoothing=False
            )

    def _publish_enable_locked(self, now: float) -> None:
        if now - self._last_enable_publish < 1.0:
            return
        message = Bool()
        message.data = True
        if self.machine.effective_mode == ControlMode.POLICY_PUPPET:
            for side in self.active_sides:
                self.puppet_enable_publishers[side].publish(message)
        elif self.machine.effective_mode in (ControlMode.RLT_SYNC, ControlMode.ALIGN_SYNC):
            for side in self.active_sides:
                self.master_enable_publishers[side].publish(message)
                self.puppet_enable_publishers[side].publish(message)
        elif self.machine.effective_mode == ControlMode.TEACH:
            for side in self.active_sides:
                if self.teach_flags[side]:
                    self.puppet_enable_publishers[side].publish(message)
        self._last_enable_publish = now

    def _joint_dict_locked(self, snapshot: Optional[JointSnapshot], now: float) -> Optional[dict[str, Any]]:
        if snapshot is None:
            return None
        return {
            "name": list(snapshot.name),
            "position": list(snapshot.position),
            "velocity": list(snapshot.velocity),
            "effort": list(snapshot.effort),
            "stamp_ns": snapshot.stamp_ns,
            "age_ms": round((now - snapshot.received_at) * 1000.0, 3),
            "fresh": self._fresh(snapshot.received_at),
        }

    def _eef_dict_locked(
        self, role: str, side: str, snapshot: Optional[EefSnapshot], now: float
    ) -> Optional[dict[str, Any]]:
        if snapshot is None:
            return None
        joint = self.master_state[side] if role == "master" else self.puppet_state[side]
        if joint is None:
            return None
        return {
            "position": [*snapshot.position, float(joint.position[6])],
            "stamp_ns": snapshot.stamp_ns,
            "age_ms": round((now - snapshot.received_at) * 1000.0, 3),
            "fresh": self._fresh(snapshot.received_at) and self._fresh(joint.received_at),
        }

    def _status_dict_locked(self, snapshot: Optional[StatusSnapshot], now: float) -> Optional[dict[str, Any]]:
        if snapshot is None:
            return None
        return {
            "ctrl_mode": snapshot.ctrl_mode,
            "teach_status": snapshot.teach_status,
            "arm_status": snapshot.arm_status,
            "motion_status": snapshot.motion_status,
            "err_code": snapshot.err_code,
            "age_ms": round((now - snapshot.received_at) * 1000.0, 3),
            "fresh": self._fresh(snapshot.received_at),
        }

    def get_observation(
        self, *, include_images: bool = True, image_names: Optional[list[str]] = None
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_teach_locked()
            now = time.monotonic()
            observation: dict[str, Any] = {
                "sequence": self._observation_sequence,
                "timestamp_ns": time.time_ns(),
                "control": self.machine.snapshot(),
                "teach_sides": dict(self.teach_flags),
                "master": {
                    side: self._joint_dict_locked(self.master_state[side], now) for side in SIDES
                },
                "puppet": {
                    side: self._joint_dict_locked(self.puppet_state[side], now) for side in SIDES
                },
                "master_eef": {
                    side: self._eef_dict_locked("master", side, self.master_eef_state[side], now)
                    for side in SIDES
                },
                "puppet_eef": {
                    side: self._eef_dict_locked("puppet", side, self.puppet_eef_state[side], now)
                    for side in SIDES
                },
                "master_status": {
                    side: self._status_dict_locked(self.master_status[side], now) for side in SIDES
                },
                "last_applied": copy.deepcopy(self.last_applied),
                "last_applied_velocity": copy.deepcopy(self.last_applied_velocity),
                "teach_alignment": {
                    side: {
                        "trajectory_active": self.alignments[side] is not None,
                        "follow_ready": self.teach_alignment_gates[side].follow_ready,
                    }
                    for side in SIDES
                },
                "applied_sequence": self._applied_sequence,
            }
            if include_images:
                selected = image_names or list(self.images)
                image_payload: dict[str, Any] = {}
                for name in selected:
                    if name not in self.images:
                        continue
                    image = self.images[name]
                    if image is None:
                        image_payload[name] = None
                    else:
                        image_payload[name] = {
                            "encoding": image.fmt,
                            "stamp_ns": image.stamp_ns,
                            "age_ms": round((now - image.received_at) * 1000.0, 3),
                            "fresh": self._fresh(image.received_at, timeout=1.0),
                            "data": image.data,
                        }
                observation["images"] = image_payload
            return observation

    def _publish_state_locked(self, now: float) -> None:
        if now - self._last_state_publish < 0.25:
            return
        state = self.machine.snapshot()
        state["teach_sides"] = dict(self.teach_flags)
        state["teach_alignment"] = {
            side: {
                "trajectory_active": self.alignments[side] is not None,
                "follow_ready": self.teach_alignment_gates[side].follow_ready,
            }
            for side in SIDES
        }
        state["align_sync"] = {
            "targets_received": {
                side: self.sync_targets[side] is not None for side in self.active_sides
            },
            "trajectory_active": self.sync_alignment is not None,
            "duration": self.sync_alignment.duration if self.sync_alignment is not None else None,
            "speed": self.sync_alignment.speed if self.sync_alignment is not None else None,
        }
        state["master_status_fresh"] = {
            side: self._status_fresh_locked(side) for side in SIDES
        }
        state_message = String()
        state_message.data = json.dumps(state, separators=(",", ":"), sort_keys=True)
        self.state_publisher.publish(state_message)
        teach_message = Bool()
        teach_message.data = self.machine.teach_active
        self.teach_publisher.publish(teach_message)
        mode_message = UInt8()
        mode_message.data = int(self.machine.effective_mode)
        self.mode_publisher.publish(mode_message)
        epoch_message = UInt64()
        epoch_message.data = int(self.machine.epoch)
        self.epoch_publisher.publish(epoch_message)
        owner_message = String()
        owner_message.data = {
            ControlMode.SAFE: "SAFE",
            ControlMode.POLICY_PUPPET: "POLICY",
            ControlMode.RLT_SYNC: "RLT",
            ControlMode.TEACH: "TEACH",
            ControlMode.RESYNC_REQUIRED: "SAFE",
            ControlMode.ALIGN_SYNC: "ALIGN",
        }[self.machine.effective_mode]
        self.owner_publisher.publish(owner_message)
        self._last_state_publish = now

    def _tick(self) -> None:
        with self._lock:
            self._refresh_teach_locked()
            now = time.monotonic()
            self._publish_state_locked(now)
            if self.machine.dry_run:
                return
            if self.machine.effective_mode == ControlMode.TEACH:
                self._publish_enable_locked(now)
                self._publish_teach_locked(now)
                return
            if self.machine.emergency_latched or not self.machine.output_enabled:
                return
            self._publish_enable_locked(now)
            if self.machine.effective_mode == ControlMode.ALIGN_SYNC:
                self._publish_sync_alignment_locked(now)
            elif self.machine.effective_mode in (
                ControlMode.POLICY_PUPPET,
                ControlMode.RLT_SYNC,
            ):
                self._publish_policy_locked(now)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cobot Magic master/puppet control arbiter")
    add_control_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rclpy.init()
    node = CobotMagicRobotControl(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
