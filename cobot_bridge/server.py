"""AGX ROS-to-ZeroMQ bridge for Cobot Magic.

The bridge is deliberately not a robot controller. It reads canonical ROS
topics and forwards policy requests only to ``/policy/*`` and ``/control/*``.
The ROS control arbiter remains the sole publisher on physical command paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import zmq

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Empty, String, UInt8, UInt64

from .protocol import (
    CAMERAS,
    ACTION_SPACES,
    JOINT_COUNT,
    REQUESTABLE_MODES,
    SIDES,
    ProtocolError,
    decode_json,
    encode_json,
    make_response,
    pack_observation_response,
    validate_action,
    validate_action_chunk,
    interpolate_action_chunk,
    validate_names,
    validate_request,
)


LOG = logging.getLogger("cobot_bridge.server")


def _stamp_ns(message: Any) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


@dataclass
class JointSample:
    name: list[str]
    position: list[float]
    velocity: list[float]
    effort: list[float]
    stamp_ns: int
    received_at: float


@dataclass
class EefSample:
    position: list[float]
    stamp_ns: int
    received_at: float


@dataclass
class ImageSample:
    data: bytes
    encoding: str
    stamp_ns: int
    received_at: float


@dataclass
class ActionChunk:
    actions: list[list[float]]
    epoch: int
    sequence: int
    waypoint_hz: float
    action_space: str
    started_at: float


class CobotBridgeRosNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("cobot_magic_zmq_bridge")
        self.args = args
        self._lock = threading.RLock()
        self._result_condition = threading.Condition(self._lock)
        self._observation_sequence = 0
        self._result_sequence = 0
        self._control_state: dict[str, Any] = {}
        self._control_epoch: int | None = None
        self._effective_mode: int | None = None
        self._teach_active: bool | None = None
        self._owner: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._action_chunks: dict[str, ActionChunk | None] = {side: None for side in SIDES}
        self._last_chunk_sequence: dict[str, int] = {side: -1 for side in SIDES}

        self._states: dict[str, dict[str, JointSample | None]] = {
            role: {side: None for side in SIDES} for role in ("master", "puppet")
        }
        self._eef_states: dict[str, dict[str, EefSample | None]] = {
            role: {side: None for side in SIDES} for role in ("master", "puppet")
        }
        self._commands: dict[str, dict[str, JointSample | None]] = {
            role: {side: None for side in SIDES} for role in ("master", "puppet")
        }
        self._images: dict[str, ImageSample | None] = {name: None for name in CAMERAS}

        state_topics = {
            ("master", "left"): "/master/left_state",
            ("master", "right"): "/master/right_state",
            ("puppet", "left"): "/puppet/left_state",
            ("puppet", "right"): "/puppet/right_state",
        }
        command_topics = {
            ("master", "left"): "/master/left_cmd",
            ("master", "right"): "/master/right_cmd",
            ("puppet", "left"): "/puppet/left_cmd",
            ("puppet", "right"): "/puppet/right_cmd",
        }
        eef_state_topics = {
            ("master", "left"): "/master/left_eef_state",
            ("master", "right"): "/master/right_eef_state",
            ("puppet", "left"): "/puppet/left_eef_state",
            ("puppet", "right"): "/puppet/right_eef_state",
        }
        camera_topics = {
            "front": "/camera_f/color/image_raw/compressed",
            "left": "/camera_l/color/image_raw/compressed",
            "right": "/camera_r/color/image_raw/compressed",
        }

        for (role, side), topic in state_topics.items():
            self.create_subscription(
                JointState,
                topic,
                lambda message, r=role, s=side: self._joint_callback(
                    self._states, r, s, message
                ),
                10,
            )
        for (role, side), topic in command_topics.items():
            self.create_subscription(
                JointState,
                topic,
                lambda message, r=role, s=side: self._joint_callback(
                    self._commands, r, s, message
                ),
                10,
            )
        for (role, side), topic in eef_state_topics.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda message, r=role, s=side: self._eef_callback(r, s, message),
                10,
            )
        for name, topic in camera_topics.items():
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, camera=name: self._image_callback(camera, message),
                10,
            )

        self.create_subscription(String, "/control/state", self._state_callback, 10)
        self.create_subscription(UInt64, "/control/epoch", self._epoch_callback, 10)
        self.create_subscription(UInt8, "/control/effective_mode", self._mode_callback, 10)
        self.create_subscription(Bool, "/control/teach_active", self._teach_callback, 10)
        self.create_subscription(String, "/control/owner", self._owner_callback, 10)
        self.create_subscription(String, "/control/action_result", self._result_callback, 10)

        self._mode_publisher = self.create_publisher(UInt8, "/control/request_mode", 10)
        self._output_publisher = self.create_publisher(Bool, "/control/output_enabled", 10)
        self._resync_publisher = self.create_publisher(Empty, "/control/resync", 10)
        self._emergency_publisher = self.create_publisher(Empty, "/control/emergency_safe", 10)
        self._clear_emergency_publisher = self.create_publisher(
            Empty, "/control/clear_emergency", 10
        )
        self._action_publishers = {
            side: self.create_publisher(JointState, f"/policy/{side}_cmd", 10)
            for side in SIDES
        }
        self._eef_action_publishers = {
            side: self.create_publisher(PosCmd, f"/policy/{side}_eef_cmd", 10)
            for side in SIDES
        }
        self.create_timer(1.0 / 200.0, self._chunk_tick)

    def _clear_chunks_locked(self) -> None:
        self._action_chunks = {side: None for side in SIDES}

    def _joint_callback(
        self,
        target: dict[str, dict[str, JointSample | None]],
        role: str,
        side: str,
        message: JointState,
    ) -> None:
        position = [float(value) for value in message.position[:JOINT_COUNT]]
        if len(position) != JOINT_COUNT or not _finite(position):
            return
        velocity = [float(value) for value in message.velocity[:JOINT_COUNT]]
        velocity += [0.0] * (JOINT_COUNT - len(velocity))
        effort = [float(value) for value in message.effort[:JOINT_COUNT]]
        effort += [0.0] * (JOINT_COUNT - len(effort))
        names = list(message.name[:JOINT_COUNT])
        names += [f"joint{index}" for index in range(len(names), JOINT_COUNT)]
        sample = JointSample(
            name=names,
            position=position,
            velocity=velocity,
            effort=effort,
            stamp_ns=_stamp_ns(message),
            received_at=time.monotonic(),
        )
        with self._lock:
            target[role][side] = sample
            self._observation_sequence += 1

    def _image_callback(self, name: str, message: CompressedImage) -> None:
        sample = ImageSample(
            data=bytes(message.data),
            encoding=str(message.format or "jpeg"),
            stamp_ns=_stamp_ns(message),
            received_at=time.monotonic(),
        )
        with self._lock:
            self._images[name] = sample
            self._observation_sequence += 1

    @staticmethod
    def _quaternion_to_rotvec(x: float, y: float, z: float, w: float) -> list[float]:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("invalid_eef_quaternion")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w
        sin_half = math.sqrt(x * x + y * y + z * z)
        if sin_half <= 1e-12:
            return [0.0, 0.0, 0.0]
        angle = 2.0 * math.atan2(sin_half, max(1e-12, w))
        scale = angle / sin_half
        return [x * scale, y * scale, z * scale]

    def _eef_callback(self, role: str, side: str, message: PoseStamped) -> None:
        pose = message.pose
        try:
            position = [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                *self._quaternion_to_rotvec(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ),
            ]
        except (TypeError, ValueError):
            return
        if not _finite(position):
            return
        sample = EefSample(
            position=position,
            stamp_ns=_stamp_ns(message),
            received_at=time.monotonic(),
        )
        with self._lock:
            self._eef_states[role][side] = sample
            self._observation_sequence += 1

    def _state_callback(self, message: String) -> None:
        try:
            state = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(state, dict):
            return
        with self._lock:
            self._control_state = state
            mode = int(state.get("effective_mode_id", -1))
            if (
                mode not in (1, 2)
                or bool(state.get("teach_active", False))
            ):
                self._clear_chunks_locked()
            self._observation_sequence += 1

    def _epoch_callback(self, message: UInt64) -> None:
        with self._lock:
            epoch = int(message.data)
            if self._control_epoch is not None and epoch != self._control_epoch:
                self._clear_chunks_locked()
            self._control_epoch = epoch

    def _mode_callback(self, message: UInt8) -> None:
        with self._lock:
            self._effective_mode = int(message.data)
            if self._effective_mode not in (1, 2):
                self._clear_chunks_locked()

    def _teach_callback(self, message: Bool) -> None:
        with self._lock:
            self._teach_active = bool(message.data)
            if self._teach_active:
                self._clear_chunks_locked()

    def _owner_callback(self, message: String) -> None:
        with self._lock:
            self._owner = str(message.data)

    def _result_callback(self, message: String) -> None:
        try:
            result = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(result, dict):
            return
        with self._result_condition:
            self._last_result = result
            if "epoch" in result:
                try:
                    self._control_epoch = int(result["epoch"])
                except (TypeError, ValueError):
                    pass
            if "effective_mode_id" in result:
                try:
                    self._effective_mode = int(result["effective_mode_id"])
                    self._owner = {
                        0: "SAFE",
                        1: "POLICY",
                        2: "RLT",
                        3: "TEACH",
                        4: "SAFE",
                        5: "ALIGN",
                    }.get(self._effective_mode, self._owner)
                except (TypeError, ValueError):
                    pass
            if (
                result.get("command") == "SET_ACTION"
                and not bool(result.get("accepted", False))
                and result.get("reason")
                in {"teach_active", "resync_required", "owner_safe", "output_disabled", "stale_epoch"}
            ):
                side = result.get("side")
                if side in SIDES:
                    self._action_chunks[side] = None
            self._result_sequence += 1
            self._result_condition.notify_all()

    def _chunk_tick(self) -> None:
        publications: list[tuple[str, JointState]] = []
        eef_publications: list[tuple[str, PosCmd]] = []
        with self._lock:
            now = time.monotonic()
            mode = self._effective_mode
            teach = self._teach_active
            epoch = self._control_epoch
            for side, chunk in self._action_chunks.items():
                if chunk is None:
                    continue
                if mode not in (1, 2) or teach or epoch != chunk.epoch:
                    self._action_chunks[side] = None
                    continue
                position, waypoint_index, finished = interpolate_action_chunk(
                    chunk.actions, now - chunk.started_at, chunk.waypoint_hz
                )
                if chunk.action_space == "eef_target":
                    # The PC/rollout boundary has already converted the
                    # checkpoint's model TCP to the SDK link6 frame.  The
                    # bridge only schedules this link6 PosCmd at 200 Hz; it
                    # must not apply another TCP offset here.
                    message = self._make_pos_cmd(
                        position,
                        epoch=chunk.epoch,
                        sequence=chunk.sequence * 1_000_000 + waypoint_index,
                    )
                    eef_publications.append((side, message))
                else:
                    message = JointState()
                    message.header.stamp = self.get_clock().now().to_msg()
                    message.header.frame_id = (
                        f"epoch={chunk.epoch};seq={chunk.sequence * 1_000_000 + waypoint_index}"
                    )
                    message.name = [f"joint{index}" for index in range(JOINT_COUNT)]
                    message.position = position
                    message.velocity = [0.0] * JOINT_COUNT
                    message.effort = [0.0] * JOINT_COUNT
                    publications.append((side, message))
                if finished:
                    self._action_chunks[side] = None
        for side, message in publications:
            self._action_publishers[side].publish(message)
        for side, message in eef_publications:
            self._eef_action_publishers[side].publish(message)

    def _joint_dict(self, sample: JointSample | None, now: float) -> dict[str, Any] | None:
        if sample is None:
            return None
        age_ms = (now - sample.received_at) * 1000.0
        return {
            "name": list(sample.name),
            "position": list(sample.position),
            "velocity": list(sample.velocity),
            "effort": list(sample.effort),
            "stamp_ns": sample.stamp_ns,
            "age_ms": round(age_ms, 3),
            "fresh": age_ms <= self.args.state_timeout_ms,
        }

    def _eef_dict(
        self, role: str, side: str, sample: EefSample | None, now: float
    ) -> dict[str, Any] | None:
        if sample is None:
            return None
        joint = self._states[role][side]
        if joint is None:
            return None
        age_ms = (now - sample.received_at) * 1000.0
        position = list(sample.position) + [float(joint.position[6])]
        return {
            "position": position,
            "stamp_ns": sample.stamp_ns,
            "age_ms": round(age_ms, 3),
            "fresh": age_ms <= self.args.state_timeout_ms
            and (now - joint.received_at) * 1000.0 <= self.args.state_timeout_ms,
        }

    @staticmethod
    def _make_pos_cmd(
        target: list[float], *, epoch: int, sequence: int | None = None
    ) -> PosCmd:
        message = PosCmd()
        message.x, message.y, message.z = (float(value) for value in target[:3])
        message.roll, message.pitch, message.yaw = (
            float(value) for value in CobotBridgeRosNode._rotvec_to_euler_xyz(target[3:6])
        )
        message.gripper = float(target[6])
        # The Piper PosCmd message has no header. These ignored fields carry
        # the bridge epoch/sequence to the arbiter for stale-command checks.
        message.mode1 = int(epoch)
        message.mode2 = int(sequence or 0) % 2_000_000_000
        return message

    @staticmethod
    def _rotvec_to_euler_xyz(rotvec: list[float]) -> tuple[float, float, float]:
        rx, ry, rz = (float(value) for value in rotvec)
        theta = math.sqrt(rx * rx + ry * ry + rz * rz)
        if theta <= 1e-12:
            return 0.0, 0.0, 0.0
        x, y, z = rx / theta, ry / theta, rz / theta
        c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
        r11 = c + x * x * one_c
        r12 = x * y * one_c - z * s
        r13 = x * z * one_c + y * s
        r21 = y * x * one_c + z * s
        r22 = c + y * y * one_c
        r23 = y * z * one_c - x * s
        r31 = z * x * one_c - y * s
        r32 = z * y * one_c + x * s
        r33 = c + z * z * one_c
        # Piper uses XYZ extrinsic RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        pitch = math.asin(max(-1.0, min(1.0, -r31)))
        if abs(math.cos(pitch)) > 1e-8:
            roll = math.atan2(r32, r33)
            yaw = math.atan2(r21, r11)
        else:
            roll = math.atan2(-r23, r22)
            yaw = 0.0
        return roll, pitch, yaw

    def snapshot(
        self, sides: list[str], cameras: list[str]
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        with self._lock:
            now = time.monotonic()
            control = copy.deepcopy(self._control_state)
            if self._control_epoch is not None:
                control["epoch"] = self._control_epoch
            if self._effective_mode is not None:
                control["effective_mode_id"] = self._effective_mode
            if self._teach_active is not None:
                control["teach_active"] = self._teach_active
            if self._owner is not None:
                control["owner"] = self._owner

            states = {
                role: {
                    side: self._joint_dict(self._states[role][side], now) for side in sides
                }
                for role in ("master", "puppet")
            }
            commands = {
                role: {
                    side: self._joint_dict(self._commands[role][side], now) for side in sides
                }
                for role in ("master", "puppet")
            }
            eef_states = {
                role: {
                    side: self._eef_dict(role, side, self._eef_states[role][side], now)
                    for side in sides
                }
                for role in ("master", "puppet")
            }

            image_meta: dict[str, Any] = {}
            image_frames: dict[str, bytes] = {}
            for name in cameras:
                sample = self._images[name]
                if sample is None:
                    image_meta[name] = None
                    continue
                age_ms = (now - sample.received_at) * 1000.0
                image_meta[name] = {
                    "encoding": sample.encoding,
                    "stamp_ns": sample.stamp_ns,
                    "age_ms": round(age_ms, 3),
                    "fresh": age_ms <= self.args.image_timeout_ms,
                }
                image_frames[name] = sample.data

            missing: list[str] = []
            if not control:
                missing.append("control")
            for role in ("master", "puppet"):
                for side in sides:
                    value = states[role][side]
                    if value is None or not value["fresh"]:
                        missing.append(f"{role}.{side}")
            for name in cameras:
                value = image_meta[name]
                if value is None or not value["fresh"]:
                    missing.append(f"image.{name}")

            observation = {
                "bridge_time_ns": time.time_ns(),
                "sequence": self._observation_sequence,
                "ready": not missing,
                "missing_or_stale": missing,
                "control": control,
                "master": states["master"],
                "puppet": states["puppet"],
                "master_eef": eef_states["master"],
                "puppet_eef": eef_states["puppet"],
                "commands": commands,
                "images": image_meta,
            }
            return observation, image_frames

    def _wait_for_result(
        self,
        previous_sequence: int,
        command: str,
        *,
        side: str | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.command_timeout_s
        with self._result_condition:
            while time.monotonic() < deadline:
                result = self._last_result
                if self._result_sequence > previous_sequence and result is not None:
                    if result.get("command") == command and (
                        side is None or result.get("side") == side
                    ):
                        return copy.deepcopy(result)
                self._result_condition.wait(timeout=max(0.0, deadline - time.monotonic()))
        return {
            "accepted": False,
            "command": command,
            "reason": "control_result_timeout",
            **({"side": side} if side is not None else {}),
        }

    def _publish_and_wait(
        self,
        publisher: Any,
        message: Any,
        command: str,
        *,
        side: str | None = None,
    ) -> dict[str, Any]:
        # NOTE: the read-only opt-in gate was removed at operator request
        # (2026-09): control forwarding is always on. Real safety still comes
        # from the ROS arbiter (mode/teach/epoch/emergency gates).
        with self._lock:
            previous = self._result_sequence
        publisher.publish(message)
        return self._wait_for_result(previous, command, side=side)

    def set_mode(self, mode: int) -> dict[str, Any]:
        message = UInt8()
        message.data = int(mode)
        result = self._publish_and_wait(self._mode_publisher, message, "SET_MODE")
        if result.get("accepted", False) and int(mode) == 0:
            with self._lock:
                self._clear_chunks_locked()
        return result

    def set_output(self, enabled: bool) -> dict[str, Any]:
        message = Bool()
        message.data = bool(enabled)
        result = self._publish_and_wait(self._output_publisher, message, "SET_OUTPUT")
        if result.get("accepted", False):
            with self._lock:
                self._control_state["output_enabled"] = bool(enabled)
                if not enabled:
                    self._clear_chunks_locked()
        return result

    def resync(self) -> dict[str, Any]:
        result = self._publish_and_wait(self._resync_publisher, Empty(), "RESYNC")
        with self._lock:
            self._clear_chunks_locked()
        return result

    def emergency_safe(self) -> dict[str, Any]:
        result = self._publish_and_wait(
            self._emergency_publisher, Empty(), "EMERGENCY_SAFE"
        )
        with self._lock:
            self._clear_chunks_locked()
        return result

    def clear_emergency(self) -> dict[str, Any]:
        result = self._publish_and_wait(
            self._clear_emergency_publisher, Empty(), "CLEAR_EMERGENCY"
        )
        with self._lock:
            self._clear_chunks_locked()
        return result

    def set_action_chunk(
        self,
        side: str,
        actions: list[list[float]],
        epoch: int,
        sequence: int,
        waypoint_hz: float,
        action_space: str = "joint",
    ) -> dict[str, Any]:
        with self._lock:
            mode = self._effective_mode
            if self._teach_active:
                return {"accepted": False, "reason": "teach_active", "side": side}
            if mode not in (1, 2):
                return {"accepted": False, "reason": "owner_safe", "side": side}
            if self._control_epoch != epoch:
                return {
                    "accepted": False,
                    "reason": "stale_epoch",
                    "side": side,
                    "epoch": self._control_epoch,
                }
            if sequence <= self._last_chunk_sequence[side]:
                return {"accepted": False, "reason": "stale_chunk_sequence", "side": side}
            self._last_chunk_sequence[side] = sequence
            self._action_chunks[side] = ActionChunk(
                actions=actions,
                epoch=epoch,
                sequence=sequence,
                waypoint_hz=waypoint_hz,
                action_space=action_space,
                started_at=time.monotonic(),
            )
        return {
            "accepted": True,
            "reason": "chunk_queued",
            "side": side,
            "epoch": epoch,
            "sequence": sequence,
            "waypoints": len(actions),
            "waypoint_hz": waypoint_hz,
            "output_hz": 200,
            "duration_s": (len(actions) - 1) / waypoint_hz,
            "action_space": action_space,
        }

    def set_action(
        self,
        side: str,
        action: list[float],
        epoch: int,
        sequence: int | None,
        alignment_speed: float | None = None,
        action_space: str = "joint",
    ) -> dict[str, Any]:
        if action_space == "eef_target":
            message = self._make_pos_cmd(action, epoch=epoch, sequence=sequence)
            return self._publish_and_wait(
                self._eef_action_publishers[side], message, "SET_ACTION", side=side
            )
        if action_space != "joint":
            raise ValueError(f"invalid_action_space:{action_space}")
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        frame_id = f"epoch={int(epoch)}"
        if sequence is not None:
            frame_id += f";seq={int(sequence)}"
        if alignment_speed is not None:
            frame_id += f";align_speed={float(alignment_speed):.6g}"
        message.header.frame_id = frame_id
        message.name = [f"joint{index}" for index in range(JOINT_COUNT)]
        message.position = list(action)
        message.velocity = [0.0] * JOINT_COUNT
        message.effort = [0.0] * JOINT_COUNT
        return self._publish_and_wait(
            self._action_publishers[side], message, "SET_ACTION", side=side
        )


class ZmqBridgeServer:
    def __init__(self, node: CobotBridgeRosNode, args: argparse.Namespace) -> None:
        self.node = node
        self.args = args
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 500)
        self._socket.setsockopt(zmq.SNDTIMEO, 2000)
        self._socket.setsockopt(zmq.MAXMSGSIZE, int(args.max_request_bytes))
        self._socket.bind(args.bind)
        self._stop = False

    def close(self) -> None:
        self._stop = True
        self._socket.close(0)
        self._context.term()

    def _authorize(self, request: Mapping[str, Any]) -> None:
        if self.args.token and request.get("token") != self.args.token:
            raise ProtocolError("unauthorized")

    def dispatch(self, request: dict[str, Any]) -> list[bytes]:
        self._authorize(request)
        command = request["command"]
        if command == "PING":
            return [
                encode_json(
                    make_response(
                        request,
                        ok=True,
                        result={
                            "server_time_ns": time.time_ns(),
                            "allow_control": bool(self.args.allow_control),
                            "ros_node": self.node.get_name(),
                        },
                    )
                )
            ]
        if command == "GET_OBSERVATION":
            sides = validate_names(request.get("sides", ["left"]), SIDES, "side")
            cameras = validate_names(request.get("images", list(CAMERAS)), CAMERAS, "camera")
            observation, images = self.node.snapshot(sides, cameras)
            return pack_observation_response(request, observation, images)
        if command == "SET_MODE":
            mode = int(request.get("mode", -1))
            if mode not in REQUESTABLE_MODES:
                raise ProtocolError(f"invalid_requestable_mode:{mode}")
            result = self.node.set_mode(mode)
        elif command == "SET_OUTPUT":
            enabled = request.get("enabled")
            if not isinstance(enabled, bool):
                raise ProtocolError("enabled_must_be_bool")
            result = self.node.set_output(enabled)
        elif command == "EMERGENCY_SAFE":
            result = self.node.emergency_safe()
        elif command == "CLEAR_EMERGENCY":
            result = self.node.clear_emergency()
        elif command == "RESYNC":
            result = self.node.resync()
        elif command == "SET_ACTION":
            side = validate_names([request.get("side")], SIDES, "side")[0]
            action_space = str(request.get("action_space", "joint")).lower()
            if action_space not in ACTION_SPACES:
                raise ProtocolError(f"invalid_action_space:{action_space}")
            action = validate_action(request.get("action"), action_space=action_space)
            if "epoch" not in request:
                raise ProtocolError("missing_epoch")
            sequence = request.get("sequence")
            alignment_speed = request.get("alignment_speed")
            if alignment_speed is not None:
                alignment_speed = float(alignment_speed)
                if (
                    not math.isfinite(alignment_speed)
                    or alignment_speed <= 0
                    or alignment_speed > 1.0
                ):
                    raise ProtocolError("invalid_alignment_speed")
            result = self.node.set_action(
                side,
                action,
                int(request["epoch"]),
                None if sequence is None else int(sequence),
                alignment_speed,
                action_space,
            )
        elif command == "SET_ACTION_CHUNK":
            side = validate_names([request.get("side")], SIDES, "side")[0]
            action_space = str(request.get("action_space", "joint")).lower()
            if action_space not in ACTION_SPACES:
                raise ProtocolError(f"invalid_action_space:{action_space}")
            actions = validate_action_chunk(
                request.get("actions"), action_space=action_space
            )
            if "epoch" not in request or "sequence" not in request:
                raise ProtocolError("chunk_requires_epoch_and_sequence")
            waypoint_hz = float(request.get("waypoint_hz", 0.0))
            if not math.isfinite(waypoint_hz) or waypoint_hz <= 0 or waypoint_hz > 200:
                raise ProtocolError("invalid_waypoint_hz")
            result = self.node.set_action_chunk(
                side,
                actions,
                int(request["epoch"]),
                int(request["sequence"]),
                waypoint_hz,
                action_space,
            )
        else:
            raise ProtocolError(f"unsupported_command:{command}")
        return [encode_json(make_response(request, ok=True, result=result))]

    def serve(self) -> None:
        LOG.warning(
            "bridge bound to %s (read-only gate removed: control forwarding always on; "
            "physical output remains owned by ROS arbiter)",
            self.args.bind,
        )
        while not self._stop and rclpy.ok():
            try:
                frames = self._socket.recv_multipart()
            except zmq.Again:
                continue
            if not frames:
                continue
            request: dict[str, Any] = {}
            try:
                if len(frames) != 1:
                    raise ProtocolError("request_must_be_single_json_frame")
                request = validate_request(decode_json(frames[0]))
                response = self.dispatch(request)
            except (ProtocolError, TypeError, ValueError) as exc:
                LOG.warning("reject request: %s", exc)
                response = [encode_json(make_response(request, ok=False, error=str(exc)))]
            try:
                self._socket.send_multipart(response)
            except zmq.Again:
                LOG.warning("reply timeout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://192.168.250.2:5555")
    parser.add_argument(
        "--allow-control", action="store_true",
        help="DEPRECATED (2026-09): the read-only gate was removed; "
        "control forwarding is always on. Kept for script compatibility.",
    )
    parser.add_argument("--token", default="")
    parser.add_argument("--state-timeout-ms", type=float, default=500.0)
    parser.add_argument("--image-timeout-ms", type=float, default=1000.0)
    parser.add_argument("--command-timeout-s", type=float, default=1.0)
    parser.add_argument("--max-request-bytes", type=int, default=1_000_000)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    rclpy.init()
    node = CobotBridgeRosNode(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="ros-executor", daemon=True)
    spin_thread.start()
    server = ZmqBridgeServer(node, args)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
