#!/usr/bin/env python3
"""Read-only AGX ROS client for the local 7D RLT IPC server.

This client creates ROS subscriptions only.  It has no ROS publishers and
never sends an action to Piper.  Joint order is joint0..joint6: six arm
joints followed by the gripper.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState


LOG = logging.getLogger("rlt_robot_ipc_client")
JOINT_NAMES = tuple(f"joint{i}" for i in range(7))
ACTION_STEPS = 16
UPDATE_KEYS = ("state", "cam_front", "cam_side")
SNAPSHOT_KEYS = ("state", "action", "cam_front", "cam_side")


def pack_arrays(
    message_type: bytes,
    arrays: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> list[bytes]:
    metadata: dict[str, dict[str, Any]] = {}
    payload: list[bytes] = []
    for key in keys:
        array = np.ascontiguousarray(arrays[key])
        metadata[key] = {"shape": list(array.shape), "dtype": array.dtype.str}
        payload.append(array.tobytes(order="C"))
    return [
        message_type,
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
        *payload,
    ]


def decode_snapshot(frames: list[bytes]) -> dict[str, np.ndarray]:
    if len(frames) != 2 + len(SNAPSHOT_KEYS) or frames[0] != b"SNAPSHOT":
        raise RuntimeError("malformed SNAPSHOT multipart message")
    metadata = json.loads(frames[1].decode("utf-8"))
    decoded: dict[str, np.ndarray] = {}
    for index, key in enumerate(SNAPSHOT_KEYS):
        descriptor = metadata[key]
        shape = tuple(int(value) for value in descriptor["shape"])
        dtype = np.dtype(descriptor["dtype"])
        raw = frames[2 + index]
        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if len(raw) != expected_bytes:
            raise RuntimeError(f"{key} bytes {len(raw)} != {expected_bytes}")
        decoded[key] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return decoded


class ReadOnlyNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("rlt_readonly_observation_client")
        self.latest_state: np.ndarray | None = None
        self.joint_samples = 0
        self.camera_samples = {"front": 0, "side": 0}
        self.camera_images: dict[str, np.ndarray | None] = {
            "front": None,
            "side": None,
        }
        self.warned_fallback = False
        self.warned_decode = False
        # Subscriptions only; no action publishers are constructed.
        self.create_subscription(JointState, args.state_topic, self.on_joint, 10)
        self.create_subscription(
            CompressedImage,
            args.front_topic,
            lambda message: self.on_camera("front", message),
            10,
        )
        self.create_subscription(
            CompressedImage,
            args.side_topic,
            lambda message: self.on_camera("side", message),
            10,
        )

    def on_joint(self, message: JointState) -> None:
        values = np.asarray(message.position, dtype=np.float32)
        if all(name in message.name for name in JOINT_NAMES):
            by_name = dict(zip(message.name, values.tolist()))
            values = np.asarray(
                [by_name[name] for name in JOINT_NAMES], dtype=np.float32
            )
        elif values.size >= 7:
            if not self.warned_fallback:
                self.get_logger().warning(
                    "using first seven JointState positions"
                )
                self.warned_fallback = True
            values = values[:7]
        else:
            return
        self.latest_state = values.reshape(1, 7)
        self.joint_samples += 1

    def on_camera(self, name: str, message: CompressedImage) -> None:
        try:
            import cv2

            encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("invalid compressed image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            self.camera_images[name] = np.transpose(rgb, (2, 0, 1))[None].astype(
                np.uint8, copy=False
            )
            self.camera_samples[name] += 1
        except Exception as exc:
            if not self.warned_decode:
                self.get_logger().warning(
                    "camera decode failed; using zero frame: %s", exc
                )
                self.warned_decode = True


class Client:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        timeout_ms = max(1, int(timeout_s * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()

    def round_trip(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.socket.send_multipart(
            pack_arrays(b"UPDATE", observation, UPDATE_KEYS)
        )
        ack = self.socket.recv_multipart()
        if not ack or ack[0] != b"ACK":
            raise RuntimeError(f"unexpected ACK: {ack!r}")
        self.socket.send_multipart([b"WAIT"])
        return decode_snapshot(self.socket.recv_multipart())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, default=7999)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--wait-for-state", type=float, default=10.0)
    parser.add_argument("--state-topic", default="/puppet/joint_left")
    parser.add_argument(
        "--front-topic",
        default="/camera_f/color/image_raw/compressed",
    )
    parser.add_argument(
        "--side-topic",
        default="/camera_l/color/image_raw/compressed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    rclpy.init()
    node = ReadOnlyNode(args)
    client = Client(args.server_host, args.server_port, args.timeout)
    try:
        deadline = time.monotonic() + args.wait_for_state
        while node.latest_state is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_state is None:
            raise RuntimeError("no 7D JointState received")

        for cycle in range(1, args.cycles + 1):
            rclpy.spin_once(node, timeout_sec=0.1)
            state = node.latest_state.copy()
            zero_image = np.zeros((1, 3, 224, 224), dtype=np.uint8)
            front_image = node.camera_images["front"]
            side_image = node.camera_images["side"]
            observation = {
                "state": state.astype(np.float32, copy=False),
                "cam_front": (
                    front_image if front_image is not None else zero_image
                ),
                "cam_side": side_image if side_image is not None else zero_image,
            }
            snapshot = client.round_trip(observation)
            action = np.asarray(snapshot["action"], dtype=np.float32)
            expected = np.repeat(state, ACTION_STEPS, axis=0)
            if action.shape != (ACTION_STEPS, 7) or not np.allclose(
                action, expected
            ):
                raise RuntimeError(
                    "hold response does not match the received 7D state"
                )
            LOG.info(
                "communication_ok cycle=%d state=%s action=%s "
                "joint_samples=%d camera_samples=%s action_publisher=False",
                cycle,
                state.shape,
                action.shape,
                node.joint_samples,
                node.camera_samples,
            )
    finally:
        client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
