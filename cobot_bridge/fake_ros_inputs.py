"""Publish synthetic left-arm feedback for isolated-domain bridge tests only."""

from __future__ import annotations

import argparse
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from piper_msgs.msg import PiperStatusMsg


class FakeInputs(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("cobot_bridge_fake_inputs")
        self.args = args
        self.started = time.monotonic()
        self.master = self.create_publisher(JointState, "/puppet/joint_rear_left", 10)
        self.puppet = self.create_publisher(JointState, "/puppet/joint_left", 10)
        self.status = self.create_publisher(PiperStatusMsg, "/rear_left/piper_ctrl/arm_status", 10)
        self.create_timer(1.0 / args.hz, self.tick)

    def tick(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.args.duration:
            rclpy.shutdown()
            return
        teach = self.args.teach_start <= elapsed < self.args.teach_start + self.args.teach_duration
        now = self.get_clock().now().to_msg()
        names = [f"joint{index}" for index in range(7)]
        for publisher, offset in ((self.master, 0.1), (self.puppet, 0.0)):
            message = JointState()
            message.header.stamp = now
            message.name = names
            message.position = [offset] * 6 + [0.04]
            message.velocity = [0.0] * 7
            message.effort = [0.0] * 7
            publisher.publish(message)
        status = PiperStatusMsg()
        status.ctrl_mode = 0x02 if teach else 0x01
        status.teach_status = 0x01 if teach else 0x00
        self.status.publish(status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--teach-start", type=float, default=1000.0)
    parser.add_argument("--teach-duration", type=float, default=0.0)
    args = parser.parse_args()
    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    if domain in ("", "0"):
        raise RuntimeError("fake inputs refuse to run in ROS domain 0")
    rclpy.init()
    node = FakeInputs(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
