#!/usr/bin/env python3
"""Publish zero-valued base telemetry for Cobot Magic single-machine tests."""

from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class FakeBaseHeartbeat(Node):
    def __init__(self, rate: float, cmd_vel_topic: str, odom_topic: str) -> None:
        super().__init__("cobot_fake_base_heartbeat")
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self._frame_id = "base_link"
        self._timer = self.create_timer(1.0 / rate, self._publish_zero)
        self.get_logger().warning(
            f"SYNTHETIC ZERO BASE HEARTBEAT: {cmd_vel_topic} and {odom_topic} "
            f"at {rate:.1f} Hz; do not use these messages for training or navigation"
        )

    def _publish_zero(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._cmd_pub.publish(Twist())

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._frame_id
        self._odom_pub.publish(odom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cobot synthetic zero base heartbeat")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    rclpy.init()
    node = FakeBaseHeartbeat(args.rate, args.cmd_vel_topic, args.odom_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
