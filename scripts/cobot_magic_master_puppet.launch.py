#!/usr/bin/env python3
"""Four Piper ROS nodes plus the ROS-only master/puppet control arbiter.

This launch deliberately excludes the legacy cobot_magic_handover.py so only
one process can publish the canonical command topics.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _piper_node(
    name: str,
    piper_name: str,
    can_argument: str,
    command_topic: str,
    eef_command_topic: str,
):
    return Node(
        package="piper",
        executable="piper_single_ctrl",
        namespace=name,
        name="piper_ctrl",
        output="screen",
        parameters=[
            {
                "can_port": LaunchConfiguration(can_argument),
                "auto_enable": False,
                "rviz_ctrl_flag": False,
                "girpper_exist": LaunchConfiguration("gripper_exist"),
                "piper_name": piper_name,
            }
        ],
        remappings=[
            ("pos_cmd", eef_command_topic),
            ("joint_ctrl_single", command_topic),
        ],
    )


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("master_left_can", default_value="can_left_m"),
        DeclareLaunchArgument("master_right_can", default_value="can_right_m"),
        DeclareLaunchArgument("puppet_left_can", default_value="can_left_s"),
        DeclareLaunchArgument("puppet_right_can", default_value="can_right_s"),
        DeclareLaunchArgument("gripper_exist", default_value="true"),
        DeclareLaunchArgument("live_output", default_value="false"),
    ]

    arms = [
        _piper_node(
            "rear_left", "rear_left", "master_left_can", "/master/left_cmd", "/master/left_eef_cmd"
        ),
        _piper_node(
            "rear_right", "rear_right", "master_right_can", "/master/right_cmd", "/master/right_eef_cmd"
        ),
        _piper_node(
            "front_left", "left", "puppet_left_can", "/puppet/left_cmd", "/puppet/left_eef_cmd"
        ),
        _piper_node(
            "front_right", "right", "puppet_right_can", "/puppet/right_cmd", "/puppet/right_eef_cmd"
        ),
    ]

    control_path = os.path.join(os.path.dirname(__file__), "cobot_magic_robot_control.py")
    control_dry_run = ExecuteProcess(
        cmd=[sys.executable, control_path, "--no-command-smoothing"],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("live_output")),
    )
    control_live = ExecuteProcess(
        cmd=[
            sys.executable,
            control_path,
            "--live-output",
            "--no-command-smoothing",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("live_output")),
    )
    return LaunchDescription(arguments + arms + [control_dry_run, control_live])
