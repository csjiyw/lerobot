#!/usr/bin/env python3
"""Launch the three serial-selected Cobot Magic RealSense cameras."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _launch_file(package_name: str, filename: str) -> str:
    return os.path.join(get_package_share_directory(package_name), "launch", filename)


def _include(package_name: str, filename: str, arguments: dict):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_launch_file(package_name, filename)),
        launch_arguments=arguments.items(),
    )


def _realsense(camera_name: str, serial_number: str):
    return _include(
        "realsense2_camera",
        "rs_launch.py",
        {
            "camera_name": camera_name,
            "camera_namespace": "",
            "serial_no": f"_{serial_number}",
            "log_level": "info",
            "enable_color": "true",
            "rgb_camera.color_profile": "640,480,30",
            "rgb_camera.color_format": "RGB8",
            "enable_depth": "false",
            "enable_infra": "false",
            "enable_infra1": "false",
            "enable_infra2": "false",
            "enable_rgbd": "false",
            "enable_sync": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
            "pointcloud.enable": "false",
            "align_depth.enable": "false",
        },
    )


def generate_launch_description():
    return LaunchDescription(
        [
            _realsense("camera_f", "042222071914"),
            _realsense("camera_l", "337122071405"),
            _realsense("camera_r", "337322073221"),
        ]
    )
