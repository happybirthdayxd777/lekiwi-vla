"""
bridge.launch.py — Unified LeKiWi launch
=========================================
Starts the ROS2 ↔ MuJoCo bridge in simulation mode.

Usage:
  ros2 launch lekiwi_ros2_bridge bridge.launch.py
  ros2 launch lekiwi_ros2_bridge bridge.launch.py sim_mode:=false
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sim_mode = LaunchConfiguration("sim_mode", default="true")

    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        output="screen",
        parameters=[{"sim_mode": sim_mode}],
        remappings=[
            ("/lekiwi/cmd_vel",      "/lekiwi/cmd_vel"),
            ("/lekiwi/joint_states", "/lekiwi/joint_states"),
            ("/lekiwi/odom",         "/lekiwi/odom"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "sim_mode",
            default_value="true",
            description="Run in simulation (MuJoCo) mode vs real robot mode",
        ),
        bridge_node,
    ])
