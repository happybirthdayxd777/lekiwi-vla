"""
bridge.launch.py — Unified launch for ROS2-LeKiWi Bridge

Usage:
  # Start bridge (sim mode)
  ros2 launch lekiwi_ros2_bridge bridge.launch.py

  # With lekiwi_modular (real or gazebo sim)
  ros2 launch lekiwi_controller control.launch.py &
  ros2 launch lekiwi_ros2_bridge bridge.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_mode = LaunchConfiguration("sim_mode", default="true")

    # The bridge node
    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        output="screen",
        parameters=[{
            "sim_mode": sim_mode,
        }],
        remappings=[
            # Bridge input: /lekiwi/cmd_vel from teleop or joystick
            ("/lekiwi/cmd_vel", "/lekiwi/cmd_vel"),
            # Bridge output: joint states in standard ROS2 format
            ("/lekiwi/joint_states", "/lekiwi/joint_states"),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "sim_mode",
            default_value="true",
            description="Use MuJoCo simulation (true) or passthrough to real robot (false)"
        ),
        bridge_node,
    ])
