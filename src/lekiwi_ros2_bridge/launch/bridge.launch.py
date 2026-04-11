"""
Unified LeKiWi Launch — Simulation mode switch
===============================================

Usage:
  # Primitive cylinders (fast, stable)
  ros2 launch lekiwi_ros2_bridge bridge.launch.py sim_type:=primitive

  # Real STL mesh geometry (from lekiwi_modular URDF)
  ros2 launch lekiwi_ros2_bridge bridge.launch.py sim_type:=urdf

The bridge node subscribes to /lekiwi/cmd_vel and publishes:
  /lekiwi/joint_states           — joint positions + velocities
  /lekiwi/camera/image_raw       — front camera image (20 Hz)
  /lekiwi/wrist_camera/image_raw — wrist camera image (20 Hz, URDF mode only)
  /lekiwi/wheel_N/cmd_vel       — wheel velocity commands
"""

import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    sim_type_arg = DeclareLaunchArgument(
        "sim_type",
        default_value="primitive",
        description=(
            "Simulation type: 'primitive' (fast cylinders) "
            "or 'urdf' (real STL meshes from lekiwi_modular)"
        ),
    )
    sim_type = LaunchConfiguration("sim_type")

    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        output="screen",
        parameters=[{"sim_type": sim_type}],
    )

    ld = LaunchDescription([sim_type_arg, bridge_node])
    return ld
