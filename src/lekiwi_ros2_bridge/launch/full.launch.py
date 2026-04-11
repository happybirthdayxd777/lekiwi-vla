"""
Unified LeKiWi Launch — Full Platform
======================================
One launch file to rule them all:

  ros2 launch lekiwi_ros2_bridge full.launch.py \
    sim_type:=primitive \
    policy:=mock

Modes:
  sim_type : primitive | urdf
  policy   : mock | pi0 | pi0_fast | act | diffusion

Starts:
  1. lekiwi_ros2_bridge (bridge node) — ROS2 ↔ MuJoCo
  2. lekiwi_vla_policy_node            — VLA policy inference
  3. Optional: Gazebo simulation (when sim_type=gazebo)
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    sim_type = DeclareLaunchArgument(
        "sim_type", default_value="primitive",
        description="primitive=fast cylinders, urdf=STL mesh geometry",
    )
    policy = DeclareLaunchArgument(
        "policy", default_value="mock",
        description="VLA policy: mock, pi0, pi0_fast, act, diffusion",
    )
    pretrained = DeclareLaunchArgument(
        "pretrained", default_value="",
        description="Path to pretrained LeRobot policy checkpoint",
    )
    device = DeclareLaunchArgument(
        "device", default_value="cpu",
        description="Device for VLA inference: cpu, cuda, mps",
    )

    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        parameters=[{
            "sim_type": LaunchConfiguration("sim_type"),
        }],
        output="screen",
    )

    vla_node = Node(
        package="lekiwi_ros2_bridge",
        executable="vla_policy_node",
        name="lekiwi_vla_policy_node",
        parameters=[{
            "policy":     LaunchConfiguration("policy"),
            "pretrained": LaunchConfiguration("pretrained"),
            "device":     LaunchConfiguration("device"),
        }],
        remappings=[
            # VLA node reads from bridge's joint_states output
            ("/lekiwi/joint_states",    "/lekiwi/joint_states"),
            ("/lekiwi/camera/image_raw","/lekiwi/camera/image_raw"),
        ],
        output="screen",
    )

    return LaunchDescription([
        sim_type,
        policy,
        pretrained,
        device,
        bridge_node,
        vla_node,
    ])
