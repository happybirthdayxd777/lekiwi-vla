"""
Unified LeKiWi Launch — Full Platform
======================================
One launch file to rule them all:

  ros2 launch lekiwi_ros2_bridge full.launch.py \
    sim_type:=primitive \
    policy:=mock

Modes:
  sim_type : primitive | urdf
  policy   : mock | pi0 | pi0_fast | act | diffusion | clip_fm

Starts:
  1. lekiwi_ros2_bridge (bridge node) — ROS2 ↔ MuJoCo
  2. lekiwi_vla_policy_node            — VLA policy inference

Topics published by bridge:
  /lekiwi/joint_states           — arm (6) + wheel (3) positions & velocities
  /lekiwi/camera/image_raw       — front camera (20 Hz)
  /lekiwi/wrist_camera/image_raw — wrist camera (20 Hz, URDF mode only)
  /lekiwi/wheel_N/cmd_vel       — wheel velocity (Float64)
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
    mode = DeclareLaunchArgument(
        "mode", default_value="sim",
        description="sim=MuJoCo, real=real hardware",
    )
    policy = DeclareLaunchArgument(
        "policy", default_value="mock",
        description="VLA policy: mock, pi0, pi0_fast, act, diffusion, clip_fm",
    )
    pretrained = DeclareLaunchArgument(
        "pretrained", default_value="",
        description="Path to pretrained policy checkpoint (LeRobot or clip_fm .pt)",
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
            "mode": LaunchConfiguration("mode"),
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
            ("/lekiwi/camera/image_raw", "/lekiwi/camera/image_raw"),
        ],
        output="screen",
    )

    return LaunchDescription([
        sim_type,
        mode,
        policy,
        pretrained,
        device,
        bridge_node,
        vla_node,
    ])
