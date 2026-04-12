"""
LeKiWi Real Hardware Mode Launch
================================
Launches the bridge in real hardware mode — no MuJoCo, reads from real servos.

Usage:
  ros2 launch lekiwi_ros2_bridge real_mode.launch.py

In this mode:
  - bridge_node connects to serial bus (/dev/ttyUSB0) instead of MuJoCo
  - Reads servo feedback → publishes /lekiwi/joint_states
  - Receives /lekiwi/cmd_vel → converts to servo commands → sends over serial
  - VLA policy node still runs (reads joint_states + camera → outputs vla_action)
  - bridge_node receives /lekiwi/vla_action → forwards to serial servos

Hardware:
  - ST3215 servo controllers on USB-serial
  - USB webcams on /dev/video0 (front) and /dev/video2 (wrist)
  - Raspberry Pi 5 or equivalent SBC

Topics (real mode):
  Input : /lekiwi/cmd_vel        (geometry_msgs/Twist) — mobile base
  Input : /lekiwi/vla_action     (Float64MultiArray)   — arm control
  Output: /lekiwi/joint_states   (sensor_msgs/JointState)
  Output: /lekiwi/camera/image_raw
  Output: /lekiwi/wrist_camera/image_raw
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    device = DeclareLaunchArgument(
        "device", default_value="cpu",
        description="Device for VLA inference: cpu, cuda, mps",
    )
    policy = DeclareLaunchArgument(
        "policy", default_value="task_oriented",
        description="VLA policy: mock, pi0, pi0_fast, act, diffusion, clip_fm, task_oriented",
    )
    pretrained = DeclareLaunchArgument(
        "pretrained", default_value="~/hermes_research/lekiwi_vla/results/task_oriented_goaldirected/checkpoint_epoch_30.pt",
        description="Path to pretrained policy checkpoint",
    )

    # ── Bridge node in real hardware mode ────────────────────────
    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        parameters=[{
            "mode": "real",           # ← tells bridge_node to use serial instead of MuJoCo
            "sim_type": "urdf",       # still use URDF joint names for state publishing
        }],
        output="screen",
    )

    # ── VLA policy node ───────────────────────────────────────────
    vla_node = Node(
        package="lekiwi_ros2_bridge",
        executable="vla_policy_node",
        name="lekiwi_vla_policy_node",
        parameters=[{
            "policy":      LaunchConfiguration("policy"),
            "pretrained":  LaunchConfiguration("pretrained"),
            "device":      LaunchConfiguration("device"),
        }],
        remappings=[
            ("/lekiwi/joint_states",     "/lekiwi/joint_states"),
            ("/lekiwi/camera/image_raw", "/lekiwi/camera/image_raw"),
        ],
        output="screen",
    )

    return LaunchDescription([
        device,
        policy,
        pretrained,
        bridge_node,
        vla_node,
    ])
