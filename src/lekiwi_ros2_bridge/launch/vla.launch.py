"""
LeKiWi VLA Policy Node Launch
==============================
Standalone launch for the VLA policy node (no bridge).

Usage:
  ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=mock device:=cpu
  ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=pi0 pretrained:=/path/to/model device:=cuda
  ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=clip_fm device:=cpu
  ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=clip_fm pretrained:=~/hermes_research/lekiwi_vla/results/fm_50ep_improved/policy_ep10.pt device:=cpu

Policies:
  mock    — sinusoidal testing (no GPU)
  pi0     — LeRobot pi0 policy
  pi0_fast— LeRobot pi0_fast policy
  act     — LeRobot ACT policy
  diffusion — LeRobot diffusion policy
  clip_fm — CLIP ViT-B/32 + Flow Matching (scripts/train_clip_fm.py), default
             checkpoint: ~/hermes_research/lekiwi_vla/results/fm_50ep_improved/policy_ep10.pt

Topics:
  Subscribes:
    /lekiwi/joint_states     — arm (6) + wheel (3) positions & velocities
    /lekiwi/camera/image_raw — camera image (20 Hz)
  Publishes:
    /lekiwi/vla_action      — 9-DOF action [arm*6, wheel*3] in native units
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    policy_arg = DeclareLaunchArgument(
        "policy", default_value="mock",
        description="VLA policy: mock, pi0, pi0_fast, act, diffusion, clip_fm",
    )
    pretrained_arg = DeclareLaunchArgument(
        "pretrained", default_value="",
        description="Path to pretrained LeRobot policy checkpoint",
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="cpu",
        description="Device for VLA inference: cpu, cuda, mps",
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
            ("/lekiwi/joint_states",    "/lekiwi/joint_states"),
            ("/lekiwi/camera/image_raw","/lekiwi/camera/image_raw"),
        ],
        output="screen",
    )

    return LaunchDescription([
        policy_arg,
        pretrained_arg,
        device_arg,
        vla_node,
    ])