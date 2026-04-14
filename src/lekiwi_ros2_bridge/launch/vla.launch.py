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
        "policy", default_value="clip_fm",
        description="VLA policy: mock, pi0, pi0_fast, act, diffusion, clip_fm",
    )
    pretrained_arg = DeclareLaunchArgument(
        "pretrained", default_value="~/hermes_research/lekiwi_vla/results/fresh_train_5k/final_policy.pt",
        description="Path to pretrained policy checkpoint (CLIP-FM .pt or LeRobot)",
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="cpu",
        description="Device for VLA inference: cpu, cuda, mps",
    )
    goal_x_arg = DeclareLaunchArgument(
        "goal_x", default_value="0.3",
        description="Goal X position for task_oriented policy (meters)",
    )
    goal_y_arg = DeclareLaunchArgument(
        "goal_y", default_value="0.2",
        description="Goal Y position for task_oriented policy (meters)",
    )
    wheel_alpha_arg = DeclareLaunchArgument(
        "wheel_alpha", default_value="0.25",
        description="ActionSmoother wheel EMA coefficient (0=smooth, 1=no smooth)",
    )
    arm_alpha_arg = DeclareLaunchArgument(
        "arm_alpha", default_value="0.70",
        description="ActionSmoother arm EMA coefficient",
    )
    wheel_max_delta_arg = DeclareLaunchArgument(
        "wheel_max_delta", default_value="0.8",
        description="Max wheel action change per step (rad/s)",
    )
    arm_max_delta_arg = DeclareLaunchArgument(
        "arm_max_delta", default_value="0.5",
        description="Max arm action change per step (rad)",
    )

    vla_node = Node(
        package="lekiwi_ros2_bridge",
        executable="vla_policy_node",
        name="lekiwi_vla_policy_node",
        parameters=[{
            "policy":     LaunchConfiguration("policy"),
            "pretrained": LaunchConfiguration("pretrained"),
            "device":     LaunchConfiguration("device"),
            "goal_x":     LaunchConfiguration("goal_x"),
            "goal_y":     LaunchConfiguration("goal_y"),
            "wheel_alpha":    LaunchConfiguration("wheel_alpha"),
            "arm_alpha":      LaunchConfiguration("arm_alpha"),
            "wheel_max_delta": LaunchConfiguration("wheel_max_delta"),
            "arm_max_delta":   LaunchConfiguration("arm_max_delta"),
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
        goal_x_arg,
        goal_y_arg,
        wheel_alpha_arg,
        arm_alpha_arg,
        wheel_max_delta_arg,
        arm_max_delta_arg,
        vla_node,
    ])