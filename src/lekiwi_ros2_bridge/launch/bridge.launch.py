"""
Unified LeKiWi Launch — Real vs Sim mode switch
===============================================

Usage:
  ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=sim
  ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=real

Modes:
  sim  — start MuJoCo bridge (LeKiWiSim) + joint_state + camera publishers
         Use this when there is no real hardware.

  real — start lekiwi_modular controllers + our bridge as passthrough
         Use this when connecting to real LeKiWi hardware.
"""

import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="sim",
        description="Execution mode: 'sim' (MuJoCo) or 'real' (hardware)",
    )
    mode = LaunchConfiguration("mode")

    # ── Sim mode: bridge node only ──────────────────────────────────────────────
    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        output="screen",
        parameters=[],
    )

    # ── Real mode: lekiwi_modular stack + bridge as monitor ─────────────────────
    omni_controller = Node(
        package="lekiwi_controller",
        executable="omni_controller",
        name="omni_controller",
        output="screen",
    )

    omni_odometry = Node(
        package="lekiwi_controller",
        executable="omni_odometry",
        name="omni_odometry",
        output="screen",
    )

    # Bridge in real mode: reads /lekiwi/cmd_vel from teleop,
    # forwards arm state to VLA policy via /lekiwi/joint_states
    # Also publishes /lekiwi/camera/image_raw from MuJoCo sim
    bridge_real = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        output="screen",
        parameters=[{"mode": "real"}],
    )

    # ── LaunchDescription ──────────────────────────────────────────────────────
    ld = LaunchDescription([mode_arg])

    # Sim mode actions
    ld.add_action(bridge_node)

    # Real mode actions (registered but择启动 depends on future expand)
    # Currently include modular controllers as reference; actual conditional
    # launch will be added once lekiwi_modular is colcon-built in this workspace.
    # ld.add_action(omni_controller)
    # ld.add_action(omni_odometry)
    # ld.add_action(bridge_real)

    return ld
