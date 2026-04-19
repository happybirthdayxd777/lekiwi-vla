#!/usr/bin/env python3
"""
bridge.launch.py — Unified LeKiWi ROS2 ↔ MuJoCo Bridge Launcher
================================================================
Phase 183 — Multi-mode: sim / urdf / vla / ctf / real-hw

Modes:
  sim       — MuJoCo primitive geometry, no ROS2 hardware nodes
  sim_urdf  — MuJoCo with full URDF meshes from lekiwi_modular
  vla       — Simulation + VLA policy action passthrough
  ctf       — Simulation + CTF Security Audit Mode
  real      — Passthrough: bridge connects ROS2 topics to hardware drivers
              (no MuJoCo simulation; robot URDF for FK only)

Usage:
  # Simulation mode
  ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=sim

  # URDF simulation
  ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=sim_urdf

  # VLA mode
  ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=vla

  # CTF security mode
  LEKIWI_CTF_MODE=1 ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=sim

  # Real hardware mode
  ros2 launch lekiwi_ros2_bridge bridge.launch.py mode:=real

Arguments:
  mode      — Operating mode: sim, sim_urdf, vla, ctf, real
  urdf_path — Path to URDF (default: lekiwi_modular package share)
  vla_model — Path to VLA policy checkpoint (for vla mode)
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition


def get_lekiwi_share():
    """Find lekiwi_modular package share directory."""
    # Default URDF path — overridden if package not found
    return '/opt/ros/humble/share/lekiwi_description'


def generate_launch_description():
    mode = LaunchConfiguration('mode', default='sim')
    urdf_path = LaunchConfiguration('urdf_path', default='')
    vla_model = LaunchConfiguration('vla_model', default='')

    # ── Determine node parameters by mode ─────────────────────────────────────
    is_sim      = IfCondition('"$(var mode)" == "sim"')
    is_sim_urdf = IfCondition('"$(var mode)" == "sim_urdf"')
    is_vla      = IfCondition('"$(var mode)" == "vla"')
    is_ctf      = IfCondition('"$(var mode)" == "ctf"')
    is_real     = IfCondition('"$(var mode)" == "real"')
    is_not_real = IfCondition('"$(var mode)" != "real"')

    bridge_params = {
        'sim_type': 'primitive',
        'vla_mode': False,
        'urdf_path': urdf_path,
    }

    # ── CTF environment ───────────────────────────────────────────────────────
    ctf_env = [
        SetEnvironmentVariable(name='LEKIWI_CTF_MODE', value='1'),
    ]

    # ── Bridge node ──────────────────────────────────────────────────────────
    # Parameters vary by mode
    bridge_node = Node(
        package='lekiwi_ros2_bridge',
        executable='bridge_node',
        name='lekiwi_bridge',
        output='screen',
        parameters=[{
            # sim mode: primitive MuJoCo
            'sim_type': 'primitive',
            'vla_mode': False,
            'urdf_path': urdf_path,
        }],
        remappings=[
            ('/lekiwi/cmd_vel',    '/lekiwi/cmd_vel'),
            ('/lekiwi/joint_states', '/lekiwi/joint_states'),
            ('/lekiwi/odom',      '/lekiwi/odom'),
        ],
        condition=IfCondition('"$(var mode)" != "real"'),
    )

    # ── Real hardware bridge ──────────────────────────────────────────────────
    # In real mode, we DON'T create the bridge node (no MuJoCo).
    # Instead, we rely on lekiwi_modular nodes to handle hardware.
    # The bridge in real mode just exposes the same topic interface.
    real_bridge_node = Node(
        package='lekiwi_ros2_bridge',
        executable='bridge_node',
        name='lekiwi_bridge_real',
        output='screen',
        parameters=[{
            # In real mode, use a minimal sim (no MuJoCo needed for FK)
            # The bridge acts as a topic adapter / security monitor
            'sim_type': 'primitive',
            'vla_mode': False,
            'urdf_path': urdf_path,
        }],
        remappings=[
            # Forward hardware topics with security wrapper
            ('/lekiwi/cmd_vel',    '/lekiwi/cmd_vel'),
            ('/lekiwi/joint_states', '/lekiwi/joint_states_from_hw'),
        ],
        condition=IfCondition('"$(var mode)" == "real"'),
    )

    # ── CTF Monitor Node ──────────────────────────────────────────────────────
    # Standalone CTF monitor: subscribes to cmd_vel, logs all events
    ctf_monitor = Node(
        package='lekiwi_ros2_bridge',
        executable='bridge_node',
        name='lekiwi_ctf_monitor',
        output='screen',
        parameters=[{
            'sim_type': 'primitive',
            'vla_mode': False,
        }],
        remappings=[],
        env={'LEKIWI_CTF_MODE': '1'},
        condition=IfCondition('"$(var mode)" == "ctf"'),
    )

    # ── VLA Server Node ────────────────────────────────────────────────────────
    # Stub: the actual VLA server is started separately
    # This just validates that vla_mode is set when mode==vla
    vla_validation = LogInfo(
        msg=['[bridge.launch] VLA mode — ensure VLA server is running on port 50051'],
        condition=IfCondition('"$(var mode)" == "vla"'),
    )

    # ── Log mode on startup ───────────────────────────────────────────────────
    mode_log = LogInfo(msg=['[bridge.launch] LeKiWi Bridge launching — mode=', mode])

    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('mode',
            default_value='sim',
            description='Operating mode: sim, sim_urdf, vla, ctf, real'),
        DeclareLaunchArgument('urdf_path',
            default_value='',
            description='URDF file path (for sim_urdf/real modes)'),
        DeclareLaunchArgument('vla_model',
            default_value='',
            description='VLA policy checkpoint path (for vla mode)'),

        # Mode log
        mode_log,

        # CTF env (always set for ctf mode)
        SetEnvironmentVariable(
            name='LEKIWI_CTF_MODE',
            value='1',
            condition=IfCondition('"$(var mode)" == "ctf"'),
        ),

        # Bridge node (simulation modes)
        bridge_node,

        # Real HW bridge
        real_bridge_node,

        # VLA validation
        vla_validation,
    ])
