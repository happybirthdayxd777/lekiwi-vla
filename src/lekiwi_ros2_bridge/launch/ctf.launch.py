"""
CTF Mode Launch — LeKiWi Bridge + Security Monitoring + CTF Integration
=======================================================================
Launches the complete CTF research environment in one command.

Usage:
  ros2 launch lekiwi_ros2_bridge ctf.launch.py

  # With CTF platform URL:
  ros2 launch lekiwi_ros2_bridge ctf.launch.py ctf_url:=http://192.168.1.50:5000

  # URDF mode with full physics:
  ros2 launch lekiwi_ros2_bridge ctf.launch.py sim_type:=urdf ctf_mode:=true

Nodes launched:
  1. lekiwi_ros2_bridge (bridge_node) — CTF mode enabled
     - /lekiwi/cmd_vel          ← teleop input
     - /lekiwi/joint_states      → robot state
     - /lekiwi/camera/image_raw  → camera (URDF mode)
     - /lekiwi/security_alert    → CTF alerts
  2. lekiwi_vla_policy_node (vla_policy_node) — mock policy for testing

CTF Security Layers:
  Layer 1: CTFSecurityAuditor — monitors all 8 CTF channels
  Layer 2: SecurityMonitor — legacy HMAC + replay protection
  Layer 3: PolicyGuardian — blocks unauthorized policy switches

CTF Platform Integration:
  python3 ctf_integration.py --mode hub --ctf-url http://localhost:5000
  # (run separately, subscribes to /lekiwi/security_alert)

CTF Challenges:
  C1: cmd_vel HMAC forged     (topic: /lekiwi/cmd_vel)
  C2: DoS rate flood          (topic: /lekiwi/cmd_vel)
  C3: Command injection       (topic: /lekiwi/cmd_vel)
  C4: Physics DoS (accel)     (topic: /lekiwi/cmd_vel)
  C5: Replay attack          (topic: /lekiwi/cmd_vel)
  C6: Sensor spoof            (topic: /lekiwi/joint_states)
  C7: Policy hijack           (topic: /lekiwi/policy)
  C8: VLA action inject       (topic: /lekiwi/vla_action)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ────────────────────────────────────────────────────────────
    sim_type_arg = DeclareLaunchArgument(
        "sim_type", default_value="primitive",
        description="urdf=STL mesh physics, primitive=fast cylinders"
    )
    rate_arg = DeclareLaunchArgument(
        "rate", default_value="50.0",
        description="Bridge loop rate (Hz)"
    )
    ctf_mode_arg = DeclareLaunchArgument(
        "ctf_mode", default_value="true",
        description="Enable CTF security monitoring (true/false)"
    )
    ctf_url_arg = DeclareLaunchArgument(
        "ctf_url", default_value="http://localhost:5000",
        description="CTF platform URL for flag submission"
    )
    render_arg = DeclareLaunchArgument(
        "render", default_value="false",
        description="Enable MuJoCo rendering (true/false)"
    )
    policy_arg = DeclareLaunchArgument(
        "policy", default_value="mock",
        description="VLA policy: mock, clip_fm, pi0"
    )

    # ── Bridge Node ──────────────────────────────────────────────────────────
    bridge_node = Node(
        package="lekiwi_ros2_bridge",
        executable="bridge_node",
        name="lekiwi_ros2_bridge",
        output="screen",
        parameters=[{
            "sim_type":     LaunchConfiguration("sim_type"),
            "rate":         LaunchConfiguration("rate"),
            "ctf_mode":     LaunchConfiguration("ctf_mode"),
            "render":       LaunchConfiguration("render"),
            # CTF: no HMAC by default (participants need to implement it)
            "enable_hmac":  False,
            "cmd_vel_secret": "",
        }],
        remappings=[
            ("/lekiwi/cmd_vel",        "/lekiwi/cmd_vel"),
            ("/lekiwi/joint_states",   "/lekiwi/joint_states"),
            ("/lekiwi/vla_action",     "/lekiwi/vla_action"),
            ("/lekiwi/security_alert", "/lekiwi/security_alert"),
        ],
    )

    # ── VLA Policy Node ──────────────────────────────────────────────────────
    vla_node = Node(
        package="lekiwi_ros2_bridge",
        executable="vla_policy_node",
        name="lekiwi_vla_policy_node",
        output="screen",
        parameters=[{
            "policy": "mock",   # CTF mode: always mock (no real GPU needed)
            "device": "cpu",
            "wheel_alpha":  0.25,
            "arm_alpha":    0.70,
        }],
        remappings=[
            ("/lekiwi/joint_states",  "/lekiwi/joint_states"),
            ("/lekiwi/camera/image_raw", "/lekiwi/camera/image_raw"),
            ("/lekiwi/vla_action",    "/lekiwi/vla_action"),
        ],
    )

    return LaunchDescription([
        sim_type_arg,
        rate_arg,
        ctf_mode_arg,
        ctf_url_arg,
        render_arg,
        policy_arg,
        bridge_node,
        vla_node,
    ])
