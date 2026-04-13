"""
bridge.launch.py — Unified launch for LeKiWi ROS2-MuJoCo bridge

Usage:
  ros2 launch lekiwi_ros2_bridge bridge.launch.py

Topics bridged:
  /lekiwi/cmd_vel         (Twist)      ← input: teleop/Nav2 base commands
  /lekiwi/arm_joint_i/cmd_pos (Float64) ← input: arm joint position (radians)
  /lekiwi/wheel_i/cmd_vel (Float64)    ← output: wheel angular velocities
  /lekiwi/odom            (Odometry)   ← output: simulated odometry
  /lekiwi/joint_states    (JointState) ← output: full joint state
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Arguments
    rate = DeclareLaunchArgument(
        'rate', default_value='50.0',
        description='Bridge loop rate (Hz)'
    )
    sim_type = DeclareLaunchArgument(
        'sim_type', default_value='primitive',  # Phase 26: urdf contact physics broken, use primitive
        description='urdf=STL mesh (recommended), primitive=fast cylinders'
    )
    mode = DeclareLaunchArgument(
        'mode', default_value='sim',
        description='sim=MuJoCo, real=real hardware'
    )
    render = DeclareLaunchArgument(
        'render', default_value='false',
        description='Enable MuJoCo rendering (true/false)'
    )

    bridge_node = Node(
        package='lekiwi_ros2_bridge',
        executable='bridge_node',
        name='lekiwi_ros2_bridge',
        output='screen',
        parameters=[{
            'rate': LaunchConfiguration('rate'),
            'sim_type': LaunchConfiguration('sim_type'),
            'mode': LaunchConfiguration('mode'),
            'render': LaunchConfiguration('render'),
        }],
        remappings=[
            # Base velocity command (from teleop)
            ('/lekiwi/cmd_vel', '/lekiwi/cmd_vel'),
            # Arm joint commands (Float64 per joint, radians)
            ('/lekiwi/arm_joint_0/cmd_pos', '/lekiwi/arm_joint_0/cmd_pos'),
            ('/lekiwi/arm_joint_1/cmd_pos', '/lekiwi/arm_joint_1/cmd_pos'),
            ('/lekiwi/arm_joint_2/cmd_pos', '/lekiwi/arm_joint_2/cmd_pos'),
            ('/lekiwi/arm_joint_3/cmd_pos', '/lekiwi/arm_joint_3/cmd_pos'),
            ('/lekiwi/arm_joint_4/cmd_pos', '/lekiwi/arm_joint_4/cmd_pos'),
            ('/lekiwi/arm_joint_5/cmd_pos', '/lekiwi/arm_joint_5/cmd_pos'),
            # Wheel velocity outputs
            ('/lekiwi/wheel_0/cmd_vel', '/lekiwi/wheel_0/cmd_vel'),
            ('/lekiwi/wheel_1/cmd_vel', '/lekiwi/wheel_1/cmd_vel'),
            ('/lekiwi/wheel_2/cmd_vel', '/lekiwi/wheel_2/cmd_vel'),
            # Odometry output
            ('/lekiwi/odom', '/lekiwi/odom'),
            # Joint state output
            ('/lekiwi/joint_states', '/lekiwi/joint_states'),
        ],
    )

    return LaunchDescription([
        rate,
        sim_type,
        mode,
        render,
        bridge_node,
    ])
