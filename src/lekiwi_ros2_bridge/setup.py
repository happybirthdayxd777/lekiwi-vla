from setuptools import setup

setup(
    name='lekiwi_ros2_bridge',
    version='0.1.0',
    # Package is lekiwi_ros2_bridge/lekiwi_ros2_bridge/
    packages=['lekiwi_ros2_bridge.lekiwi_ros2_bridge'],
    package_dir={
        'lekiwi_ros2_bridge.lekiwi_ros2_bridge': 'lekiwi_ros2_bridge',
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/lekiwi_ros2_bridge']),
        ('share/lekiwi_ros2_bridge', ['package.xml']),
        ('share/lekiwi_ros2_bridge/launch', [
            'launch/bridge.launch.py',
            'launch/full.launch.py',
            'launch/vla.launch.py',
            'launch/real_mode.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LeKiWi Researcher',
    description='ROS2 ↔ LeKiWi-MuJoCo Bridge Node',
    license='MIT',
    tests_require=['ament_copyright', 'ament_flake8', 'ament_pep257'],
    entry_points={
        'console_scripts': [
            # Entry points reference the package: lekiwi_ros2_bridge.bridge_node:main
            # which maps to lekiwi_ros2_bridge/lekiwi_ros2_bridge/bridge_node.py
            'bridge_node = lekiwi_ros2_bridge.lekiwi_ros2_bridge.bridge_node:main',
            'vla_policy_node = lekiwi_ros2_bridge.lekiwi_ros2_bridge.vla_policy_node:main',
            'replay_node = lekiwi_ros2_bridge.lekiwi_ros2_bridge.replay_node:main',
        ],
    },
)
