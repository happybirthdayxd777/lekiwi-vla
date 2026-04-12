from setuptools import setup

package_name = "lekiwi_ros2_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LeKiWi Research",
    maintainer_email="research@lekiwi.local",
    description="ROS2 ↔ MuJoCo bridge node for LeKiWi VLA research platform",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge_node = lekiwi_ros2_bridge.bridge_node:main",
            "vla_policy_node = lekiwi_ros2_bridge.vla_policy_node:main",
            "replay_node = lekiwi_ros2_bridge.replay_node:main",
        ],
    },
)
