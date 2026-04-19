from setuptools import setup

package_name = "lekiwi_ros2_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name + "/launch", ["launch/bridge.launch.py"]),
    ],
    install_scripts=["scripts/bridge_node"],
    python_requires=">=3.8",
)
