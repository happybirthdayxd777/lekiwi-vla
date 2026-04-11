from setuptools import setup

setup(
    name="lekiwi_ros2_bridge",
    version="0.1.0",
    packages=["lekiwi_ros2_bridge"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/lekiwi_ros2_bridge"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aaron Luo",
    description="ROS2 ↔ MuJoCo Bridge for LeKiWi",
    license="MIT",
)
