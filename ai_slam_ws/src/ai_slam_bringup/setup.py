from setuptools import setup
from glob import glob

package_name = "ai_slam_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "odom_corruptor=ai_slam_bringup.odom_corruptor:main",
            "scan_fix=ai_slam_bringup.scan_fix:main",
            "gt_pose_publisher=ai_slam_bringup.gt_pose_publisher:main",
            "auto_driver=ai_slam_bringup.auto_driver:main",
            "lifecycle_manager=ai_slam_bringup.lifecycle_manager:main",
        ],
    },
)
