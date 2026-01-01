from setuptools import setup
from glob import glob
import os

package_name = "ai_slam_description"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
        (f"share/{package_name}/models", glob("models/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)
