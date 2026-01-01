from setuptools import setup

package_name = "ai_slam_ai"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "dataset_recorder=ai_slam_ai.dataset_recorder:main",
            "train_model=ai_slam_ai.train_model:main",
            "infer_node=ai_slam_ai.infer_node:main",
        ],
    },
)
