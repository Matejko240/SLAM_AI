from setuptools import setup
from glob import glob
from pathlib import Path


def collect_data_files(source_dir: str, target_prefix: str):
    base = Path(source_dir)
    entries = []
    if not base.is_dir():
        return entries
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel_parent = path.parent.relative_to(base)
        target = f"{target_prefix}/{rel_parent.as_posix()}" if rel_parent.as_posix() != "." else target_prefix
        entries.append((target, [str(path)]))
    return entries

package_name = "ai_slam_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        *collect_data_files("models", f"share/{package_name}/models"),
        *collect_data_files("media", f"share/{package_name}/media"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)
