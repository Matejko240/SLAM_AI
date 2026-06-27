from __future__ import annotations

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_ref(ref: str, current_dir: Path, cfg_root: Path) -> Path:
    if os.path.isabs(ref):
        return Path(ref).resolve()
    current = (current_dir / ref).resolve()
    if current.exists():
        return current
    return (cfg_root / ref).resolve()


def _load_cfg(path: Path, cfg_root: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise RuntimeError(f"Config extends cycle detected: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    extends = data.pop("extends", None)
    if not extends:
        return data
    parents = [extends] if isinstance(extends, str) else list(extends)
    merged_parent = {}
    for parent_ref in parents:
        parent_path = _resolve_ref(str(parent_ref), path.parent, cfg_root)
        merged_parent = _deep_merge(merged_parent, _load_cfg(parent_path, cfg_root, (*stack, path)))
    return _deep_merge(merged_parent, data)


def _setup(context, *_args, **_kwargs):
    bringup_share = Path(get_package_share_directory("ai_slam_bringup"))
    cfg_root = bringup_share / "config"
    cfg_arg = LaunchConfiguration("config").perform(context)
    cfg_path = Path(cfg_arg)
    if not cfg_path.is_absolute():
        cfg_path = (cfg_root / cfg_arg).resolve()
    cfg = _load_cfg(cfg_path, cfg_root)

    infer = cfg.get("inference", {}) or {}
    robak = cfg.get("robak", {}) or {}
    rywak = cfg.get("rywak", {}) or {}
    jetracer_scan = cfg.get("jetracer_scan", {}) or {}

    scan_in = str(LaunchConfiguration("scan_topic").perform(context))
    if scan_in == "__USE_CONFIG__":
        scan_in = str(infer.get("scan_topic", "/scan"))

    scan_out = str(LaunchConfiguration("scan_out_topic").perform(context))
    if scan_out == "__USE_CONFIG__":
        scan_out = "/scan_jetracer_ai"

    odom_topic = str(LaunchConfiguration("odom_topic").perform(context))
    if odom_topic == "__USE_CONFIG__":
        odom_topic = str(infer.get("odom_topic", "/odom"))

    out_dir = str(LaunchConfiguration("out_dir").perform(context))
    if out_dir == "__USE_CONFIG__":
        out_dir = "out"

    scan_reverse = bool(jetracer_scan.get("scan_reverse", False))
    scan_shift_deg = float(jetracer_scan.get("scan_shift_deg", 0.0))

    nodes = [
        Node(
            package="ai_slam_bringup",
            executable="jetracer_scan_adapter",
            output="screen",
            parameters=[
                {
                    "in_topic": scan_in,
                    "out_topic": scan_out,
                    "target_beams": 360,
                    "clip_min": 0.08,
                    "clip_max": 10.0,
                    "frame_id_override": "base_scan",
                    "scan_reverse": scan_reverse,
                    "scan_shift_deg": scan_shift_deg,
                }
            ],
        ),
        Node(
            package="ai_slam_ai",
            executable="infer_robak_node",
            output="screen",
            parameters=[
                {
                    "out_dir": out_dir,
                    "scan_topic": scan_out,
                    "odom_topic": odom_topic,
                    "pose_topic": str(robak.get("pose_topic", "/pose_robak_jetracer")),
                    "tf_parent": "odom_ai_jetracer",
                    "tf_child": "base_link_ai_jetracer",
                    "model_name": str(robak.get("model_name", "model_robak.pt")),
                    "infer_odom_topic": str(robak.get("infer_odom_topic", odom_topic)),
                    "init_from": "odom",
                    "use_odom_corrections": False,
                    "force_odom_pose": False,
                    "odom_guard_enabled": False,
                    "write_experiment_metadata": True,
                }
            ],
        ),
        Node(
            package="ai_slam_ai",
            executable="infer_rywak_node",
            output="screen",
            parameters=[
                {
                    "out_dir": out_dir,
                    "scan_topic": scan_out,
                    "odom_topic": odom_topic,
                    "init_from_odom_topic": str(rywak.get("infer_init_from_odom_topic", odom_topic)),
                    "pose_topic": str(rywak.get("pose_topic", "/pose_rywak_jetracer")),
                    "tf_parent": "odom_ai_jetracer",
                    "tf_child": "base_link_ai_jetracer",
                    "model_name": str(rywak.get("model_name", "model_rywak.pt")),
                    "use_odom_corrections": False,
                    "force_odom_pose": False,
                    "odom_guard_enabled": False,
                    "fuse_odom_v_weight": 0.0,
                    "fuse_odom_w_weight": 0.0,
                    "fuse_odom_v_gain": 0.0,
                    "fuse_odom_w_gain": 0.0,
                    "anchor_yaw_to_odom": 0.0,
                    "anchor_xy_to_odom": 0.0,
                    "write_experiment_metadata": True,
                }
            ],
        ),
    ]
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="jetracer_online_strict.yaml"),
            DeclareLaunchArgument("scan_topic", default_value="__USE_CONFIG__"),
            DeclareLaunchArgument("scan_out_topic", default_value="__USE_CONFIG__"),
            DeclareLaunchArgument("odom_topic", default_value="__USE_CONFIG__"),
            DeclareLaunchArgument("out_dir", default_value="__USE_CONFIG__"),
            OpaqueFunction(function=_setup),
        ]
    )
