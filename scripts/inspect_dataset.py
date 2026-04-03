#!/usr/bin/env python3
"""Generuje czytelny raport inspekcji datasetu do artefaktow eksperymentu."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_DIR = REPO_ROOT / "ai_slam_ws"
REF_MAP_YAML = WORKSPACE_DIR / "src" / "ai_slam_eval" / "maps" / "reference_map.yaml"
WORLD_REFERENCE_MAPS = {
    "world_house.sdf": "reference_map.yaml",
    "world_office.sdf": "reference_map_office.yaml",
    "world_hospital.sdf": "reference_map_hospital.yaml",
}
DEFAULT_WORLD_SPAWN_POSES = {
    "world_house.sdf": (5.0, 0.0, 0.0),
    "world_house": (5.0, 0.0, 0.0),
    "world_office.sdf": (0.03, 2.27, 0.0),
    "world_office": (0.03, 2.27, 0.0),
    "world_hospital.sdf": (0.0, -25.0, 0.0),
    "world_hospital": (0.0, -25.0, 0.0),
}
LEGACY_WORLD_SPAWN_POSES = {
    "world_hospital.sdf": (0.72, 11.6, 0.0),
    "world_hospital": (0.72, 11.6, 0.0),
}

SUMMARY_NAME = "dataset_inspection_summary.json"
OVERVIEW_NAME = "dataset_inspection_overview.png"
SCANS_NAME = "dataset_inspection_scans.png"
LEGACY_NAME = "dataset_analysis.png"
TARGET_COMPONENTS_NAME = "dataset_target_components.png"
ROBAK_SUMMARY_NAME = "dataset_robak_coverage_summary.json"
ROBAK_DISTANCE_NAME = "dataset_robak_coverage_distance.png"
ROBAK_ROTATION_NAME = "dataset_robak_coverage_rotation.png"
ROBAK_COMPONENTS_NAME = "dataset_robak_target_components.png"
RYWAK_SUMMARY_NAME = "dataset_rywak_coverage_summary.json"
RYWAK_LINEAR_NAME = "dataset_rywak_coverage_linear_velocity.png"
RYWAK_ANGULAR_NAME = "dataset_rywak_coverage_angular_velocity.png"
RYWAK_SIGNED_NAME = "dataset_rywak_target_signed_velocity.png"
TRAJECTORY_SPEED_NAME = "eval_trajectory_speed.png"
TRAINING_SUMMARY_NAME = "train_inspection_summary.json"
EXPERIMENT_SUMMARY_NAME = "experiment_inspection_summary.json"


def resolve_existing_path(dataset_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = dataset_dir / name
        if path.exists():
            return path
    return dataset_dir / candidates[0]


def resolve_eval_trajectory_data_path(dataset_dir: Path) -> Path:
    return resolve_existing_path(dataset_dir, ["eval_trajectory_data.npz", "trajectory_data.npz"])


def resolve_eval_map_layers_path(dataset_dir: Path) -> Path:
    return resolve_existing_path(dataset_dir, ["eval_map_layers.npz", "map_layers.npz"])


def _trajectory_alignment_metrics(
    poses_xy: np.ndarray,
    ref_grid: np.ndarray,
    ref_resolution: float,
    ref_origin: list[float],
) -> tuple[float, float]:
    xy = np.asarray(poses_xy, dtype=np.float32)
    if xy.size == 0:
        return 1.0, 1.0
    if xy.ndim == 2 and xy.shape[1] >= 2:
        x = xy[:, 0]
        y = xy[:, 1]
    else:
        flat = xy.reshape((-1,))
        if flat.size < 2:
            return 1.0, 1.0
        x = flat[0::2]
        y = flat[1::2]
    h, w = ref_grid.shape
    x_min = float(ref_origin[0])
    y_min = float(ref_origin[1])
    jj = np.floor((x - x_min) / float(ref_resolution)).astype(np.int32)
    ii = np.floor((y - y_min) / float(ref_resolution)).astype(np.int32)
    inside = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    outside_ratio = float(1.0 - np.mean(inside.astype(np.float32)))
    if not np.any(inside):
        return outside_ratio, 1.0
    occ = np.zeros_like(inside, dtype=np.bool_)
    occ_inside = np.asarray(ref_grid[ii[inside], jj[inside]], dtype=np.uint8) == 0
    occ[inside] = occ_inside
    occupied_hit_ratio = float(np.mean(occ_inside.astype(np.float32)))
    return outside_ratio, occupied_hit_ratio


def choose_reference_display_alignment(
    dataset_dir: Path,
    world_name: str | None,
    gt_xy: np.ndarray,
) -> tuple[np.ndarray, float, list[float]]:
    best_score: float | None = None
    best: tuple[np.ndarray, float, list[float]] | None = None
    for keep_world_origin in (True, False):
        _yaml, grid_base, resolution, origin = load_reference_map_local(
            dataset_dir,
            world_name=world_name,
            keep_world_origin=keep_world_origin,
        )
        for rotate_180 in (False, True):
            grid = np.rot90(grid_base, 2) if rotate_180 else grid_base
            outside_ratio, occ_hit_ratio = _trajectory_alignment_metrics(gt_xy, grid, resolution, origin)
            # Prioritize staying in-map, then minimize path through occupied cells.
            score = outside_ratio * 5.0 + occ_hit_ratio
            if best_score is None or score < best_score:
                best_score = score
                best = (grid.copy(), float(resolution), [float(origin[0]), float(origin[1]), float(origin[2])])
    if best is not None:
        return best
    _yaml, grid, resolution, origin = load_reference_map_local(dataset_dir, world_name=world_name, keep_world_origin=True)
    return grid, float(resolution), [float(origin[0]), float(origin[1]), float(origin[2])]


def load_reference_map(yaml_path: Path) -> tuple[np.ndarray, float, list[float]]:
    """Wczytuje mape referencyjna z YAML oraz wskazanego w nim pliku PGM."""
    meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    resolution = float(meta["resolution"])
    origin = list(meta["origin"])
    image_path = Path(str(meta.get("image", "reference_map.pgm")))
    if not image_path.is_absolute():
        image_path = (yaml_path.parent / image_path).resolve()

    lines = image_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines) and (not lines[idx].strip() or lines[idx].lstrip().startswith("#")):
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "P2":
        raise ValueError(f"Nieobslugiwany format mapy PGM: {image_path}")
    idx += 1

    while idx < len(lines) and lines[idx].lstrip().startswith("#"):
        idx += 1
    width, height = map(int, lines[idx].split())
    idx += 1
    while idx < len(lines) and lines[idx].lstrip().startswith("#"):
        idx += 1
    _max_val = int(lines[idx].strip())
    idx += 1

    pixels: list[int] = []
    for line in lines[idx:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pixels.extend(int(part) for part in stripped.split())

    grid = np.asarray(pixels, dtype=np.uint8).reshape((height, width))
    return grid, resolution, origin


def make_reference_display_grid(ref_grid: np.ndarray) -> np.ndarray:
    """Render reference map as free=bright, occupied=dark."""
    return np.where(ref_grid == 254, 1.0, np.where(ref_grid == 0, 0.08, 0.32))


def make_probabilistic_map_readable(
    prob_map: np.ndarray,
    neutral_band: float = 0.06,
    occupied_only: bool = True,
) -> np.ndarray:
    """Reduce noisy map rays; optionally show only confident occupied structure."""
    prob = np.asarray(prob_map, dtype=np.float32)
    show = np.full(prob.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(prob)
    if not np.any(finite):
        return show
    eps = float(max(1e-6, neutral_band))
    occ = finite & (prob >= (0.5 + eps))
    free = finite & (prob <= (0.5 - eps))
    if occupied_only:
        show[occ] = 1.0
    else:
        show[occ] = 0.08
        show[free] = 0.90
    return show


def render_map_match_overlay(reference_occ: np.ndarray, candidate_map: np.ndarray) -> np.ndarray:
    """Render occupancy agreement map (TP/FP/FN) for better readability."""
    ref_occ = np.asarray(reference_occ, dtype=np.float32) > 0.5
    cand_occ = np.asarray(candidate_map, dtype=np.float32) > 0.5
    if ref_occ.shape != cand_occ.shape or ref_occ.size == 0:
        return np.zeros((0, 0, 3), dtype=np.float32)

    h, w = ref_occ.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Dark neutral background
    rgb[:, :, :] = np.asarray([0.02, 0.03, 0.06], dtype=np.float32)
    # True negatives: faint reference-free area
    tn = ~ref_occ & ~cand_occ
    rgb[tn] = np.asarray([0.06, 0.08, 0.12], dtype=np.float32)
    # False negatives: reference occupied but missing in candidate (blue)
    fn = ref_occ & ~cand_occ
    rgb[fn] = np.asarray([0.23, 0.57, 0.97], dtype=np.float32)
    # False positives: candidate occupied but absent in reference (red)
    fp = ~ref_occ & cand_occ
    rgb[fp] = np.asarray([0.97, 0.32, 0.26], dtype=np.float32)
    # True positives: overlap with reference (green)
    tp = ref_occ & cand_occ
    rgb[tp] = np.asarray([0.40, 0.95, 0.47], dtype=np.float32)
    return rgb


def orient_map_layer_to_reference(layer: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pick orientation (0 or 180 deg) that best overlaps reference occupied cells."""
    src = np.asarray(layer, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if src.shape != ref.shape or src.size == 0:
        return src
    ref_occ = ref < 0.25
    candidates = [src, np.rot90(src, 2)]
    best = src
    best_score = -1.0
    for cand in candidates:
        cand_occ = cand > 0.55
        union = float(np.sum(ref_occ | cand_occ))
        if union <= 0.0:
            score = 0.0
        else:
            score = float(np.sum(ref_occ & cand_occ)) / union
        if score > best_score:
            best_score = score
            best = cand
    return best


def outside_ratio_xy(poses_xy: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    if poses_xy.size == 0:
        return 0.0
    inside = (
        (poses_xy[:, 0] >= float(x_min))
        & (poses_xy[:, 0] <= float(x_max))
        & (poses_xy[:, 1] >= float(y_min))
        & (poses_xy[:, 1] <= float(y_max))
    )
    return float(100.0 * (1.0 - np.mean(inside.astype(np.float32))))


def configure_axes(ax) -> None:
    ax.set_facecolor("#111827")
    ax.grid(True, alpha=0.22, color="#475569")
    ax.tick_params(colors="#dbe7ff")
    ax.xaxis.label.set_color("#dbe7ff")
    ax.yaxis.label.set_color("#dbe7ff")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#334155")


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [normalize_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    return value


def configure_figure(fig) -> None:
    fig.patch.set_facecolor("#0b1120")


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values)).astype(np.float32)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    normalized = normalize_json_value(payload)
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def stats_1d(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "positive_ratio_pct": 0.0,
            "negative_ratio_pct": 0.0,
            "zero_ratio_pct": 0.0,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": percentile_or_zero(arr, 95.0),
        "positive_ratio_pct": float(np.mean(arr > 0.0) * 100.0),
        "negative_ratio_pct": float(np.mean(arr < 0.0) * 100.0),
        "zero_ratio_pct": float(np.mean(arr == 0.0) * 100.0),
    }


def load_history(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_results_artifacts(dataset_dir: Path, artifact_updates: dict[str, Path]) -> None:
    results_path = dataset_dir / "results.json"
    if not results_path.exists():
        return

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts

    changed = False
    for key, path in artifact_updates.items():
        if path.exists():
            resolved = str(path.resolve())
            if artifacts.get(key) != resolved:
                artifacts[key] = resolved
                changed = True

    if not changed:
        return

    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(results_path)


def resolve_reference_map_yaml(
    dataset_dir: Path,
    world_name: str | None = None,
    *,
    prefer_world_mapping: bool = False,
) -> Path:
    config_snapshot_path = dataset_dir / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        try:
            cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
            sim_cfg = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
            world_candidates: list[str] = []
            for candidate_world in [world_name, sim_cfg.get("test_world"), sim_cfg.get("train_world")]:
                text = str(candidate_world or "").strip()
                if not text:
                    continue
                if text not in world_candidates:
                    world_candidates.append(text)
            candidate = (
                cfg.get("evaluation", {}).get("reference_map_yaml")
                if isinstance(cfg.get("evaluation"), dict)
                else None
            )

            def _resolve_from_worlds() -> Path | None:
                for world in world_candidates:
                    mapped_ref_name = WORLD_REFERENCE_MAPS.get(world)
                    if mapped_ref_name:
                        mapped_ref_path = (WORKSPACE_DIR / "src" / "ai_slam_eval" / "maps" / mapped_ref_name).resolve()
                        if mapped_ref_path.exists():
                            return mapped_ref_path
                return None

            def _resolve_from_eval_candidate() -> Path | None:
                if isinstance(candidate, str) and candidate.strip():
                    candidate_path = Path(candidate.strip())
                    if candidate_path.is_absolute() and candidate_path.exists():
                        return candidate_path.resolve()
                    eval_maps_path = (WORKSPACE_DIR / "src" / "ai_slam_eval" / "maps" / candidate_path.name).resolve()
                    if eval_maps_path.exists():
                        return eval_maps_path
                    cfg_relative = (config_snapshot_path.parent / candidate_path).resolve()
                    if cfg_relative.exists():
                        return cfg_relative
                return None

            if prefer_world_mapping:
                mapped = _resolve_from_worlds()
                if mapped is not None:
                    return mapped
                eval_mapped = _resolve_from_eval_candidate()
                if eval_mapped is not None:
                    return eval_mapped
            else:
                eval_mapped = _resolve_from_eval_candidate()
                if eval_mapped is not None:
                    return eval_mapped
                mapped = _resolve_from_worlds()
                if mapped is not None:
                    return mapped
        except Exception:
            pass

    results_path = dataset_dir / "results.json"
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            artifacts = payload.get("artifacts", {})
            candidate = artifacts.get("reference_map_yaml")
            if isinstance(candidate, str) and candidate:
                candidate_path = Path(candidate).expanduser()
                if candidate_path.exists():
                    return candidate_path.resolve()
        except Exception:
            pass

    return REF_MAP_YAML


def resolve_include_ai_in_artifacts(dataset_dir: Path) -> bool:
    config_snapshot_path = dataset_dir / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        try:
            cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
            evaluation = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
            value = evaluation.get("include_ai_in_artifacts")
            if isinstance(value, bool):
                return value
        except Exception:
            pass
    return False


def resolve_maps_rotate_180(dataset_dir: Path, results: dict[str, Any] | None = None) -> bool:
    map_layers_npz = resolve_eval_map_layers_path(dataset_dir)
    if map_layers_npz.exists():
        try:
            with np.load(map_layers_npz, allow_pickle=True) as data:
                if "rotate_180" in data:
                    return bool(int(np.asarray(data["rotate_180"]).reshape(-1)[0]))
        except Exception:
            pass
    if isinstance(results, dict):
        try:
            cfg = results.get("config_snapshot", {})
            evaluation = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
            if isinstance(evaluation, dict):
                value = evaluation.get("maps_rotate_180")
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(int(value))
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            pass
    return False


def inverse_pose_transform_xy_scalar(x: float, y: float, tx: float, ty: float, yaw: float) -> tuple[float, float]:
    dx = float(x) - float(tx)
    dy = float(y) - float(ty)
    c = float(np.cos(float(yaw)))
    s = float(np.sin(float(yaw)))
    return (
        c * dx + s * dy,
        -s * dx + c * dy,
    )


def resolve_spawn_pose(dataset_dir: Path, world_name: str | None = None) -> tuple[float, float, float]:
    config_snapshot_path = dataset_dir / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        return 0.0, 0.0, 0.0

    try:
        cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return 0.0, 0.0, 0.0

    simulation = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
    spawn_poses = simulation.get("spawn_poses", {}) if isinstance(simulation.get("spawn_poses"), dict) else {}

    candidates: list[str] = []
    for candidate in [world_name, simulation.get("train_world"), simulation.get("test_world")]:
        text = str(candidate or "").strip()
        if not text:
            continue
        for key in [text, text if text.endswith(".sdf") else f"{text}.sdf"]:
            if key not in candidates:
                candidates.append(key)

    for key in candidates:
        pose = spawn_poses.get(key)
        if isinstance(pose, dict):
            return (
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )
    for key in candidates:
        if key in DEFAULT_WORLD_SPAWN_POSES:
            return DEFAULT_WORLD_SPAWN_POSES[key]
    return 0.0, 0.0, 0.0


def load_reference_map_local(
    dataset_dir: Path,
    *,
    world_name: str | None = None,
    prefer_world_mapping: bool = False,
    keep_world_origin: bool = False,
) -> tuple[Path, np.ndarray, float, list[float]]:
    ref_map_yaml = resolve_reference_map_yaml(
        dataset_dir,
        world_name=world_name,
        prefer_world_mapping=prefer_world_mapping,
    )
    ref_grid, ref_resolution, ref_origin_world = load_reference_map(ref_map_yaml)
    spawn_x, spawn_y, spawn_yaw = resolve_spawn_pose(dataset_dir, world_name=world_name)
    if keep_world_origin:
        return ref_map_yaml, ref_grid, ref_resolution, [float(ref_origin_world[0]), float(ref_origin_world[1]), float(ref_origin_world[2])]

    def _local_origin_for_spawn(spawn_pose: tuple[float, float, float]) -> list[float]:
        sx, sy, syaw = spawn_pose
        ox_local, oy_local = inverse_pose_transform_xy_scalar(
            float(ref_origin_world[0]),
            float(ref_origin_world[1]),
            float(sx),
            float(sy),
            float(syaw),
        )
        return [
            ox_local,
            oy_local,
            float(wrap_angle(np.asarray([float(ref_origin_world[2]) - float(syaw)], dtype=np.float32))[0]),
        ]

    chosen_spawn = (float(spawn_x), float(spawn_y), float(spawn_yaw))
    trajectory_path = resolve_eval_trajectory_data_path(dataset_dir)
    if trajectory_path.exists():
        try:
            world_candidates: list[str] = []
            config_snapshot_path = dataset_dir / "config_snapshot.yaml"
            if config_snapshot_path.exists():
                cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
                simulation = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
                for value in [world_name, simulation.get("test_world"), simulation.get("train_world")]:
                    text = str(value or "").strip()
                    if not text:
                        continue
                    for key in [text, text if text.endswith(".sdf") else f"{text}.sdf"]:
                        if key not in world_candidates:
                            world_candidates.append(key)
            elif world_name:
                world_candidates = [world_name, world_name if world_name.endswith(".sdf") else f"{world_name}.sdf"]

            candidate_spawns: list[tuple[float, float, float]] = [chosen_spawn]
            for key in world_candidates:
                for mapping in (DEFAULT_WORLD_SPAWN_POSES, LEGACY_WORLD_SPAWN_POSES):
                    pose = mapping.get(key)
                    if pose is not None and pose not in candidate_spawns:
                        candidate_spawns.append(pose)

            with np.load(trajectory_path, allow_pickle=True) as data:
                gt = np.asarray(data["gt_xytheta"], dtype=np.float32).reshape((-1, 3)) if "gt_xytheta" in data else np.zeros((0, 3), dtype=np.float32)

            if gt.shape[0] > 0 and len(candidate_spawns) > 1:
                h, w = ref_grid.shape
                best_spawn = chosen_spawn
                best_ratio = None
                for pose in candidate_spawns:
                    origin_local = _local_origin_for_spawn(pose)
                    x_min = origin_local[0]
                    y_min = origin_local[1]
                    x_max = x_min + w * ref_resolution
                    y_max = y_min + h * ref_resolution
                    outside = (
                        (gt[:, 0] < x_min)
                        | (gt[:, 0] > x_max)
                        | (gt[:, 1] < y_min)
                        | (gt[:, 1] > y_max)
                    )
                    ratio = float(np.mean(outside.astype(np.float32)))
                    if best_ratio is None or ratio < best_ratio:
                        best_ratio = ratio
                        best_spawn = pose
                chosen_spawn = best_spawn
        except Exception:
            pass

    local_origin = _local_origin_for_spawn(chosen_spawn)
    return ref_map_yaml, ref_grid, ref_resolution, local_origin


def reconstruct_gt_poses(odom: np.ndarray, corrections: np.ndarray) -> np.ndarray:
    gt = np.asarray(odom, dtype=np.float32).copy()
    gt[:, :2] += np.asarray(corrections[:, :2], dtype=np.float32)
    if gt.shape[1] >= 3 and corrections.shape[1] >= 3:
        gt[:, 2] = wrap_angle(gt[:, 2] + np.asarray(corrections[:, 2], dtype=np.float32))
    return gt


def trajectory_step_lengths(poses: np.ndarray) -> np.ndarray:
    deltas = np.diff(poses[:, :2], axis=0) if poses.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
    return np.linalg.norm(deltas, axis=1).astype(np.float32) if deltas.size else np.zeros((0,), dtype=np.float32)


def trajectory_length(poses: np.ndarray) -> float:
    steps = trajectory_step_lengths(poses)
    return float(steps.sum()) if steps.size else 0.0


def trajectory_jump_threshold(odom: np.ndarray) -> float:
    odom_steps = trajectory_step_lengths(odom)
    if odom_steps.size == 0:
        return 0.5
    return max(0.5, float(np.percentile(odom_steps, 99)) * 8.0)


def segmented_trajectory_for_plot(poses: np.ndarray, jump_threshold: float) -> tuple[np.ndarray, int]:
    segmented = np.asarray(poses, dtype=np.float32).copy()
    step_lengths = trajectory_step_lengths(segmented)
    jump_indices = np.where(step_lengths > jump_threshold)[0] + 1
    if jump_indices.size:
        segmented[jump_indices, 0] = np.nan
        segmented[jump_indices, 1] = np.nan
    return segmented, int(jump_indices.size)


def piecewise_trajectory_length(poses: np.ndarray, jump_threshold: float) -> float:
    steps = trajectory_step_lengths(poses)
    if steps.size == 0:
        return 0.0
    return float(steps[steps <= jump_threshold].sum())


def trajectory_span_xy(poses: np.ndarray) -> tuple[float, float]:
    if poses.size == 0:
        return 0.0, 0.0
    return (
        float(np.max(poses[:, 0]) - np.min(poses[:, 0])),
        float(np.max(poses[:, 1]) - np.min(poses[:, 1])),
    )


def scan_to_points(scan: np.ndarray, pose: np.ndarray, max_range: float) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(-np.pi, np.pi, scan.shape[-1], endpoint=False, dtype=np.float32)
    valid = np.isfinite(scan) & (scan > 0.05) & (scan < max_range)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    local_x = scan[valid] * np.cos(angles[valid])
    local_y = scan[valid] * np.sin(angles[valid])

    x, y, theta = pose.astype(np.float32)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    global_x = x + cos_t * local_x - sin_t * local_y
    global_y = y + sin_t * local_x + cos_t * local_y
    return global_x.astype(np.float32), global_y.astype(np.float32)


def local_scan_points(scan: np.ndarray, max_range: float) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(-np.pi, np.pi, scan.shape[-1], endpoint=False, dtype=np.float32)
    valid = np.isfinite(scan) & (scan > 0.05) & (scan < max_range)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    x = scan[valid] * np.cos(angles[valid])
    y = scan[valid] * np.sin(angles[valid])
    return x.astype(np.float32), y.astype(np.float32)


def build_sampled_map_points(
    scans: np.ndarray,
    poses: np.ndarray,
    max_range: float,
    max_scans: int = 1800,
) -> tuple[np.ndarray, np.ndarray, int]:
    sample_step = max(1, int(np.ceil(scans.shape[0] / float(max_scans))))
    points_x: list[np.ndarray] = []
    points_y: list[np.ndarray] = []
    sampled_scans = 0
    for index in range(0, scans.shape[0], sample_step):
        px, py = scan_to_points(scans[index], poses[index], max_range=max_range)
        if px.size == 0:
            continue
        points_x.append(px)
        points_y.append(py)
        sampled_scans += 1
    if not points_x:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), sampled_scans
    return np.concatenate(points_x), np.concatenate(points_y), sampled_scans


def compute_summary(
    scans: np.ndarray,
    odom: np.ndarray,
    gt: np.ndarray,
    corrections: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
    valid_ranges = scans[np.isfinite(scans) & (scans > 0.05)]
    correction_xy = np.linalg.norm(corrections[:, :2], axis=1)
    theta_abs = np.abs(corrections[:, 2])
    jump_threshold = trajectory_jump_threshold(odom)
    gt_traj_len_raw = trajectory_length(gt)
    gt_traj_len = piecewise_trajectory_length(gt, jump_threshold)
    odom_traj_len = trajectory_length(odom)
    gt_span_x, gt_span_y = trajectory_span_xy(gt)
    odom_span_x, odom_span_y = trajectory_span_xy(odom)
    gt_jump_count = int(np.sum(trajectory_step_lengths(gt) > jump_threshold))

    return {
        "sample_count": int(scans.shape[0]),
        "scan_beam_count": int(scans.shape[1]),
        "valid_return_ratio": float(np.mean(np.isfinite(scans) & (scans > 0.05))),
        "valid_return_count": int(np.sum(np.isfinite(scans) & (scans > 0.05))),
        "range_mean_m": float(np.mean(valid_ranges)) if valid_ranges.size else 0.0,
        "range_median_m": float(np.median(valid_ranges)) if valid_ranges.size else 0.0,
        "range_p95_m": float(np.percentile(valid_ranges, 95)) if valid_ranges.size else 0.0,
        "range_max_m": float(np.max(valid_ranges)) if valid_ranges.size else 0.0,
        "trajectory_length_m": gt_traj_len,
        "trajectory_pose_source": "ground_truth",
        "trajectory_x_span_m": gt_span_x,
        "trajectory_y_span_m": gt_span_y,
        "gt_trajectory_length_m": gt_traj_len,
        "gt_raw_trajectory_length_m": gt_traj_len_raw,
        "gt_trajectory_x_span_m": gt_span_x,
        "gt_trajectory_y_span_m": gt_span_y,
        "gt_discontinuity_jump_threshold_m": jump_threshold,
        "gt_discontinuity_jump_count": gt_jump_count,
        "odom_trajectory_length_m": odom_traj_len,
        "odom_trajectory_x_span_m": odom_span_x,
        "odom_trajectory_y_span_m": odom_span_y,
        "correction_xy_rmse_m": float(np.sqrt(np.mean(correction_xy ** 2))) if correction_xy.size else 0.0,
        "correction_theta_rmse_rad": float(np.sqrt(np.mean(corrections[:, 2] ** 2))) if corrections.size else 0.0,
        "correction_xy_mean_mm": float(np.mean(correction_xy) * 1000.0) if correction_xy.size else 0.0,
        "correction_theta_mean_deg": float(np.rad2deg(np.mean(theta_abs))) if theta_abs.size else 0.0,
        "meta": meta,
    }


def render_overview(
    dataset_dir: Path,
    scans: np.ndarray,
    odom: np.ndarray,
    gt: np.ndarray,
    corrections: np.ndarray,
    summary: dict[str, Any],
    ref_grid: np.ndarray,
    ref_resolution: float,
    ref_origin: list[float],
) -> Path:
    valid_ranges = scans[np.isfinite(scans) & (scans > 0.05)]
    plot_max_range = max(5.0, float(np.percentile(valid_ranges, 99))) if valid_ranges.size else 5.0
    points_x, points_y, sampled_scans = build_sampled_map_points(scans, gt, max_range=plot_max_range)
    gt_plot, gt_jump_count = segmented_trajectory_for_plot(
        gt, float(summary.get("gt_discontinuity_jump_threshold_m", 0.5))
    )
    gt_scatter_step = max(1, gt.shape[0] // 1200)
    summary["sampled_map_scan_count"] = int(sampled_scans)
    summary["sampled_map_point_count"] = int(points_x.size)
    summary["sampled_map_pose_source"] = "ground_truth"
    summary["gt_plot_discontinuity_count"] = int(gt_jump_count)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    configure_figure(fig)
    for ax in axes.flat:
        configure_axes(ax)

    ref_x_min = ref_origin[0]
    ref_y_min = ref_origin[1]
    ref_x_max = ref_x_min + ref_grid.shape[1] * ref_resolution
    ref_y_max = ref_y_min + ref_grid.shape[0] * ref_resolution
    ref_display = make_reference_display_grid(ref_grid)

    ax = axes[0, 0]
    ax.imshow(
        ref_display,
        extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max],
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=0.65,
        aspect="equal",
    )
    if points_x.size:
        scatter_step = max(1, points_x.size // 16000)
        ax.scatter(points_x[::scatter_step], points_y[::scatter_step], s=1.6, c="#a3e635", alpha=0.22, linewidths=0)
    ax.scatter(
        gt[::gt_scatter_step, 0],
        gt[::gt_scatter_step, 1],
        s=11,
        c="#f8fafc",
        alpha=0.58,
        linewidths=0,
        zorder=4,
        label="GT próbki",
    )
    ax.plot(gt_plot[:, 0], gt_plot[:, 1], color="#f8fafc", linewidth=2.1, alpha=0.92, label="GT")
    ax.plot(odom[:, 0], odom[:, 1], color="#38bdf8", linewidth=1.8, alpha=0.92, label="Odometra")
    ax.scatter(gt[0, 0], gt[0, 1], s=68, c="#22c55e", zorder=5, label="Start GT")
    ax.scatter(gt[-1, 0], gt[-1, 1], s=72, c="#f97316", marker="X", zorder=5, label="Koniec GT")
    ax.scatter(odom[-1, 0], odom[-1, 1], s=40, c="#0ea5e9", marker="D", zorder=5, label="Koniec odometrii")
    legend = ax.legend(loc="lower right")
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#334155")
    for text in legend.get_texts():
        text.set_color("#e5eefc")
    ax.set_title("Trajektorie GT i odometrii + mapa punktowa")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    in_bounds = (
        (odom[:, 0] >= ref_x_min)
        & (odom[:, 0] <= ref_x_max)
        & (odom[:, 1] >= ref_y_min)
        & (odom[:, 1] <= ref_y_max)
    )
    outside_pct = float(100.0 * (1.0 - np.mean(in_bounds.astype(np.float32)))) if odom.shape[0] else 0.0
    ax.text(
        0.02,
        0.02,
        f"Poza granicą mapy (odom): {outside_pct:.1f}%",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        color="#dbe7ff",
        fontsize=8.5,
        bbox={"facecolor": "#0f172a", "edgecolor": "#334155", "boxstyle": "round,pad=0.3", "alpha": 0.8},
    )

    ax = axes[0, 1]
    if points_x.size:
        hb = ax.hexbin(points_x, points_y, gridsize=85, mincnt=1, cmap="viridis", linewidths=0)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Gęstość punktów", color="#dbe7ff")
        cbar.ax.yaxis.set_tick_params(color="#dbe7ff")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#dbe7ff")
    ax.scatter(
        gt[::gt_scatter_step, 0],
        gt[::gt_scatter_step, 1],
        s=8,
        c="#f8fafc",
        alpha=0.42,
        linewidths=0,
        zorder=4,
        label="GT próbki",
    )
    ax.plot(gt_plot[:, 0], gt_plot[:, 1], color="#e2e8f0", linewidth=1.4, alpha=0.82, label="GT")
    ax.plot(odom[:, 0], odom[:, 1], color="#38bdf8", linewidth=1.1, alpha=0.72, label="Odometra")
    legend = ax.legend(loc="upper right")
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#334155")
    for text in legend.get_texts():
        text.set_color("#e5eefc")
    ax.set_title("Gęstość punktów LiDAR")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1, 0]
    if valid_ranges.size:
        clip_max = float(np.percentile(valid_ranges, 99.5))
        bins = np.linspace(0.0, max(clip_max, 1.0), 48)
        ax.hist(np.clip(valid_ranges, 0.0, clip_max), bins=bins, color="#38bdf8", alpha=0.82, edgecolor="#0f172a")
        ax.axvline(summary["range_mean_m"], color="#a3e635", linestyle="--", linewidth=1.4, label="Średnia")
        ax.axvline(summary["range_median_m"], color="#f97316", linestyle=":", linewidth=1.6, label="Mediana")
        legend = ax.legend(loc="upper right")
        legend.get_frame().set_facecolor("#111827")
        legend.get_frame().set_edgecolor("#334155")
        for text in legend.get_texts():
            text.set_color("#e5eefc")
    ax.set_title("Rozkład odległości LiDAR")
    ax.set_xlabel("Odległość [m]")
    ax.set_ylabel("Liczba pomiarów")

    ax = axes[1, 1]
    dx_mm = corrections[:, 0] * 1000.0
    dy_mm = corrections[:, 1] * 1000.0
    hb = ax.hexbin(dx_mm, dy_mm, gridsize=55, mincnt=1, cmap="plasma", linewidths=0)
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axvline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.45)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Liczba próbek", color="#dbe7ff")
    cbar.ax.yaxis.set_tick_params(color="#dbe7ff")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#dbe7ff")
    ax.set_title("Korekty pozycji GT vs odometria")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("dy [mm]")
    info_text = (
        f"RMSE XY: {summary['correction_xy_rmse_m']:.4f} m\n"
        f"RMSE theta: {summary['correction_theta_rmse_rad']:.4f} rad\n"
        f"Długość GT / odom: {summary['gt_trajectory_length_m']:.1f} / {summary['odom_trajectory_length_m']:.1f} m\n"
        f"Przerwy GT po skokach > {summary['gt_discontinuity_jump_threshold_m']:.2f} m: {summary['gt_discontinuity_jump_count']}"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#e5eefc",
        fontsize=9,
        bbox={"facecolor": "#0f172a", "edgecolor": "#334155", "boxstyle": "round,pad=0.4", "alpha": 0.86},
    )

    fig.suptitle(
        f"Inspekcja datasetu: {dataset_dir.name} | próbki={summary['sample_count']} | promieni/skan={summary['scan_beam_count']}",
        color="#f8fafc",
        fontsize=15,
        y=0.995,
    )

    output_path = dataset_dir / OVERVIEW_NAME
    save_figure(fig, output_path)
    return output_path


def render_scan_gallery(dataset_dir: Path, scans: np.ndarray) -> Path:
    valid_ranges = scans[np.isfinite(scans) & (scans > 0.05)]
    plot_max_range = max(4.0, float(np.percentile(valid_ranges, 99))) if valid_ranges.size else 4.0
    indices = sorted({0, scans.shape[0] // 2, scans.shape[0] - 1})

    fig, axes = plt.subplots(1, len(indices), figsize=(15.5, 4.8))
    configure_figure(fig)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, sample_index in zip(axes, indices):
        configure_axes(ax)
        local_x, local_y = local_scan_points(scans[sample_index], max_range=plot_max_range)
        ax.scatter(local_x, local_y, s=5, c="#38bdf8", alpha=0.88, linewidths=0)
        ax.scatter([0.0], [0.0], s=60, c="#a3e635", marker="o", zorder=5)
        ax.set_title(f"Skan lokalny #{sample_index}")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim(-plot_max_range, plot_max_range)
        ax.set_ylim(-plot_max_range, plot_max_range)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Reprezentatywne skany lokalne", color="#f8fafc", fontsize=15, y=0.98)
    output_path = dataset_dir / SCANS_NAME
    save_figure(fig, output_path)
    return output_path


def render_baseline_target_components(dataset_dir: Path, corrections: np.ndarray) -> tuple[Path, dict[str, Any]]:
    dx_mm = corrections[:, 0].astype(np.float32) * 1000.0
    dy_mm = corrections[:, 1].astype(np.float32) * 1000.0
    dtheta_deg = np.rad2deg(wrap_angle(corrections[:, 2])).astype(np.float32)

    dx_summary = stats_1d(dx_mm)
    dy_summary = stats_1d(dy_mm)
    dtheta_summary = stats_1d(dtheta_deg)

    dx_limit = nice_symmetric_limit(dx_mm, 500.0, 100.0)
    dy_limit = nice_symmetric_limit(dy_mm, 250.0, 50.0)
    dtheta_limit = nice_symmetric_limit(dtheta_deg, 45.0, 15.0)

    output_path = render_component_histograms(
        dataset_dir / TARGET_COMPONENTS_NAME,
        title="AI: rozkład podpisanych etykiet modelu",
        ncols=3,
        specs=[
            {
                "values": dx_mm,
                "bins": np.linspace(-dx_limit, dx_limit, 41),
                "title": "dx korekty",
                "xlabel": "dx [mm]",
                "color": "#38bdf8",
                "xlim": (-dx_limit, dx_limit),
                "vertical_lines": [(0.0, "0 mm", "#a3e635")],
                "annotations": [
                    f"Średnia / mediana: {dx_summary['mean']:.1f} / {dx_summary['median']:.1f}",
                    f"Min / max: {dx_summary['min']:.1f} / {dx_summary['max']:.1f}",
                    f"+ / -: {dx_summary['positive_ratio_pct']:.1f}% / {dx_summary['negative_ratio_pct']:.1f}%",
                ],
            },
            {
                "values": dy_mm,
                "bins": np.linspace(-dy_limit, dy_limit, 41),
                "title": "dy korekty",
                "xlabel": "dy [mm]",
                "color": "#22c55e",
                "xlim": (-dy_limit, dy_limit),
                "vertical_lines": [(0.0, "0 mm", "#a3e635")],
                "annotations": [
                    f"Średnia / mediana: {dy_summary['mean']:.1f} / {dy_summary['median']:.1f}",
                    f"Min / max: {dy_summary['min']:.1f} / {dy_summary['max']:.1f}",
                    f"+ / -: {dy_summary['positive_ratio_pct']:.1f}% / {dy_summary['negative_ratio_pct']:.1f}%",
                ],
            },
            {
                "values": dtheta_deg,
                "bins": np.linspace(-dtheta_limit, dtheta_limit, 49),
                "title": "dtheta korekty",
                "xlabel": "dtheta [deg]",
                "color": "#f97316",
                "xlim": (-dtheta_limit, dtheta_limit),
                "vertical_lines": [(0.0, "0 deg", "#a3e635")],
                "annotations": [
                    f"Średnia / mediana: {dtheta_summary['mean']:.1f} / {dtheta_summary['median']:.1f}",
                    f"Min / max: {dtheta_summary['min']:.1f} / {dtheta_summary['max']:.1f}",
                    f"+ / -: {dtheta_summary['positive_ratio_pct']:.1f}% / {dtheta_summary['negative_ratio_pct']:.1f}%",
                ],
            },
        ],
    )
    return output_path, {
        "correction_dx_mm_signed": dx_summary,
        "correction_dy_mm_signed": dy_summary,
        "correction_dtheta_deg_signed": dtheta_summary,
    }


def generate_baseline_report(dataset_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        return {}, {}

    print(f"[DATASET] Wczytywanie: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    scans = np.asarray(data["X_scan"], dtype=np.float32)
    odom = np.asarray(data["X_odom"], dtype=np.float32)
    corrections = np.asarray(data["Y"], dtype=np.float32)
    gt = reconstruct_gt_poses(odom, corrections)
    meta = data["meta"].item() if "meta" in data else {}
    meta = meta if isinstance(meta, dict) else {}

    dataset_world_name = ""
    config_snapshot_path = dataset_dir / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        try:
            cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
            simulation = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
            dataset_world_name = str(simulation.get("train_world", "")).strip()
        except Exception:
            dataset_world_name = ""
    ref_map_yaml, ref_grid, ref_resolution, ref_origin = load_reference_map_local(
        dataset_dir,
        world_name=dataset_world_name or None,
        prefer_world_mapping=True,
        keep_world_origin=True,
    )
    summary = compute_summary(scans, odom, gt, corrections, meta)
    summary.update(
        {
            "dataset_dir": str(dataset_dir.resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "reference_map_yaml": str(ref_map_yaml.resolve()),
        }
    )

    overview_path = render_overview(dataset_dir, scans, odom, gt, corrections, summary, ref_grid, ref_resolution, ref_origin)
    scans_path = render_scan_gallery(dataset_dir, scans)
    target_components_path, target_summary = render_baseline_target_components(dataset_dir, corrections)
    summary.update(target_summary)
    print(f"[DATASET] Zapisano widok ogólny: {overview_path}")
    print(f"[DATASET] Zapisano galerię skanów: {scans_path}")
    print(f"[DATASET] Zapisano etykiety modelu: {target_components_path}")
    print(
        "[DATASET] Najważniejsze: "
        f"próbki={summary['sample_count']}, "
        f"valid_ratio={summary['valid_return_ratio']:.3f}, "
        f"traj={summary['trajectory_length_m']:.2f} m, "
        f"rmse_xy={summary['correction_xy_rmse_m']:.4f} m"
    )
    return {
        "overview": overview_path,
        "scans": scans_path,
        "target_components": target_components_path,
        "legacy": overview_path,
    }, normalize_json_value(summary)


def render_histogram_coverage(
    output_path: Path,
    values: np.ndarray,
    *,
    bins: np.ndarray,
    title: str,
    xlabel: str,
    color: str,
    annotations: list[str],
    xlim: tuple[float, float] | None = None,
    vertical_lines: list[tuple[float, str, str]] | None = None,
    y_log: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    configure_figure(fig)
    configure_axes(ax)

    ax.hist(values, bins=bins, color=color, alpha=0.86, edgecolor="#0f172a")
    if xlim is not None:
        ax.set_xlim(*xlim)

    for line_x, label, line_color in vertical_lines or []:
        ax.axvline(line_x, color=line_color, linestyle="--", linewidth=1.5, label=label)

    if vertical_lines:
        legend = ax.legend(loc="upper right")
        legend.get_frame().set_facecolor("#111827")
        legend.get_frame().set_edgecolor("#334155")
        for text in legend.get_texts():
            text.set_color("#e5eefc")
    if y_log:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Liczba próbek")
    ax.text(
        0.02,
        0.98,
        "\n".join(annotations),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#e5eefc",
        fontsize=9,
        bbox={"facecolor": "#0f172a", "edgecolor": "#334155", "boxstyle": "round,pad=0.4", "alpha": 0.86},
    )
    save_figure(fig, output_path)
    return output_path


def nice_symmetric_limit(values: np.ndarray, minimum: float, step: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float(minimum)
    max_abs = float(np.max(np.abs(arr)))
    if step <= 0.0:
        return float(max(minimum, max_abs))
    return float(max(minimum, np.ceil(max_abs / step) * step))


def nice_positive_limit(values: np.ndarray, minimum: float, step: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float(minimum)
    max_val = float(np.max(arr))
    if step <= 0.0:
        return float(max(minimum, max_val))
    return float(max(minimum, np.ceil(max_val / step) * step))


def render_component_histograms(
    output_path: Path,
    *,
    title: str,
    specs: list[dict[str, Any]],
    ncols: int,
) -> Path:
    n_items = max(1, len(specs))
    ncols = max(1, min(ncols, n_items))
    nrows = int(np.ceil(n_items / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows))
    configure_figure(fig)

    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes_arr.flat:
        configure_axes(ax)

    for ax, spec in zip(axes_arr.flat, specs):
        values = np.asarray(spec["values"], dtype=np.float32).reshape(-1)
        bins = np.asarray(spec["bins"], dtype=np.float32)
        ax.hist(values, bins=bins, color=str(spec["color"]), alpha=0.86, edgecolor="#0f172a")

        xlim = spec.get("xlim")
        if xlim is not None:
            ax.set_xlim(*xlim)

        for line_x, label, line_color in spec.get("vertical_lines", []) or []:
            ax.axvline(line_x, color=line_color, linestyle="--", linewidth=1.4, label=label)

        if spec.get("vertical_lines"):
            legend = ax.legend(loc="upper right")
            legend.get_frame().set_facecolor("#111827")
            legend.get_frame().set_edgecolor("#334155")
            for text in legend.get_texts():
                text.set_color("#e5eefc")

        ax.set_title(str(spec["title"]))
        ax.set_xlabel(str(spec["xlabel"]))
        ax.set_ylabel("Liczba próbek")

        annotations = [str(item) for item in spec.get("annotations", []) or []]
        if annotations:
            ax.text(
                0.02,
                0.98,
                "\n".join(annotations),
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="#e5eefc",
                fontsize=8.5,
                bbox={"facecolor": "#0f172a", "edgecolor": "#334155", "boxstyle": "round,pad=0.35", "alpha": 0.86},
            )

    for ax in axes_arr.flat[n_items:]:
        fig.delaxes(ax)

    fig.suptitle(title, color="#f8fafc", fontsize=15)
    save_figure(fig, output_path)
    return output_path


def generate_robak_coverage_report(dataset_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    dataset_path = dataset_dir / "dataset_robak.npz"
    if not dataset_path.exists():
        return {}, {}

    print(f"[ROBAK] Wczytywanie: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as data:
        labels = np.asarray(data["Y"], dtype=np.float32)
        meta = data["meta"].item() if "meta" in data else {}

    if labels.size == 0:
        return {}, {}

    translation_cm = np.linalg.norm(labels[:, :2], axis=1).astype(np.float32) * 100.0
    rotation_deg = np.rad2deg(wrap_angle(labels[:, 2])).astype(np.float32)

    translation_abs_summary = stats_1d(translation_cm)
    rotation_abs_deg = np.abs(rotation_deg).astype(np.float32)
    rotation_summary = stats_1d(rotation_deg)
    rotation_abs_summary = stats_1d(rotation_abs_deg)

    distance_target_50 = float(np.mean(translation_cm <= 50.0) * 100.0)
    distance_target_100 = float(np.mean(translation_cm <= 100.0) * 100.0)
    rotation_target_90 = float(np.mean(rotation_abs_deg <= 90.0) * 100.0)
    rotation_target_180 = float(np.mean(rotation_abs_deg <= 180.0) * 100.0)
    overflow_100cm = int(np.sum(translation_cm > 100.0))
    base_pair_count = int(meta.get("pair_accept_count", labels.shape[0]))
    augmented_sample_count = int(meta.get("augment_added_samples", 0))
    dx_cm = labels[:, 0].astype(np.float32) * 100.0
    dy_cm = labels[:, 1].astype(np.float32) * 100.0
    dx_summary = stats_1d(dx_cm)
    dy_summary = stats_1d(dy_cm)
    dx_limit_cm = nice_symmetric_limit(dx_cm, 25.0, 5.0)
    dy_limit_cm = nice_symmetric_limit(dy_cm, 25.0, 5.0)
    rotation_limit_deg = nice_symmetric_limit(rotation_deg, 5.0, 5.0)

    translation_path = render_histogram_coverage(
        dataset_dir / ROBAK_DISTANCE_NAME,
        dx_cm,
        bins=np.linspace(-dx_limit_cm, dx_limit_cm, 41),
        title="Robak: podpisane dx lokalne",
        xlabel="dx [cm]",
        color="#f97316",
        xlim=(-dx_limit_cm, dx_limit_cm),
        vertical_lines=[
            (0.0, "0", "#a3e635"),
        ],
        annotations=[
            f"Probki: {dx_summary['count']}",
            f"Pary bazowe / augment: {base_pair_count} / {augmented_sample_count}",
            f"Srednia / mediana: {dx_summary['mean']:.2f} / {dx_summary['median']:.2f} cm",
            f"P95 / max: {dx_summary['p95']:.2f} / {dx_summary['max']:.1f} cm",
            f"+ / - / 0: {dx_summary['positive_ratio_pct']:.1f}% / {dx_summary['negative_ratio_pct']:.1f}% / {dx_summary['zero_ratio_pct']:.1f}%",
            f"Offsety: {normalize_json_value(meta.get('offsets', []))}",
        ],
        y_log=True,
    )
    rotation_path = render_histogram_coverage(
        dataset_dir / ROBAK_ROTATION_NAME,
        rotation_deg,
        bins=np.linspace(-rotation_limit_deg, rotation_limit_deg, 49),
        title="Robak: podpisane dtheta miedzy skanami",
        xlabel="dtheta [deg]",
        color="#22c55e",
        xlim=(-rotation_limit_deg, rotation_limit_deg),
        vertical_lines=[
            (0.0, "0", "#a3e635"),
        ],
        annotations=[
            f"Probki: {rotation_summary['count']}",
            f"Pary bazowe / augment: {base_pair_count} / {augmented_sample_count}",
            f"Srednia / mediana: {rotation_summary['mean']:.2f} / {rotation_summary['median']:.2f} deg",
            f"P95 / max: {rotation_summary['p95']:.2f} / {rotation_summary['max']:.1f} deg",
            f"+ / - / 0: {rotation_summary['positive_ratio_pct']:.1f}% / {rotation_summary['negative_ratio_pct']:.1f}% / {rotation_summary['zero_ratio_pct']:.1f}%",
            f"Offsety: {normalize_json_value(meta.get('offsets', []))}",
        ],
        y_log=True,
    )
    components_path = render_component_histograms(
        dataset_dir / ROBAK_COMPONENTS_NAME,
        title="Robak: podpisane skladowe etykiet",
        ncols=3,
        specs=[
            {
                "values": dx_cm,
                "bins": np.linspace(-dx_limit_cm, dx_limit_cm, 41),
                "title": "dx lokalne",
                "xlabel": "dx [cm]",
                "color": "#f97316",
                "xlim": (-dx_limit_cm, dx_limit_cm),
                "vertical_lines": [(0.0, "0", "#a3e635")],
                "annotations": [
                    f"Srednia / mediana: {dx_summary['mean']:.2f} / {dx_summary['median']:.2f}",
                    f"P95: {dx_summary['p95']:.2f}, min/max: {dx_summary['min']:.1f}/{dx_summary['max']:.1f}",
                ],
            },
            {
                "values": dy_cm,
                "bins": np.linspace(-dy_limit_cm, dy_limit_cm, 41),
                "title": "dy lokalne",
                "xlabel": "dy [cm]",
                "color": "#22c55e",
                "xlim": (-dy_limit_cm, dy_limit_cm),
                "vertical_lines": [(0.0, "0", "#a3e635")],
                "annotations": [
                    f"Srednia / mediana: {dy_summary['mean']:.2f} / {dy_summary['median']:.2f}",
                    f"P95: {dy_summary['p95']:.2f}, min/max: {dy_summary['min']:.1f}/{dy_summary['max']:.1f}",
                ],
            },
            {
                "values": rotation_deg,
                "bins": np.linspace(-rotation_limit_deg, rotation_limit_deg, 49),
                "title": "dtheta",
                "xlabel": "dtheta [deg]",
                "color": "#0ea5e9",
                "xlim": (-rotation_limit_deg, rotation_limit_deg),
                "vertical_lines": [(0.0, "0", "#a3e635")],
                "annotations": [
                    f"Srednia / mediana: {rotation_summary['mean']:.2f} / {rotation_summary['median']:.2f}",
                    f"P95: {rotation_summary['p95']:.2f}, min/max: {rotation_summary['min']:.1f}/{rotation_summary['max']:.1f}",
                ],
            },
        ],
    )

    summary = {
        "dataset_path": str(dataset_path.resolve()),
        "sample_count": int(labels.shape[0]),
        "base_pair_count": base_pair_count,
        "augmented_sample_count": augmented_sample_count,
        "translation_cm": translation_abs_summary,
        "dx_local_cm_signed": dx_summary,
        "dy_local_cm_signed": dy_summary,
        "rotation_deg_signed": rotation_summary,
        "rotation_deg_abs": rotation_abs_summary,
        "coverage_pct_0_50cm": distance_target_50,
        "coverage_pct_0_100cm": distance_target_100,
        "coverage_pct_abs_rotation_0_90deg": rotation_target_90,
        "coverage_pct_abs_rotation_0_180deg": rotation_target_180,
        "samples_above_100cm": overflow_100cm,
        "meta": normalize_json_value(meta if isinstance(meta, dict) else {}),
    }
    return {
        "summary": dataset_dir / ROBAK_SUMMARY_NAME,
        "distance": translation_path,
        "rotation": rotation_path,
        "components": components_path,
    }, summary


def generate_rywak_coverage_report(dataset_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    dataset_path = dataset_dir / "dataset_rywak.npz"
    if not dataset_path.exists():
        return {}, {}

    print(f"[RYWAK] Wczytywanie: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as data:
        labels = np.asarray(data["Y"], dtype=np.float32)
        meta = data["meta"].item() if "meta" in data else {}

    if labels.size == 0:
        return {}, {}

    linear_velocity = labels[:, 0].astype(np.float32)
    angular_velocity = labels[:, 1].astype(np.float32)
    linear_abs = np.abs(linear_velocity).astype(np.float32)
    angular_abs = np.abs(angular_velocity).astype(np.float32)

    linear_abs_summary = stats_1d(linear_abs)
    angular_abs_summary = stats_1d(angular_abs)
    signed_linear_summary = stats_1d(linear_velocity)
    signed_angular_summary = stats_1d(angular_velocity)

    linear_target_1 = float(np.mean(linear_abs <= 1.0) * 100.0)
    linear_target_2 = float(np.mean(linear_abs <= 2.0) * 100.0)
    angular_target_3 = float(np.mean(angular_abs <= 3.0) * 100.0)
    linear_over_2 = int(np.sum(linear_abs > 2.0))
    angular_over_3 = int(np.sum(angular_abs > 3.0))
    v_clip_abs = float(meta.get("v_clip_abs", 0.0)) if isinstance(meta, dict) else 0.0
    w_clip_abs = float(meta.get("w_clip_abs", 0.0)) if isinstance(meta, dict) else 0.0
    linear_signed_limit = max(nice_symmetric_limit(linear_velocity, 0.5, 0.1), v_clip_abs if v_clip_abs > 0.0 else 0.0)
    angular_signed_limit = max(nice_symmetric_limit(angular_velocity, 1.0, 0.25), w_clip_abs if w_clip_abs > 0.0 else 0.0)

    linear_path = render_histogram_coverage(
        dataset_dir / RYWAK_LINEAR_NAME,
        linear_velocity,
        bins=np.linspace(-linear_signed_limit, linear_signed_limit, 41),
        title="Rywak: podpisana prędkość liniowa",
        xlabel="v [m/s]",
        color="#22c55e",
        xlim=(-linear_signed_limit, linear_signed_limit),
        vertical_lines=[
            item
            for item in [
                (0.0, "0 m/s", "#a3e635"),
                (-v_clip_abs, "-clip", "#38bdf8") if v_clip_abs > 0.0 else None,
                (v_clip_abs, "+clip", "#38bdf8") if v_clip_abs > 0.0 else None,
            ]
            if item is not None
        ],
        annotations=[
            f"Próbki: {signed_linear_summary['count']}",
            f"Zapisane / odrzucone: {labels.shape[0]} / {int(meta.get('sample_filter_reject_count', 0))}",
            f"Średnia / mediana: {signed_linear_summary['mean']:.3f} / {signed_linear_summary['median']:.3f} m/s",
            f"Min / max: {signed_linear_summary['min']:.3f} / {signed_linear_summary['max']:.3f} m/s",
            f"+ / - / 0: {signed_linear_summary['positive_ratio_pct']:.1f}% / {signed_linear_summary['negative_ratio_pct']:.1f}% / {signed_linear_summary['zero_ratio_pct']:.1f}%",
            f"Clip v: +/-{v_clip_abs:.3f} m/s" if v_clip_abs > 0.0 else "Clip v: brak",
            "Histogram pokazuje podpisane v, bez przechodzenia do modułu.",
        ],
    )
    angular_path = render_histogram_coverage(
        dataset_dir / RYWAK_ANGULAR_NAME,
        angular_velocity,
        bins=np.linspace(-angular_signed_limit, angular_signed_limit, 49),
        title="Rywak: podpisana prędkość kątowa",
        xlabel="omega [rad/s]",
        color="#a855f7",
        xlim=(-angular_signed_limit, angular_signed_limit),
        vertical_lines=[
            item
            for item in [
                (0.0, "0 rad/s", "#a3e635"),
                (-w_clip_abs, "-clip", "#38bdf8") if w_clip_abs > 0.0 else None,
                (w_clip_abs, "+clip", "#38bdf8") if w_clip_abs > 0.0 else None,
            ]
            if item is not None
        ],
        annotations=[
            f"Próbki: {signed_angular_summary['count']}",
            f"Zapisane / odrzucone: {labels.shape[0]} / {int(meta.get('sample_filter_reject_count', 0))}",
            f"Średnia / mediana: {signed_angular_summary['mean']:.3f} / {signed_angular_summary['median']:.3f} rad/s",
            f"Min / max: {signed_angular_summary['min']:.3f} / {signed_angular_summary['max']:.3f} rad/s",
            f"+ / - / 0: {signed_angular_summary['positive_ratio_pct']:.1f}% / {signed_angular_summary['negative_ratio_pct']:.1f}% / {signed_angular_summary['zero_ratio_pct']:.1f}%",
            f"Clip omega: +/-{w_clip_abs:.3f} rad/s" if w_clip_abs > 0.0 else "Clip omega: brak",
            "Znak omega rozróżnia kierunek skrętu i jest istotny.",
        ],
    )
    signed_path = render_component_histograms(
        dataset_dir / RYWAK_SIGNED_NAME,
        title="Rywak: rozkład podpisanych etykiet modelu",
        ncols=2,
        specs=[
            {
                "values": linear_velocity,
                "bins": np.linspace(-linear_signed_limit, linear_signed_limit, 41),
                "title": "v podpisane",
                "xlabel": "v [m/s]",
                "color": "#22c55e",
                "xlim": (-linear_signed_limit, linear_signed_limit),
                "vertical_lines": [(0.0, "0 m/s", "#a3e635")],
                "annotations": [
                    f"Średnia / mediana: {signed_linear_summary['mean']:.3f} / {signed_linear_summary['median']:.3f}",
                    f"Min / max: {signed_linear_summary['min']:.3f} / {signed_linear_summary['max']:.3f}",
                ],
            },
            {
                "values": angular_velocity,
                "bins": np.linspace(-angular_signed_limit, angular_signed_limit, 49),
                "title": "omega podpisane",
                "xlabel": "omega [rad/s]",
                "color": "#a855f7",
                "xlim": (-angular_signed_limit, angular_signed_limit),
                "vertical_lines": [(0.0, "0 rad/s", "#a3e635")],
                "annotations": [
                    f"Średnia / mediana: {signed_angular_summary['mean']:.3f} / {signed_angular_summary['median']:.3f}",
                    f"Min / max: {signed_angular_summary['min']:.3f} / {signed_angular_summary['max']:.3f}",
                ],
            },
        ],
    )

    summary = {
        "dataset_path": str(dataset_path.resolve()),
        "sample_count": int(labels.shape[0]),
        "linear_velocity_abs_mps": linear_abs_summary,
        "linear_velocity_signed_mps": signed_linear_summary,
        "angular_velocity_abs_radps": angular_abs_summary,
        "angular_velocity_signed_radps": signed_angular_summary,
        "coverage_pct_abs_linear_0_1mps": linear_target_1,
        "coverage_pct_abs_linear_0_2mps": linear_target_2,
        "coverage_pct_abs_angular_0_3radps": angular_target_3,
        "samples_above_2mps": linear_over_2,
        "samples_above_3radps": angular_over_3,
        "meta": normalize_json_value(meta if isinstance(meta, dict) else {}),
    }
    return {
        "summary": dataset_dir / RYWAK_SUMMARY_NAME,
        "linear_velocity": linear_path,
        "angular_velocity": angular_path,
        "signed_velocity": signed_path,
    }, summary


def load_trajectory_series(path: Path, time_key: str, pose_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if time_key not in data or pose_key not in data:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        times = np.asarray(data[time_key], dtype=np.float32).reshape(-1)
        poses = np.asarray(data[pose_key], dtype=np.float32)
    if poses.size == 0:
        return times[:0], np.zeros((0, 3), dtype=np.float32)
    poses = poses.reshape((-1, 3))
    n = min(times.shape[0], poses.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return times[:n].astype(np.float32), poses[:n].astype(np.float32)


def compute_segment_speeds(times: np.ndarray, poses: np.ndarray) -> np.ndarray:
    tt = np.asarray(times, dtype=np.float32).reshape(-1)
    pp = np.asarray(poses, dtype=np.float32).reshape((-1, 3))
    n = min(tt.shape[0], pp.shape[0])
    if n <= 1:
        return np.zeros((0,), dtype=np.float32)
    dt = np.diff(tt[:n])
    step = np.linalg.norm(np.diff(pp[:n, :2], axis=0), axis=1).astype(np.float32)
    out = np.zeros_like(step, dtype=np.float32)
    valid = np.isfinite(dt) & (dt > 1e-6)
    out[valid] = step[valid] / dt[valid]
    return out.astype(np.float32)


def render_speed_trajectory_report(
    dataset_dir: Path,
    trajectory_path: Path,
    ref_grid: np.ndarray,
    ref_resolution: float,
    ref_origin: list[float],
    include_ai: bool = False,
) -> tuple[Path, dict[str, Any]]:
    series_specs = [
        ("time_s", "baseline_xytheta", "SLAM ROS", "#c2410c"),
        ("robak_time_s", "robak_xytheta", "Robak", "#dc2626"),
        ("rywak_time_s", "rywak_xytheta", "Rywak", "#4d7c0f"),
    ]
    if include_ai:
        series_specs.append(("ai_time_s", "ai_xytheta", "AI", "#16a34a"))

    loaded: list[dict[str, Any]] = []
    for time_key, pose_key, label, color in series_specs:
        times, poses = load_trajectory_series(trajectory_path, time_key, pose_key)
        if times.size == 0 or poses.size == 0 or poses.shape[0] < 2:
            continue
        segment_speeds = compute_segment_speeds(times, poses)
        if segment_speeds.size == 0:
            continue
        loaded.append(
            {
                "label": label,
                "color": color,
                "times": times,
                "poses": poses,
                "segment_speeds": segment_speeds,
                "speed_summary": stats_1d(segment_speeds),
            }
        )

    if not loaded:
        return dataset_dir / TRAJECTORY_SPEED_NAME, {}

    global_speed_limit = max(
        0.1,
        float(
            np.max(
                [
                    max(
                        item["speed_summary"]["p95"],
                        item["speed_summary"]["max"],
                    )
                    for item in loaded
                ]
            )
        ),
    )
    speed_norm = Normalize(vmin=0.0, vmax=global_speed_limit)
    speed_cmap = LinearSegmentedColormap.from_list(
        "speed_heatmap",
        ["#1d4ed8", "#38bdf8", "#facc15", "#dc2626"],
    )

    ref_x_min = ref_origin[0]
    ref_y_min = ref_origin[1]
    ref_x_max = ref_x_min + ref_grid.shape[1] * ref_resolution
    ref_y_max = ref_y_min + ref_grid.shape[0] * ref_resolution
    ref_display = make_reference_display_grid(ref_grid)

    n = len(loaded)
    ncols = min(2, n)
    nrows = int(np.ceil(n / max(1, ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.5 * nrows), squeeze=False)
    configure_figure(fig)
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, loaded):
        configure_axes(ax)
        poses = np.asarray(item["poses"], dtype=np.float32)
        segment_speeds = np.asarray(item["segment_speeds"], dtype=np.float32)
        ax.imshow(
            ref_display,
            extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.58,
            aspect="equal",
            zorder=0,
        )
        map_rect = Rectangle(
            (ref_x_min, ref_y_min),
            ref_x_max - ref_x_min,
            ref_y_max - ref_y_min,
            fill=False,
            edgecolor="#e2e8f0",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            alpha=0.9,
        )
        ax.add_patch(map_rect)

        points = poses[:, :2].reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap=speed_cmap,
            norm=speed_norm,
            linewidths=2.8,
            alpha=0.97,
            zorder=3,
        )
        lc.set_array(segment_speeds)
        ax.add_collection(lc)
        ax.scatter(poses[0, 0], poses[0, 1], s=52, c="#22c55e", zorder=4, label="Start")
        ax.scatter(poses[-1, 0], poses[-1, 1], s=58, c="#f97316", marker="X", zorder=4, label="Koniec")

        summary = item["speed_summary"]
        ax.set_title(str(item["label"]))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")
        x_vals = poses[:, 0]
        y_vals = poses[:, 1]
        x_min_data = float(np.min(x_vals))
        x_max_data = float(np.max(x_vals))
        y_min_data = float(np.min(y_vals))
        y_max_data = float(np.max(y_vals))
        x_min = min(ref_x_min, x_min_data)
        x_max = max(ref_x_max, x_max_data)
        y_min = min(ref_y_min, y_min_data)
        y_max = max(ref_y_max, y_max_data)
        x_pad = max(1.0, 0.05 * max(x_max - x_min, 1.0))
        y_pad = max(1.0, 0.05 * max(y_max - y_min, 1.0))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"Średnia: {summary['mean']:.3f} m/s",
                    f"Mediana: {summary['median']:.3f} m/s",
                    f"P95 / max: {summary['p95']:.3f} / {summary['max']:.3f} m/s",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="#e5eefc",
            fontsize=8.5,
            bbox={"facecolor": "#0f172a", "edgecolor": "#334155", "boxstyle": "round,pad=0.35", "alpha": 0.86},
        )

    for ax in axes_flat[len(loaded):]:
        fig.delaxes(ax)

    sm = plt.cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap)
    sm.set_array(np.asarray([0.0, global_speed_limit], dtype=np.float32))
    # Keep colorbar fully outside subplot grid to avoid overlap with the last panel.
    cbar_ax = fig.add_axes([0.905, 0.14, 0.014, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Prędkość translacyjna [m/s]", color="#dbe7ff")
    cbar.ax.yaxis.set_tick_params(color="#dbe7ff")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#dbe7ff")

    fig.suptitle("Trajektorie kolorowane prędkością", color="#f8fafc", fontsize=15, y=0.99)
    output_path = dataset_dir / TRAJECTORY_SPEED_NAME
    fig.subplots_adjust(left=0.04, right=0.885, top=0.90, bottom=0.08, wspace=0.16, hspace=0.24)
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path, {
        item["label"].lower().replace(" ", "_"): normalize_json_value(item["speed_summary"])
        for item in loaded
    }


def generate_trajectory_speed_report(
    dataset_dir: Path,
    include_ai: bool = False,
) -> tuple[dict[str, Path], dict[str, Any]]:
    trajectory_path = resolve_eval_trajectory_data_path(dataset_dir)
    if not trajectory_path.exists():
        return {}, {}

    results = load_history(dataset_dir / "results.json")
    world_name = str(results.get("world_name", "")).strip() if isinstance(results, dict) else ""
    _gt_t, gt_pose = load_trajectory_series(trajectory_path, "time_s", "gt_xytheta")
    gt_xy = gt_pose[:, :2] if gt_pose.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    ref_grid, ref_resolution, ref_origin = choose_reference_display_alignment(
        dataset_dir,
        world_name,
        gt_xy,
    )
    trajectory_speed_path, speed_summary = render_speed_trajectory_report(
        dataset_dir,
        trajectory_path,
        ref_grid,
        ref_resolution,
        ref_origin,
        include_ai=include_ai,
    )
    if not trajectory_speed_path.exists():
        return {}, {}
    return {"trajectory_speed": trajectory_speed_path}, {"trajectory_speed_profiles": speed_summary}


def _traj_series_xy(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    arr = arr.reshape((-1, 3))
    return arr[:, :2]


def _traj_err_mag(data: np.lib.npyio.NpzFile, err_xy_key: str) -> np.ndarray:
    if err_xy_key not in data:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(data[err_xy_key], dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    arr = arr.reshape((-1, 2))
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2).astype(np.float32)


def _traj_abs_theta(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.abs(arr).astype(np.float32)


def generate_clean_eval_plots(dataset_dir: Path, include_ai: bool = False) -> dict[str, Path]:
    trajectory_npz = resolve_eval_trajectory_data_path(dataset_dir)
    results_path = dataset_dir / "results.json"
    map_layers_npz = resolve_eval_map_layers_path(dataset_dir)
    if not trajectory_npz.exists():
        return {}

    artifacts: dict[str, Path] = {}
    results = load_history(results_path)
    world_name = str(results.get("world_name", "")).strip() if isinstance(results, dict) else ""
    with np.load(trajectory_npz, allow_pickle=True) as data:
        gt_xy = _traj_series_xy(data, "gt_xytheta")
        baseline_xy = _traj_series_xy(data, "baseline_xytheta")
        ai_xy = _traj_series_xy(data, "ai_xytheta")
        robak_xy = _traj_series_xy(data, "robak_xytheta")
        rywak_xy = _traj_series_xy(data, "rywak_xytheta")
        t = np.asarray(data["time_s"], dtype=np.float32).reshape(-1) if "time_s" in data else np.zeros((0,), dtype=np.float32)
        e_baseline = _traj_err_mag(data, "baseline_err_xy")
        eth_baseline = _traj_abs_theta(data, "baseline_err_theta")
        t_robak = np.asarray(data["robak_time_s"], dtype=np.float32).reshape(-1) if "robak_time_s" in data else np.zeros((0,), dtype=np.float32)
        e_robak = _traj_err_mag(data, "robak_err_xy")
        eth_robak = _traj_abs_theta(data, "robak_err_theta")
        t_rywak = np.asarray(data["rywak_time_s"], dtype=np.float32).reshape(-1) if "rywak_time_s" in data else np.zeros((0,), dtype=np.float32)
        e_rywak = _traj_err_mag(data, "rywak_err_xy")
        eth_rywak = _traj_abs_theta(data, "rywak_err_theta")
        t_ai = np.asarray(data["ai_time_s"], dtype=np.float32).reshape(-1) if "ai_time_s" in data else np.zeros((0,), dtype=np.float32)
        e_ai = _traj_err_mag(data, "ai_err_xy")
        eth_ai = _traj_abs_theta(data, "ai_err_theta")

    ref_grid, ref_resolution, ref_origin = choose_reference_display_alignment(
        dataset_dir,
        world_name,
        gt_xy,
    )
    ref_x_min = ref_origin[0]
    ref_y_min = ref_origin[1]
    ref_x_max = ref_x_min + ref_grid.shape[1] * ref_resolution
    ref_y_max = ref_y_min + ref_grid.shape[0] * ref_resolution
    ref_display = make_reference_display_grid(ref_grid)

    # Trajectory plot without AI (for readability)
    traj_path = dataset_dir / "eval_trajectory.png"
    if gt_xy.shape[0] > 0 and baseline_xy.shape[0] > 0:
        fig, ax_full = plt.subplots(1, 1, figsize=(9.0, 7.2))
        ax_full.imshow(
            ref_display,
            extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.6,
            zorder=0,
        )
        ax_full.set_aspect("equal")
        ax_full.set_xlabel("x [m]")
        ax_full.set_ylabel("y [m]")
        ax_full.grid(True, alpha=0.25)
        map_rect = Rectangle(
            (ref_x_min, ref_y_min),
            ref_x_max - ref_x_min,
            ref_y_max - ref_y_min,
            fill=False,
            edgecolor="#f8fafc",
            linewidth=1.25,
            linestyle="--",
            zorder=2,
            alpha=0.95,
        )
        ax_full.add_patch(map_rect)

        series = [
            (gt_xy, "tab:blue", "GT"),
            (baseline_xy, "tab:orange", "baseline"),
            (robak_xy, "tab:red", "robak"),
            (rywak_xy, "tab:purple", "rywak"),
        ]
        if include_ai:
            series.append((ai_xy, "tab:green", "ai"))
        for arr, color, label in series:
            if arr.shape[0] > 0:
                ax_full.plot(arr[:, 0], arr[:, 1], color=color, linewidth=1.4, alpha=0.9, label=label)

        all_xy = np.concatenate([arr for arr, _, _ in series if arr.shape[0] > 0], axis=0)
        x_min = min(float(np.min(all_xy[:, 0])), ref_x_min)
        x_max = max(float(np.max(all_xy[:, 0])), ref_x_max)
        y_min = min(float(np.min(all_xy[:, 1])), ref_y_min)
        y_max = max(float(np.max(all_xy[:, 1])), ref_y_max)
        x_pad = max(2.0, 0.12 * max(x_max - x_min, 1.0))
        y_pad = max(2.0, 0.12 * max(y_max - y_min, 1.0))
        ax_full.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_full.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_full.set_title("Trajektorie wzgledem mapy referencyjnej")
        ax_full.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
        fig.savefig(traj_path, dpi=150)
        plt.close(fig)
        artifacts["trajectory_png"] = traj_path
        coord_payload = {
            "reference_map": {
                "x_min": float(ref_x_min),
                "x_max": float(ref_x_max),
                "y_min": float(ref_y_min),
                "y_max": float(ref_y_max),
                "resolution_m_per_cell": float(ref_resolution),
                "grid_shape": [int(ref_grid.shape[0]), int(ref_grid.shape[1])],
            },
            "series": {},
        }
        for arr, _color, label in series:
            if arr.shape[0] == 0:
                continue
            outside_ratio, occ_hit_ratio = _trajectory_alignment_metrics(arr, ref_grid, ref_resolution, ref_origin)
            coord_payload["series"][str(label)] = {
                "count": int(arr.shape[0]),
                "x_min": float(np.min(arr[:, 0])),
                "x_max": float(np.max(arr[:, 0])),
                "y_min": float(np.min(arr[:, 1])),
                "y_max": float(np.max(arr[:, 1])),
                "outside_map_pct": float(100.0 * outside_ratio),
                "occupied_cell_hit_pct": float(100.0 * occ_hit_ratio),
            }
        coords_path = dataset_dir / "eval_trajectory_coordinates.json"
        write_json(coords_path, coord_payload)
        artifacts["trajectory_coordinates_json"] = coords_path

    # Error plot without AI
    err_path = dataset_dir / "eval_errors.png"
    if t.size > 0 and e_baseline.size > 0 and eth_baseline.size > 0:
        fig, (ax_pos, ax_theta) = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
        ax_pos.plot(t[: e_baseline.shape[0]], e_baseline, label="baseline", linewidth=1.5)
        ax_theta.plot(t[: eth_baseline.shape[0]], eth_baseline, label="baseline", linewidth=1.5)
        if t_robak.size > 0 and e_robak.size > 0 and eth_robak.size > 0:
            n = min(t_robak.shape[0], e_robak.shape[0], eth_robak.shape[0])
            ax_pos.plot(t_robak[:n], e_robak[:n], label="robak", alpha=0.85)
            ax_theta.plot(t_robak[:n], eth_robak[:n], label="robak", alpha=0.85)
        if t_rywak.size > 0 and e_rywak.size > 0 and eth_rywak.size > 0:
            n = min(t_rywak.shape[0], e_rywak.shape[0], eth_rywak.shape[0])
            ax_pos.plot(t_rywak[:n], e_rywak[:n], label="rywak", alpha=0.85)
            ax_theta.plot(t_rywak[:n], eth_rywak[:n], label="rywak", alpha=0.85)
        if include_ai and t_ai.size > 0 and e_ai.size > 0 and eth_ai.size > 0:
            n = min(t_ai.shape[0], e_ai.shape[0], eth_ai.shape[0])
            ax_pos.plot(t_ai[:n], e_ai[:n], label="ai", alpha=0.85)
            ax_theta.plot(t_ai[:n], eth_ai[:n], label="ai", alpha=0.85)
        t_max = float(
            np.max(
                np.asarray(
                    [
                        np.max(t[: e_baseline.shape[0]]) if e_baseline.size > 0 else 0.0,
                        np.max(t_robak[: min(t_robak.shape[0], e_robak.shape[0])]) if e_robak.size > 0 else 0.0,
                        np.max(t_rywak[: min(t_rywak.shape[0], e_rywak.shape[0])]) if e_rywak.size > 0 else 0.0,
                        np.max(t_ai[: min(t_ai.shape[0], e_ai.shape[0])]) if (include_ai and e_ai.size > 0) else 0.0,
                    ],
                    dtype=np.float32,
                )
            )
        )
        if t_max > 0.0:
            x_pad = max(0.5, 0.01 * t_max)
            ax_theta.set_xlim(0.0, t_max + x_pad)
        pos_series = [e_baseline]
        th_series = [eth_baseline]
        if e_robak.size > 0:
            pos_series.append(e_robak[: min(t_robak.shape[0], e_robak.shape[0])])
            th_series.append(eth_robak[: min(t_robak.shape[0], eth_robak.shape[0])])
        if e_rywak.size > 0:
            pos_series.append(e_rywak[: min(t_rywak.shape[0], e_rywak.shape[0])])
            th_series.append(eth_rywak[: min(t_rywak.shape[0], eth_rywak.shape[0])])
        if include_ai and e_ai.size > 0:
            pos_series.append(e_ai[: min(t_ai.shape[0], e_ai.shape[0])])
            th_series.append(eth_ai[: min(t_ai.shape[0], eth_ai.shape[0])])
        pos_all = np.concatenate([np.asarray(s, dtype=np.float32).reshape(-1) for s in pos_series if np.asarray(s).size > 0])
        th_all = np.concatenate([np.asarray(s, dtype=np.float32).reshape(-1) for s in th_series if np.asarray(s).size > 0])
        if pos_all.size > 0:
            pos_min = float(np.min(pos_all))
            pos_max = float(np.max(pos_all))
            pos_pad = max(0.05, 0.07 * max(pos_max - pos_min, 1e-3))
            ax_pos.set_ylim(pos_min - pos_pad, pos_max + pos_pad)
        if th_all.size > 0:
            th_min = float(np.min(th_all))
            th_max = float(np.max(th_all))
            th_pad = max(0.03, 0.07 * max(th_max - th_min, 1e-3))
            ax_theta.set_ylim(th_min - th_pad, th_max + th_pad)
        ax_pos.set_title("Błąd pozycji")
        ax_pos.set_ylabel("error [m]")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="best")
        ax_theta.set_title("Błąd orientacji")
        ax_theta.set_xlabel("t [s]")
        ax_theta.set_ylabel("|error| [rad]")
        ax_theta.grid(True, alpha=0.3)
        ax_theta.legend(loc="best")
        fig.tight_layout()
        fig.savefig(err_path, dpi=150)
        plt.close(fig)
        artifacts["errors_png"] = err_path

    # Map layers plot without AI
    maps_path = dataset_dir / "eval_maps.png"
    if map_layers_npz.exists():
        with np.load(map_layers_npz) as m:
            key_order = ["ref", "baseline", "robak", "rywak"] + (["ai"] if include_ai else [])
            keys = [k for k in key_order if k in m.files]
            rotate_180 = bool(int(np.asarray(m["rotate_180"]).reshape(-1)[0])) if "rotate_180" in m.files else False
            if keys:
                n = len(keys)
                ncols = min(2, n)
                nrows = int(np.ceil(n / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.4 * nrows), squeeze=False)
                configure_figure(fig)
                axes_flat = axes.ravel()
                ref_occ = np.asarray(m["ref"], dtype=np.float32) if "ref" in m.files else None
                if ref_occ is not None and rotate_180:
                    ref_occ = np.rot90(ref_occ, 2)
                for i, key in enumerate(keys):
                    ax = axes_flat[i]
                    configure_axes(ax)
                    disp = np.asarray(m[key], dtype=np.float32)
                    if rotate_180:
                        disp = np.rot90(disp, 2)
                    # Keep IoU values from raw maps, but render as match-overlay for readability.
                    if key == "ref" or ref_occ is None or ref_occ.shape != disp.shape:
                        ax.imshow(disp, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                    else:
                        overlay = render_map_match_overlay(ref_occ, disp)
                        if overlay.size == 0:
                            ax.imshow(disp, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                        else:
                            ax.imshow(overlay, origin="lower", interpolation="nearest")
                    metric_key = f"iou_map_{key}" if key != "baseline" else "iou_map_baseline"
                    iou = None
                    if isinstance(results, dict):
                        metrics = results.get("metrics", {})
                        if isinstance(metrics, dict):
                            iou = metrics.get(metric_key)
                    title = key
                    if isinstance(iou, (int, float)):
                        title = f"{key} (IoU={float(iou):.3f})"
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                for j in range(n, len(axes_flat)):
                    axes_flat[j].axis("off")
                fig.suptitle("Porównanie map", fontsize=12, color="#f8fafc")
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
                fig.savefig(maps_path, dpi=160, facecolor=fig.get_facecolor())
                plt.close(fig)
                artifacts["maps_png"] = maps_path
    return artifacts


def render_training_curve(
    history_path: Path,
    output_path: Path,
    *,
    title: str,
) -> tuple[Path, dict[str, Any]] | None:
    history = load_history(history_path)
    epochs = history.get("epochs", [])
    if not isinstance(epochs, list) or not epochs:
        return None

    epoch_idx = np.asarray([int(item.get("epoch", idx + 1)) for idx, item in enumerate(epochs)], dtype=np.int32)
    train_loss = np.asarray([float(item.get("train_loss", 0.0)) for item in epochs], dtype=np.float32)
    val_loss = np.asarray([float(item.get("val_loss", 0.0)) for item in epochs], dtype=np.float32)
    best_idx = int(np.argmin(val_loss))
    best_epoch = int(epoch_idx[best_idx])
    best_val = float(val_loss[best_idx])

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    configure_figure(fig)
    configure_axes(ax)
    ax.plot(epoch_idx, train_loss, color="#38bdf8", linewidth=2.0, label="Błąd uczenia")
    ax.plot(epoch_idx, val_loss, color="#f97316", linewidth=2.0, label="Błąd walidacji")
    ax.axvline(best_epoch, color="#a3e635", linestyle="--", linewidth=1.4, label=f"Best epoch = {best_epoch}")
    ax.set_title(title)
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Loss")
    legend = ax.legend(loc="upper right")
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#334155")
    for text in legend.get_texts():
        text.set_color("#e5eefc")
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Epoki: {len(epoch_idx)}",
                f"Best val_loss: {best_val:.6f}",
                f"Final train_loss: {float(train_loss[-1]):.6f}",
                f"Final val_loss: {float(val_loss[-1]):.6f}",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#e5eefc",
        fontsize=9,
        bbox={"facecolor": "#0f172a", "edgecolor": "#334155", "boxstyle": "round,pad=0.4", "alpha": 0.86},
    )
    save_figure(fig, output_path)
    return output_path, {
        "history_path": str(history_path.resolve()),
        "epoch_count": int(len(epoch_idx)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
    }


def generate_training_curves_report(dataset_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    specs = [
        ("ai", dataset_dir / "train_history.json", dataset_dir / "train_curve_ai.png", "AI: błąd uczenia i walidacji"),
        ("robak", dataset_dir / "train_history_robak.json", dataset_dir / "train_curve_robak.png", "Robak: błąd uczenia i walidacji"),
        ("rywak", dataset_dir / "train_history_rywak.json", dataset_dir / "train_curve_rywak.png", "Rywak: błąd uczenia i walidacji"),
    ]

    artifact_paths: dict[str, Path] = {}
    summary: dict[str, Any] = {}

    for key, history_path, output_path, title in specs:
        if not history_path.exists():
            continue
        rendered = render_training_curve(history_path, output_path, title=title)
        if rendered is None:
            continue
        plot_path, model_summary = rendered
        artifact_paths[key] = plot_path
        summary[key] = model_summary
        print(f"[TRAIN] Zapisano krzywą: {plot_path}")

    if not summary:
        return {}, {}

    return artifact_paths, summary


def generate_report(dataset_dir: Path) -> dict[str, Path]:
    artifact_updates: dict[str, Path] = {}
    combined_summary: dict[str, Any] = {}
    include_ai = resolve_include_ai_in_artifacts(dataset_dir)

    baseline_artifacts, baseline_summary = generate_baseline_report(dataset_dir)
    if baseline_artifacts:
        legacy_path = dataset_dir / LEGACY_NAME
        if legacy_path.exists() and baseline_artifacts["legacy"].resolve() != legacy_path.resolve():
            try:
                legacy_path.unlink()
            except Exception:
                pass
        artifact_updates.update(
            {
                "dataset_inspection_overview_png": baseline_artifacts["overview"],
                "dataset_inspection_scans_png": baseline_artifacts["scans"],
                "dataset_target_components_png": baseline_artifacts["target_components"],
                "dataset_analysis_png": baseline_artifacts["legacy"],
            }
        )
    if baseline_summary:
        combined_summary.update(baseline_summary)

    robak_artifacts, robak_summary = generate_robak_coverage_report(dataset_dir)
    if robak_summary:
        combined_summary["robak_coverage"] = robak_summary
        write_json(robak_artifacts["summary"], robak_summary)
        artifact_updates.update(
            {
                "dataset_robak_coverage_summary_json": robak_artifacts["summary"],
                "dataset_robak_coverage_distance_png": robak_artifacts["distance"],
                "dataset_robak_coverage_rotation_png": robak_artifacts["rotation"],
                "dataset_robak_target_components_png": robak_artifacts["components"],
            }
        )
        print(f"[ROBAK] Zapisano podsumowanie: {robak_artifacts['summary']}")

    rywak_artifacts, rywak_summary = generate_rywak_coverage_report(dataset_dir)
    if rywak_summary:
        combined_summary["rywak_coverage"] = rywak_summary
        write_json(rywak_artifacts["summary"], rywak_summary)
        artifact_updates.update(
            {
                "dataset_rywak_coverage_summary_json": rywak_artifacts["summary"],
                "dataset_rywak_coverage_linear_velocity_png": rywak_artifacts["linear_velocity"],
                "dataset_rywak_coverage_angular_velocity_png": rywak_artifacts["angular_velocity"],
                "dataset_rywak_target_signed_velocity_png": rywak_artifacts["signed_velocity"],
            }
        )
        print(f"[RYWAK] Zapisano podsumowanie: {rywak_artifacts['summary']}")

    training_artifacts, training_summary = generate_training_curves_report(dataset_dir)
    if training_summary:
        combined_summary["training_curves"] = training_summary
        training_summary_path = dataset_dir / TRAINING_SUMMARY_NAME
        write_json(training_summary_path, training_summary)
        artifact_updates["training_inspection_summary_json"] = training_summary_path
        for key, path in training_artifacts.items():
            artifact_updates[f"training_curve_{key}_png"] = path
        print(f"[TRAIN] Zapisano podsumowanie: {training_summary_path}")

    trajectory_artifacts, trajectory_summary = generate_trajectory_speed_report(dataset_dir, include_ai=include_ai)
    if trajectory_artifacts:
        combined_summary.update(trajectory_summary)
        artifact_updates["trajectory_speed_png"] = trajectory_artifacts["trajectory_speed"]
        print(f"[EVAL] Zapisano trajektorie z heatmapą prędkości: {trajectory_artifacts['trajectory_speed']}")

    clean_eval_artifacts = generate_clean_eval_plots(dataset_dir, include_ai=include_ai)
    if clean_eval_artifacts:
        artifact_updates.update(clean_eval_artifacts)
        print("[EVAL] Zapisano czytelne wykresy eval: trajectory/errors/maps")

    if not combined_summary:
        raise FileNotFoundError(
            f"Nie znaleziono obsługiwanych artefaktów datasetu ani historii treningu w: {dataset_dir}"
        )

    summary_path = dataset_dir / SUMMARY_NAME
    experiment_summary_path = dataset_dir / EXPERIMENT_SUMMARY_NAME
    write_json(summary_path, combined_summary)
    write_json(experiment_summary_path, combined_summary)
    artifact_updates["dataset_inspection_summary_json"] = summary_path
    artifact_updates["experiment_inspection_summary_json"] = experiment_summary_path
    update_results_artifacts(dataset_dir, artifact_updates)

    print(f"[REPORT] Zapisano podsumowanie zbiorcze: {summary_path}")
    print(f"[REPORT] Zapisano podsumowanie eksperymentu: {experiment_summary_path}")
    return {key: path for key, path in artifact_updates.items() if path.exists()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generuje raport inspekcji datasetu dla eksperymentu.")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(REPO_ROOT / "out"),
        help="Ścieżka do folderu eksperymentu, np. out/exp_20260312_172054.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Nie istnieje folder datasetu: {dataset_dir}")
    generate_report(dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
