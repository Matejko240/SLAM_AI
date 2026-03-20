#!/usr/bin/env python3
"""Generuje czytelny raport inspekcji datasetu do artefaktow eksperymentu."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_DIR = REPO_ROOT / "ai_slam_ws"
REF_MAP_YAML = WORKSPACE_DIR / "src" / "ai_slam_eval" / "maps" / "reference_map.yaml"
WORLD_REFERENCE_MAPS = {
    "world_house.sdf": "reference_map.yaml",
    "world_office.sdf": "reference_map_office.yaml",
    "world_hospital.sdf": "reference_map_hospital.yaml",
}

SUMMARY_NAME = "dataset_inspection_summary.json"
OVERVIEW_NAME = "dataset_inspection_overview.png"
SCANS_NAME = "dataset_inspection_scans.png"
LEGACY_NAME = "dataset_analysis.png"
ROBAK_SUMMARY_NAME = "dataset_robak_coverage_summary.json"
ROBAK_DISTANCE_NAME = "dataset_robak_coverage_distance.png"
ROBAK_ROTATION_NAME = "dataset_robak_coverage_rotation.png"
RYWAK_SUMMARY_NAME = "dataset_rywak_coverage_summary.json"
RYWAK_LINEAR_NAME = "dataset_rywak_coverage_linear_velocity.png"
RYWAK_ANGULAR_NAME = "dataset_rywak_coverage_angular_velocity.png"
TRAINING_SUMMARY_NAME = "training_inspection_summary.json"
EXPERIMENT_SUMMARY_NAME = "experiment_inspection_summary.json"


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
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": percentile_or_zero(arr, 95.0),
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


def resolve_reference_map_yaml(dataset_dir: Path) -> Path:
    config_snapshot_path = dataset_dir / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        try:
            cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
            sim_cfg = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
            train_world = str(sim_cfg.get("train_world", "")).strip()
            mapped_ref_name = WORLD_REFERENCE_MAPS.get(train_world)
            if mapped_ref_name:
                mapped_ref_path = (WORKSPACE_DIR / "src" / "ai_slam_eval" / "maps" / mapped_ref_name).resolve()
                if mapped_ref_path.exists():
                    return mapped_ref_path
            candidate = (
                cfg.get("evaluation", {}).get("reference_map_yaml")
                if isinstance(cfg.get("evaluation"), dict)
                else None
            )
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
    ref_display = np.where(ref_grid == 0, 1.0, np.where(ref_grid == 254, 0.08, 0.32))

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
    shutil.copy2(output_path, dataset_dir / LEGACY_NAME)
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

    ref_map_yaml = resolve_reference_map_yaml(dataset_dir)
    ref_grid, ref_resolution, ref_origin = load_reference_map(ref_map_yaml)
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
    print(f"[DATASET] Zapisano widok ogólny: {overview_path}")
    print(f"[DATASET] Zapisano galerię skanów: {scans_path}")
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
        "legacy": dataset_dir / LEGACY_NAME,
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

    distance_summary = stats_1d(translation_cm)
    rotation_abs_deg = np.abs(rotation_deg).astype(np.float32)
    rotation_summary = stats_1d(rotation_deg)
    rotation_abs_summary = stats_1d(rotation_abs_deg)

    distance_target_50 = float(np.mean(translation_cm <= 50.0) * 100.0)
    distance_target_100 = float(np.mean(translation_cm <= 100.0) * 100.0)
    rotation_target_90 = float(np.mean(rotation_abs_deg <= 90.0) * 100.0)
    rotation_target_180 = float(np.mean(rotation_abs_deg <= 180.0) * 100.0)
    overflow_100cm = int(np.sum(translation_cm > 100.0))

    distance_path = render_histogram_coverage(
        dataset_dir / ROBAK_DISTANCE_NAME,
        np.clip(translation_cm, 0.0, 100.0),
        bins=np.linspace(0.0, 100.0, 41),
        title="Robak: pokrycie przesunięć między skanami",
        xlabel="Przesunięcie [cm]",
        color="#f97316",
        xlim=(0.0, 100.0),
        vertical_lines=[
            (50.0, "50 cm", "#a3e635"),
            (100.0, "100 cm", "#38bdf8"),
        ],
        annotations=[
            f"Próbki: {distance_summary['count']}",
            f"Średnia / mediana: {distance_summary['mean']:.1f} / {distance_summary['median']:.1f} cm",
            f"95 percentyl / max: {distance_summary['p95']:.1f} / {distance_summary['max']:.1f} cm",
            f"Pokrycie 0-50 cm: {distance_target_50:.1f}%",
            f"Pokrycie 0-100 cm: {distance_target_100:.1f}%",
            f"> 100 cm: {overflow_100cm}",
        ],
    )
    rotation_path = render_histogram_coverage(
        dataset_dir / ROBAK_ROTATION_NAME,
        rotation_deg,
        bins=np.linspace(-180.0, 180.0, 49),
        title="Robak: pokrycie rotacji między skanami",
        xlabel="Rotacja [deg]",
        color="#0ea5e9",
        xlim=(-180.0, 180.0),
        vertical_lines=[
            (-90.0, "-90 deg", "#a3e635"),
            (90.0, "+90 deg", "#a3e635"),
        ],
        annotations=[
            f"Próbki: {rotation_summary['count']}",
            f"Min / max: {rotation_summary['min']:.1f} / {rotation_summary['max']:.1f} deg",
            f"Średnia |rot| / p95 |rot|: {rotation_abs_summary['mean']:.1f} / {rotation_abs_summary['p95']:.1f} deg",
            f"Pokrycie |rot| <= 90 deg: {rotation_target_90:.1f}%",
            f"Pokrycie |rot| <= 180 deg: {rotation_target_180:.1f}%",
            f"Offsety: {normalize_json_value(meta.get('offsets', []))}",
        ],
    )

    summary = {
        "dataset_path": str(dataset_path.resolve()),
        "sample_count": int(labels.shape[0]),
        "translation_cm": distance_summary,
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
        "distance": distance_path,
        "rotation": rotation_path,
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

    linear_summary = stats_1d(linear_abs)
    angular_summary = stats_1d(angular_abs)
    signed_linear_summary = stats_1d(linear_velocity)
    signed_angular_summary = stats_1d(angular_velocity)

    linear_target_1 = float(np.mean(linear_abs <= 1.0) * 100.0)
    linear_target_2 = float(np.mean(linear_abs <= 2.0) * 100.0)
    angular_target_3 = float(np.mean(angular_abs <= 3.0) * 100.0)
    linear_over_2 = int(np.sum(linear_abs > 2.0))
    angular_over_3 = int(np.sum(angular_abs > 3.0))

    linear_path = render_histogram_coverage(
        dataset_dir / RYWAK_LINEAR_NAME,
        np.clip(linear_abs, 0.0, 2.0),
        bins=np.linspace(0.0, 2.0, 41),
        title="Rywak: pokrycie prędkości liniowej",
        xlabel="|v| [m/s]",
        color="#22c55e",
        xlim=(0.0, 2.0),
        vertical_lines=[
            (1.0, "1 m/s", "#f97316"),
            (2.0, "2 m/s", "#38bdf8"),
        ],
        annotations=[
            f"Próbki: {linear_summary['count']}",
            f"Średnia / mediana: {linear_summary['mean']:.3f} / {linear_summary['median']:.3f} m/s",
            f"95 percentyl / max: {linear_summary['p95']:.3f} / {linear_summary['max']:.3f} m/s",
            f"Pokrycie |v| <= 1 m/s: {linear_target_1:.1f}%",
            f"Pokrycie |v| <= 2 m/s: {linear_target_2:.1f}%",
            f"> 2 m/s: {linear_over_2}",
        ],
    )
    angular_path = render_histogram_coverage(
        dataset_dir / RYWAK_ANGULAR_NAME,
        np.clip(angular_abs, 0.0, 3.0),
        bins=np.linspace(0.0, 3.0, 37),
        title="Rywak: pokrycie prędkości kątowej",
        xlabel="|omega| [rad/s]",
        color="#a855f7",
        xlim=(0.0, 3.0),
        vertical_lines=[(3.0, "3 rad/s", "#38bdf8")],
        annotations=[
            f"Próbki: {angular_summary['count']}",
            f"Średnia / mediana: {angular_summary['mean']:.3f} / {angular_summary['median']:.3f} rad/s",
            f"95 percentyl / max: {angular_summary['p95']:.3f} / {angular_summary['max']:.3f} rad/s",
            f"Pokrycie |omega| <= 3 rad/s: {angular_target_3:.1f}%",
            f"> 3 rad/s: {angular_over_3}",
            f"Zakres signed omega: {signed_angular_summary['min']:.3f} .. {signed_angular_summary['max']:.3f}",
        ],
    )

    summary = {
        "dataset_path": str(dataset_path.resolve()),
        "sample_count": int(labels.shape[0]),
        "linear_velocity_abs_mps": linear_summary,
        "linear_velocity_signed_mps": signed_linear_summary,
        "angular_velocity_abs_radps": angular_summary,
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
    }, summary


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
        ("ai", dataset_dir / "train_history.json", dataset_dir / "training_curve_ai.png", "AI: błąd uczenia i walidacji"),
        ("robak", dataset_dir / "train_history_robak.json", dataset_dir / "training_curve_robak.png", "Robak: błąd uczenia i walidacji"),
        ("rywak", dataset_dir / "train_history_rywak.json", dataset_dir / "training_curve_rywak.png", "Rywak: błąd uczenia i walidacji"),
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

    baseline_artifacts, baseline_summary = generate_baseline_report(dataset_dir)
    if baseline_artifacts:
        artifact_updates.update(
            {
                "dataset_inspection_overview_png": baseline_artifacts["overview"],
                "dataset_inspection_scans_png": baseline_artifacts["scans"],
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
