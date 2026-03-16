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

SUMMARY_NAME = "dataset_inspection_summary.json"
OVERVIEW_NAME = "dataset_inspection_overview.png"
SCANS_NAME = "dataset_inspection_scans.png"
LEGACY_NAME = "dataset_analysis.png"


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
    odom: np.ndarray,
    max_range: float,
    max_scans: int = 1800,
) -> tuple[np.ndarray, np.ndarray, int]:
    sample_step = max(1, int(np.ceil(scans.shape[0] / float(max_scans))))
    points_x: list[np.ndarray] = []
    points_y: list[np.ndarray] = []
    sampled_scans = 0
    for index in range(0, scans.shape[0], sample_step):
        px, py = scan_to_points(scans[index], odom[index], max_range=max_range)
        if px.size == 0:
            continue
        points_x.append(px)
        points_y.append(py)
        sampled_scans += 1
    if not points_x:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), sampled_scans
    return np.concatenate(points_x), np.concatenate(points_y), sampled_scans


def compute_summary(scans: np.ndarray, odom: np.ndarray, corrections: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    valid_ranges = scans[np.isfinite(scans) & (scans > 0.05)]
    correction_xy = np.linalg.norm(corrections[:, :2], axis=1)
    theta_abs = np.abs(corrections[:, 2])
    deltas = np.diff(odom[:, :2], axis=0) if odom.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
    traj_len = float(np.linalg.norm(deltas, axis=1).sum()) if deltas.size else 0.0

    return {
        "sample_count": int(scans.shape[0]),
        "scan_beam_count": int(scans.shape[1]),
        "valid_return_ratio": float(np.mean(np.isfinite(scans) & (scans > 0.05))),
        "valid_return_count": int(np.sum(np.isfinite(scans) & (scans > 0.05))),
        "range_mean_m": float(np.mean(valid_ranges)) if valid_ranges.size else 0.0,
        "range_median_m": float(np.median(valid_ranges)) if valid_ranges.size else 0.0,
        "range_p95_m": float(np.percentile(valid_ranges, 95)) if valid_ranges.size else 0.0,
        "range_max_m": float(np.max(valid_ranges)) if valid_ranges.size else 0.0,
        "trajectory_length_m": traj_len,
        "trajectory_x_span_m": float(np.max(odom[:, 0]) - np.min(odom[:, 0])) if odom.size else 0.0,
        "trajectory_y_span_m": float(np.max(odom[:, 1]) - np.min(odom[:, 1])) if odom.size else 0.0,
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
    corrections: np.ndarray,
    summary: dict[str, Any],
    ref_grid: np.ndarray,
    ref_resolution: float,
    ref_origin: list[float],
) -> Path:
    valid_ranges = scans[np.isfinite(scans) & (scans > 0.05)]
    plot_max_range = max(5.0, float(np.percentile(valid_ranges, 99))) if valid_ranges.size else 5.0
    points_x, points_y, sampled_scans = build_sampled_map_points(scans, odom, max_range=plot_max_range)
    summary["sampled_map_scan_count"] = int(sampled_scans)
    summary["sampled_map_point_count"] = int(points_x.size)

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
    ax.plot(odom[:, 0], odom[:, 1], color="#38bdf8", linewidth=1.8, alpha=0.92)
    ax.scatter(odom[0, 0], odom[0, 1], s=68, c="#22c55e", zorder=5)
    ax.scatter(odom[-1, 0], odom[-1, 1], s=72, c="#f97316", marker="X", zorder=5)
    ax.set_title("Trajektoria i próbka mapy punktowej")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    ax = axes[0, 1]
    if points_x.size:
        hb = ax.hexbin(points_x, points_y, gridsize=85, mincnt=1, cmap="viridis", linewidths=0)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Gęstość punktów", color="#dbe7ff")
        cbar.ax.yaxis.set_tick_params(color="#dbe7ff")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#dbe7ff")
    ax.plot(odom[:, 0], odom[:, 1], color="#e2e8f0", linewidth=1.1, alpha=0.75)
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
        f"Długość trajektorii: {summary['trajectory_length_m']:.1f} m"
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


def generate_report(dataset_dir: Path) -> dict[str, Path]:
    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku datasetu: {dataset_path}")

    print(f"[DATASET] Wczytywanie: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    scans = np.asarray(data["X_scan"], dtype=np.float32)
    odom = np.asarray(data["X_odom"], dtype=np.float32)
    corrections = np.asarray(data["Y"], dtype=np.float32)
    meta = data["meta"].item() if "meta" in data else {}
    meta = meta if isinstance(meta, dict) else {}

    ref_grid, ref_resolution, ref_origin = load_reference_map(REF_MAP_YAML)
    summary = compute_summary(scans, odom, corrections, meta)
    summary.update(
        {
            "dataset_dir": str(dataset_dir.resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "reference_map_yaml": str(REF_MAP_YAML.resolve()),
        }
    )

    overview_path = render_overview(dataset_dir, scans, odom, corrections, summary, ref_grid, ref_resolution, ref_origin)
    scans_path = render_scan_gallery(dataset_dir, scans)
    summary_path = dataset_dir / SUMMARY_NAME
    summary = normalize_json_value(summary)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DATASET] Zapisano podsumowanie: {summary_path}")
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
        "summary": summary_path,
        "overview": overview_path,
        "scans": scans_path,
        "legacy": dataset_dir / LEGACY_NAME,
    }


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
