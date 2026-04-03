#!/usr/bin/env python3
"""
Wizualizacja ścieżki z planned_paths/*.yaml: mapa referencyjna (PGM) + polilinia bez kolizji (A*)
+ zagęszczona ścieżka pure pursuit (jak w planned_path_driver).

Przykład:
  cd ~/SLAM_AI
  python3 scripts/plot_planned_path.py \\
    --spec ai_slam_ws/src/ai_slam_bringup/config/planned_paths/office_example.yaml \\
    --reference-map ai_slam_ws/src/ai_slam_eval/maps/reference_map_office.yaml \\
    --out out/planned_path_preview.png

RViz (przy uruchomionym planned_path_driver): dodaj Marker, topic /planned_path_reference
i ewentualnie /planned_path_dense, Fixed Frame = world (lub jak w reference_path_marker_frame).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_BRINGUP_SRC = _REPO / "ai_slam_ws" / "src" / "ai_slam_bringup"
if str(_BRINGUP_SRC) not in sys.path:
    sys.path.insert(0, str(_BRINGUP_SRC))

import yaml

from ai_slam_bringup.occupancy_grid_plan import (  # type: ignore  # noqa: E402
    densify_polyline,
    load_reference_map_layers,
    plan_polyline_through_anchors,
)


def _chain_densify_straight(anchors: list[tuple[float, float]], step_m: float) -> list[tuple[float, float]]:
    if len(anchors) < 2:
        return list(anchors)
    out: list[tuple[float, float]] = [anchors[0]]
    for i in range(len(anchors) - 1):
        seg = densify_polyline([anchors[i], anchors[i + 1]], step_m)
        out.extend(seg[1:])
    return out


def _imshow_pgm(ax, pgm: np.ndarray, meta: dict) -> None:
    """Pokrycie z world_to_cell / cell_to_world (flip_y jak w planowaniu)."""
    ox, oy, _ = meta["origin"]
    res = float(meta["resolution"])
    h, w = int(meta["height"]), int(meta["width"])
    img = np.flipud(pgm)
    extent = (ox, ox + w * res, oy, oy + h * res)
    ax.imshow(
        img,
        cmap="gray",
        extent=extent,
        origin="lower",
        interpolation="nearest",
        alpha=1.0,
        zorder=0,
        vmin=0,
        vmax=255,
    )


def _path_heading_deltas(pts: list[tuple[float, float]]) -> tuple[float, list[float]]:
    """Długość łamanej + |Δ nagłówka| między kolejnymi odcinkami (proxy krzywizny ścieżki)."""
    if len(pts) < 2:
        return 0.0, []
    total = 0.0
    headings: list[float] = []
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        ds = math.hypot(x1 - x0, y1 - y0)
        total += ds
        if ds > 1e-9:
            headings.append(math.atan2(y1 - y0, x1 - x0))
    abs_dh: list[float] = []
    for i in range(len(headings) - 1):
        d = headings[i + 1] - headings[i]
        d = math.atan2(math.sin(d), math.cos(d))
        abs_dh.append(abs(d))
    return total, abs_dh


def _print_path_stats(label: str, pts: list[tuple[float, float]]) -> None:
    length, dhs = _path_heading_deltas(pts)
    if not dhs:
        print(f"  [{label}] długość ≈ {length:.2f} m, brak segmentów pod kąt (prawie prosta).")
        return
    arr = np.array(dhs, dtype=np.float64)
    print(
        f"  [{label}] długość ≈ {length:.2f} m, |Δθ| między odcinkami [rad]: "
        f"mean={arr.mean():.3f}, p50={np.percentile(arr, 50):.3f}, p90={np.percentile(arr, 90):.3f}, max={arr.max():.3f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Podgląd planned path: mapa + polilinie (PNG)")
    p.add_argument("--spec", type=Path, required=True, help="YAML spec (anchors, dense_step_m, use_astar, …)")
    p.add_argument(
        "--reference-map",
        type=Path,
        default=None,
        help="Mapa referencyjna YAML (PGM w tle; wymagana przy use_astar)",
    )
    p.add_argument("--out", type=Path, default=Path("out/planned_path_preview.png"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--no-astar",
        action="store_true",
        help="Ignoruj use_astar w YAML — tylko proste między kotwicami.",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Nie drukuj przybliżonych statystyk ścieżki (krzywizna).",
    )
    args = p.parse_args()

    spec_path = args.spec.resolve()
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    anchors_raw = spec.get("anchors") or spec.get("waypoints") or []
    anchors: list[tuple[float, float]] = []
    for a in anchors_raw:
        if isinstance(a, dict):
            anchors.append((float(a["x"]), float(a["y"])))
    if len(anchors) < 2:
        print("Potrzeba co najmniej 2 kotwic.", file=sys.stderr)
        return 1

    dense_step = float(spec.get("dense_step_m", 0.2))
    use_astar = bool(spec.get("use_astar", False))
    map_flip_y = bool(spec.get("map_flip_y", True))
    inflate_m = float(spec.get("inflate_robot_m", 0.35))

    ref_path = args.reference_map.resolve() if args.reference_map else None
    pgm: np.ndarray | None = None
    blocked_map: np.ndarray | None = None
    meta: dict | None = None
    if ref_path is not None and ref_path.is_file():
        pgm, blocked_map, meta = load_reference_map_layers(str(ref_path))

    collision_poly: list[tuple[float, float]] = []
    path_xy: list[tuple[float, float]]
    astar_fallback = False

    if use_astar and not args.no_astar:
        if ref_path is None or not ref_path.is_file():
            print("use_astar=true wymaga istniejącego --reference-map", file=sys.stderr)
            return 1
        assert meta is not None and blocked_map is not None
        blocked = blocked_map
        res = float(meta["resolution"])
        inflate_cells = max(1, int(math.ceil(inflate_m / res)))
        try:
            poly = plan_polyline_through_anchors(
                anchors,
                blocked,
                meta,
                flip_y=map_flip_y,
                inflate_cells=inflate_cells,
            )
            collision_poly = list(poly)
            path_xy = densify_polyline(poly, dense_step)
        except ValueError as e:
            print(
                f"Ostrzeżenie: A* nie zadziałał ({e}). Kotwice + proste odcinki (nie „bez kolizji” na mapie).",
                file=sys.stderr,
            )
            astar_fallback = True
            collision_poly = list(anchors)
            path_xy = _chain_densify_straight(anchors, dense_step)
    else:
        path_xy = _chain_densify_straight(anchors, dense_step)
        collision_poly = list(anchors)

    xs = np.array([p[0] for p in path_xy], dtype=np.float64)
    ys = np.array([p[1] for p in path_xy], dtype=np.float64)
    axs = np.array([a[0] for a in anchors], dtype=np.float64)
    ays = np.array([a[1] for a in anchors], dtype=np.float64)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Zainstaluj matplotlib (np. pip install matplotlib).", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(11, 10))

    if pgm is not None and meta is not None:
        _imshow_pgm(ax, pgm, meta)

    cpx = np.array([p[0] for p in collision_poly], dtype=np.float64)
    cpy = np.array([p[1] for p in collision_poly], dtype=np.float64)
    if len(collision_poly) >= 2:
        lbl_cf = "polilinia A* (środek korytarza, bez kolizji z mapą)" if (use_astar and not args.no_astar and not astar_fallback) else "łamana kotwic (bez A* / fallback)"
        ax.plot(
            cpx,
            cpy,
            "--",
            color="lime",
            linewidth=2.4,
            alpha=0.95,
            zorder=3,
            label=lbl_cf,
        )

    ax.plot(xs, ys, "-", color="tab:blue", linewidth=1.35, alpha=0.9, zorder=4, label="ścieżka zagęszczona (pure pursuit)")
    ax.plot(axs, ays, "o", color="tab:red", markersize=6, zorder=5, label="kotwice")
    ax.plot(xs[0], ys[0], "s", color="tab:green", markersize=8, zorder=6, label="start")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, zorder=2, color="white", linewidth=0.4)
    leg = ax.legend(loc="upper right", framealpha=0.92)
    for text in leg.get_texts():
        text.set_color("black")
    ax.set_xlabel("x [m] (jak /ground_truth_pose)")
    ax.set_ylabel("y [m]")
    title = (
        f"{spec_path.name}  |  punkty zagęszczone={len(path_xy)}  |  astar={use_astar and not args.no_astar}"
    )
    if astar_fallback:
        title += "  (fallback)"
    ax.set_title(title)
    if pgm is None:
        ax.set_facecolor("#f0f0f0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Zapisano: {args.out.resolve()}  ({len(path_xy)} punktów zagęszczonych)")
    if not args.no_stats:
        print("Przybliżenie geometryczne (nie tożsame z rozkładem v/ω w datasetcie):")
        if len(collision_poly) >= 2:
            _print_path_stats("polilinia referencyjna", collision_poly)
        _print_path_stats("ścieżka zagęszczona", path_xy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
