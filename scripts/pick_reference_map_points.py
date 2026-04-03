#!/usr/bin/env python3
"""
Zbieranie współrzędnych ścieżki na mapie referencyjnej (YAML + PGM) — bez Gazebo i bez SLAM.

Współrzędne są w tym samym układzie co wizualizacja w scripts/plot_planned_path.py
(oraz jak mapa w occupancy_grid_plan / planned_path_driver przy map_flip_y zgodnym z planowaniem).

Użycie:
  cd ~/SLAM_AI
  python3 scripts/pick_reference_map_points.py \\
    --reference-map ai_slam_ws/src/ai_slam_eval/maps/reference_map_hospital.yaml

Sterowanie w oknie matplotlib:
  LPM — dodaj punkt (wypisze się w terminalu)
  PPM — cofnij ostatni punkt
  środkowy przycisk myszy — zakończ wybór i wypisz blok YAML polyline

Wymaga wyświetlacza (WSLg / X11 / natywny Linux). Bez DISPLAY uruchomienie się nie powiedzie.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_BRINGUP_SRC = _REPO / "ai_slam_ws" / "src" / "ai_slam_bringup"
if str(_BRINGUP_SRC) not in sys.path:
    sys.path.insert(0, str(_BRINGUP_SRC))

from ai_slam_bringup.occupancy_grid_plan import load_reference_map_layers  # noqa: E402


def _imshow_pgm(ax, pgm: np.ndarray, meta: dict) -> None:
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


def main() -> int:
    p = argparse.ArgumentParser(
        description="Klikaj punkty na mapie referencyjnej; na końcu środkowy przycisk myszy — YAML polyline."
    )
    p.add_argument(
        "--reference-map",
        type=Path,
        required=True,
        help="Ścieżka do reference_map_*.yaml (PGM obok)",
    )
    args = p.parse_args()
    ref = args.reference_map.resolve()
    if not ref.is_file():
        print(f"Brak pliku: {ref}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Zainstaluj matplotlib (np. pip install matplotlib).", file=sys.stderr)
        return 1

    pgm, _blocked, meta = load_reference_map_layers(str(ref))

    fig, ax = plt.subplots(figsize=(12, 10))
    _imshow_pgm(ax, pgm, meta)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"{ref.name}  |  LPM=punkt, PPM=cofnij, środek myszy=koniec + YAML\n"
        f"(układ jak plot_planned_path / mapa ref.)"
    )
    ax.grid(True, alpha=0.2, color="cyan", linewidth=0.35)

    scatter = ax.scatter([], [], s=80, c="red", zorder=5, label="kotwice")
    points: list[tuple[float, float]] = []
    yaml_done = False

    def redraw():
        if points:
            scatter.set_offsets(np.array(points))
        else:
            scatter.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()

    def finish_yaml():
        nonlocal yaml_done
        if yaml_done:
            return
        yaml_done = True
        _print_yaml(points, ref)

    def on_click(event):
        nonlocal points
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.button == 1:
            points.append((x, y))
            print(f"  [{len(points)}] x={x:.4f}, y={y:.4f}")
            redraw()
        elif event.button == 3:
            if points:
                removed = points.pop()
                print(f"  (cofnięto) x={removed[0]:.4f}, y={removed[1]:.4f}")
                redraw()
        elif event.button == 2:
            fig.canvas.mpl_disconnect(cid)
            finish_yaml()
            plt.close(fig)

    def on_close(_event):
        finish_yaml()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", on_close)
    print(
        "\nOkno mapy: LPM = punkt | PPM = cofnij | środek myszy = koniec i YAML w terminalu\n",
        flush=True,
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        print(f"Nie udało się wyświetlić okna (DISPLAY / backend): {exc}", file=sys.stderr)
        return 1
    return 0


def _print_yaml(points: list[tuple[float, float]], ref: Path) -> None:
    if len(points) < 2:
        print("\nZa mało punktów (min. 2). Dodaj punkty LPM, zakończ środkiem myszy.", flush=True)
        return
    print("\n--- Skopiuj do YAML (polyline) ---\n", flush=True)
    print("polyline:")
    for x, y in points:
        print(f"  - {{x: {x:.6g}, y: {y:.6g}}}")
    print(
        "\nNastępnie np.:\n"
        f"  python3 scripts/planned_path_from_polyline.py \\\n"
        f"    --input twoj_plik.yaml \\\n"
        f"    --reference-map {ref} \\\n"
        f"    --out ai_slam_ws/src/ai_slam_bringup/config/planned_paths/twoja_sciezka.yaml\n",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
