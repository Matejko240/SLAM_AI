#!/usr/bin/env python3
"""
Z ręcznej polilinii (lista punktów w świecie) generuje plik planned_paths/*.yaml z kotwicami.

Dla każdego odcinka między kolejnymi punktami: rzut na najbliższą wolną komórkę (po inflacji),
potem A*; gdy A* się nie uda — wstawianie punktu pośredniego (półśrodek) i ponowienie.

Wejście: YAML z kluczem ``polyline`` (lista {x, y}). Opcjonalnie w tym samym pliku:
``inflate_robot_m``, ``dense_step_m``, ``map_flip_y``, ``use_astar``, ``strip_grid_loops`` (domyślnie true).

  polyline:
    - {x: 0.0, y: -25.0}
    - {x: -5.0, y: -22.0}
  # reference_map w pliku tylko jeśli nie podasz --reference-map

Przykład:
  python3 scripts/planned_path_from_polyline.py \\
    --reference-map ai_slam_ws/src/ai_slam_eval/maps/reference_map_hospital.yaml \\
    --input my_route.yaml \\
    --out ai_slam_ws/src/ai_slam_bringup/config/planned_paths/hospital_custom.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ai_slam_ws" / "src" / "ai_slam_bringup"))

from ai_slam_bringup.occupancy_grid_plan import (  # noqa: E402
    astar,
    cell_to_world_center,
    inflate_obstacles,
    load_reference_map_layers,
    plan_polyline_through_anchors,
    remove_grid_loops,
    world_to_cell,
)


def nearest_walkable(
    tx: float,
    ty: float,
    walk: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
) -> tuple[float, float]:
    h, w = walk.shape
    ox, oy, _ = meta["origin"]
    rs, cs = np.where(walk)
    if rs.size == 0:
        raise RuntimeError("Brak wolnych komórek na mapie.")
    wx = ox + (cs.astype(np.float64) + 0.5) * float(meta["resolution"])
    iy = (h - 1 - rs).astype(np.float64) if flip_y else rs.astype(np.float64)
    wy = oy + (iy + 0.5) * float(meta["resolution"])
    k = int(np.argmin((wx - tx) ** 2 + (wy - ty) ** 2))
    return float(wx[k]), float(wy[k])


def merge_cell_paths(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not a:
        return b
    if not b:
        return a
    # Najdłuższe nakładanie końcówki a z początkiem b (nie tylko jedna komórka).
    maxk = 0
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            maxk = k
            break
    return a + b[maxk:]


def chain_polyline_with_astar(
    world_points: list[tuple[float, float]],
    walk: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
    max_mid_insertions: int,
    strip_grid_loops: bool = True,
) -> list[tuple[int, int]]:
    h, w = walk.shape

    def cell_free(x: float, y: float) -> tuple[int, int] | None:
        c = world_to_cell(x, y, meta, flip_y=flip_y)
        if not c:
            return None
        r, col = c
        if not (0 <= r < h and 0 <= col < w and walk[r, col]):
            return None
        return c

    segments: list[list[tuple[int, int]]] = []
    working = [(float(x), float(y)) for x, y in world_points]
    insert_budget = max_mid_insertions

    i = 0
    while i < len(working) - 1:
        ax, ay = working[i]
        bx, by = working[i + 1]
        sa, sb = cell_free(ax, ay), cell_free(bx, by)
        if sa is None:
            nx, ny = nearest_walkable(ax, ay, walk, meta, flip_y=flip_y)
            sa = cell_free(nx, ny)
        if sb is None:
            nx, ny = nearest_walkable(bx, by, walk, meta, flip_y=flip_y)
            sb = cell_free(nx, ny)
        if sa is None or sb is None:
            raise ValueError(f"Nie można znaleźć wolnej komórki przy ({ax},{ay}) lub ({bx},{by}).")

        seg = astar(walk, sa, sb)
        if seg is None:
            if insert_budget <= 0:
                raise ValueError(f"A* nie znalazł ścieżki ({ax},{ay}) -> ({bx},{by}); dodaj punkt pośredni ręcznie.")
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            nx, ny = nearest_walkable(mx, my, walk, meta, flip_y=flip_y)
            working.insert(i + 1, (nx, ny))
            insert_budget -= 1
            continue
        segments.append(seg)
        i += 1

    out: list[tuple[int, int]] = []
    for seg in segments:
        out = merge_cell_paths(out, seg)
    if strip_grid_loops:
        out = remove_grid_loops(out)
    return out


def subsample_world(
    merged: list[tuple[float, float]], step_m: float
) -> list[tuple[float, float]]:
    if not merged:
        return []
    anchors = [merged[0]]
    acc = 0.0
    for i in range(len(merged) - 1):
        d = math.hypot(merged[i + 1][0] - merged[i][0], merged[i + 1][1] - merged[i][1])
        acc += d
        if acc >= step_m:
            anchors.append(merged[i + 1])
            acc = 0.0
    if math.hypot(anchors[-1][0] - merged[-1][0], anchors[-1][1] - merged[-1][1]) > 0.15:
        anchors.append(merged[-1])
    return anchors


def main() -> int:
    ap = argparse.ArgumentParser(description="Ręczna polilinia → planned_paths YAML z kotwicami")
    ap.add_argument("--input", type=Path, required=True, help="YAML z polyline: [{x,y}, ...]")
    ap.add_argument("--reference-map", type=Path, default=None, help="YAML mapy (PGM obok)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inflate-m", type=float, default=None)
    ap.add_argument("--subsample-m", type=float, default=None)
    ap.add_argument("--map-flip-y", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
    ap.add_argument("--max-mid-insertions", type=int, default=80)
    ap.add_argument(
        "--no-strip-grid-loops",
        action="store_true",
        help="Nie usuwaj pętli na siatce (możliwe powtórzenia komórek i krzyżowanie trasy)",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    raw_pl = spec.get("polyline") or spec.get("waypoints") or spec.get("anchors_hint")
    if not raw_pl or len(raw_pl) < 2:
        print("Plik musi zawierać polyline: z co najmniej 2 punktami.", file=sys.stderr)
        return 1

    pts: list[tuple[float, float]] = []
    for a in raw_pl:
        if isinstance(a, dict):
            pts.append((float(a["x"]), float(a["y"])))
        elif isinstance(a, (list, tuple)) and len(a) >= 2:
            pts.append((float(a[0]), float(a[1])))

    ref_map = args.reference_map
    if ref_map is None:
        rm = spec.get("reference_map") or spec.get("reference_map_yaml")
        if not rm:
            print("Podaj --reference-map lub pole reference_map w YAML.", file=sys.stderr)
            return 1
        ref_map = Path(str(rm))
    ref_map = ref_map.resolve()
    if not ref_map.is_file():
        print(f"Brak mapy: {ref_map}", file=sys.stderr)
        return 1

    inflate_m = float(args.inflate_m if args.inflate_m is not None else spec.get("inflate_robot_m", 0.35))
    dense_step = float(spec.get("dense_step_m", 0.22))
    flip_y = bool(spec.get("map_flip_y", True)) if args.map_flip_y is None else args.map_flip_y
    use_astar = bool(spec.get("use_astar", True))
    subsample_m = float(args.subsample_m if args.subsample_m is not None else spec.get("subsample_m", 1.1))
    strip_grid_loops = bool(spec.get("strip_grid_loops", True)) and not args.no_strip_grid_loops

    pgm, blocked, meta = load_reference_map_layers(str(ref_map))
    res = float(meta["resolution"])
    inflate_cells = max(1, int(math.ceil(inflate_m / res)))
    walk = ~inflate_obstacles(blocked, inflate_cells)

    try:
        cells = chain_polyline_with_astar(
            pts,
            walk,
            meta,
            flip_y=flip_y,
            max_mid_insertions=args.max_mid_insertions,
            strip_grid_loops=strip_grid_loops,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    merged = [cell_to_world_center(r, c, meta, flip_y=flip_y) for r, c in cells]
    anchors = subsample_world(merged, subsample_m)

    try:
        poly = plan_polyline_through_anchors(
            anchors, blocked, meta, flip_y=flip_y, inflate_cells=inflate_cells
        )
    except ValueError as e:
        print(f"Walidacja kotwic (plan_polyline_through_anchors): {e}", file=sys.stderr)
        return 1

    lp = sum(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )
    n_unique = len(set(cells))
    reuse = len(cells) - n_unique

    out_lines = [
        "# Wygenerowano: scripts/planned_path_from_polyline.py",
        f"# Mapa: {ref_map.name}  |  długość A* ~{lp:.0f} m  |  kotwice: {len(anchors)}  |  reuse komórek: {reuse}",
        "anchors:",
    ]
    for x, y in anchors:
        out_lines.append(f"  - {{x: {round(x, 3)}, y: {round(y, 3)}}}")
    out_lines.append(f"dense_step_m: {dense_step}")
    out_lines.append(f"use_astar: {str(use_astar).lower()}")
    out_lines.append(f"map_flip_y: {str(flip_y).lower()}")
    out_lines.append(f"inflate_robot_m: {inflate_m}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Zapisano {args.out}  ({len(anchors)} kotwic, ~{lp:.0f} m)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
