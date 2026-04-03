#!/usr/bin/env python3
"""
Generuje trasę office (kotwice YAML) od spawna (0.03, 2.27), bez powtarzania komórek siatki
(ścieżka prosta w sensie grafu — brak „cofania się” tym samym korytarzem).

Strategia:
  1) A* od startu do najdalszego węzła BFS (długa pojedyncza łamana, ~0 nakładania).
  2) Opcjonalnie: seria przedłużeń A* od bieżącego końca do dalekich celów, odrzucane gdy
     udział już odwiedzonych komórek (poza krótkim początkiem odcinka) przekracza próg.

Uruchomienie (office):
  python3 scripts/generate_office_long_tour_yaml.py > ai_slam_ws/src/ai_slam_bringup/config/planned_paths/office_example.yaml

Szpital (spawn z experiment_config):
  python3 scripts/generate_office_long_tour_yaml.py \\
    --map-yaml ai_slam_ws/src/ai_slam_eval/maps/reference_map_hospital.yaml \\
    --start-x 0 --start-y -25 \\
    > ai_slam_ws/src/ai_slam_bringup/config/planned_paths/hospital_example.yaml

Ręczna polilinia zamiast auto-trasy: scripts/planned_path_from_polyline.py
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ai_slam_ws" / "src" / "ai_slam_bringup"))

from ai_slam_bringup.occupancy_grid_plan import (  # noqa: E402
    astar,
    cell_to_world_center,
    inflate_obstacles,
    load_reference_map_layers,
    plan_polyline_through_anchors,
    world_to_cell,
)

MAP_YAML = _REPO / "ai_slam_ws" / "src" / "ai_slam_eval" / "maps" / "reference_map_office.yaml"
FLIP_Y = True


def bfs_dist_from(
    walk, h: int, w: int, seed: tuple[int, int]
) -> dict[tuple[int, int], int]:
    dist = {seed: 0}
    q = deque([seed])
    while q:
        r, c = q.popleft()
        d0 = dist[(r, c)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= h or nc >= w or not walk[nr, nc]:
                continue
            if dr and dc:
                if not walk[r + dr, c] or not walk[r, c + dc]:
                    continue
            if (nr, nc) in dist:
                continue
            dist[(nr, nc)] = d0 + 1
            q.append((nr, nc))
    return dist


def path_overlap_fraction(path: list[tuple[int, int]], used: set[tuple[int, int]], skip_prefix: int) -> float:
    if len(path) <= skip_prefix:
        return 0.0
    body = path[skip_prefix:]
    if not body:
        return 0.0
    hit = sum(1 for cell in body if cell in used)
    return hit / float(len(body))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inflate-m", type=float, default=0.35)
    ap.add_argument("--subsample-m", type=float, default=1.1)
    ap.add_argument(
        "--max-cell-reuse",
        type=float,
        default=0.08,
        help="Max. ułamek komórek NOWEGO odcinka (po --skip-cells) już odwiedzonych. 0 = tylko pierwsza noga A* (~40 m, zero reuse).",
    )
    ap.add_argument("--skip-cells", type=int, default=10, help="Początek odcinka ignorowany przy liczeniu reuse.")
    ap.add_argument("--max-extensions", type=int, default=30)
    ap.add_argument("--min-extension-len", type=int, default=25, help="Min. długość odcinka w komórkach.")
    ap.add_argument(
        "--candidates-per-ext",
        type=int,
        default=36,
        help="Ile celów A* testować na jedno przedłużenie (losowo z najdalszych nieodwiedzonych).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--map-yaml",
        type=Path,
        default=MAP_YAML,
        help="YAML mapy referencyjnej (katalog z PGM).",
    )
    ap.add_argument("--start-x", type=float, default=0.03, help="Punkt startowy trasy (świat).")
    ap.add_argument("--start-y", type=float, default=2.27)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    map_path = args.map_yaml.resolve()
    if not map_path.is_file():
        print(f"Brak pliku mapy: {map_path}", file=sys.stderr)
        return 1

    pgm, blocked, meta = load_reference_map_layers(str(map_path))
    res = float(meta["resolution"])
    inflate_cells = max(1, int(math.ceil(args.inflate_m / res)))
    walk = ~inflate_obstacles(blocked, inflate_cells)
    h, w = walk.shape

    def ok(x: float, y: float):
        c = world_to_cell(x, y, meta, flip_y=FLIP_Y)
        return c if c and walk[c[0], c[1]] else None

    start = ok(args.start_x, args.start_y)
    if not start:
        print(
            f"Start ({args.start_x}, {args.start_y}) nie jest wolny po inflacji — popraw --start-x/--start-y.",
            file=sys.stderr,
        )
        return 1

    d0 = bfs_dist_from(walk, h, w, start)
    far_a = max(d0, key=lambda k: d0[k])
    p = astar(walk, start, far_a)
    if not p:
        return 1

    cells: list[tuple[int, int]] = list(p)
    used: set[tuple[int, int]] = set(cells)

    for _ext in range(args.max_extensions):
        cur = cells[-1]
        dcur = bfs_dist_from(walk, h, w, cur)
        # Kandydaci: daleko od bieżącego końca (pomijamy już odwiedzone jako CELE — wymuszamy „nowe” miejsce)
        pool = [g for g in dcur if g not in used]
        pool.sort(key=lambda g: -dcur[g])
        pool = pool[: min(500, len(pool))]
        if len(pool) > args.candidates_per_ext:
            candidates = rng.sample(pool, args.candidates_per_ext)
        else:
            candidates = pool

        best_path: list[tuple[int, int]] | None = None
        best_score = -1.0

        for goal in candidates:
            seg = astar(walk, cur, goal)
            if seg is None or len(seg) < args.min_extension_len:
                continue
            ov = path_overlap_fraction(seg, used, args.skip_cells)
            if ov > args.max_cell_reuse + 1e-15:
                continue
            # Preferuj długie odcinki i niskie nakładanie
            new_cells = len(set(seg[args.skip_cells :]) - used)
            score = len(seg) * (1.0 - ov) + 0.5 * new_cells
            if score > best_score:
                best_score = score
                best_path = seg

        if best_path is None:
            break
        # doklej bez duplikatu pierwszego węzła
        if best_path[0] == cells[-1]:
            cells.extend(best_path[1:])
            for c in best_path[1:]:
                used.add(c)
        else:
            cells.extend(best_path)
            for c in best_path:
                used.add(c)

    merged = [cell_to_world_center(r, c, meta, flip_y=FLIP_Y) for r, c in cells]

    anchors = [merged[0]]
    acc = 0.0
    for i in range(len(merged) - 1):
        d = math.hypot(merged[i + 1][0] - merged[i][0], merged[i + 1][1] - merged[i][1])
        acc += d
        if acc >= args.subsample_m:
            anchors.append(merged[i + 1])
            acc = 0.0
    if math.hypot(anchors[-1][0] - merged[-1][0], anchors[-1][1] - merged[-1][1]) > 0.15:
        anchors.append(merged[-1])

    poly = plan_polyline_through_anchors(
        anchors, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
    )
    lm = sum(
        math.hypot(merged[i + 1][0] - merged[i][0], merged[i + 1][1] - merged[i][1])
        for i in range(len(merged) - 1)
    )
    lp = sum(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )

    # Statystyka powtórzeń komórek (0 = idealna ścieżka prosta w grafie)
    n_cells = len(cells)
    n_unique = len(set(cells))
    reuse = n_cells - n_unique
    reuse_pct = 100.0 * reuse / max(1, n_cells)

    print("# Trasa: pierwsza noga = A* do najdalszego BFS; dalsze odcinki tylko jeśli reuse komórek ≤ --max-cell-reuse.")
    if "hospital" in map_path.name.lower():
        print("# Mapa: szpital — dopasuj spawn w simulation.spawn_poses.world_hospital.sdf do --start-x/--start-y.")
    else:
        print("# Office: część wschodniego skrzydła może być niedostępna z spawnu — trasa omija te rejony.")
    print(
        "# Komórki: {} łącznie, {} unikalnych (powtórzenia: {}, {:.2f}%); długość ~{:.0f} m; kotwice: {}; A* ~{:.0f} m.".format(
            n_cells, n_unique, reuse, reuse_pct, lm, len(anchors), lp
        )
    )
    if reuse_pct > 12.0:
        print("# UWAGA: duży udział powtórzeń — rozważ --max-cell-reuse 0.0 (krótsza trasa) lub inną mapę/inflację.")
    print("# Acykliczna geometria: driver.planned_path.loop_path: false w experiment_config.yaml.")
    print("anchors:")
    for x, y in anchors:
        print(f"  - {{x: {round(x, 3)}, y: {round(y, 3)}}}")
    print("dense_step_m: 0.22")
    print("use_astar: true")
    print("map_flip_y: true")
    print(f"inflate_robot_m: {args.inflate_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
