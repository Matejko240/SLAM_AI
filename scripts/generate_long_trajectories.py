#!/usr/bin/env python3
"""
Generuje 4 długie planned_paths (YAML) pod dataset / ewaluację:

  1) office_trajectory_acyclic.yaml   — jak najdłuższa, mało powtórzeń komórek (BFS+A*+przedłużenia)
  2) office_trajectory_cyclic_2lap.yaml — ta sama geometria co acyclic; loop_path: true w planned_path_driver
     (po dojechaniu do końca ślad resetuje się na początek = kolejny pełny przejazd tą samą trasą).
  3) hospital_trajectory_acyclic.yaml
  4) hospital_trajectory_cyclic_2lap.yaml

Uruchom z katalogu głównego repo:
  python3 scripts/generate_long_trajectories.py

Podgląd PNG na mapach referencyjnych:
  python3 scripts/plot_long_trajectories.py

Wymaga aktualnych reference_map_*.yaml + PGM (scripts/generate_reference_map.py po zmianach świata).

Jeśli w datasetcie brakuje próbek w „koszykach” prędkości: najpierw kręć `driver.planned_path.linear_vel_max` /
`angular_vel_max` w experiment_config.yaml; ewentualnie dodaj osobne krótkie trasy (np. długi prosty odcinek
vs serpentyna) przez scripts/planned_path_from_polyline.py — bez konieczności duplikowania tych czterech plików.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

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

FLIP_Y = True
OUT_DIR = _REPO / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config" / "planned_paths"
MAP_OFFICE = _REPO / "ai_slam_ws" / "src" / "ai_slam_eval" / "maps" / "reference_map_office.yaml"
MAP_HOSPITAL = _REPO / "ai_slam_ws" / "src" / "ai_slam_eval" / "maps" / "reference_map_hospital.yaml"

# Office: twarde wymuszenie dużego nawrotu "U" w lewym górnym rogu mapy.
# Uwaga: "U" ma być widoczne (dwie różne gałęzie y + łącznik pionowy po lewej),
# a nie zawracanie po tej samej linii.
_OFFICE_UL_U_CANDIDATES: list[tuple[tuple[float, float], ...]] = [
    ((-22.45, 17.35), (-26.75, 17.35), (-26.75, 18.65), (-22.75, 18.65)),
    ((-22.35, 17.25), (-26.55, 17.25), (-26.55, 18.55), (-22.65, 18.55)),
    ((-22.55, 17.45), (-26.85, 17.45), (-26.85, 18.75), (-22.85, 18.75)),
]

# Przecięcia polilinii (środki komórek → świat); eps ~1 mm przy współrzędnych w metrach.
_GEOM_EPS = 1e-7
# Krótsze odcinki pomijamy w teście — inaczej siatka 8-sąsiedzka daje fałszywe „krzyże” przy ostrych zakrętach.
_MIN_SEG_LEN_CROSS_M = 0.28
# Para odcinków musi być oddalona w indeksie wzdłuż łańcucha (sąsiednie i prawie-sąsiednie i tak dzielą wierzchołek).
_MIN_VERTEX_GAP_SEGMENTS = 3


def _cells_world_xy(cells: list[tuple[int, int]], meta: dict) -> list[tuple[float, float]]:
    return [cell_to_world_center(r, c, meta, flip_y=FLIP_Y) for r, c in cells]


def _orient_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_closed_segment(
    a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]
) -> bool:
    if abs(_orient_2d(a, b, p)) > _GEOM_EPS:
        return False
    return (
        min(a[0], b[0]) - _GEOM_EPS <= p[0] <= max(a[0], b[0]) + _GEOM_EPS
        and min(a[1], b[1]) - _GEOM_EPS <= p[1] <= max(a[1], b[1]) + _GEOM_EPS
    )


def segments_intersect_open(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """True jeśli odcinki domknięte AB i CD się przecinają (w tym kolinearne nakładanie)."""
    o1 = _orient_2d(a, b, c)
    o2 = _orient_2d(a, b, d)
    o3 = _orient_2d(c, d, a)
    o4 = _orient_2d(c, d, b)

    if (o1 > _GEOM_EPS and o2 > _GEOM_EPS) or (o1 < -_GEOM_EPS and o2 < -_GEOM_EPS):
        return False
    if (o3 > _GEOM_EPS and o4 > _GEOM_EPS) or (o3 < -_GEOM_EPS and o4 < -_GEOM_EPS):
        return False

    if abs(o1) <= _GEOM_EPS and _point_on_closed_segment(a, b, c):
        return True
    if abs(o2) <= _GEOM_EPS and _point_on_closed_segment(a, b, d):
        return True
    if abs(o3) <= _GEOM_EPS and _point_on_closed_segment(c, d, a):
        return True
    if abs(o4) <= _GEOM_EPS and _point_on_closed_segment(c, d, b):
        return True
    return o1 * o2 < -(_GEOM_EPS**2) and o3 * o4 < -(_GEOM_EPS**2)


def _bbox_sep(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    m1x, m2x = (min(ax, bx), max(ax, bx))
    m1y, m2y = (min(ay, by), max(ay, by))
    n1x, n2x = (min(cx, dx), max(cx, dx))
    n1y, n2y = (min(cy, dy), max(cy, dy))
    pad = _GEOM_EPS
    return m2x < n1x - pad or n2x < m1x - pad or m2y < n1y - pad or n2y < m1y - pad


def extension_adds_polyline_cross(
    existing_xy: list[tuple[float, float]],
    seg_xy: list[tuple[float, float]],
) -> bool:
    """Czy dołączenie seg (od tego samego co koniec existing) tworzy przecięcie nie-sąsiednich odcinków."""
    if len(existing_xy) < 2 or len(seg_xy) < 2:
        return False
    if (
        math.hypot(existing_xy[-1][0] - seg_xy[0][0], existing_xy[-1][1] - seg_xy[0][1])
        > 0.02
    ):
        return True
    full = existing_xy + seg_xy[1:]
    ne = len(existing_xy)
    first_new_seg = ne - 1
    nseg = len(full) - 1
    for s in range(first_new_seg, nseg):
        a, b = full[s], full[s + 1]
        if math.hypot(b[0] - a[0], b[1] - a[1]) < _MIN_SEG_LEN_CROSS_M:
            continue
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        for t in range(nseg):
            if abs(s - t) < _MIN_VERTEX_GAP_SEGMENTS:
                continue
            c, d = full[t], full[t + 1]
            if math.hypot(d[0] - c[0], d[1] - c[1]) < _MIN_SEG_LEN_CROSS_M:
                continue
            if _bbox_sep(ax, ay, bx, by, c[0], c[1], d[0], d[1]):
                continue
            if segments_intersect_open(a, b, c, d):
                return True
    return False


def open_polyline_nonadjacent_segments_cross(
    pts: list[tuple[float, float]],
    *,
    min_seg_m: float = 0.11,
) -> bool:
    """Przecięcie odcinków polilinii otwartej (np. wynik plan_polyline_through_anchors)."""
    n = len(pts)
    if n < 4:
        return False
    nseg = n - 1
    for i in range(nseg):
        a, b = pts[i], pts[i + 1]
        if math.hypot(b[0] - a[0], b[1] - a[1]) < min_seg_m:
            continue
        for j in range(i + 2, nseg):
            c, d = pts[j], pts[j + 1]
            if math.hypot(d[0] - c[0], d[1] - c[1]) < min_seg_m:
                continue
            if _bbox_sep(a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]):
                continue
            if segments_intersect_open(a, b, c, d):
                return True
    return False


def bfs_dist_from(walk, h: int, w: int, seed: tuple[int, int]) -> dict[tuple[int, int], int]:
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


def _sample_free_cells(walk: np.ndarray, rng: random.Random, n: int) -> list[tuple[int, int]]:
    coords = np.argwhere(walk)
    if coords.size == 0:
        return []
    if len(coords) <= n:
        return [(int(row[0]), int(row[1])) for row in coords]
    pick = rng.sample(range(len(coords)), k=n)
    return [(int(coords[i][0]), int(coords[i][1])) for i in pick]


def _cell_quadrant(cell: tuple[int, int], r_med: float, c_med: float) -> int:
    r, c = cell
    return (0 if r < r_med else 2) + (0 if c < c_med else 1)


def _pick_extension_candidates(
    pool: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rng: random.Random,
    k: int,
    *,
    r_med: float | None,
    c_med: float | None,
) -> list[tuple[int, int]]:
    """Wybór celów A*: opcjonalnie równoważy ćwiartki mapy (mniej „klejenia” w jednym rogu)."""
    if not pool:
        return []
    if r_med is None or c_med is None or k <= 4:
        if len(pool) > k:
            return rng.sample(pool, k)
        return list(pool)

    cnt = [0, 0, 0, 0]
    for g in used:
        qi = _cell_quadrant(g, r_med, c_med)
        cnt[qi] += 1
    inv = [1.0 / (1.0 + float(c)) for c in cnt]
    s_inv = sum(inv)
    # Proporcje inv/s_inv; bez wymuszania min. 1 na ćwiartkę (dla małego k nie zablokuje redukcji).
    raw = [k * inv[q] / s_inv for q in range(4)]
    targets = [int(x) for x in raw]
    while sum(targets) < k:
        j = max(range(4), key=lambda q: inv[q] / (1.0 + float(targets[q])))
        targets[j] += 1
    while sum(targets) > k:
        j = max(range(4), key=lambda i: targets[i])
        if targets[j] <= 0:
            break
        targets[j] -= 1

    buckets: list[list[tuple[int, int]]] = [[], [], [], []]
    for g in pool:
        buckets[_cell_quadrant(g, r_med, c_med)].append(g)

    picked: list[tuple[int, int]] = []
    for q in range(4):
        need = targets[q]
        bq = buckets[q]
        if len(bq) < 2:
            bq = list(pool)
        if need >= len(bq):
            picked.extend(bq)
        else:
            picked.extend(rng.sample(bq, need))
    # unikalne, zachowaj do k
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for g in picked:
        if g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= k:
            break
    while len(out) < k and pool:
        g = rng.choice(pool)
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out[:k]


def build_acyclic_cells(
    walk,
    h: int,
    w: int,
    meta: dict,
    start: tuple[int, int],
    rng: random.Random,
    *,
    max_extensions: int,
    max_cell_reuse: float,
    skip_cells: int,
    min_extension_len: int,
    candidates_per_ext: int,
    balance_quadrants: bool,
) -> list[tuple[int, int]]:
    rs, cs = np.where(walk)
    r_med = float(np.median(rs)) if balance_quadrants and rs.size > 0 else None
    c_med = float(np.median(cs)) if balance_quadrants and cs.size > 0 else None

    d0 = bfs_dist_from(walk, h, w, start)
    far_a = max(d0, key=lambda k: d0[k])
    p = astar(walk, start, far_a)
    if not p:
        raise RuntimeError("A* start→far failed")
    cells: list[tuple[int, int]] = list(p)
    used: set[tuple[int, int]] = set(cells)

    for _ext in range(max_extensions):
        cur = cells[-1]
        cr, cc = cur
        existing_xy = _cells_world_xy(cells, meta)
        # Bez pełnego BFS od cur — losuj kandydatów, sortuj po Manhattan od bieżącego końca.
        raw = _sample_free_cells(walk, rng, max(650, candidates_per_ext * 90))
        pool = [g for g in raw if g not in used]
        pool.sort(key=lambda g: -(abs(g[0] - cr) + abs(g[1] - cc)))
        pool = pool[: min(280, len(pool))]
        candidates = _pick_extension_candidates(
            pool,
            used,
            rng,
            candidates_per_ext,
            r_med=r_med,
            c_med=c_med,
        )

        best_path: list[tuple[int, int]] | None = None
        best_score = -1.0

        for goal in candidates:
            seg = astar(walk, cur, goal)
            if seg is None or len(seg) < min_extension_len:
                continue
            ov = path_overlap_fraction(seg, used, skip_cells)
            if ov > max_cell_reuse + 1e-15:
                continue
            seg_xy = _cells_world_xy(seg, meta)
            if extension_adds_polyline_cross(existing_xy, seg_xy):
                continue
            new_cells = len(set(seg[skip_cells:]) - used)
            score = len(seg) * (1.0 - ov) + 0.5 * new_cells
            if score > best_score:
                best_score = score
                best_path = seg

        if best_path is None:
            break
        if best_path[0] == cells[-1]:
            cells.extend(best_path[1:])
            for c in best_path[1:]:
                used.add(c)
        else:
            cells.extend(best_path)
            for c in best_path:
                used.add(c)
    return cells


def anchors_from_grid_stride(
    cells: list[tuple[int, int]],
    meta: dict,
    stride: int,
) -> list[tuple[float, float]]:
    """Kotwice co `stride` komórek — A* między sąsiednimi leży blisko oryginalnej ścieżki (mniej skrótów)."""
    if not cells or stride < 1:
        return []
    out: list[tuple[float, float]] = []
    for i in range(0, len(cells), stride):
        r, c = cells[i]
        out.append(cell_to_world_center(r, c, meta, flip_y=FLIP_Y))
    last = cells[-1]
    tail = cell_to_world_center(last[0], last[1], meta, flip_y=FLIP_Y)
    if not out:
        return [tail]
    if math.hypot(out[-1][0] - tail[0], out[-1][1] - tail[1]) > 0.05:
        out.append(tail)
    return out


def cells_to_anchors(
    cells: list[tuple[int, int]],
    meta: dict,
    subsample_m: float,
) -> list[tuple[float, float]]:
    merged = [cell_to_world_center(r, c, meta, flip_y=FLIP_Y) for r, c in cells]
    anchors = [merged[0]]
    acc = 0.0
    for i in range(len(merged) - 1):
        d = math.hypot(merged[i + 1][0] - merged[i][0], merged[i + 1][1] - merged[i][1])
        acc += d
        if acc >= subsample_m:
            anchors.append(merged[i + 1])
            acc = 0.0
    if math.hypot(anchors[-1][0] - merged[-1][0], anchors[-1][1] - merged[-1][1]) > 0.15:
        anchors.append(merged[-1])
    return anchors


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _has_office_upper_left_u_turn(anchors: list[tuple[float, float]]) -> bool:
    """Detekcja WIDOCZNEGO U (dwie gałęzie y) w lewym górnym rogu office."""
    if len(anchors) < 4:
        return False
    # Lewa pionowa część U musi mieć dolną i górną gałąź na wyraźnie różnych wysokościach.
    has_left_low = any((x <= -25.0 and 16.8 <= y <= 17.8) for x, y in anchors)
    has_left_high = any((x <= -25.0 and 18.2 <= y <= 19.2) for x, y in anchors)
    # Po prawej stronie też oczekujemy obu wysokości, aby nie zliczać samego "dojazdu" i "odbicia".
    has_right_low = any((x >= -23.0 and 16.6 <= y <= 17.9) for x, y in anchors)
    has_right_high = any((x >= -23.0 and 18.2 <= y <= 19.2) for x, y in anchors)
    if not (has_left_low and has_left_high and has_right_low and has_right_high):
        return False

    # Dodatkowo sprawdź, że sąsiednie odcinki tworzą lokalnie mocny zwrot po lewej.
    for i in range(1, len(anchors) - 1):
        x, y = anchors[i]
        if x > -25.0:
            continue
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i + 1]
        h1 = math.atan2(y - y0, x - x0)
        h2 = math.atan2(y1 - y, x1 - x)
        ddeg = math.degrees(abs(_wrap_pi(h2 - h1)))
        if ddeg >= 80.0:
            return True
    return False


def _is_walkable_xy(
    x: float,
    y: float,
    walk: np.ndarray,
    meta: dict,
) -> bool:
    cell = world_to_cell(x, y, meta, flip_y=FLIP_Y)
    if cell is None:
        return False
    r, c = cell
    if r < 0 or c < 0 or r >= walk.shape[0] or c >= walk.shape[1]:
        return False
    return bool(walk[r, c])


def _closest_anchor_index(
    anchors: list[tuple[float, float]],
    target: tuple[float, float],
) -> int:
    tx, ty = target
    if not anchors:
        return 0
    best_i = 0
    best_d = float("inf")
    for i, (x, y) in enumerate(anchors):
        d = math.hypot(x - tx, y - ty)
        if d < best_d:
            best_d = d
            best_i = i
    if best_i >= len(anchors) - 1:
        return max(0, len(anchors) - 2)
    return best_i


def _dedup_anchor_seq(
    anchors: list[tuple[float, float]],
    *,
    min_step_m: float,
) -> list[tuple[float, float]]:
    if not anchors:
        return []
    out = [anchors[0]]
    for p in anchors[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_step_m:
            out.append(p)
    if len(out) == 1 and len(anchors) > 1:
        out.append(anchors[-1])
    return out


def inject_office_upper_left_u_turn(
    anchors: list[tuple[float, float]],
    blocked: np.ndarray,
    meta: dict,
    *,
    inflate_cells: int,
) -> tuple[list[tuple[float, float]], str]:
    """
    Wymuś duże U w lewym górnym rogu office.
    Zwraca (anchors, status): status in {"already_present","injected","fallback_no_valid_candidate"}.
    """
    if _has_office_upper_left_u_turn(anchors):
        return anchors, "already_present"

    walk = ~inflate_obstacles(blocked, inflate_cells)
    for seq in _OFFICE_UL_U_CANDIDATES:
        entry = seq[0]
        if not (
            all(_is_walkable_xy(x, y, walk, meta) for x, y in seq)
        ):
            continue

        idx = _closest_anchor_index(anchors, entry)
        trial = anchors[: idx + 1] + list(seq) + anchors[idx + 1 :]
        trial = _dedup_anchor_seq(trial, min_step_m=0.14)
        if len(trial) < 3:
            continue

        try:
            poly = plan_polyline_through_anchors(
                trial, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
            )
        except ValueError:
            continue
        if open_polyline_nonadjacent_segments_cross(poly):
            continue
        if _has_office_upper_left_u_turn(trial):
            return trial, "injected"

    return anchors, "fallback_no_valid_candidate"


def write_planned_yaml(
    path: Path,
    *,
    anchors: list[tuple[float, float]],
    blocked,
    meta: dict,
    inflate_cells: int,
    header_lines: list[str],
    loop_path: bool,
    dense_step_m: float,
    inflate_m: float,
) -> tuple[float, float]:
    poly = plan_polyline_through_anchors(
        anchors, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
    )
    lp = sum(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )
    lines = header_lines + ["anchors:"]
    for x, y in anchors:
        lines.append(f"  - {{x: {round(x, 3)}, y: {round(y, 3)}}}")
    lines.append(f"dense_step_m: {dense_step_m}")
    lines.append("use_astar: true")
    lines.append("map_flip_y: true")
    lines.append(f"inflate_robot_m: {inflate_m}")
    lines.append(f"loop_path: {str(loop_path).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lp, float(len(anchors))


def build_acyclic_cells_for_map(
    map_yaml: Path,
    start_xy: tuple[float, float],
    rng: random.Random,
    *,
    name: str,
    inflate_m: float,
    acyclic_max_ext: int,
    acyclic_max_reuse: float,
    balance_quadrants: bool,
    acyclic_candidates: int,
    max_extensions_override: int | None,
    acyclic_min_ext_len: int | None,
    acyclic_reuse_override: float | None,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, dict, int]:
    """Ładuje mapę, zwraca (cells, blocked, walk, meta, inflate_cells)."""
    _pgm, blocked, meta = load_reference_map_layers(str(map_yaml))
    res = float(meta["resolution"])
    inflate_cells = max(1, int(math.ceil(inflate_m / res)))
    walk = ~inflate_obstacles(blocked, inflate_cells)
    h, w = walk.shape

    def ok(x: float, y: float):
        c = world_to_cell(x, y, meta, flip_y=FLIP_Y)
        return c if c and walk[c[0], c[1]] else None

    start = ok(start_xy[0], start_xy[1])
    if not start:
        raise RuntimeError(f"{name}: start {start_xy} not free after inflation")

    n_ext = acyclic_max_ext if max_extensions_override is None else max_extensions_override
    amin_len = 18 if acyclic_min_ext_len is None else acyclic_min_ext_len
    areuse = acyclic_max_reuse if acyclic_reuse_override is None else acyclic_reuse_override

    cells = build_acyclic_cells(
        walk,
        h,
        w,
        meta,
        start,
        rng,
        max_extensions=n_ext,
        max_cell_reuse=areuse,
        skip_cells=10,
        min_extension_len=amin_len,
        candidates_per_ext=acyclic_candidates,
        balance_quadrants=balance_quadrants,
    )
    return cells, blocked, walk, meta, inflate_cells


def compute_anchors_avoid_plan_cross(
    cells: list[tuple[int, int]],
    blocked,
    meta: dict,
    inflate_cells: int,
    subsample_m: float,
) -> list[tuple[float, float]]:
    # Ponowne A* między rzadkimi kotwicami może „uciąć” rogi i przecinać wcześniejsze odcinki w XY.
    sm_hi = float(subsample_m)
    sm_lo = 0.38
    anchors = cells_to_anchors(cells, meta, sm_hi)
    poly_try = plan_polyline_through_anchors(
        anchors, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
    )
    if open_polyline_nonadjacent_segments_cross(poly_try):
        anchors_lo = cells_to_anchors(cells, meta, sm_lo)
        poly_lo = plan_polyline_through_anchors(
            anchors_lo, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
        )
        if not open_polyline_nonadjacent_segments_cross(poly_lo):
            best = (anchors_lo, poly_lo, sm_lo)
            lo_ok, hi_bad = sm_lo, sm_hi
            for _ in range(24):
                sm_mid = (lo_ok + hi_bad) / 2.0
                a_mid = cells_to_anchors(cells, meta, sm_mid)
                p_mid = plan_polyline_through_anchors(
                    a_mid, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
                )
                if open_polyline_nonadjacent_segments_cross(p_mid):
                    hi_bad = sm_mid
                else:
                    best = (a_mid, p_mid, sm_mid)
                    lo_ok = sm_mid
            anchors = best[0]
        else:
            for stride in (6, 5, 4, 3, 2):
                anchors = anchors_from_grid_stride(cells, meta, stride)
                if len(anchors) < 2:
                    continue
                poly_try = plan_polyline_through_anchors(
                    anchors, blocked, meta, flip_y=FLIP_Y, inflate_cells=inflate_cells
                )
                if not open_polyline_nonadjacent_segments_cross(poly_try):
                    break
    return anchors


def write_trajectory_yaml(
    name: str,
    map_yaml: Path,
    out_path: Path,
    *,
    cells: list[tuple[int, int]],
    blocked,
    meta: dict,
    inflate_cells: int,
    anchors: list[tuple[float, float]],
    loop_path: bool,
    tag: str,
    laps_note: int,
    inflate_m: float,
) -> None:
    n_unique = len(set(cells))
    reuse = len(cells) - n_unique
    reuse_pct = 100.0 * reuse / max(1, len(cells))

    world_name = "world_office.sdf" if "office" in map_yaml.name else "world_hospital.sdf"
    ref_rel = map_yaml.name
    hdr = [
        f"# {name}: {tag} | map {ref_rel} | world {world_name}",
        f"# Komórki: {len(cells)} (unikalnych {n_unique}, reuse {reuse_pct:.1f}%). Regeneracja: python3 scripts/generate_long_trajectories.py",
        "# loop_path w tym pliku nadpisuje domyślny parametr ROS w planned_path_driver (jeśli obecny).",
    ]
    if loop_path:
        hdr.append(
            f"# Cyclic: identyczna trasa jak *_acyclic.yaml; loop_path=true — driver po końcu wraca na początek "
            f"(wiele pełnych przejazdów; typowo min. {laps_note} dla datasetu, parametr --laps={laps_note})."
        )
    lp, _nanchors = write_planned_yaml(
        out_path,
        anchors=anchors,
        blocked=blocked,
        meta=meta,
        inflate_cells=inflate_cells,
        header_lines=hdr,
        loop_path=loop_path,
        dense_step_m=0.22,
        inflate_m=inflate_m,
    )
    print(
        f"[OK] {out_path.name}: anchors={len(anchors)}  A* poly ~{lp:.0f} m  "
        f"cells={len(cells)}  reuse%={reuse_pct:.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--laps",
        type=int,
        default=2,
        help="Cel przejazdów (cyclic = ta sama trasa co acyclic); wpis do nagłówka YAML — driver zapętla aż do stop",
    )
    ap.add_argument("--inflate-m", type=float, default=0.35)
    ap.add_argument("--subsample-m", type=float, default=1.05, help="Rzadziej kotwice = krótszy YAML, dłuższe A* odcinki")
    ap.add_argument(
        "--acyclic-max-ext",
        type=int,
        default=68,
        help="Więcej = dłuższa trasa acykliczna (wolniejszy generator)",
    )
    ap.add_argument(
        "--acyclic-max-reuse",
        type=float,
        default=0.12,
        help="Max. ułamek komórek nowego odcinka już odwiedzonych (poza skip prefix)",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if not MAP_OFFICE.is_file() or not MAP_HOSPITAL.is_file():
        print("Brak map reference_map_office/hospital.yaml", file=sys.stderr)
        return 1

    # Para (acyclic, cyclic) na mapę: cyclic = te same komórki + loop_path (dwa i więcej pełnych przejazdów w driverze).
    job_pairs = [
        (
            "Office acyclic",
            "Office cyclic",
            MAP_OFFICE,
            (0.03, 2.27),
            OUT_DIR / "office_trajectory_acyclic.yaml",
            OUT_DIR / "office_trajectory_cyclic_2lap.yaml",
            True,
            38,
            None,
            None,
            None,
        ),
        (
            "Hospital acyclic",
            "Hospital cyclic",
            MAP_HOSPITAL,
            (0.022, -24.996),
            OUT_DIR / "hospital_trajectory_acyclic.yaml",
            OUT_DIR / "hospital_trajectory_cyclic_2lap.yaml",
            True,
            42,
            92,
            12,
            0.19,
        ),
    ]

    for ta, tc, mpath, start, pa, pc, bq, acyc_cand, ext_ov, amin_ov, areuse_ov in job_pairs:
        try:
            cells, blocked, _walk, meta, inflate_cells = build_acyclic_cells_for_map(
                mpath,
                start,
                rng,
                name=f"{ta} / {tc}",
                inflate_m=args.inflate_m,
                acyclic_max_ext=args.acyclic_max_ext,
                acyclic_max_reuse=args.acyclic_max_reuse,
                balance_quadrants=bq,
                acyclic_candidates=acyc_cand,
                max_extensions_override=ext_ov,
                acyclic_min_ext_len=amin_ov,
                acyclic_reuse_override=areuse_ov,
            )
            anchors = compute_anchors_avoid_plan_cross(
                cells, blocked, meta, inflate_cells, args.subsample_m
            )
            if "office" in mpath.name.lower():
                anchors, u_status = inject_office_upper_left_u_turn(
                    anchors,
                    blocked,
                    meta,
                    inflate_cells=inflate_cells,
                )
                if u_status == "injected":
                    print("[INFO] Office: enforced upper-left U-turn (injected anchors).", flush=True)
                elif u_status == "already_present":
                    print("[INFO] Office: upper-left U-turn already present.", flush=True)
                else:
                    print(
                        "[WARN] Office: upper-left U-turn enforcement fallback (no valid candidate).",
                        flush=True,
                    )
            write_trajectory_yaml(
                ta,
                mpath,
                pa,
                cells=cells,
                blocked=blocked,
                meta=meta,
                inflate_cells=inflate_cells,
                anchors=anchors,
                loop_path=False,
                tag="acyclic",
                laps_note=args.laps,
                inflate_m=args.inflate_m,
            )
            write_trajectory_yaml(
                tc,
                mpath,
                pc,
                cells=cells,
                blocked=blocked,
                meta=meta,
                inflate_cells=inflate_cells,
                anchors=anchors,
                loop_path=True,
                tag="cyclic_same_track_loop_path",
                laps_note=args.laps,
                inflate_m=args.inflate_m,
            )
        except Exception as e:
            print(f"[FAIL] {mpath.name}: {e}", file=sys.stderr)
            return 1

    print("\nPliki zapisane w:", OUT_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
