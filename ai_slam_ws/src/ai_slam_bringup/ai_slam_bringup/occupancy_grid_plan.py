"""
Planowanie ścieżki na siatce mapy referencyjnej (PGM + YAML jak map_server).
Używane przez planned_path_driver: A* między kotwicami, inflacja przeszkód.
"""
from __future__ import annotations

import heapq
import math
import os
from typing import Iterable

import numpy as np
import yaml


def _parse_origin_field(raw) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        ox, oy = float(raw[0]), float(raw[1])
        oyaw = float(raw[2]) if len(raw) >= 3 else 0.0
        return ox, oy, oyaw
    if isinstance(raw, str):
        s = raw.strip().strip("[]")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        vals = [float(p) for p in parts]
        while len(vals) < 3:
            vals.append(0.0)
        return float(vals[0]), float(vals[1]), float(vals[2])
    return 0.0, 0.0, 0.0


def load_reference_map_layers(yaml_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Zwraca:
      pgm: uint8 (H,W) wartości z pliku PGM
      blocked: bool (H,W), True = przeszkoda (pgm < occ_thresh; jak eval)
      meta: resolution, origin, shape, occ_thresh, yaml_dir, …
    """
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    ox, oy, oyaw = _parse_origin_field(y.get("origin", [0.0, 0.0, 0.0]))
    res = float(y.get("resolution", 0.05))
    img_name = str(y.get("image", "reference_map.pgm")).strip()
    occ_thresh = int(y.get("occ_thresh", 128))
    base = os.path.dirname(yaml_path)
    pgm_path = os.path.join(base, img_name)
    pgm = _load_pgm(pgm_path)
    blocked = (pgm < occ_thresh).astype(bool)
    h, w = blocked.shape
    meta = {
        "resolution": res,
        "origin": [ox, oy, oyaw],
        "width": w,
        "height": h,
        "yaml_dir": base,
        "yaml_path": yaml_path,
        "occ_thresh": occ_thresh,
    }
    return pgm, blocked, meta


def load_reference_map(yaml_path: str) -> tuple[np.ndarray, dict]:
    """
    Zwraca:
      occ: bool (H,W), True = przeszkoda (jak eval: pgm < occ_thresh)
      meta: resolution, origin (ox,oy,yaw), shape (h,w), yaml_dir
    """
    _pgm, blocked, meta = load_reference_map_layers(yaml_path)
    return blocked, meta


def _read_token(f) -> bytes | None:
    while True:
        c = f.read(1)
        if not c:
            return None
        if c.isspace():
            continue
        if c == b"#":
            f.readline()
            continue
        token = c
        break
    while True:
        c = f.read(1)
        if not c or c.isspace():
            break
        token += c
    return token


def _load_pgm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = _read_token(f)
        if magic not in (b"P2", b"P5"):
            raise ValueError(f"Unsupported PGM: {magic!r}")
        w = int(_read_token(f))
        h = int(_read_token(f))
        maxval = int(_read_token(f))
        if magic == b"P2":
            vals = []
            while True:
                t = _read_token(f)
                if t is None:
                    break
                vals.append(int(t))
            arr = np.array(vals, dtype=np.uint16 if maxval > 255 else np.uint8)
            if arr.size != w * h:
                raise ValueError(f"PGM P2 size mismatch: expected {w * h}, got {arr.size}")
        else:
            if maxval < 256:
                raw = f.read(w * h)
                if len(raw) < w * h:
                    raise ValueError("PGM P5 size mismatch")
                arr = np.frombuffer(raw[: w * h], dtype=np.uint8)
            else:
                raw = f.read(w * h * 2)
                if len(raw) < w * h * 2:
                    raise ValueError("PGM P5 size mismatch")
                arr = np.frombuffer(raw[: w * h * 2], dtype=">u2")
        return arr.reshape((h, w))


def inflate_obstacles(blocked: np.ndarray, radius_cells: int) -> np.ndarray:
    """True = nadal przeszkoda po inflacji (rozrost wolnej przestrzeni)."""
    if radius_cells <= 0:
        return blocked.astype(bool)
    h, w = blocked.shape
    occ = blocked.astype(np.uint8)
    out = occ.copy()
    r = int(radius_cells)
    for i in range(h):
        for j in range(w):
            if occ[i, j] == 0:
                continue
            i0, i1 = max(0, i - r), min(h, i + r + 1)
            j0, j1 = max(0, j - r), min(w, j + r + 1)
            out[i0:i1, j0:j1] = 1
    return out.astype(bool)


def remove_grid_loops(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Usuwa pętle ze śladu po komórkach siatki: ponowne wejście w już odwiedzoną komórkę obcina ścieżkę
    do pierwszego wystąpienia (żadna komórka nie występuje więcej niż raz). Stosować po sklejaniu
    kilku odcinków A*, które mogły nakładać się na tym samym korytarzu.
    """
    out: list[tuple[int, int]] = []
    idx_in_out: dict[tuple[int, int], int] = {}
    for c in cells:
        if c in idx_in_out:
            cut = idx_in_out[c]
            out = out[: cut + 1]
            idx_in_out = {out[j]: j for j in range(len(out))}
        else:
            idx_in_out[c] = len(out)
            out.append(c)
    return out


def world_to_cell(
    x: float,
    y: float,
    meta: dict,
    *,
    flip_y: bool,
) -> tuple[int, int] | None:
    ox, oy, _ = meta["origin"]
    res = float(meta["resolution"])
    h, w = int(meta["height"]), int(meta["width"])
    col = int(math.floor((x - ox) / res))
    iy = int(math.floor((y - oy) / res))
    row = (h - 1 - iy) if flip_y else iy
    if col < 0 or row < 0 or col >= w or row >= h:
        return None
    return row, col


def cell_to_world_center(
    row: int,
    col: int,
    meta: dict,
    *,
    flip_y: bool,
) -> tuple[float, float]:
    ox, oy, _ = meta["origin"]
    res = float(meta["resolution"])
    h = int(meta["height"])
    iy = (h - 1 - row) if flip_y else row
    wx = ox + (col + 0.5) * res
    wy = oy + (iy + 0.5) * res
    return wx, wy


def astar(
    walkable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """
    walkable[row,col] True = wolne. 8-sąsiedztwo.
    Zwraca listę komórek [start ... goal] lub None.
    """
    h, w = walkable.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if not walkable[sr, sc] or not walkable[gr, gc]:
        return None

    def heur(r: int, c: int) -> float:
        dr, dc = gr - r, gc - c
        return math.hypot(dr, dc)

    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    gscore: dict[tuple[int, int], float] = {start: 0.0}
    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heur(sr, sc), start))

    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == (gr, gc):
            path = [cur]
            p = came[cur]
            while p is not None:
                path.append(p)
                p = came[p]
            path.reverse()
            return path

        cr, cc = cur
        for dr, dc in neigh:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nc < 0 or nr >= h or nc >= w:
                continue
            if not walkable[nr, nc]:
                continue
            # nie tnij rogów przez zajęte
            if dr != 0 and dc != 0:
                if not walkable[cr + dr, cc] or not walkable[cr, cc + dc]:
                    continue
            step = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
            tentative = gscore[cur] + step
            nxt = (nr, nc)
            if tentative < gscore.get(nxt, float("inf")):
                came[nxt] = cur
                gscore[nxt] = tentative
                f = tentative + heur(nr, nc)
                heapq.heappush(open_heap, (f, nxt))
    return None


def plan_polyline_through_anchors(
    anchors_xy: list[tuple[float, float]],
    blocked: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
    inflate_cells: int,
) -> list[tuple[float, float]]:
    """Łączy kotwice łańcuchem A*; kotwice muszą leżeć w wolnej komórce po inflacji."""
    if len(anchors_xy) < 2:
        return list(anchors_xy)
    walkable = ~inflate_obstacles(blocked, inflate_cells)
    out: list[tuple[float, float]] = []
    for i in range(len(anchors_xy) - 1):
        a = anchors_xy[i]
        b = anchors_xy[i + 1]
        sa = world_to_cell(a[0], a[1], meta, flip_y=flip_y)
        sb = world_to_cell(b[0], b[1], meta, flip_y=flip_y)
        if sa is None or sb is None:
            raise ValueError(f"Anchor poza mapą: {a} / {b}")
        cells = astar(walkable, sa, sb)
        if cells is None:
            raise ValueError(f"A* nie znalazł ścieżki {a} -> {b}")
        for idx, cell in enumerate(cells):
            if i > 0 and idx == 0:
                continue  # unikaj duplikatu węzła
            out.append(cell_to_world_center(cell[0], cell[1], meta, flip_y=flip_y))
    return out


def densify_polyline(points: Iterable[tuple[float, float]], step_m: float) -> list[tuple[float, float]]:
    """Równomierne próbkowanie odcinków co ~step_m."""
    pts = list(points)
    if len(pts) < 2:
        return pts
    step_m = max(0.05, float(step_m))
    dense: list[tuple[float, float]] = [pts[0]]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        n = max(1, int(math.ceil(L / step_m)))
        for k in range(1, n + 1):
            t = min(1.0, k / n)
            dense.append((x0 + t * dx, y0 + t * dy))
    # usuń prawie-duplikaty
    cleaned = [dense[0]]
    for p in dense[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 1e-3:
            cleaned.append(p)
    return cleaned
