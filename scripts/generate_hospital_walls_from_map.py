#!/usr/bin/env python3
"""
Buduje ``clinic_interior_walls``: pionowe boxy ze scalonych prostokątów na siatce
``reference_map_hospital.pgm`` — **1:1** z maską occupied (bez poszerzania drzwi,
morfologii i nadpisywania PGM).

  python3 scripts/generate_hospital_walls_from_map.py

Wynik: ai_slam_ws/src/ai_slam_gazebo/models/clinic_interior_walls/{model.config,model.sdf}
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_MAP_YAML = _REPO / "ai_slam_ws/src/ai_slam_eval/maps/reference_map_hospital.yaml"
_OUT_DIR = _REPO / "ai_slam_ws/src/ai_slam_gazebo/models/clinic_interior_walls"

FLIP_Y = True
WALL_HEIGHT = 2.85
OCC_THRESH = 128


def _f(x: float) -> str:
    return f"{float(x):.6g}"


def _read_pgm_p2(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            lines.append(s)
    assert lines[0] == "P2", lines[0]
    w, h = map(int, lines[1].split())
    _maxv = int(lines[2].split()[0])
    vals = list(map(int, " ".join(lines[3:]).split()))
    return np.array(vals, dtype=np.uint8).reshape((h, w))


def horizontal_runs(row: np.ndarray) -> list[tuple[int, int]]:
    w = int(row.shape[0])
    runs: list[tuple[int, int]] = []
    j = 0
    while j < w:
        if not row[j]:
            j += 1
            continue
        j0 = j
        while j < w and row[j]:
            j += 1
        runs.append((j0, j - 1))
    return runs


def merge_vertical_slabs(blocked: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = blocked.shape
    active: dict[tuple[int, int], tuple[int, int]] = {}
    finalized: list[tuple[int, int, int, int]] = []

    for i in range(h):
        runs = horizontal_runs(blocked[i])
        run_set = set(runs)
        for key in list(active.keys()):
            if key not in run_set:
                j0, j1 = key
                i0, i1 = active.pop(key)
                finalized.append((i0, i1, j0, j1))
        for (j0, j1) in runs:
            if (j0, j1) in active:
                i0, _ = active[(j0, j1)]
                active[(j0, j1)] = (i0, i)
            else:
                active[(j0, j1)] = (i, i)
    for (j0, j1), (i0, i1) in active.items():
        finalized.append((i0, i1, j0, j1))
    return finalized


def rect_world_pose_size(
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    *,
    meta: dict,
    flip_y: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ox, oy, _ = meta["origin"]
    res = float(meta["resolution"])
    h = int(meta["height"])

    sx = (j1 - j0 + 1) * res
    sy = (i1 - i0 + 1) * res

    cx = ox + (j0 + j1 + 1) * 0.5 * res

    if flip_y:
        iy0 = h - 1 - i0
        iy1 = h - 1 - i1
    else:
        iy0, iy1 = i0, i1
    iy_lo = min(iy0, iy1)
    iy_hi = max(iy0, iy1)
    y_min = oy + iy_lo * res
    y_max = oy + (iy_hi + 1) * res
    cy = 0.5 * (y_min + y_max)

    cz = WALL_HEIGHT * 0.5
    return (cx, cy, cz), (sx, sy, WALL_HEIGHT)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ściany szpitala z PGM (1:1, bez modyfikacji mapy)")
    ap.add_argument(
        "--occ-thresh",
        type=int,
        default=None,
        help=f"Próg occupied zamiast z YAML (domyślnie: {OCC_THRESH} lub pole occ_thresh w YAML)",
    )
    args = ap.parse_args()

    with open(_MAP_YAML, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    base = _MAP_YAML.parent
    pgm_path = base / str(y.get("image", "reference_map_hospital.pgm"))
    occ = int(args.occ_thresh if args.occ_thresh is not None else y.get("occ_thresh", OCC_THRESH))
    res = float(y.get("resolution", 0.05))
    ox, oy, oyaw = float(y["origin"][0]), float(y["origin"][1]), float(y["origin"][2]) if len(y["origin"]) > 2 else 0.0

    pgm = _read_pgm_p2(pgm_path)
    blocked = (pgm < occ).astype(bool)
    h, w = blocked.shape
    meta = {"origin": [ox, oy, oyaw], "resolution": res, "height": h, "width": w}

    n_blk = int(blocked.sum())
    print(f"PGM: {pgm_path.name}, occ_thresh={occ}, blocked={n_blk} komórek", flush=True)

    rects = merge_vertical_slabs(blocked)
    print(f"Prostokąty po scaleniu: {len(rects)}", flush=True)

    parts: list[str] = []
    for k, (i0, i1, j0, j1) in enumerate(rects):
        pos, size = rect_world_pose_size(i0, i1, j0, j1, meta=meta, flip_y=FLIP_Y)
        px, py, pz = pos
        sx, sy, sz = size
        parts.append(
            f"""      <collision name="c_{k}">
        <pose>{_f(px)} {_f(py)} {_f(pz)} 0 0 0</pose>
        <geometry><box><size>{_f(sx)} {_f(sy)} {_f(sz)}</size></box></geometry>
      </collision>
      <visual name="v_{k}">
        <pose>{_f(px)} {_f(py)} {_f(pz)} 0 0 0</pose>
        <geometry><box><size>{_f(sx)} {_f(sy)} {_f(sz)}</size></box></geometry>
        <material>
          <ambient>0.82 0.84 0.88 1</ambient>
          <diffuse>0.82 0.84 0.88 1</diffuse>
        </material>
      </visual>"""
        )

    body = "\n".join(parts)
    # Bez wiodących spacji przed <?xml — wymóg poprawnego XML (parsowanie / walidacja).
    sdf = textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="clinic_interior_walls">
            <static>true</static>
            <link name="link">
        {body}
            </link>
          </model>
        </sdf>
        """
    ).lstrip()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "model.sdf").write_text(sdf, encoding="utf-8")
    (_OUT_DIR / "model.config").write_text(
        textwrap.dedent(
            """\
            <?xml version="1.0"?>
            <model>
              <name>clinic_interior_walls</name>
              <version>1.0</version>
              <sdf version="1.6">model.sdf</sdf>
              <description>Ściany z reference_map_hospital.pgm (generate_hospital_walls_from_map.py, 1:1)</description>
            </model>
            """
        ),
        encoding="utf-8",
    )
    print(f"Zapisano {_OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
