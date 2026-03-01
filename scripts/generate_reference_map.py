#!/usr/bin/env python3
"""
Generuje mapę referencyjną (PGM+YAML) na podstawie świata SDF.

Naprawia problem "starej mapy": zamiast hardcodowanych ścian/przeszkód,
parsuje <model>/<link>/<collision>/<geometry><box> ze wskazanego pliku .sdf.

Zapisuje do:
- ai_slam_ws/src/ai_slam_eval/maps
- oraz (jeśli istnieje) ai_slam_ws/install/ai_slam_eval/share/ai_slam_eval/maps
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET
import numpy as np


def parse_pose(text: str):
    """pose: x y z roll pitch yaw (mogą być krótsze)"""
    if not text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = [float(x) for x in text.strip().split()]
    parts += [0.0] * (6 - len(parts))
    return tuple(parts[:6])


def compose_pose(a, b):
    """
    Składanie tylko 2D (x,y,yaw). Roll/pitch ignorujemy.
    a,b: (x,y,z,roll,pitch,yaw)
    """
    ax, ay, az, _, _, ayaw = a
    bx, by, bz, _, _, byaw = b
    c = math.cos(ayaw)
    s = math.sin(ayaw)
    rx = c * bx - s * by
    ry = s * bx + c * by
    return (ax + rx, ay + ry, az + bz, 0.0, 0.0, ayaw + byaw)


def extract_boxes_from_sdf(sdf_path: str):
    """
    Zwraca listę boxów z kolizji:
    [{x,y,yaw,sx,sy}]
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    boxes = []

    for model in root.iter("model"):
        mname = model.attrib.get("name", "")
        mpose = parse_pose(model.findtext("pose"))

        # pomijamy typowe elementy
        if mname in ("ground_plane", "sun"):
            continue

        for link in model.findall("link"):
            lpose = parse_pose(link.findtext("pose"))

            for coll in link.findall("collision"):
                cpose = parse_pose(coll.findtext("pose"))
                geom = coll.find("geometry")
                if geom is None:
                    continue
                box = geom.find("box")
                if box is None:
                    continue
                size_txt = box.findtext("size")
                if not size_txt:
                    continue
                sx, sy, sz = [float(x) for x in size_txt.strip().split()]

                pose = compose_pose(compose_pose(mpose, lpose), cpose)
                x, y, z, _, _, yaw = pose

                # UWAGA: u Ciebie w world_*_house wszystkie yaw są 0, ale zostawiamy na przyszłość.
                boxes.append({"x": x, "y": y, "yaw": yaw, "sx": sx, "sy": sy})

    return boxes


def world_to_pixel(x, y, origin_x, origin_y, resolution):
    px = int((x - origin_x) / resolution)
    py = int((y - origin_y) / resolution)
    return px, py


def draw_aabb(grid, cx, cy, w, h, origin_x, origin_y, resolution, value=0):
    """
    Rysuje axis-aligned box (AABB). Dla Twoich world_house (yaw=0) to idealne.
    """
    x1, y1 = world_to_pixel(cx - w / 2, cy - h / 2, origin_x, origin_y, resolution)
    x2, y2 = world_to_pixel(cx + w / 2, cy + h / 2, origin_x, origin_y, resolution)

    h_px, w_px = grid.shape

    x1 = max(0, min(x1, w_px - 1))
    x2 = max(0, min(x2, w_px - 1))
    y1 = max(0, min(y1, h_px - 1))
    y2 = max(0, min(y2, h_px - 1))

    grid[y1 : y2 + 1, x1 : x2 + 1] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--world",
        required=True,
        help="Ścieżka do pliku .sdf (np. ai_slam_ws/src/ai_slam_gazebo/worlds/world_test_house.sdf)",
    )
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()

    world_path = os.path.abspath(args.world)
    if not os.path.isfile(world_path):
        raise FileNotFoundError(f"Nie znaleziono świata SDF: {world_path}")

    boxes = extract_boxes_from_sdf(world_path)
    if len(boxes) == 0:
        raise RuntimeError("Nie znaleziono żadnych <box> w kolizjach świata SDF.")

    # bounds
    min_x = min(b["x"] - b["sx"] / 2 for b in boxes) - args.margin
    max_x = max(b["x"] + b["sx"] / 2 for b in boxes) + args.margin
    min_y = min(b["y"] - b["sy"] / 2 for b in boxes) - args.margin
    max_y = max(b["y"] + b["sy"] / 2 for b in boxes) + args.margin

    res = float(args.resolution)
    width_px = int(math.ceil((max_x - min_x) / res))
    height_px = int(math.ceil((max_y - min_y) / res))

    origin_x = float(min_x)
    origin_y = float(min_y)

    # grid: unknown=205
    grid = np.full((height_px, width_px), 205, dtype=np.uint8)
    # free=254 w całym zakresie
    grid[:, :] = 254

    # obstacles occupied=0
    for b in boxes:
        # Dla świata house yaw=0 – rysujemy AABB.
        # Jeśli kiedyś pojawią się obroty, trzeba dodać rysowanie OBB.
        draw_aabb(grid, b["x"], b["y"], b["sx"], b["sy"], origin_x, origin_y, res, value=0)

    # odwróć Y (PGM ma Y w dół)
    grid = np.flipud(grid)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    out_dirs = [
        os.path.join(repo_root, "ai_slam_ws", "src", "ai_slam_eval", "maps"),
    ]
    install_maps = os.path.join(
        repo_root, "ai_slam_ws", "install", "ai_slam_eval", "share", "ai_slam_eval", "maps"
    )
    if os.path.isdir(install_maps):
        out_dirs.append(install_maps)

    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        pgm_path = os.path.join(out_dir, "reference_map.pgm")
        yaml_path = os.path.join(out_dir, "reference_map.yaml")

        with open(pgm_path, "w", encoding="utf-8") as f:
            f.write("P2\n")
            f.write(f"{width_px} {height_px}\n")
            f.write("255\n")
            for row in grid:
                f.write(" ".join(str(int(v)) for v in row) + "\n")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("image: reference_map.pgm\n")
            f.write(f"resolution: {res}\n")
            f.write(f"origin: [{origin_x}, {origin_y}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")

        print(f"[OK] reference map written to: {out_dir}")
        print(f"     PGM:  {pgm_path}")
        print(f"     YAML: {yaml_path}")

    print(f"\nWorld: {world_path}")
    print(f"Bounds: x=[{min_x:.2f},{max_x:.2f}] y=[{min_y:.2f},{max_y:.2f}]")
    print(f"Size: {width_px}x{height_px} px, res={res} m/px, boxes={len(boxes)}")


if __name__ == "__main__":
    main()