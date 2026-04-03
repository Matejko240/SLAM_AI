#!/usr/bin/env python3
"""
Generuje mapę referencyjną (PGM+YAML) na podstawie świata SDF.

Obsługuje:
- kolizje box/cylinder z samego świata,
- modele przez <include> (w tym URI model://nazwa → ai_slam_gazebo/models/nazwa/model.sdf),
- collision mesh w lokalnych plikach Collada (.dae).

Zapisuje do:
- ai_slam_ws/src/ai_slam_eval/maps
- oraz (jeśli istnieje) ai_slam_ws/install/ai_slam_eval/share/ai_slam_eval/maps
"""

import argparse
import math
import os
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.path import Path as MplPath


def parse_pose(text: str):
    """pose: x y z roll pitch yaw (mogą być krótsze)"""
    if not text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = [float(x) for x in text.strip().split()]
    parts += [0.0] * (6 - len(parts))
    return tuple(parts[:6])


def compose_pose(a, b):
    """
    Składanie tylko 2D (x,y,z,yaw). Roll/pitch ignorujemy.
    a,b: (x,y,z,roll,pitch,yaw)
    """
    ax, ay, az, _, _, ayaw = a
    bx, by, bz, _, _, byaw = b
    c = math.cos(ayaw)
    s = math.sin(ayaw)
    rx = c * bx - s * by
    ry = s * bx + c * by
    return (ax + rx, ay + ry, az + bz, 0.0, 0.0, ayaw + byaw)


def transform_xy(points: np.ndarray, pose) -> np.ndarray:
    """Obrót + translacja 2D dla listy punktów [N,2]."""
    x, y, _, _, _, yaw = pose
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T + np.array([x, y], dtype=float)


def transform_xyz(points: np.ndarray, pose) -> np.ndarray:
    """Obrót wokół Z + translacja 3D dla listy punktów [N,3]."""
    x, y, z, _, _, yaw = pose
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return points @ rot.T + np.array([x, y, z], dtype=float)


def _gazebo_models_root(base_dir: Path) -> Path:
    """Katalog .../ai_slam_gazebo/models względem świata (worlds/) lub modelu (models/nazwa/)."""
    b = base_dir.resolve()
    if b.name == "worlds":
        m = b.parent / "models"
        if m.is_dir():
            return m
    if b.parent.name == "models":
        return b.parent
    cur = b
    for _ in range(16):
        m, w = cur / "models", cur / "worlds"
        if m.is_dir() and w.is_dir():
            return m
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(f"Nie znaleziono ai_slam_gazebo/models (kontekst: {base_dir})")


def resolve_model_sdf(uri: str, base_dir: Path) -> Path:
    uri = str(uri or "").strip()
    if not uri:
        raise FileNotFoundError("Pusty URI modelu")
    if uri.startswith("model://"):
        name = uri.replace("model://", "", 1).strip().strip("/")
        if not name:
            raise FileNotFoundError("Pusty model:// URI")
        root = _gazebo_models_root(base_dir)
        model_sdf = root / name / "model.sdf"
        if model_sdf.is_file():
            return model_sdf.resolve()
        raise FileNotFoundError(f"Brak model.sdf dla model://{name}: {model_sdf}")

    candidate = (base_dir / uri).resolve()
    if candidate.is_dir():
        model_sdf = candidate / "model.sdf"
        if model_sdf.is_file():
            return model_sdf
    if candidate.is_file():
        return candidate

    model_sdf = (base_dir / uri / "model.sdf").resolve()
    if model_sdf.is_file():
        return model_sdf

    raise FileNotFoundError(f"Nie znaleziono model.sdf dla URI: {uri}")


def resolve_mesh_path(uri: str, base_dir: Path) -> Path:
    uri = str(uri or "").strip()
    if not uri:
        raise FileNotFoundError("Pusty URI mesha")
    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1)).resolve()
    if uri.startswith("model://"):
        raise FileNotFoundError(f"Nieobsługiwany model:// URI mesha: {uri}")
    return (base_dir / uri).resolve()


def rectangle_polygon(pose, sx: float, sy: float) -> np.ndarray:
    local = np.array(
        [
            [-sx / 2.0, -sy / 2.0],
            [sx / 2.0, -sy / 2.0],
            [sx / 2.0, sy / 2.0],
            [-sx / 2.0, sy / 2.0],
        ],
        dtype=float,
    )
    return transform_xy(local, pose)


def cylinder_polygon(pose, radius: float, samples: int = 24) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, num=samples, endpoint=False)
    local = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return transform_xy(local, pose)


@lru_cache(maxsize=128)
def load_collada_polygons(mesh_path_str: str):
    """
    Wczytuje listę polygonów 3D z pliku .dae.
    Dla mapy wykorzystujemy później ich rzut XY.
    """
    mesh_path = Path(mesh_path_str)
    root = ET.parse(mesh_path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    asset = root.find("c:asset", ns)
    unit_scale = 1.0
    up_axis = "Z_UP"
    if asset is not None:
        unit = asset.find("c:unit", ns)
        axis = asset.findtext("c:up_axis", default="Z_UP", namespaces=ns).strip() or "Z_UP"
        up_axis = axis
        if unit is not None:
            unit_scale = float(unit.attrib.get("meter", "1.0"))

    def axis_adjust(values: np.ndarray) -> np.ndarray:
        if up_axis == "Y_UP":
            return np.column_stack((values[:, 0], values[:, 2], values[:, 1]))
        if up_axis == "X_UP":
            return np.column_stack((values[:, 1], values[:, 2], values[:, 0]))
        return values

    def rotation_matrix(axis_x: float, axis_y: float, axis_z: float, angle_deg: float) -> np.ndarray:
        axis = np.asarray([axis_x, axis_y, axis_z], dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.eye(4, dtype=float)
        axis = axis / norm
        x, y, z = axis
        angle = math.radians(angle_deg)
        c = math.cos(angle)
        s = math.sin(angle)
        cc = 1.0 - c
        rot = np.array(
            [
                [x * x * cc + c, x * y * cc - z * s, x * z * cc + y * s, 0.0],
                [y * x * cc + z * s, y * y * cc + c, y * z * cc - x * s, 0.0],
                [z * x * cc - y * s, z * y * cc + x * s, z * z * cc + c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return rot

    def transform_matrix(elem) -> np.ndarray | None:
        tag = elem.tag.rsplit("}", 1)[-1]
        txt = (elem.text or "").strip()
        if tag == "translate":
            vals = [float(x) for x in txt.split()]
            if len(vals) != 3:
                return None
            mat = np.eye(4, dtype=float)
            mat[:3, 3] = vals
            return mat
        if tag == "scale":
            vals = [float(x) for x in txt.split()]
            if len(vals) != 3:
                return None
            return np.diag([vals[0], vals[1], vals[2], 1.0]).astype(float)
        if tag == "rotate":
            vals = [float(x) for x in txt.split()]
            if len(vals) != 4:
                return None
            return rotation_matrix(*vals)
        if tag == "matrix":
            vals = [float(x) for x in txt.split()]
            if len(vals) != 16:
                return None
            return np.asarray(vals, dtype=float).reshape((4, 4)).T
        return None

    def collect_mesh_polygons(mesh, convert_vertices) -> list[np.ndarray]:
        local_polygons = []
        sources = {}
        for source in mesh.findall("c:source", ns):
            source_id = source.attrib.get("id", "")
            float_array = source.find("c:float_array", ns)
            accessor = source.find(".//c:accessor", ns)
            if float_array is None or accessor is None:
                continue
            stride = int(accessor.attrib.get("stride", "3"))
            values = np.fromstring(float_array.text or "", sep=" ", dtype=float)
            if values.size == 0:
                continue
            values = values.reshape((-1, stride))[:, :3]
            sources[source_id] = convert_vertices(values)

        vertex_map = {}
        for vertices in mesh.findall("c:vertices", ns):
            vid = vertices.attrib.get("id", "")
            pos_input = vertices.find("c:input[@semantic='POSITION']", ns)
            if pos_input is not None:
                vertex_map[vid] = pos_input.attrib.get("source", "").lstrip("#")

        def append_primitives(node, default_vcount=None):
            inputs = node.findall("c:input", ns)
            if not inputs:
                return
            stride = max(int(inp.attrib.get("offset", "0")) for inp in inputs) + 1

            vertex_input = None
            for inp in inputs:
                if inp.attrib.get("semantic") in ("VERTEX", "POSITION"):
                    vertex_input = inp
                    break
            if vertex_input is None:
                return

            source_id = vertex_input.attrib.get("source", "").lstrip("#")
            if vertex_input.attrib.get("semantic") == "VERTEX":
                source_id = vertex_map.get(source_id, "")
            vertices = sources.get(source_id)
            if vertices is None:
                return

            p_text = node.findtext("c:p", default="", namespaces=ns)
            if not p_text.strip():
                return
            p_values = [int(x) for x in p_text.split()]

            if default_vcount is None:
                counts = [int(x) for x in node.findtext("c:vcount", default="", namespaces=ns).split()]
            else:
                counts = [default_vcount] * int(node.attrib.get("count", "0"))

            cursor = 0
            v_offset = int(vertex_input.attrib.get("offset", "0"))
            for count in counts:
                if count < 3:
                    cursor += count * stride
                    continue
                poly = []
                for i in range(count):
                    idx = p_values[cursor + i * stride + v_offset]
                    poly.append(vertices[idx])
                local_polygons.append(np.asarray(poly, dtype=float))
                cursor += count * stride

        for polylist in mesh.findall("c:polylist", ns):
            append_primitives(polylist)
        for triangles in mesh.findall("c:triangles", ns):
            append_primitives(triangles, default_vcount=3)
        return local_polygons

    geometry_polygons = {}
    for geom in root.findall("c:library_geometries/c:geometry", ns):
        mesh = geom.find("c:mesh", ns)
        if mesh is None:
            continue
        geom_id = geom.attrib.get("id", "")
        geometry_polygons[geom_id] = collect_mesh_polygons(mesh, lambda values: values)

    scene_polygons = []
    node_library = {
        node.attrib.get("id", ""): node for node in root.findall("c:library_nodes/c:node", ns)
    }
    visual_scene = root.find("c:library_visual_scenes/c:visual_scene", ns)

    def apply_matrix(poly: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        pts = np.column_stack((poly, np.ones((poly.shape[0],), dtype=float)))
        return (matrix @ pts.T).T[:, :3]

    def walk_node(node, parent_matrix):
        local_matrix = np.eye(4, dtype=float)
        for child in node:
            mat = transform_matrix(child)
            if mat is not None:
                local_matrix = local_matrix @ mat
        total_matrix = parent_matrix @ local_matrix

        for inst in node.findall("c:instance_geometry", ns):
            geom_id = inst.attrib.get("url", "").lstrip("#")
            for poly in geometry_polygons.get(geom_id, ()):
                scene_polygons.append(apply_matrix(poly, total_matrix) * unit_scale)

        for nested in node.findall("c:node", ns):
            walk_node(nested, total_matrix)

        for inst_node in node.findall("c:instance_node", ns):
            ref = inst_node.attrib.get("url", "").lstrip("#")
            target = node_library.get(ref)
            if target is not None:
                walk_node(target, total_matrix)

    if visual_scene is not None:
        for node in visual_scene.findall("c:node", ns):
            walk_node(node, np.eye(4, dtype=float))
    if scene_polygons:
        return tuple(scene_polygons)

    fallback_polygons = []
    for mesh in root.findall(".//c:mesh", ns):
        for poly in collect_mesh_polygons(mesh, lambda values: axis_adjust(values * unit_scale)):
            fallback_polygons.append(poly)

    return tuple(fallback_polygons)


def mesh_polygon_hits_scan(poly3: np.ndarray, scan_height: float, scan_band: float) -> bool:
    if poly3.shape[0] < 3:
        return False
    z_min = float(np.min(poly3[:, 2]))
    z_max = float(np.max(poly3[:, 2]))
    if z_max < (scan_height - scan_band) or z_min > (scan_height + scan_band):
        return False

    origin = poly3[0]
    normal = None
    for i in range(1, poly3.shape[0] - 1):
        v1 = poly3[i] - origin
        v2 = poly3[i + 1] - origin
        cross = np.cross(v1, v2)
        norm = float(np.linalg.norm(cross))
        if norm > 1e-8:
            normal = cross / norm
            break

    if normal is None:
        return False

    # Odrzucamy powierzchnie poziome i prawie poziome: podłogi, sufity, blaty.
    return abs(float(normal[2])) <= 0.45


def extract_polygons_from_model(model_elem, source_sdf: Path, parent_pose, polygons, scan_height: float, scan_band: float):
    model_name = model_elem.attrib.get("name", "")
    model_pose = compose_pose(parent_pose, parse_pose(model_elem.findtext("pose")))

    if model_name in ("ground_plane", "sun"):
        return

    source_dir = source_sdf.parent

    for include in model_elem.findall("include"):
        include_uri = include.findtext("uri", "").strip()
        if not include_uri:
            continue
        include_pose = compose_pose(model_pose, parse_pose(include.findtext("pose")))
        included_sdf = resolve_model_sdf(include_uri, source_dir)
        included_root = ET.parse(included_sdf).getroot()
        included_model = included_root.find("model")
        if included_model is not None:
            extract_polygons_from_model(included_model, included_sdf, include_pose, polygons, scan_height, scan_band)

    for nested_model in model_elem.findall("model"):
        extract_polygons_from_model(nested_model, source_sdf, model_pose, polygons, scan_height, scan_band)

    for link in model_elem.findall("link"):
        link_pose = compose_pose(model_pose, parse_pose(link.findtext("pose")))
        for coll in link.findall("collision"):
            coll_pose = compose_pose(link_pose, parse_pose(coll.findtext("pose")))
            geom = coll.find("geometry")
            if geom is None:
                continue

            box = geom.find("box")
            if box is not None:
                size_txt = box.findtext("size", "").strip()
                if size_txt:
                    sx, sy, sz = [float(x) for x in size_txt.split()]
                    z_center = coll_pose[2]
                    if (z_center - sz / 2.0) <= (scan_height + scan_band) and (z_center + sz / 2.0) >= (scan_height - scan_band):
                        polygons.append(rectangle_polygon(coll_pose, sx, sy))
                continue

            cyl = geom.find("cylinder")
            if cyl is not None:
                radius_txt = cyl.findtext("radius", "").strip()
                length_txt = cyl.findtext("length", "").strip()
                if radius_txt and length_txt:
                    radius = float(radius_txt)
                    length = float(length_txt)
                    z_center = coll_pose[2]
                    if (z_center - length / 2.0) <= (scan_height + scan_band) and (z_center + length / 2.0) >= (scan_height - scan_band):
                        polygons.append(cylinder_polygon(coll_pose, radius))
                continue

            mesh = geom.find("mesh")
            if mesh is None:
                continue
            mesh_uri = mesh.findtext("uri", "").strip()
            if not mesh_uri:
                continue

            mesh_path = resolve_mesh_path(mesh_uri, source_dir)
            for poly3 in load_collada_polygons(str(mesh_path)):
                if poly3.shape[0] < 3:
                    continue
                poly3_world = transform_xyz(poly3, coll_pose)
                if not mesh_polygon_hits_scan(poly3_world, scan_height, scan_band):
                    continue
                poly2 = poly3_world[:, :2]
                if np.ptp(poly2[:, 0]) < 1e-4 and np.ptp(poly2[:, 1]) < 1e-4:
                    continue
                polygons.append(poly2)


def extract_polygons_from_sdf(sdf_path: str, scan_height: float, scan_band: float):
    """
    Zwraca listę polygonów 2D reprezentujących kolizje świata.
    """
    sdf_file = Path(sdf_path).resolve()
    root = ET.parse(sdf_file).getroot()

    polygons = []

    world = root.find("world")
    if world is not None:
        for model in world.findall("model"):
            extract_polygons_from_model(
                model,
                sdf_file,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                polygons,
                scan_height,
                scan_band,
            )
        return polygons

    model = root.find("model")
    if model is not None:
        extract_polygons_from_model(
            model,
            sdf_file,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            polygons,
            scan_height,
            scan_band,
        )
        return polygons

    return polygons


def world_to_pixel(x, y, origin_x, origin_y, resolution):
    px = int((x - origin_x) / resolution)
    py = int((y - origin_y) / resolution)
    return px, py


def draw_polygon(grid, poly_xy, origin_x, origin_y, resolution, value=0):
    poly = np.asarray(poly_xy, dtype=float)
    if poly.shape[0] < 3:
        return

    min_x = float(np.min(poly[:, 0]))
    max_x = float(np.max(poly[:, 0]))
    min_y = float(np.min(poly[:, 1]))
    max_y = float(np.max(poly[:, 1]))

    x1, y1 = world_to_pixel(min_x, min_y, origin_x, origin_y, resolution)
    x2, y2 = world_to_pixel(max_x, max_y, origin_x, origin_y, resolution)

    h_px, w_px = grid.shape
    x1 = max(0, min(x1, w_px - 1))
    x2 = max(0, min(x2, w_px - 1))
    y1 = max(0, min(y1, h_px - 1))
    y2 = max(0, min(y2, h_px - 1))

    if x2 < x1 or y2 < y1:
        return

    # Dla osiowo ustawionych prostokątów i ścian można wypełnić AABB bez
    # drogiego rasteryzowania wielokąta. To istotnie przyspiesza mapę
    # referencyjną dla dużych shelli biura/szpitala.
    x_unique = np.unique(np.round(poly[:, 0], 5))
    y_unique = np.unique(np.round(poly[:, 1], 5))
    if len(x_unique) <= 2 and len(y_unique) <= 2:
        grid[y1 : y2 + 1, x1 : x2 + 1] = value
        return

    xs = origin_x + (np.arange(x1, x2 + 1) + 0.5) * resolution
    ys = origin_y + (np.arange(y1, y2 + 1) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    mask = MplPath(poly).contains_points(points, radius=resolution * 0.35)
    mask = mask.reshape((y2 - y1 + 1, x2 - x1 + 1))
    sub = grid[y1 : y2 + 1, x1 : x2 + 1]
    sub[mask] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--world",
        required=True,
        help="Ścieżka do pliku .sdf (np. ai_slam_ws/src/ai_slam_gazebo/worlds/world_office.sdf)",
    )
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--scan-height", type=float, default=0.15)
    ap.add_argument("--scan-band", type=float, default=0.03)
    ap.add_argument(
        "--output-stem",
        default="reference_map",
        help="Nazwa bazowa plików wyjściowych bez rozszerzenia, np. reference_map_office",
    )
    args = ap.parse_args()

    world_path = os.path.abspath(args.world)
    if not os.path.isfile(world_path):
        raise FileNotFoundError(f"Nie znaleziono świata SDF: {world_path}")

    polygons = extract_polygons_from_sdf(
        world_path,
        scan_height=float(args.scan_height),
        scan_band=float(args.scan_band),
    )
    if len(polygons) == 0:
        raise RuntimeError("Nie znaleziono żadnej obsługiwanej geometrii kolizji w świecie SDF.")

    min_x = min(float(np.min(poly[:, 0])) for poly in polygons) - args.margin
    max_x = max(float(np.max(poly[:, 0])) for poly in polygons) + args.margin
    min_y = min(float(np.min(poly[:, 1])) for poly in polygons) - args.margin
    max_y = max(float(np.max(poly[:, 1])) for poly in polygons) + args.margin

    res = float(args.resolution)
    width_px = int(math.ceil((max_x - min_x) / res))
    height_px = int(math.ceil((max_y - min_y) / res))

    origin_x = float(min_x)
    origin_y = float(min_y)

    grid = np.full((height_px, width_px), 254, dtype=np.uint8)

    for poly in polygons:
        draw_polygon(grid, poly, origin_x, origin_y, res, value=0)

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

    output_stem = str(args.output_stem).strip() or "reference_map"
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        pgm_path = os.path.join(out_dir, f"{output_stem}.pgm")
        yaml_path = os.path.join(out_dir, f"{output_stem}.yaml")

        with open(pgm_path, "w", encoding="utf-8") as f:
            f.write("P2\n")
            f.write(f"{width_px} {height_px}\n")
            f.write("255\n")
            for row in grid:
                f.write(" ".join(str(int(v)) for v in row) + "\n")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(
                "# Map_server: occupied_thresh / free_thresh. Planowanie (occ_thresh): piksel < occ_thresh = przeszkoda.\n"
            )
            f.write(f"image: {output_stem}.pgm\n")
            f.write(f"resolution: {res}\n")
            f.write(f"origin: [{origin_x}, {origin_y}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")
            f.write("occ_thresh: 128\n")

        print(f"[OK] reference map written to: {out_dir}")
        print(f"     PGM:  {pgm_path}")
        print(f"     YAML: {yaml_path}")

    print(f"\nWorld: {world_path}")
    print(f"Bounds: x=[{min_x:.2f},{max_x:.2f}] y=[{min_y:.2f},{max_y:.2f}]")
    print(f"Size: {width_px}x{height_px} px, res={res} m/px, polygons={len(polygons)}")


if __name__ == "__main__":
    main()
