#!/usr/bin/env python3
"""
World SDF: dopisuje brakujące <visual> jako klony <collision> (ten sam pose + geometry).

  • link tylko z collision → visual po każdym collision (od końca).
  • link z 3 collision i 1 visual, gdy ostatni collision ma tę samą skrzynkę co visual
    (porównanie liczbowe, nie surowy XML) → usuń ten visual, dodaj 3 visual-e po collision.

  • gdy len(collision) > len(visual) ≥ 1 (np. focus_cluster): dopisz brakujące visual-e
    klonując ostatnie (len(c)-len(v)) collision (nie uruchamia się, jeśli zadziałał wariant „corner”).

  python3 scripts/sync_world_sdf_visual_collision.py --world ai_slam_ws/.../world_office.sdf
"""
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def _ns(root: ET.Element) -> str:
    return root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""


def _box_size(geom: ET.Element | None, ns: str) -> tuple[float, float, float] | None:
    if geom is None:
        return None
    g = geom.find(f"{ns}box") if ns else geom.find("box")
    if g is None:
        return None
    sz = g.find(f"{ns}size") if ns else g.find("size")
    if sz is None:
        return None
    parts = "".join(sz.itertext()).split()
    if len(parts) < 3:
        return None
    return tuple(float(parts[i]) for i in range(3))


def _cylinder_r_l(geom: ET.Element | None, ns: str) -> tuple[float, float] | None:
    if geom is None:
        return None
    cyl = geom.find(f"{ns}cylinder") if ns else geom.find("cylinder")
    if cyl is None:
        return None
    r_el = cyl.find(f"{ns}radius") if ns else cyl.find("radius")
    l_el = cyl.find(f"{ns}length") if ns else cyl.find("length")
    if r_el is None or l_el is None:
        return None
    return (float(r_el.text.strip()), float(l_el.text.strip()))


def _plane_size(geom: ET.Element | None, ns: str) -> tuple[str, tuple[float, float]] | None:
    if geom is None:
        return None
    pl = geom.find(f"{ns}plane") if ns else geom.find("plane")
    if pl is None:
        return None
    n_el = pl.find(f"{ns}normal") if ns else pl.find("normal")
    s_el = pl.find(f"{ns}size") if ns else pl.find("size")
    if n_el is None or s_el is None:
        return None
    n = " ".join("".join(n_el.itertext()).split())
    sp = "".join(s_el.itertext()).split()
    if len(sp) < 2:
        return None
    return (n, (float(sp[0]), float(sp[1])))


def _geom_semantic_equal(a: ET.Element | None, b: ET.Element | None, ns: str) -> bool:
    if a is None or b is None:
        return False
    ba, bb = _box_size(a, ns), _box_size(b, ns)
    if ba is not None and bb is not None:
        return all(abs(x - y) < 1e-5 for x, y in zip(ba, bb))
    ca, cb = _cylinder_r_l(a, ns), _cylinder_r_l(b, ns)
    if ca is not None and cb is not None:
        return abs(ca[0] - cb[0]) < 1e-5 and abs(ca[1] - cb[1]) < 1e-5
    pa, pb = _plane_size(a, ns), _plane_size(b, ns)
    if pa is not None and pb is not None:
        return pa[0] == pb[0] and abs(pa[1][0] - pb[1][0]) < 1e-5 and abs(pa[1][1] - pb[1][1]) < 1e-5
    return ET.tostring(a, encoding="unicode") == ET.tostring(b, encoding="unicode")


def _default_material(ns: str) -> ET.Element:
    mat = ET.Element(f"{ns}material")
    amb = ET.SubElement(mat, f"{ns}ambient")
    amb.text = "0.78 0.78 0.8 1"
    dif = ET.SubElement(mat, f"{ns}diffuse")
    dif.text = "0.78 0.78 0.8 1"
    return mat


def _pose_tuple6(pose_el: ET.Element | None, ns: str) -> tuple[float, ...]:
    if pose_el is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = [float(x) for x in "".join(pose_el.itertext()).split()]
    while len(parts) < 6:
        parts.append(0.0)
    return tuple(parts[:6])


def _poses_close(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-5) -> bool:
    return all(abs(x - y) < eps for x, y in zip(a, b))


def _collision_has_visual_twin(col: ET.Element, link: ET.Element, ns: str) -> bool:
    cg = col.find(f"{ns}geometry")
    cp = _pose_tuple6(col.find(f"{ns}pose"), ns)
    for v in link.findall(f"{ns}visual"):
        vg = v.find(f"{ns}geometry")
        if not _geom_semantic_equal(cg, vg, ns):
            continue
        vp = _pose_tuple6(v.find(f"{ns}pose"), ns)
        if _poses_close(cp, vp):
            return True
    return False


def _visual_from_collision(col: ET.Element, ns: str, suffix: str) -> ET.Element:
    vis = ET.Element(f"{ns}visual")
    vis.set("name", f"auto_vis_{col.get('name', 'c')}_{suffix}")
    p = col.find(f"{ns}pose")
    if p is not None:
        vis.append(copy.deepcopy(p))
    g = col.find(f"{ns}geometry")
    if g is not None:
        vis.append(copy.deepcopy(g))
    vis.append(_default_material(ns))
    return vis


def _fix_link(link: ET.Element, ns: str, dry_run: bool) -> int:
    cols = list(link.findall(f"{ns}collision"))
    vises = list(link.findall(f"{ns}visual"))
    if not cols:
        return 0
    n = 0

    # corner / open_cluster: 3× collision, 1× visual == ostatni collision (semantycznie)
    if len(cols) == 3 and len(vises) == 1:
        cg = cols[2].find(f"{ns}geometry")
        vg = vises[0].find(f"{ns}geometry")
        if _geom_semantic_equal(cg, vg, ns):
            if not dry_run:
                link.remove(vises[0])
            n += 1
            cols = list(link.findall(f"{ns}collision"))
            for i, col in enumerate(cols):
                if dry_run:
                    n += 1
                    continue
                idx = list(link).index(col)
                link.insert(idx + 1, _visual_from_collision(col, ns, str(i)))
                n += 1
            return n

    if len(vises) == 0:
        for col in reversed(cols):
            if dry_run:
                n += 1
                continue
            idx = list(link).index(col)
            link.insert(idx + 1, _visual_from_collision(col, ns, "novis"))
            n += 1
        return n

    if len(cols) > len(vises) >= 1:
        k = len(cols) - len(vises)
        extras = cols[-k:]
        if dry_run:
            return k
        for j, col in enumerate(extras):
            link.append(_visual_from_collision(col, ns, f"extra{j}"))
        return k

    return 0


def _twin_repair_pass(root: ET.Element, ns: str, dry_run: bool) -> int:
    """Dopina visual dla każdej collision, która nie ma pary (pose + geometry)."""
    n = 0
    for link in root.iter(f"{ns}link"):
        for col in list(link.findall(f"{ns}collision")):
            if _collision_has_visual_twin(col, link, ns):
                continue
            n += 1
            if not dry_run:
                link.append(_visual_from_collision(col, ns, "twin"))
    return n


def process_world(path: Path, *, dry_run: bool) -> int:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = _ns(root)
    total = 0
    for link in root.iter(f"{ns}link"):
        total += _fix_link(link, ns, dry_run)
    total += _twin_repair_pass(root, ns, dry_run)
    if not dry_run and total > 0:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = args.world.resolve()
    n = process_world(path, dry_run=args.dry_run)
    print(f"{'[dry-run] ' if args.dry_run else ''}Operacji: {n} ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
