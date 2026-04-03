"""Wspólna logika: collision ↔ visual te same wymiary (geometry) i pose w każdym linku."""
from __future__ import annotations

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
    if r_el is None or l_el is None or r_el.text is None or l_el.text is None:
        return None
    return (float(r_el.text.strip()), float(l_el.text.strip()))


def _plane_normal_size(geom: ET.Element | None, ns: str) -> tuple[str, tuple[float, float]] | None:
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
    pa, pb = _plane_normal_size(a, ns), _plane_normal_size(b, ns)
    if pa is not None and pb is not None:
        return pa[0] == pb[0] and abs(pa[1][0] - pb[1][0]) < 1e-5 and abs(pa[1][1] - pb[1][1]) < 1e-5
    return ET.tostring(a, encoding="unicode") == ET.tostring(b, encoding="unicode")


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


def validate_sdf_links(path: Path) -> list[str]:
    """Zwraca listę komunikatów błędów (pusta = OK)."""
    errs: list[str] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return [f"{path}: XML {e}"]
    root = tree.getroot()
    ns = _ns(root)
    for link in root.iter(f"{ns}link"):
        lname = link.get("name", "?")
        cols = link.findall(f"{ns}collision")
        vises = link.findall(f"{ns}visual")
        if not cols and not vises:
            continue
        if not cols:
            errs.append(f"{path}: link {lname}: visual bez collision")
            continue
        for col in cols:
            cname = col.get("name", "?")
            if not _collision_has_visual_twin(col, link, ns):
                errs.append(f"{path}: link {lname}: collision {cname!r} bez pary visual (ta sama geometria+pose)")
    return errs


def validate_models_dir(models_root: Path) -> list[str]:
    all_errs: list[str] = []
    for sdf in sorted(models_root.rglob("model.sdf")):
        all_errs.extend(validate_sdf_links(sdf))
    return all_errs
