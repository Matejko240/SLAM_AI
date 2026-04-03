#!/usr/bin/env python3
"""
Usuwa wskazane modele z poziomu <world> w pliku SDF (bez zagnieżdżonych <model> wewnątrz innych modeli).

Przykład:
  python3 scripts/remove_world_sdf_models.py \\
    ai_slam_ws/src/ai_slam_gazebo/worlds/world_office.sdf \\
    zone_meet_c staff_service_b workroom_a workroom_b \\
    focus_room_a focus_room_b focus_room_c focus_room_d focus_room_e
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_sdf", type=Path, help="Ścieżka do pliku .sdf")
    ap.add_argument("names", nargs="+", help="Atrybut name modeli do usunięcia")
    ap.add_argument("--dry-run", action="store_true", help="Tylko wypisz, co by usunięto")
    args = ap.parse_args()
    path = args.world_sdf.resolve()
    if not path.is_file():
        print(f"Brak pliku: {path}", file=sys.stderr)
        return 1

    want = set(args.names)
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "sdf":
        print("Oczekiwano korzenia <sdf>", file=sys.stderr)
        return 1
    world = root.find("world")
    if world is None:
        print("Brak <world> w SDF", file=sys.stderr)
        return 1

    removed: list[str] = []
    for child in list(world):
        if child.tag != "model":
            continue
        name = child.get("name")
        if name in want:
            world.remove(child)
            removed.append(name)

    missing = sorted(want - set(removed))
    if missing:
        print("UWAGA: nie znaleziono jako bezpośrednich dzieci <world>:", ", ".join(missing))
    print("Usunięto (world-level):", ", ".join(removed) if removed else "(nic)")

    if args.dry_run:
        return 0

    # Czytelny zapis (Python 3.9+)
    try:
        ET.indent(tree.getroot(), space="  ")
    except AttributeError:
        pass

    tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
