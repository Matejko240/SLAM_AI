#!/usr/bin/env python3
"""
Wizualizacja tras z planned_paths (długie + opcjonalnie przykłady) na mapach referencyjnych PGM.

Zapis (domyślnie):
  out/trajectories/trajectory_office_acyclic.png
  out/trajectories/trajectory_office_cyclic_2lap.png  (ta sama geometria co acyclic; inny tylko loop_path w YAML)
  out/trajectories/trajectory_hospital_acyclic.png
  out/trajectories/trajectory_hospital_cyclic_2lap.png

Uruchom z katalogu głównego repo:
  python3 scripts/plot_long_trajectories.py
  python3 scripts/plot_long_trajectories.py --include-examples
  python3 scripts/plot_long_trajectories.py --out-dir out/moje_podglady

Pojedyncza trasa (dowolny YAML):
  python3 scripts/plot_planned_path.py \\
    --spec ai_slam_ws/src/ai_slam_bringup/config/planned_paths/office_trajectory_acyclic.yaml \\
    --reference-map ai_slam_ws/src/ai_slam_eval/maps/reference_map_office.yaml \\
    --out out/trajectory_office_acyclic.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PLOT = _REPO / "scripts" / "plot_planned_path.py"
_MAPS = _REPO / "ai_slam_ws" / "src" / "ai_slam_eval" / "maps"
_PLANNED = _REPO / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config" / "planned_paths"

_DEFAULT_JOBS: list[tuple[str, Path, Path, str]] = [
    (
        "office_acyclic",
        _PLANNED / "office_trajectory_acyclic.yaml",
        _MAPS / "reference_map_office.yaml",
        "trajectory_office_acyclic.png",
    ),
    (
        "office_cyclic_2lap",
        _PLANNED / "office_trajectory_cyclic_2lap.yaml",
        _MAPS / "reference_map_office.yaml",
        "trajectory_office_cyclic_2lap.png",
    ),
    (
        "hospital_acyclic",
        _PLANNED / "hospital_trajectory_acyclic.yaml",
        _MAPS / "reference_map_hospital.yaml",
        "trajectory_hospital_acyclic.png",
    ),
    (
        "hospital_cyclic_2lap",
        _PLANNED / "hospital_trajectory_cyclic_2lap.yaml",
        _MAPS / "reference_map_hospital.yaml",
        "trajectory_hospital_cyclic_2lap.png",
    ),
]

_EXAMPLE_JOBS: list[tuple[str, Path, Path, str]] = [
    (
        "office_example",
        _PLANNED / "office_example.yaml",
        _MAPS / "reference_map_office.yaml",
        "trajectory_office_example_legacy.png",
    ),
    (
        "hospital_example",
        _PLANNED / "hospital_example.yaml",
        _MAPS / "reference_map_hospital.yaml",
        "trajectory_hospital_example_legacy.png",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "out" / "trajectories")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--include-examples",
        action="store_true",
        help="Dorzuć office_example.yaml / hospital_example.yaml",
    )
    args = ap.parse_args()

    if not _PLOT.is_file():
        print(f"Brak {_PLOT}", file=sys.stderr)
        return 1

    jobs = list(_DEFAULT_JOBS)
    if args.include_examples:
        jobs.extend(_EXAMPLE_JOBS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for _label, spec, ref_map, out_name in jobs:
        if not spec.is_file():
            print(f"Pomijam (brak pliku): {spec}", file=sys.stderr)
            continue
        if not ref_map.is_file():
            print(f"Pomijam (brak mapy): {ref_map}", file=sys.stderr)
            continue
        out_path = args.out_dir / out_name
        cmd = [
            sys.executable,
            str(_PLOT),
            "--spec",
            str(spec),
            "--reference-map",
            str(ref_map),
            "--out",
            str(out_path),
            "--dpi",
            str(args.dpi),
            "--view",
            "full_map",
            "--no-stats",
        ]
        r = subprocess.run(cmd, cwd=str(_REPO))
        if r.returncode != 0:
            print(f"Błąd rysowania: {spec.name}", file=sys.stderr)
            return r.returncode
        ok += 1

    print(f"Zapisano {ok} obraz(ów) w: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
