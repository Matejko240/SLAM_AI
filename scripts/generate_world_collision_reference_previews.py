#!/usr/bin/env python3
"""
1) Sprawdza, że w world_office.sdf, world_hospital.sdf oraz wszystkich model.sdf w ai_slam_gazebo/models
   każda collision ma parę visual (ta sama geometria + pose) — ta sama logika co test ai_slam_gazebo.
2) Rysuje PNG z *istniejących na dysku* plików reference_map_office.pgm / reference_map_hospital.pgm
   (ai_slam_eval/maps). To NIE jest automatyczny raster aktualnego SDF — po edycji świata uruchom:
     python3 scripts/generate_reference_map.py --world .../world_office.sdf --output-stem reference_map_office
     python3 scripts/generate_reference_map.py --world .../world_hospital.sdf --output-stem reference_map_hospital
   potem ponownie ten skrypt.

Uruchom z katalogu głównego repo:
  python3 scripts/generate_world_collision_reference_previews.py
  python3 scripts/generate_world_collision_reference_previews.py --out-dir out
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_GAZEBO_PKG = _REPO / "ai_slam_ws" / "src" / "ai_slam_gazebo"
_WORLDS = _GAZEBO_PKG / "worlds"
_MODELS = _GAZEBO_PKG / "models"
_VALIDATION = _GAZEBO_PKG / "test" / "collision_visual_validation.py"
_BRINGUP_SRC = _REPO / "ai_slam_ws" / "src" / "ai_slam_bringup"
_MAPS = _REPO / "ai_slam_ws" / "src" / "ai_slam_eval" / "maps"


def _load_validation():
    spec = importlib.util.spec_from_file_location("collision_visual_validation", _VALIDATION)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.validate_models_dir, mod.validate_sdf_links


def _imshow_pgm(ax, pgm: np.ndarray, meta: dict) -> None:
    ox, oy, _ = meta["origin"]
    res = float(meta["resolution"])
    h, w = int(meta["height"]), int(meta["width"])
    img = np.flipud(pgm)
    extent = (ox, ox + w * res, oy, oy + h * res)
    ax.imshow(
        img,
        cmap="gray",
        extent=extent,
        origin="lower",
        interpolation="nearest",
        alpha=1.0,
        zorder=0,
        vmin=0,
        vmax=255,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "out", help="Katalog wyjściowy PNG")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    validate_models_dir, validate_sdf_links = _load_validation()

    errs: list[str] = []
    errs.extend(validate_models_dir(_MODELS))
    for wname in ("world_office.sdf", "world_hospital.sdf"):
        p = _WORLDS / wname
        if p.is_file():
            errs.extend(validate_sdf_links(p))

    if errs:
        print("BŁĄD walidacji collision↔visual:", file=sys.stderr)
        for e in errs[:80]:
            print(" ", e, file=sys.stderr)
        if len(errs) > 80:
            print(f"  … ({len(errs)} łącznie)", file=sys.stderr)
        return 1

    print(f"OK: collision↔visual — 0 błędów (modele + world_office + world_hospital).")

    if str(_BRINGUP_SRC) not in sys.path:
        sys.path.insert(0, str(_BRINGUP_SRC))
    from ai_slam_bringup.occupancy_grid_plan import load_reference_map_layers  # noqa: E402

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Zainstaluj matplotlib (np. pip install matplotlib), aby zapisać PNG.", file=sys.stderr)
        return 1

    pairs = [
        (
            "office",
            _MAPS / "reference_map_office.yaml",
            "world_office.sdf",
        ),
        (
            "hospital",
            _MAPS / "reference_map_hospital.yaml",
            "world_hospital.sdf",
        ),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = (
        "Walidacja SDF: każda collision ma parę visual (identyczna geometria + pose).\n"
        "Zakres: ai_slam_gazebo/models/*.sdf, worlds/world_office.sdf, worlds/world_hospital.sdf."
    )

    for key, yaml_path, sdf_name in pairs:
        if not yaml_path.is_file():
            print(f"Ostrzeżenie: brak mapy {yaml_path}, pomijam PNG dla {key}.", file=sys.stderr)
            continue
        with open(yaml_path, "r", encoding="utf-8") as yf:
            yraw = yaml.safe_load(yf) or {}
        img_name = str(yraw.get("image", "")).strip()
        pgm_on_disk = yaml_path.parent / img_name if img_name else None
        sdf_on_disk = _WORLDS / sdf_name
        if (
            pgm_on_disk is not None
            and pgm_on_disk.is_file()
            and sdf_on_disk.is_file()
            and sdf_on_disk.stat().st_mtime > pgm_on_disk.stat().st_mtime
        ):
            print(
                f"UWAGA: {sdf_name} jest nowszy niż {pgm_on_disk.name} — PNG może być nieaktualny. "
                f"Uruchom: python3 scripts/generate_reference_map.py --world .../{sdf_name} --output-stem reference_map_{key}",
                file=sys.stderr,
            )
        pgm, _blocked, meta = load_reference_map_layers(str(yaml_path))
        fig, ax = plt.subplots(figsize=(12, 10))
        _imshow_pgm(ax, pgm, meta)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m] (układ jak mapa referencyjna / planowanie)")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Mapa referencyjna: {key}  |  {sdf_name}\n{status}", fontsize=11)
        ax.grid(True, alpha=0.2, color="cyan", linewidth=0.35)
        out_png = args.out_dir / f"reference_collision_visual_{key}.png"
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Zapisano: {out_png.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
