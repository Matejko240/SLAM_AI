#!/usr/bin/env python3
"""
Generuje minimalne modele Gazebo (SDF: box / cylinder) pod model://… używane w world_office / world_hospital.
Uruchom z katalogu repo:
  python3 scripts/generate_placeholder_gazebo_models.py

Nadpisuje tylko brakujące lub --force wszystkie z manifestu.
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WORLDS = _REPO / "ai_slam_ws/src/ai_slam_gazebo/worlds"
_MODELS = _REPO / "ai_slam_ws/src/ai_slam_gazebo/models"


# Modele generowane przez generate_hospital_walls_from_map.py — nie nadpisywać placeholderami.
_SKIP_PLACEHOLDER_MODELS: frozenset[str] = frozenset({"clinic_interior_walls"})


def discover_model_names() -> list[str]:
    names: set[str] = set()
    for sdf in _WORLDS.glob("*.sdf"):
        text = sdf.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        for m in re.finditer(r"model://([^<\"'\s]+)", text):
            n = m.group(1).strip()
            if n not in _SKIP_PLACEHOLDER_MODELS:
                names.add(n)
    return sorted(names)


def model_config_xml(name: str) -> str:
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <model>
          <name>{name}</name>
          <version>1.0</version>
          <sdf version="1.6">model.sdf</sdf>
          <description>Placeholder primitive (ai_slam_gazebo)</description>
        </model>
        """
    )


def sdf_box_model(name: str, sx: float, sy: float, sz: float, rgba: tuple[float, float, float]) -> str:
    r, g, b = rgba
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="{name}">
            <static>true</static>
            <link name="link">
              <collision name="col">
                <geometry>
                  <box>
                    <size>{sx} {sy} {sz}</size>
                  </box>
                </geometry>
              </collision>
              <visual name="vis">
                <geometry>
                  <box>
                    <size>{sx} {sy} {sz}</size>
                  </box>
                </geometry>
                <material>
                  <ambient>{r} {g} {b} 1</ambient>
                  <diffuse>{r} {g} {b} 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
    )


def sdf_cylinder_model(name: str, radius: float, length: float, rgba: tuple[float, float, float]) -> str:
    r, g, b = rgba
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="{name}">
            <static>true</static>
            <link name="link">
              <collision name="col">
                <geometry>
                  <cylinder>
                    <radius>{radius}</radius>
                    <length>{length}</length>
                  </cylinder>
                </geometry>
              </collision>
              <visual name="vis">
                <geometry>
                  <cylinder>
                    <radius>{radius}</radius>
                    <length>{length}</length>
                  </cylinder>
                </geometry>
                <material>
                  <ambient>{r} {g} {b} 1</ambient>
                  <diffuse>{r} {g} {b} 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
    )


def sdf_clinic_wall_shell() -> str:
    """
    Obrys dopasowany do world_hospital.sdf: pozycje include ~ x∈[-12.5,12], y∈[-31,20].
    Wcześniejsze 24×24 m zostawiało większość obiektów na zewnątrz „pudełka”.
    """
    t = 0.22
    h = 2.85
    z = h / 2.0
    # Środek ciężkości obszaru w Y (asymetria względem 0)
    cy = -5.5
    half_x = 14.0
    half_y = 26.0
    lx = 2.0 * half_x + t
    ly = 2.0 * half_y + t
    mat = "<ambient>0.82 0.83 0.86 1</ambient><diffuse>0.82 0.83 0.86 1</diffuse>"

    yn = cy + half_y + t / 2.0
    ys = cy - half_y - t / 2.0
    xe = half_x + t / 2.0
    xw = -half_x - t / 2.0

    def wall(link: str, pose: str, sx: float, sy: float, sz: float) -> str:
        return f"""
          <link name="{link}">
            <pose>{pose}</pose>
            <collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
            <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry><material>{mat}</material></visual>
          </link>"""

    body = "".join(
        [
            wall("wall_n", f"0 {yn} {z} 0 0 0", lx, t, h),
            wall("wall_s", f"0 {ys} {z} 0 0 0", lx, t, h),
            wall("wall_e", f"{xe} {cy} {z} 0 0 0", t, ly, h),
            wall("wall_w", f"{xw} {cy} {z} 0 0 0", t, ly, h),
        ]
    )
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="clinic_wall_shell">
            <static>true</static>
            {body}
          </model>
        </sdf>
        """
    )


def sdf_clinic_floor_base() -> str:
    """Posadzka pod ten sam prostokąt co ściany (link przesunięty w Y)."""
    cy = -5.5
    half_x = 14.0
    half_y = 26.0
    sx = 2.0 * half_x
    sy = 2.0 * half_y
    sz = 0.02
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="clinic_floor_base">
            <static>true</static>
            <link name="link">
              <pose>0 {cy} {sz / 2.0} 0 0 0</pose>
              <collision name="col">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
              </collision>
              <visual name="vis">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
                <material>
                  <ambient>0.88 0.88 0.9 1</ambient>
                  <diffuse>0.88 0.88 0.9 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
    )


SPECIAL_SDF: dict[str, str] = {
    "clinic_wall_shell": sdf_clinic_wall_shell(),
    "clinic_floor_base": sdf_clinic_floor_base(),
}


# Jawne rozmiary [m]; reszta wg heurystyki poniżej
EXACT: dict[str, tuple[float, float, float]] = {
    "clinic_core_station": (5.0, 3.0, 2.2),
    "clinic_lift_portal": (2.4, 0.35, 2.5),
    "clinic_lift_car": (2.0, 2.0, 2.4),
    "clinic_screen_half": (2.2, 0.06, 2.0),
    "clinic_screen_closed": (3.2, 0.06, 2.0),
    "clinic_screen_open": (3.2, 0.06, 2.0),
    "clinic_ramp": (4.0, 1.8, 0.35),
    "conference_table_large": (3.2, 1.3, 0.72),
    "conference_table_small": (1.9, 0.95, 0.72),
    "office_cafe_table": (1.15, 1.15, 0.72),
    "office_table": (1.35, 0.7, 0.72),
    "reception_desk": (2.4, 0.75, 1.05),
    "desk": (1.35, 0.65, 0.73),
    "whiteboard": (1.9, 0.04, 1.15),
    "tv_stand": (1.5, 0.45, 0.55),
    "refrigerator": (0.85, 0.85, 1.85),
    "dishwasher": (0.65, 0.65, 0.88),
    "microwave": (0.52, 0.42, 0.32),
    "coffee_maker": (0.35, 0.35, 0.48),
    "mini_fridge": (0.55, 0.55, 0.88),
    "utility_fridge": (0.7, 0.7, 1.7),
    "toilet": (0.55, 0.7, 0.42),
    "Toilet": (0.55, 0.7, 0.42),
    "Shower": (1.0, 1.0, 2.1),
    "KitchenSink": (0.65, 0.55, 0.9),
    "office_couch": (1.85, 0.82, 0.78),
    "lounge_sofa_a": (1.9, 0.85, 0.78),
    "lounge_sofa_b": (1.9, 0.85, 0.78),
    "lounge_table": (1.25, 0.65, 0.45),
    "lounge_display": (1.2, 0.35, 1.7),
    "lounge_mat": (2.5, 1.8, 0.02),
    "TrolleyBed": (2.1, 0.95, 0.65),
    "TrolleyBedPatient": (2.1, 0.95, 0.72),
    "MalePatientBed": (2.1, 1.0, 0.65),
    "AdjTable": (1.15, 0.65, 0.72),
    "BedTable": (0.45, 0.35, 0.65),
    "BedsideTable": (0.55, 0.45, 0.65),
    "StorageRack": (1.2, 0.55, 1.85),
    "StorageRackCovered": (1.25, 0.6, 1.9),
    "StorageRackCoverOpen": (1.25, 0.6, 1.9),
    "MetalCabinet": (1.0, 0.55, 1.8),
    "utility_cabinet": (0.9, 0.5, 1.75),
    "Drawer": (0.55, 0.5, 0.75),
    "storage_block_a": (1.2, 1.2, 1.5),
    "storage_block_c": (0.9, 0.9, 1.2),
    "CGMClassic": (0.55, 0.55, 1.45),
    "XRayMachine": (1.1, 0.85, 1.65),
    "AnesthesiaMachine": (0.75, 0.65, 1.35),
    "VendingMachine": (1.0, 0.75, 1.85),
    "InstrumentCart1": (0.65, 0.45, 1.05),
    "InstrumentCart2": (0.7, 0.48, 1.08),
    "SurgicalTrolley": (0.75, 0.48, 1.0),
    "SurgicalTrolleyMed": (0.72, 0.46, 0.98),
    "MopCart3": (0.55, 0.45, 1.05),
    "BMWCart": (0.55, 0.42, 0.95),
    "BPCart": (0.48, 0.38, 0.92),
    "ParkingTrolleyMin": (0.85, 0.5, 0.95),
    "ParkingTrolleyMax": (1.05, 0.58, 1.05),
    "IVStand": (0.35, 0.35, 1.35),
    "BloodPressureMonitor": (0.25, 0.22, 0.95),
    "PatientWheelChair": (0.65, 0.85, 0.95),
    "office_box": (0.55, 0.55, 0.55),
}


CYLINDER_MODELS: dict[str, tuple[float, float]] = {
    "wastebasket": (0.16, 0.38),
    "utility_bin": (0.18, 0.42),
    "desk_chair": (0.26, 0.92),
    "office_chair": (0.26, 0.92),
    "hangout_chair": (0.28, 0.95),
    "Chair": (0.26, 0.9),
    "OfficeChairBlack": (0.27, 0.93),
    "PotatoChipChair": (0.3, 0.88),
    "WhiteChipChair": (0.3, 0.88),
}


def rgba_for(name: str) -> tuple[float, float, float]:
    n = name.lower()
    if "chair" in n or n in ("chair",) or name in ("Chair", "OfficeChairBlack"):
        return (0.35, 0.38, 0.42)
    if "sofa" in n or "couch" in n:
        return (0.45, 0.42, 0.5)
    if "table" in n or "desk" in n.lower() or name == "AdjTable":
        return (0.55, 0.48, 0.4)
    if "toilet" in n or name == "Toilet":
        return (0.92, 0.92, 0.95)
    if "fridge" in n or "refrigerator" in n:
        return (0.75, 0.78, 0.82)
    if "scrubs" in n or "patient" in n or "visitor" in n or "kid" in n or name in (
        "Scrubs",
        "ElderLadyPatient",
        "ElderMalePatient",
        "FemaleVisitorSit",
        "MaleVisitorSit",
        "VisitorKidSit",
        "PatientFSit",
        "MaleVisitorOnPhone",
        "FemaleVisitor",
    ):
        return (0.55, 0.5, 0.65)
    if "metal" in n or "rack" in n or "cart" in n.lower() or "trolley" in n.lower():
        return (0.55, 0.55, 0.58)
    return (0.62, 0.6, 0.58)


def default_dims(name: str) -> tuple[float, float, float]:
    if name in EXACT:
        return EXACT[name]
    n = name.lower()
    if "bed" in n:
        return (2.0, 1.0, 0.6)
    if "chair" in n:
        return (0.55, 0.55, 0.9)
    if "table" in n or "desk" in n:
        return (1.2, 0.7, 0.72)
    if "cabinet" in n or "drawer" in n:
        return (0.8, 0.5, 1.5)
    if "cart" in n or "trolley" in n:
        return (0.7, 0.45, 1.0)
    if "machine" in n or "monitor" in n.lower():
        return (0.8, 0.6, 1.4)
    return (0.6, 0.5, 1.0)


def write_model(name: str, force: bool) -> bool:
    target = _MODELS / name
    mc = target / "model.config"
    ms = target / "model.sdf"
    if target.is_dir() and mc.is_file() and ms.is_file() and not force:
        return False

    target.mkdir(parents=True, exist_ok=True)
    mc.write_text(model_config_xml(name), encoding="utf-8")

    if name in SPECIAL_SDF:
        ms.write_text(SPECIAL_SDF[name], encoding="utf-8")
    elif name in CYLINDER_MODELS:
        rad, length = CYLINDER_MODELS[name]
        ms.write_text(sdf_cylinder_model(name, rad, length, rgba_for(name)), encoding="utf-8")
    else:
        sx, sy, sz = default_dims(name)
        ms.write_text(sdf_box_model(name, sx, sy, sz, rgba_for(name)), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Nadpisz istniejące modele")
    args = ap.parse_args()

    names = discover_model_names()
    if not names:
        print("Brak model:// w worlds/*.sdf", flush=True)
        return 1

    _MODELS.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for name in names:
        if write_model(name, args.force):
            n_new += 1
    print(f"Modele w {_MODELS}: {len(list(_MODELS.iterdir()))} katalogów, zapisano/zaktualizowano: {n_new}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
