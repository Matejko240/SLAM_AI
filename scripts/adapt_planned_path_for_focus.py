#!/usr/bin/env python3
"""
Generuje tymczasowy planned_path spec z adaptacją kotwic pod fokus rundy (np. mocniejszy slalom).

Założenie:
- nie wydłużamy czasu datasetu; zwiększamy pokrycie histogramów przez zmianę geometrii trasy,
  głównie przez dodatkowe kotwice o naprzemiennym odchyleniu bocznym.
- dla bezpieczeństwa kotwice są walidowane na mapie referencyjnej (A* + inflacja).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_BRINGUP_SRC = _REPO / "ai_slam_ws" / "src" / "ai_slam_bringup"
if str(_BRINGUP_SRC) not in sys.path:
    sys.path.insert(0, str(_BRINGUP_SRC))

from ai_slam_bringup.occupancy_grid_plan import (  # type: ignore  # noqa: E402
    densify_polyline,
    inflate_obstacles,
    load_reference_map,
    plan_polyline_through_anchors,
    world_to_cell,
)


@dataclass(frozen=True)
class Profile:
    enabled: bool
    spacing_m: float
    amplitude_m: float
    max_anchor_count: int
    min_spacing_m: float = 0.55
    max_amplitude_m: float = 0.60


_PROFILES: dict[str, Profile] = {
    "balanced": Profile(enabled=True, spacing_m=2.20, amplitude_m=0.16, max_anchor_count=550),
    "rotation": Profile(enabled=True, spacing_m=1.70, amplitude_m=0.26, max_anchor_count=650),
    "slalom": Profile(enabled=True, spacing_m=1.30, amplitude_m=0.34, max_anchor_count=800),
    # Translation fokus: bez agresywnego slalomu, ale z delikatnym przesunięciem toru,
    # żeby kolejne rundy dawały nowe próbki wysokich v.
    "translation": Profile(
        enabled=True,
        spacing_m=2.20,
        amplitude_m=0.10,
        max_anchor_count=680,
        min_spacing_m=1.10,
        max_amplitude_m=0.26,
    ),
}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be mapping: {path}")
    return data


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _resolve_path_from_config(path_value: str, cfg_path: Path) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p if p.exists() else None

    cfg_dir = cfg_path.resolve().parent
    src_dir = cfg_path.resolve().parents[2] if len(cfg_path.resolve().parents) >= 3 else None
    candidates = [
        cfg_dir / p,
        cfg_dir / "planned_paths" / p,
    ]
    if src_dir is not None:
        candidates.extend(
            [
                src_dir / "ai_slam_bringup" / "config" / p,
                src_dir / "ai_slam_bringup" / "config" / "planned_paths" / p,
                src_dir / "ai_slam_eval" / "maps" / p,
            ]
        )
    repo_ws_src = _REPO / "ai_slam_ws" / "src"
    candidates.extend(
        [
            _REPO / p,
            _REPO / "ai_slam_ws" / p,
            repo_ws_src / "ai_slam_bringup" / "config" / p,
            repo_ws_src / "ai_slam_bringup" / "config" / "planned_paths" / p,
            repo_ws_src / "ai_slam_eval" / "maps" / p,
        ]
    )
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _anchors_from_spec(spec: dict) -> list[tuple[float, float]]:
    raw = spec.get("anchors") or spec.get("waypoints") or []
    out: list[tuple[float, float]] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        if "x" not in p or "y" not in p:
            continue
        out.append((float(p["x"]), float(p["y"])))
    return out


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _dedup_points(points: list[tuple[float, float]], min_step: float = 0.10) -> list[tuple[float, float]]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_step:
            out.append(p)
    if len(out) == 1 and len(points) > 1:
        out.append(points[-1])
    return out


def _is_walkable(
    x: float,
    y: float,
    walkable: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
) -> bool:
    cell = world_to_cell(x, y, meta, flip_y=flip_y)
    if cell is None:
        return False
    r, c = cell
    return bool(walkable[r, c])


def _snap_candidate(
    base: tuple[float, float],
    nx: float,
    ny: float,
    sign: float,
    amplitude_m: float,
    walkable: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
) -> tuple[float, float]:
    # Najpierw próbujemy kilka skal odchylenia (w obie strony), potem fallback do punktu bazowego.
    for scale in (1.0, 0.75, 0.50, 0.25, -0.25, -0.50, -0.75, -1.0, 0.0):
        x = base[0] + sign * scale * amplitude_m * nx
        y = base[1] + sign * scale * amplitude_m * ny
        if _is_walkable(x, y, walkable, meta, flip_y=flip_y):
            return (x, y)
    return base


def _generate_slalom_anchors(
    base_poly: list[tuple[float, float]],
    walkable: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
    spacing_m: float,
    amplitude_m: float,
    run_idx: int,
    max_anchor_count: int,
) -> list[tuple[float, float]]:
    dense = densify_polyline(base_poly, spacing_m)
    if len(dense) < 3:
        return dense

    out: list[tuple[float, float]] = [dense[0]]
    sign = 1.0 if (run_idx % 2) == 1 else -1.0
    for i in range(1, len(dense) - 1):
        prev_p = dense[i - 1]
        cur_p = dense[i]
        next_p = dense[i + 1]
        tx = next_p[0] - prev_p[0]
        ty = next_p[1] - prev_p[1]
        norm = math.hypot(tx, ty)
        if norm < 1e-6:
            continue
        nx = -ty / norm
        ny = tx / norm

        h1 = math.atan2(cur_p[1] - prev_p[1], cur_p[0] - prev_p[0])
        h2 = math.atan2(next_p[1] - cur_p[1], next_p[0] - cur_p[0])
        turn = abs(_wrap_pi(h2 - h1))
        curv_scale = 1.0 + min(0.8, turn / 1.25)
        amp_local = min(0.70, amplitude_m * curv_scale)
        candidate = _snap_candidate(
            cur_p,
            nx,
            ny,
            sign,
            amp_local,
            walkable,
            meta,
            flip_y=flip_y,
        )
        out.append(candidate)
        sign *= -1.0
    out.append(dense[-1])
    cleaned = _dedup_points(out, min_step=max(0.08, spacing_m * 0.25))
    if len(cleaned) <= max_anchor_count:
        return cleaned
    idxs = np.linspace(0, len(cleaned) - 1, num=max_anchor_count, dtype=np.int64)
    keep_idx = sorted(set(int(i) for i in idxs.tolist()))
    reduced = [cleaned[i] for i in keep_idx]
    if len(reduced) == 1 and len(cleaned) > 1:
        reduced.append(cleaned[-1])
    return reduced


def _generate_translation_anchors(
    base_poly: list[tuple[float, float]],
    walkable: np.ndarray,
    meta: dict,
    *,
    flip_y: bool,
    spacing_m: float,
    amplitude_m: float,
    run_idx: int,
    max_anchor_count: int,
) -> list[tuple[float, float]]:
    dense = densify_polyline(base_poly, spacing_m)
    if len(dense) < 3:
        return dense

    out: list[tuple[float, float]] = [dense[0]]
    phase = (run_idx % 4) * (math.pi / 4.0)
    wave_period_pts = max(10, int(round(6.0 / max(0.2, spacing_m))))

    for i in range(1, len(dense) - 1):
        prev_p = dense[i - 1]
        cur_p = dense[i]
        next_p = dense[i + 1]
        tx = next_p[0] - prev_p[0]
        ty = next_p[1] - prev_p[1]
        norm = math.hypot(tx, ty)
        if norm < 1e-6:
            continue
        nx = -ty / norm
        ny = tx / norm

        # Delikatne wygładzanie narożników zamiast częstego zygzaka.
        smooth_w = 0.06
        base_x = (1.0 - 2.0 * smooth_w) * cur_p[0] + smooth_w * (prev_p[0] + next_p[0])
        base_y = (1.0 - 2.0 * smooth_w) * cur_p[1] + smooth_w * (prev_p[1] + next_p[1])
        base_pt = (base_x, base_y)
        if not _is_walkable(base_pt[0], base_pt[1], walkable, meta, flip_y=flip_y):
            base_pt = cur_p

        h1 = math.atan2(cur_p[1] - prev_p[1], cur_p[0] - prev_p[0])
        h2 = math.atan2(next_p[1] - cur_p[1], next_p[0] - cur_p[0])
        turn = abs(_wrap_pi(h2 - h1))

        candidate = base_pt
        if amplitude_m > 1e-6 and turn < math.radians(24.0):
            wave = math.sin((2.0 * math.pi * float(i) / float(wave_period_pts)) + phase)
            amp_local = amplitude_m * wave
            if abs(amp_local) > 1e-6:
                sign = 1.0 if amp_local >= 0.0 else -1.0
                candidate = _snap_candidate(
                    base_pt,
                    nx,
                    ny,
                    sign,
                    abs(amp_local),
                    walkable,
                    meta,
                    flip_y=flip_y,
                )
        out.append(candidate)

    out.append(dense[-1])
    cleaned = _dedup_points(out, min_step=max(0.08, spacing_m * 0.20))
    if len(cleaned) <= max_anchor_count:
        return cleaned
    idxs = np.linspace(0, len(cleaned) - 1, num=max_anchor_count, dtype=np.int64)
    keep_idx = sorted(set(int(i) for i in idxs.tolist()))
    reduced = [cleaned[i] for i in keep_idx]
    if len(reduced) == 1 and len(cleaned) > 1:
        reduced.append(cleaned[-1])
    return reduced


def _adaptive_profile(mode: str, run_idx: int) -> Profile:
    base = _PROFILES.get(mode, _PROFILES["balanced"])
    if not base.enabled:
        return base
    growth = 1.0 + 0.08 * max(0, run_idx - 1)
    amp = min(base.max_amplitude_m, base.amplitude_m * growth)
    spacing = max(base.min_spacing_m, base.spacing_m / (1.0 + 0.06 * max(0, run_idx - 1)))
    return Profile(
        enabled=True,
        spacing_m=spacing,
        amplitude_m=amp,
        max_anchor_count=base.max_anchor_count,
        min_spacing_m=base.min_spacing_m,
        max_amplitude_m=base.max_amplitude_m,
    )


def _rounded_anchors(
    anchors: list[tuple[float, float]],
    *,
    ndigits: int,
) -> list[tuple[float, float]]:
    return [(round(float(x), ndigits), round(float(y), ndigits)) for x, y in anchors]


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptacja planned path (kotwice) pod fokus rundy")
    ap.add_argument("--config", type=Path, required=True, help="Run config YAML do modyfikacji (in-place)")
    ap.add_argument("--focus-mode", type=str, required=True, help="balanced|rotation|translation|slalom")
    ap.add_argument("--run-idx", type=int, required=True, help="Numer rundy (>=1)")
    ap.add_argument("--work-dir", type=Path, required=True, help="Katalog na tymczasowy spec YAML")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    cfg = _load_yaml(cfg_path)
    mode = args.focus_mode.strip().lower()
    run_idx = max(1, int(args.run_idx))
    profile = _adaptive_profile(mode, run_idx)

    driver = cfg.setdefault("driver", {})
    pp = driver.setdefault("planned_path", {})
    experiment = cfg.setdefault("experiment", {})
    spec_val = str(pp.get("spec_yaml", "")).strip()
    if not spec_val:
        experiment["adaptive_path_profile"] = mode
        experiment["adaptive_path_status"] = "skipped:no_spec"
        _save_yaml(cfg_path, cfg)
        print(json.dumps({"status": "skipped", "reason": "no_spec_yaml"}, ensure_ascii=False))
        return 0

    spec_path = _resolve_path_from_config(spec_val, cfg_path)
    if spec_path is None:
        experiment["adaptive_path_profile"] = mode
        experiment["adaptive_path_status"] = "skipped:spec_not_found"
        _save_yaml(cfg_path, cfg)
        print(json.dumps({"status": "skipped", "reason": "spec_not_found", "spec_yaml": spec_val}, ensure_ascii=False))
        return 0

    if not profile.enabled:
        experiment["adaptive_path_profile"] = mode
        experiment["adaptive_path_status"] = "skipped:profile_disabled"
        _save_yaml(cfg_path, cfg)
        print(json.dumps({"status": "skipped", "reason": "profile_disabled", "mode": mode}, ensure_ascii=False))
        return 0

    spec = _load_yaml(spec_path)
    anchors = _anchors_from_spec(spec)
    if len(anchors) < 2:
        experiment["adaptive_path_profile"] = mode
        experiment["adaptive_path_status"] = "skipped:not_enough_anchors"
        _save_yaml(cfg_path, cfg)
        print(json.dumps({"status": "skipped", "reason": "not_enough_anchors"}, ensure_ascii=False))
        return 0

    use_astar = bool(spec.get("use_astar", bool(pp.get("use_astar", False))))
    map_flip_y = bool(spec.get("map_flip_y", bool(pp.get("map_flip_y", True))))
    inflate_m = float(spec.get("inflate_robot_m", float(pp.get("inflate_robot_m", 0.35))))
    ref_map_val = str(pp.get("reference_map_yaml", cfg.get("evaluation", {}).get("reference_map_yaml", ""))).strip()
    ref_map_path = _resolve_path_from_config(ref_map_val, cfg_path) if ref_map_val else None

    if not use_astar or ref_map_path is None:
        experiment["adaptive_path_profile"] = mode
        experiment["adaptive_path_status"] = "skipped:no_astar_or_map"
        _save_yaml(cfg_path, cfg)
        print(
            json.dumps(
                {"status": "skipped", "reason": "no_astar_or_reference_map", "use_astar": use_astar, "reference_map": ref_map_val},
                ensure_ascii=False,
            )
        )
        return 0

    blocked, meta = load_reference_map(str(ref_map_path))
    res = float(meta["resolution"])
    inflate_cells = max(1, int(math.ceil(inflate_m / res)))
    walkable = ~inflate_obstacles(blocked, inflate_cells)
    base_poly = plan_polyline_through_anchors(
        anchors,
        blocked,
        meta,
        flip_y=map_flip_y,
        inflate_cells=inflate_cells,
    )

    chosen: list[tuple[float, float]] | None = None
    chosen_serialized: list[tuple[float, float]] | None = None
    chosen_amp = 0.0
    attempts = []
    serialize_ndigits = 3
    for attempt_idx in range(5):
        amp_floor = 0.0 if mode == "translation" else 0.06
        amp_try = max(amp_floor, profile.amplitude_m * (0.70 ** attempt_idx))
        if mode == "translation":
            anchors_try = _generate_translation_anchors(
                base_poly,
                walkable,
                meta,
                flip_y=map_flip_y,
                spacing_m=profile.spacing_m,
                amplitude_m=amp_try,
                run_idx=run_idx,
                max_anchor_count=profile.max_anchor_count,
            )
        else:
            anchors_try = _generate_slalom_anchors(
                base_poly,
                walkable,
                meta,
                flip_y=map_flip_y,
                spacing_m=profile.spacing_m,
                amplitude_m=amp_try,
                run_idx=run_idx,
                max_anchor_count=profile.max_anchor_count,
            )
        anchors_serialized = _rounded_anchors(anchors_try, ndigits=serialize_ndigits)
        try:
            _ = plan_polyline_through_anchors(
                anchors_serialized,
                blocked,
                meta,
                flip_y=map_flip_y,
                inflate_cells=inflate_cells,
            )
            chosen = anchors_try
            chosen_serialized = anchors_serialized
            chosen_amp = amp_try
            attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "amplitude_m": amp_try,
                    "serialized_ndigits": serialize_ndigits,
                    "status": "ok",
                }
            )
            break
        except ValueError as exc:
            attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "amplitude_m": amp_try,
                    "serialized_ndigits": serialize_ndigits,
                    "status": f"fail:{exc}",
                }
            )

    if chosen is None or chosen_serialized is None:
        # Twardy fallback A*: zostawiamy oryginalny spec bez adaptacji.
        pp["spec_yaml"] = str(spec_path)
        experiment["adaptive_path_profile"] = mode
        experiment["adaptive_path_status"] = "fallback:source_spec"
        experiment["adaptive_selected_path"] = str(spec_path)
        experiment["adaptive_path_attempts"] = attempts
        _save_yaml(cfg_path, cfg)
        print(
            json.dumps(
                {"status": "fallback", "reason": "validation_failed", "attempts": attempts, "spec_yaml": str(spec_path)},
                ensure_ascii=False,
            )
        )
        return 0

    out_spec = args.work_dir.resolve() / f"adaptive_path_{mode}_r{run_idx}_{spec_path.name}"
    out_spec_data = dict(spec)
    out_spec_data["anchors"] = [{"x": float(x), "y": float(y)} for x, y in chosen_serialized]
    out_spec_data["adaptive_source_spec_yaml"] = str(spec_path)
    out_spec_data["adaptive_focus_mode"] = mode
    out_spec_data["adaptive_spacing_m"] = float(profile.spacing_m)
    out_spec_data["adaptive_amplitude_m"] = float(chosen_amp)
    out_spec_data["adaptive_anchor_round_ndigits"] = int(serialize_ndigits)
    out_spec_data["adaptive_run_idx"] = int(run_idx)
    _save_yaml(out_spec, out_spec_data)

    pp["spec_yaml"] = str(out_spec)
    experiment["adaptive_selected_path"] = str(out_spec)
    experiment["adaptive_path_profile"] = mode
    experiment["adaptive_path_status"] = "generated"
    experiment["adaptive_path_anchor_count"] = int(len(chosen))
    experiment["adaptive_path_attempts"] = attempts
    _save_yaml(cfg_path, cfg)

    print(
        json.dumps(
            {
                "status": "generated",
                "mode": mode,
                "source_spec": str(spec_path),
                "output_spec": str(out_spec),
                "anchor_count": len(chosen),
                "spacing_m": profile.spacing_m,
                "amplitude_m": chosen_amp,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
