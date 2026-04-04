#!/usr/bin/env python3
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

from ai_slam_bringup.occupancy_grid_plan import load_reference_map_layers  # type: ignore  # noqa: E402


@dataclass
class RunInfo:
    experiment_id: str
    run_config: Path
    world_name: str
    reference_map: Path | None


def _resolve_reference_map(candidate: str, run_cfg: Path) -> Path | None:
    raw = str(candidate or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p.resolve()

    ws_src = _REPO / "ai_slam_ws" / "src"
    run_dir = run_cfg.parent
    checks = [
        run_dir / p,
        _REPO / p,
        _REPO / "ai_slam_ws" / p,
        ws_src / "ai_slam_eval" / "maps" / p,
        ws_src / "ai_slam_bringup" / "config" / p,
    ]
    for c in checks:
        if c.exists():
            return c.resolve()
    return None


def _run_info(exp_id: str, run_cfg: Path) -> RunInfo:
    cfg = yaml.safe_load(run_cfg.read_text(encoding="utf-8")) or {}
    sim = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
    driver = cfg.get("driver", {}) if isinstance(cfg.get("driver"), dict) else {}
    planned = driver.get("planned_path", {}) if isinstance(driver.get("planned_path"), dict) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}

    world_name = str(sim.get("train_world", "")).strip() or "unknown_world"
    ref_raw = str(planned.get("reference_map_yaml", "")).strip() or str(eval_cfg.get("reference_map_yaml", "")).strip()
    ref_map = _resolve_reference_map(ref_raw, run_cfg)
    return RunInfo(experiment_id=exp_id, run_config=run_cfg, world_name=world_name, reference_map=ref_map)


def _load_dataset_traj(exp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    ds = exp_dir / "dataset.npz"
    if not ds.exists():
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    with np.load(ds, allow_pickle=True) as data:
        if "X_odom" not in data or "Y" not in data:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        odom = np.asarray(data["X_odom"], dtype=np.float32).reshape((-1, 3))
        corr = np.asarray(data["Y"], dtype=np.float32).reshape((-1, 3))

    n = min(int(odom.shape[0]), int(corr.shape[0]))
    if n <= 1:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    odom_xy = odom[:n, :2].astype(np.float32)
    gt_xy = (odom[:n, :2] + corr[:n, :2]).astype(np.float32)

    mask = np.isfinite(odom_xy[:, 0]) & np.isfinite(odom_xy[:, 1]) & np.isfinite(gt_xy[:, 0]) & np.isfinite(gt_xy[:, 1])
    odom_xy = odom_xy[mask]
    gt_xy = gt_xy[mask]
    return odom_xy, gt_xy


def _stuck_mask(gt_xy: np.ndarray, *, min_step_m: float, min_run_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if gt_xy.shape[0] <= 2:
        return np.zeros((gt_xy.shape[0],), dtype=bool), []
    steps = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).astype(np.float32)
    low = steps < float(min_step_m)
    out = np.zeros((gt_xy.shape[0],), dtype=bool)
    segs: list[tuple[int, int]] = []

    start = -1
    for i, v in enumerate(low.tolist()):
        if v and start < 0:
            start = i
        if (not v) and start >= 0:
            end = i - 1
            if (end - start + 1) >= min_run_len:
                p0 = int(start)
                p1 = int(end + 1)
                out[p0 : p1 + 1] = True
                segs.append((p0, p1))
            start = -1
    if start >= 0:
        end = int(low.shape[0] - 1)
        if (end - start + 1) >= min_run_len:
            p0 = int(start)
            p1 = int(end + 1)
            out[p0 : p1 + 1] = True
            segs.append((p0, p1))
    return out, segs


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)


def _plot_group(
    out_dir: Path,
    map_path: Path,
    items: list[RunInfo],
    *,
    stuck_step_m: float,
    stuck_min_steps: int,
) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib is required: {exc}") from exc

    pgm, _blocked, meta = load_reference_map_layers(str(map_path))
    ox, oy, _ = meta["origin"]
    res = float(meta["resolution"])
    h = int(meta["height"])
    w = int(meta["width"])
    extent = (ox, ox + w * res, oy, oy + h * res)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(np.flipud(pgm), cmap="gray", extent=extent, origin="lower", interpolation="nearest", alpha=0.96, vmin=0, vmax=255)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.4)

    cmap = plt.get_cmap("tab20")
    summary_runs: list[dict] = []
    max_legend = 16
    for idx, info in enumerate(items):
        exp_dir = _REPO / "out" / info.experiment_id
        odom_xy, gt_xy = _load_dataset_traj(exp_dir)
        if gt_xy.shape[0] <= 1:
            summary_runs.append(
                {
                    "experiment_id": info.experiment_id,
                    "world_name": info.world_name,
                    "samples": int(gt_xy.shape[0]),
                    "trajectory_length_m": 0.0,
                    "stuck_segments": 0,
                    "stuck_points": 0,
                    "status": "missing_or_empty_dataset",
                }
            )
            continue

        color = cmap(idx % 20)
        label = info.experiment_id
        steps = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).astype(np.float32)
        traj_len = float(np.sum(steps))
        stuck, segs = _stuck_mask(gt_xy, min_step_m=stuck_step_m, min_run_len=stuck_min_steps)

        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=color, linewidth=1.7, alpha=0.92, label=label if idx < max_legend else None, zorder=3)
        ax.plot(odom_xy[:, 0], odom_xy[:, 1], color=color, linewidth=0.8, alpha=0.30, linestyle="--", zorder=2)
        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], s=18, c=[color], marker="o", alpha=0.9, zorder=4)
        ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], s=22, c=[color], marker="X", alpha=0.9, zorder=4)
        if np.any(stuck):
            ax.scatter(gt_xy[stuck, 0], gt_xy[stuck, 1], s=14, c=[color], marker="x", alpha=0.9, zorder=5)

        stuck_segments = int(len(segs))
        stuck_points = int(np.sum(stuck))
        status = "ok"
        quality_gate_fail = False
        if stuck_points >= 120:
            status = "stuck_points"
            quality_gate_fail = True
        elif stuck_segments >= 3:
            status = "stuck_segments"
            quality_gate_fail = True
        elif stuck_segments >= 2 and stuck_points >= 60:
            status = "stuck_mixed"
            quality_gate_fail = True
        elif traj_len > 0.0 and traj_len < 100.0:
            status = "low_progress"
            quality_gate_fail = True

        summary_runs.append(
            {
                "experiment_id": info.experiment_id,
                "world_name": info.world_name,
                "samples": int(gt_xy.shape[0]),
                "trajectory_length_m": traj_len,
                "stuck_segments": stuck_segments,
                "stuck_points": stuck_points,
                "status": status,
                "quality_gate_fail": bool(quality_gate_fail),
            }
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Merged dataset trajectories | map={map_path.name} | runs={len(items)}\n"
        f"solid=GT, dashed=odom, x=possible stuck (step<{stuck_step_m:.3f}m for >= {stuck_min_steps} steps)"
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    out_png = out_dir / f"merged_trajectories_overlay_{_safe_stem(map_path)}.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "reference_map_yaml": str(map_path),
        "world_names": sorted({i.world_name for i in items}),
        "overlay_png": str(out_png),
        "runs": summary_runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rysuje overlay trajektorii wszystkich rund datasetowych użytych w merge "
            "(na mapie referencyjnej), żeby wykryć utknięcia i problemy przejazdu."
        )
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Katalog merged experiment (out/exp_multi_merged_*)")
    ap.add_argument("--experiment-ids", nargs="+", required=True, help="Lista exp_multi_dataset_* użytych w merge")
    ap.add_argument("--run-configs", nargs="+", required=True, help="Lista run_*.yaml odpowiadająca --experiment-ids")
    ap.add_argument("--stuck-step-m", type=float, default=0.01, help="Próg kroku [m] do flagi possible stuck")
    ap.add_argument("--stuck-min-steps", type=int, default=12, help="Minimalna liczba kolejnych małych kroków")
    ap.add_argument("--summary-name", default="merged_trajectory_overview_summary.json")
    args = ap.parse_args()

    exp_ids = [str(x).strip() for x in args.experiment_ids if str(x).strip()]
    cfgs = [Path(x).expanduser().resolve() for x in args.run_configs if str(x).strip()]
    if len(exp_ids) != len(cfgs):
        raise SystemExit(f"Length mismatch: experiment_ids={len(exp_ids)} run_configs={len(cfgs)}")

    infos: list[RunInfo] = []
    for exp_id, cfg_path in zip(exp_ids, cfgs):
        if not cfg_path.exists():
            raise SystemExit(f"Missing run config: {cfg_path}")
        infos.append(_run_info(exp_id, cfg_path))

    by_map: dict[str, list[RunInfo]] = {}
    missing_map_runs: list[dict] = []
    for info in infos:
        if info.reference_map is None:
            missing_map_runs.append(
                {
                    "experiment_id": info.experiment_id,
                    "run_config": str(info.run_config),
                    "world_name": info.world_name,
                    "status": "missing_reference_map",
                }
            )
            continue
        key = str(info.reference_map)
        by_map.setdefault(key, []).append(info)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    for map_key, items in sorted(by_map.items(), key=lambda kv: kv[0]):
        outputs.append(
            _plot_group(
                out_dir,
                Path(map_key),
                items,
                stuck_step_m=float(args.stuck_step_m),
                stuck_min_steps=max(1, int(args.stuck_min_steps)),
            )
        )

    summary = {
        "experiment_count": len(exp_ids),
        "map_group_count": len(outputs),
        "map_groups": outputs,
        "missing_map_runs": missing_map_runs,
        "stuck_step_m": float(args.stuck_step_m),
        "stuck_min_steps": int(args.stuck_min_steps),
    }
    summary_path = out_dir / str(args.summary_name)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] Trajectory overview summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
