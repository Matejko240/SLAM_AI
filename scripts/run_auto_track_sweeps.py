#!/usr/bin/env python3
"""Automatyczny, sekwencyjny sweep per-track dla testów hospital na BIG_DATASET.

Cel:
- tor5 Robak+SLAM: ograniczyć drift translacyjny (RMSE vs classic_slam),
- tor6 Rywak+SLAM: ograniczyć drift yaw/trajektorii (RMSE vs classic_slam),
- tor7/tor8 no-SLAM: wyjść ponad odometrię (głównie IoU, pomocniczo RMSE).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml
from out_layout import ensure_grouped_out_layout, ensure_sweep_storage, resolve_experiment_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config" / "experiment_config.yaml"
RUN_FULL_CYCLE = REPO_ROOT / "scripts" / "run_full_cycle.sh"
STRICT_NO_SLAM_ODOM = True
STRICT_SLAM_CLASSIC = False


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def get_nested(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    current = payload
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def fmt_value(value: Any) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "_")


def metric(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = metrics.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_score(value: float) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return float("inf")
    return float(value)


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def aggregate_metrics(metric_runs: list[dict[str, Any]]) -> dict[str, float]:
    if not metric_runs:
        return {}
    keys: set[str] = set()
    for payload in metric_runs:
        keys.update(payload.keys())
    aggregated: dict[str, float] = {}
    for key in sorted(keys):
        vals: list[float] = []
        for payload in metric_runs:
            value = payload.get(key)
            if value is None:
                continue
            try:
                casted = float(value)
            except Exception:
                continue
            if math.isnan(casted) or math.isinf(casted):
                continue
            vals.append(casted)
        if vals:
            aggregated[key] = mean(vals)
    return aggregated


def hard_must_beat_odom_penalty(
    rmse_no_slam: float,
    rmse_odom: float,
    iou_no_slam: float,
    iou_odom: float,
) -> float:
    return hard_must_beat_reference_penalty(
        rmse_candidate=rmse_no_slam,
        rmse_reference=rmse_odom,
        iou_candidate=iou_no_slam,
        iou_reference=iou_odom,
        base_penalty=1000.0,
        scale_penalty=5000.0,
    )


def hard_must_beat_reference_penalty(
    rmse_candidate: float,
    rmse_reference: float,
    iou_candidate: float,
    iou_reference: float,
    base_penalty: float,
    scale_penalty: float,
) -> float:
    penalty = 0.0
    rmse_gap = rmse_candidate - rmse_reference
    iou_gap = iou_reference - iou_candidate
    if rmse_gap > 0.0:
        penalty += float(base_penalty) + float(scale_penalty) * rmse_gap
    if iou_gap > 0.0:
        penalty += float(base_penalty) + float(scale_penalty) * iou_gap
    return penalty


def load_metrics(experiment_id: str) -> dict[str, Any]:
    try:
        exp_dir = resolve_experiment_dir(experiment_id)
    except Exception:
        return {}
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


@dataclass(frozen=True)
class ParamSweep:
    path: str
    values: list[Any]


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    objective_name: str
    objective: Callable[[dict[str, Any]], float]
    params: list[ParamSweep]


def objective_robak_slam(metrics: dict[str, Any]) -> float:
    rmse_robak = metric(metrics, "rmse_xy_robak_slam")
    rmse_classic = metric(metrics, "rmse_xy_classic_slam")
    iou_robak = metric(metrics, "iou_map_robak")
    iou_classic = metric(metrics, "iou_map_classic_slam")
    rmse_delta = metric(metrics, "rmse_xy_robak_slam") - metric(metrics, "rmse_xy_classic_slam")
    iou_gap = metric(metrics, "iou_map_classic_slam") - metric(metrics, "iou_map_robak")
    score = 100.0 * rmse_delta + 25.0 * max(0.0, iou_gap)
    if STRICT_SLAM_CLASSIC:
        score += hard_must_beat_reference_penalty(
            rmse_candidate=rmse_robak,
            rmse_reference=rmse_classic,
            iou_candidate=iou_robak,
            iou_reference=iou_classic,
            base_penalty=1500.0,
            scale_penalty=7000.0,
        )
    return safe_score(score)


def objective_rywak_slam(metrics: dict[str, Any]) -> float:
    rmse_rywak = metric(metrics, "rmse_xy_rywak_slam")
    rmse_classic = metric(metrics, "rmse_xy_classic_slam")
    iou_rywak = metric(metrics, "iou_map_rywak")
    iou_classic = metric(metrics, "iou_map_classic_slam")
    rmse_delta = metric(metrics, "rmse_xy_rywak_slam") - metric(metrics, "rmse_xy_classic_slam")
    iou_gap = metric(metrics, "iou_map_classic_slam") - metric(metrics, "iou_map_rywak")
    score = 100.0 * rmse_delta + 20.0 * max(0.0, iou_gap)
    if STRICT_SLAM_CLASSIC:
        score += hard_must_beat_reference_penalty(
            rmse_candidate=rmse_rywak,
            rmse_reference=rmse_classic,
            iou_candidate=iou_rywak,
            iou_reference=iou_classic,
            base_penalty=1500.0,
            scale_penalty=7000.0,
        )
    return safe_score(score)


def objective_robak_no_slam(metrics: dict[str, Any]) -> float:
    rmse_no = metric(metrics, "rmse_xy_robak_no_slam")
    rmse_odom = metric(metrics, "rmse_xy_odom_topic")
    iou_no = metric(metrics, "iou_map_robak_no_slam")
    iou_odom = metric(metrics, "iou_map_odom_points")
    rmse_delta = rmse_no - rmse_odom
    iou_gap = iou_odom - iou_no
    score = 8.0 * rmse_delta + 220.0 * max(0.0, iou_gap)
    if STRICT_NO_SLAM_ODOM:
        score += hard_must_beat_odom_penalty(rmse_no, rmse_odom, iou_no, iou_odom)
    return safe_score(score)


def objective_rywak_no_slam(metrics: dict[str, Any]) -> float:
    rmse_no = metric(metrics, "rmse_xy_rywak_no_slam")
    rmse_odom = metric(metrics, "rmse_xy_odom_topic")
    iou_no = metric(metrics, "iou_map_rywak_no_slam")
    iou_odom = metric(metrics, "iou_map_odom_points")
    rmse_delta = rmse_no - rmse_odom
    iou_gap = iou_odom - iou_no
    score = 8.0 * rmse_delta + 220.0 * max(0.0, iou_gap)
    if STRICT_NO_SLAM_ODOM:
        score += hard_must_beat_odom_penalty(rmse_no, rmse_odom, iou_no, iou_odom)
    return safe_score(score)


def build_phase_specs() -> list[PhaseSpec]:
    return [
        PhaseSpec(
            name="robak_slam_translation",
            objective_name="min(robak_slam_rmse_vs_classic + iou_penalty)",
            objective=objective_robak_slam,
            params=[
                ParamSweep("robak.infer_max_step_trans", [0.05, 0.06, 0.07]),
                ParamSweep("robak.infer_odom_delta_xy_gain", [0.65, 0.75, 0.85]),
                ParamSweep("robak.odom_guard_xy_error_m", [0.24, 0.32, 0.40]),
            ],
        ),
        PhaseSpec(
            name="rywak_slam_yaw",
            objective_name="min(rywak_slam_rmse_vs_classic + iou_penalty)",
            objective=objective_rywak_slam,
            params=[
                ParamSweep("rywak.anchor_yaw_to_odom", [0.62, 0.72, 0.82]),
                ParamSweep("rywak.max_step_yaw", [0.12, 0.16, 0.20]),
                ParamSweep("rywak.odom_guard_yaw_error_rad", [0.20, 0.26, 0.34]),
            ],
        ),
        PhaseSpec(
            name="robak_no_slam",
            objective_name="min(robak_no_slam_rmse_vs_odom + iou_gap_vs_odom)",
            objective=objective_robak_no_slam,
            params=[
                ParamSweep("robak_no_slam.infer_odom_delta_xy_alpha", [0.12, 0.22, 0.32]),
                ParamSweep("robak_no_slam.infer_odom_pose_xy_alpha", [0.00, 0.01, 0.02]),
                ParamSweep("robak_no_slam.odom_guard_xy_error_m", [0.8, 1.2, 1.6]),
            ],
        ),
        PhaseSpec(
            name="rywak_no_slam",
            objective_name="min(rywak_no_slam_rmse_vs_odom + iou_gap_vs_odom)",
            objective=objective_rywak_no_slam,
            params=[
                ParamSweep("rywak_no_slam.fuse_odom_v_weight", [0.18, 0.26, 0.34]),
                ParamSweep("rywak_no_slam.anchor_yaw_to_odom", [0.25, 0.38, 0.55]),
                ParamSweep("rywak_no_slam.xy_step_odom_weight", [0.30, 0.42, 0.55]),
                ParamSweep("rywak_no_slam.odom_guard_xy_error_m", [0.9, 1.1, 1.4]),
            ],
        ),
    ]


def build_fine_phase_specs() -> list[PhaseSpec]:
    return [
        PhaseSpec(
            name="robak_slam_translation_fine120",
            objective_name="min(robak_slam_rmse_vs_classic + iou_penalty)",
            objective=objective_robak_slam,
            params=[
                ParamSweep("robak.infer_max_step_trans", [0.045, 0.055, 0.065]),
                ParamSweep("robak.infer_odom_delta_xy_gain", [0.70, 0.78, 0.86]),
                ParamSweep("robak.odom_guard_xy_error_m", [0.22, 0.28, 0.34]),
            ],
        ),
        PhaseSpec(
            name="rywak_slam_yaw_fine120",
            objective_name="min(rywak_slam_rmse_vs_classic + iou_penalty)",
            objective=objective_rywak_slam,
            params=[
                ParamSweep("rywak.anchor_yaw_to_odom", [0.64, 0.72, 0.80]),
                ParamSweep("rywak.max_step_yaw", [0.10, 0.12, 0.14]),
                ParamSweep("rywak.odom_guard_yaw_error_rad", [0.18, 0.22, 0.26]),
            ],
        ),
        PhaseSpec(
            name="robak_no_slam_fine120",
            objective_name="min(robak_no_slam_rmse_vs_odom + iou_gap_vs_odom)",
            objective=objective_robak_no_slam,
            params=[
                ParamSweep("robak_no_slam.infer_odom_delta_xy_alpha", [0.18, 0.26, 0.34]),
                ParamSweep("robak_no_slam.infer_odom_pose_xy_alpha", [0.00, 0.005, 0.01]),
                ParamSweep("robak_no_slam.odom_guard_xy_error_m", [1.0, 1.3, 1.6]),
            ],
        ),
        PhaseSpec(
            name="rywak_no_slam_fine120",
            objective_name="min(rywak_no_slam_rmse_vs_odom + iou_gap_vs_odom)",
            objective=objective_rywak_no_slam,
            params=[
                ParamSweep("rywak_no_slam.fuse_odom_v_weight", [0.12, 0.18, 0.24]),
                ParamSweep("rywak_no_slam.anchor_yaw_to_odom", [0.18, 0.25, 0.32]),
                ParamSweep("rywak_no_slam.xy_step_odom_weight", [0.26, 0.34, 0.42]),
                ParamSweep("rywak_no_slam.odom_guard_xy_error_m", [1.0, 1.2, 1.4]),
            ],
        ),
    ]


def patch_common_test_settings(cfg: dict[str, Any], dataset_source: str, eval_duration: float | None) -> None:
    set_nested(cfg, "experiment.dataset_source_experiment_id", dataset_source)
    set_nested(cfg, "tracks.tor2_ai_slam", False)
    set_nested(cfg, "tracks.tor5_robak", True)
    set_nested(cfg, "tracks.tor6_rywak", True)
    set_nested(cfg, "tracks.tor7_robak_no_slam", True)
    set_nested(cfg, "tracks.tor8_rywak_no_slam", True)
    if eval_duration is not None:
        set_nested(cfg, "pipeline.evaluation_sec", float(eval_duration))
        set_nested(cfg, "timing.eval_duration", float(eval_duration))


def run_test_cycle(config_path: Path, experiment_id: str, cuda_device: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    env["RUN_FULL_CYCLE_SKIP_LOCK"] = "1"
    cmd = [
        "bash",
        str(RUN_FULL_CYCLE),
        str(config_path),
        "phase:=test",
        f"experiment_id:={experiment_id}",
        "gui:=false",
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def persist_rows(csv_path: Path, json_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ts",
        "phase",
        "objective",
        "param",
        "candidate_value",
        "chosen_value_before",
        "score",
        "status",
        "return_code",
        "repeats_target",
        "repeats_ok",
        "repeat_experiment_ids",
        "experiment_id",
        "config_path",
        "rmse_xy_classic_slam",
        "rmse_xy_robak_slam",
        "rmse_xy_rywak_slam",
        "rmse_xy_odom_topic",
        "rmse_xy_robak_no_slam",
        "rmse_xy_rywak_no_slam",
        "iou_map_classic_slam",
        "iou_map_robak",
        "iou_map_rywak",
        "iou_map_odom_points",
        "iou_map_robak_no_slam",
        "iou_map_rywak_no_slam",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    global STRICT_NO_SLAM_ODOM, STRICT_SLAM_CLASSIC
    parser = argparse.ArgumentParser(description="Automatyczny sweep per-track (hospital test).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Bazowy config YAML.")
    parser.add_argument("--dataset-source", default="BIG_DATASET", help="Źródło modeli/datasetu.")
    parser.add_argument("--eval-duration", type=float, default=120.0, help="Czas testu (s).")
    parser.add_argument("--cuda-device", default="0", help="CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Liczba powtórzeń każdego kandydata (score liczony ze średnich metryk).",
    )
    parser.add_argument(
        "--phases",
        default="",
        help=(
            "Lista faz do uruchomienia (comma-separated), np. "
            "'robak_slam_translation,rywak_slam_yaw'. Puste = wszystkie fazy."
        ),
    )
    parser.add_argument(
        "--skip-params",
        default="",
        help=(
            "Lista param path do pominięcia (comma-separated), np. "
            "'rywak.anchor_yaw_to_odom,robak.infer_max_step_trans'."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Limit runów (0 = bez limitu).")
    parser.add_argument(
        "--include-fine-phases",
        action="store_true",
        default=False,
        help="Dodaj dodatkowe, węższe fazy strojenia (fine 120s).",
    )
    parser.add_argument(
        "--strict-no-slam-odom",
        action="store_true",
        default=True,
        help="Twarda kara, gdy no-SLAM przegrywa z odometrią (RMSE/IoU).",
    )
    parser.add_argument(
        "--no-strict-no-slam-odom",
        action="store_false",
        dest="strict_no_slam_odom",
        help="Wyłącz twardą karę no-SLAM vs odometria.",
    )
    parser.add_argument(
        "--strict-slam-classic",
        action="store_true",
        default=False,
        help="Twarda kara, gdy SLAM (robak/rywak) przegrywa z classic_slam (RMSE/IoU).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=True,
        help="Zapisz finalnie najlepsze parametry do --config.",
    )
    parser.add_argument(
        "--no-apply",
        action="store_false",
        dest="apply",
        help="Nie nadpisuj --config (tylko raport i plik final_config.yaml w sweepie).",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        print(f"[AUTO-SWEEP] --repeats musi być >=1 (podano {args.repeats})", file=sys.stderr)
        return 2
    STRICT_NO_SLAM_ODOM = bool(args.strict_no_slam_odom)
    STRICT_SLAM_CLASSIC = bool(args.strict_slam_classic)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[AUTO-SWEEP] Brak configu: {config_path}", file=sys.stderr)
        return 2
    if not RUN_FULL_CYCLE.exists():
        print(f"[AUTO-SWEEP] Brak skryptu: {RUN_FULL_CYCLE}", file=sys.stderr)
        return 2

    ensure_grouped_out_layout()
    sweep_id = f"sweep_auto_tracks_{time.strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = ensure_sweep_storage(sweep_id)
    configs_dir = sweep_dir / "configs"
    reports_dir = sweep_dir / "reports"
    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = sweep_dir / "summary.csv"
    summary_json = sweep_dir / "summary.json"

    base_cfg = load_yaml(config_path)
    patch_common_test_settings(base_cfg, args.dataset_source, args.eval_duration)
    working_cfg = copy.deepcopy(base_cfg)

    phase_specs = build_phase_specs()
    if args.include_fine_phases:
        phase_specs.extend(build_fine_phase_specs())
    if args.phases:
        wanted = [name.strip() for name in str(args.phases).split(",") if name.strip()]
        known = {phase.name for phase in phase_specs}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            print(
                f"[AUTO-SWEEP] Nieznane fazy: {unknown}. Dostępne: {sorted(known)}",
                file=sys.stderr,
            )
            return 2
        by_name = {phase.name: phase for phase in phase_specs}
        phase_specs = [by_name[name] for name in wanted]
        print(f"[AUTO-SWEEP] Filtrowanie faz (--phases): {wanted}")
    skip_params = {name.strip() for name in str(args.skip_params).split(",") if name.strip()}
    if skip_params:
        print(f"[AUTO-SWEEP] Pomijane parametry (--skip-params): {sorted(skip_params)}")
    rows: list[dict[str, Any]] = []
    applied_changes: dict[str, Any] = {}
    run_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[AUTO-SWEEP] Sweep ID: {sweep_id}")
    print(f"[AUTO-SWEEP] Config bazowy: {config_path}")
    print(f"[AUTO-SWEEP] Dataset source: {args.dataset_source}")
    print(f"[AUTO-SWEEP] Eval duration: {args.eval_duration}s")
    print(f"[AUTO-SWEEP] Repeats per candidate: {args.repeats}")
    print(f"[AUTO-SWEEP] Strict no-SLAM vs odom: {STRICT_NO_SLAM_ODOM}")
    print(f"[AUTO-SWEEP] Strict SLAM vs classic: {STRICT_SLAM_CLASSIC}")
    print(f"[AUTO-SWEEP] Include fine phases: {args.include_fine_phases}")
    print(f"[AUTO-SWEEP] CUDA_VISIBLE_DEVICES={args.cuda_device}")
    print(f"[AUTO-SWEEP] Output: {sweep_dir}")

    for phase in phase_specs:
        print(f"\n[AUTO-SWEEP] Faza: {phase.name} ({phase.objective_name})")
        for param in phase.params:
            if param.path in skip_params:
                print(f"[AUTO-SWEEP]  Pomijam parametr: {param.path}")
                continue
            current_value = get_nested(working_cfg, param.path)
            best_value = current_value
            best_score = float("inf")
            best_metrics: dict[str, Any] = {}
            any_success = False

            print(f"[AUTO-SWEEP]  Parametr: {param.path}, start={current_value}, kandydaci={param.values}")
            for candidate in param.values:
                if args.max_runs > 0 and run_counter >= args.max_runs:
                    print("[AUTO-SWEEP]  Osiągnięto --max-runs, zatrzymuję sweep.")
                    persist_rows(summary_csv, summary_json, rows)
                    final_cfg_path = sweep_dir / "final_config.yaml"
                    dump_yaml(final_cfg_path, working_cfg)
                    if args.apply:
                        dump_yaml(config_path, working_cfg)
                        print(f"[AUTO-SWEEP]  Zapisano aktualny best config do: {config_path}")
                    print(f"[AUTO-SWEEP]  Snapshot best config: {final_cfg_path}")
                    return 0

                cfg_candidate = copy.deepcopy(working_cfg)
                set_nested(cfg_candidate, param.path, candidate)
                patch_common_test_settings(cfg_candidate, args.dataset_source, args.eval_duration)

                cfg_filename = (
                    f"{(run_counter + 1):03d}_{phase.name}_{param.path.replace('.', '_')}_{fmt_value(candidate)}.yaml"
                )
                cfg_path = configs_dir / cfg_filename
                dump_yaml(cfg_path, cfg_candidate)

                repeat_metrics: list[dict[str, Any]] = []
                repeat_ids: list[str] = []
                repeat_rcs: list[int] = []
                repeat_elapsed: list[float] = []
                for repeat_idx in range(1, args.repeats + 1):
                    if args.max_runs > 0 and run_counter >= args.max_runs:
                        print("[AUTO-SWEEP]  Osiągnięto --max-runs, zatrzymuję sweep.")
                        persist_rows(summary_csv, summary_json, rows)
                        final_cfg_path = sweep_dir / "final_config.yaml"
                        dump_yaml(final_cfg_path, working_cfg)
                        if args.apply:
                            dump_yaml(config_path, working_cfg)
                            print(f"[AUTO-SWEEP]  Zapisano aktualny best config do: {config_path}")
                        print(f"[AUTO-SWEEP]  Snapshot best config: {final_cfg_path}")
                        return 0

                    run_counter += 1
                    repeat_experiment_id = f"exp_auto_tune_{timestamp}_{run_counter:03d}"
                    repeat_ids.append(repeat_experiment_id)
                    print(
                        f"[AUTO-SWEEP]    Run {run_counter} (rep {repeat_idx}/{args.repeats}): "
                        f"{param.path}={candidate} -> {repeat_experiment_id}"
                    )
                    started = time.time()
                    rc = run_test_cycle(cfg_path, repeat_experiment_id, args.cuda_device)
                    elapsed = time.time() - started
                    repeat_rcs.append(rc)
                    repeat_elapsed.append(elapsed)
                    metrics = load_metrics(repeat_experiment_id)
                    if rc == 0 and metrics:
                        repeat_metrics.append(metrics)

                metrics = (
                    aggregate_metrics(repeat_metrics)
                    if len(repeat_metrics) == args.repeats
                    else {}
                )
                score = phase.objective(metrics) if metrics else float("inf")
                status = "done" if metrics else "failed"
                first_experiment_id = repeat_ids[0] if repeat_ids else ""
                return_code = max(repeat_rcs) if repeat_rcs else 1
                elapsed = sum(repeat_elapsed)

                row = {
                    "ts": datetime.now().isoformat(),
                    "phase": phase.name,
                    "objective": phase.objective_name,
                    "param": param.path,
                    "candidate_value": candidate,
                    "chosen_value_before": current_value,
                    "score": score,
                    "status": status,
                    "return_code": return_code,
                    "repeats_target": args.repeats,
                    "repeats_ok": len(repeat_metrics),
                    "repeat_experiment_ids": ";".join(repeat_ids),
                    "elapsed_sec": round(elapsed, 3),
                    "experiment_id": first_experiment_id,
                    "config_path": str(cfg_path.resolve()),
                    "rmse_xy_classic_slam": metric(metrics, "rmse_xy_classic_slam"),
                    "rmse_xy_robak_slam": metric(metrics, "rmse_xy_robak_slam"),
                    "rmse_xy_rywak_slam": metric(metrics, "rmse_xy_rywak_slam"),
                    "rmse_xy_odom_topic": metric(metrics, "rmse_xy_odom_topic"),
                    "rmse_xy_robak_no_slam": metric(metrics, "rmse_xy_robak_no_slam"),
                    "rmse_xy_rywak_no_slam": metric(metrics, "rmse_xy_rywak_no_slam"),
                    "iou_map_classic_slam": metric(metrics, "iou_map_classic_slam"),
                    "iou_map_robak": metric(metrics, "iou_map_robak"),
                    "iou_map_rywak": metric(metrics, "iou_map_rywak"),
                    "iou_map_odom_points": metric(metrics, "iou_map_odom_points"),
                    "iou_map_robak_no_slam": metric(metrics, "iou_map_robak_no_slam"),
                    "iou_map_rywak_no_slam": metric(metrics, "iou_map_rywak_no_slam"),
                }
                rows.append(row)
                persist_rows(summary_csv, summary_json, rows)
                print(
                    f"[AUTO-SWEEP]      status={status}, rc={return_code}, "
                    f"repeats_ok={len(repeat_metrics)}/{args.repeats}, score={score:.6f}, "
                    f"elapsed={elapsed:.1f}s"
                )

                if status == "done":
                    any_success = True
                    if score < best_score:
                        best_score = score
                        best_value = candidate
                        best_metrics = metrics

            if any_success:
                set_nested(working_cfg, param.path, best_value)
                applied_changes[param.path] = best_value
                print(
                    f"[AUTO-SWEEP]  Wybrano: {param.path}={best_value} "
                    f"(score={best_score:.6f})"
                )
                if best_metrics:
                    print(
                        "[AUTO-SWEEP]    snapshot: "
                        f"rmse classic={metric(best_metrics, 'rmse_xy_classic_slam'):.4f}, "
                        f"robak_slam={metric(best_metrics, 'rmse_xy_robak_slam'):.4f}, "
                        f"rywak_slam={metric(best_metrics, 'rmse_xy_rywak_slam'):.4f}, "
                        f"iou odom={metric(best_metrics, 'iou_map_odom_points'):.4f}, "
                        f"robak_no={metric(best_metrics, 'iou_map_robak_no_slam'):.4f}, "
                        f"rywak_no={metric(best_metrics, 'iou_map_rywak_no_slam'):.4f}"
                    )
            else:
                print(f"[AUTO-SWEEP]  Brak udanych runów dla {param.path}; zostawiam {current_value}.")

    final_cfg_path = sweep_dir / "final_config.yaml"
    dump_yaml(final_cfg_path, working_cfg)
    if args.apply:
        dump_yaml(config_path, working_cfg)
        print(f"\n[AUTO-SWEEP] Zapisano finalny config do: {config_path}")
    else:
        print(f"\n[AUTO-SWEEP] --no-apply: config źródłowy bez zmian.")
    print(f"[AUTO-SWEEP] Snapshot finalnego configu: {final_cfg_path}")

    report_payload = {
        "sweep_id": sweep_id,
        "base_config": str(config_path),
        "dataset_source": args.dataset_source,
        "eval_duration": args.eval_duration,
        "repeats": args.repeats,
        "cuda_device": args.cuda_device,
        "strict_no_slam_odom": bool(STRICT_NO_SLAM_ODOM),
        "strict_slam_classic": bool(STRICT_SLAM_CLASSIC),
        "include_fine_phases": bool(args.include_fine_phases),
        "run_count": run_counter,
        "applied_changes": applied_changes,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "final_config_snapshot": str(final_cfg_path),
        "config_applied": bool(args.apply),
    }
    report_path = reports_dir / "auto_sweep_report.json"
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[AUTO-SWEEP] Raport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
