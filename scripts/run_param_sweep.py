#!/usr/bin/env python3
"""Uruchamia sweep jednego parametru na stalych datasetach lub w trybie legacy full-cycle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from out_layout import (
    OUT_DIR,
    ensure_experiment_storage,
    ensure_grouped_out_layout,
    ensure_sweep_storage,
    iter_experiment_dirs,
    resolve_experiment_dir,
    resolve_venv_site_packages,
)
from results_metric_keys import metrics_rmse_theta_odom, metrics_rmse_xy_odom


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config"
RUN_FULL_CYCLE = REPO_ROOT / "scripts" / "run_full_cycle.sh"
CLEANUP_SCRIPT = REPO_ROOT / "scripts" / "cleanup.sh"
DEFAULT_CONFIG_NAME = "experiment_config.yaml"
VENV_SITE = resolve_venv_site_packages(REPO_ROOT)

ROBAK_TRAIN_PARAM_KEYS = {
    "robak.lr",
    "robak.max_epochs",
    "robak.batch_size",
    "robak.val_ratio",
}
ROBAK_INFER_PARAM_KEYS = {
    "robak.infer_delta_ema_alpha",
    "robak.infer_odom_heading_alpha",
    "robak.infer_odom_delta_xy_alpha",
    "robak.infer_odom_delta_yaw_alpha",
    "robak.infer_odom_pose_xy_alpha",
    "robak.infer_odom_pose_xy_gain",
}
RYWAK_TRAIN_PARAM_KEYS = {
    "rywak.lr",
    "rywak.max_epochs",
    "rywak.batch_size",
    "rywak.val_ratio",
    "rywak.dropout",
    "rywak.weight_decay",
    "rywak.huber_delta",
    "rywak.input_noise_std",
    "rywak.clip_grad_norm",
    "rywak.loss_v_weight",
    "rywak.loss_w_weight",
}
RYWAK_INFER_PARAM_KEYS = {
    "rywak.delta_scan_clip",
    "rywak.fuse_odom_v_weight",
    "rywak.fuse_odom_w_weight",
    "rywak.fuse_odom_v_gain",
    "rywak.fuse_odom_w_gain",
    "rywak.vel_ema_alpha",
    "rywak.anchor_yaw_to_odom",
    "rywak.anchor_xy_to_odom",
    "rywak.anchor_xy_to_odom_gain",
}
SHARED_TRAIN_PARAM_SPECS = {
    "shared.lr": {
        "target_paths": (("robak", "lr"), ("rywak", "lr")),
        "reference_paths": (("robak", "lr"), ("rywak", "lr"), ("training", "learning_rate")),
    },
    "shared.max_epochs": {
        "target_paths": (("robak", "max_epochs"), ("rywak", "max_epochs")),
        "reference_paths": (("robak", "max_epochs"), ("rywak", "max_epochs"), ("training", "max_epochs")),
    },
    "shared.batch_size": {
        "target_paths": (("robak", "batch_size"), ("rywak", "batch_size")),
        "reference_paths": (("robak", "batch_size"), ("rywak", "batch_size"), ("training", "batch_size")),
    },
    "shared.val_ratio": {
        "target_paths": (("robak", "val_ratio"), ("rywak", "val_ratio")),
        "reference_paths": (("robak", "val_ratio"), ("rywak", "val_ratio"), ("training", "validation_ratio")),
    },
}
MAP_FILTER_PARAM_KEYS = {
    "evaluation.points_min_translation",
    "evaluation.points_min_rotation",
    "evaluation.points_min_time_gap_sec",
}


def resolve_config_path(config_name: str) -> Path:
    candidate = Path(config_name)
    if not candidate.is_absolute():
        candidate = CONFIG_DIR / config_name
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku config: {candidate}")
    return candidate


def resolve_source_experiment(exp_id: str) -> Path:
    if not exp_id:
        raise ValueError("Brak eksperymentu zrodlowego dla sweepu na stalych datasetach.")
    return resolve_experiment_dir(exp_id)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def get_nested_value(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested_value(data: dict[str, Any], path: list[str], value: Any) -> None:
    current = data
    for part in path[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[path[-1]] = value


def cast_like(reference: Any, value: float) -> Any:
    if isinstance(reference, bool):
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(round(value))
    if isinstance(reference, float):
        return float(value)
    return float(value)


def generate_values(start: float, stop: float, step: float, as_int: bool) -> list[Any]:
    if step == 0.0:
        raise ValueError("Krok sweepu nie moze byc rowny 0.")
    direction = 1.0 if stop >= start else -1.0
    if step * direction <= 0.0:
        raise ValueError("Znak kroku nie pasuje do zakresu od-do.")

    values: list[Any] = []
    current = start
    eps = abs(step) * 1e-6 + 1e-9
    while (current <= stop + eps) if direction > 0 else (current >= stop - eps):
        values.append(int(round(current)) if as_int else float(round(current, 10)))
        current += step
    return values


def sanitize_value(value: Any) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "_")


def read_metrics(exp_id: str) -> dict[str, Any]:
    if not exp_id:
        return {}
    try:
        exp_dir = resolve_experiment_dir(exp_id)
    except FileNotFoundError:
        return {}
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        return {}
    payload = read_json(results_path)
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "source_experiment_id",
        "param_path",
        "param_value",
        "status",
        "elapsed_sec",
        "experiment_id",
        "rmse_xy_robak",
        "rmse_theta_robak",
        "iou_map_robak",
        "rmse_xy_rywak",
        "rmse_theta_rywak",
        "iou_map_rywak",
        "rmse_xy_ai",
        "rmse_theta_ai",
        "iou_map_ai",
        "rmse_xy_odom_topic",
        "rmse_theta_odom_topic",
        "rmse_xy_baseline",
        "rmse_theta_baseline",
        "iou_map_baseline",
        "config_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def persist_summary_files(summary_csv: Path, summary_json: Path, rows: list[dict[str, Any]]) -> None:
    write_summary(summary_csv, rows)
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def effective_base_config(config_name: str, source_dir: Path | None) -> Path:
    source_snapshot = source_dir / "config_snapshot.yaml" if source_dir else None
    if source_snapshot and source_snapshot.exists() and config_name in {"", DEFAULT_CONFIG_NAME}:
        return source_snapshot
    return resolve_config_path(config_name or DEFAULT_CONFIG_NAME)


def base_file_names(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    robak_cfg = cfg.get("robak", {}) if isinstance(cfg.get("robak"), dict) else {}
    rywak_cfg = cfg.get("rywak", {}) if isinstance(cfg.get("rywak"), dict) else {}
    return {
        "baseline": {
            "dataset_name": "dataset.npz",
            "model_name": "model.pt",
            "history_name": "train_history.json",
        },
        "robak": {
            "dataset_name": str(robak_cfg.get("dataset_name", "dataset_robak.npz")),
            "model_name": str(robak_cfg.get("model_name", "model_robak.pt")),
            "history_name": str(robak_cfg.get("history_name", "train_history_robak.json")),
        },
        "rywak": {
            "dataset_name": str(rywak_cfg.get("dataset_name", "dataset_rywak.npz")),
            "model_name": str(rywak_cfg.get("model_name", "model_rywak.pt")),
            "history_name": str(rywak_cfg.get("history_name", "train_history_rywak.json")),
        },
    }


def source_artifact_paths(source_dir: Path, fallback_cfg: dict[str, Any]) -> dict[str, Path]:
    source_results = read_json(source_dir / "results.json")
    artifacts = source_results.get("artifacts", {}) if isinstance(source_results.get("artifacts"), dict) else {}
    snapshot_path = Path(str(artifacts.get("config_snapshot_yaml", source_dir / "config_snapshot.yaml")))
    source_cfg = load_yaml(snapshot_path) if snapshot_path.exists() else fallback_cfg
    names = base_file_names(source_cfg)

    return {
        "baseline_dataset": Path(str(artifacts.get("dataset_npz", source_dir / names["baseline"]["dataset_name"]))),
        "baseline_model": Path(str(artifacts.get("model_pt", source_dir / names["baseline"]["model_name"]))),
        "baseline_history": Path(str(artifacts.get("train_history_json", source_dir / names["baseline"]["history_name"]))),
        "robak_dataset": Path(str(artifacts.get("robak_dataset_npz", source_dir / names["robak"]["dataset_name"]))),
        "robak_model": Path(str(artifacts.get("robak_model_pt", source_dir / names["robak"]["model_name"]))),
        "robak_history": Path(str(artifacts.get("robak_train_history_json", source_dir / names["robak"]["history_name"]))),
        "rywak_dataset": Path(str(artifacts.get("rywak_dataset_npz", source_dir / names["rywak"]["dataset_name"]))),
        "rywak_model": Path(str(artifacts.get("rywak_model_pt", source_dir / names["rywak"]["model_name"]))),
        "rywak_history": Path(str(artifacts.get("rywak_train_history_json", source_dir / names["rywak"]["history_name"]))),
        "config_snapshot": snapshot_path,
        "metadata": source_dir / "experiment_metadata.json",
    }


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def classify_fixed_dataset_param(param_key: str) -> tuple[str, set[str]]:
    if param_key in SHARED_TRAIN_PARAM_SPECS:
        return "shared", {"robak", "rywak"}
    if param_key in ROBAK_TRAIN_PARAM_KEYS:
        return "robak", {"robak"}
    if param_key in ROBAK_INFER_PARAM_KEYS:
        return "robak", set()
    if param_key in RYWAK_TRAIN_PARAM_KEYS:
        return "rywak", {"rywak"}
    if param_key in RYWAK_INFER_PARAM_KEYS:
        return "rywak", set()
    if param_key in MAP_FILTER_PARAM_KEYS:
        return "map_filter", set()

    if param_key.startswith("robak."):
        raise ValueError(
            f"Parametr {param_key} dotyczy budowy datasetu Robaka. "
            "Dla sweepu na stalych datasetach wybierz parametr treningu albo inferencji."
        )
    if param_key.startswith("rywak."):
        raise ValueError(
            f"Parametr {param_key} dotyczy budowy datasetu Rywaka. "
            "Dla sweepu na stalych datasetach wybierz parametr treningu albo inferencji."
        )
    if param_key.startswith("evaluation."):
        raise ValueError(
            f"Parametr {param_key} nie jest obslugiwany przez sweep na stalych datasetach."
        )
    if param_key.startswith("shared."):
        supported = ", ".join(sorted(SHARED_TRAIN_PARAM_SPECS))
        raise ValueError(
            f"Parametr {param_key} nie jest obslugiwanym wspolnym sweepem. "
            f"Dostepne wspolne parametry: {supported}."
        )
    raise ValueError(f"Nieobslugiwany parametr sweepu: {param_key}")


def resolve_fixed_dataset_param_targets(
    param_key: str,
    base_config: dict[str, Any],
) -> tuple[str, set[str], list[list[str]], Any]:
    group, train_models = classify_fixed_dataset_param(param_key)
    if param_key in SHARED_TRAIN_PARAM_SPECS:
        spec = SHARED_TRAIN_PARAM_SPECS[param_key]
        target_paths = [list(path) for path in spec["target_paths"]]
        reference_paths = [list(path) for path in spec["reference_paths"]]
        reference_value = None
        for path in reference_paths:
            candidate = get_nested_value(base_config, path)
            if candidate is not None:
                reference_value = candidate
                break
        if reference_value is None:
            raise ValueError(
                f"Brak referencyjnej wartosci dla wspolnego sweepu {param_key} w configu bazowym."
            )
        return group, train_models, target_paths, reference_value
    else:
        target_paths = [[part.strip() for part in param_key.split(".") if part.strip()]]

    reference_values = [get_nested_value(base_config, path) for path in target_paths]
    missing_paths = [".".join(path) for path, value in zip(target_paths, reference_values) if value is None]
    if missing_paths:
        raise ValueError(
            f"Brak parametrow {', '.join(missing_paths)} w configu bazowym."
        )

    return group, train_models, target_paths, reference_values[0]


def prepare_target_metadata(
    source_dir: Path,
    target_dir: Path,
    target_experiment_id: str,
    source_experiment_id: str,
    param_key: str,
    param_value: Any,
    base_config_path: Path,
) -> None:
    source_meta = read_json(source_dir / "experiment_metadata.json")
    now_iso = datetime.now().isoformat()
    notes = list(source_meta.get("notes", [])) if isinstance(source_meta.get("notes"), list) else []
    notes.append(
        f"[{now_iso}] Sweep fixed_dataset: source={source_experiment_id}, "
        f"param={param_key}, value={param_value}, base_config={base_config_path}"
    )

    metadata = {
        "experiment_id": target_experiment_id,
        "created_at": now_iso,
        "system_info": source_meta.get("system_info", {}),
        "dataset": source_meta.get("dataset", {}),
        "training": source_meta.get("training", {}),
        "inference": {},
        "evaluation": {},
        "total_experiment_time_sec": None,
        "notes": notes,
        "sweep": {
            "mode": "fixed_dataset",
            "source_experiment_id": source_experiment_id,
            "param_path": param_key,
            "param_value": param_value,
            "base_config_path": str(base_config_path.resolve()),
        },
    }
    (target_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_target_experiment(
    source_dir: Path,
    target_dir: Path,
    cfg: dict[str, Any],
    source_experiment_id: str,
    target_experiment_id: str,
    param_key: str,
    param_value: Any,
    base_config_path: Path,
    train_models: set[str],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_names = base_file_names(cfg)
    source_paths = source_artifact_paths(source_dir, cfg)

    copy_if_exists(source_paths["baseline_dataset"], target_dir / target_names["baseline"]["dataset_name"])
    copy_if_exists(source_paths["robak_dataset"], target_dir / target_names["robak"]["dataset_name"])
    copy_if_exists(source_paths["rywak_dataset"], target_dir / target_names["rywak"]["dataset_name"])

    if "robak" not in train_models:
        copy_if_exists(source_paths["robak_model"], target_dir / target_names["robak"]["model_name"])
        copy_if_exists(source_paths["robak_history"], target_dir / target_names["robak"]["history_name"])
    if "rywak" not in train_models:
        copy_if_exists(source_paths["rywak_model"], target_dir / target_names["rywak"]["model_name"])
        copy_if_exists(source_paths["rywak_history"], target_dir / target_names["rywak"]["history_name"])

    copy_if_exists(source_paths["baseline_model"], target_dir / target_names["baseline"]["model_name"])
    copy_if_exists(source_paths["baseline_history"], target_dir / target_names["baseline"]["history_name"])
    prepare_target_metadata(
        source_dir=source_dir,
        target_dir=target_dir,
        target_experiment_id=target_experiment_id,
        source_experiment_id=source_experiment_id,
        param_key=param_key,
        param_value=param_value,
        base_config_path=base_config_path,
    )


def shell_preamble() -> str:
    parts = [
        "source /opt/ros/jazzy/setup.bash",
        "if [ -f ai_slam_ws/install/setup.bash ]; then source ai_slam_ws/install/setup.bash; fi",
    ]
    if VENV_SITE is not None and VENV_SITE.is_dir():
        parts.append(f"export PYTHONPATH=\"${{PYTHONPATH:+$PYTHONPATH:}}{VENV_SITE}\"")
    parts.append(
        "if grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then "
        "export FASTDDS_BUILTIN_TRANSPORTS=\"${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}\"; fi"
    )
    return " && ".join(parts)


def run_shell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=REPO_ROOT,
        text=True,
    )


def make_ros_params_file(path: Path, params: dict[str, Any]) -> None:
    payload = {"/**": {"ros__parameters": params}}
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def common_config_values(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    timing_cfg = cfg.get("timing", {}) if isinstance(cfg.get("timing"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    return {
        "seed": int(experiment_cfg.get("seed", 123)),
        "dataset_wait_timeout": float(timing_cfg.get("dataset_wait_timeout", 0.0)),
        "training_max_epochs": int(training_cfg.get("max_epochs", 200)),
        "training_patience": int(training_cfg.get("patience", 20)),
        "training_min_delta": float(training_cfg.get("min_delta", 1e-5)),
        "training_lr": float(training_cfg.get("learning_rate", 0.001)),
        "training_batch_size": int(training_cfg.get("batch_size", 128)),
        "training_val_ratio": float(training_cfg.get("validation_ratio", 0.2)),
    }


def apply_eval_duration_override(cfg: dict[str, Any], eval_duration: float | None) -> None:
    if eval_duration is None:
        return
    set_nested_value(cfg, ["pipeline", "evaluation_sec"], float(eval_duration))
    set_nested_value(cfg, ["timing", "eval_duration"], float(eval_duration))


def build_trainer_params(model_type: str, cfg: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    common = common_config_values(cfg)
    out_dir = str(OUT_DIR.resolve())

    if model_type == "robak":
        robak_cfg = cfg.get("robak", {}) if isinstance(cfg.get("robak"), dict) else {}
        return {
            "seed": common["seed"],
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "dataset_name": str(robak_cfg.get("dataset_name", "dataset_robak.npz")),
            "model_name": str(robak_cfg.get("model_name", "model_robak.pt")),
            "history_name": str(robak_cfg.get("history_name", "train_history_robak.json")),
            "skip_if_model_exists": False,
            "dataset_wait_timeout": common["dataset_wait_timeout"],
            "max_epochs": int(robak_cfg.get("max_epochs", common["training_max_epochs"])),
            "patience": int(robak_cfg.get("patience", common["training_patience"])),
            "min_delta": common["training_min_delta"],
            "lr": float(robak_cfg.get("lr", common["training_lr"])),
            "batch_size": int(robak_cfg.get("batch_size", common["training_batch_size"])),
            "val_ratio": float(robak_cfg.get("val_ratio", common["training_val_ratio"])),
            "write_experiment_metadata": True,
        }

    if model_type == "rywak":
        rywak_cfg = cfg.get("rywak", {}) if isinstance(cfg.get("rywak"), dict) else {}
        return {
            "seed": common["seed"],
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "dataset_name": str(rywak_cfg.get("dataset_name", "dataset_rywak.npz")),
            "model_name": str(rywak_cfg.get("model_name", "model_rywak.pt")),
            "history_name": str(rywak_cfg.get("history_name", "train_history_rywak.json")),
            "skip_if_model_exists": False,
            "dataset_wait_timeout": common["dataset_wait_timeout"],
            "max_epochs": int(rywak_cfg.get("max_epochs", common["training_max_epochs"])),
            "patience": int(rywak_cfg.get("patience", common["training_patience"])),
            "min_delta": common["training_min_delta"],
            "lr": float(rywak_cfg.get("lr", common["training_lr"])),
            "batch_size": int(rywak_cfg.get("batch_size", common["training_batch_size"])),
            "val_ratio": float(rywak_cfg.get("val_ratio", common["training_val_ratio"])),
            "hidden_dims": list(rywak_cfg.get("hidden_dims", [192, 96, 48])),
            "dropout": float(rywak_cfg.get("dropout", 0.1)),
            "weight_decay": float(rywak_cfg.get("weight_decay", 1e-4)),
            "huber_delta": float(rywak_cfg.get("huber_delta", 1.0)),
            "input_noise_std": float(rywak_cfg.get("input_noise_std", 0.02)),
            "clip_grad_norm": float(rywak_cfg.get("clip_grad_norm", 1.0)),
            "loss_v_weight": float(rywak_cfg.get("loss_v_weight", 1.0)),
            "loss_w_weight": float(rywak_cfg.get("loss_w_weight", 1.5)),
            "write_experiment_metadata": True,
        }

    raise ValueError(f"Nieznany model do trenowania: {model_type}")


def run_trainer(model_type: str, cfg: dict[str, Any], experiment_id: str, params_dir: Path, logs_dir: Path | None = None) -> tuple[int, Path | None]:
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / f"{model_type}_{experiment_id}.yaml"
    make_ros_params_file(params_path, build_trainer_params(model_type, cfg, experiment_id))
    executable = "train_model_robak" if model_type == "robak" else "train_model_rywak"
    log_path: Path | None = None
    command = f"{shell_preamble()} && ros2 run ai_slam_ai {executable} --ros-args --params-file {params_path}"
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{model_type}_{experiment_id}.log"
        command = f"{command} > {log_path} 2>&1"
    return run_shell(command).returncode, log_path


def cleanup_environment() -> None:
    if CLEANUP_SCRIPT.exists():
        run_shell(f"{CLEANUP_SCRIPT} || true")


def run_test_phase(cfg_path: Path, experiment_id: str) -> int:
    cleanup_environment()
    command = (
        f"{shell_preamble()} && ros2 launch ai_slam_bringup demo.launch.py"
        f" config:={cfg_path}"
        f" phase:=test"
        f" experiment_id:={experiment_id}"
    )
    completed = run_shell(command)
    cleanup_environment()
    return int(completed.returncode)


def discover_new_experiment(before: set[str], after: set[str]) -> str:
    new_items = sorted(after - before)
    if new_items:
        return new_items[-1]
    return ""


def run_full_cycle_sweep(args: argparse.Namespace) -> int:
    if not RUN_FULL_CYCLE.exists():
        raise FileNotFoundError(f"Brak skryptu: {RUN_FULL_CYCLE}")

    ensure_grouped_out_layout()
    config_path = resolve_config_path(args.config)
    base_config = load_yaml(config_path)
    param_path = [part.strip() for part in str(args.param).split(".") if part.strip()]
    if not param_path:
        raise ValueError("Niepoprawna sciezka parametru.")

    original_value = get_nested_value(base_config, param_path)
    as_int = isinstance(original_value, int) and not isinstance(original_value, bool)
    values = generate_values(args.start, args.stop, args.step, as_int=as_int)
    if not values:
        raise ValueError("Zakres sweepu nie wygenerowal zadnej wartosci.")

    sweep_id = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}_{'_'.join(param_path)}"
    sweep_dir = ensure_sweep_storage(sweep_id)
    config_out_dir = sweep_dir / "configs"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = sweep_dir / "summary.csv"
    summary_json = sweep_dir / "summary.json"
    persist_summary_files(summary_csv, summary_json, [])

    print(f"[SWEEP] Tryb legacy full-cycle.")
    print(f"[SWEEP] Config bazowy: {config_path}")
    print(f"[SWEEP] Parametr: {'.'.join(param_path)}")
    print(f"[SWEEP] Wartosci: {values}")
    if args.eval_duration is not None:
        print(f"[SWEEP] Nadpisany czas testu i ewaluacji: {args.eval_duration}s")
    print(f"[SWEEP] Katalog wynikow sweepu: {sweep_dir}")

    rows: list[dict[str, Any]] = []
    had_failures = False
    for index, raw_value in enumerate(values, start=1):
        cfg = load_yaml(config_path)
        casted_value = cast_like(original_value, float(raw_value)) if original_value is not None else raw_value
        set_nested_value(cfg, param_path, casted_value)
        apply_eval_duration_override(cfg, args.eval_duration)

        temp_config = config_out_dir / f"{index:02d}_{sanitize_value(casted_value)}.yaml"
        temp_config.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

        before = {path.name for path in iter_experiment_dirs()}
        started_at = time.time()
        print(f"\n[SWEEP] [{index}/{len(values)}] {'.'.join(param_path)} = {casted_value}")
        completed = subprocess.run([str(RUN_FULL_CYCLE), str(temp_config)], cwd=REPO_ROOT)
        elapsed = time.time() - started_at
        after = {path.name for path in iter_experiment_dirs()}
        exp_id = discover_new_experiment(before, after)
        metrics = read_metrics(exp_id)
        odom_xy = metrics_rmse_xy_odom(metrics)
        odom_th = metrics_rmse_theta_odom(metrics)
        status = "done" if completed.returncode == 0 else f"failed({completed.returncode})"
        if completed.returncode != 0:
            had_failures = True
        rows.append(
            {
                "mode": "full_cycle",
                "source_experiment_id": "",
                "param_path": ".".join(param_path),
                "param_value": casted_value,
                "status": status,
                "elapsed_sec": round(elapsed, 3),
                "experiment_id": exp_id,
                "config_path": str(temp_config.resolve()),
                "rmse_xy_robak": metrics.get("rmse_xy_robak"),
                "rmse_theta_robak": metrics.get("rmse_theta_robak"),
                "iou_map_robak": metrics.get("iou_map_robak"),
                "rmse_xy_rywak": metrics.get("rmse_xy_rywak"),
                "rmse_theta_rywak": metrics.get("rmse_theta_rywak"),
                "iou_map_rywak": metrics.get("iou_map_rywak"),
                "rmse_xy_ai": metrics.get("rmse_xy_ai"),
                "rmse_theta_ai": metrics.get("rmse_theta_ai"),
                "iou_map_ai": metrics.get("iou_map_ai"),
                "rmse_xy_odom_topic": odom_xy,
                "rmse_theta_odom_topic": odom_th,
                "rmse_xy_baseline": odom_xy,
                "rmse_theta_baseline": odom_th,
                "iou_map_baseline": metrics.get("iou_map_baseline"),
            }
        )
        persist_summary_files(summary_csv, summary_json, rows)

    print(f"\n[SWEEP] Zapisano podsumowanie CSV: {summary_csv}")
    print(f"[SWEEP] Zapisano podsumowanie JSON: {summary_json}")
    return 1 if had_failures else 0


def run_fixed_dataset_sweep(args: argparse.Namespace) -> int:
    ensure_grouped_out_layout()
    source_dir = resolve_source_experiment(args.source_experiment)
    base_config_path = effective_base_config(args.config, source_dir)
    base_config = load_yaml(base_config_path)
    param_path = [part.strip() for part in str(args.param).split(".") if part.strip()]
    if not param_path:
        raise ValueError("Niepoprawna sciezka parametru.")
    param_key = ".".join(param_path)
    _group, train_models, target_paths, original_value = resolve_fixed_dataset_param_targets(param_key, base_config)

    as_int = isinstance(original_value, int) and not isinstance(original_value, bool)
    values = generate_values(args.start, args.stop, args.step, as_int=as_int)
    if not values:
        raise ValueError("Zakres sweepu nie wygenerowal zadnej wartosci.")

    sweep_id = f"sweep_fixed_{time.strftime('%Y%m%d_%H%M%S')}_{'_'.join(param_path)}"
    sweep_dir = ensure_sweep_storage(sweep_id)
    config_out_dir = sweep_dir / "configs"
    params_out_dir = sweep_dir / "train_params"
    logs_out_dir = sweep_dir / "logs"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    params_out_dir.mkdir(parents=True, exist_ok=True)
    logs_out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = sweep_dir / "summary.csv"
    summary_json = sweep_dir / "summary.json"
    persist_summary_files(summary_csv, summary_json, [])

    print(f"[SWEEP] Tryb: fixed_dataset")
    print(f"[SWEEP] Eksperyment zrodlowy: {source_dir.name}")
    print(f"[SWEEP] Config bazowy: {base_config_path}")
    print(f"[SWEEP] Parametr: {param_key}")
    print(f"[SWEEP] Wartosci: {values}")
    print(f"[SWEEP] Trenowane modele: {sorted(train_models) if train_models else ['brak retreningu']}")
    if args.eval_duration is not None:
        print(f"[SWEEP] Nadpisany czas testu i ewaluacji: {args.eval_duration}s")
    print(f"[SWEEP] Katalog wynikow sweepu: {sweep_dir}")

    rows: list[dict[str, Any]] = []
    had_failures = False
    for index, raw_value in enumerate(values, start=1):
        cfg = load_yaml(base_config_path)
        casted_value = cast_like(original_value, float(raw_value)) if original_value is not None else raw_value
        for target_path in target_paths:
            set_nested_value(cfg, target_path, casted_value)
        apply_eval_duration_override(cfg, args.eval_duration)

        temp_config = config_out_dir / f"{index:02d}_{sanitize_value(casted_value)}.yaml"
        temp_config.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

        exp_id = f"exp_sweep_{time.strftime('%Y%m%d_%H%M%S')}_{index:02d}"
        target_dir = ensure_experiment_storage(exp_id)
        started_at = time.time()
        status = "done"
        trainer_log_paths: dict[str, str] = {}

        print(f"\n[SWEEP] [{index}/{len(values)}] {param_key} = {casted_value} -> {exp_id}")
        try:
            prepare_target_experiment(
                source_dir=source_dir,
                target_dir=target_dir,
                cfg=cfg,
                source_experiment_id=source_dir.name,
                target_experiment_id=exp_id,
                param_key=param_key,
                param_value=casted_value,
                base_config_path=base_config_path,
                train_models=train_models,
            )

            train_rc = 0
            for model_type in sorted(train_models):
                print(f"[SWEEP]   trening modelu: {model_type}")
                train_rc, trainer_log_path = run_trainer(model_type, cfg, exp_id, params_out_dir, logs_out_dir)
                if trainer_log_path is not None:
                    trainer_log_paths[model_type] = str(trainer_log_path.resolve())
                if train_rc != 0:
                    status = f"failed_train_{model_type}({train_rc})"
                    had_failures = True
                    if trainer_log_path is not None:
                        print(f"[SWEEP]   log bledu: {trainer_log_path}")
                    break

            if status == "done":
                print("[SWEEP]   test / ewaluacja")
                test_rc = run_test_phase(temp_config.resolve(), exp_id)
                if test_rc != 0:
                    status = f"failed_test({test_rc})"
                    had_failures = True
                elif not (target_dir / "results.json").exists():
                    status = "failed_results_missing"
                    had_failures = True

        except Exception as exc:
            status = f"failed_prepare({exc})"
            had_failures = True
            print(f"[SWEEP]   BLAD przygotowania: {exc}", file=sys.stderr)

        elapsed = time.time() - started_at
        metrics = read_metrics(exp_id)
        odom_xy = metrics_rmse_xy_odom(metrics)
        odom_th = metrics_rmse_theta_odom(metrics)
        rows.append(
            {
                "mode": "fixed_dataset",
                "source_experiment_id": source_dir.name,
                "param_path": param_key,
                "param_value": casted_value,
                "status": status,
                "elapsed_sec": round(elapsed, 3),
                "experiment_id": exp_id,
                "config_path": str(temp_config.resolve()),
                "trainer_logs": trainer_log_paths,
                "rmse_xy_robak": metrics.get("rmse_xy_robak"),
                "rmse_theta_robak": metrics.get("rmse_theta_robak"),
                "iou_map_robak": metrics.get("iou_map_robak"),
                "rmse_xy_rywak": metrics.get("rmse_xy_rywak"),
                "rmse_theta_rywak": metrics.get("rmse_theta_rywak"),
                "iou_map_rywak": metrics.get("iou_map_rywak"),
                "rmse_xy_ai": metrics.get("rmse_xy_ai"),
                "rmse_theta_ai": metrics.get("rmse_theta_ai"),
                "iou_map_ai": metrics.get("iou_map_ai"),
                "rmse_xy_odom_topic": odom_xy,
                "rmse_theta_odom_topic": odom_th,
                "rmse_xy_baseline": odom_xy,
                "rmse_theta_baseline": odom_th,
                "iou_map_baseline": metrics.get("iou_map_baseline"),
            }
        )
        persist_summary_files(summary_csv, summary_json, rows)
        print(f"[SWEEP]   status={status}")

    print(f"\n[SWEEP] Zapisano podsumowanie CSV: {summary_csv}")
    print(f"[SWEEP] Zapisano podsumowanie JSON: {summary_json}")
    return 1 if had_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep jednego parametru dla eksperymentow AI SLAM.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="Bazowy plik YAML lub fallback config.")
    parser.add_argument("--source-experiment", default="", help="Eksperyment zrodlowy dla sweepu na stalych datasetach.")
    parser.add_argument("--param", required=True, help="Sciezka parametru, np. rywak.dropout")
    parser.add_argument("--start", required=True, type=float, help="Wartosc poczatkowa.")
    parser.add_argument("--stop", required=True, type=float, help="Wartosc koncowa.")
    parser.add_argument("--step", required=True, type=float, help="Krok sweepu.")
    parser.add_argument(
        "--eval-duration",
        type=float,
        default=None,
        help="Nadpisanie pipeline.evaluation_sec (i timing.eval_duration) dla każdej iteracji sweepa.",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed_dataset", "full_cycle"),
        default="fixed_dataset",
        help="fixed_dataset = ten sam dataset, full_cycle = legacy pelny pipeline.",
    )
    args = parser.parse_args()

    if args.mode == "full_cycle":
        return run_full_cycle_sweep(args)
    return run_fixed_dataset_sweep(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[SWEEP] BLAD: {exc}", file=sys.stderr)
        raise
