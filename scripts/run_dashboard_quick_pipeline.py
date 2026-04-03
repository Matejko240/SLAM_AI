#!/usr/bin/env python3
"""Tworzy tymczasowy config z nadpisanymi czasami i uruchamia szybki pipeline z dashboardu."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from out_layout import (
    DASHBOARD_QUICK_CONFIG_DIR,
    ensure_experiment_storage,
    ensure_grouped_out_layout,
    resolve_experiment_dir,
    resolve_venv_site_packages,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config"
TEMP_CONFIG_DIR = DASHBOARD_QUICK_CONFIG_DIR
VENV_SITE = resolve_venv_site_packages(REPO_ROOT)
ROS_DISTRO = os.environ.get("ROS_DISTRO", "jazzy")
ROS_SETUP = Path(f"/opt/ros/{ROS_DISTRO}/setup.bash")
WS_SETUP = REPO_ROOT / "ai_slam_ws" / "install" / "setup.bash"
WORLD_REFERENCE_MAPS = {
    "world_house.sdf": "reference_map.yaml",
    "world_office.sdf": "reference_map_office.yaml",
    "world_hospital.sdf": "reference_map_hospital.yaml",
}


def resolve_config_path(raw_value: str) -> Path:
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = CONFIG_DIR / candidate
    candidate = candidate.resolve()
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Nie znaleziono configu: {raw_value}")
    return candidate


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def get_cfg_value(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return default if cursor is None else cursor


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def set_nested(payload: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = payload
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    normalized = normalized.strip("_")
    return normalized[:48]


def reference_map_for_world(world_name: str) -> str:
    return WORLD_REFERENCE_MAPS.get(str(world_name).strip(), "reference_map.yaml")


def build_experiment_id(name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(name)
    if slug:
        return f"exp_{slug}_{timestamp}"
    return f"exp_{timestamp}"


def create_temp_config(
    base_config_path: Path,
    dataset_duration: float | None,
    eval_duration: float | None,
    experiment_name: str,
    dataset_world: str | None = None,
    test_world: str | None = None,
    forced_experiment_id: str | None = None,
) -> tuple[Path, str]:
    payload = load_yaml(base_config_path)
    experiment_id = forced_experiment_id or build_experiment_id(experiment_name)

    if dataset_duration is not None:
        set_nested(payload, ["pipeline", "dataset_collection_sec"], float(dataset_duration))
        set_nested(payload, ["timing", "dataset_duration"], float(dataset_duration))
        set_nested(payload, ["dataset", "max_samples"], 0)
        set_nested(payload, ["timing", "dataset_wait_timeout"], float(dataset_duration) * 2.0)
    if eval_duration is not None:
        set_nested(payload, ["pipeline", "evaluation_sec"], float(eval_duration))
        set_nested(payload, ["timing", "eval_duration"], float(eval_duration))
    if dataset_world:
        set_nested(payload, ["simulation", "train_world"], str(dataset_world))
    if test_world:
        set_nested(payload, ["simulation", "test_world"], str(test_world))
        set_nested(payload, ["evaluation", "reference_map_yaml"], reference_map_for_world(str(test_world)))

    evaluation_cfg = payload.get("evaluation")
    if isinstance(evaluation_cfg, dict):
        evaluation_cfg.pop("test_scenarios", None)

    set_nested(payload, ["dashboard_quick_launch", "base_config"], str(base_config_path))
    if dataset_duration is not None:
        set_nested(payload, ["dashboard_quick_launch", "dataset_duration_common"], float(dataset_duration))
    if eval_duration is not None:
        set_nested(payload, ["dashboard_quick_launch", "eval_duration"], float(eval_duration))
    if experiment_name.strip():
        set_nested(payload, ["dashboard_quick_launch", "requested_name"], experiment_name.strip())
    if dataset_world:
        set_nested(payload, ["dashboard_quick_launch", "dataset_world"], str(dataset_world))
    if test_world:
        set_nested(payload, ["dashboard_quick_launch", "test_world"], str(test_world))
    set_nested(payload, ["dashboard_quick_launch", "generated_experiment_id"], experiment_id)

    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temp_name = f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(experiment_name) or 'run'}.yaml"
    temp_path = TEMP_CONFIG_DIR / temp_name
    save_yaml(temp_path, payload)
    return temp_path, experiment_id


def run_command(command: list[str]) -> int:
    print("[QUICK] Uruchamiam:")
    print("[QUICK]   " + " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def format_ros_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_ros_run_command(executable: str, params: list[tuple[str, Any]]) -> str:
    param_args = " ".join(
        f"-p {name}:={shlex.quote(format_ros_value(value))}"
        for name, value in params
        if value is not None
    )
    parts = [
        f"if [ ! -f {shlex.quote(str(ROS_SETUP))} ]; then echo '[QUICK][ERROR] Missing ROS 2 setup: {ROS_SETUP}' >&2; exit 1; fi",
        f"source {shlex.quote(str(ROS_SETUP))}",
        f"if [ ! -f {shlex.quote(str(WS_SETUP))} ]; then echo '[QUICK][ERROR] Workspace is not built yet: {WS_SETUP}' >&2; exit 1; fi",
        f"source {shlex.quote(str(WS_SETUP))}",
        'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
    ]
    if VENV_SITE is not None and VENV_SITE.is_dir():
        parts.append(f"export PYTHONPATH=\"${{PYTHONPATH:+$PYTHONPATH:}}{VENV_SITE}\"")
    parts.append(f"ros2 run ai_slam_ai {executable} --ros-args {param_args}")
    return " && ".join(parts)


def run_shell_command(command: str) -> int:
    print("[QUICK] Uruchamiam:")
    print("[QUICK]   " + command)
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def run_best_effort_cleanup(reason: str) -> None:
    print(f"[QUICK] Cleanup: {reason}")
    return_code = run_command(["bash", "./scripts/cleanup.sh"])
    if return_code != 0:
        print(f"[QUICK][WARN] cleanup.sh zakończył się kodem {return_code}; kontynuuję.")


def ensure_runtime_ready() -> None:
    missing: list[str] = []
    if not ROS_SETUP.is_file():
        missing.append(f"- Missing ROS 2 setup: {ROS_SETUP}")
    if not WS_SETUP.is_file():
        missing.append(f"- Workspace is not built yet: {WS_SETUP}")
    if not missing:
        return

    details = "\n".join(missing)
    raise RuntimeError(
        "Brakuje środowiska ROS 2 potrzebnego do quick pipeline.\n"
        f"{details}\n\n"
        "Najpierw uruchom:\n"
        "  ./scripts/install_deps.sh\n"
        f"  cd {REPO_ROOT / 'ai_slam_ws'}\n"
        f"  source {ROS_SETUP}\n"
        "  colcon build --symlink-install\n"
    )


def dataset_artifacts_complete(config_path: Path, experiment_id: str) -> bool:
    cfg = load_yaml(config_path)
    exp_dir = resolve_experiment_dir(experiment_id)

    expected_files: list[str] = []
    mode = str(get_cfg_value(cfg, "experiment", "mode", default="ai")).lower()
    if mode == "ai":
        expected_files.extend(["dataset.npz", "experiment_metadata.json"])

    tracks_cfg = get_cfg_value(cfg, "tracks", default={})
    if parse_bool(get_cfg_value(tracks_cfg, "tor5_robak", default=False), default=False):
        expected_files.append(str(get_cfg_value(cfg, "robak", "dataset_name", default="dataset_robak.npz")))
    if parse_bool(get_cfg_value(tracks_cfg, "tor6_rywak", default=False), default=False):
        expected_files.append(str(get_cfg_value(cfg, "rywak", "dataset_name", default="dataset_rywak.npz")))

    if not expected_files:
        return False

    return all((exp_dir / name).is_file() and (exp_dir / name).stat().st_size > 0 for name in expected_files)


def training_artifacts_complete(config_path: Path, experiment_id: str) -> bool:
    cfg = load_yaml(config_path)
    exp_dir = resolve_experiment_dir(experiment_id)

    expected_files: list[str] = []
    mode = str(get_cfg_value(cfg, "experiment", "mode", default="ai")).lower()
    if mode == "ai":
        expected_files.extend(["model.pt", "train_history.json"])

    tracks_cfg = get_cfg_value(cfg, "tracks", default={})
    if parse_bool(get_cfg_value(tracks_cfg, "tor5_robak", default=False), default=False):
        expected_files.extend(
            [
                str(get_cfg_value(cfg, "robak", "model_name", default="model_robak.pt")),
                str(get_cfg_value(cfg, "robak", "history_name", default="train_history_robak.json")),
            ]
        )
    if parse_bool(get_cfg_value(tracks_cfg, "tor6_rywak", default=False), default=False):
        expected_files.extend(
            [
                str(get_cfg_value(cfg, "rywak", "model_name", default="model_rywak.pt")),
                str(get_cfg_value(cfg, "rywak", "history_name", default="train_history_rywak.json")),
            ]
        )

    if not expected_files:
        return False

    return all((exp_dir / name).is_file() and (exp_dir / name).stat().st_size > 0 for name in expected_files)


def test_artifacts_complete(experiment_id: str) -> bool:
    exp_dir = resolve_experiment_dir(experiment_id)
    results_path = exp_dir / "results.json"
    return results_path.is_file() and results_path.stat().st_size > 0


def quick_artifacts_complete(config_path: Path, experiment_id: str, phases: set[str]) -> bool:
    checks: list[bool] = []
    if "dataset" in phases:
        checks.append(dataset_artifacts_complete(config_path, experiment_id))
    if "train" in phases:
        checks.append(training_artifacts_complete(config_path, experiment_id))
    if "test" in phases:
        checks.append(test_artifacts_complete(experiment_id))
    return bool(checks) and all(checks)


def run_launch_phase(phase: str, config_path: Path, experiment_id: str, success_phases: set[str]) -> int:
    ensure_experiment_storage(experiment_id)
    return_code = run_command(
        [
            "bash",
            "./scripts/run_experiment.sh",
            phase,
            f"config:={config_path}",
            f"experiment_id:={experiment_id}",
        ]
    )
    if return_code != 0 and quick_artifacts_complete(config_path, experiment_id, success_phases):
        print(
            "[QUICK][WARN] Requested artifacts were saved successfully; "
            "ignoring non-zero shutdown return code."
        )
        return 0
    return return_code


def run_full_cycle_quick(config_path: Path, experiment_id: str) -> int:
    ensure_experiment_storage(experiment_id)
    print("[QUICK] Tryb sekwencyjny: dataset -> train_existing -> test")
    run_best_effort_cleanup("przed fazą dataset")

    dataset_rc = run_launch_phase("dataset", config_path, experiment_id, {"dataset"})
    if dataset_rc != 0:
        run_best_effort_cleanup("po nieudanej fazie dataset")
        return dataset_rc

    train_rc = train_existing_experiment(config_path, experiment_id)
    if train_rc != 0:
        run_best_effort_cleanup("po nieudanej fazie treningu")
        return train_rc

    run_best_effort_cleanup("przed fazą test")
    test_rc = run_launch_phase("test", config_path, experiment_id, {"test"})
    run_best_effort_cleanup("po zakończeniu full_cycle")
    return test_rc


def run_dataset_then_train_quick(config_path: Path, experiment_id: str) -> int:
    ensure_experiment_storage(experiment_id)
    print("[QUICK] Tryb sekwencyjny: dataset -> train_existing")
    run_best_effort_cleanup("przed fazą dataset")

    dataset_rc = run_launch_phase("dataset", config_path, experiment_id, {"dataset"})
    if dataset_rc != 0:
        run_best_effort_cleanup("po nieudanej fazie dataset")
        return dataset_rc

    train_rc = train_existing_experiment(config_path, experiment_id)
    run_best_effort_cleanup("po zakończeniu dataset_train")
    return train_rc


def train_existing_experiment(base_config_path: Path, experiment_id: str) -> int:
    ensure_experiment_storage(experiment_id)
    cfg = load_yaml(base_config_path)
    mode = str(get_cfg_value(cfg, "experiment", "mode", default="ai")).lower()
    if mode != "ai":
        raise ValueError("Szybki trening istniejącego eksperymentu jest dostępny tylko dla experiment.mode=ai.")

    out_dir = str((REPO_ROOT / "out").resolve())
    seed = int(get_cfg_value(cfg, "experiment", "seed", default=123))
    dataset_wait_timeout = float(get_cfg_value(cfg, "timing", "dataset_wait_timeout", default=0.0))
    max_epochs = int(get_cfg_value(cfg, "training", "max_epochs", default=200))
    patience = int(get_cfg_value(cfg, "training", "patience", default=20))
    min_delta = float(get_cfg_value(cfg, "training", "min_delta", default=1e-5))
    learning_rate = float(get_cfg_value(cfg, "training", "learning_rate", default=1e-3))
    batch_size = int(get_cfg_value(cfg, "training", "batch_size", default=128))
    validation_ratio = float(get_cfg_value(cfg, "training", "validation_ratio", default=0.2))

    commands: list[str] = []
    commands.append(
        build_ros_run_command(
            "train_model",
            [
                ("seed", seed),
                ("out_dir", out_dir),
                ("experiment_id", experiment_id),
                ("dataset_name", "dataset.npz"),
                ("model_name", "model.pt"),
                ("history_name", "train_history.json"),
                ("skip_if_model_exists", False),
                ("dataset_wait_timeout", dataset_wait_timeout),
                ("max_epochs", max_epochs),
                ("patience", patience),
                ("min_delta", min_delta),
                ("lr", learning_rate),
                ("batch_size", batch_size),
                ("val_ratio", validation_ratio),
            ],
        )
    )

    tracks_cfg = get_cfg_value(cfg, "tracks", default={})
    robak_enabled = parse_bool(get_cfg_value(tracks_cfg, "tor5_robak", default=False), default=False)
    rywak_enabled = parse_bool(get_cfg_value(tracks_cfg, "tor6_rywak", default=False), default=False)

    robak_cfg = get_cfg_value(cfg, "robak", default={})
    if robak_enabled:
        commands.append(
            build_ros_run_command(
                "train_model_robak",
                [
                    ("seed", seed),
                    ("out_dir", out_dir),
                    ("experiment_id", experiment_id),
                    ("dataset_name", str(get_cfg_value(robak_cfg, "dataset_name", default="dataset_robak.npz"))),
                    ("model_name", str(get_cfg_value(robak_cfg, "model_name", default="model_robak.pt"))),
                    ("history_name", str(get_cfg_value(robak_cfg, "history_name", default="train_history_robak.json"))),
                    ("skip_if_model_exists", False),
                    ("write_experiment_metadata", True),
                    ("dataset_wait_timeout", dataset_wait_timeout),
                    ("max_epochs", int(get_cfg_value(robak_cfg, "max_epochs", default=max_epochs))),
                    ("patience", int(get_cfg_value(robak_cfg, "patience", default=patience))),
                    ("min_delta", min_delta),
                    ("lr", float(get_cfg_value(robak_cfg, "lr", default=learning_rate))),
                    ("batch_size", int(get_cfg_value(robak_cfg, "batch_size", default=batch_size))),
                    ("val_ratio", float(get_cfg_value(robak_cfg, "val_ratio", default=validation_ratio))),
                ],
            )
        )

    rywak_cfg = get_cfg_value(cfg, "rywak", default={})
    if rywak_enabled:
        commands.append(
            build_ros_run_command(
                "train_model_rywak",
                [
                    ("seed", seed),
                    ("out_dir", out_dir),
                    ("experiment_id", experiment_id),
                    ("dataset_name", str(get_cfg_value(rywak_cfg, "dataset_name", default="dataset_rywak.npz"))),
                    ("model_name", str(get_cfg_value(rywak_cfg, "model_name", default="model_rywak.pt"))),
                    ("history_name", str(get_cfg_value(rywak_cfg, "history_name", default="train_history_rywak.json"))),
                    ("skip_if_model_exists", False),
                    ("write_experiment_metadata", True),
                    ("dataset_wait_timeout", dataset_wait_timeout),
                    ("max_epochs", int(get_cfg_value(rywak_cfg, "max_epochs", default=max_epochs))),
                    ("patience", int(get_cfg_value(rywak_cfg, "patience", default=patience))),
                    ("min_delta", min_delta),
                    ("lr", float(get_cfg_value(rywak_cfg, "lr", default=learning_rate))),
                    ("batch_size", int(get_cfg_value(rywak_cfg, "batch_size", default=batch_size))),
                    ("val_ratio", float(get_cfg_value(rywak_cfg, "val_ratio", default=validation_ratio))),
                    ("hidden_dims", list(get_cfg_value(rywak_cfg, "hidden_dims", default=[192, 96, 48]))),
                    ("dropout", float(get_cfg_value(rywak_cfg, "dropout", default=0.1))),
                    ("weight_decay", float(get_cfg_value(rywak_cfg, "weight_decay", default=1e-4))),
                    ("huber_delta", float(get_cfg_value(rywak_cfg, "huber_delta", default=1.0))),
                    ("input_noise_std", float(get_cfg_value(rywak_cfg, "input_noise_std", default=0.02))),
                    ("clip_grad_norm", float(get_cfg_value(rywak_cfg, "clip_grad_norm", default=1.0))),
                    ("loss_v_weight", float(get_cfg_value(rywak_cfg, "loss_v_weight", default=1.0))),
                    ("loss_w_weight", float(get_cfg_value(rywak_cfg, "loss_w_weight", default=1.5))),
                ],
            )
        )

    for command in commands:
        return_code = run_shell_command(command)
        if return_code != 0:
            return return_code
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Szybkie uruchamianie pipeline z dashboardu.")
    parser.add_argument(
        "--mode",
        choices=("dataset", "dataset_train", "full_cycle", "train_existing", "test_existing", "train_test_existing"),
        required=True,
    )
    parser.add_argument("--base-config", required=True, help="Nazwa configu z config/ albo ścieżka absolutna.")
    parser.add_argument("--name", default="", help="Przyjazna nazwa uruchomienia. Zostanie dołączona do experiment_id.")
    parser.add_argument("--dataset-duration", type=float, default=None, help="Wspólny czas datasetu dla AI, Robaka i Rywaka.")
    parser.add_argument("--eval-duration", type=float, default=None, help="Czas testu i ewaluacji.")
    parser.add_argument("--dataset-world", default="", help="Świat używany do zbierania datasetu.")
    parser.add_argument("--test-world", default="", help="Świat używany do testu i ewaluacji.")
    parser.add_argument("--experiment-id", default="", help="Istniejący eksperyment do treningu lub testu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_grouped_out_layout()
    ensure_runtime_ready()
    base_config_path = resolve_config_path(args.base_config)
    if args.mode in {"dataset", "dataset_train", "full_cycle"} and args.dataset_duration is None:
        raise ValueError("Tryb dataset/dataset_train/full_cycle wymaga --dataset-duration.")
    if args.mode in {"full_cycle", "test_existing", "train_test_existing"} and args.eval_duration is None:
        raise ValueError("Tryb z testem wymaga --eval-duration.")
    if args.mode in {"train_existing", "test_existing", "train_test_existing"} and not args.experiment_id.strip():
        raise ValueError("Tryb bez datasetu wymaga --experiment-id.")

    if args.mode == "train_existing":
        print(f"[QUICK] Bazowy config: {base_config_path}")
        print(f"[QUICK] Existing experiment: {args.experiment_id}")
        return train_existing_experiment(base_config_path=base_config_path, experiment_id=args.experiment_id.strip())

    temp_config_path, experiment_id = create_temp_config(
        base_config_path=base_config_path,
        dataset_duration=args.dataset_duration,
        eval_duration=args.eval_duration,
        experiment_name=args.name,
        dataset_world=args.dataset_world.strip() or None,
        test_world=args.test_world.strip() or None,
        forced_experiment_id=args.experiment_id.strip() or None,
    )

    print(f"[QUICK] Bazowy config: {base_config_path}")
    print(f"[QUICK] Tymczasowy config: {temp_config_path}")
    print(f"[QUICK] Experiment ID: {experiment_id}")

    if args.mode == "dataset":
        return run_launch_phase("dataset", temp_config_path, experiment_id, {"dataset"})

    if args.mode == "dataset_train":
        return run_dataset_then_train_quick(temp_config_path, experiment_id)

    if args.mode == "test_existing":
        return run_launch_phase("test", temp_config_path, args.experiment_id.strip(), {"test"})

    if args.mode == "train_test_existing":
        print(f"[QUICK] Existing experiment: {args.experiment_id}")
        train_rc = train_existing_experiment(base_config_path=base_config_path, experiment_id=args.experiment_id.strip())
        if train_rc != 0:
            return train_rc
        return run_launch_phase("test", temp_config_path, args.experiment_id.strip(), {"test"})

    return run_full_cycle_quick(temp_config_path, experiment_id)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[QUICK][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
