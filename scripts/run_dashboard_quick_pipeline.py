#!/usr/bin/env python3
"""Tworzy tymczasowy config z nadpisanymi czasami i uruchamia szybki pipeline z dashboardu."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from out_layout import DASHBOARD_QUICK_CONFIG_DIR, ensure_experiment_storage, ensure_grouped_out_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config"
TEMP_CONFIG_DIR = DASHBOARD_QUICK_CONFIG_DIR
VENV_SITE = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"


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
    forced_experiment_id: str | None = None,
) -> tuple[Path, str]:
    payload = load_yaml(base_config_path)
    experiment_id = forced_experiment_id or build_experiment_id(experiment_name)

    if dataset_duration is not None:
        set_nested(payload, ["timing", "dataset_duration"], float(dataset_duration))
        set_nested(payload, ["robak", "dataset_duration"], float(dataset_duration))
        set_nested(payload, ["rywak", "dataset_duration"], float(dataset_duration))
    if eval_duration is not None:
        set_nested(payload, ["timing", "eval_duration"], float(eval_duration))

    set_nested(payload, ["dashboard_quick_launch", "base_config"], str(base_config_path))
    if dataset_duration is not None:
        set_nested(payload, ["dashboard_quick_launch", "dataset_duration_common"], float(dataset_duration))
    if eval_duration is not None:
        set_nested(payload, ["dashboard_quick_launch", "eval_duration"], float(eval_duration))
    if experiment_name.strip():
        set_nested(payload, ["dashboard_quick_launch", "requested_name"], experiment_name.strip())
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
        "source /opt/ros/jazzy/setup.bash",
        "if [ -f ai_slam_ws/install/setup.bash ]; then source ai_slam_ws/install/setup.bash; fi",
    ]
    if VENV_SITE.is_dir():
        parts.append(f"export PYTHONPATH=\"${{PYTHONPATH:+$PYTHONPATH:}}{VENV_SITE}\"")
    parts.append(f"ros2 run ai_slam_ai {executable} --ros-args {param_args}")
    return " && ".join(parts)


def run_shell_command(command: str) -> int:
    print("[QUICK] Uruchamiam:")
    print("[QUICK]   " + command)
    completed = subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def train_existing_experiment(base_config_path: Path, experiment_id: str) -> int:
    ensure_experiment_storage(experiment_id)
    cfg = load_yaml(base_config_path)
    mode = str(get_cfg_value(cfg, "experiment", "mode", default="ai")).lower()
    if mode != "ai":
        raise ValueError("Szybki trening istniejącego eksperymentu jest dostępny tylko dla experiment.mode=ai.")

    out_dir = str((REPO_ROOT / "out").resolve())
    seed = int(get_cfg_value(cfg, "experiment", "seed", default=123))
    dataset_wait_timeout = float(get_cfg_value(cfg, "timing", "dataset_wait_timeout", default=120.0))
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
    parser.add_argument("--mode", choices=("dataset", "full_cycle", "train_existing", "test_existing"), required=True)
    parser.add_argument("--base-config", required=True, help="Nazwa configu z config/ albo ścieżka absolutna.")
    parser.add_argument("--name", default="", help="Przyjazna nazwa uruchomienia. Zostanie dołączona do experiment_id.")
    parser.add_argument("--dataset-duration", type=float, default=None, help="Wspólny czas datasetu dla AI, Robaka i Rywaka.")
    parser.add_argument("--eval-duration", type=float, default=None, help="Czas testu i ewaluacji.")
    parser.add_argument("--experiment-id", default="", help="Istniejący eksperyment do treningu lub testu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_grouped_out_layout()
    base_config_path = resolve_config_path(args.base_config)
    if args.mode in {"dataset", "full_cycle"} and args.dataset_duration is None:
        raise ValueError("Tryb dataset/full_cycle wymaga --dataset-duration.")
    if args.mode in {"full_cycle", "test_existing"} and args.eval_duration is None:
        raise ValueError("Tryb full_cycle/test_existing wymaga --eval-duration.")
    if args.mode in {"train_existing", "test_existing"} and not args.experiment_id.strip():
        raise ValueError("Tryb train_existing/test_existing wymaga --experiment-id.")

    if args.mode == "train_existing":
        print(f"[QUICK] Bazowy config: {base_config_path}")
        print(f"[QUICK] Existing experiment: {args.experiment_id}")
        return train_existing_experiment(base_config_path=base_config_path, experiment_id=args.experiment_id.strip())

    temp_config_path, experiment_id = create_temp_config(
        base_config_path=base_config_path,
        dataset_duration=args.dataset_duration,
        eval_duration=args.eval_duration,
        experiment_name=args.name,
        forced_experiment_id=args.experiment_id.strip() or None,
    )

    print(f"[QUICK] Bazowy config: {base_config_path}")
    print(f"[QUICK] Tymczasowy config: {temp_config_path}")
    print(f"[QUICK] Experiment ID: {experiment_id}")

    if args.mode == "dataset":
        ensure_experiment_storage(experiment_id)
        return run_command(
            [
                "bash",
                "./scripts/run_experiment.sh",
                "dataset",
                f"config:={temp_config_path}",
                f"experiment_id:={experiment_id}",
            ]
        )

    if args.mode == "test_existing":
        ensure_experiment_storage(args.experiment_id.strip())
        return run_command(
            [
                "bash",
                "./scripts/run_experiment.sh",
                "test",
                f"config:={temp_config_path}",
                f"experiment_id:={args.experiment_id.strip()}",
            ]
        )

    ensure_experiment_storage(experiment_id)
    return run_command(
        [
            "bash",
            "./scripts/run_full_cycle.sh",
            str(temp_config_path),
            f"experiment_id:={experiment_id}",
        ]
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[QUICK][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
