#!/usr/bin/env python3
"""Lekki dashboard HTTP do przegladania eksperymentow i uruchamiania skryptow."""

from __future__ import annotations

import argparse
import csv
import errno
import html
import io
import json
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from out_layout import (
    DASHBOARD_JOBS_DIR,
    OUT_DIR,
    ensure_grouped_out_layout,
    iter_experiment_dirs,
    iter_sweep_dirs,
    resolve_experiment_dir,
    resolve_sweep_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
CONFIG_DIR = REPO_ROOT / "ai_slam_ws" / "src" / "ai_slam_bringup" / "config"
FUNCTION_INDEX_MD = DOCS_DIR / "function_index.md"
FUNCTION_INDEX_JSON = DOCS_DIR / "function_index.json"
JOB_LOG_DIR = DASHBOARD_JOBS_DIR
VENV_SITE = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
POSITION_SERIES = {
    "gt": ("time_s", "gt_xytheta", "GT", "#e2e8f0"),
    "baseline": ("time_s", "baseline_xytheta", "Baseline", "#c2410c"),
    "ai": ("ai_time_s", "ai_xytheta", "AI", "#0f766e"),
    "scanmatch": ("scanmatch_time_s", "scanmatch_xytheta", "ScanMatcher", "#2563eb"),
    "bruteforce": ("bruteforce_time_s", "bruteforce_xytheta", "Bruteforce", "#7c3aed"),
    "robak": ("robak_time_s", "robak_xytheta", "Robak", "#b91c1c"),
    "rywak": ("rywak_time_s", "rywak_xytheta", "Rywak", "#4d7c0f"),
}

mimetypes.add_type("text/markdown", ".md")
ERROR_SERIES = {
    "baseline": ("time_s", "baseline_err_xy", "baseline_err_theta", "Baseline", "#c2410c"),
    "ai": ("ai_time_s", "ai_err_xy", "ai_err_theta", "AI", "#0f766e"),
    "scanmatch": ("scanmatch_time_s", "scanmatch_err_xy", "scanmatch_err_theta", "ScanMatcher", "#2563eb"),
    "bruteforce": ("bruteforce_time_s", "bruteforce_err_xy", "bruteforce_err_theta", "Bruteforce", "#7c3aed"),
    "robak": ("robak_time_s", "robak_err_xy", "robak_err_theta", "Robak", "#b91c1c"),
    "rywak": ("rywak_time_s", "rywak_err_xy", "rywak_err_theta", "Rywak", "#4d7c0f"),
}
COMPARISON_GROUPS = {
    "robak": "Robak",
    "rywak": "Rywak",
    "map_filter": "Filtr mapy",
}
COMPARISON_PARAM_SPECS = [
    {"key": "robak.offsets", "label": "Robak offsets", "group": "robak", "kind": "categorical", "sources": [("config", ("robak", "offsets")), ("robak_dataset_meta", ("offsets",))]},
    {"key": "robak.lr", "label": "Robak lr", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "lr"))]},
    {"key": "robak.max_epochs", "label": "Robak max_epochs", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "max_epochs"))]},
    {"key": "robak.batch_size", "label": "Robak batch_size", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "batch_size"))]},
    {"key": "robak.val_ratio", "label": "Robak val_ratio", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "val_ratio"))]},
    {"key": "robak.min_pair_dist", "label": "Robak min_pair_dist", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "min_pair_dist")), ("robak_dataset_meta", ("min_pair_dist",))]},
    {"key": "robak.min_pair_dyaw", "label": "Robak min_pair_dyaw", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "min_pair_dyaw")), ("robak_dataset_meta", ("min_pair_dyaw",))]},
    {"key": "robak.min_pair_dt_sec", "label": "Robak min_pair_dt_sec", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "min_pair_dt_sec")), ("robak_dataset_meta", ("min_pair_dt_sec",))]},
    {"key": "robak.pair_filter_mode", "label": "Robak pair_filter_mode", "group": "robak", "kind": "categorical", "sources": [("config", ("robak", "pair_filter_mode")), ("robak_dataset_meta", ("pair_filter_mode",))]},
    {"key": "robak.max_pair_dist", "label": "Robak max_pair_dist", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "max_pair_dist")), ("robak_dataset_meta", ("max_pair_dist",))]},
    {"key": "robak.max_pair_dyaw", "label": "Robak max_pair_dyaw", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "max_pair_dyaw")), ("robak_dataset_meta", ("max_pair_dyaw",))]},
    {"key": "robak.augment_noise_std_scale", "label": "Robak augment_noise_std_scale", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "augment_noise_std_scale")), ("robak_dataset_meta", ("augment_noise_std_scale",))]},
    {"key": "robak.augment_cut_fraction", "label": "Robak augment_cut_fraction", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "augment_cut_fraction")), ("robak_dataset_meta", ("augment_cut_fraction",))]},
    {"key": "robak.augment_cut_max_points", "label": "Robak augment_cut_max_points", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "augment_cut_max_points")), ("robak_dataset_meta", ("augment_cut_max_points",))]},
    {"key": "robak.infer_delta_ema_alpha", "label": "Robak infer_delta_ema_alpha", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "infer_delta_ema_alpha"))]},
    {"key": "robak.infer_odom_heading_alpha", "label": "Robak infer_odom_heading_alpha", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "infer_odom_heading_alpha"))]},
    {"key": "robak.infer_odom_delta_xy_alpha", "label": "Robak infer_odom_delta_xy_alpha", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "infer_odom_delta_xy_alpha"))]},
    {"key": "robak.infer_odom_delta_yaw_alpha", "label": "Robak infer_odom_delta_yaw_alpha", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "infer_odom_delta_yaw_alpha"))]},
    {"key": "robak.infer_odom_pose_xy_alpha", "label": "Robak infer_odom_pose_xy_alpha", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "infer_odom_pose_xy_alpha"))]},
    {"key": "robak.infer_odom_pose_xy_gain", "label": "Robak infer_odom_pose_xy_gain", "group": "robak", "kind": "numeric", "sources": [("config", ("robak", "infer_odom_pose_xy_gain"))]},
    {"key": "rywak.min_sample_dist", "label": "Rywak min_sample_dist", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "min_sample_dist")), ("rywak_dataset_meta", ("min_sample_dist",))]},
    {"key": "rywak.lr", "label": "Rywak lr", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "lr"))]},
    {"key": "rywak.max_epochs", "label": "Rywak max_epochs", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "max_epochs"))]},
    {"key": "rywak.batch_size", "label": "Rywak batch_size", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "batch_size"))]},
    {"key": "rywak.val_ratio", "label": "Rywak val_ratio", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "val_ratio"))]},
    {"key": "rywak.min_sample_dyaw", "label": "Rywak min_sample_dyaw", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "min_sample_dyaw")), ("rywak_dataset_meta", ("min_sample_dyaw",))]},
    {"key": "rywak.min_sample_dt_sec", "label": "Rywak min_sample_dt_sec", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "min_sample_dt_sec")), ("rywak_dataset_meta", ("min_sample_dt_sec",))]},
    {"key": "rywak.min_delta_scan_rms", "label": "Rywak min_delta_scan_rms", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "min_delta_scan_rms")), ("rywak_dataset_meta", ("min_delta_scan_rms",))]},
    {"key": "rywak.sample_filter_mode", "label": "Rywak sample_filter_mode", "group": "rywak", "kind": "categorical", "sources": [("config", ("rywak", "sample_filter_mode")), ("rywak_dataset_meta", ("sample_filter_mode",))]},
    {"key": "rywak.delta_scan_clip", "label": "Rywak delta_scan_clip", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "delta_scan_clip")), ("rywak_dataset_meta", ("delta_scan_clip",))]},
    {"key": "rywak.hidden_dims", "label": "Rywak hidden_dims", "group": "rywak", "kind": "categorical", "sources": [("config", ("rywak", "hidden_dims")), ("rywak_history", ("hidden_dims",))]},
    {"key": "rywak.dropout", "label": "Rywak dropout", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "dropout")), ("rywak_history", ("dropout",))]},
    {"key": "rywak.weight_decay", "label": "Rywak weight_decay", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "weight_decay")), ("rywak_history", ("weight_decay",))]},
    {"key": "rywak.huber_delta", "label": "Rywak huber_delta", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "huber_delta")), ("rywak_history", ("huber_delta",))]},
    {"key": "rywak.input_noise_std", "label": "Rywak input_noise_std", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "input_noise_std")), ("rywak_history", ("input_noise_std",))]},
    {"key": "rywak.clip_grad_norm", "label": "Rywak clip_grad_norm", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "clip_grad_norm")), ("rywak_history", ("clip_grad_norm",))]},
    {"key": "rywak.loss_v_weight", "label": "Rywak loss_v_weight", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "loss_v_weight")), ("rywak_history", ("loss_v_weight",))]},
    {"key": "rywak.loss_w_weight", "label": "Rywak loss_w_weight", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "loss_w_weight")), ("rywak_history", ("loss_w_weight",))]},
    {"key": "rywak.fuse_odom_v_weight", "label": "Rywak fuse_odom_v_weight", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "fuse_odom_v_weight"))]},
    {"key": "rywak.fuse_odom_w_weight", "label": "Rywak fuse_odom_w_weight", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "fuse_odom_w_weight"))]},
    {"key": "rywak.fuse_odom_v_gain", "label": "Rywak fuse_odom_v_gain", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "fuse_odom_v_gain"))]},
    {"key": "rywak.fuse_odom_w_gain", "label": "Rywak fuse_odom_w_gain", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "fuse_odom_w_gain"))]},
    {"key": "rywak.vel_ema_alpha", "label": "Rywak vel_ema_alpha", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "vel_ema_alpha"))]},
    {"key": "rywak.anchor_yaw_to_odom", "label": "Rywak anchor_yaw_to_odom", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "anchor_yaw_to_odom"))]},
    {"key": "rywak.anchor_xy_to_odom", "label": "Rywak anchor_xy_to_odom", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "anchor_xy_to_odom"))]},
    {"key": "rywak.anchor_xy_to_odom_gain", "label": "Rywak anchor_xy_to_odom_gain", "group": "rywak", "kind": "numeric", "sources": [("config", ("rywak", "anchor_xy_to_odom_gain"))]},
    {"key": "evaluation.points_min_translation", "label": "Mapa points_min_translation", "group": "map_filter", "kind": "numeric", "sources": [("config", ("evaluation", "points_min_translation"))]},
    {"key": "evaluation.points_min_rotation", "label": "Mapa points_min_rotation", "group": "map_filter", "kind": "numeric", "sources": [("config", ("evaluation", "points_min_rotation"))]},
    {"key": "evaluation.points_min_time_gap_sec", "label": "Mapa points_min_time_gap_sec", "group": "map_filter", "kind": "numeric", "sources": [("config", ("evaluation", "points_min_time_gap_sec"))]},
    {"key": "evaluation.points_filter_mode", "label": "Mapa points_filter_mode", "group": "map_filter", "kind": "categorical", "sources": [("config", ("evaluation", "points_filter_mode"))]},
]
COMPARISON_METRIC_SPECS = [
    {"key": "rmse_xy_robak", "label": "Robak RMSE XY [m]", "group": "robak"},
    {"key": "rmse_theta_robak", "label": "Robak RMSE theta [rad]", "group": "robak"},
    {"key": "iou_map_robak", "label": "Robak IoU mapy", "group": "robak"},
    {"key": "rmse_xy_rywak", "label": "Rywak RMSE XY [m]", "group": "rywak"},
    {"key": "rmse_theta_rywak", "label": "Rywak RMSE theta [rad]", "group": "rywak"},
    {"key": "iou_map_rywak", "label": "Rywak IoU mapy", "group": "rywak"},
    {"key": "iou_map_robak", "label": "IoU mapy Robak", "group": "map_filter"},
    {"key": "iou_map_rywak", "label": "IoU mapy Rywak", "group": "map_filter"},
]
PARAM_SPEC_BY_KEY = {spec["key"]: spec for spec in COMPARISON_PARAM_SPECS}
METRIC_SPEC_BY_KEY = {spec["key"]: spec for spec in COMPARISON_METRIC_SPECS}
SHARED_SWEEP_PARAM_LABELS = {
    "shared.lr": "Wspólny learning rate Robaka i Rywaka",
    "shared.max_epochs": "Wspólna liczba epok Robaka i Rywaka",
    "shared.batch_size": "Wspólny batch size Robaka i Rywaka",
    "shared.val_ratio": "Wspólny udział walidacji Robaka i Rywaka",
}
SWEEP_PLOT_FAMILIES = {
    "rmse_xy": {
        "label": "RMSE XY [m]",
        "y_label": "RMSE XY [m]",
        "series": [
            ("baseline", "rmse_xy_baseline", "Baseline", "#c2410c"),
            ("ai", "rmse_xy_ai", "AI", "#0f766e"),
            ("robak", "rmse_xy_robak", "Robak", "#b91c1c"),
            ("rywak", "rmse_xy_rywak", "Rywak", "#4d7c0f"),
        ],
    },
    "rmse_theta": {
        "label": "RMSE theta [rad]",
        "y_label": "RMSE theta [rad]",
        "series": [
            ("baseline", "rmse_theta_baseline", "Baseline", "#c2410c"),
            ("ai", "rmse_theta_ai", "AI", "#0f766e"),
            ("robak", "rmse_theta_robak", "Robak", "#b91c1c"),
            ("rywak", "rmse_theta_rywak", "Rywak", "#4d7c0f"),
        ],
    },
    "iou_map": {
        "label": "IoU mapy",
        "y_label": "IoU",
        "series": [
            ("baseline", "iou_map_baseline", "Baseline", "#c2410c"),
            ("ai", "iou_map_ai", "AI", "#0f766e"),
            ("robak", "iou_map_robak", "Robak", "#b91c1c"),
            ("rywak", "iou_map_rywak", "Rywak", "#4d7c0f"),
        ],
    },
}
SWEEP_NOTE_RE = re.compile(
    r"Sweep (?P<mode>[a-z_]+): source=(?P<source>[^,]+), param=(?P<param>[^,]+), "
    r"value=(?P<value>.*?), base_config=(?P<base_config>.+)$"
)
SWEEP_EXPERIMENT_RE = re.compile(r"(exp_[A-Za-z0-9_]+)")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]
    except Exception:
        return []


def read_csv_list(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader if isinstance(row, dict)]
    except Exception:
        return []


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path.resolve())


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return normalize_json_value(value.item())
        return [normalize_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): normalize_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def get_nested_value(data: Any, path: tuple[str, ...] | list[str]) -> Any:
    value = data
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return normalize_json_value(value)


def read_npz_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as payload:
            if "meta" not in payload:
                return {}
            meta = payload["meta"]
            if isinstance(meta, np.ndarray) and meta.shape == ():
                meta = meta.item()
            return normalize_json_value(meta if isinstance(meta, dict) else {})
    except Exception:
        return {}


def format_param_value(value: Any) -> str:
    value = normalize_json_value(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def is_numeric_param_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_sweep_value(raw_value: str) -> Any:
    text = str(raw_value).strip()
    if not text:
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.startswith(("[", "{", "\"")):
        try:
            return json.loads(text)
        except Exception:
            pass
    try:
        if any(char in text for char in ".eE"):
            return float(text)
        return int(text)
    except Exception:
        return text


def extract_sweep_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sweep = metadata.get("sweep")
    if isinstance(sweep, dict) and sweep.get("param_path"):
        return {
            "mode": str(sweep.get("mode", "")).strip() or "fixed_dataset",
            "source_experiment_id": str(sweep.get("source_experiment_id", "")).strip(),
            "param_path": str(sweep.get("param_path", "")).strip(),
            "param_value": normalize_json_value(sweep.get("param_value")),
            "base_config_path": str(sweep.get("base_config_path", "")).strip(),
        }

    notes = metadata.get("notes", [])
    if not isinstance(notes, list):
        return {}
    for note in reversed(notes):
        match = SWEEP_NOTE_RE.search(str(note))
        if not match:
            continue
        return {
            "mode": match.group("mode").strip() or "fixed_dataset",
            "source_experiment_id": match.group("source").strip(),
            "param_path": match.group("param").strip(),
            "param_value": parse_sweep_value(match.group("value")),
            "base_config_path": match.group("base_config").strip(),
        }
    return {}


def sweep_config_path(sweep_dir: Path, experiment_id: str) -> str:
    index = experiment_id.rsplit("_", 1)[-1]
    if not index.isdigit():
        return ""
    config_dir = sweep_dir / "configs"
    if not config_dir.exists():
        return ""
    matches = sorted(config_dir.glob(f"{index}_*.yaml"))
    if not matches:
        return ""
    return str(matches[0].resolve())


def collect_sweep_experiment_ids(sweep_dir: Path) -> list[str]:
    experiment_ids: set[str] = set()
    for subdir_name, pattern in (("logs", "*.log"), ("train_params", "*.yaml")):
        subdir = sweep_dir / subdir_name
        if not subdir.exists():
            continue
        for path in subdir.glob(pattern):
            match = SWEEP_EXPERIMENT_RE.search(path.stem)
            if match:
                experiment_ids.add(match.group(1))
    return sorted(experiment_ids)


def recover_sweep_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    recovered_rows: list[dict[str, Any]] = []
    for experiment_id in collect_sweep_experiment_ids(sweep_dir):
        try:
            exp_dir = resolve_experiment_dir(experiment_id)
        except FileNotFoundError:
            continue

        metadata = read_json(exp_dir / "experiment_metadata.json")
        sweep_meta = extract_sweep_metadata(metadata)
        if not sweep_meta.get("param_path"):
            continue

        results = read_json(exp_dir / "results.json")
        metrics = results.get("metrics", {}) if isinstance(results.get("metrics"), dict) else {}
        total_time = metadata.get("total_experiment_time_sec")
        recovered_rows.append(
            {
                "mode": sweep_meta.get("mode", "fixed_dataset"),
                "source_experiment_id": sweep_meta.get("source_experiment_id", ""),
                "param_path": sweep_meta.get("param_path", ""),
                "param_value": sweep_meta.get("param_value"),
                "status": "done" if (exp_dir / "results.json").exists() else "failed_results_missing",
                "elapsed_sec": round(float(total_time), 3) if isinstance(total_time, (int, float)) else total_time,
                "experiment_id": experiment_id,
                "config_path": sweep_config_path(sweep_dir, experiment_id),
                "rmse_xy_robak": metrics.get("rmse_xy_robak"),
                "rmse_theta_robak": metrics.get("rmse_theta_robak"),
                "iou_map_robak": metrics.get("iou_map_robak"),
                "rmse_xy_rywak": metrics.get("rmse_xy_rywak"),
                "rmse_theta_rywak": metrics.get("rmse_theta_rywak"),
                "iou_map_rywak": metrics.get("iou_map_rywak"),
                "rmse_xy_ai": metrics.get("rmse_xy_ai"),
                "rmse_theta_ai": metrics.get("rmse_theta_ai"),
                "iou_map_ai": metrics.get("iou_map_ai"),
                "rmse_xy_baseline": metrics.get("rmse_xy_baseline"),
                "rmse_theta_baseline": metrics.get("rmse_theta_baseline"),
                "iou_map_baseline": metrics.get("iou_map_baseline"),
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[int, float | str]:
        numeric_value = metric_float(row.get("param_value"))
        if numeric_value is not None:
            return (0, numeric_value)
        return (1, str(row.get("experiment_id", "")))

    return sorted(recovered_rows, key=sort_key)


def load_sweep_rows(sweep_dir: Path) -> tuple[list[dict[str, Any]], str]:
    summary_json_path = sweep_dir / "summary.json"
    rows = read_json_list(summary_json_path)
    if rows:
        return rows, "summary_json"

    summary_csv_path = sweep_dir / "summary.csv"
    rows = read_csv_list(summary_csv_path)
    if rows:
        return rows, "summary_csv"

    rows = recover_sweep_rows(sweep_dir)
    if rows:
        return rows, "recovered_experiments"
    return [], "missing"


def list_config_files() -> list[dict[str, str]]:
    configs: list[dict[str, str]] = []
    if not CONFIG_DIR.exists():
        return configs
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        configs.append({"name": path.name, "path": str(path.resolve())})
    return configs


def resolve_config_name(name: str) -> Path:
    target = (CONFIG_DIR / name).resolve()
    if not str(target).startswith(str(CONFIG_DIR.resolve())):
        raise ValueError("Niepoprawna nazwa pliku konfiguracyjnego.")
    if target.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Dozwolone są tylko pliki YAML.")
    return target


def load_config_payload(name: str) -> dict[str, Any]:
    path = resolve_config_name(name)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {name}")
    content = read_text(path)
    parsed = yaml.safe_load(content) if content.strip() else {}
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "content": content,
        "parsed": parsed if parsed is not None else {},
    }


def render_yaml_content(parsed: Any) -> str:
    return yaml.safe_dump(
        parsed if parsed is not None else {},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def save_config_payload(name: str, content: str | None = None, parsed: Any | None = None) -> dict[str, Any]:
    path = resolve_config_name(name)
    if parsed is not None:
        content = render_yaml_content(parsed)
    if content is None:
        raise ValueError("Brak treści pliku konfiguracyjnego.")
    parsed_payload = yaml.safe_load(content) if content.strip() else {}
    path.write_text(content, encoding="utf-8")
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "content": content,
        "parsed": parsed_payload if parsed_payload is not None else {},
    }


def ensure_function_index() -> None:
    if FUNCTION_INDEX_JSON.exists() and FUNCTION_INDEX_MD.exists():
        return
    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "generate_function_index.py"),
            "--output-md",
            str(FUNCTION_INDEX_MD),
            "--output-json",
            str(FUNCTION_INDEX_JSON),
        ],
        cwd=REPO_ROOT,
        check=False,
    )


def safe_resolve_local_path(raw_path: str) -> Path:
    candidate = Path(unquote(raw_path))
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not str(candidate).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Sciezka jest poza repozytorium.")
    return candidate


def load_trajectory_npz(experiment_id: str) -> np.lib.npyio.NpzFile:
    exp_dir = resolve_experiment_dir(experiment_id)
    traj_path = exp_dir / "trajectory_data.npz"
    if not traj_path.exists():
        raise FileNotFoundError(f"Brak pliku trajectory_data.npz dla {experiment_id}")
    return np.load(traj_path, allow_pickle=True)


def wrap_angle_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return ((arr + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def nearest_time_indices(reference_time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_time, dtype=np.float32).reshape((-1,))
    query = np.asarray(query_time, dtype=np.float32).reshape((-1,))
    if ref.size == 0 or query.size == 0:
        return np.zeros((0,), dtype=np.int64)

    idx = np.searchsorted(ref, query, side="left")
    idx = np.clip(idx, 0, ref.size - 1)
    prev_idx = np.clip(idx - 1, 0, ref.size - 1)
    use_prev = np.abs(ref[prev_idx] - query) <= np.abs(ref[idx] - query)
    return np.where(use_prev, prev_idx, idx).astype(np.int64)


def get_pose_series(data: np.lib.npyio.NpzFile, series_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    spec = POSITION_SERIES.get(series_name)
    if spec is None:
        return None
    time_key, pose_key, _label, _color = spec
    if time_key not in data or pose_key not in data:
        return None
    time_arr = np.asarray(data[time_key], dtype=np.float32).reshape((-1,))
    pose_arr = np.asarray(data[pose_key], dtype=np.float32)
    if pose_arr.size == 0 or time_arr.size == 0:
        return None
    pose_arr = pose_arr.reshape((-1, 3))
    n = min(time_arr.shape[0], pose_arr.shape[0])
    if n <= 0:
        return None
    return time_arr[:n], pose_arr[:n]


def get_error_series(
    data: np.lib.npyio.NpzFile,
    series_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    spec = ERROR_SERIES.get(series_name)
    if spec is None:
        return None
    time_key, err_xy_key, err_th_key, _label, _color = spec
    if time_key in data and err_xy_key in data and err_th_key in data:
        time_arr = np.asarray(data[time_key], dtype=np.float32).reshape((-1,))
        err_xy = np.asarray(data[err_xy_key], dtype=np.float32)
        err_th = np.asarray(data[err_th_key], dtype=np.float32).reshape((-1,))
        if err_xy.size == 0 or err_th.size == 0 or time_arr.size == 0:
            return None
        err_xy = err_xy.reshape((-1, 2))
        n = min(time_arr.shape[0], err_xy.shape[0], err_th.shape[0])
        if n <= 0:
            return None
        return time_arr[:n], err_xy[:n], err_th[:n], "stored"

    if "time_s" not in data or "gt_xytheta" not in data:
        return None
    pose_series = get_pose_series(data, series_name)
    if pose_series is None:
        return None

    gt_time = np.asarray(data["time_s"], dtype=np.float32).reshape((-1,))
    gt_pose = np.asarray(data["gt_xytheta"], dtype=np.float32)
    if gt_time.size == 0 or gt_pose.size == 0:
        return None
    gt_pose = gt_pose.reshape((-1, 3))

    time_arr, pose_arr = pose_series
    n = min(gt_time.shape[0], gt_pose.shape[0])
    gt_time = gt_time[:n]
    gt_pose = gt_pose[:n]
    idx = nearest_time_indices(gt_time, time_arr)
    if idx.size == 0:
        return None
    gt_match = gt_pose[idx]
    err_xy = (gt_match[:, :2] - pose_arr[:, :2]).astype(np.float32)
    err_th = wrap_angle_array(gt_match[:, 2] - pose_arr[:, 2])
    return time_arr, err_xy, err_th, "reconstructed"


def inspect_trajectory_capabilities(exp_dir: Path) -> dict[str, Any]:
    traj_path = exp_dir / "trajectory_data.npz"
    if not traj_path.exists():
        return {"has_series": False, "error_mode": "missing"}
    try:
        data = np.load(traj_path, allow_pickle=True)
        mode = "missing"
        for series_name in ERROR_SERIES:
            result = get_error_series(data, series_name)
            if result is None:
                continue
            _time_arr, _err_xy, _err_th, source = result
            if source == "stored":
                mode = "stored"
                break
            if mode != "stored":
                mode = "reconstructed"
        return {"has_series": True, "error_mode": mode}
    except Exception:
        return {"has_series": True, "error_mode": "broken"}


def extract_experiment_parameters(exp_dir: Path, metadata: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    artifacts = results.get("artifacts", {}) if isinstance(results, dict) else {}
    config_snapshot_path = Path(str(artifacts.get("config_snapshot_yaml", exp_dir / "config_snapshot.yaml")))
    sources = {
        "config": read_yaml(config_snapshot_path) if config_snapshot_path.exists() else {},
        "robak_dataset_meta": read_npz_meta(Path(str(artifacts.get("robak_dataset_npz", exp_dir / "dataset_robak.npz")))),
        "rywak_dataset_meta": read_npz_meta(Path(str(artifacts.get("rywak_dataset_npz", exp_dir / "dataset_rywak.npz")))),
        "robak_history": read_json(Path(str(artifacts.get("robak_train_history_json", exp_dir / "train_history_robak.json")))),
        "rywak_history": read_json(Path(str(artifacts.get("rywak_train_history_json", exp_dir / "train_history_rywak.json")))),
        "training_metadata": metadata.get("training", {}).get("parameters", {}) if isinstance(metadata, dict) else {},
        "evaluation_metadata": metadata.get("evaluation", {}).get("parameters", {}) if isinstance(metadata, dict) else {},
    }

    values: dict[str, Any] = {}
    for spec in COMPARISON_PARAM_SPECS:
        for source_name, source_path in spec["sources"]:
            source_value = get_nested_value(sources.get(source_name, {}), source_path)
            if source_value is None:
                continue
            values[spec["key"]] = source_value
            break
    return values


def build_comparison_catalog(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group_key, group_label in COMPARISON_GROUPS.items():
        params: list[dict[str, Any]] = []
        for spec in COMPARISON_PARAM_SPECS:
            if spec["group"] != group_key:
                continue
            available_count = sum(1 for exp in experiments if spec["key"] in exp.get("parameter_values", {}))
            if available_count <= 0:
                continue
            params.append(
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "kind": spec["kind"],
                    "available_count": available_count,
                }
            )

        metrics: list[dict[str, Any]] = []
        for spec in COMPARISON_METRIC_SPECS:
            if spec["group"] != group_key:
                continue
            available_count = sum(1 for exp in experiments if exp.get("metrics", {}).get(spec["key"]) is not None)
            if available_count <= 0:
                continue
            metrics.append(
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "available_count": available_count,
                }
            )

        if params and metrics:
            groups.append(
                {
                    "key": group_key,
                    "label": group_label,
                    "params": params,
                    "metrics": metrics,
                }
            )

    return {"groups": groups}


def metric_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def discover_sweeps() -> list[dict[str, Any]]:
    sweeps: list[dict[str, Any]] = []
    ensure_grouped_out_layout()
    if not OUT_DIR.exists():
        return sweeps

    for sweep_dir in iter_sweep_dirs():
        summary_json_path = sweep_dir / "summary.json"
        summary_csv_path = sweep_dir / "summary.csv"
        rows, rows_source = load_sweep_rows(sweep_dir)

        first_row = rows[0] if rows else {}
        param_path = str(first_row.get("param_path", "")).strip()
        source_experiment_id = str(first_row.get("source_experiment_id", "")).strip()
        mode = str(first_row.get("mode", "")).strip() or "unknown"
        success_count = sum(1 for row in rows if str(row.get("status", "")).strip() == "done")
        failed_count = len(rows) - success_count
        families: list[dict[str, Any]] = []

        for family_key, family_spec in SWEEP_PLOT_FAMILIES.items():
            available_series = []
            for series_key, metric_key, label, _color in family_spec["series"]:
                count = sum(
                    1
                    for row in rows
                    if str(row.get("status", "")).strip() == "done" and metric_float(row.get(metric_key)) is not None
                )
                if count > 0:
                    available_series.append({"key": series_key, "label": label, "count": count})
            if available_series:
                families.append(
                    {
                        "key": family_key,
                        "label": family_spec["label"],
                        "series": available_series,
                    }
                )

        created_at_source = summary_json_path if summary_json_path.exists() else (
            summary_csv_path if summary_csv_path.exists() else sweep_dir
        )
        sweeps.append(
            {
                "id": sweep_dir.name,
                "path": str(sweep_dir.resolve()),
                "mode": mode,
                "source_experiment_id": source_experiment_id,
                "param_path": param_path,
                "total_count": len(rows),
                "success_count": success_count,
                "failed_count": failed_count,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created_at_source.stat().st_mtime)),
                "summary_json_path": str(summary_json_path.resolve()) if summary_json_path.exists() else "",
                "summary_csv_path": str(summary_csv_path.resolve()) if summary_csv_path.exists() else "",
                "rows_source": rows_source,
                "families": families,
                "has_metric_data": bool(families),
            }
        )

    return sweeps


def discover_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    ensure_grouped_out_layout()
    if not OUT_DIR.exists():
        return experiments

    for exp_dir in iter_experiment_dirs():
        metadata = read_json(exp_dir / "experiment_metadata.json")
        results = read_json(exp_dir / "results.json")
        series_info = inspect_trajectory_capabilities(exp_dir)
        parameter_values = extract_experiment_parameters(exp_dir, metadata, results)
        dataset_inspection_summary = read_json(exp_dir / "dataset_inspection_summary.json")

        datasets: list[dict[str, Any]] = []
        for dataset_path in sorted(exp_dir.glob("dataset*.npz")):
            if dataset_path.name == "trajectory_data.npz":
                continue
            kind = "ai"
            if "robak" in dataset_path.name:
                kind = "robak"
            elif "rywak" in dataset_path.name:
                kind = "rywak"
            datasets.append(
                {
                    "name": dataset_path.name,
                    "kind": kind,
                    "path": str(dataset_path.resolve()),
                    "size_mb": round(dataset_path.stat().st_size / (1024 * 1024), 3),
                }
            )

        artifacts = results.get("artifacts", {})
        extra_artifacts = {
            "dataset_inspection_overview_png": exp_dir / "dataset_inspection_overview.png",
            "dataset_inspection_scans_png": exp_dir / "dataset_inspection_scans.png",
            "dataset_inspection_summary_json": exp_dir / "dataset_inspection_summary.json",
            "dataset_analysis_png": exp_dir / "dataset_analysis.png",
        }
        metrics = results.get("metrics", {})
        artifact_map = {
            key: value
            for key, value in artifacts.items()
            if isinstance(value, str) and Path(value).suffix.lower() in {".png", ".json", ".npz", ".pt", ".yaml"}
        }
        for key, path in extra_artifacts.items():
            if path.exists() and path.is_file():
                artifact_map[key] = str(path.resolve())
        experiments.append(
            {
                "id": exp_dir.name,
                "path": str(exp_dir.resolve()),
                "created_at": metadata.get("created_at"),
                "dataset_samples": metadata.get("dataset", {}).get("statistics", {}).get("n_samples"),
                "train_epochs": metadata.get("training", {}).get("training_results", {}).get("epochs_run"),
                "eval_samples": metadata.get("evaluation", {}).get("metrics", {}).get("n_evaluation_samples"),
                "metrics": metrics,
                "diagnostics": results.get("diagnostics", {}),
                "datasets": datasets,
                "parameter_values": parameter_values,
                "has_series": bool(series_info["has_series"]),
                "error_series_mode": series_info["error_mode"],
                "dataset_inspection": dataset_inspection_summary if dataset_inspection_summary else None,
                "artifacts": artifact_map,
            }
        )
    return experiments


@dataclass
class Job:
    id: str
    label: str
    command: str
    cwd: str
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    return_code: int | None
    log_path: str


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return [asdict(job) for job in jobs]

    def read_log(self, job_id: str, tail: int = 40000) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        path = Path(job.log_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-tail:]

    def start(self, label: str, command: str, cwd: Path | None = None) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        cwd = cwd or REPO_ROOT
        log_path = JOB_LOG_DIR / f"{job_id}.log"
        job = Job(
            id=job_id,
            label=label,
            command=command,
            cwd=str(cwd),
            status="queued",
            created_at=time.time(),
            started_at=None,
            finished_at=None,
            return_code=None,
            log_path=str(log_path),
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return asdict(job)

    def _run(self, job_id: str):
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()

        log_path = Path(job.log_path)
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"$ {job.command}\n\n")
            logf.flush()
            process = subprocess.Popen(
                ["bash", "-lc", job.command],
                cwd=job.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                logf.write(line)
                logf.flush()
            return_code = process.wait()

        with self._lock:
            job = self._jobs[job_id]
            job.return_code = int(return_code)
            job.finished_at = time.time()
            job.status = "done" if return_code == 0 else "failed"


JOB_MANAGER = JobManager()


def command_for_training(model_type: str, experiment_id: str) -> tuple[str, str]:
    out_dir = shlex.quote(str(OUT_DIR.resolve()))
    exp = shlex.quote(experiment_id)
    preamble_parts = [
        "source /opt/ros/jazzy/setup.bash",
        "if [ -f ai_slam_ws/install/setup.bash ]; then source ai_slam_ws/install/setup.bash; fi",
    ]
    if VENV_SITE.is_dir():
        preamble_parts.append(f"export PYTHONPATH=\"${{PYTHONPATH:+$PYTHONPATH:}}{VENV_SITE}\"")
    preamble = " && ".join(preamble_parts)

    if model_type == "baseline":
        label = f"Trening AI dla {experiment_id}"
        command = (
            f"{preamble}"
            f" && ros2 run ai_slam_ai train_model --ros-args"
            f" -p out_dir:={out_dir}"
            f" -p experiment_id:={exp}"
            f" -p dataset_name:=dataset.npz"
            f" -p model_name:=model.pt"
            f" -p history_name:=train_history.json"
            f" -p skip_if_model_exists:=false"
        )
        return label, command

    if model_type == "robak":
        label = f"Trening Robaka dla {experiment_id}"
        command = (
            f"{preamble}"
            f" && ros2 run ai_slam_ai train_model_robak --ros-args"
            f" -p out_dir:={out_dir}"
            f" -p experiment_id:={exp}"
            f" -p dataset_name:=dataset_robak.npz"
            f" -p model_name:=model_robak.pt"
            f" -p history_name:=train_history_robak.json"
            f" -p skip_if_model_exists:=false"
            f" -p write_experiment_metadata:=true"
        )
        return label, command

    if model_type == "rywak":
        label = f"Trening Rywaka dla {experiment_id}"
        command = (
            f"{preamble}"
            f" && ros2 run ai_slam_ai train_model_rywak --ros-args"
            f" -p out_dir:={out_dir}"
            f" -p experiment_id:={exp}"
            f" -p dataset_name:=dataset_rywak.npz"
            f" -p model_name:=model_rywak.pt"
            f" -p history_name:=train_history_rywak.json"
            f" -p skip_if_model_exists:=false"
            f" -p write_experiment_metadata:=true"
        )
        return label, command

    raise ValueError(f"Nieznany typ modelu: {model_type}")


def build_job_command(payload: dict[str, Any]) -> tuple[str, str]:
    action = str(payload.get("action", "")).strip()
    if action == "run_all":
        return "Budowanie środowiska i pełny pipeline", "./scripts/run_all.sh"
    if action == "run_full_cycle":
        config = str(payload.get("config", "experiment_config.yaml")).strip() or "experiment_config.yaml"
        return (
            f"Pełny eksperyment: trening i test ({config})",
            f"./scripts/run_full_cycle.sh {shlex.quote(config)}",
        )
    if action == "run_experiment":
        mode = str(payload.get("mode", "")).strip()
        extra_args = str(payload.get("extra_args", "")).strip()
        mode_labels = {
            "fast": "Szybki eksperyment",
            "full": "Pełny eksperyment",
            "dataset": "Tylko dataset",
            "train": "Tylko trening",
            "test": "Tylko test",
        }
        command = "./scripts/run_experiment.sh"
        if mode:
            command += f" {shlex.quote(mode)}"
        if extra_args:
            command += f" {extra_args}"
        return f"Uruchomienie eksperymentu: {mode_labels.get(mode, mode or 'tryb własny')}", command
    if action == "inspect_dataset":
        experiment_id = str(payload.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ValueError("Brak experiment_id dla inspekcji datasetu.")
        dataset_dir = resolve_experiment_dir(experiment_id)
        return (
            f"Generowanie raportu datasetu dla {experiment_id}",
            f"python3 scripts/inspect_dataset.py {shlex.quote(str(dataset_dir))}",
        )
    if action == "train_selected":
        model_type = str(payload.get("model_type", "")).strip()
        experiment_id = str(payload.get("experiment_id", "")).strip()
        if not model_type or not experiment_id:
            raise ValueError("Brak model_type lub experiment_id.")
        return command_for_training(model_type, experiment_id)
    if action == "sweep_parameter":
        source_experiment = str(payload.get("source_experiment", "")).strip()
        config = str(payload.get("config", "experiment_config.yaml")).strip() or "experiment_config.yaml"
        param = str(payload.get("param", "")).strip()
        start = str(payload.get("start", "")).strip()
        stop = str(payload.get("stop", "")).strip()
        step = str(payload.get("step", "")).strip()
        if not source_experiment or not param or not start or not stop or not step:
            raise ValueError("Sweep wymaga eksperymentu zrodlowego, configu, parametru, zakresu od-do i kroku.")
        float(start)
        float(stop)
        float(step)
        return (
            f"Sweep na stalym datasecie {source_experiment}: {param} ({start} -> {stop}, krok {step})",
            "python3 scripts/run_param_sweep.py"
            f" --mode fixed_dataset"
            f" --config {shlex.quote(config)}"
            f" --source-experiment {shlex.quote(source_experiment)}"
            f" --param {shlex.quote(param)}"
            f" --start {shlex.quote(start)}"
            f" --stop {shlex.quote(stop)}"
            f" --step {shlex.quote(step)}",
        )
    if action == "quick_pipeline":
        mode = str(payload.get("mode", "")).strip()
        config = str(payload.get("config", "experiment_config.yaml")).strip() or "experiment_config.yaml"
        run_name = str(payload.get("name", "")).strip()
        dataset_duration = str(payload.get("dataset_duration", "")).strip()
        eval_duration = str(payload.get("eval_duration", "")).strip()
        experiment_id = str(payload.get("experiment_id", "")).strip()
        if mode not in {"dataset", "full_cycle", "train_existing", "test_existing"}:
            raise ValueError("Nieznany tryb szybkiego uruchomienia.")
        if mode in {"dataset", "full_cycle"} and not dataset_duration:
            raise ValueError("Brak wspólnego czasu datasetu.")
        if dataset_duration:
            float(dataset_duration)
        if mode in {"full_cycle", "test_existing"}:
            if not eval_duration:
                raise ValueError("Ten tryb wymaga czasu testu i ewaluacji.")
            float(eval_duration)
        if mode in {"train_existing", "test_existing"} and not experiment_id:
            raise ValueError("Brak wybranego eksperymentu.")

        label_suffix = experiment_id or run_name or config
        command = (
            "python3 scripts/run_dashboard_quick_pipeline.py"
            f" --mode {shlex.quote(mode)}"
            f" --base-config {shlex.quote(config)}"
        )
        if dataset_duration:
            command += f" --dataset-duration {shlex.quote(dataset_duration)}"
        if run_name:
            command += f" --name {shlex.quote(run_name)}"
        if eval_duration:
            command += f" --eval-duration {shlex.quote(eval_duration)}"
        if experiment_id:
            command += f" --experiment-id {shlex.quote(experiment_id)}"
        label_map = {
            "dataset": "Szybki start: tylko datasety",
            "full_cycle": "Szybki start: dataset + trening + test",
            "train_existing": "Szybki start: trening wybranego eksperymentu",
            "test_existing": "Szybki start: test wybranego eksperymentu",
        }
        label = f"{label_map.get(mode, 'Szybki start')} ({label_suffix})"
        return label, command
    if action == "generate_function_index":
        return (
            "Generowanie indeksu funkcji",
            "python3 scripts/generate_function_index.py"
            f" --output-md {shlex.quote(str(FUNCTION_INDEX_MD))}"
            f" --output-json {shlex.quote(str(FUNCTION_INDEX_JSON))}",
        )
    raise ValueError(f"Nieznana akcja: {action}")


def apply_plot_style(fig, ax):
    fig.patch.set_facecolor("#0f1724")
    ax.set_facecolor("#151d2d")
    ax.tick_params(colors="#d9e2f2")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.xaxis.label.set_color("#d9e2f2")
    ax.yaxis.label.set_color("#d9e2f2")
    ax.title.set_color("#f8fafc")
    ax.grid(True, alpha=0.28, color="#475569")


def style_plot_legend(ax):
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#334155")
    for text in legend.get_texts():
        text.set_color("#e5eefc")


def make_placeholder_figure(title: str, message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    apply_plot_style(fig, ax)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, color="#e5eefc")
    ax.set_title(title)
    ax.axis("off")
    return figure_to_png(fig)


def figure_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_trajectory_image(
    experiment_id: str,
    series_names: list[str],
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> bytes:
    try:
        data = load_trajectory_npz(experiment_id)
    except Exception as exc:
        return make_placeholder_figure("Trajektorie", str(exc))

    fig, ax = plt.subplots(figsize=(8, 6))
    apply_plot_style(fig, ax)
    plotted = 0
    for series_name in series_names:
        spec = POSITION_SERIES.get(series_name)
        if spec is None:
            continue
        _time_key, pose_key, label, color = spec
        if pose_key not in data:
            continue
        arr = np.asarray(data[pose_key], dtype=np.float32)
        if arr.size == 0:
            continue
        arr = arr.reshape((-1, 3))
        ax.plot(arr[:, 0], arr[:, 1], label=label, linewidth=1.6, color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return make_placeholder_figure("Trajektorie", f"Brak danych trajektorii dla {experiment_id}.")

    ax.set_title(f"Trajektorie: {experiment_id}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    style_plot_legend(ax)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    return figure_to_png(fig)


def plot_error_image(
    experiment_id: str,
    series_names: list[str],
    metric: str,
    time_min: float | None,
    time_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> bytes:
    try:
        data = load_trajectory_npz(experiment_id)
    except Exception as exc:
        return make_placeholder_figure(
            "Błędy",
            f"{exc}\nNiestandardowy wykres błędu wymaga pliku trajectory_data.npz z ewaluacji.",
        )

    is_orientation = metric.startswith("orientation")
    in_degrees = metric == "orientation_deg"
    y_label = "błąd orientacji [deg]" if in_degrees else ("błąd orientacji [rad]" if is_orientation else "błąd pozycji [m]")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    apply_plot_style(fig, ax)
    plotted = 0
    used_sources: set[str] = set()

    for series_name in series_names:
        spec = ERROR_SERIES.get(series_name)
        if spec is None:
            continue
        _time_key, _err_xy_key, _err_th_key, label, color = spec
        error_series = get_error_series(data, series_name)
        if error_series is None:
            continue
        time_arr, err_xy, err_th, source = error_series
        used_sources.add(source)

        if is_orientation:
            value_arr = np.abs(err_th.reshape((-1,)))
            if in_degrees:
                value_arr = np.rad2deg(value_arr)
        else:
            value_arr = np.sqrt(err_xy[:, 0] ** 2 + err_xy[:, 1] ** 2)

        n = min(time_arr.shape[0], value_arr.shape[0])
        if n <= 0:
            continue
        time_arr = time_arr[:n]
        value_arr = value_arr[:n]

        mask = np.ones(n, dtype=bool)
        if time_min is not None:
            mask &= time_arr >= float(time_min)
        if time_max is not None:
            mask &= time_arr <= float(time_max)
        if not np.any(mask):
            continue

        ax.plot(time_arr[mask], value_arr[mask], label=label, linewidth=1.4, color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return make_placeholder_figure(
            "Błędy",
            (
                f"Brak danych błędów dla {experiment_id}.\n"
                "Dla starszych eksperymentów wykres działa tylko jeśli istnieje trajectory_data.npz."
            ),
        )

    source_note = ""
    if used_sources == {"reconstructed"}:
        source_note = " (odtworzone z trajektorii)"
    elif "reconstructed" in used_sources:
        source_note = " (część serii odtworzona)"

    ax.set_title(f"Błędy: {experiment_id}{source_note}")
    ax.set_xlabel("czas [s]")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    style_plot_legend(ax)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    return figure_to_png(fig)


def plot_comparison_image(group: str, param_key: str, metric_key: str) -> bytes:
    param_spec = PARAM_SPEC_BY_KEY.get(param_key)
    metric_spec = METRIC_SPEC_BY_KEY.get(metric_key)
    if param_spec is None or metric_spec is None:
        return make_placeholder_figure("Porównanie", "Nieznany parametr lub metryka.")
    if param_spec["group"] != group or metric_spec["group"] != group:
        return make_placeholder_figure("Porównanie", "Parametr i metryka nie należą do tej samej rodziny.")

    experiments = discover_experiments()
    points: list[dict[str, Any]] = []
    for exp in experiments:
        value = exp.get("parameter_values", {}).get(param_key)
        metric_value = exp.get("metrics", {}).get(metric_key)
        if value is None or metric_value is None:
            continue
        points.append(
            {
                "experiment_id": exp["id"],
                "x_raw": normalize_json_value(value),
                "x_label": format_param_value(value),
                "y": float(metric_value),
            }
        )

    if len(points) < 2:
        return make_placeholder_figure(
            "Porównanie",
            "Za mało eksperymentów z zapisanym parametrem i metryką, aby narysować wykres.\n"
            "Dla części parametrów potrzebny jest nowszy eksperyment z pełnym snapshotem configu.",
        )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    apply_plot_style(fig, ax)
    color = "#38bdf8" if group == "robak" else ("#34d399" if group == "rywak" else "#f59e0b")

    if param_spec["kind"] == "numeric" and all(is_numeric_param_value(point["x_raw"]) for point in points):
        points.sort(key=lambda item: float(item["x_raw"]))
        x_vals = np.asarray([float(point["x_raw"]) for point in points], dtype=np.float32)
        y_vals = np.asarray([float(point["y"]) for point in points], dtype=np.float32)
        ax.scatter(x_vals, y_vals, s=56, color=color, edgecolors="#e2e8f0", linewidths=0.8, alpha=0.92)

        uniq = sorted(set(float(value) for value in x_vals.tolist()))
        if len(uniq) >= 2:
            mean_x = []
            mean_y = []
            for value in uniq:
                mask = np.isclose(x_vals, value)
                mean_x.append(value)
                mean_y.append(float(y_vals[mask].mean()))
            ax.plot(mean_x, mean_y, color="#f8fafc", linewidth=1.4, alpha=0.9, label="Średnia dla wartości parametru")
            style_plot_legend(ax)

        ax.set_xlabel(param_spec["label"])
    else:
        categories = sorted({point["x_label"] for point in points})
        positions = {label: idx for idx, label in enumerate(categories)}
        grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in categories}
        for point in points:
            grouped[point["x_label"]].append(point)

        x_plot: list[float] = []
        y_plot: list[float] = []
        for label, items in grouped.items():
            base_x = float(positions[label])
            if len(items) == 1:
                x_offsets = [0.0]
            else:
                x_offsets = np.linspace(-0.14, 0.14, len(items)).tolist()
            for offset, item in zip(x_offsets, items):
                x_plot.append(base_x + float(offset))
                y_plot.append(float(item["y"]))

        ax.scatter(x_plot, y_plot, s=56, color=color, edgecolors="#e2e8f0", linewidths=0.8, alpha=0.92)

        mean_x = []
        mean_y = []
        for label in categories:
            values = [float(item["y"]) for item in grouped[label]]
            mean_x.append(positions[label])
            mean_y.append(float(np.mean(values)))
        ax.plot(mean_x, mean_y, color="#f8fafc", linewidth=1.4, alpha=0.9, label="Średnia dla kategorii")
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(categories, rotation=18, ha="right")
        ax.set_xlabel(param_spec["label"])
        style_plot_legend(ax)

    ax.set_title(f"{metric_spec['label']} vs {param_spec['label']} ({len(points)} eksperymentów)")
    ax.set_ylabel(metric_spec["label"])
    return figure_to_png(fig)


def plot_sweep_image(sweep_id: str, family_key: str) -> bytes:
    if not sweep_id:
        return make_placeholder_figure("Analiza sweepa", "Wybierz wynik sweepa.")

    try:
        sweep_dir = resolve_sweep_dir(sweep_id)
    except FileNotFoundError:
        return make_placeholder_figure("Analiza sweepa", f"Nie znaleziono katalogu sweepa: {sweep_id}")

    rows, rows_source = load_sweep_rows(sweep_dir)
    if not rows:
        return make_placeholder_figure(
            "Analiza sweepa",
            f"Brak danych sweepa dla {sweep_id}.",
        )

    family_spec = SWEEP_PLOT_FAMILIES.get(family_key)
    if family_spec is None:
        return make_placeholder_figure(
            "Analiza sweepa",
            "Ten sweep nie ma jeszcze dostępnych metryk do narysowania.\n"
            "Najpierw uruchom sweep, który zakończy się poprawnym testem i ewaluacją.",
        )

    done_rows = [row for row in rows if str(row.get("status", "")).strip() == "done"]
    if not done_rows:
        return make_placeholder_figure(
            "Analiza sweepa",
            "Ten sweep nie ma żadnych udanych przebiegów.\n"
            "Nie da się narysować RMSE ani IoU, dopóki przynajmniej jedna iteracja nie zakończy się sukcesem.",
        )

    param_path = str(done_rows[0].get("param_path") or rows[0].get("param_path") or "parametr")
    param_label = (
        PARAM_SPEC_BY_KEY.get(param_path, {}).get("label")
        or SHARED_SWEEP_PARAM_LABELS.get(param_path)
        or param_path
    )
    source_experiment_id = str(
        done_rows[0].get("source_experiment_id") or rows[0].get("source_experiment_id") or "brak"
    )
    numeric_points = []
    is_numeric = True
    for row in done_rows:
        numeric_value = metric_float(row.get("param_value"))
        if numeric_value is None:
            is_numeric = False
            break
        numeric_points.append((numeric_value, row))

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    apply_plot_style(fig, ax)
    plotted = 0

    if is_numeric:
        numeric_points.sort(key=lambda item: item[0])
        for _series_key, metric_key, label, color in family_spec["series"]:
            x_vals: list[float] = []
            y_vals: list[float] = []
            for x_value, row in numeric_points:
                metric_value = metric_float(row.get(metric_key))
                if metric_value is None:
                    continue
                x_vals.append(float(x_value))
                y_vals.append(float(metric_value))
            if not x_vals:
                continue
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=5.2,
                linewidth=1.8,
                color=color,
                label=label,
            )
            plotted += 1
        ax.set_xlabel(param_label)
    else:
        categories = list(dict.fromkeys(format_param_value(row.get("param_value")) for row in done_rows))
        positions = {label: idx for idx, label in enumerate(categories)}
        for _series_key, metric_key, label, color in family_spec["series"]:
            x_vals: list[float] = []
            y_vals: list[float] = []
            for row in done_rows:
                metric_value = metric_float(row.get(metric_key))
                if metric_value is None:
                    continue
                x_vals.append(float(positions[format_param_value(row.get("param_value"))]))
                y_vals.append(float(metric_value))
            if not x_vals:
                continue
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=5.2,
                linewidth=1.8,
                color=color,
                label=label,
            )
            plotted += 1
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(categories, rotation=18, ha="right")
        ax.set_xlabel(param_label)

    if plotted == 0:
        plt.close(fig)
        return make_placeholder_figure(
            "Analiza sweepa",
            "Sweep istnieje, ale wybrana rodzina metryk nie ma jeszcze żadnych wartości.\n"
            "Sprawdź summary.json albo uruchom sweep ponownie.",
        )

    ax.set_ylabel(family_spec["y_label"])
    source_suffix = " | odtworzone z eksperymentow" if rows_source == "recovered_experiments" else ""
    ax.set_title(
        f"{family_spec['label']} vs {param_label} | dataset: {source_experiment_id} | "
        f"udane: {len(done_rows)}/{len(rows)}{source_suffix}"
    )
    ax.legend(loc="best")
    style_plot_legend(ax)
    return figure_to_png(fig)


HTML_PAGE = """<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SLAM AI Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090d15;
      --panel: rgba(12, 18, 29, 0.94);
      --ink: #edf2ff;
      --muted: #93a1c2;
      --line: rgba(148, 163, 184, 0.18);
      --accent: #a3e635;
      --accent-dark: #4ade80;
      --good: #34d399;
      --bad: #f87171;
      --shadow: 0 24px 52px rgba(2, 6, 23, 0.42);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(163, 230, 53, 0.18), transparent 24%),
        radial-gradient(circle at left center, rgba(52, 211, 153, 0.12), transparent 32%),
        linear-gradient(180deg, #0b1120 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 1760px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: grid;
      gap: 12px;
      padding: 24px 26px;
      margin-bottom: 18px;
      background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.94));
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 0;
      max-width: 82ch;
      color: var(--muted);
    }
    .layout {
      display: grid;
      gap: 18px;
      align-items: start;
      grid-template-columns: minmax(280px, 320px) minmax(0, 1fr);
    }
    .stack {
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .sidebar {
      align-self: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 18px;
      backdrop-filter: blur(10px);
    }
    .panel h2, .panel h3 {
      margin-top: 0;
      margin-bottom: 12px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .panel h2 { font-size: 1.05rem; }
    .panel h3 { font-size: 0.92rem; }
    .controls { display: grid; gap: 10px; }
    label {
      display: grid;
      gap: 6px;
      font-size: 0.92rem;
      color: var(--muted);
    }
    input, select, textarea, button {
      font: inherit;
    }
    input[type="checkbox"] {
      width: auto;
      padding: 0;
    }
    input, select, textarea {
      width: 100%;
      padding: 11px 12px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      background: rgba(10, 15, 25, 0.92);
      color: var(--ink);
    }
    textarea {
      min-height: 260px;
      resize: vertical;
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.84rem;
      line-height: 1.45;
    }
    button {
      cursor: pointer;
      border: none;
      border-radius: 999px;
      padding: 11px 16px;
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: #fffaf5;
      letter-spacing: 0.02em;
    }
    button.secondary {
      background: rgba(148, 163, 184, 0.12);
      color: var(--ink);
    }
    button.ghost {
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.18);
      color: var(--ink);
    }
    .top-actions, .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .workspace-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }
    .workspace-tab {
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(148, 163, 184, 0.18);
      color: var(--ink);
      box-shadow: none;
    }
    .workspace-tab.active {
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: #08110d;
      box-shadow: 0 14px 30px rgba(74, 222, 128, 0.2);
    }
    .workspace-hidden {
      display: none !important;
    }
    .two {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .three {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .checkboxes {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      padding: 6px 0 2px;
    }
    .checkboxes label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--ink);
      width: auto;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.16);
      line-height: 1.1;
    }
    .checkboxes input[type="checkbox"] {
      margin: 0;
      flex: 0 0 auto;
      transform: translateY(-0.5px);
    }
    .metric {
      padding: 14px;
      border-radius: 14px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .metric strong {
      display: block;
      font-size: 1.25rem;
      margin-top: 4px;
    }
    .metric small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
    }
    .metric-columns {
      display: grid;
      gap: 18px;
      grid-template-columns: 1.4fr 1fr;
    }
    .config-toolbar {
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 1fr);
    }
    .config-priority {
      display: grid;
      gap: 14px;
    }
    .config-priority-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }
    .config-priority-card {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(24, 34, 52, 0.96), rgba(15, 23, 42, 0.92));
      border: 1px solid rgba(163, 230, 53, 0.28);
      box-shadow: 0 12px 26px rgba(2, 6, 23, 0.22);
    }
    .config-priority-card strong {
      font-size: 0.96rem;
    }
    .config-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--ink);
      line-height: 1.25;
    }
    .config-priority-badge {
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-dark);
    }
    .config-groups {
      display: grid;
      gap: 14px;
      max-height: 520px;
      overflow: auto;
      padding-right: 4px;
    }
    .config-group {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .config-group h3 {
      margin: 0;
    }
    .config-fields-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .config-field {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 14px;
      background: rgba(12, 18, 29, 0.96);
      border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .config-field input[type="checkbox"] {
      width: auto;
      justify-self: start;
      transform: scale(1.15);
    }
    .config-path {
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.8rem;
      color: var(--accent-dark);
      word-break: break-word;
    }
    .config-type {
      font-size: 0.78rem;
      color: var(--muted);
    }
    .config-help {
      font-size: 0.84rem;
      color: var(--muted);
    }
    .metric-stack, .artifact-list, .dataset-list, .job-list, .info-list {
      display: grid;
      gap: 8px;
    }
    .inspection-summary {
      display: grid;
      gap: 14px;
    }
    .inspection-hero {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    }
    .inspection-kpi {
      padding: 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(21, 31, 47, 0.98), rgba(12, 18, 29, 0.96));
      border: 1px solid rgba(163, 230, 53, 0.22);
      box-shadow: 0 14px 32px rgba(2, 6, 23, 0.22);
    }
    .inspection-kpi-head, .inspection-group-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .inspection-kpi strong {
      display: block;
      margin-top: 6px;
      font-size: 1.34rem;
    }
    .inspection-kpi.status-good, .inspection-group.status-good {
      border-color: rgba(74, 222, 128, 0.38);
      box-shadow: 0 14px 32px rgba(34, 197, 94, 0.12);
    }
    .inspection-kpi.status-medium, .inspection-group.status-medium {
      border-color: rgba(250, 204, 21, 0.34);
      box-shadow: 0 14px 32px rgba(234, 179, 8, 0.1);
    }
    .inspection-kpi.status-bad, .inspection-group.status-bad {
      border-color: rgba(248, 113, 113, 0.34);
      box-shadow: 0 14px 32px rgba(239, 68, 68, 0.12);
    }
    .inspection-groups {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(235px, 1fr));
    }
    .inspection-group {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .inspection-group-head {
      display: grid;
      gap: 4px;
    }
    .inspection-group-head h3 {
      margin: 0;
      font-size: 1rem;
    }
    .inspection-group-head p {
      margin: 0;
      color: var(--muted);
      font-size: 0.85rem;
    }
    .inspection-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .inspection-group .metric {
      padding: 12px;
      min-height: 92px;
    }
    .inspection-group .metric strong {
      font-size: 1.05rem;
    }
    .inspection-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 82px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .inspection-badge.good {
      color: #d1fae5;
      background: rgba(22, 163, 74, 0.18);
      border-color: rgba(74, 222, 128, 0.36);
    }
    .inspection-badge.medium {
      color: #fef3c7;
      background: rgba(202, 138, 4, 0.2);
      border-color: rgba(250, 204, 21, 0.34);
    }
    .inspection-badge.bad {
      color: #fee2e2;
      background: rgba(220, 38, 38, 0.18);
      border-color: rgba(248, 113, 113, 0.34);
    }
    .info-box {
      padding: 14px;
      border-radius: 14px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.14);
      color: var(--ink);
    }
    .info-box code {
      background: rgba(30, 41, 59, 0.92);
      padding: 2px 6px;
      border-radius: 6px;
    }
    .artifact-list a, .link {
      color: var(--accent-dark);
      text-decoration: none;
    }
    .artifact-list a:hover, .link:hover {
      text-decoration: underline;
    }
    .plot-grid {
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    }
    .plot-card {
      display: grid;
      gap: 10px;
    }
    .plot-card img {
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #0b1120;
      min-height: 360px;
      max-height: 680px;
      object-fit: contain;
      cursor: zoom-in;
    }
    .plot-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .status .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--muted);
    }
    .status.running .dot { background: var(--accent); }
    .status.done .dot { background: var(--good); }
    .status.failed .dot { background: var(--bad); }
    .muted { color: var(--muted); }
    .section-note, .flash {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .flash.error { color: var(--bad); }
    .flash.ok { color: var(--good); }
    .image-modal {
      position: fixed;
      inset: 0;
      display: none;
      background: rgba(14, 17, 27, 0.78);
      backdrop-filter: blur(4px);
      z-index: 999;
      padding: 24px;
    }
    .image-modal.open {
      display: grid;
      place-items: center;
    }
    .image-modal-card {
      width: min(96vw, 1600px);
      max-height: 92vh;
      background: rgba(11, 17, 32, 0.98);
      border-radius: 22px;
      padding: 16px;
      display: grid;
      gap: 10px;
      box-shadow: var(--shadow);
    }
    .image-modal-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .image-modal img {
      width: 100%;
      max-height: 80vh;
      object-fit: contain;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #0b1120;
    }
    @media (max-width: 1240px) {
      .layout { grid-template-columns: 1fr; }
      .metric-columns { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
      .plot-grid, .checkboxes, .three, .two, .config-toolbar {
        grid-template-columns: 1fr;
      }
      .shell { padding: 14px; }
      .plot-card img { min-height: 260px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>SLAM AI Dashboard</h1>
      <p>Panel do wyboru zebranych datasetów, uruchamiania pipeline z poziomu przeglądarki, edycji plików konfiguracyjnych oraz generowania wykresów z własnym zakresem czasu i osi. Kliknięcie wykresu otwiera go w powiększeniu.</p>
      <div class="top-actions">
        <button onclick="startJob({action:'run_all'})">Zbuduj środowisko i uruchom pełny pipeline</button>
        <button class="secondary" onclick="startJob({action:'generate_function_index'})">Odśwież indeks funkcji</button>
      </div>
    </section>

    <section class="workspace-tabs">
      <button id="workspace-tab-analysis" class="workspace-tab active" data-workspace-tab="analysis" onclick="setWorkspace('analysis')">Analiza</button>
      <button id="workspace-tab-experiments" class="workspace-tab" data-workspace-tab="experiments" onclick="setWorkspace('experiments')">Eksperymenty</button>
      <button id="workspace-tab-settings" class="workspace-tab" data-workspace-tab="settings" onclick="setWorkspace('settings')">Ustawienia</button>
    </section>

    <div class="layout">
      <aside class="stack sidebar">
        <section class="panel controls" data-workspaces="analysis experiments">
          <h2>Eksperyment</h2>
          <label>Wybierz eksperyment
            <select id="experiment-select" onchange="renderExperiment()"></select>
          </label>
          <div id="experiment-meta" class="muted"></div>
          <div class="dataset-list" id="dataset-list"></div>
        </section>

        <section class="panel controls" data-workspaces="experiments">
          <h2>Szybki Start</h2>
          <p class="section-note" id="quick-launch-config-note">Ładowanie ustawień szybkiego startu...</p>
          <label>Nazwa nowego uruchomienia
            <input id="quick-experiment-name" placeholder="np. robak_porownanie_01">
          </label>
          <div class="two">
            <label>Wspólny czas datasetu [s]
              <input id="quick-dataset-duration" placeholder="np. 30.0">
            </label>
            <label>Czas testu i ewaluacji [s]
              <input id="quick-eval-duration" placeholder="np. 100.0">
            </label>
          </div>
          <div class="button-row">
            <button class="secondary" onclick="startQuickPipeline('dataset')">Zbierz tylko datasety</button>
            <button onclick="startQuickPipeline('full_cycle')">Dataset + trening + test</button>
          </div>
          <div class="button-row">
            <button class="secondary" onclick="startQuickPipeline('train_existing')">Trenuj wybrany eksperyment</button>
            <button class="secondary" onclick="startQuickPipeline('test_existing')">Testuj wybrany eksperyment</button>
            <button class="secondary" onclick="inspectSelected()">Raport datasetu</button>
          </div>
          <p class="section-note">Nowe uruchomienia używają nazwy z pola powyżej. Trening i test korzystają z aktualnie wybranego eksperymentu z listy po lewej. Test zapisze nowe wyniki w folderze tego eksperymentu.</p>
        </section>

        <section class="panel controls" data-workspaces="experiments">
          <h2>Zadania</h2>
          <div id="job-list" class="job-list"></div>
          <label>Log zadania
            <textarea id="job-log" readonly></textarea>
          </label>
        </section>

        <section class="panel controls" data-workspaces="settings">
          <h2>Indeks funkcji</h2>
          <p class="section-note" id="function-summary">Ładowanie indeksu funkcji...</p>
          <div class="artifact-list">
            <a id="function-md-link" class="link" target="_blank">Otwórz Markdown</a>
            <a id="function-json-link" class="link" target="_blank">Otwórz JSON</a>
          </div>
        </section>
      </aside>

      <main class="stack">
        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Metryki</h2>
            <p class="section-note">Błędy RMSE są po lewej, IoU map po prawej.</p>
          </div>
          <div class="metric-columns">
            <div class="stack">
              <h3>RMSE</h3>
              <div id="rmse-grid" class="metric-stack"></div>
            </div>
            <div class="stack">
              <h3>IoU</h3>
              <div id="iou-grid" class="metric-stack"></div>
              <div class="info-box">
                <strong>Jak liczone jest IoU</strong>
                <div class="info-list">
                  <div>1. Mapa SLAM jest rzutowana do siatki mapy referencyjnej.</div>
                  <div>2. Komórka jest zajęta, gdy occupancy spełnia <code>value &gt;= 50</code>.</div>
                  <div>3. Komórki <code>-1</code> są traktowane jako nieznane i nie wchodzą do porównania.</div>
                  <div>4. Liczone jest <code>|ref ∩ pred| / |ref ∪ pred|</code> tylko na znanych komórkach.</div>
                  <div>5. Dla fallbacku punktowego Robak/Rywak maska znanych komórek to tylko komórki, do których faktycznie dodano punkty.</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Raport Datasetu</h2>
            <p class="section-note">Raport generowany jest dla <code>dataset.npz</code> z wybranego eksperymentu i zapisuje czytelne artefakty do folderu eksperymentu.</p>
          </div>
          <div class="button-row">
            <button class="secondary" onclick="inspectSelected()">Wygeneruj / odśwież raport datasetu</button>
            <a id="dataset-inspection-summary-link" class="link" target="_blank">Otwórz summary.json</a>
          </div>
          <p class="section-note" id="dataset-inspection-note">Brak raportu datasetu dla wybranego eksperymentu.</p>
          <div id="dataset-inspection-summary" class="inspection-summary"></div>
          <div class="plot-grid">
            <div class="plot-card">
              <div class="plot-actions">
                <h3>Widok ogólny datasetu</h3>
                <button class="ghost" onclick="openImageModalById('dataset-inspection-overview', 'Raport datasetu - widok ogólny')">Powiększ</button>
              </div>
              <img id="dataset-inspection-overview" alt="Raport datasetu - widok ogólny" onclick="openImageModalById('dataset-inspection-overview', 'Raport datasetu - widok ogólny')">
            </div>
            <div class="plot-card">
              <div class="plot-actions">
                <h3>Reprezentatywne skany</h3>
                <button class="ghost" onclick="openImageModalById('dataset-inspection-scans', 'Raport datasetu - skany')">Powiększ</button>
              </div>
              <img id="dataset-inspection-scans" alt="Raport datasetu - skany" onclick="openImageModalById('dataset-inspection-scans', 'Raport datasetu - skany')">
            </div>
          </div>
        </section>

        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Porównanie Parametrów</h2>
            <p class="section-note">Wykres pokazuje RMSE albo IoU względem wybranego parametru Robaka, Rywaka lub filtracji map. Starsze eksperymenty bez snapshotu configu mogą nie mieć pełnego kompletu parametrów.</p>
          </div>
          <div class="three">
            <label>Rodzina
              <select id="compare-group" onchange="renderComparisonOptions()"></select>
            </label>
            <label>Parametr
              <select id="compare-param" onchange="refreshComparisonPlot()"></select>
            </label>
            <label>Metryka
              <select id="compare-metric" onchange="refreshComparisonPlot()"></select>
            </label>
          </div>
          <p class="section-note" id="comparison-note">Ładowanie danych porównań...</p>
          <div class="plot-grid">
            <div class="plot-card">
              <div class="plot-actions">
                <h3>RMSE / IoU vs parametr</h3>
                <button class="ghost" onclick="openImageModalById('comparison-plot', 'Porównanie parametrów')">Powiększ</button>
              </div>
              <img id="comparison-plot" alt="Porównanie parametrów" onclick="openImageModalById('comparison-plot', 'Porównanie parametrów')">
            </div>
          </div>
        </section>

        <section class="panel stack" data-workspaces="experiments">
          <div>
            <h2>Sweep Parametru Na Stałym Datasecie</h2>
            <p class="section-note">Ta funkcja klonuje jeden wybrany eksperyment jako źródło datasetów, a potem uruchamia serię treningów i testów tylko na tym samym zbiorze. Wybierane są wyłącznie parametry, które mają sens bez ponownego nagrywania datasetu.</p>
          </div>
          <div class="two">
            <label>Eksperyment źródłowy datasetu
              <select id="sweep-source-experiment" onchange="renderSweepOptions()"></select>
            </label>
            <label>Config bazowy / fallback
              <select id="sweep-config" onchange="renderSweepOptions()"></select>
            </label>
          </div>
          <div class="two">
            <label>Rodzina parametru
              <select id="sweep-group" onchange="renderSweepOptions()"></select>
            </label>
            <label>Parametr sweepu
              <select id="sweep-param" onchange="renderSweepDefaults(true)"></select>
            </label>
          </div>
          <div class="three">
            <label>Zakres od
              <input id="sweep-start" placeholder="np. 0.02">
            </label>
            <label>Zakres do
              <input id="sweep-stop" placeholder="np. 0.12">
            </label>
            <label>Krok
              <input id="sweep-step" placeholder="np. 0.01">
            </label>
          </div>
          <div class="button-row">
            <button onclick="startSweep()">Uruchom serię treningów i testów</button>
          </div>
          <p class="section-note" id="sweep-note">Wybierz eksperyment źródłowy, parametr oraz zakres wartości.</p>
        </section>

        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Analiza Jednego Sweepa</h2>
            <p class="section-note">Tutaj porównujesz wyniki tylko dla jednego, konkretnego sweepa uruchomionego na stałym źródłowym datasecie.</p>
          </div>
          <div class="two">
            <label>Wynik sweepa
              <select id="sweep-result-select" onchange="renderSweepResultOptions()"></select>
            </label>
            <label>Rodzina metryk
              <select id="sweep-result-family" onchange="refreshSweepResultPlot()"></select>
            </label>
          </div>
          <p class="section-note" id="sweep-result-note">Ładowanie wyników sweepów...</p>
          <div class="artifact-list">
            <a id="sweep-summary-json-link" class="link" target="_blank">Otwórz summary.json</a>
            <a id="sweep-summary-csv-link" class="link" target="_blank">Otwórz summary.csv</a>
          </div>
          <div class="plot-grid">
            <div class="plot-card">
              <div class="plot-actions">
                <h3>RMSE / IoU dla jednego sweepa</h3>
                <button class="ghost" onclick="openImageModalById('sweep-result-plot', 'Analiza jednego sweepa')">Powiększ</button>
              </div>
              <img id="sweep-result-plot" alt="Analiza jednego sweepa" onclick="openImageModalById('sweep-result-plot', 'Analiza jednego sweepa')">
            </div>
          </div>
        </section>

        <section class="panel stack" data-workspaces="settings">
          <div>
            <h2>Edytor Config</h2>
            <p class="section-note">Pola mają opis po polsku, a pod spodem widać dokładną ścieżkę zapisu w YAML. Na górze są najważniejsze ustawienia Robaka, Rywaka oraz filtracji datasetu i map.</p>
          </div>
          <div class="two">
            <label>Plik konfiguracyjny
              <select id="config-select" onchange="loadConfigEditor(true)"></select>
            </label>
            <div class="button-row" style="align-self:end;">
              <button class="secondary" onclick="loadConfigEditor(true)">Wczytaj</button>
              <button onclick="saveConfigEditor()">Zapisz</button>
            </div>
          </div>
          <p id="config-status" class="flash">Brak danych.</p>
          <div id="config-priority" class="config-priority"></div>
          <div class="config-toolbar">
            <label>Filtr parametrów
              <input id="config-filter" placeholder="np. evaluation.points, robak.min_pair" oninput="renderConfigFields()">
            </label>
          </div>
          <div id="config-fields" class="config-groups"></div>
          <textarea id="config-editor" spellcheck="false" oninput="markConfigDirty()"></textarea>
        </section>

        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Wykres trajektorii</h2>
            <p class="section-note">Zakres osi X/Y podajesz w metrach. Kliknięcie na obraz otwiera pełny podgląd.</p>
          </div>
          <div class="checkboxes" id="trajectory-series"></div>
          <div class="two">
            <label>X min [m]<input id="traj-x-min" placeholder="-2.0"></label>
            <label>X max [m]<input id="traj-x-max" placeholder="2.0"></label>
          </div>
          <div class="two">
            <label>Y min [m]<input id="traj-y-min" placeholder="-2.0"></label>
            <label>Y max [m]<input id="traj-y-max" placeholder="2.0"></label>
          </div>
          <button onclick="refreshPlots()">Generuj wykresy</button>
          <div class="plot-grid">
            <div class="plot-card">
              <div class="plot-actions">
                <h3>Trajektoria niestandardowa</h3>
                <button class="ghost" onclick="openImageModalById('trajectory-custom', 'Trajektoria niestandardowa')">Powiększ</button>
              </div>
              <img id="trajectory-custom" alt="Trajektoria niestandardowa" onclick="openImageModalById('trajectory-custom', 'Trajektoria niestandardowa')">
            </div>
            <div class="plot-card">
              <div class="plot-actions">
                <h3>`trajectory.png`</h3>
                <button class="ghost" onclick="openImageModalById('trajectory-static', 'trajectory.png')">Powiększ</button>
              </div>
              <img id="trajectory-static" alt="trajectory.png" onclick="openImageModalById('trajectory-static', 'trajectory.png')">
            </div>
          </div>
        </section>

        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Wykres błędu</h2>
            <p class="section-note">Na osi OY jednostka jest jawna: metry dla błędu pozycji albo radiany/stopnie dla błędu orientacji.</p>
            <p class="flash" id="error-availability">Sprawdzanie dostępności danych błędu...</p>
          </div>
          <div class="checkboxes" id="error-series"></div>
          <div class="three">
            <label>Typ błędu
              <select id="error-metric">
                <option value="position_m">Pozycja [m]</option>
                <option value="orientation_rad">Orientacja [rad]</option>
                <option value="orientation_deg">Orientacja [deg]</option>
              </select>
            </label>
            <label>Czas min [s]<input id="err-time-min" placeholder="0"></label>
            <label>Czas max [s]<input id="err-time-max" placeholder="60"></label>
          </div>
          <div class="two">
            <label>OY min<input id="err-y-min" placeholder="0"></label>
            <label>OY max<input id="err-y-max" placeholder="1.0"></label>
          </div>
          <div class="plot-grid">
            <div class="plot-card">
              <div class="plot-actions">
                <h3>Błąd niestandardowy</h3>
                <button class="ghost" onclick="openImageModalById('error-custom', 'Błąd niestandardowy')">Powiększ</button>
              </div>
              <img id="error-custom" alt="Błąd niestandardowy" onclick="openImageModalById('error-custom', 'Błąd niestandardowy')">
            </div>
            <div class="plot-card">
              <div class="plot-actions">
                <h3>`errors.png`</h3>
                <button class="ghost" onclick="openImageModalById('error-static', 'errors.png')">Powiększ</button>
              </div>
              <img id="error-static" alt="errors.png" onclick="openImageModalById('error-static', 'errors.png')">
            </div>
          </div>
        </section>

        <section class="panel stack" data-workspaces="analysis">
          <div>
            <h2>Mapa i diagnostyka</h2>
            <p class="section-note">Poniżej widać też diagnostykę filtrowania punktów fallback mapy dla Robaka i Rywaka.</p>
          </div>
          <div id="map-diagnostics" class="muted"></div>
          <div class="plot-grid">
            <div class="plot-card">
              <div class="plot-actions">
                <h3>`maps.png`</h3>
                <button class="ghost" onclick="openImageModalById('maps-static', 'maps.png')">Powiększ</button>
              </div>
              <img id="maps-static" alt="maps.png" onclick="openImageModalById('maps-static', 'maps.png')">
            </div>
          </div>
        </section>

        <section class="panel" data-workspaces="analysis">
          <h2>Artefakty</h2>
          <div id="artifact-list" class="artifact-list"></div>
        </section>
      </main>
    </div>
  </div>

  <div id="image-modal" class="image-modal" onclick="closeImageModal(event)">
    <div class="image-modal-card" onclick="event.stopPropagation()">
      <div class="image-modal-head">
        <strong id="image-modal-title">Podgląd</strong>
        <button class="ghost" onclick="closeImageModal()">Zamknij</button>
      </div>
      <img id="image-modal-img" alt="Podgląd">
    </div>
  </div>

  <script>
    const state = {
      experiments: [],
      jobs: [],
      function_index: null,
      configs: [],
      comparisonCatalog: null,
      sweeps: [],
      selectedJobId: null,
      currentConfigName: null,
      currentConfigParsed: null,
      currentWorkspace: 'analysis',
      configDirty: false,
      configRenderTimer: null,
    };
    const TRAJ_DEFAULT = ["gt", "baseline", "ai", "robak", "rywak"];
    const ERR_DEFAULT = ["baseline", "ai", "robak", "rywak"];
    const CONFIG_GROUP_LABELS = {
      experiment: 'Eksperyment',
      simulation: 'Symulacja i świat',
      tracks: 'Tory porównawcze',
      scan_matcher: 'ScanMatcher',
      shared: 'Wspólne Robak + Rywak',
      robak: 'Robak',
      rywak: 'Rywak',
      timing: 'Czasowanie',
      dataset: 'Dataset',
      ground_truth: 'Pozycja referencyjna',
      training: 'Trening AI',
      inference: 'Inferencja AI',
      slam: 'SLAM Toolbox',
      evaluation: 'Ewaluacja',
      odometry: 'Odometria wejściowa',
      driver: 'Sterownik robota',
      output: 'Wyjście',
    };
    const CONFIG_SUFFIX_LABELS = {
      offsets: 'Przesunięcia par skanów',
      lr: 'Learning rate',
      max_epochs: 'Maksymalna liczba epok',
      batch_size: 'Rozmiar batcha',
      val_ratio: 'Udział walidacji',
      min_pair_dist: 'Minimalne przesunięcie między skanami w parze',
      min_pair_dyaw: 'Minimalna rotacja między skanami w parze',
      min_pair_dt_sec: 'Minimalny odstęp czasu między skanami w parze',
      pair_filter_mode: 'Tryb filtracji par skanów',
      max_pair_dist: 'Maksymalne przesunięcie pary skanów',
      max_pair_dyaw: 'Maksymalna rotacja pary skanów',
      augment_noise_std_scale: 'Skala szumu augmentacji',
      augment_cut_fraction: 'Udział maskowania skanu',
      augment_cut_max_points: 'Maksymalna liczba maskowanych punktów',
      infer_delta_ema_alpha: 'Wygładzanie predykcji delta',
      infer_odom_heading_alpha: 'Dociąganie yaw do odometrii',
      infer_odom_delta_xy_alpha: 'Mieszanie delta XY z odometrią',
      infer_odom_delta_yaw_alpha: 'Mieszanie delta yaw z odometrią',
      infer_odom_pose_xy_alpha: 'Bazowe kotwiczenie pozycji XY do odometrii',
      infer_odom_pose_xy_gain: 'Dodatkowe kotwiczenie XY przy dryfcie',
      dataset_duration: 'Czas zbierania datasetu',
      max_samples: 'Maksymalna liczba próbek',
      min_sample_dist: 'Minimalne przesunięcie między próbkami',
      min_sample_dyaw: 'Minimalna rotacja między próbkami',
      min_sample_dt_sec: 'Minimalny odstęp czasu między próbkami',
      min_delta_scan_rms: 'Minimalna różnica RMS skanu',
      sample_filter_mode: 'Tryb filtracji próbek',
      delta_scan_clip: 'Ograniczenie różnicy skanu',
      hidden_dims: 'Rozmiary warstw ukrytych',
      dropout: 'Dropout sieci',
      weight_decay: 'Regularizacja weight decay',
      huber_delta: 'Delta funkcji Hubera',
      input_noise_std: 'Szum wejściowy',
      clip_grad_norm: 'Limit normy gradientu',
      loss_v_weight: 'Waga błędu prędkości liniowej',
      loss_w_weight: 'Waga błędu prędkości kątowej',
      fuse_odom_v_weight: 'Bazowa waga odometrii dla v',
      fuse_odom_w_weight: 'Bazowa waga odometrii dla w',
      fuse_odom_v_gain: 'Wzmocnienie odometrii dla v',
      fuse_odom_w_gain: 'Wzmocnienie odometrii dla w',
      vel_ema_alpha: 'Wygładzanie EMA prędkości',
      anchor_yaw_to_odom: 'Kotwiczenie yaw do odometrii',
      anchor_xy_to_odom: 'Bazowe kotwiczenie XY do odometrii',
      anchor_xy_to_odom_gain: 'Dodatkowe kotwiczenie XY do odometrii',
      points_min_translation: 'Minimalne przesunięcie do dodania punktów mapy',
      points_min_rotation: 'Minimalna rotacja do dodania punktów mapy',
      points_min_time_gap_sec: 'Minimalny odstęp czasu między stemplowaniami punktów',
      points_filter_mode: 'Tryb filtracji dodawania punktów mapy',
      mode: 'Tryb pracy',
      phase: 'Faza eksperymentu',
      seed: 'Seed losowy',
      gui: 'Widok GUI Gazebo',
    };
    const CONFIG_FIELD_METADATA = {
      'shared.lr': {
        label: 'Wspólny learning rate Robaka i Rywaka',
        category: 'Robak + Rywak · trening wspólny',
        sweep: { group: 'shared', start: '0.0002', stop: '0.0020', step: '0.0002' },
      },
      'shared.max_epochs': {
        label: 'Wspólna liczba epok Robaka i Rywaka',
        category: 'Robak + Rywak · trening wspólny',
        sweep: { group: 'shared', start: '50', stop: '250', step: '25' },
      },
      'shared.batch_size': {
        label: 'Wspólny batch size Robaka i Rywaka',
        category: 'Robak + Rywak · trening wspólny',
        sweep: { group: 'shared', start: '32', stop: '256', step: '32' },
      },
      'shared.val_ratio': {
        label: 'Wspólny udział walidacji Robaka i Rywaka',
        category: 'Robak + Rywak · trening wspólny',
        sweep: { group: 'shared', start: '0.10', stop: '0.35', step: '0.05' },
      },
      'robak.offsets': {
        category: 'Robak · inferencja',
      },
      'robak.lr': {
        category: 'Robak · trening',
        sweep: { group: 'robak', start: '0.0002', stop: '0.0020', step: '0.0002' },
      },
      'robak.max_epochs': {
        category: 'Robak · trening',
        sweep: { group: 'robak', start: '50', stop: '250', step: '25' },
      },
      'robak.batch_size': {
        category: 'Robak · trening',
        sweep: { group: 'robak', start: '32', stop: '256', step: '32' },
      },
      'robak.val_ratio': {
        category: 'Robak · trening',
        sweep: { group: 'robak', start: '0.10', stop: '0.35', step: '0.05' },
      },
      'robak.min_pair_dist': {
        category: 'Robak · filtracja datasetu',
        sweep: { group: 'robak', start: '0.01', stop: '0.10', step: '0.01' },
      },
      'robak.min_pair_dyaw': {
        category: 'Robak · filtracja datasetu',
        sweep: { group: 'robak', start: '0.01', stop: '0.20', step: '0.01' },
      },
      'robak.min_pair_dt_sec': {
        category: 'Robak · filtracja datasetu',
        sweep: { group: 'robak', start: '0.0', stop: '0.5', step: '0.05' },
      },
      'robak.pair_filter_mode': {
        category: 'Robak · filtracja datasetu',
      },
      'robak.max_pair_dist': {
        category: 'Robak · filtracja datasetu',
        sweep: { group: 'robak', start: '0.04', stop: '0.20', step: '0.02' },
      },
      'robak.max_pair_dyaw': {
        category: 'Robak · filtracja datasetu',
        sweep: { group: 'robak', start: '0.10', stop: '0.50', step: '0.05' },
      },
      'robak.augment_noise_std_scale': {
        category: 'Robak · augmentacja',
        sweep: { group: 'robak', start: '0.0', stop: '0.05', step: '0.005' },
      },
      'robak.augment_cut_fraction': {
        category: 'Robak · augmentacja',
        sweep: { group: 'robak', start: '0.0', stop: '0.35', step: '0.05' },
      },
      'robak.augment_cut_max_points': {
        category: 'Robak · augmentacja',
      },
      'robak.infer_delta_ema_alpha': {
        category: 'Robak · inferencja',
        sweep: { group: 'robak', start: '0.10', stop: '0.90', step: '0.10' },
      },
      'robak.infer_odom_heading_alpha': {
        category: 'Robak · inferencja',
        sweep: { group: 'robak', start: '0.0', stop: '0.60', step: '0.05' },
      },
      'robak.infer_odom_delta_xy_alpha': {
        category: 'Robak · inferencja',
        sweep: { group: 'robak', start: '0.10', stop: '0.90', step: '0.10' },
      },
      'robak.infer_odom_delta_yaw_alpha': {
        category: 'Robak · inferencja',
        sweep: { group: 'robak', start: '0.10', stop: '0.90', step: '0.10' },
      },
      'robak.infer_odom_pose_xy_alpha': {
        category: 'Robak · inferencja',
        sweep: { group: 'robak', start: '0.0', stop: '0.30', step: '0.02' },
      },
      'robak.infer_odom_pose_xy_gain': {
        category: 'Robak · inferencja',
        sweep: { group: 'robak', start: '0.0', stop: '0.40', step: '0.04' },
      },
      'rywak.lr': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '0.0002', stop: '0.0020', step: '0.0002' },
      },
      'rywak.max_epochs': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '50', stop: '250', step: '25' },
      },
      'rywak.batch_size': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '32', stop: '256', step: '32' },
      },
      'rywak.val_ratio': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '0.10', stop: '0.35', step: '0.05' },
      },
      'rywak.min_sample_dist': {
        category: 'Rywak · filtracja datasetu',
        sweep: { group: 'rywak', start: '0.01', stop: '0.10', step: '0.01' },
      },
      'rywak.min_sample_dyaw': {
        category: 'Rywak · filtracja datasetu',
        sweep: { group: 'rywak', start: '0.01', stop: '0.20', step: '0.01' },
      },
      'rywak.min_sample_dt_sec': {
        category: 'Rywak · filtracja datasetu',
        sweep: { group: 'rywak', start: '0.0', stop: '0.5', step: '0.05' },
      },
      'rywak.min_delta_scan_rms': {
        category: 'Rywak · filtracja datasetu',
        sweep: { group: 'rywak', start: '0.0', stop: '0.20', step: '0.01' },
      },
      'rywak.sample_filter_mode': {
        category: 'Rywak · filtracja datasetu',
      },
      'rywak.delta_scan_clip': {
        category: 'Rywak · wejście modelu',
        sweep: { group: 'rywak', start: '0.5', stop: '3.0', step: '0.25' },
      },
      'rywak.hidden_dims': {
        category: 'Rywak · architektura',
      },
      'rywak.dropout': {
        category: 'Rywak · architektura',
        sweep: { group: 'rywak', start: '0.0', stop: '0.50', step: '0.05' },
      },
      'rywak.weight_decay': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '0.0', stop: '0.0010', step: '0.0001' },
      },
      'rywak.huber_delta': {
        category: 'Rywak · funkcja straty',
        sweep: { group: 'rywak', start: '0.2', stop: '2.0', step: '0.2' },
      },
      'rywak.input_noise_std': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '0.0', stop: '0.10', step: '0.01' },
      },
      'rywak.clip_grad_norm': {
        category: 'Rywak · trening',
        sweep: { group: 'rywak', start: '0.2', stop: '3.0', step: '0.2' },
      },
      'rywak.loss_v_weight': {
        category: 'Rywak · funkcja straty',
        sweep: { group: 'rywak', start: '0.2', stop: '3.0', step: '0.2' },
      },
      'rywak.loss_w_weight': {
        category: 'Rywak · funkcja straty',
        sweep: { group: 'rywak', start: '0.2', stop: '3.0', step: '0.2' },
      },
      'rywak.fuse_odom_v_weight': {
        category: 'Rywak · fuzja z odometrią',
        sweep: { group: 'rywak', start: '0.0', stop: '1.0', step: '0.05' },
      },
      'rywak.fuse_odom_w_weight': {
        category: 'Rywak · fuzja z odometrią',
        sweep: { group: 'rywak', start: '0.0', stop: '1.0', step: '0.05' },
      },
      'rywak.fuse_odom_v_gain': {
        category: 'Rywak · fuzja z odometrią',
        sweep: { group: 'rywak', start: '0.0', stop: '1.0', step: '0.05' },
      },
      'rywak.fuse_odom_w_gain': {
        category: 'Rywak · fuzja z odometrią',
        sweep: { group: 'rywak', start: '0.0', stop: '1.0', step: '0.05' },
      },
      'rywak.vel_ema_alpha': {
        category: 'Rywak · wygładzanie',
        sweep: { group: 'rywak', start: '0.10', stop: '0.95', step: '0.05' },
      },
      'rywak.anchor_yaw_to_odom': {
        category: 'Rywak · kotwiczenie',
        sweep: { group: 'rywak', start: '0.0', stop: '1.0', step: '0.05' },
      },
      'rywak.anchor_xy_to_odom': {
        category: 'Rywak · kotwiczenie',
        sweep: { group: 'rywak', start: '0.0', stop: '0.30', step: '0.02' },
      },
      'rywak.anchor_xy_to_odom_gain': {
        category: 'Rywak · kotwiczenie',
        sweep: { group: 'rywak', start: '0.0', stop: '0.40', step: '0.04' },
      },
      'evaluation.points_min_translation': {
        category: 'Mapa · filtracja punktów',
        sweep: { group: 'map_filter', start: '0.01', stop: '0.15', step: '0.01' },
      },
      'evaluation.points_min_rotation': {
        category: 'Mapa · filtracja punktów',
        sweep: { group: 'map_filter', start: '0.01', stop: '0.20', step: '0.01' },
      },
      'evaluation.points_min_time_gap_sec': {
        category: 'Mapa · filtracja punktów',
        sweep: { group: 'map_filter', start: '0.0', stop: '0.5', step: '0.05' },
      },
      'evaluation.points_filter_mode': {
        category: 'Mapa · filtracja punktów',
      },
    };
    const CONFIG_PRIORITY_FIELDS = [
      { path: ['robak', 'lr'], category: 'Robak · trening' },
      { path: ['robak', 'max_epochs'], category: 'Robak · trening' },
      { path: ['robak', 'batch_size'], category: 'Robak · trening' },
      { path: ['robak', 'val_ratio'], category: 'Robak · trening' },
      { path: ['robak', 'min_pair_dist'], category: 'Robak · filtracja datasetu' },
      { path: ['robak', 'min_pair_dyaw'], category: 'Robak · filtracja datasetu' },
      { path: ['robak', 'min_pair_dt_sec'], category: 'Robak · filtracja datasetu' },
      { path: ['robak', 'pair_filter_mode'], category: 'Robak · filtracja datasetu' },
      { path: ['robak', 'max_pair_dist'], category: 'Robak · filtracja datasetu' },
      { path: ['robak', 'max_pair_dyaw'], category: 'Robak · filtracja datasetu' },
      { path: ['robak', 'augment_noise_std_scale'], category: 'Robak · augmentacja' },
      { path: ['robak', 'augment_cut_fraction'], category: 'Robak · augmentacja' },
      { path: ['robak', 'infer_delta_ema_alpha'], category: 'Robak · inferencja' },
      { path: ['robak', 'infer_odom_heading_alpha'], category: 'Robak · inferencja' },
      { path: ['robak', 'infer_odom_delta_xy_alpha'], category: 'Robak · inferencja' },
      { path: ['robak', 'infer_odom_delta_yaw_alpha'], category: 'Robak · inferencja' },
      { path: ['robak', 'infer_odom_pose_xy_alpha'], category: 'Robak · inferencja' },
      { path: ['robak', 'infer_odom_pose_xy_gain'], category: 'Robak · inferencja' },
      { path: ['rywak', 'lr'], category: 'Rywak · trening' },
      { path: ['rywak', 'max_epochs'], category: 'Rywak · trening' },
      { path: ['rywak', 'batch_size'], category: 'Rywak · trening' },
      { path: ['rywak', 'val_ratio'], category: 'Rywak · trening' },
      { path: ['rywak', 'min_sample_dist'], category: 'Rywak · filtracja datasetu' },
      { path: ['rywak', 'min_sample_dyaw'], category: 'Rywak · filtracja datasetu' },
      { path: ['rywak', 'min_sample_dt_sec'], category: 'Rywak · filtracja datasetu' },
      { path: ['rywak', 'min_delta_scan_rms'], category: 'Rywak · filtracja datasetu' },
      { path: ['rywak', 'sample_filter_mode'], category: 'Rywak · filtracja datasetu' },
      { path: ['rywak', 'delta_scan_clip'], category: 'Rywak · wejście modelu' },
      { path: ['rywak', 'hidden_dims'], category: 'Rywak · architektura' },
      { path: ['rywak', 'dropout'], category: 'Rywak · architektura' },
      { path: ['rywak', 'weight_decay'], category: 'Rywak · trening' },
      { path: ['rywak', 'huber_delta'], category: 'Rywak · funkcja straty' },
      { path: ['rywak', 'loss_v_weight'], category: 'Rywak · funkcja straty' },
      { path: ['rywak', 'loss_w_weight'], category: 'Rywak · funkcja straty' },
      { path: ['rywak', 'fuse_odom_v_weight'], category: 'Rywak · fuzja z odometrią' },
      { path: ['rywak', 'fuse_odom_w_weight'], category: 'Rywak · fuzja z odometrią' },
      { path: ['rywak', 'fuse_odom_v_gain'], category: 'Rywak · fuzja z odometrią' },
      { path: ['rywak', 'fuse_odom_w_gain'], category: 'Rywak · fuzja z odometrią' },
      { path: ['rywak', 'vel_ema_alpha'], category: 'Rywak · wygładzanie' },
      { path: ['rywak', 'anchor_yaw_to_odom'], category: 'Rywak · kotwiczenie' },
      { path: ['rywak', 'anchor_xy_to_odom'], category: 'Rywak · kotwiczenie' },
      { path: ['rywak', 'anchor_xy_to_odom_gain'], category: 'Rywak · kotwiczenie' },
      { path: ['evaluation', 'points_min_translation'], category: 'Mapa · filtracja punktów' },
      { path: ['evaluation', 'points_min_rotation'], category: 'Mapa · filtracja punktów' },
      { path: ['evaluation', 'points_min_time_gap_sec'], category: 'Mapa · filtracja punktów' },
      { path: ['evaluation', 'points_filter_mode'], category: 'Mapa · filtracja punktów' },
    ];
    const DATASET_SWEEP_KEYS = new Set([
      'shared.lr',
      'shared.max_epochs',
      'shared.batch_size',
      'shared.val_ratio',
      'robak.lr',
      'robak.max_epochs',
      'robak.batch_size',
      'robak.val_ratio',
      'robak.infer_delta_ema_alpha',
      'robak.infer_odom_heading_alpha',
      'robak.infer_odom_delta_xy_alpha',
      'robak.infer_odom_delta_yaw_alpha',
      'robak.infer_odom_pose_xy_alpha',
      'robak.infer_odom_pose_xy_gain',
      'rywak.lr',
      'rywak.max_epochs',
      'rywak.batch_size',
      'rywak.val_ratio',
      'rywak.delta_scan_clip',
      'rywak.dropout',
      'rywak.weight_decay',
      'rywak.huber_delta',
      'rywak.input_noise_std',
      'rywak.clip_grad_norm',
      'rywak.loss_v_weight',
      'rywak.loss_w_weight',
      'rywak.fuse_odom_v_weight',
      'rywak.fuse_odom_w_weight',
      'rywak.fuse_odom_v_gain',
      'rywak.fuse_odom_w_gain',
      'rywak.vel_ema_alpha',
      'rywak.anchor_yaw_to_odom',
      'rywak.anchor_xy_to_odom',
      'rywak.anchor_xy_to_odom_gain',
      'evaluation.points_min_translation',
      'evaluation.points_min_rotation',
      'evaluation.points_min_time_gap_sec',
    ]);
    const SWEEP_FIELD_SPECS = Object.entries(CONFIG_FIELD_METADATA)
      .filter(([key, meta]) => meta.sweep && DATASET_SWEEP_KEYS.has(key))
      .map(([key, meta]) => ({ key, ...meta.sweep }));

    function valueOf(id) { return document.getElementById(id).value.trim(); }
    function selectedExperimentId() { return valueOf('experiment-select'); }
    function selectedExperiment() { return state.experiments.find((exp) => exp.id === selectedExperimentId()); }
    function selectedConfigName() { return valueOf('config-select') || 'experiment_config.yaml'; }
    function selectedSweepSourceExperimentId() { return valueOf('sweep-source-experiment'); }
    function selectedSweepSourceExperiment() {
      return state.experiments.find((exp) => exp.id === selectedSweepSourceExperimentId());
    }
    function selectedSweepResultId() { return valueOf('sweep-result-select'); }
    function selectedSweepResult() {
      return state.sweeps.find((item) => item.id === selectedSweepResultId());
    }
    function query(params) { return new URLSearchParams(params).toString(); }
    function pathKey(path) { return Array.isArray(path) ? path.join('.') : String(path || ''); }
    function configMeta(path) { return CONFIG_FIELD_METADATA[pathKey(path)] || {}; }
    function humanizeToken(token) {
      return String(token || '')
        .split('_')
        .filter(Boolean)
        .map((part) => {
          if (part === 'xy') return 'XY';
          if (part === 'dt') return 'dt';
          if (part === 'sec') return 'sek';
          if (part === 'rms') return 'RMS';
          if (part === 'ema') return 'EMA';
          if (part === 'yaw') return 'yaw';
          if (part.length <= 3) return part.toUpperCase();
          return part.charAt(0).toUpperCase() + part.slice(1);
        })
        .join(' ');
    }
    function describeConfigField(path) {
      const meta = configMeta(path);
      if (meta.label) {
        return meta.label;
      }
      if (!Array.isArray(path) || path.length === 0) {
        return 'Parametr';
      }
      const suffix = path[path.length - 1];
      return CONFIG_SUFFIX_LABELS[suffix] || humanizeToken(suffix);
    }
    function describeConfigGroup(groupName) {
      return CONFIG_GROUP_LABELS[groupName] || humanizeToken(groupName);
    }
    function priorityCategory(path, fallback = 'Parametr') {
      return configMeta(path).category || fallback;
    }
    function numericConfigValue(path, fallback = null) {
      const value = getPathValue(state.currentConfigParsed, path);
      if (typeof value === 'number') {
        return value;
      }
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : fallback;
    }
    function setWorkspace(workspace) {
      state.currentWorkspace = workspace;
      renderWorkspace();
    }
    function renderWorkspace() {
      document.querySelectorAll('[data-workspaces]').forEach((node) => {
        const workspaces = String(node.dataset.workspaces || '')
          .trim()
          .split(' ')
          .filter(Boolean);
        const shouldHide = workspaces.length > 0 && !workspaces.includes(state.currentWorkspace);
        node.hidden = shouldHide;
        node.classList.toggle('workspace-hidden', shouldHide);
      });
      document.querySelectorAll('[data-workspace-tab]').forEach((button) => {
        button.classList.toggle('active', button.dataset.workspaceTab === state.currentWorkspace);
      });
    }
    function hasMetricValue(value) {
      return !(value === null || value === undefined || value === '' || value === 'brak');
    }

    function renderSeriesCheckboxes(containerId, series, defaults) {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      Object.entries(series).forEach(([key, value]) => {
        const label = Array.isArray(value) ? value[2] || key : value.label || key;
        const wrapper = document.createElement('label');
        wrapper.innerHTML = `<input type="checkbox" value="${key}" ${defaults.includes(key) ? 'checked' : ''}> ${label}`;
        container.appendChild(wrapper);
      });
    }

    function checkedValues(containerId) {
      return Array.from(document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`)).map((node) => node.value);
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      return response.json();
    }

    function metricCard(label, value, note = '') {
      return `<div class="metric"><span class="muted">${label}</span><strong>${value ?? 'brak'}</strong>${note ? `<small>${note}</small>` : ''}</div>`;
    }

    function formatMetric(value, digits = 4) {
      if (value === null || value === undefined || value === '') {
        return 'brak';
      }
      if (typeof value === 'number') {
        return value.toFixed(digits);
      }
      return value;
    }

    function buildRmseCard(label, xyValue, thetaValue) {
      const parts = [];
      if (hasMetricValue(xyValue)) {
        parts.push(`XY: ${formatMetric(xyValue)} m`);
      }
      if (hasMetricValue(thetaValue)) {
        parts.push(`θ: ${formatMetric(thetaValue)} rad`);
      }
      return parts.length ? metricCard(label, parts.join(' | ')) : '';
    }

    function buildIouCard(label, value) {
      return hasMetricValue(value) ? metricCard(label, formatMetric(value)) : '';
    }

    async function loadState() {
      const payload = await fetchJson('/api/state');
      state.experiments = payload.experiments || [];
      state.jobs = payload.jobs || [];
      state.function_index = payload.function_index || null;
      state.configs = payload.configs || [];
      state.comparisonCatalog = payload.comparison_catalog || { groups: [] };
      state.sweeps = payload.sweeps || [];

      renderExperimentSelect();
      renderConfigSelect();
      renderQuickLaunchPanel();
      renderComparisonControls();
      renderSweepControls();
      renderSweepResultControls();
      renderExperiment();
      renderJobs();
      renderFunctionIndex();
      renderWorkspace();

      if (!state.currentConfigName) {
        state.currentConfigName = selectedConfigName();
        await loadConfigEditor(false);
      }
    }

    function renderExperimentSelect() {
      const select = document.getElementById('experiment-select');
      const previous = select.value;
      select.innerHTML = '';
      state.experiments.forEach((exp) => {
        const option = document.createElement('option');
        option.value = exp.id;
        option.textContent = exp.id;
        select.appendChild(option);
      });
      if (state.experiments.length > 0) {
        select.value = state.experiments.some((exp) => exp.id === previous) ? previous : state.experiments[0].id;
      }
    }

    function renderConfigSelect() {
      const select = document.getElementById('config-select');
      const previous = state.currentConfigName || select.value || 'experiment_config.yaml';
      select.innerHTML = '';
      state.configs.forEach((cfg) => {
        const option = document.createElement('option');
        option.value = cfg.name;
        option.textContent = cfg.name;
        select.appendChild(option);
      });
      if (state.configs.length > 0) {
        const exists = state.configs.some((cfg) => cfg.name === previous);
        select.value = exists ? previous : state.configs[0].name;
      }
    }

    function renderComparisonControls() {
      const groupSelect = document.getElementById('compare-group');
      const previous = groupSelect.value;
      groupSelect.innerHTML = '';

      (state.comparisonCatalog?.groups || []).forEach((group) => {
        const option = document.createElement('option');
        option.value = group.key;
        option.textContent = group.label;
        groupSelect.appendChild(option);
      });

      if (groupSelect.options.length > 0) {
        const hasPrevious = Array.from(groupSelect.options).some((option) => option.value === previous);
        groupSelect.value = hasPrevious ? previous : groupSelect.options[0].value;
      }

      renderComparisonOptions();
    }

    function renderComparisonOptions() {
      const groupKey = valueOf('compare-group');
      const group = (state.comparisonCatalog?.groups || []).find((item) => item.key === groupKey);
      const paramSelect = document.getElementById('compare-param');
      const metricSelect = document.getElementById('compare-metric');
      const note = document.getElementById('comparison-note');
      const prevParam = paramSelect.value;
      const prevMetric = metricSelect.value;

      paramSelect.innerHTML = '';
      metricSelect.innerHTML = '';

      if (!group) {
        note.textContent = 'Brak dostępnych danych porównawczych.';
        setImageTarget('comparison-plot', '');
        return;
      }

      group.params.forEach((item) => {
        const option = document.createElement('option');
        option.value = item.key;
        option.textContent = `${describeConfigField(item.key.split('.'))} (${item.available_count})`;
        paramSelect.appendChild(option);
      });
      group.metrics.forEach((item) => {
        const option = document.createElement('option');
        option.value = item.key;
        option.textContent = `${item.label} (${item.available_count})`;
        metricSelect.appendChild(option);
      });

      if (paramSelect.options.length > 0) {
        const hasPrevious = Array.from(paramSelect.options).some((option) => option.value === prevParam);
        paramSelect.value = hasPrevious ? prevParam : paramSelect.options[0].value;
      }
      if (metricSelect.options.length > 0) {
        const hasPrevious = Array.from(metricSelect.options).some((option) => option.value === prevMetric);
        metricSelect.value = hasPrevious ? prevMetric : metricSelect.options[0].value;
      }

      refreshComparisonPlot();
    }

    function refreshComparisonPlot() {
      const groupKey = valueOf('compare-group');
      const paramKey = valueOf('compare-param');
      const metricKey = valueOf('compare-metric');
      const group = (state.comparisonCatalog?.groups || []).find((item) => item.key === groupKey);
      const note = document.getElementById('comparison-note');

      if (!group || !paramKey || !metricKey) {
        note.textContent = 'Brak danych do porównania.';
        setImageTarget('comparison-plot', '');
        return;
      }

      const paramInfo = group.params.find((item) => item.key === paramKey);
      const metricInfo = group.metrics.find((item) => item.key === metricKey);
      const paramLabel = paramInfo ? describeConfigField(paramInfo.key.split('.')) : paramKey;
      note.textContent = `Porównanie używa eksperymentów, które mają zapisane: ${paramLabel} oraz ${metricInfo?.label || metricKey}.`;
      setImageTarget('comparison-plot', `/api/plot/comparison?${query({
        group: groupKey,
        param: paramKey,
        metric: metricKey,
        t: Date.now(),
      })}`);
    }

    function renderSweepControls() {
      const groupSelect = document.getElementById('sweep-group');
      const configSelect = document.getElementById('sweep-config');
      const sourceSelect = document.getElementById('sweep-source-experiment');
      const prevGroup = groupSelect.value;
      const prevConfig = configSelect.value;
      const prevSource = sourceSelect.value;

      groupSelect.innerHTML = '';
      sourceSelect.innerHTML = '';

      state.experiments.forEach((exp) => {
        const option = document.createElement('option');
        option.value = exp.id;
        option.textContent = exp.id;
        sourceSelect.appendChild(option);
      });
      if (sourceSelect.options.length > 0) {
        const preferredSource = prevSource || selectedExperimentId();
        const hasPreferred = Array.from(sourceSelect.options).some((option) => option.value === preferredSource);
        sourceSelect.value = hasPreferred ? preferredSource : sourceSelect.options[0].value;
      }

      ['shared', 'robak', 'rywak', 'map_filter'].forEach((groupKey) => {
        const hasOptions = SWEEP_FIELD_SPECS.some((item) => item.group === groupKey);
        if (!hasOptions) {
          return;
        }
        const option = document.createElement('option');
        option.value = groupKey;
        option.textContent = (state.comparisonCatalog?.groups || []).find((item) => item.key === groupKey)?.label
          || (groupKey === 'map_filter' ? 'Filtr mapy' : describeConfigGroup(groupKey));
        groupSelect.appendChild(option);
      });
      if (groupSelect.options.length > 0) {
        const hasPrevious = Array.from(groupSelect.options).some((option) => option.value === prevGroup);
        groupSelect.value = hasPrevious ? prevGroup : groupSelect.options[0].value;
      }

      configSelect.innerHTML = '';
      state.configs.forEach((cfg) => {
        const option = document.createElement('option');
        option.value = cfg.name;
        option.textContent = cfg.name;
        configSelect.appendChild(option);
      });
      if (configSelect.options.length > 0) {
        const preferredConfig = prevConfig || state.currentConfigName || selectedConfigName();
        const hasPreferred = Array.from(configSelect.options).some((option) => option.value === preferredConfig);
        configSelect.value = hasPreferred ? preferredConfig : configSelect.options[0].value;
      }

      renderSweepOptions();
    }

    function renderSweepOptions() {
      const groupKey = valueOf('sweep-group');
      const select = document.getElementById('sweep-param');
      const note = document.getElementById('sweep-note');
      const previous = select.value;
      select.innerHTML = '';

      const options = SWEEP_FIELD_SPECS.filter((item) => item.group === groupKey);
      const sourceExp = selectedSweepSourceExperiment();
      const usesSnapshot = Boolean(sourceExp?.artifacts?.config_snapshot_yaml) && (valueOf('sweep-config') || 'experiment_config.yaml') === 'experiment_config.yaml';
      options.forEach((item) => {
        const option = document.createElement('option');
        option.value = item.key;
        option.textContent = describeConfigField(item.key.split('.'));
        select.appendChild(option);
      });

      if (select.options.length > 0) {
        const hasPrevious = Array.from(select.options).some((option) => option.value === previous);
        select.value = hasPrevious ? previous : select.options[0].value;
        renderSweepDefaults(!hasPrevious);
      } else {
        document.getElementById('sweep-start').value = '';
        document.getElementById('sweep-stop').value = '';
        document.getElementById('sweep-step').value = '';
      }

      if (!options.length) {
        note.textContent = 'Brak parametrów sweepu dla tej grupy.';
      } else if (groupKey === 'shared') {
        note.textContent = 'Sweep utworzy nowe eksperymenty na bazie jednego wybranego datasetu i dla każdej wartości parametru przetrenuje jednocześnie Robaka oraz Rywaka.';
      } else if (usesSnapshot) {
        note.textContent = 'Sweep utworzy nowe eksperymenty na bazie jednego wybranego datasetu. Bazowy config zostanie wzięty ze snapshotu eksperymentu źródłowego.';
      } else if (sourceExp?.artifacts?.config_snapshot_yaml) {
        note.textContent = 'Sweep utworzy nowe eksperymenty na bazie jednego wybranego datasetu. Wybrany config nadpisze snapshot konfiguracji eksperymentu źródłowego.';
      } else {
        note.textContent = 'Sweep utworzy nowe eksperymenty na bazie jednego wybranego datasetu. Wybrany config posłuży jako baza, bo eksperyment źródłowy nie ma snapshotu konfiguracji.';
      }
    }

    function renderSweepDefaults(force = false) {
      const paramKey = valueOf('sweep-param');
      const spec = SWEEP_FIELD_SPECS.find((item) => item.key === paramKey);
      if (!spec) {
        return;
      }
      const startInput = document.getElementById('sweep-start');
      const stopInput = document.getElementById('sweep-stop');
      const stepInput = document.getElementById('sweep-step');

      if (force || !startInput.value) {
        startInput.value = spec.start;
      }
      if (force || !stopInput.value) {
        stopInput.value = spec.stop;
      }
      if (force || !stepInput.value) {
        stepInput.value = spec.step;
      }
    }

    function startSweep() {
      const source_experiment = valueOf('sweep-source-experiment');
      const config = valueOf('sweep-config') || selectedConfigName();
      const param = valueOf('sweep-param');
      const start = valueOf('sweep-start');
      const stop = valueOf('sweep-stop');
      const step = valueOf('sweep-step');
      if (!source_experiment || !config || !param || !start || !stop || !step) {
        alert('Sweep wymaga eksperymentu źródłowego, configu, parametru oraz zakresu od-do z krokiem.');
        return;
      }
      const sourceExp = selectedSweepSourceExperiment();
      const usesSnapshot = Boolean(sourceExp?.artifacts?.config_snapshot_yaml) && config === 'experiment_config.yaml';
      if (state.configDirty && config === selectedConfigName() && !usesSnapshot) {
        alert('Masz niezapisane zmiany w aktualnym configu. Zapisz YAML przed uruchomieniem sweepu.');
        return;
      }
      startJob({
        action: 'sweep_parameter',
        source_experiment,
        config,
        param,
        start,
        stop,
        step,
      });
    }

    function setSweepSummaryLink(id, path, label) {
      const link = document.getElementById(id);
      if (!link) {
        return;
      }
      if (!path) {
        link.style.display = 'none';
        link.removeAttribute('href');
        return;
      }
      link.style.display = '';
      link.href = `/api/artifact?path=${encodeURIComponent(path)}`;
      link.textContent = label;
    }

    function formatInspectionNumber(value, digits = 2) {
      if (value === null || value === undefined || value === '') {
        return 'brak';
      }
      if (typeof value === 'number') {
        if (Number.isInteger(value)) {
          return new Intl.NumberFormat('pl-PL').format(value);
        }
        return value.toFixed(digits);
      }
      return String(value);
    }

    function inspectionLevelLabel(level) {
      if (level === 'good') {
        return 'Dobry';
      }
      if (level === 'medium') {
        return 'Średni';
      }
      if (level === 'bad') {
        return 'Słaby';
      }
      return 'Brak';
    }

    function inspectionBadge(level) {
      if (!level) {
        return '';
      }
      return `<span class="inspection-badge ${level}">${inspectionLevelLabel(level)}</span>`;
    }

    function inspectionStatusClass(level) {
      return level ? ` status-${level}` : '';
    }

    function tierHigherIsBetter(value, goodMin, mediumMin) {
      if (!Number.isFinite(Number(value))) {
        return 'bad';
      }
      const numeric = Number(value);
      if (numeric >= goodMin) {
        return 'good';
      }
      if (numeric >= mediumMin) {
        return 'medium';
      }
      return 'bad';
    }

    function tierLowerIsBetter(value, goodMax, mediumMax) {
      if (!Number.isFinite(Number(value))) {
        return 'bad';
      }
      const numeric = Number(value);
      if (numeric <= goodMax) {
        return 'good';
      }
      if (numeric <= mediumMax) {
        return 'medium';
      }
      return 'bad';
    }

    function inspectionLevelScore(level) {
      if (level === 'good') {
        return 3;
      }
      if (level === 'medium') {
        return 2;
      }
      return 1;
    }

    function scoreToInspectionLevel(score, goodMin = 2.7, mediumMin = 1.7) {
      if (score >= goodMin) {
        return 'good';
      }
      if (score >= mediumMin) {
        return 'medium';
      }
      return 'bad';
    }

    function averageInspectionLevel(levels, goodMin = 2.7, mediumMin = 1.7) {
      const scores = levels.map(inspectionLevelScore);
      const avg = scores.reduce((sum, value) => sum + value, 0) / Math.max(scores.length, 1);
      return scoreToInspectionLevel(avg, goodMin, mediumMin);
    }

    function evaluateInspectionQuality(summary) {
      const scanLevel = averageInspectionLevel([
        tierHigherIsBetter(summary.valid_return_ratio, 0.98, 0.9),
        tierHigherIsBetter(summary.scan_beam_count, 270, 180),
      ], 2.5, 1.7);

      const trajectoryLevel = averageInspectionLevel([
        tierHigherIsBetter(summary.trajectory_length_m, 80, 30),
        tierHigherIsBetter(Math.min(Number(summary.trajectory_x_span_m || 0), Number(summary.trajectory_y_span_m || 0)), 4, 2),
        tierHigherIsBetter(Math.max(Number(summary.trajectory_x_span_m || 0), Number(summary.trajectory_y_span_m || 0)), 8, 4),
      ], 2.5, 1.7);

      const correctionLevel = averageInspectionLevel([
        tierLowerIsBetter(summary.correction_xy_rmse_m, 0.4, 1.5),
        tierLowerIsBetter(summary.correction_theta_rmse_rad, 0.25, 0.8),
      ], 2.5, 1.7);

      const mapLevel = averageInspectionLevel([
        tierHigherIsBetter(summary.sampled_map_point_count, 120000, 40000),
        tierHigherIsBetter(summary.sampled_map_scan_count, 600, 200),
      ], 2.5, 1.7);

      const sampleLevel = tierHigherIsBetter(summary.sample_count, 2000, 500);
      const overallLevel = averageInspectionLevel(
        [scanLevel, trajectoryLevel, correctionLevel, mapLevel, sampleLevel],
        2.7,
        1.8,
      );
      const weakestKey = [
        ['skany', scanLevel],
        ['trajektoria', trajectoryLevel],
        ['korekty', correctionLevel],
        ['mapa', mapLevel],
      ].sort((left, right) => inspectionLevelScore(left[1]) - inspectionLevelScore(right[1]))[0]?.[0] || 'brak';

      return {
        overall: overallLevel,
        scans: scanLevel,
        trajectory: trajectoryLevel,
        corrections: correctionLevel,
        map: mapLevel,
        sample: sampleLevel,
        weakestKey,
      };
    }

    function inspectionKpiCard(label, value, note = '', status = '') {
      return `
        <div class="inspection-kpi${inspectionStatusClass(status)}">
          <div class="inspection-kpi-head">
            <span class="muted">${label}</span>
            ${inspectionBadge(status)}
          </div>
          <strong>${value ?? 'brak'}</strong>
          ${note ? `<small>${note}</small>` : ''}
        </div>
      `;
    }

    function inspectionGroupCard(title, description, cards, status = '') {
      return `
        <section class="inspection-group${inspectionStatusClass(status)}">
          <div class="inspection-group-head">
            <div class="inspection-group-title">
              <h3>${title}</h3>
              ${inspectionBadge(status)}
            </div>
            <p>${description}</p>
          </div>
          <div class="inspection-grid">${cards.join('')}</div>
        </section>
      `;
    }

    function renderDatasetInspection(exp) {
      const note = document.getElementById('dataset-inspection-note');
      const summaryGrid = document.getElementById('dataset-inspection-summary');
      const artifacts = exp?.artifacts || {};
      const summary = exp?.dataset_inspection || null;
      const overviewPath = artifacts.dataset_inspection_overview_png || artifacts.dataset_analysis_png || '';
      const scansPath = artifacts.dataset_inspection_scans_png || '';
      const summaryPath = artifacts.dataset_inspection_summary_json || '';

      setSweepSummaryLink('dataset-inspection-summary-link', summaryPath, 'Otwórz summary.json');
      setImageTarget('dataset-inspection-overview', overviewPath ? `/api/artifact?path=${encodeURIComponent(overviewPath)}` : '');
      setImageTarget('dataset-inspection-scans', scansPath ? `/api/artifact?path=${encodeURIComponent(scansPath)}` : '');

      if (!exp) {
        note.textContent = 'Brak wybranego eksperymentu.';
        summaryGrid.innerHTML = '';
        return;
      }

      if (summary) {
        const quality = evaluateInspectionQuality(summary);
        const heroCards = [
          inspectionKpiCard('Ocena datasetu', inspectionLevelLabel(quality.overall), `Najsłabszy obszar: ${quality.weakestKey}`, quality.overall),
          inspectionKpiCard('Próbki datasetu', formatInspectionNumber(summary.sample_count, 0), `Plik: ${summary.dataset_path?.split('/').pop() || 'dataset.npz'}`, quality.sample),
          inspectionKpiCard('Poprawne pomiary', `${formatInspectionNumber((summary.valid_return_ratio || 0) * 100.0, 1)} %`, `N = ${formatInspectionNumber(summary.valid_return_count || 0, 0)}`, quality.scans),
          inspectionKpiCard('Długość trajektorii', `${formatInspectionNumber(summary.trajectory_length_m)} m`, `Rozpiętość X/Y: ${formatInspectionNumber(summary.trajectory_x_span_m)} m / ${formatInspectionNumber(summary.trajectory_y_span_m)} m`, quality.trajectory),
          inspectionKpiCard('RMSE korekty XY', `${formatInspectionNumber(summary.correction_xy_rmse_m, 4)} m`, `RMSE θ = ${formatInspectionNumber(summary.correction_theta_rmse_rad, 4)} rad`, quality.corrections),
        ];
        const groups = [
          inspectionGroupCard('Jakość skanów', 'Statystyki surowych odczytów LiDAR zapisanych w datasecie.', [
            metricCard('Promienie / skan', formatInspectionNumber(summary.scan_beam_count, 0)),
            metricCard('Średni zasięg', `${formatInspectionNumber(summary.range_mean_m)} m`),
            metricCard('Mediana zasięgu', `${formatInspectionNumber(summary.range_median_m)} m`),
            metricCard('P95 zasięgu', `${formatInspectionNumber(summary.range_p95_m)} m`),
            metricCard('Maksymalny zasięg', `${formatInspectionNumber(summary.range_max_m)} m`),
            metricCard('Seed / meta', formatInspectionNumber(summary.meta?.seed, 0), `N = ${formatInspectionNumber(summary.meta?.n, 0)}, beams = ${formatInspectionNumber(summary.meta?.scan_dim, 0)}`),
          ], quality.scans),
          inspectionGroupCard('Trajektoria', 'Podstawowy opis ruchu robota użyty do budowy datasetu.', [
            metricCard('Długość', `${formatInspectionNumber(summary.trajectory_length_m)} m`),
            metricCard('Rozpiętość X', `${formatInspectionNumber(summary.trajectory_x_span_m)} m`),
            metricCard('Rozpiętość Y', `${formatInspectionNumber(summary.trajectory_y_span_m)} m`),
          ], quality.trajectory),
          inspectionGroupCard('Korekty', 'Jak duże poprawki pozycji i orientacji zapisano względem odometrii.', [
            metricCard('RMSE korekty XY', `${formatInspectionNumber(summary.correction_xy_rmse_m, 4)} m`),
            metricCard('Średnia korekta XY', `${formatInspectionNumber(summary.correction_xy_mean_mm)} mm`),
            metricCard('RMSE korekty θ', `${formatInspectionNumber(summary.correction_theta_rmse_rad, 4)} rad`),
            metricCard('Średnia korekta θ', `${formatInspectionNumber(summary.correction_theta_mean_deg)} deg`),
          ], quality.corrections),
          inspectionGroupCard('Mapa punktowa', 'Ile danych wizualizacyjnych weszło do raportu i z ilu skanów je zebrano.', [
            metricCard('Punkty mapy', formatInspectionNumber(summary.sampled_map_point_count || 0, 0)),
            metricCard('Skanów do wizualizacji', formatInspectionNumber(summary.sampled_map_scan_count || 0, 0)),
            metricCard('Mapa referencyjna', summary.reference_map_yaml?.split('/').pop() || 'brak', summary.dataset_dir?.split('/').pop() || ''),
          ], quality.map),
        ];
        summaryGrid.innerHTML = `
          <div class="inspection-hero">${heroCards.join('')}</div>
          <div class="inspection-groups">${groups.join('')}</div>
        `;
        note.textContent = 'Raport datasetu jest zapisany w folderze eksperymentu i odświeża się po ponownym uruchomieniu inspekcji. Oceny kolorystyczne są heurystyczne i służą do szybkiego porównania jakości datasetów.';
        return;
      }

      summaryGrid.innerHTML = '';
      if (overviewPath || scansPath) {
        note.textContent = 'Znaleziono starsze artefakty inspekcji datasetu. Wygeneruj raport ponownie, aby dostać nowy podgląd i podsumowanie JSON.';
      } else {
        note.textContent = 'Brak raportu datasetu dla wybranego eksperymentu. Kliknij przycisk powyżej, aby go wygenerować.';
      }
    }

    function renderSweepResultControls() {
      const select = document.getElementById('sweep-result-select');
      const previous = select.value;
      select.innerHTML = '';

      state.sweeps.forEach((sweep) => {
        const option = document.createElement('option');
        const sourceText = sweep.source_experiment_id || 'brak źródła';
        const paramPath = String(sweep.param_path || '');
        const paramLabel = paramPath ? describeConfigField(paramPath.split('.')) : 'Parametr sweepa';
        option.value = sweep.id;
        option.textContent = `${sweep.id} · ${sourceText} · ${paramLabel} (${sweep.success_count}/${sweep.total_count})`;
        select.appendChild(option);
      });

      if (select.options.length > 0) {
        const hasPrevious = Array.from(select.options).some((option) => option.value === previous);
        select.value = hasPrevious ? previous : select.options[0].value;
      }

      renderSweepResultOptions();
    }

    function renderSweepResultOptions() {
      const sweep = selectedSweepResult();
      const familySelect = document.getElementById('sweep-result-family');
      const note = document.getElementById('sweep-result-note');
      const previous = familySelect.value;
      familySelect.innerHTML = '';

      if (!sweep) {
        note.textContent = 'Brak zapisanych wyników sweepu w katalogu out/.';
        setSweepSummaryLink('sweep-summary-json-link', '', '');
        setSweepSummaryLink('sweep-summary-csv-link', '', '');
        setImageTarget('sweep-result-plot', '');
        return;
      }

      (sweep.families || []).forEach((family) => {
        const option = document.createElement('option');
        option.value = family.key;
        option.textContent = family.label;
        familySelect.appendChild(option);
      });

      if (familySelect.options.length > 0) {
        const hasPrevious = Array.from(familySelect.options).some((option) => option.value === previous);
        familySelect.value = hasPrevious ? previous : familySelect.options[0].value;
      }

      const paramPath = String(sweep.param_path || '');
      const paramLabel = paramPath ? describeConfigField(paramPath.split('.')) : 'Parametr sweepa';
      const sourceText = sweep.source_experiment_id || 'brak źródła';
      const failurePart = sweep.failed_count ? ` | nieudane: ${sweep.failed_count}` : '';
      const sourcePart = sweep.rows_source === 'recovered_experiments' ? ' | dane odtworzone z exp_sweep_*' : '';
      if (familySelect.options.length > 0) {
        note.textContent = `Źródło datasetu: ${sourceText} | parametr: ${paramLabel} | udane przebiegi: ${sweep.success_count}/${sweep.total_count}${failurePart}${sourcePart}`;
      } else {
        note.textContent = `Źródło datasetu: ${sourceText} | parametr: ${paramLabel} | sweep nie ma jeszcze metryk do narysowania. Udane przebiegi: ${sweep.success_count}/${sweep.total_count}${failurePart}${sourcePart}`;
      }
      setSweepSummaryLink('sweep-summary-json-link', sweep.summary_json_path, 'Otwórz summary.json');
      setSweepSummaryLink('sweep-summary-csv-link', sweep.summary_csv_path, 'Otwórz summary.csv');
      refreshSweepResultPlot();
    }

    function refreshSweepResultPlot() {
      const sweep = selectedSweepResult();
      if (!sweep) {
        setImageTarget('sweep-result-plot', '');
        return;
      }
      const familyKey = valueOf('sweep-result-family');
      if (!familyKey) {
        setImageTarget('sweep-result-plot', `/api/plot/sweep?${query({
          sweep_id: sweep.id,
          family: '',
          t: Date.now(),
        })}`);
        return;
      }
      setImageTarget('sweep-result-plot', `/api/plot/sweep?${query({
        sweep_id: sweep.id,
        family: familyKey,
        t: Date.now(),
      })}`);
    }

    function renderExperiment() {
      const exp = selectedExperiment();
      const meta = document.getElementById('experiment-meta');
      const datasets = document.getElementById('dataset-list');
      const artifacts = document.getElementById('artifact-list');
      const diagnostics = document.getElementById('map-diagnostics');
      const rmseGrid = document.getElementById('rmse-grid');
      const iouGrid = document.getElementById('iou-grid');
      const errorAvailability = document.getElementById('error-availability');

      if (!exp) {
        meta.textContent = 'Brak eksperymentów w katalogu out/.';
        datasets.innerHTML = '';
        artifacts.innerHTML = '';
        diagnostics.textContent = '';
        rmseGrid.innerHTML = '';
        iouGrid.innerHTML = '';
        errorAvailability.textContent = '';
        renderDatasetInspection(null);
        renderQuickLaunchPanel();
        return;
      }

      meta.innerHTML = `
        <div><strong>${exp.id}</strong></div>
        <div>Utworzono: ${exp.created_at || 'brak danych'}</div>
        <div>Próbki datasetu: ${exp.dataset_samples ?? 'brak'}</div>
        <div>Próbki ewaluacji: ${exp.eval_samples ?? 'brak'}</div>
      `;

      const m = exp.metrics || {};
      const rmseCards = [
        buildRmseCard('Baseline', m.rmse_xy_baseline, m.rmse_theta_baseline),
        buildRmseCard('AI', m.rmse_xy_ai, m.rmse_theta_ai),
        buildRmseCard('Robak', m.rmse_xy_robak, m.rmse_theta_robak),
        buildRmseCard('Rywak', m.rmse_xy_rywak, m.rmse_theta_rywak),
        buildRmseCard('ScanMatcher', m.rmse_xy_scanmatch, m.rmse_theta_scanmatch),
        buildRmseCard('Bruteforce', m.rmse_xy_bruteforce, m.rmse_theta_bruteforce),
      ].filter(Boolean);
      rmseGrid.innerHTML = rmseCards.length ? rmseCards.join('') : '<div class="info-box">Brak dostępnych metryk RMSE dla tego eksperymentu.</div>';

      const iouCards = [
        buildIouCard('Baseline', m.iou_map_baseline),
        buildIouCard('AI', m.iou_map_ai),
        buildIouCard('Robak', m.iou_map_robak),
        buildIouCard('Rywak', m.iou_map_rywak),
      ].filter(Boolean);
      iouGrid.innerHTML = iouCards.length ? iouCards.join('') : '<div class="info-box">Brak dostępnych metryk IoU dla tego eksperymentu.</div>';

      datasets.innerHTML = '';
      (exp.datasets || []).forEach((dataset) => {
        const node = document.createElement('div');
        node.className = 'metric';
        node.innerHTML = `<span class="muted">${dataset.kind}</span><strong>${dataset.name}</strong><small>${dataset.size_mb} MB</small>`;
        datasets.appendChild(node);
      });

      artifacts.innerHTML = '';
      Object.entries(exp.artifacts || {}).forEach(([key, path]) => {
        const a = document.createElement('a');
        a.target = '_blank';
        a.href = `/api/artifact?path=${encodeURIComponent(path)}`;
        a.textContent = `${key}: ${path.split('/').pop()}`;
        artifacts.appendChild(a);
      });

      const pointDiag = exp.diagnostics?.point_map_filter || {};
      diagnostics.innerHTML = Object.entries(pointDiag).map(([name, stat]) =>
        `<div><strong>${name}</strong>: stamped_scans=${stat.stamped_scans}, skipped_scans=${stat.skipped_scans}, stamped_points=${stat.stamped_points}</div>`
      ).join('') || 'Brak diagnostyki map punktowych.';
      renderQuickLaunchPanel();
      renderDatasetInspection(exp);

      if (exp.error_series_mode === 'stored') {
        errorAvailability.className = 'flash ok';
        errorAvailability.textContent = 'Niestandardowy wykres błędu działa na zapisanych surowych seriach błędu.';
      } else if (exp.error_series_mode === 'reconstructed') {
        errorAvailability.className = 'flash';
        errorAvailability.textContent = 'Niestandardowy wykres błędu jest odtwarzany z trajektorii, bo eksperyment nie miał jeszcze zapisanych surowych błędów.';
      } else if (exp.error_series_mode === 'missing') {
        errorAvailability.className = 'flash error';
        errorAvailability.textContent = 'Brak trajectory_data.npz. Dla pełnego wykresu niestandardowego trzeba uruchomić nową ewaluację.';
      } else {
        errorAvailability.className = 'flash error';
        errorAvailability.textContent = 'Nie udało się odczytać danych do niestandardowego wykresu błędu.';
      }

      refreshPlots();
    }

    function renderJobs() {
      const list = document.getElementById('job-list');
      list.innerHTML = '';
      state.jobs.forEach((job) => {
        const button = document.createElement('button');
        button.className = 'secondary';
        button.style.textAlign = 'left';
        button.innerHTML = `
          <div><strong>${job.label}</strong></div>
          <div class="status ${job.status}"><span class="dot"></span>${job.status} (${job.return_code ?? '...'})</div>
        `;
        button.onclick = () => selectJob(job.id);
        list.appendChild(button);
      });
      if (!state.selectedJobId && state.jobs.length > 0) {
        state.selectedJobId = state.jobs[0].id;
      }
      refreshJobLog();
    }

    async function selectJob(jobId) {
      state.selectedJobId = jobId;
      await refreshJobLog();
    }

    async function refreshJobLog() {
      const target = document.getElementById('job-log');
      if (!state.selectedJobId) {
        target.value = '';
        return;
      }
      try {
        const payload = await fetchJson(`/api/job-log?id=${encodeURIComponent(state.selectedJobId)}`);
        target.value = payload.log || '';
      } catch (error) {
        target.value = String(error);
      }
    }

    async function startJob(payload) {
      try {
        await fetchJson('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        await loadState();
      } catch (error) {
        alert(error);
      }
    }

    function trainSelected() {
      const experiment_id = selectedExperimentId();
      if (!experiment_id) {
        alert('Najpierw wybierz eksperyment.');
        return;
      }
      startJob({
        action: 'train_selected',
        experiment_id,
        model_type: valueOf('train-model-type'),
      });
    }

    function inspectSelected() {
      const experiment_id = selectedExperimentId();
      if (!experiment_id) {
        alert('Najpierw wybierz eksperyment.');
        return;
      }
      startJob({ action: 'inspect_dataset', experiment_id });
    }

    function renderQuickLaunchPanel(forceDefaults = false) {
      const note = document.getElementById('quick-launch-config-note');
      const datasetInput = document.getElementById('quick-dataset-duration');
      const evalInput = document.getElementById('quick-eval-duration');
      const configName = selectedConfigName() || state.currentConfigName || 'experiment_config.yaml';
      const experimentId = selectedExperimentId();

      note.textContent = `Bazowy config: ${configName}. Wybrany eksperyment do treningu/testu: ${experimentId || 'brak'}. Wspólny czas datasetu nadpisuje timing.dataset_duration, robak.dataset_duration i rywak.dataset_duration tylko dla nowych uruchomień.`;

      const datasetDuration = numericConfigValue(['timing', 'dataset_duration'], 30.0);
      const evalDuration = numericConfigValue(['timing', 'eval_duration'], 100.0);

      if (forceDefaults || !datasetInput.value) {
        datasetInput.value = datasetDuration === null ? '' : String(datasetDuration);
      }
      if (forceDefaults || !evalInput.value) {
        evalInput.value = evalDuration === null ? '' : String(evalDuration);
      }
    }

    function startQuickPipeline(mode) {
      const config = selectedConfigName();
      const runName = valueOf('quick-experiment-name');
      const datasetDuration = valueOf('quick-dataset-duration');
      const evalDuration = valueOf('quick-eval-duration');
      const experimentId = selectedExperimentId();

      if (!config) {
        alert('Najpierw wczytaj plik konfiguracyjny.');
        return;
      }
      if (mode === 'dataset' || mode === 'full_cycle') {
        if (!datasetDuration) {
          alert('Podaj wspólny czas zbierania datasetu.');
          return;
        }
      }
      if (mode === 'full_cycle' && !evalDuration) {
        alert('Podaj czas testu i ewaluacji dla pełnego przebiegu.');
        return;
      }
      if (mode === 'test_existing' && !evalDuration) {
        alert('Podaj czas testu i ewaluacji dla testu wybranego eksperymentu.');
        return;
      }
      if ((mode === 'train_existing' || mode === 'test_existing') && !experimentId) {
        alert('Najpierw wybierz eksperyment z listy.');
        return;
      }
      if (mode === 'dataset' || mode === 'full_cycle') {
        if (!runName) {
          alert('Podaj nazwę nowego uruchomienia, żeby łatwo odróżnić eksperymenty.');
          return;
        }
      }
      if (state.configDirty) {
        alert('Masz niezapisane zmiany w configu. Zapisz YAML przed szybkim uruchomieniem.');
        return;
      }

      startJob({
        action: 'quick_pipeline',
        mode,
        config,
        name: runName,
        experiment_id: experimentId,
        dataset_duration: datasetDuration,
        eval_duration: evalDuration,
      });
    }

    function cloneValue(value) {
      return value === undefined ? null : JSON.parse(JSON.stringify(value));
    }

    function flattenConfigEntries(value, path = [], entries = []) {
      if (Array.isArray(value) || value === null || typeof value !== 'object') {
        entries.push({ path, value });
        return entries;
      }
      Object.entries(value).forEach(([key, child]) => flattenConfigEntries(child, path.concat(key), entries));
      return entries;
    }

    function getPathValue(root, path) {
      return path.reduce((acc, key) => (acc == null ? undefined : acc[key]), root);
    }

    function setPathValue(root, path, value) {
      if (!path.length) {
        return;
      }
      let cursor = root;
      for (let i = 0; i < path.length - 1; i += 1) {
        if (cursor[path[i]] === undefined || cursor[path[i]] === null || typeof cursor[path[i]] !== 'object' || Array.isArray(cursor[path[i]])) {
          cursor[path[i]] = {};
        }
        cursor = cursor[path[i]];
      }
      cursor[path[path.length - 1]] = value;
    }

    function parseLooseScalar(text) {
      const trimmed = text.trim();
      if (trimmed === '') {
        return '';
      }
      if (trimmed === 'true') {
        return true;
      }
      if (trimmed === 'false') {
        return false;
      }
      if (trimmed === 'null' || trimmed === '~') {
        return null;
      }
      const numeric = Number(trimmed);
      if (!Number.isNaN(numeric)) {
        return numeric;
      }
      return text;
    }

    function parseTypedValue(rawValue, originalValue) {
      if (Array.isArray(originalValue)) {
        try {
          const parsed = JSON.parse(rawValue);
          if (!Array.isArray(parsed)) {
            throw new Error('To nie jest lista.');
          }
          return parsed;
        } catch (error) {
          throw new Error('Lista musi być podana jako JSON, np. [1, 2, 3].');
        }
      }
      if (originalValue === null) {
        return parseLooseScalar(rawValue);
      }
      if (typeof originalValue === 'number') {
        const numeric = Number(rawValue);
        if (Number.isNaN(numeric)) {
          throw new Error('Wartość musi być liczbą.');
        }
        return Number.isInteger(originalValue) ? Math.trunc(numeric) : numeric;
      }
      if (typeof originalValue === 'boolean') {
        return Boolean(rawValue);
      }
      return rawValue;
    }

    function describeConfigValue(value) {
      if (Array.isArray(value)) {
        return 'lista JSON';
      }
      if (value === null) {
        return 'null / tekst / liczba';
      }
      if (typeof value === 'boolean') {
        return 'przełącznik';
      }
      if (typeof value === 'number') {
        return 'liczba';
      }
      if (typeof value === 'string') {
        return 'tekst';
      }
      return typeof value;
    }

    function markConfigDirty(message = null) {
      state.configDirty = true;
      const status = document.getElementById('config-status');
      status.className = 'flash';
      status.textContent = message || `Edytujesz ${selectedConfigName()} — są niezapisane zmiany.`;
    }

    function scheduleConfigPreviewRender(note = 'Zmodyfikowano parametry w formularzu.') {
      if (state.configRenderTimer) {
        clearTimeout(state.configRenderTimer);
      }
      markConfigDirty(note);
      state.configRenderTimer = setTimeout(renderParsedConfigPreview, 220);
    }

    async function renderParsedConfigPreview() {
      if (state.currentConfigParsed === null) {
        return;
      }
      try {
        const payload = await fetchJson('/api/config/render', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ parsed: state.currentConfigParsed }),
        });
        document.getElementById('config-editor').value = payload.content || '';
      } catch (error) {
        const status = document.getElementById('config-status');
        status.className = 'flash error';
        status.textContent = String(error);
      }
    }

    function updateParsedConfigValue(path, rawValue, originalValue, options = {}) {
      if (state.currentConfigParsed === null) {
        return;
      }
      try {
        const nextValue = options.direct ? rawValue : parseTypedValue(rawValue, originalValue);
        setPathValue(state.currentConfigParsed, path, nextValue);
        scheduleConfigPreviewRender(`Zmodyfikowano ${path.join('.')}. Są niezapisane zmiany.`);
      } catch (error) {
        const status = document.getElementById('config-status');
        status.className = 'flash error';
        status.textContent = `${path.join('.')} — ${error.message || error}`;
      }
    }

    function appendConfigValueEditor(container, path, currentValue) {
      let input;
      if (typeof currentValue === 'boolean') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = currentValue;
        input.onchange = () => updateParsedConfigValue(path, input.checked, currentValue, { direct: true });
      } else if (Array.isArray(currentValue)) {
        input = document.createElement('input');
        input.value = JSON.stringify(currentValue);
        input.spellcheck = false;
        input.onchange = () => updateParsedConfigValue(path, input.value, currentValue);
      } else {
        input = document.createElement('input');
        input.value = currentValue === null ? '' : String(currentValue);
        input.oninput = () => updateParsedConfigValue(path, input.value, currentValue);
      }
      container.appendChild(input);

      if (Array.isArray(currentValue)) {
        const help = document.createElement('span');
        help.className = 'config-help';
        help.textContent = 'Listy edytuj jako JSON, np. [256, 128, 64].';
        container.appendChild(help);
      }
    }

    function renderConfigPriorityFields() {
      const container = document.getElementById('config-priority');
      container.innerHTML = '';
      const data = state.currentConfigParsed;
      if (data === null || typeof data !== 'object' || Array.isArray(data)) {
        return;
      }

      const section = document.createElement('section');
      section.className = 'config-group';

      const title = document.createElement('h3');
      title.textContent = 'Najważniejsze parametry do trenowania i porównywania modeli';
      section.appendChild(title);

      const grid = document.createElement('div');
      grid.className = 'config-priority-grid';

      CONFIG_PRIORITY_FIELDS.forEach((spec) => {
        const currentValue = getPathValue(data, spec.path);
        if (currentValue === undefined) {
          return;
        }

        const card = document.createElement('div');
        card.className = 'config-priority-card';

        const badge = document.createElement('span');
        badge.className = 'config-priority-badge';
        badge.textContent = priorityCategory(spec.path, spec.category);
        card.appendChild(badge);

        const label = document.createElement('span');
        label.className = 'config-title';
        label.textContent = describeConfigField(spec.path);
        card.appendChild(label);

        const pathEl = document.createElement('span');
        pathEl.className = 'config-path';
        pathEl.textContent = spec.path.join('.');
        card.appendChild(pathEl);

        appendConfigValueEditor(card, spec.path, currentValue);
        grid.appendChild(card);
      });

      if (!grid.childNodes.length) {
        return;
      }

      section.appendChild(grid);
      container.appendChild(section);
    }

    function renderConfigFields() {
      const container = document.getElementById('config-fields');
      container.innerHTML = '';
      const data = state.currentConfigParsed;
      if (data === null || typeof data !== 'object' || Array.isArray(data)) {
        container.innerHTML = '<div class="info-box">Brak struktury YAML do wyświetlenia w formularzu.</div>';
        return;
      }

      const filter = valueOf('config-filter').toLowerCase();
      const groups = new Map();
      flattenConfigEntries(data).forEach((entry) => {
        const fullPath = entry.path.join('.');
        if (filter && !fullPath.toLowerCase().includes(filter)) {
          return;
        }
        const groupName = entry.path[0] || 'root';
        if (!groups.has(groupName)) {
          groups.set(groupName, []);
        }
        groups.get(groupName).push(entry);
      });

      if (groups.size === 0) {
        container.innerHTML = '<div class="info-box">Filtr nie pasuje do żadnego parametru.</div>';
        return;
      }

      groups.forEach((entries, groupName) => {
        const group = document.createElement('section');
        group.className = 'config-group';

        const title = document.createElement('h3');
        title.textContent = describeConfigGroup(groupName);
        group.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'config-fields-grid';

        entries.forEach((entry) => {
          const currentValue = getPathValue(state.currentConfigParsed, entry.path);
          const field = document.createElement('label');
          field.className = 'config-field';

          const titleEl = document.createElement('span');
          titleEl.className = 'config-title';
          titleEl.textContent = describeConfigField(entry.path);
          field.appendChild(titleEl);

          const pathEl = document.createElement('span');
          pathEl.className = 'config-path';
          pathEl.textContent = entry.path.join('.');
          field.appendChild(pathEl);

          const typeEl = document.createElement('span');
          typeEl.className = 'config-type';
          typeEl.textContent = describeConfigValue(currentValue);
          field.appendChild(typeEl);

          appendConfigValueEditor(field, entry.path, currentValue);

          grid.appendChild(field);
        });

        group.appendChild(grid);
        container.appendChild(group);
      });
    }

    async function loadConfigEditor(force = false) {
      if (state.configDirty && !force) {
        return;
      }
      const configName = selectedConfigName();
      try {
        const payload = await fetchJson(`/api/config?name=${encodeURIComponent(configName)}`);
        document.getElementById('config-editor').value = payload.content || '';
        state.currentConfigParsed = cloneValue(payload.parsed || {});
        state.currentConfigName = payload.name;
        state.configDirty = false;
        document.getElementById('config-filter').value = '';
        renderConfigPriorityFields();
        renderConfigFields();
        renderQuickLaunchPanel(true);
        const sweepConfig = document.getElementById('sweep-config');
        if (sweepConfig) {
          sweepConfig.value = payload.name;
        }
        const status = document.getElementById('config-status');
        status.className = 'flash ok';
        status.textContent = `Wczytano ${payload.name}.`;
      } catch (error) {
        const status = document.getElementById('config-status');
        status.className = 'flash error';
        status.textContent = String(error);
      }
    }

    async function saveConfigEditor() {
      try {
        const payload = await fetchJson('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: selectedConfigName(),
            content: document.getElementById('config-editor').value,
          }),
        });
        document.getElementById('config-editor').value = payload.content || '';
        state.currentConfigParsed = cloneValue(payload.parsed || {});
        state.currentConfigName = payload.name;
        state.configDirty = false;
        renderConfigPriorityFields();
        renderConfigFields();
        renderQuickLaunchPanel(true);
        const sweepConfig = document.getElementById('sweep-config');
        if (sweepConfig) {
          sweepConfig.value = payload.name;
        }
        const status = document.getElementById('config-status');
        status.className = 'flash ok';
        status.textContent = `Zapisano ${payload.name}.`;
      } catch (error) {
        const status = document.getElementById('config-status');
        status.className = 'flash error';
        status.textContent = String(error);
      }
    }

    function setImageTarget(id, src) {
      const img = document.getElementById(id);
      img.src = src || '';
    }

    function refreshPlots() {
      const exp = selectedExperiment();
      if (!exp) {
        return;
      }

      const trajParams = {
        experiment_id: exp.id,
        series: checkedValues('trajectory-series').join(','),
        x_min: valueOf('traj-x-min'),
        x_max: valueOf('traj-x-max'),
        y_min: valueOf('traj-y-min'),
        y_max: valueOf('traj-y-max'),
        t: Date.now(),
      };
      setImageTarget('trajectory-custom', `/api/plot/trajectory?${query(trajParams)}`);

      const errParams = {
        experiment_id: exp.id,
        series: checkedValues('error-series').join(','),
        metric: valueOf('error-metric'),
        time_min: valueOf('err-time-min'),
        time_max: valueOf('err-time-max'),
        y_min: valueOf('err-y-min'),
        y_max: valueOf('err-y-max'),
        t: Date.now(),
      };
      setImageTarget('error-custom', `/api/plot/error?${query(errParams)}`);

      const artifacts = exp.artifacts || {};
      setImageTarget('trajectory-static', artifacts.trajectory_png ? `/api/artifact?path=${encodeURIComponent(artifacts.trajectory_png)}` : '');
      setImageTarget('error-static', artifacts.errors_png ? `/api/artifact?path=${encodeURIComponent(artifacts.errors_png)}` : '');
      setImageTarget('maps-static', artifacts.maps_png ? `/api/artifact?path=${encodeURIComponent(artifacts.maps_png)}` : '');
    }

    function renderFunctionIndex() {
      const summary = document.getElementById('function-summary');
      const mdLink = document.getElementById('function-md-link');
      const jsonLink = document.getElementById('function-json-link');
      if (!state.function_index) {
        summary.textContent = 'Brak indeksu funkcji.';
        return;
      }
      summary.textContent = `Wygenerowano ${state.function_index.count} pozycji.`;
      mdLink.href = `/api/artifact?path=${encodeURIComponent(state.function_index.markdown_path)}`;
      mdLink.textContent = 'Otwórz function_index.md';
      jsonLink.href = `/api/artifact?path=${encodeURIComponent(state.function_index.json_path)}`;
      jsonLink.textContent = 'Otwórz function_index.json';
    }

    function openImageModalById(id, title) {
      const img = document.getElementById(id);
      if (!img || !img.src) {
        return;
      }
      const modal = document.getElementById('image-modal');
      document.getElementById('image-modal-title').textContent = title;
      document.getElementById('image-modal-img').src = img.src;
      modal.classList.add('open');
    }

    function closeImageModal(event) {
      if (event && event.target && event.target.id && event.target.id !== 'image-modal') {
        return;
      }
      document.getElementById('image-modal').classList.remove('open');
    }

    async function bootstrap() {
      renderSeriesCheckboxes('trajectory-series', {
        gt: ['time_s', 'gt_xytheta', 'GT'],
        baseline: ['time_s', 'baseline_xytheta', 'Baseline'],
        ai: ['ai_time_s', 'ai_xytheta', 'AI'],
        robak: ['robak_time_s', 'robak_xytheta', 'Robak'],
        rywak: ['rywak_time_s', 'rywak_xytheta', 'Rywak'],
        scanmatch: ['scanmatch_time_s', 'scanmatch_xytheta', 'ScanMatcher'],
        bruteforce: ['bruteforce_time_s', 'bruteforce_xytheta', 'Bruteforce'],
      }, TRAJ_DEFAULT);
      renderSeriesCheckboxes('error-series', {
        baseline: ['', '', 'Baseline'],
        ai: ['', '', 'AI'],
        robak: ['', '', 'Robak'],
        rywak: ['', '', 'Rywak'],
        scanmatch: ['', '', 'ScanMatcher'],
        bruteforce: ['', '', 'Bruteforce'],
      }, ERR_DEFAULT);
      await loadState();
      setInterval(loadState, 3000);
    }

    bootstrap().catch((error) => alert(error));
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SLAMAIDashboard/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTML_PAGE, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            ensure_function_index()
            experiments = discover_experiments()
            payload = {
                "experiments": experiments,
                "jobs": JOB_MANAGER.list_jobs(),
                "configs": list_config_files(),
                "comparison_catalog": build_comparison_catalog(experiments),
                "sweeps": discover_sweeps(),
                "function_index": {
                    "count": read_json(FUNCTION_INDEX_JSON).get("count", 0) if FUNCTION_INDEX_JSON.exists() else 0,
                    "markdown_path": str(FUNCTION_INDEX_MD.resolve()),
                    "json_path": str(FUNCTION_INDEX_JSON.resolve()),
                },
            }
            self._send_json(payload)
            return
        if parsed.path == "/api/config":
            config_name = parse_qs(parsed.query).get("name", ["experiment_config.yaml"])[0]
            try:
                payload = load_config_payload(config_name)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/job-log":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            if not job_id:
                self._send_json({"error": "Brak id zadania."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                log_text = JOB_MANAGER.read_log(job_id)
            except KeyError:
                self._send_json({"error": f"Nie znaleziono zadania {job_id}."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"id": job_id, "log": log_text})
            return
        if parsed.path == "/api/plot/trajectory":
            query = parse_qs(parsed.query)
            experiment_id = query.get("experiment_id", [""])[0]
            series_names = [item for item in query.get("series", [""])[0].split(",") if item]
            png = plot_trajectory_image(
                experiment_id=experiment_id,
                series_names=series_names,
                x_min=safe_float(query.get("x_min", [""])[0]),
                x_max=safe_float(query.get("x_max", [""])[0]),
                y_min=safe_float(query.get("y_min", [""])[0]),
                y_max=safe_float(query.get("y_max", [""])[0]),
            )
            self._send_bytes(png, content_type="image/png")
            return
        if parsed.path == "/api/plot/error":
            query = parse_qs(parsed.query)
            experiment_id = query.get("experiment_id", [""])[0]
            metric = query.get("metric", ["position_m"])[0]
            series_names = [item for item in query.get("series", [""])[0].split(",") if item]
            png = plot_error_image(
                experiment_id=experiment_id,
                series_names=series_names,
                metric=metric,
                time_min=safe_float(query.get("time_min", [""])[0]),
                time_max=safe_float(query.get("time_max", [""])[0]),
                y_min=safe_float(query.get("y_min", [""])[0]),
                y_max=safe_float(query.get("y_max", [""])[0]),
            )
            self._send_bytes(png, content_type="image/png")
            return
        if parsed.path == "/api/plot/comparison":
            query = parse_qs(parsed.query)
            group = query.get("group", ["robak"])[0]
            param_key = query.get("param", [""])[0]
            metric_key = query.get("metric", [""])[0]
            png = plot_comparison_image(group=group, param_key=param_key, metric_key=metric_key)
            self._send_bytes(png, content_type="image/png")
            return
        if parsed.path == "/api/plot/sweep":
            query = parse_qs(parsed.query)
            sweep_id = query.get("sweep_id", [""])[0]
            family_key = query.get("family", [""])[0]
            png = plot_sweep_image(sweep_id=sweep_id, family_key=family_key)
            self._send_bytes(png, content_type="image/png")
            return
        if parsed.path == "/api/artifact":
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            if not raw_path:
                self._send_text("Brak sciezki.", status=HTTPStatus.BAD_REQUEST)
                return
            try:
                path = safe_resolve_local_path(raw_path)
            except Exception as exc:
                self._send_text(str(exc), status=HTTPStatus.BAD_REQUEST)
                return
            if not path.exists() or not path.is_file():
                self._send_text(f"Nie znaleziono pliku: {path}", status=HTTPStatus.NOT_FOUND)
                return
            content_type, _ = mimetypes.guess_type(str(path))
            if path.suffix.lower() in {".md", ".txt", ".yaml", ".yml", ".json", ".csv"}:
                content_type = "text/plain; charset=utf-8"
            elif content_type and content_type.startswith("text/") and "charset=" not in content_type:
                content_type = f"{content_type}; charset=utf-8"
            self._send_bytes(path.read_bytes(), content_type=content_type or "application/octet-stream")
            return

        self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config/render":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                content = render_yaml_content(payload.get("parsed", {}))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"content": content}, status=HTTPStatus.OK)
            return
        if parsed.path == "/api/config":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                name = str(payload.get("name", "experiment_config.yaml")).strip()
                result = save_config_payload(
                    name,
                    content=None if "parsed" in payload else str(payload.get("content", "")),
                    parsed=payload.get("parsed") if "parsed" in payload else None,
                )
            except yaml.YAMLError as exc:
                self._send_json({"error": f"Błąd YAML: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.OK)
            return
        if parsed.path != "/api/jobs":
            self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            label, command = build_job_command(payload)
            job = JOB_MANAGER.start(label=label, command=command)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(job, status=HTTPStatus.CREATED)

    def log_message(self, format: str, *args):
        return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, status=status, content_type="application/json; charset=utf-8")

    def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain; charset=utf-8"):
        self._send_bytes(text.encode("utf-8"), status=status, content_type=content_type)

    def _send_bytes(self, data: bytes, status: HTTPStatus = HTTPStatus.OK, content_type: str = "application/octet-stream"):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError as exc:
            if exc.errno in {errno.EPIPE, errno.ECONNRESET}:
                return
            raise


def main():
    parser = argparse.ArgumentParser(description="Uruchamia dashboard HTTP dla projektu SLAM_AI.")
    parser.add_argument("--host", default="127.0.0.1", help="Adres nasluchu.")
    parser.add_argument("--port", type=int, default=8765, help="Port HTTP.")
    args = parser.parse_args()

    ensure_function_index()
    ensure_grouped_out_layout()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Repo: {REPO_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
