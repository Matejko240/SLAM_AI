#!/usr/bin/env python3
"""Pomocnicza, zgodna wstecznie organizacja katalogu out/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out"
EXPERIMENTS_DIR = OUT_DIR / "experiments"
SWEEPS_DIR = OUT_DIR / "sweeps"
DATASETS_DIR = OUT_DIR / "datasets"
JOBS_DIR = OUT_DIR / "jobs"
QUICK_CONFIGS_DIR = OUT_DIR / "quick_configs"
DASHBOARD_JOBS_DIR = JOBS_DIR / "dashboard_jobs"
DASHBOARD_QUICK_CONFIG_DIR = QUICK_CONFIGS_DIR / "dashboard_quick_configs"

_GROUP_DIRS = {
    EXPERIMENTS_DIR.name,
    SWEEPS_DIR.name,
    DATASETS_DIR.name,
    JOBS_DIR.name,
    QUICK_CONFIGS_DIR.name,
}


def _cleanup_stale_symlink(path: Path) -> None:
    if path.is_symlink() and not path.exists():
        path.unlink()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_symlink(link_path: Path, target_path: Path) -> bool:
    _cleanup_stale_symlink(link_path)
    target_path = target_path.resolve()
    if link_path.exists():
        if link_path.is_symlink():
            try:
                if link_path.resolve() == target_path:
                    return True
            except Exception:
                pass
            link_path.unlink()
        else:
            return False

    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target_path, link_path.parent)
    try:
        os.symlink(relative_target, link_path, target_is_directory=True)
    except TypeError:
        os.symlink(relative_target, link_path)
    except OSError:
        return False
    return True


def _path_is_live_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


def _iter_named_dirs(root: Path, prefix: str) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.iterdir(), key=lambda item: item.name, reverse=True):
        _cleanup_stale_symlink(path)
        if path.name in _GROUP_DIRS:
            continue
        if not path.name.startswith(prefix):
            continue
        if path.is_dir():
            yield path


def _sync_named_group(group_dir: Path, prefix: str) -> None:
    _ensure_dir(group_dir)

    for legacy_dir in _iter_named_dirs(OUT_DIR, prefix):
        _ensure_symlink(group_dir / legacy_dir.name, legacy_dir)

    for grouped_dir in _iter_named_dirs(group_dir, prefix):
        _ensure_symlink(OUT_DIR / grouped_dir.name, grouped_dir)


def _sync_special_dir(actual_dir: Path, legacy_name: str) -> None:
    _ensure_dir(actual_dir)
    _ensure_symlink(OUT_DIR / legacy_name, actual_dir)


def _looks_like_experiment_storage(path: Path) -> bool:
    if _path_is_live_file(path / "experiment_metadata.json"):
        return True
    if _path_is_live_file(path / "results.json"):
        return True
    return any(_path_is_live_file(candidate) for candidate in path.glob("dataset*.npz"))


def _adopt_legacy_dataset_experiments() -> None:
    _ensure_dir(EXPERIMENTS_DIR)
    _ensure_dir(DATASETS_DIR)

    for dataset_dir in _iter_named_dirs(DATASETS_DIR, "exp_"):
        experiment_dir = EXPERIMENTS_DIR / dataset_dir.name
        legacy_dir = OUT_DIR / dataset_dir.name
        _cleanup_stale_symlink(experiment_dir)
        _cleanup_stale_symlink(legacy_dir)

        if experiment_dir.exists() or legacy_dir.exists():
            continue
        if not _looks_like_experiment_storage(dataset_dir):
            continue

        dataset_dir.rename(experiment_dir)
        _ensure_symlink(OUT_DIR / experiment_dir.name, experiment_dir)


def _sync_dataset_views() -> None:
    _ensure_dir(DATASETS_DIR)
    patterns = (
        "dataset*.npz",
        "trajectory_data.npz",
        "dataset_inspection_*.png",
        "dataset_inspection_*.json",
    )
    for exp_dir in _iter_named_dirs(EXPERIMENTS_DIR, "exp_"):
        dataset_view_dir = DATASETS_DIR / exp_dir.name
        _ensure_dir(dataset_view_dir)
        for pattern in patterns:
            for artifact in sorted(exp_dir.glob(pattern)):
                if artifact.is_file():
                    _ensure_symlink(dataset_view_dir / artifact.name, artifact)


def ensure_grouped_out_layout() -> None:
    _ensure_dir(OUT_DIR)
    _ensure_dir(EXPERIMENTS_DIR)
    _ensure_dir(SWEEPS_DIR)
    _ensure_dir(DATASETS_DIR)
    _ensure_dir(JOBS_DIR)
    _ensure_dir(QUICK_CONFIGS_DIR)

    _adopt_legacy_dataset_experiments()
    _sync_named_group(EXPERIMENTS_DIR, "exp_")
    _sync_named_group(SWEEPS_DIR, "sweep")
    _sync_special_dir(DASHBOARD_JOBS_DIR, "dashboard_jobs")
    _sync_special_dir(DASHBOARD_QUICK_CONFIG_DIR, "dashboard_quick_configs")
    _sync_dataset_views()


def iter_experiment_dirs() -> list[Path]:
    ensure_grouped_out_layout()
    return list(_iter_named_dirs(EXPERIMENTS_DIR, "exp_"))


def iter_sweep_dirs() -> list[Path]:
    ensure_grouped_out_layout()
    return list(_iter_named_dirs(SWEEPS_DIR, "sweep"))


def resolve_experiment_dir(experiment_id: str) -> Path:
    ensure_grouped_out_layout()
    for candidate in (EXPERIMENTS_DIR / experiment_id, OUT_DIR / experiment_id):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Nie znaleziono eksperymentu: {experiment_id}")


def resolve_sweep_dir(sweep_id: str) -> Path:
    ensure_grouped_out_layout()
    for candidate in (SWEEPS_DIR / sweep_id, OUT_DIR / sweep_id):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Nie znaleziono sweepa: {sweep_id}")


def ensure_experiment_storage(experiment_id: str) -> Path:
    ensure_grouped_out_layout()
    legacy_dir = OUT_DIR / experiment_id
    if legacy_dir.exists() and legacy_dir.is_dir() and not legacy_dir.is_symlink():
        _ensure_symlink(EXPERIMENTS_DIR / experiment_id, legacy_dir)
        _sync_dataset_views()
        return legacy_dir.resolve()

    grouped_dir = EXPERIMENTS_DIR / experiment_id
    grouped_dir.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(OUT_DIR / experiment_id, grouped_dir)
    _sync_dataset_views()
    return grouped_dir.resolve()


def ensure_sweep_storage(sweep_id: str) -> Path:
    ensure_grouped_out_layout()
    legacy_dir = OUT_DIR / sweep_id
    if legacy_dir.exists() and legacy_dir.is_dir() and not legacy_dir.is_symlink():
        _ensure_symlink(SWEEPS_DIR / sweep_id, legacy_dir)
        return legacy_dir.resolve()

    grouped_dir = SWEEPS_DIR / sweep_id
    grouped_dir.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(OUT_DIR / sweep_id, grouped_dir)
    return grouped_dir.resolve()
