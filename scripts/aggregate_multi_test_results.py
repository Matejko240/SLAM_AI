#!/usr/bin/env python3
"""Scala wyniki wielu scenariuszy testowych do jednego results.json eksperymentu."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def aggregate_metrics(runs: list[dict[str, Any]]) -> tuple[dict[str, float | None], dict[str, dict[str, float | int | None]]]:
    metric_values: dict[str, list[float]] = {}
    for run in runs:
        metrics = run.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            num = normalize_number(value)
            if num is None:
                continue
            metric_values.setdefault(str(key), []).append(num)

    means: dict[str, float | None] = {}
    stats: dict[str, dict[str, float | int | None]] = {}
    for key, values in metric_values.items():
        if not values:
            means[key] = None
            stats[key] = {"count": 0, "mean": None, "min": None, "max": None}
            continue
        means[key] = sum(values) / len(values)
        stats[key] = {
            "count": len(values),
            "mean": means[key],
            "min": min(values),
            "max": max(values),
        }
    return means, stats


def run_label(run_payload: dict[str, Any], fallback: str) -> str:
    label = str(run_payload.get("evaluation_label", "")).strip()
    if label:
        return label
    world_name = str(run_payload.get("world_name", "")).strip()
    if world_name:
        return world_name
    return fallback


def summarize_runs(run_paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for index, results_path in enumerate(run_paths, start=1):
        payload = load_json(results_path)
        label = run_label(payload, fallback=f"run_{index:02d}")
        artifacts = payload.get("artifacts", {})
        reference_map_yaml = artifacts.get("reference_map_yaml") if isinstance(artifacts, dict) else None
        runs.append(
            {
                "index": index,
                "label": label,
                "world_name": payload.get("world_name"),
                "artifact_subdir": payload.get("artifact_subdir"),
                "reference_map_yaml": reference_map_yaml,
                "results_json": str(results_path.resolve()),
                "metrics": payload.get("metrics", {}),
                "diagnostics": payload.get("diagnostics", {}),
                "artifacts": artifacts if isinstance(artifacts, dict) else {},
            }
        )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="Katalog eksperymentu, np. out/exp_20260319_123456")
    parser.add_argument("run_results", nargs="+", type=Path, help="Pliki results.json z poszczególnych scenariuszy testowych")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    run_paths = [path.resolve() for path in args.run_results]
    runs = summarize_runs(run_paths)
    if not runs:
        raise SystemExit("Brak wyników do agregacji.")

    aggregate_mean, aggregate_stats = aggregate_metrics(runs)
    primary = runs[0]
    primary_payload = load_json(run_paths[0])

    summary_path = experiment_dir / "multi_test_summary.json"
    results_path = experiment_dir / "results.json"

    top_artifacts = dict(primary.get("artifacts", {}))
    top_artifacts["evaluation_runs_summary_json"] = str(summary_path.resolve())
    top_artifacts["primary_results_json"] = str(run_paths[0].resolve())
    if primary.get("reference_map_yaml"):
        top_artifacts["reference_map_yaml"] = str(primary["reference_map_yaml"])

    payload = {
        "mode": primary_payload.get("mode"),
        "seed": primary_payload.get("seed"),
        "duration_sec": primary_payload.get("duration_sec"),
        "world_name": primary_payload.get("world_name"),
        "evaluation_label": primary_payload.get("evaluation_label"),
        "multi_test": True,
        "test_run_count": len(runs),
        "metrics": aggregate_mean,
        "metrics_summary": aggregate_stats,
        "evaluation_runs": runs,
        "diagnostics": {
            "primary": primary_payload.get("diagnostics", {}),
            "runs": {run["label"]: run.get("diagnostics", {}) for run in runs},
        },
        "artifacts": top_artifacts,
    }

    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] zapisano podsumowanie wielotestowe: {summary_path}")
    print(f"[OK] zaktualizowano główne wyniki eksperymentu: {results_path}")
    print(f"[OK] liczba scenariuszy: {len(runs)}")
    for run in runs:
        print(f"  - {run['label']}: {run['results_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
