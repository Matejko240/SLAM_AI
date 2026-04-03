#!/usr/bin/env python3
"""Kanoniczne klucze metryk w results.json i aliasy legace (stare eksperymenty)."""

from __future__ import annotations

from typing import Any

# Kolejność: najpierw kanon, potem alias dla starych plików wyników.
RMSE_XY_ODOM_KEYS: tuple[str, ...] = ("rmse_xy_odom_topic", "rmse_xy_baseline")
RMSE_THETA_ODOM_KEYS: tuple[str, ...] = ("rmse_theta_odom_topic", "rmse_theta_baseline")


def metrics_coalesce(metrics: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if value is None or value == "":
            continue
        return value
    return None


def metrics_rmse_xy_odom(metrics: dict[str, Any] | None) -> Any:
    return metrics_coalesce(metrics, RMSE_XY_ODOM_KEYS)


def metrics_rmse_theta_odom(metrics: dict[str, Any] | None) -> Any:
    return metrics_coalesce(metrics, RMSE_THETA_ODOM_KEYS)
