#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _meta_obj(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as d:
        meta = d.get("meta", {})
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return meta if isinstance(meta, dict) else {}


def _arr(v) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype == object and a.shape == ():
        a = np.asarray(a.item())
    return np.asarray(a)


def _scalar(v, default, cast):
    try:
        a = _arr(v).reshape(-1)
        if a.size == 0:
            return cast(default)
        return cast(a[0])
    except Exception:
        return cast(default)


def _to_bool(v, default: bool) -> bool:
    try:
        raw = _arr(v).reshape(-1)
        if raw.size == 0:
            return bool(default)
        first = raw[0]
        if isinstance(first, str):
            val = first.strip().lower()
            if val in {"1", "true", "yes", "on"}:
                return True
            if val in {"0", "false", "no", "off"}:
                return False
            return bool(default)
        return bool(int(first))
    except Exception:
        return bool(default)


def _counts(meta: dict, key: str, bins: int) -> np.ndarray:
    arr = _arr(meta.get(key, np.zeros((bins,), dtype=np.int64))).astype(np.int64).reshape(-1)
    return np.pad(arr, (0, max(0, bins - arr.size)))[:bins]


def _finite_or_none(value: float) -> float | None:
    out = float(value)
    return out if np.isfinite(out) else None


def _meta_path(meta: dict, key: str) -> Path | None:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        arr = _arr(raw).reshape(-1)
    except Exception:
        arr = np.asarray([raw], dtype=object)
    if arr.size == 0:
        return None
    text = str(arr[0]).strip()
    if not text:
        return None
    return Path(text)


def _meta_paths(meta: dict, key: str) -> list[Path]:
    raw = meta.get(key, [])
    try:
        arr = _arr(raw).reshape(-1)
    except Exception:
        arr = np.asarray([], dtype=object)
    out: list[Path] = []
    for it in arr.tolist():
        text = str(it).strip()
        if not text:
            continue
        out.append(Path(text))
    return out


def _load_y(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as d:
        y = np.asarray(d["Y"], dtype=np.float32)
    return y


def _concat_metrics_from_npz_sources(
    source_paths: list[Path],
    metric_fn,
) -> np.ndarray:
    vals: list[np.ndarray] = []
    for p in source_paths:
        if not p.exists():
            continue
        try:
            y = _load_y(p)
        except Exception:
            continue
        v = np.asarray(metric_fn(y), dtype=np.float32).reshape(-1)
        if v.size > 0:
            vals.append(v)
    if not vals:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vals, axis=0).astype(np.float32)


def _rywak_metrics_from_y(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            "unknown",
        )
    if arr.shape[1] >= 3:
        # New format: local GT deltas (dx, dy, dtheta)
        return (
            np.linalg.norm(arr[:, :2], axis=1).astype(np.float32),
            np.abs(arr[:, 2]).astype(np.float32),
            "translation_rotation",
        )
    # Legacy format: (v, w)
    return arr[:, 0].astype(np.float32), arr[:, 1].astype(np.float32), "velocity"


def _load_rywak_metrics(npz_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    y = _load_y(npz_path)
    return _rywak_metrics_from_y(y)


def _concat_rywak_metric_from_npz_sources(source_paths: list[Path], which: str) -> np.ndarray:
    vals: list[np.ndarray] = []
    for p in source_paths:
        if not p.exists():
            continue
        try:
            lin, ang, _mode = _load_rywak_metrics(p)
        except Exception:
            continue
        v = lin if which == "linear" else ang
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size > 0:
            vals.append(v)
    if not vals:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vals, axis=0).astype(np.float32)


def _resolve_hist_range(
    *,
    use_abs: bool,
    hist_min: float | None,
    hist_max: float | None,
    values: list[np.ndarray],
) -> tuple[float, float]:
    if hist_min is not None and hist_max is not None:
        try:
            lo = float(hist_min)
            hi = float(hist_max)
            if np.isfinite(lo) and np.isfinite(hi):
                if hi < lo:
                    lo, hi = hi, lo
                if hi > lo:
                    return lo, hi
        except Exception:
            pass

    finite_chunks: list[np.ndarray] = []
    for arr in values:
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if x.size == 0:
            continue
        if use_abs:
            x = np.abs(x)
        x = x[np.isfinite(x)]
        if x.size > 0:
            finite_chunks.append(x)

    if not finite_chunks:
        return float("nan"), float("nan")

    all_vals = np.concatenate(finite_chunks, axis=0)
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1e-6
    return lo, hi


def _hist(
    values: np.ndarray,
    bins: int,
    use_abs: bool,
    *,
    hist_min: float | None = None,
    hist_max: float | None = None,
    clip_to_range: bool = False,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    if use_abs:
        x = np.abs(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros((bins,), dtype=np.int64)

    lo = float(np.min(finite)) if hist_min is None else float(hist_min)
    hi = float(np.max(finite)) if hist_max is None else float(hist_max)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi < lo:
        lo, hi = hi, lo

    if clip_to_range:
        vals = np.clip(finite, lo, hi)
    else:
        vals = finite[(finite >= lo) & (finite <= hi)]
    if vals.size == 0:
        return np.zeros((bins,), dtype=np.int64)

    if float(hi - lo) < 1e-9:
        out = np.zeros((bins,), dtype=np.int64)
        out[0] = vals.size
        return out

    edges = np.linspace(lo, hi, bins + 1, dtype=np.float32)
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, bins - 1)
    return np.bincount(idx, minlength=bins).astype(np.int64)


def _hist_counts_means(
    values: np.ndarray,
    bins: int,
    use_abs: bool,
    *,
    hist_min: float,
    hist_max: float,
    clip_to_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if use_abs:
        x = np.abs(x)
    finite = x[np.isfinite(x)]
    counts = np.zeros((bins,), dtype=np.int64)
    means = np.full((bins,), np.nan, dtype=np.float64)
    if finite.size == 0:
        return counts, means
    lo = float(hist_min)
    hi = float(hist_max)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return counts, means
    if hi < lo:
        lo, hi = hi, lo
    if clip_to_range:
        vals = np.clip(finite, lo, hi)
    else:
        vals = finite[(finite >= lo) & (finite <= hi)]
    if vals.size == 0:
        return counts, means
    if float(hi - lo) < 1e-9:
        counts[0] = int(vals.size)
        means[0] = float(np.mean(vals))
        return counts, means

    edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(np.int64)
    sums = np.bincount(idx, weights=vals.astype(np.float64), minlength=bins).astype(np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz].astype(np.float64)
    return counts, means


def _write_bin_table(
    out_csv: Path,
    *,
    bins: int,
    hist_min: float,
    hist_max: float,
    raw_counts: np.ndarray,
    component_counts: np.ndarray,
    final_counts: np.ndarray,
    component_means: np.ndarray,
    final_means: np.ndarray,
) -> None:
    use_physical_axis = (
        np.isfinite(hist_min) and np.isfinite(hist_max) and float(hist_max) > float(hist_min)
    )
    if use_physical_axis:
        edges = np.linspace(float(hist_min), float(hist_max), bins + 1, dtype=np.float64)
    else:
        edges = np.arange(bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bin_index",
                "bin_min",
                "bin_max",
                "bin_center",
                "raw_count",
                "component_count",
                "final_count",
                "component_mean_value",
                "final_mean_value",
            ]
        )
        for i in range(bins):
            comp_mean = component_means[i]
            fin_mean = final_means[i]
            w.writerow(
                [
                    int(i),
                    float(edges[i]),
                    float(edges[i + 1]),
                    float(centers[i]),
                    int(raw_counts[i]),
                    int(component_counts[i]),
                    int(final_counts[i]),
                    "" if not np.isfinite(comp_mean) else float(comp_mean),
                    "" if not np.isfinite(fin_mean) else float(fin_mean),
                ]
            )


def _plot_single(
    counts: np.ndarray,
    *,
    title: str,
    out_png: Path,
    x_label: str,
    series_label: str = "surowe",
    color: str = "#60a5fa",
    hist_min: float | None = None,
    hist_max: float | None = None,
) -> None:
    bins = len(counts)
    use_physical_axis = (
        hist_min is not None
        and hist_max is not None
        and np.isfinite(hist_min)
        and np.isfinite(hist_max)
        and float(hist_max) > float(hist_min)
    )

    if use_physical_axis:
        edges = np.linspace(float(hist_min), float(hist_max), bins + 1, dtype=np.float64)
        x = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.72 if bins > 0 else 0.72
    else:
        x = np.arange(bins, dtype=np.float64)
        width = 0.72

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, counts, width=width, label=series_label, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label if use_physical_axis else f"{x_label} (bin)")
    ax.set_ylabel("liczba probek")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_pair(
    left_counts: np.ndarray,
    right_counts: np.ndarray,
    *,
    title: str,
    out_png: Path,
    x_label: str,
    left_label: str = "przed balansem",
    right_label: str = "po balansie",
    hist_min: float | None = None,
    hist_max: float | None = None,
) -> None:
    bins = len(left_counts)
    use_physical_axis = (
        hist_min is not None
        and hist_max is not None
        and np.isfinite(hist_min)
        and np.isfinite(hist_max)
        and float(hist_max) > float(hist_min)
    )

    if use_physical_axis:
        edges = np.linspace(float(hist_min), float(hist_max), bins + 1, dtype=np.float64)
        x = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.42 if bins > 0 else 0.42
    else:
        x = np.arange(bins, dtype=np.float64)
        width = 0.42

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, left_counts, width=width, label=left_label, color="#60a5fa")
    ax.bar(x + width / 2, right_counts, width=width, label=right_label, color="#34d399")
    ax.set_title(title)
    ax.set_xlabel(x_label if use_physical_axis else f"{x_label} (bin)")
    ax.set_ylabel("liczba probek")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _load_y_component(npz_path: Path, col: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as d:
        y = np.asarray(d["Y"], dtype=np.float32)
    return y[:, col] if y.ndim == 2 else np.zeros((0,), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Raport histogramow: bez uciecia vs po ucieciu.")
    ap.add_argument("--experiment-dir", required=True, help="np. out/exp_20260329_123456")
    args = ap.parse_args()

    exp = Path(args.experiment_dir).resolve()
    if not exp.is_dir():
        raise SystemExit(f"Brak katalogu: {exp}")
    out_dir = exp / "hist_balance_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    # RYWAK
    rywak = exp / "dataset_rywak.npz"
    if rywak.exists():
        m = _meta_obj(rywak)
        strict_mode = _to_bool(m.get("strict_unique_equalized", 0), False)
        bins = max(1, _scalar(m.get("balance_bins", m.get("bins", 24)), 24, int))
        use_abs_lin = _to_bool(m.get("balance_linear_use_abs", 1), True)
        use_abs_ang = _to_bool(m.get("balance_angular_use_abs", 1), True)
        raw_lin = _counts(m, "balance_linear_counts_per_bin", bins)
        raw_ang = _counts(m, "balance_angular_counts_per_bin", bins)
        cut_lin = _counts(m, "balance_linear_selected_counts_per_bin", bins)
        cut_ang = _counts(m, "balance_angular_selected_counts_per_bin", bins)
        lin_min = _scalar(
            m.get(
                "balance_linear_hist_min_mps",
                m.get("balance_translation_hist_min_m", m.get("row_hist_min", np.nan)),
            ),
            np.nan,
            float,
        )
        lin_max = _scalar(
            m.get(
                "balance_linear_hist_max_mps",
                m.get("balance_translation_hist_max_m", m.get("row_hist_max", np.nan)),
            ),
            np.nan,
            float,
        )
        ang_min = _scalar(
            m.get(
                "balance_angular_hist_min_radps",
                m.get("balance_rotation_hist_min_rad", m.get("col_hist_min", np.nan)),
            ),
            np.nan,
            float,
        )
        ang_max = _scalar(
            m.get(
                "balance_angular_hist_max_radps",
                m.get("balance_rotation_hist_max_rad", m.get("col_hist_max", np.nan)),
            ),
            np.nan,
            float,
        )

        lin_cut = exp / "dataset_rywak_linear_balanced.npz"
        ang_cut = exp / "dataset_rywak_angular_balanced.npz"
        final_lin_vals, final_ang_vals, rywak_metric_mode = _load_rywak_metrics(rywak)
        if lin_cut.exists():
            lin_cut_vals, _ang_unused, _mode_unused = _load_rywak_metrics(lin_cut)
        else:
            lin_cut_vals = np.zeros((0,), dtype=np.float32)
        if ang_cut.exists():
            _lin_unused, ang_cut_vals, _mode_unused = _load_rywak_metrics(ang_cut)
        else:
            ang_cut_vals = np.zeros((0,), dtype=np.float32)
        source_lin_vals = np.zeros((0,), dtype=np.float32)  # unique before balance
        source_ang_vals = np.zeros((0,), dtype=np.float32)  # unique before balance
        raw_source_lin_vals = np.zeros((0,), dtype=np.float32)  # raw before unique
        raw_source_ang_vals = np.zeros((0,), dtype=np.float32)  # raw before unique
        source_path = _meta_path(m, "source_path")
        if strict_mode and source_path is not None and source_path.exists():
            source_lin_vals, source_ang_vals, _mode_unused = _load_rywak_metrics(source_path)
            source_meta = _meta_obj(source_path)
            source_paths = _meta_paths(source_meta, "source_paths")
            raw_source_lin_vals = _concat_rywak_metric_from_npz_sources(source_paths, "linear")
            raw_source_ang_vals = _concat_rywak_metric_from_npz_sources(source_paths, "angular")

        lin_lo, lin_hi = _resolve_hist_range(
            use_abs=use_abs_lin,
            hist_min=lin_min,
            hist_max=lin_max,
            values=[final_lin_vals, lin_cut_vals, source_lin_vals, raw_source_lin_vals],
        )
        ang_lo, ang_hi = _resolve_hist_range(
            use_abs=use_abs_ang,
            hist_min=ang_min,
            hist_max=ang_max,
            values=[final_ang_vals, ang_cut_vals, source_ang_vals, raw_source_ang_vals],
        )

        raw_lin_mean = np.full((bins,), np.nan, dtype=np.float64)
        raw_ang_mean = np.full((bins,), np.nan, dtype=np.float64)
        if raw_source_lin_vals.size > 0:
            raw_lin, raw_lin_mean = _hist_counts_means(
                raw_source_lin_vals,
                bins,
                use_abs_lin,
                hist_min=lin_lo,
                hist_max=lin_hi,
                clip_to_range=True,
            )
        elif source_lin_vals.size > 0:
            raw_lin, raw_lin_mean = _hist_counts_means(
                source_lin_vals,
                bins,
                use_abs_lin,
                hist_min=lin_lo,
                hist_max=lin_hi,
                clip_to_range=True,
            )
        elif np.all(raw_lin == 0):
            raw_lin, raw_lin_mean = _hist_counts_means(
                final_lin_vals,
                bins,
                use_abs_lin,
                hist_min=lin_lo,
                hist_max=lin_hi,
                clip_to_range=True,
            )
        if raw_source_ang_vals.size > 0:
            raw_ang, raw_ang_mean = _hist_counts_means(
                raw_source_ang_vals,
                bins,
                use_abs_ang,
                hist_min=ang_lo,
                hist_max=ang_hi,
                clip_to_range=True,
            )
        elif source_ang_vals.size > 0:
            raw_ang, raw_ang_mean = _hist_counts_means(
                source_ang_vals,
                bins,
                use_abs_ang,
                hist_min=ang_lo,
                hist_max=ang_hi,
                clip_to_range=True,
            )
        elif np.all(raw_ang == 0):
            raw_ang, raw_ang_mean = _hist_counts_means(
                final_ang_vals,
                bins,
                use_abs_ang,
                hist_min=ang_lo,
                hist_max=ang_hi,
                clip_to_range=True,
            )
        merged_lin, merged_lin_mean = _hist_counts_means(
            final_lin_vals,
            bins,
            use_abs_lin,
            hist_min=lin_lo,
            hist_max=lin_hi,
            clip_to_range=True,
        )
        merged_ang, merged_ang_mean = _hist_counts_means(
            final_ang_vals,
            bins,
            use_abs_ang,
            hist_min=ang_lo,
            hist_max=ang_hi,
            clip_to_range=True,
        )
        if strict_mode and source_lin_vals.size > 0:
            cut_lin, cut_lin_mean = _hist_counts_means(
                source_lin_vals,
                bins,
                use_abs_lin,
                hist_min=lin_lo,
                hist_max=lin_hi,
                clip_to_range=True,
            )
        elif lin_cut_vals.size > 0:
            cut_lin, cut_lin_mean = _hist_counts_means(
                lin_cut_vals,
                bins,
                use_abs_lin,
                hist_min=lin_lo,
                hist_max=lin_hi,
                clip_to_range=True,
            )
        else:
            cut_lin_mean = np.full((bins,), np.nan, dtype=np.float64)
        if strict_mode and source_ang_vals.size > 0:
            cut_ang, cut_ang_mean = _hist_counts_means(
                source_ang_vals,
                bins,
                use_abs_ang,
                hist_min=ang_lo,
                hist_max=ang_hi,
                clip_to_range=True,
            )
        elif ang_cut_vals.size > 0:
            cut_ang, cut_ang_mean = _hist_counts_means(
                ang_cut_vals,
                bins,
                use_abs_ang,
                hist_min=ang_lo,
                hist_max=ang_hi,
                clip_to_range=True,
            )
        else:
            cut_ang_mean = np.full((bins,), np.nan, dtype=np.float64)

        if rywak_metric_mode == "translation_rotation":
            lin_title = "Rywak: translacja miedzy skanami (dane surowe)"
            lin_title_pair = "Rywak: translacja miedzy skanami (unikalne: przed/po balansie)"
            lin_x_label = "|delta_trans| [m]"
            ang_title = "Rywak: rotacja miedzy skanami (dane surowe)"
            ang_title_pair = "Rywak: rotacja miedzy skanami (unikalne: przed/po balansie)"
            ang_x_label = "|delta_yaw| [rad]"
        else:
            lin_title = "Rywak: predkosc liniowa (dane surowe)"
            lin_title_pair = "Rywak: predkosc liniowa (unikalne: przed/po balansie)"
            lin_x_label = "v [m/s]"
            ang_title = "Rywak: predkosc katowa (dane surowe)"
            ang_title_pair = "Rywak: predkosc katowa (unikalne: przed/po balansie)"
            ang_x_label = "omega [rad/s]"

        _plot_single(
            raw_lin,
            title=lin_title,
            out_png=out_dir / "rywak_linear_hist_raw.png",
            x_label=lin_x_label,
            hist_min=lin_lo,
            hist_max=lin_hi,
        )
        _plot_pair(
            cut_lin,
            merged_lin,
            title=lin_title_pair,
            out_png=out_dir / "rywak_linear_hist_unique_before_after.png",
            x_label=lin_x_label,
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=lin_lo,
            hist_max=lin_hi,
        )
        _plot_single(
            raw_ang,
            title=ang_title,
            out_png=out_dir / "rywak_angular_hist_raw.png",
            x_label=ang_x_label,
            hist_min=ang_lo,
            hist_max=ang_hi,
        )
        _plot_pair(
            cut_ang,
            merged_ang,
            title=ang_title_pair,
            out_png=out_dir / "rywak_angular_hist_unique_before_after.png",
            x_label=ang_x_label,
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=ang_lo,
            hist_max=ang_hi,
        )
        # Backward-compatibility names.
        _plot_pair(
            cut_lin,
            merged_lin,
            title=lin_title_pair,
            out_png=out_dir / "rywak_linear_hist_compare.png",
            x_label=lin_x_label,
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=lin_lo,
            hist_max=lin_hi,
        )
        _plot_pair(
            cut_ang,
            merged_ang,
            title=ang_title_pair,
            out_png=out_dir / "rywak_angular_hist_compare.png",
            x_label=ang_x_label,
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=ang_lo,
            hist_max=ang_hi,
        )
        _write_bin_table(
            out_dir / "rywak_linear_bins.csv",
            bins=bins,
            hist_min=lin_lo,
            hist_max=lin_hi,
            raw_counts=raw_lin,
            component_counts=cut_lin,
            final_counts=merged_lin,
            component_means=cut_lin_mean,
            final_means=merged_lin_mean,
        )
        _write_bin_table(
            out_dir / "rywak_angular_bins.csv",
            bins=bins,
            hist_min=ang_lo,
            hist_max=ang_hi,
            raw_counts=raw_ang,
            component_counts=cut_ang,
            final_counts=merged_ang,
            component_means=cut_ang_mean,
            final_means=merged_ang_mean,
        )

        summary["rywak"] = {
            "raw_samples": int(raw_source_lin_vals.size) if raw_source_lin_vals.size > 0 else _scalar(
                m.get("balance_raw_sample_count", m.get("n_after_dedup_before_balance", 0)),
                0,
                int,
            ),
            "selected_linear": int(np.sum(cut_lin)),
            "selected_angular": int(np.sum(cut_ang)),
            "selected_merged": _scalar(m.get("balance_merged_selected_count", m.get("n_output", 0)), 0, int),
            "cut_for_equalization": _scalar(
                (
                    int(raw_source_lin_vals.size)
                    if raw_source_lin_vals.size > 0
                    else m.get("balance_raw_sample_count", m.get("n_after_dedup_before_balance", 0))
                ),
                0,
                int,
            )
            - _scalar(m.get("balance_merged_selected_count", m.get("n_output", 0)), 0, int),
            "metric_mode": rywak_metric_mode,
            "linear_hist_range_mps": [_finite_or_none(lin_lo), _finite_or_none(lin_hi)],
            "angular_hist_range_radps": [_finite_or_none(ang_lo), _finite_or_none(ang_hi)],
            "linear_metric_unit": "m" if rywak_metric_mode == "translation_rotation" else "m/s",
            "angular_metric_unit": "rad" if rywak_metric_mode == "translation_rotation" else "rad/s",
            "linear_non_empty_bins": _scalar(m.get("balance_linear_bins_non_empty", np.count_nonzero(raw_lin)), 0, int),
            "angular_non_empty_bins": _scalar(m.get("balance_angular_bins_non_empty", np.count_nonzero(raw_ang)), 0, int),
            "merge_strategy": (
                "strict_unique_equalized"
                if strict_mode
                else str(_scalar(m.get("balance_merge_strategy", "union_unique"), "union_unique", str))
            ),
            "final_linear_non_empty_bins": int(np.count_nonzero(merged_lin)),
            "final_angular_non_empty_bins": int(np.count_nonzero(merged_ang)),
            "component_linear_non_empty_bins": int(np.count_nonzero(cut_lin)),
            "component_angular_non_empty_bins": int(np.count_nonzero(cut_ang)),
            "raw_source_path": str(source_path.resolve()) if source_path is not None and source_path.exists() else None,
            "linear_raw_hist_png": str((out_dir / "rywak_linear_hist_raw.png").resolve()),
            "linear_unique_before_after_hist_png": str(
                (out_dir / "rywak_linear_hist_unique_before_after.png").resolve()
            ),
            "angular_raw_hist_png": str((out_dir / "rywak_angular_hist_raw.png").resolve()),
            "angular_unique_before_after_hist_png": str(
                (out_dir / "rywak_angular_hist_unique_before_after.png").resolve()
            ),
            "linear_bin_table_csv": str((out_dir / "rywak_linear_bins.csv").resolve()),
            "angular_bin_table_csv": str((out_dir / "rywak_angular_bins.csv").resolve()),
        }

    # ROBAK
    robak = exp / "dataset_robak.npz"
    if robak.exists():
        m = _meta_obj(robak)
        strict_mode = _to_bool(m.get("strict_unique_equalized", 0), False)
        bins = max(1, _scalar(m.get("balance_bins", m.get("bins", 24)), 24, int))
        use_abs_t = _to_bool(m.get("balance_translation_use_abs", 0), False)
        use_abs_r = _to_bool(m.get("balance_rotation_use_abs", 1), True)
        raw_t = _counts(m, "balance_translation_counts_per_bin", bins)
        raw_r = _counts(m, "balance_rotation_counts_per_bin", bins)
        cut_t = _counts(m, "balance_translation_selected_counts_per_bin", bins)
        cut_r = _counts(m, "balance_rotation_selected_counts_per_bin", bins)
        trans_min = _scalar(m.get("balance_translation_hist_min_m", m.get("row_hist_min", np.nan)), np.nan, float)
        trans_max = _scalar(m.get("balance_translation_hist_max_m", m.get("row_hist_max", np.nan)), np.nan, float)
        rot_min = _scalar(m.get("balance_rotation_hist_min_deg", m.get("col_hist_min", np.nan)), np.nan, float)
        rot_max = _scalar(m.get("balance_rotation_hist_max_deg", m.get("col_hist_max", np.nan)), np.nan, float)

        t_cut = exp / "dataset_robak_translation_balanced.npz"
        r_cut = exp / "dataset_robak_rotation_balanced.npz"
        t_cut_vals = np.zeros((0,), dtype=np.float32)
        r_cut_vals = np.zeros((0,), dtype=np.float32)
        if t_cut.exists():
            with np.load(t_cut, allow_pickle=True) as d:
                y = np.asarray(d["Y"], dtype=np.float32)
            t_cut_vals = np.linalg.norm(y[:, :2], axis=1).astype(np.float32)
        if r_cut.exists():
            r_cut_vals = np.rad2deg(_load_y_component(r_cut, 2)).astype(np.float32)
        with np.load(robak, allow_pickle=True) as d:
            y_final_robak = np.asarray(d["Y"], dtype=np.float32)
        final_t_vals = np.linalg.norm(y_final_robak[:, :2], axis=1).astype(np.float32)
        final_r_vals = np.rad2deg(np.asarray(y_final_robak[:, 2], dtype=np.float32))
        source_t_vals = np.zeros((0,), dtype=np.float32)  # unique before balance
        source_r_vals = np.zeros((0,), dtype=np.float32)  # unique before balance
        raw_source_t_vals = np.zeros((0,), dtype=np.float32)  # raw before unique
        raw_source_r_vals = np.zeros((0,), dtype=np.float32)  # raw before unique
        source_path = _meta_path(m, "source_path")
        if strict_mode and source_path is not None and source_path.exists():
            with np.load(source_path, allow_pickle=True) as d:
                y_src = np.asarray(d["Y"], dtype=np.float32)
            source_t_vals = np.linalg.norm(y_src[:, :2], axis=1).astype(np.float32)
            source_r_vals = np.rad2deg(np.asarray(y_src[:, 2], dtype=np.float32))
            source_meta = _meta_obj(source_path)
            source_paths = _meta_paths(source_meta, "source_paths")
            raw_source_t_vals = _concat_metrics_from_npz_sources(
                source_paths,
                lambda y: np.linalg.norm(y[:, :2], axis=1).astype(np.float32) if y.ndim == 2 else np.zeros((0,), dtype=np.float32),
            )
            raw_source_r_vals = _concat_metrics_from_npz_sources(
                source_paths,
                lambda y: np.rad2deg(np.asarray(y[:, 2], dtype=np.float32)) if y.ndim == 2 else np.zeros((0,), dtype=np.float32),
            )

        trans_lo, trans_hi = _resolve_hist_range(
            use_abs=use_abs_t,
            hist_min=trans_min,
            hist_max=trans_max,
            values=[final_t_vals, t_cut_vals, source_t_vals, raw_source_t_vals],
        )
        rot_lo, rot_hi = _resolve_hist_range(
            use_abs=use_abs_r,
            hist_min=rot_min,
            hist_max=rot_max,
            values=[final_r_vals, r_cut_vals, source_r_vals, raw_source_r_vals],
        )
        raw_t_mean = np.full((bins,), np.nan, dtype=np.float64)
        raw_r_mean = np.full((bins,), np.nan, dtype=np.float64)
        if raw_source_t_vals.size > 0:
            raw_t, raw_t_mean = _hist_counts_means(
                raw_source_t_vals,
                bins,
                use_abs_t,
                hist_min=trans_lo,
                hist_max=trans_hi,
                clip_to_range=True,
            )
        elif source_t_vals.size > 0:
            raw_t, raw_t_mean = _hist_counts_means(
                source_t_vals,
                bins,
                use_abs_t,
                hist_min=trans_lo,
                hist_max=trans_hi,
                clip_to_range=True,
            )
        elif np.all(raw_t == 0):
            raw_t, raw_t_mean = _hist_counts_means(
                final_t_vals,
                bins,
                use_abs_t,
                hist_min=trans_lo,
                hist_max=trans_hi,
                clip_to_range=True,
            )
        if raw_source_r_vals.size > 0:
            raw_r, raw_r_mean = _hist_counts_means(
                raw_source_r_vals,
                bins,
                use_abs_r,
                hist_min=rot_lo,
                hist_max=rot_hi,
                clip_to_range=True,
            )
        elif source_r_vals.size > 0:
            raw_r, raw_r_mean = _hist_counts_means(
                source_r_vals,
                bins,
                use_abs_r,
                hist_min=rot_lo,
                hist_max=rot_hi,
                clip_to_range=True,
            )
        elif np.all(raw_r == 0):
            raw_r, raw_r_mean = _hist_counts_means(
                final_r_vals,
                bins,
                use_abs_r,
                hist_min=rot_lo,
                hist_max=rot_hi,
                clip_to_range=True,
            )
        final_t, final_t_mean = _hist_counts_means(
            final_t_vals,
            bins,
            use_abs_t,
            hist_min=trans_lo,
            hist_max=trans_hi,
            clip_to_range=True,
        )
        final_r, final_r_mean = _hist_counts_means(
            final_r_vals,
            bins,
            use_abs_r,
            hist_min=rot_lo,
            hist_max=rot_hi,
            clip_to_range=True,
        )
        if strict_mode and source_t_vals.size > 0:
            cut_t, cut_t_mean = _hist_counts_means(
                source_t_vals,
                bins,
                use_abs_t,
                hist_min=trans_lo,
                hist_max=trans_hi,
                clip_to_range=True,
            )
        elif t_cut_vals.size > 0:
            cut_t, cut_t_mean = _hist_counts_means(
                t_cut_vals,
                bins,
                use_abs_t,
                hist_min=trans_lo,
                hist_max=trans_hi,
                clip_to_range=True,
            )
        else:
            cut_t_mean = np.full((bins,), np.nan, dtype=np.float64)
        if strict_mode and source_r_vals.size > 0:
            cut_r, cut_r_mean = _hist_counts_means(
                source_r_vals,
                bins,
                use_abs_r,
                hist_min=rot_lo,
                hist_max=rot_hi,
                clip_to_range=True,
            )
        elif r_cut_vals.size > 0:
            cut_r, cut_r_mean = _hist_counts_means(
                r_cut_vals,
                bins,
                use_abs_r,
                hist_min=rot_lo,
                hist_max=rot_hi,
                clip_to_range=True,
            )
        else:
            cut_r_mean = np.full((bins,), np.nan, dtype=np.float64)

        _plot_single(
            raw_t,
            title="Robak: translacja pary skanow (dane surowe)",
            out_png=out_dir / "robak_translation_hist_raw.png",
            x_label="|delta_xy| [m]" if use_abs_t else "delta_xy [m]",
            hist_min=trans_lo,
            hist_max=trans_hi,
        )
        _plot_pair(
            cut_t,
            final_t,
            title="Robak: translacja pary skanow (unikalne: przed/po balansie)",
            out_png=out_dir / "robak_translation_hist_unique_before_after.png",
            x_label="|delta_xy| [m]" if use_abs_t else "delta_xy [m]",
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=trans_lo,
            hist_max=trans_hi,
        )
        _plot_single(
            raw_r,
            title="Robak: rotacja pary skanow (dane surowe)",
            out_png=out_dir / "robak_rotation_hist_raw.png",
            x_label="|delta_yaw| [deg]" if use_abs_r else "delta_yaw [deg]",
            hist_min=rot_lo,
            hist_max=rot_hi,
        )
        _plot_pair(
            cut_r,
            final_r,
            title="Robak: rotacja pary skanow (unikalne: przed/po balansie)",
            out_png=out_dir / "robak_rotation_hist_unique_before_after.png",
            x_label="|delta_yaw| [deg]" if use_abs_r else "delta_yaw [deg]",
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=rot_lo,
            hist_max=rot_hi,
        )
        # Backward-compatibility names.
        _plot_pair(
            cut_t,
            final_t,
            title="Robak: translacja pary skanow (unikalne: przed/po balansie)",
            out_png=out_dir / "robak_translation_hist_compare.png",
            x_label="|delta_xy| [m]" if use_abs_t else "delta_xy [m]",
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=trans_lo,
            hist_max=trans_hi,
        )
        _plot_pair(
            cut_r,
            final_r,
            title="Robak: rotacja pary skanow (unikalne: przed/po balansie)",
            out_png=out_dir / "robak_rotation_hist_compare.png",
            x_label="|delta_yaw| [deg]" if use_abs_r else "delta_yaw [deg]",
            left_label="unikalne przed balansem",
            right_label="po balansie",
            hist_min=rot_lo,
            hist_max=rot_hi,
        )
        _write_bin_table(
            out_dir / "robak_translation_bins.csv",
            bins=bins,
            hist_min=trans_lo,
            hist_max=trans_hi,
            raw_counts=raw_t,
            component_counts=cut_t,
            final_counts=final_t,
            component_means=cut_t_mean,
            final_means=final_t_mean,
        )
        _write_bin_table(
            out_dir / "robak_rotation_bins.csv",
            bins=bins,
            hist_min=rot_lo,
            hist_max=rot_hi,
            raw_counts=raw_r,
            component_counts=cut_r,
            final_counts=final_r,
            component_means=cut_r_mean,
            final_means=final_r_mean,
        )

        summary["robak"] = {
            "raw_samples": int(raw_source_t_vals.size) if raw_source_t_vals.size > 0 else _scalar(
                m.get("balance_raw_sample_count", m.get("n_after_dedup_before_balance", 0)),
                0,
                int,
            ),
            "selected_translation": int(np.sum(cut_t)),
            "selected_rotation": int(np.sum(cut_r)),
            "selected_merged": _scalar(m.get("balance_merged_selected_count", m.get("n_output", 0)), 0, int),
            "cut_for_equalization": _scalar(
                (
                    int(raw_source_t_vals.size)
                    if raw_source_t_vals.size > 0
                    else m.get("balance_raw_sample_count", m.get("n_after_dedup_before_balance", 0))
                ),
                0,
                int,
            )
            - _scalar(m.get("balance_merged_selected_count", m.get("n_output", 0)), 0, int),
            "translation_hist_range_m": [_finite_or_none(trans_lo), _finite_or_none(trans_hi)],
            "rotation_hist_range_deg": [_finite_or_none(rot_lo), _finite_or_none(rot_hi)],
            "translation_non_empty_bins": _scalar(m.get("balance_translation_bins_non_empty", np.count_nonzero(raw_t)), 0, int),
            "rotation_non_empty_bins": _scalar(m.get("balance_rotation_bins_non_empty", np.count_nonzero(raw_r)), 0, int),
            "merge_strategy": (
                "strict_unique_equalized"
                if strict_mode
                else str(_scalar(m.get("balance_merge_strategy", "union_unique"), "union_unique", str))
            ),
            "final_translation_non_empty_bins": int(np.count_nonzero(final_t)),
            "final_rotation_non_empty_bins": int(np.count_nonzero(final_r)),
            "component_translation_non_empty_bins": int(np.count_nonzero(cut_t)),
            "component_rotation_non_empty_bins": int(np.count_nonzero(cut_r)),
            "raw_source_path": str(source_path.resolve()) if source_path is not None and source_path.exists() else None,
            "translation_raw_hist_png": str((out_dir / "robak_translation_hist_raw.png").resolve()),
            "translation_unique_before_after_hist_png": str(
                (out_dir / "robak_translation_hist_unique_before_after.png").resolve()
            ),
            "rotation_raw_hist_png": str((out_dir / "robak_rotation_hist_raw.png").resolve()),
            "rotation_unique_before_after_hist_png": str(
                (out_dir / "robak_rotation_hist_unique_before_after.png").resolve()
            ),
            "translation_bin_table_csv": str((out_dir / "robak_translation_bins.csv").resolve()),
            "rotation_bin_table_csv": str((out_dir / "robak_rotation_bins.csv").resolve()),
        }

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Raport: {out_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
