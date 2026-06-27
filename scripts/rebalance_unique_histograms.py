#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Edge:
    to: int
    rev: int
    cap: int


class Dinic:
    def __init__(self, n: int):
        self.n = int(n)
        self.g: list[list[Edge]] = [[] for _ in range(self.n)]

    def add_edge(self, fr: int, to: int, cap: int) -> None:
        cap_i = int(cap)
        fwd = Edge(to=to, rev=len(self.g[to]), cap=cap_i)
        rev = Edge(to=fr, rev=len(self.g[fr]), cap=0)
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def _bfs_level(self, s: int, t: int) -> list[int]:
        level = [-1] * self.n
        q: list[int] = [s]
        level[s] = 0
        head = 0
        while head < len(q):
            v = q[head]
            head += 1
            for e in self.g[v]:
                if e.cap > 0 and level[e.to] < 0:
                    level[e.to] = level[v] + 1
                    if e.to == t:
                        return level
                    q.append(e.to)
        return level

    def _dfs(self, v: int, t: int, f: int, level: list[int], it: list[int]) -> int:
        if v == t:
            return f
        while it[v] < len(self.g[v]):
            i = it[v]
            e = self.g[v][i]
            if e.cap > 0 and level[v] < level[e.to]:
                d = self._dfs(e.to, t, min(f, e.cap), level, it)
                if d > 0:
                    e.cap -= d
                    self.g[e.to][e.rev].cap += d
                    return d
            it[v] += 1
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        inf = 10**18
        while True:
            level = self._bfs_level(s, t)
            if level[t] < 0:
                return flow
            it = [0] * self.n
            while True:
                f = self._dfs(s, t, inf, level, it)
                if f == 0:
                    break
                flow += f


def _bin_indices(values: np.ndarray, bins: int, vmin: float, vmax: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid histogram range: [{vmin}, {vmax}]")
    clipped = np.clip(vals, vmin, vmax)
    scaled = (clipped - vmin) / (vmax - vmin)
    idx = np.floor(scaled * float(bins)).astype(np.int64)
    return np.clip(idx, 0, int(bins) - 1)


def _deduplicate_xy(
    features: np.ndarray,
    labels: np.ndarray,
    pose_key: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray, str]:
    if features.shape[0] == 0:
        return (
            features,
            labels,
            pose_key,
            0,
            np.zeros((0,), dtype=np.int64),
            "X+Y",
        )
    flat_x = np.ascontiguousarray(features.reshape((features.shape[0], -1)))
    flat_y = np.ascontiguousarray(labels.reshape((labels.shape[0], -1)))
    merged = np.concatenate([flat_x, flat_y], axis=1)
    dedup_key = "X+Y"
    merged_view = np.ascontiguousarray(merged).view(
        np.dtype((np.void, merged.dtype.itemsize * merged.shape[1]))
    )
    _, unique_idx = np.unique(merged_view, return_index=True)
    unique_idx = np.sort(unique_idx.astype(np.int64))
    removed = int(features.shape[0] - unique_idx.size)
    pose_out = pose_key[unique_idx] if pose_key is not None else None
    return features[unique_idx], labels[unique_idx], pose_out, removed, unique_idx, dedup_key


def _build_cell_counts(r_idx: np.ndarray, c_idx: np.ndarray, bins: int) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    counts = np.zeros((bins, bins), dtype=np.int64)
    cell_lists: list[list[list[int]]] = [[[] for _ in range(bins)] for _ in range(bins)]
    for k, (ri, ci) in enumerate(zip(r_idx.tolist(), c_idx.tolist())):
        counts[ri, ci] += 1
        cell_lists[ri][ci].append(k)
    cells = [[np.asarray(cell_lists[i][j], dtype=np.int64) for j in range(bins)] for i in range(bins)]
    return counts, cells


def _flow_for_target(cell_counts: np.ndarray, target_per_bin: int) -> tuple[bool, np.ndarray]:
    bins = int(cell_counts.shape[0])
    source = 0
    row_base = 1
    col_base = row_base + bins
    sink = col_base + bins
    node_count = sink + 1

    flow_net = Dinic(node_count)
    edge_refs: dict[tuple[int, int], Edge] = {}

    for i in range(bins):
        flow_net.add_edge(source, row_base + i, int(target_per_bin))
    for j in range(bins):
        flow_net.add_edge(col_base + j, sink, int(target_per_bin))
    for i in range(bins):
        for j in range(bins):
            cap = int(cell_counts[i, j])
            if cap <= 0:
                continue
            from_node = row_base + i
            to_node = col_base + j
            prev_len = len(flow_net.g[from_node])
            flow_net.add_edge(from_node, to_node, cap)
            edge_refs[(i, j)] = flow_net.g[from_node][prev_len]

    required = int(bins * target_per_bin)
    maxflow = flow_net.max_flow(source, sink)
    if maxflow != required:
        return False, np.zeros_like(cell_counts, dtype=np.int64)

    selected_counts = np.zeros_like(cell_counts, dtype=np.int64)
    for (i, j), e in edge_refs.items():
        used = int(cell_counts[i, j] - e.cap)
        if used > 0:
            selected_counts[i, j] = used
    return True, selected_counts


def _max_feasible_target(cell_counts: np.ndarray, row_counts: np.ndarray, col_counts: np.ndarray) -> tuple[int, np.ndarray]:
    hi = int(min(int(np.min(row_counts)), int(np.min(col_counts))))
    if hi <= 0:
        return 0, np.zeros_like(cell_counts, dtype=np.int64)
    lo = 0
    best = np.zeros_like(cell_counts, dtype=np.int64)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ok, selected = _flow_for_target(cell_counts, mid)
        if ok:
            lo = mid
            best = selected
        else:
            hi = mid - 1
    if lo <= 0:
        return 0, np.zeros_like(cell_counts, dtype=np.int64)
    if best.sum() == 0:
        ok, selected = _flow_for_target(cell_counts, lo)
        if not ok:
            return 0, np.zeros_like(cell_counts, dtype=np.int64)
        best = selected
    return lo, best


def _select_indices_from_cells(
    cell_indices: list[list[np.ndarray]],
    selected_counts: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    chunks: list[np.ndarray] = []
    bins = int(selected_counts.shape[0])
    for i in range(bins):
        for j in range(bins):
            k = int(selected_counts[i, j])
            if k <= 0:
                continue
            pool = cell_indices[i][j]
            if pool.size < k:
                raise RuntimeError(f"Cell[{i},{j}] has {pool.size} samples, requires {k}")
            if pool.size == k:
                chosen = pool
            else:
                chosen = np.sort(rng.choice(pool, size=k, replace=False).astype(np.int64))
            chunks.append(chosen)
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.sort(np.concatenate(chunks, axis=0).astype(np.int64))


def _rywak_metrics(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    y = np.asarray(labels, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"Invalid Rywak labels shape: {y.shape}. Expected (N,2) or (N,3).")
    if y.shape[1] >= 3:
        metric_row = np.linalg.norm(y[:, :2], axis=1).astype(np.float32)
        metric_col = np.abs(y[:, 2]).astype(np.float32)
        return metric_row, metric_col, "translation_rotation", "m", "rad"
    metric_row = np.abs(y[:, 0]).astype(np.float32)
    metric_col = np.abs(y[:, 1]).astype(np.float32)
    return metric_row, metric_col, "velocity", "m/s", "rad/s"


def _mirror_augment_robak(
    features: np.ndarray, labels: np.ndarray, pose: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mirror-augment Robak: reverse scan beams, negate dy and dθ."""
    feat_m = features[:, :, ::-1].copy()
    lab_m = labels.copy()
    lab_m[:, 1] *= -1
    lab_m[:, 2] *= -1
    pose_m = np.full_like(pose, np.nan) if pose is not None else None
    feat_cat = np.concatenate([features, feat_m], axis=0)
    lab_cat = np.concatenate([labels, lab_m], axis=0)
    pose_cat = np.concatenate([pose, pose_m], axis=0) if pose is not None else None
    return feat_cat, lab_cat, pose_cat


def _mirror_augment_rywak(
    features: np.ndarray, labels: np.ndarray, pose: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mirror-augment Rywak: negate dθ, reverse delta scan, negate ω."""
    feat_m = features.copy()
    feat_m[:, 0] *= -1
    feat_m[:, 1] *= -1
    feat_m[:, 2:] = feat_m[:, 2:][:, ::-1]
    lab_m = labels.copy()
    lab_m[:, 1] *= -1
    pose_m = np.full_like(pose, np.nan) if pose is not None else None
    feat_cat = np.concatenate([features, feat_m], axis=0)
    lab_cat = np.concatenate([labels, lab_m], axis=0)
    pose_cat = np.concatenate([pose, pose_m], axis=0) if pose is not None else None
    return feat_cat, lab_cat, pose_cat


def _rebalance_one(
    npz_path: Path,
    out_path: Path,
    *,
    feature_key: str,
    bins: int,
    metric_row: np.ndarray,
    metric_col: np.ndarray,
    row_range: tuple[float, float],
    col_range: tuple[float, float],
    seed: int,
    require_all_bins: bool,
    label: str,
    mirror_augment: str = "none",
    require_offsets: bool = False,
) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        if feature_key not in data or "Y" not in data:
            raise KeyError(f"{npz_path}: expected keys {feature_key!r} and 'Y'")
        features_raw = np.asarray(data[feature_key], dtype=np.float32)
        labels_raw = np.asarray(data["Y"], dtype=np.float32)
        pose_raw = np.asarray(data["P"], dtype=np.float32) if "P" in data else None
        noff_raw = np.asarray(data["N"], dtype=np.int32) if "N" in data else None

    if require_offsets and noff_raw is None:
        return {
            "success": False,
            "reason": "missing_offsets",
            "label": label,
            "input_path": str(npz_path),
            "output_path": str(out_path),
            "bins": int(bins),
            "n_input": int(features_raw.shape[0]),
            "mirror_augment": mirror_augment,
            "message": (
                "Robak rebalance requires per-sample scan offsets 'N' in the input dataset. "
                "Regenerate dataset_robak_merged.npz from canonical inputs that preserve N."
            ),
        }

    if features_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError(f"{npz_path}: feature/label rows mismatch")
    if pose_raw is not None and pose_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError(f"{npz_path}: pose/label rows mismatch")

    n_before_mirror = int(features_raw.shape[0])
    if mirror_augment == "robak":
        features_raw, labels_raw, pose_raw = _mirror_augment_robak(features_raw, labels_raw, pose_raw)
        if noff_raw is not None:
            noff_raw = np.concatenate([noff_raw, noff_raw], axis=0)
    elif mirror_augment == "rywak":
        features_raw, labels_raw, pose_raw = _mirror_augment_rywak(features_raw, labels_raw, pose_raw)
        if noff_raw is not None:
            noff_raw = np.concatenate([noff_raw, noff_raw], axis=0)
    n_after_mirror = int(features_raw.shape[0])

    # Recompute metrics on the augmented data.
    if mirror_augment != "none":
        metric_row = np.concatenate([metric_row, metric_row], axis=0)
        metric_col = np.concatenate([metric_col, metric_col], axis=0)

    # Deduplication policy: always X+Y (pose key deliberately ignored).
    # Skip full dedup after mirror augmentation to avoid OOM on large datasets:
    # mirror never creates exact duplicates (scan beams are reversed).
    dedup_pose_requested = False
    dedup_pose_available = pose_raw is not None
    if mirror_augment != "none" and n_after_mirror > 2_000_000:
        # Fast path: skip costly dedup for large mirrored datasets
        features = features_raw
        labels = labels_raw
        pose = pose_raw
        noff = noff_raw
        removed_dups = 0
        unique_idx = np.arange(features_raw.shape[0], dtype=np.int64)
        dedup_key_used = "X+Y (skipped: mirror-augmented)"
    else:
        features, labels, pose, removed_dups, unique_idx, dedup_key_used = _deduplicate_xy(
            features_raw,
            labels_raw,
            pose_raw,
        )
        noff = noff_raw[unique_idx] if noff_raw is not None else None
    metric_row_dedup = np.asarray(metric_row, dtype=np.float64).reshape(-1)[unique_idx]
    metric_col_dedup = np.asarray(metric_col, dtype=np.float64).reshape(-1)[unique_idx]

    r_idx = _bin_indices(metric_row_dedup, bins, row_range[0], row_range[1])
    c_idx = _bin_indices(metric_col_dedup, bins, col_range[0], col_range[1])

    cell_counts, cell_indices = _build_cell_counts(r_idx, c_idx, bins)
    row_counts = np.sum(cell_counts, axis=1).astype(np.int64)
    col_counts = np.sum(cell_counts, axis=0).astype(np.int64)

    missing_rows = np.where(row_counts <= 0)[0].astype(np.int64).tolist()
    missing_cols = np.where(col_counts <= 0)[0].astype(np.int64).tolist()
    if require_all_bins and (missing_rows or missing_cols):
        return {
            "success": False,
            "reason": "missing_bins",
            "label": label,
            "input_path": str(npz_path),
            "output_path": str(out_path),
            "bins": int(bins),
            "n_input": int(n_before_mirror),
            "n_after_mirror": int(n_after_mirror),
            "mirror_augment": mirror_augment,
            "n_after_dedup": int(features.shape[0]),
            "duplicates_removed_before_balance": int(removed_dups),
            "dedup_key_used": dedup_key_used,
            "dedup_use_pose_key_requested": dedup_pose_requested,
            "dedup_use_pose_key_available": dedup_pose_available,
            "missing_row_bins": missing_rows,
            "missing_col_bins": missing_cols,
            "row_counts_before": row_counts.tolist(),
            "col_counts_before": col_counts.tolist(),
            "row_count_min_before": int(np.min(row_counts)),
            "row_count_max_before": int(np.max(row_counts)),
            "col_count_min_before": int(np.min(col_counts)),
            "col_count_max_before": int(np.max(col_counts)),
        }

    target_per_bin, selected_matrix = _max_feasible_target(cell_counts, row_counts, col_counts)
    if target_per_bin <= 0:
        return {
            "success": False,
            "reason": "no_feasible_target",
            "label": label,
            "input_path": str(npz_path),
            "output_path": str(out_path),
            "bins": int(bins),
            "n_input": int(n_before_mirror),
            "n_after_dedup": int(features.shape[0]),
            "duplicates_removed_before_balance": int(removed_dups),
            "dedup_key_used": dedup_key_used,
            "dedup_use_pose_key_requested": dedup_pose_requested,
            "dedup_use_pose_key_available": dedup_pose_available,
            "missing_row_bins": missing_rows,
            "missing_col_bins": missing_cols,
            "row_counts_before": row_counts.tolist(),
            "col_counts_before": col_counts.tolist(),
            "row_count_min_before": int(np.min(row_counts)),
            "row_count_max_before": int(np.max(row_counts)),
            "col_count_min_before": int(np.min(col_counts)),
            "col_count_max_before": int(np.max(col_counts)),
        }

    selected_idx = _select_indices_from_cells(cell_indices, selected_matrix, seed=seed)
    features_out = features[selected_idx]
    labels_out = labels[selected_idx]
    pose_out = pose[selected_idx] if pose is not None else None
    noff_out = noff[selected_idx] if noff is not None else None

    # Validate strict uniqueness after balancing.
    features_chk, labels_chk, pose_chk, removed_again, _, _ = _deduplicate_xy(
        features_out,
        labels_out,
        pose_out,
    )
    if removed_again != 0 or features_chk.shape[0] != features_out.shape[0]:
        raise RuntimeError(f"{label}: non-unique samples remained after balancing")
    if pose_out is not None and pose_chk is not None and pose_chk.shape[0] != pose_out.shape[0]:
        raise RuntimeError(f"{label}: pose key validation failed")

    # Validate equal histogram counts.
    r_sel = _bin_indices(metric_row_dedup[selected_idx], bins, row_range[0], row_range[1])
    c_sel = _bin_indices(metric_col_dedup[selected_idx], bins, col_range[0], col_range[1])
    row_after = np.bincount(r_sel, minlength=bins).astype(np.int64)
    col_after = np.bincount(c_sel, minlength=bins).astype(np.int64)
    if not (np.all(row_after == target_per_bin) and np.all(col_after == target_per_bin)):
        raise RuntimeError(f"{label}: equalization validation failed")

    meta = {
        "strict_unique_equalized": np.int64(1),
        "label": np.asarray([label], dtype=object),
        "source_path": np.asarray([str(npz_path)], dtype=object),
        "bins": np.int64(bins),
        "target_per_bin": np.int64(target_per_bin),
        "n_input": np.int64(features_raw.shape[0]),
        "n_after_dedup_before_balance": np.int64(features.shape[0]),
        "duplicates_removed_before_balance": np.int64(removed_dups),
        "n_output": np.int64(features_out.shape[0]),
        "dedup_key_used": np.asarray([dedup_key_used], dtype=object),
        "dedup_use_pose_key_requested": np.int64(1 if dedup_pose_requested else 0),
        "dedup_use_pose_key_available": np.int64(1 if dedup_pose_available else 0),
        "row_hist_min": np.float32(row_range[0]),
        "row_hist_max": np.float32(row_range[1]),
        "col_hist_min": np.float32(col_range[0]),
        "col_hist_max": np.float32(col_range[1]),
        "row_counts_after": row_after.astype(np.int64),
        "col_counts_after": col_after.astype(np.int64),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload: dict[str, object] = {feature_key: features_out, "Y": labels_out, "meta": meta}
    if pose_out is not None:
        out_payload["P"] = pose_out
    if noff_out is not None:
        out_payload["N"] = noff_out
    np.savez_compressed(out_path, **out_payload)

    return {
        "success": True,
        "reason": "ok",
        "label": label,
        "input_path": str(npz_path),
        "output_path": str(out_path),
        "bins": int(bins),
        "target_per_bin": int(target_per_bin),
        "n_input": int(n_before_mirror),
        "n_after_mirror": int(n_after_mirror),
        "mirror_augment": mirror_augment,
        "n_after_dedup_before_balance": int(features.shape[0]),
        "duplicates_removed_before_balance": int(removed_dups),
        "n_output": int(features_out.shape[0]),
        "dedup_key_used": dedup_key_used,
        "dedup_use_pose_key_requested": dedup_pose_requested,
        "dedup_use_pose_key_available": dedup_pose_available,
        "missing_row_bins": missing_rows,
        "missing_col_bins": missing_cols,
        "row_counts_before": row_counts.tolist(),
        "col_counts_before": col_counts.tolist(),
        "row_count_min_before": int(np.min(row_counts)),
        "row_count_max_before": int(np.max(row_counts)),
        "col_count_min_before": int(np.min(col_counts)),
        "col_count_max_before": int(np.max(col_counts)),
        "row_counts_after": row_after.tolist(),
        "col_counts_after": col_after.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strict histogram rebalance with unique samples (no replacement), for merged Rywak/Robak datasets."
    )
    ap.add_argument("--rywak-npz", type=Path, help="Input dataset_rywak_merged.npz")
    ap.add_argument("--robak-npz", type=Path, help="Input dataset_robak_merged.npz")
    ap.add_argument("--rywak-out", type=Path, help="Output strict-balanced Rywak npz")
    ap.add_argument("--robak-out", type=Path, help="Output strict-balanced Robak npz")
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--require-all-bins", action="store_true", default=False)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--rywak-v-min", type=float, default=0.0)
    ap.add_argument("--rywak-v-max", type=float, default=1.2)
    ap.add_argument("--rywak-w-min", type=float, default=0.0)
    ap.add_argument("--rywak-w-max", type=float, default=3.0)
    ap.add_argument("--robak-t-min", type=float, default=0.0)
    ap.add_argument("--robak-t-max", type=float, default=1.0)
    ap.add_argument("--robak-r-min", type=float, default=0.0)
    ap.add_argument("--robak-r-max", type=float, default=180.0)
    ap.add_argument(
        "--dedup-use-pose-key",
        action="store_true",
        help="DEPRECATED (ignored). Deduplication is always performed on X+Y only.",
    )
    ap.add_argument(
        "--mirror-augment",
        action="store_true",
        default=False,
        help="Apply mirror augmentation inline before rebalancing (doubles unique samples).",
    )
    args = ap.parse_args()

    if args.bins < 1:
        raise SystemExit("--bins must be >= 1")

    summary: dict[str, object] = {
        "bins": int(args.bins),
        "seed": int(args.seed),
        "require_all_bins": bool(args.require_all_bins),
    }
    failed = False

    if args.rywak_npz is not None:
        if args.rywak_out is None:
            raise SystemExit("--rywak-out is required when --rywak-npz is provided")
        with np.load(args.rywak_npz, allow_pickle=True) as d:
            y = np.asarray(d["Y"], dtype=np.float32)
        (
            ry_metric_row,
            ry_metric_col,
            ry_metric_mode,
            ry_row_unit,
            ry_col_unit,
        ) = _rywak_metrics(y)
        ry_label = "rywak_translation_rotation" if ry_metric_mode == "translation_rotation" else "rywak_v_w"
        ry = _rebalance_one(
            args.rywak_npz,
            args.rywak_out,
            feature_key="X",
            bins=int(args.bins),
            metric_row=ry_metric_row,
            metric_col=ry_metric_col,
            row_range=(float(args.rywak_v_min), float(args.rywak_v_max)),
            col_range=(float(args.rywak_w_min), float(args.rywak_w_max)),
            seed=int(args.seed) + 17,
            require_all_bins=bool(args.require_all_bins),
            label=ry_label,
            mirror_augment="rywak" if args.mirror_augment else "none",
        )
        ry["metric_mode"] = ry_metric_mode
        ry["row_metric_unit"] = ry_row_unit
        ry["col_metric_unit"] = ry_col_unit
        summary["rywak"] = ry
        failed = failed or (not bool(ry.get("success", False)))

    if args.robak_npz is not None:
        if args.robak_out is None:
            raise SystemExit("--robak-out is required when --robak-npz is provided")
        with np.load(args.robak_npz, allow_pickle=True) as d:
            y = np.asarray(d["Y"], dtype=np.float32)
        rb_metric_t = np.linalg.norm(y[:, :2], axis=1).astype(np.float32)
        rb_metric_r = np.abs(np.rad2deg(y[:, 2])).astype(np.float32)
        rb = _rebalance_one(
            args.robak_npz,
            args.robak_out,
            feature_key="X_pairs",
            bins=int(args.bins),
            metric_row=rb_metric_t,
            metric_col=rb_metric_r,
            row_range=(float(args.robak_t_min), float(args.robak_t_max)),
            col_range=(float(args.robak_r_min), float(args.robak_r_max)),
            seed=int(args.seed) + 29,
            require_all_bins=bool(args.require_all_bins),
            label="robak_translation_rotation",
            mirror_augment="robak" if args.mirror_augment else "none",
            require_offsets=True,
        )
        summary["robak"] = rb
        failed = failed or (not bool(rb.get("success", False)))

    if "rywak" not in summary and "robak" not in summary:
        raise SystemExit("At least one of --rywak-npz / --robak-npz must be provided")

    summary["all_success"] = bool(not failed)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] Summary written: {args.summary_json}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
