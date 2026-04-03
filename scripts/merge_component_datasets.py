#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz_arrays(path: Path, feature_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as data:
        if feature_key not in data or "Y" not in data:
            raise KeyError(f"{path}: expected keys {feature_key!r} and 'Y'")
        x = np.asarray(data[feature_key], dtype=np.float32)
        y = np.asarray(data["Y"], dtype=np.float32)
        p = np.asarray(data["P"], dtype=np.float32) if "P" in data else None
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{path}: mismatched rows {x.shape[0]} vs {y.shape[0]}")
    if p is not None and p.shape[0] != y.shape[0]:
        raise ValueError(f"{path}: mismatched P rows {p.shape[0]} vs {y.shape[0]}")
    return x, y, p


def _deduplicate(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray | None,
    *,
    use_pose_key: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, str]:
    if x.shape[0] == 0:
        return x, y, p, 0, "X+Y+P" if use_pose_key and p is not None else "X+Y"
    flat_x = np.ascontiguousarray(x.reshape((x.shape[0], -1)))
    flat_y = np.ascontiguousarray(y.reshape((y.shape[0], -1)))
    if use_pose_key and p is not None:
        flat_p = np.ascontiguousarray(p.reshape((p.shape[0], -1)))
        merged = np.concatenate([flat_x, flat_y, flat_p], axis=1)
        dedup_key = "X+Y+P"
    else:
        merged = np.concatenate([flat_x, flat_y], axis=1)
        dedup_key = "X+Y"
    merged_view = np.ascontiguousarray(merged).view(
        np.dtype((np.void, merged.dtype.itemsize * merged.shape[1]))
    )
    _, unique_idx = np.unique(merged_view, return_index=True)
    unique_idx = np.sort(unique_idx.astype(np.int64))
    removed = int(x.shape[0] - unique_idx.size)
    p_out = p[unique_idx] if p is not None else None
    return x[unique_idx], y[unique_idx], p_out, removed, dedup_key


def _merge_group(
    inputs: list[Path],
    *,
    feature_key: str,
    out_path: Path,
    deduplicate: bool,
    dedup_use_pose_key: bool,
    group_name: str,
) -> dict:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    pose_sources = 0
    for p in inputs:
        x, y, pose = _load_npz_arrays(p, feature_key)
        xs.append(x)
        ys.append(y)
        if pose is not None:
            pose_sources += 1
            ps.append(pose)
    x_cat = np.concatenate(xs, axis=0) if xs else np.zeros((0,), dtype=np.float32)
    y_cat = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    p_cat = np.concatenate(ps, axis=0) if pose_sources == len(inputs) and ps else None
    n_before = int(y_cat.shape[0]) if y_cat.ndim >= 1 else 0
    removed = 0
    dedup_key_used = "X+Y"
    pose_key_requested = bool(dedup_use_pose_key)
    pose_key_available = p_cat is not None
    pose_key_fallback_reason = ""
    if deduplicate and n_before > 0:
        if pose_key_requested and not pose_key_available:
            if pose_sources == 0:
                pose_key_fallback_reason = "pose_key_not_present_in_sources"
            else:
                pose_key_fallback_reason = "pose_key_missing_in_some_sources"
        x_cat, y_cat, p_cat, removed, dedup_key_used = _deduplicate(
            x_cat, y_cat, p_cat, use_pose_key=pose_key_requested and pose_key_available
        )
    n_after = int(y_cat.shape[0]) if y_cat.ndim >= 1 else 0

    meta = {
        "merged_group": np.asarray([group_name], dtype=object),
        "source_paths": np.asarray([str(p) for p in inputs], dtype=object),
        "source_count": np.int64(len(inputs)),
        "n_before_dedup": np.int64(n_before),
        "n_after_dedup": np.int64(n_after),
        "deduplicate_enabled": np.int64(1 if deduplicate else 0),
        "dedup_key_used": np.asarray([dedup_key_used], dtype=object),
        "dedup_use_pose_key_requested": np.int64(1 if pose_key_requested else 0),
        "dedup_use_pose_key_available": np.int64(1 if pose_key_available else 0),
        "pose_key_sources_count": np.int64(pose_sources),
        "duplicates_removed": np.int64(removed),
    }
    if pose_key_fallback_reason:
        meta["dedup_pose_key_fallback_reason"] = np.asarray([pose_key_fallback_reason], dtype=object)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload: dict[str, object] = {feature_key: x_cat, "Y": y_cat, "meta": meta}
    if p_cat is not None:
        out_payload["P"] = p_cat
    np.savez_compressed(out_path, **out_payload)
    return {
        "group": group_name,
        "feature_key": feature_key,
        "out_path": str(out_path),
        "source_count": len(inputs),
        "pose_key_sources_count": pose_sources,
        "pose_key_in_merged_dataset": bool(p_cat is not None),
        "n_before_dedup": n_before,
        "n_after_dedup": n_after,
        "dedup_key_used": dedup_key_used,
        "dedup_use_pose_key_requested": pose_key_requested,
        "dedup_pose_key_fallback_reason": pose_key_fallback_reason,
        "duplicates_removed": removed,
    }


def _paths(values: list[str]) -> list[Path]:
    out: list[Path] = []
    for v in values:
        p = Path(v).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing file: {p}")
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge component-balanced datasets into training-ready merged datasets "
            "(Rywak: X/Y, Robak: X_pairs/Y)."
        )
    )
    ap.add_argument("--out-dir", required=True, help="Output directory, e.g. out/exp_xxx")
    ap.add_argument(
        "--rywak-linear",
        nargs="*",
        default=[],
        help="Paths to dataset_rywak_linear_balanced.npz (one or many experiments).",
    )
    ap.add_argument(
        "--rywak-angular",
        nargs="*",
        default=[],
        help="Paths to dataset_rywak_angular_balanced.npz (one or many experiments).",
    )
    ap.add_argument(
        "--robak-translation",
        nargs="*",
        default=[],
        help="Paths to dataset_robak_translation_balanced.npz (one or many experiments).",
    )
    ap.add_argument(
        "--robak-rotation",
        nargs="*",
        default=[],
        help="Paths to dataset_robak_rotation_balanced.npz (one or many experiments).",
    )
    ap.add_argument("--rywak-out-name", default="dataset_rywak_merged.npz")
    ap.add_argument("--robak-out-name", default="dataset_robak_merged.npz")
    ap.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove exact duplicate (X,Y) rows after concatenation.",
    )
    ap.add_argument(
        "--dedup-use-pose-key",
        action="store_true",
        help="When deduplicating, use X+Y+P key if P (pose_prev/curr) exists in all input sources.",
    )
    ap.add_argument(
        "--summary-name",
        default="merge_component_datasets_summary.json",
        help="Summary JSON file name in --out-dir.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    rywak_inputs = _paths(args.rywak_linear) + _paths(args.rywak_angular)
    if rywak_inputs:
        summary["rywak"] = _merge_group(
            rywak_inputs,
            feature_key="X",
            out_path=out_dir / args.rywak_out_name,
            deduplicate=bool(args.deduplicate),
            dedup_use_pose_key=bool(args.dedup_use_pose_key),
            group_name="rywak_component_merge",
        )

    robak_inputs = _paths(args.robak_translation) + _paths(args.robak_rotation)
    if robak_inputs:
        summary["robak"] = _merge_group(
            robak_inputs,
            feature_key="X_pairs",
            out_path=out_dir / args.robak_out_name,
            deduplicate=bool(args.deduplicate),
            dedup_use_pose_key=bool(args.dedup_use_pose_key),
            group_name="robak_component_merge",
        )

    if not summary:
        raise SystemExit("No input component datasets provided.")

    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
