import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from typing import Any
import numpy as np

CUBLAS_WORKSPACE_CONFIG_DEFAULT = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_DEFAULT)


@dataclass(frozen=True)
class TorchDeviceInfo:
    requested: str
    resolved: str
    reason: str
    warning: str | None = None


def seed_all(seed: int, *, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = not bool(deterministic)
                torch.backends.cudnn.allow_tf32 = not bool(deterministic)
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high" if not bool(deterministic) else "highest")
            except Exception:
                pass
        try:
            torch.use_deterministic_algorithms(bool(deterministic))
        except Exception:
            pass
    except Exception:
        pass


def select_torch_device(requested: str = "auto") -> TorchDeviceInfo:
    req = str(requested or "auto").strip().lower()
    if req in {"gpu"}:
        req = "cuda"

    try:
        import torch
    except Exception as exc:
        warning = f"PyTorch is unavailable, falling back to CPU: {exc}"
        return TorchDeviceInfo(requested=req, resolved="cpu", reason="PyTorch unavailable", warning=warning)

    if req == "cpu":
        return TorchDeviceInfo(requested=req, resolved="cpu", reason="CPU requested explicitly")

    candidate = "cuda:0"
    if req not in {"auto", "cuda"} and not req.startswith("cuda:"):
        warning = f"Unknown torch_device='{requested}', using auto device selection."
        req = "auto"
    else:
        warning = None
        if req.startswith("cuda:"):
            candidate = req

    if not torch.cuda.is_available():
        fallback_warning = "CUDA is not available in the current PyTorch runtime."
        combined_warning = fallback_warning if warning is None else f"{warning} {fallback_warning}"
        return TorchDeviceInfo(
            requested=req,
            resolved="cpu",
            reason="CUDA unavailable, falling back to CPU",
            warning=combined_warning,
        )

    try:
        dev = torch.device(candidate)
        probe = torch.zeros(1, device=dev, dtype=torch.float32)
        _ = probe + 1.0
        torch.cuda.synchronize(dev)
        gpu_name = torch.cuda.get_device_name(dev)
        return TorchDeviceInfo(
            requested=req,
            resolved=str(dev),
            reason=f"Using CUDA device '{gpu_name}'",
            warning=warning,
        )
    except Exception as exc:
        fallback_warning = f"CUDA probe failed for '{candidate}', falling back to CPU: {exc}"
        combined_warning = fallback_warning if warning is None else f"{warning} {fallback_warning}"
        return TorchDeviceInfo(
            requested=req,
            resolved="cpu",
            reason="CUDA unusable, falling back to CPU",
            warning=combined_warning,
        )


def torch_state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in state_dict.items()
    }


def synchronize_torch_device(device) -> None:
    try:
        import torch
    except Exception:
        return

    try:
        dev = torch.device(device)
    except Exception:
        return

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        ensure_dir(directory)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def wait_for_npz_dataset(
    path: str,
    timeout_sec: float,
    *,
    poll_interval_sec: float = 0.5,
    required_keys: tuple[str, ...] | None = None,
) -> tuple[bool, Exception | None]:
    timeout = max(0.0, float(timeout_sec))
    deadline = time.time() + timeout
    last_error: Exception | None = None

    # Ensure at least one validation attempt even when timeout_sec == 0.
    first_attempt = True
    while first_attempt or time.time() < deadline:
        first_attempt = False
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with np.load(path, allow_pickle=True) as data:
                    keys = tuple(required_keys or ())
                    if keys:
                        available = set(data.keys())
                        missing = [key for key in keys if key not in available]
                        if missing:
                            raise KeyError(f"Missing required npz keys: {missing}")
                        for key in keys:
                            value = data[key]
                            if isinstance(value, np.ndarray) and value.size == 0:
                                raise ValueError(f"Empty required array in npz key: {key}")
                return True, None
            except (EOFError, ValueError, OSError, zipfile.BadZipFile, KeyError) as exc:
                last_error = exc
        if time.time() < deadline:
            time.sleep(max(0.05, float(poll_interval_sec)))

    return False, last_error


def normalize_split_strategy(value: str | None) -> str:
    strategy = str(value or "").strip().lower()
    if strategy in {"random", "shuffle", "random_shuffle"}:
        return "random_shuffle"
    return "tail_holdout_no_shuffle"


def split_train_val_indices(
    n_samples: int,
    val_ratio: float,
    *,
    seed: int,
    split_strategy: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    n = int(n_samples)
    if n < 2:
        raise ValueError(f"Need at least 2 samples for train/val split, got {n}")
    n_val = int(max(1, min(n - 1, round(float(val_ratio) * n))))
    strategy = normalize_split_strategy(split_strategy)

    if strategy == "random_shuffle":
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        val_idx = np.sort(perm[:n_val].astype(np.int64))
        train_idx = np.sort(perm[n_val:].astype(np.int64))
        return train_idx, val_idx, strategy

    split_idx = n - n_val
    train_idx = np.arange(0, split_idx, dtype=np.int64)
    val_idx = np.arange(split_idx, n, dtype=np.int64)
    return train_idx, val_idx, strategy


def split_time_coverage_stats(
    n_samples: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    split_strategy: str,
) -> dict[str, Any]:
    n = max(1, int(n_samples))

    def _window(idx: np.ndarray) -> tuple[float, float]:
        if idx.size == 0 or n <= 1:
            return 0.0, 0.0
        return float(idx.min()) / float(n - 1), float(idx.max()) / float(n - 1)

    train_start, train_end = _window(train_idx)
    val_start, val_end = _window(val_idx)
    return {
        "strategy": normalize_split_strategy(split_strategy),
        "n_total": int(n_samples),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "train_time_window_start_ratio": train_start,
        "train_time_window_end_ratio": train_end,
        "val_time_window_start_ratio": val_start,
        "val_time_window_end_ratio": val_end,
        "train_time_fraction_estimate": float(train_idx.size) / float(n),
        "val_time_fraction_estimate": float(val_idx.size) / float(n),
    }


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def parse_filter_mode(value: str, default: str = "any") -> str:
    mode = str(value).strip().lower()
    if mode in {"all", "and"}:
        return "all"
    if mode in {"any", "or"}:
        return "any"
    return default


@dataclass(frozen=True)
class PoseDelta:
    dx: float
    dy: float
    dtheta: float
    distance: float


def pose_delta(prev_xyth, curr_xyth) -> PoseDelta:
    dx = float(curr_xyth[0]) - float(prev_xyth[0])
    dy = float(curr_xyth[1]) - float(prev_xyth[1])
    dtheta = wrap(float(curr_xyth[2]) - float(prev_xyth[2]))
    return PoseDelta(
        dx=dx,
        dy=dy,
        dtheta=dtheta,
        distance=float(math.hypot(dx, dy)),
    )


def pose_delta_local(prev_xyth, curr_xyth) -> PoseDelta:
    delta_world = pose_delta(prev_xyth, curr_xyth)
    th0 = float(prev_xyth[2])
    c = math.cos(th0)
    s = math.sin(th0)
    dx_local = c * delta_world.dx + s * delta_world.dy
    dy_local = -s * delta_world.dx + c * delta_world.dy
    return PoseDelta(
        dx=float(dx_local),
        dy=float(dy_local),
        dtheta=float(delta_world.dtheta),
        distance=float(math.hypot(dx_local, dy_local)),
    )


def pose_pairs_to_local_deltas(pose_pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pose_pairs, dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[1] < 6:
        raise ValueError(f"Expected pose_pairs shape (N,6+), got {pairs.shape}")

    prev_xy = pairs[:, 0:2]
    prev_th = pairs[:, 2]
    curr_xy = pairs[:, 3:5]
    curr_th = pairs[:, 5]

    delta_world = curr_xy - prev_xy
    c = np.cos(prev_th)
    s = np.sin(prev_th)
    dx_local = c * delta_world[:, 0] + s * delta_world[:, 1]
    dy_local = -s * delta_world[:, 0] + c * delta_world[:, 1]
    dtheta = ((curr_th - prev_th + np.pi) % (2.0 * np.pi)) - np.pi

    return np.stack([dx_local, dy_local, dtheta], axis=1).astype(np.float32)


def estimate_dt_from_pose_pairs_and_velocity_labels(
    pose_pairs: np.ndarray,
    velocity_labels: np.ndarray,
    *,
    min_linear_speed: float = 0.03,
    min_angular_speed: float = 0.05,
    clip_min: float = 1e-3,
    clip_max: float = 1.0,
    fallback_dt: float | None = None,
) -> np.ndarray:
    deltas = pose_pairs_to_local_deltas(pose_pairs)
    labels = np.asarray(velocity_labels, dtype=np.float32)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f"Expected velocity_labels shape (N,2+), got {labels.shape}")

    trans = np.linalg.norm(deltas[:, :2], axis=1)
    rot = np.abs(deltas[:, 2])
    v_abs = np.abs(labels[:, 0])
    w_abs = np.abs(labels[:, 1])

    dt_v = np.full(trans.shape, np.nan, dtype=np.float32)
    dt_w = np.full(rot.shape, np.nan, dtype=np.float32)
    v_mask = v_abs >= float(min_linear_speed)
    w_mask = w_abs >= float(min_angular_speed)
    dt_v[v_mask] = trans[v_mask] / np.maximum(v_abs[v_mask], 1e-6)
    dt_w[w_mask] = rot[w_mask] / np.maximum(w_abs[w_mask], 1e-6)

    dt = np.nanmean(np.stack([dt_v, dt_w], axis=1), axis=1).astype(np.float32)
    valid = np.isfinite(dt) & (dt >= float(clip_min)) & (dt <= float(clip_max))
    if fallback_dt is None:
        if np.any(valid):
            fallback_dt = float(np.median(dt[valid]))
        else:
            fallback_dt = 0.1
    dt = np.where(valid, dt, float(fallback_dt)).astype(np.float32)
    return np.clip(dt, float(clip_min), float(clip_max)).astype(np.float32)


def velocity_predictions_to_local_deltas(
    predicted_velocities: np.ndarray,
    dt_estimates: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(predicted_velocities, dtype=np.float32)
    dt = np.asarray(dt_estimates, dtype=np.float32).reshape(-1)
    if pred.ndim != 2 or pred.shape[1] < 2:
        raise ValueError(f"Expected predicted_velocities shape (N,2+), got {pred.shape}")
    if dt.shape[0] != pred.shape[0]:
        raise ValueError(f"dt_estimates length {dt.shape[0]} does not match predictions {pred.shape[0]}")

    dx = pred[:, 0] * dt
    dy = np.zeros_like(dx, dtype=np.float32)
    dtheta = pred[:, 1] * dt
    return np.stack([dx, dy, dtheta], axis=1).astype(np.float32)


def compose_local_pose_sequence(start_pose, local_deltas: np.ndarray) -> tuple[float, float, float]:
    x = float(start_pose[0])
    y = float(start_pose[1])
    th = float(start_pose[2])
    for dx, dy, dtheta in np.asarray(local_deltas, dtype=np.float32):
        c = math.cos(th)
        s = math.sin(th)
        x += c * float(dx) - s * float(dy)
        y += s * float(dx) + c * float(dy)
        th = wrap(th + float(dtheta))
    return float(x), float(y), float(th)


def split_pose_pair_sequences(
    pose_pairs: np.ndarray,
    *,
    continuity_pos_tol_m: float = 1e-3,
    continuity_yaw_tol_rad: float = 1e-3,
    min_segment_len: int = 2,
) -> list[tuple[int, int]]:
    pairs = np.asarray(pose_pairs, dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[1] < 6:
        return []
    n = int(pairs.shape[0])
    if n < max(2, int(min_segment_len)):
        return []

    segments: list[tuple[int, int]] = []
    start = 0
    for idx in range(n - 1):
        curr_pose = pairs[idx, 3:6]
        next_prev_pose = pairs[idx + 1, 0:3]
        pos_ok = float(np.linalg.norm(curr_pose[:2] - next_prev_pose[:2])) <= float(continuity_pos_tol_m)
        yaw_ok = abs(wrap(float(curr_pose[2]) - float(next_prev_pose[2]))) <= float(continuity_yaw_tol_rad)
        if pos_ok and yaw_ok:
            continue
        if (idx + 1 - start) >= int(min_segment_len):
            segments.append((start, idx + 1))
        start = idx + 1
    if (n - start) >= int(min_segment_len):
        segments.append((start, n))
    return segments


def collect_rollout_window_starts(
    pose_pairs: np.ndarray,
    horizon: int,
    *,
    continuity_pos_tol_m: float = 1e-3,
    continuity_yaw_tol_rad: float = 1e-3,
) -> np.ndarray:
    pairs = np.asarray(pose_pairs, dtype=np.float32)
    horizon_int = int(horizon)
    if pairs.ndim != 2 or pairs.shape[1] < 6 or horizon_int < 1:
        return np.zeros((0,), dtype=np.int64)

    starts: list[int] = []
    segments = split_pose_pair_sequences(
        pairs,
        continuity_pos_tol_m=continuity_pos_tol_m,
        continuity_yaw_tol_rad=continuity_yaw_tol_rad,
        min_segment_len=max(2, horizon_int),
    )
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < horizon_int:
            continue
        starts.extend(range(seg_start, seg_end - horizon_int + 1))
    return np.asarray(starts, dtype=np.int64)


def compute_rollout_metrics_from_local_deltas(
    predicted_local_deltas: np.ndarray,
    pose_pairs: np.ndarray,
    *,
    horizons: list[int] | tuple[int, ...] = (5, 10, 20),
    continuity_pos_tol_m: float = 1e-3,
    continuity_yaw_tol_rad: float = 1e-3,
) -> dict[str, Any]:
    pred = np.asarray(predicted_local_deltas, dtype=np.float32)
    pairs = np.asarray(pose_pairs, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[0] != pairs.shape[0] or pred.shape[1] < 3 or pairs.ndim != 2 or pairs.shape[1] < 6:
        return {"n_rollout_segments": 0}

    segments = split_pose_pair_sequences(
        pairs,
        continuity_pos_tol_m=continuity_pos_tol_m,
        continuity_yaw_tol_rad=continuity_yaw_tol_rad,
        min_segment_len=2,
    )
    metrics: dict[str, Any] = {"n_rollout_segments": int(len(segments))}
    unique_horizons = sorted({int(h) for h in horizons if int(h) >= 1})
    for horizon in unique_horizons:
        xy_err_sq: list[float] = []
        yaw_err_sq: list[float] = []
        n_windows = 0
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < horizon:
                continue
            for win_start in range(seg_start, seg_end - horizon + 1):
                gt_start = pairs[win_start, 0:3]
                gt_end = pairs[win_start + horizon - 1, 3:6]
                pred_end = compose_local_pose_sequence(gt_start, pred[win_start : win_start + horizon, :3])
                dx = float(pred_end[0]) - float(gt_end[0])
                dy = float(pred_end[1]) - float(gt_end[1])
                dth = wrap(float(pred_end[2]) - float(gt_end[2]))
                xy_err_sq.append(dx * dx + dy * dy)
                yaw_err_sq.append(dth * dth)
                n_windows += 1

        metrics[f"n_rollout_windows_h{horizon}"] = int(n_windows)
        if n_windows > 0:
            metrics[f"rollout_xy_rmse_h{horizon}"] = float(math.sqrt(float(np.mean(xy_err_sq))))
            metrics[f"rollout_yaw_rmse_h{horizon}"] = float(math.sqrt(float(np.mean(yaw_err_sq))))
        else:
            metrics[f"rollout_xy_rmse_h{horizon}"] = None
            metrics[f"rollout_yaw_rmse_h{horizon}"] = None
    return metrics


def velocity_label_from_poses(
    prev_xyth,
    curr_xyth,
    dt_sec: float,
    *,
    frame: str = "local",
) -> tuple[float, float]:
    dt = max(float(dt_sec), 1e-6)
    frame_mode = str(frame).strip().lower()
    if frame_mode == "world":
        delta = pose_delta(prev_xyth, curr_xyth)
        dominant = delta.dx if abs(delta.dx) >= abs(delta.dy) else delta.dy
        linear = math.copysign(delta.distance, dominant) if abs(dominant) > 1e-9 else 0.0
    else:
        delta = pose_delta_local(prev_xyth, curr_xyth)
        linear = float(delta.dx)
    angular = float(delta.dtheta)
    return float(linear / dt), float(angular / dt)


def scan_delta_rms(scan_prev: np.ndarray, scan_curr: np.ndarray) -> float:
    if scan_prev is None or scan_curr is None:
        return 0.0
    delta = np.asarray(scan_curr, dtype=np.float32) - np.asarray(scan_prev, dtype=np.float32)
    if delta.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(delta * delta)))


def scan_motion_confidence(
    scan_prev: np.ndarray | None,
    scan_curr: np.ndarray | None,
    *,
    low_rms: float = 0.02,
    high_rms: float = 0.20,
) -> float:
    rms = scan_delta_rms(scan_prev, scan_curr)
    lo = float(low_rms)
    hi = float(high_rms)
    if hi <= lo:
        return 1.0 if rms >= hi else 0.0
    return float(np.clip((rms - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def passes_motion_filter(
    prev_xyth,
    curr_xyth,
    *,
    dt_sec: float | None = None,
    min_translation: float = 0.0,
    min_rotation: float = 0.0,
    min_time_gap_sec: float = 0.0,
    min_scan_delta_rms: float = 0.0,
    scan_delta_rms_value: float | None = None,
    mode: str = "any",
) -> tuple[bool, PoseDelta]:
    delta = pose_delta(prev_xyth, curr_xyth)
    checks = []

    if min_translation > 0.0:
        checks.append(delta.distance >= float(min_translation))
    if min_rotation > 0.0:
        checks.append(abs(delta.dtheta) >= float(min_rotation))
    if min_time_gap_sec > 0.0:
        checks.append(dt_sec is not None and float(dt_sec) >= float(min_time_gap_sec))
    if min_scan_delta_rms > 0.0:
        checks.append(
            scan_delta_rms_value is not None
            and float(scan_delta_rms_value) >= float(min_scan_delta_rms)
        )

    if not checks:
        return True, delta

    filt_mode = parse_filter_mode(mode)
    return (all(checks) if filt_mode == "all" else any(checks)), delta


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return (0.0, 0.0, qz, qw)


def xytheta_from_odom(odom_msg):
    x = float(odom_msg.pose.pose.position.x)
    y = float(odom_msg.pose.pose.position.y)
    th = float(yaw_from_quat(odom_msg.pose.pose.orientation))
    return x, y, th


def xytheta_from_pose_stamped(ps):
    x = float(ps.pose.position.x)
    y = float(ps.pose.position.y)
    th = float(yaw_from_quat(ps.pose.orientation))
    return x, y, th


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def uniform_histogram_sample_indices(
    values: np.ndarray,
    *,
    bins: int,
    seed: int,
    use_abs: bool,
    hist_min: float | None = None,
    hist_max: float | None = None,
    target_quantile: float = 0.35,
    target_min_per_bin: int = 8,
    upsample: bool = True,
) -> tuple[np.ndarray, dict]:
    """Select indices to flatten histogram counts without collapsing to sparse tail bins.

    Strategy:
    - optional fixed histogram range (hist_min..hist_max) to ignore out-of-range tails,
    - target count from quantile of non-empty bin counts (instead of strict minimum),
    - optional upsampling with replacement for underrepresented bins.
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    bins_eff = max(1, int(bins))
    quantile_eff = float(np.clip(float(target_quantile), 0.0, 1.0))
    min_target_eff = max(1, int(target_min_per_bin))
    upsample_eff = bool(upsample)
    stats = {
        "n_total": int(arr.size),
        "bins_requested": int(bins_eff),
        "use_abs": bool(use_abs),
        "value_min": None,
        "value_max": None,
        "n_finite": 0,
        "hist_min": None,
        "hist_max": None,
        "n_in_range": 0,
        "n_out_of_range": 0,
        "bins_non_empty": 0,
        "target_per_bin": 0,
        "target_quantile": quantile_eff,
        "target_min_per_bin": min_target_eff,
        "upsample": upsample_eff,
        "n_selected": 0,
        "counts_per_bin": np.zeros((bins_eff,), dtype=np.int64),
        "selected_counts_per_bin": np.zeros((bins_eff,), dtype=np.int64),
    }
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64), stats

    if use_abs:
        arr = np.abs(arr).astype(np.float32)

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros((0,), dtype=np.int64), stats

    valid_idx = np.nonzero(finite_mask)[0].astype(np.int64)
    valid_values = arr[finite_mask]
    stats["n_finite"] = int(valid_values.size)
    value_min = float(np.min(valid_values))
    value_max = float(np.max(valid_values))
    stats["value_min"] = value_min
    stats["value_max"] = value_max

    def _finite_float_or_none(raw: float | None) -> float | None:
        if raw is None:
            return None
        try:
            parsed = float(raw)
        except Exception:
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    hist_min_eff = _finite_float_or_none(hist_min)
    hist_max_eff = _finite_float_or_none(hist_max)
    if hist_min_eff is None:
        hist_min_eff = value_min
    if hist_max_eff is None:
        hist_max_eff = value_max
    if hist_max_eff < hist_min_eff:
        hist_min_eff, hist_max_eff = hist_max_eff, hist_min_eff
    if float(hist_max_eff - hist_min_eff) < 1e-9:
        hist_min_eff = value_min
        hist_max_eff = value_max

    stats["hist_min"] = float(hist_min_eff)
    stats["hist_max"] = float(hist_max_eff)

    in_range_mask = (valid_values >= float(hist_min_eff)) & (valid_values <= float(hist_max_eff))
    in_range_idx = valid_idx[in_range_mask].astype(np.int64)
    in_range_values = valid_values[in_range_mask]
    stats["n_in_range"] = int(in_range_values.size)
    stats["n_out_of_range"] = int(valid_values.size - in_range_values.size)
    if in_range_values.size == 0:
        return np.zeros((0,), dtype=np.int64), stats

    hist_span = float(hist_max_eff - hist_min_eff)
    if bins_eff <= 1 or hist_span < 1e-9:
        stats["bins_non_empty"] = 1
        stats["target_per_bin"] = int(in_range_values.size)
        stats["n_selected"] = int(in_range_idx.size)
        one_bin_counts = np.zeros((bins_eff,), dtype=np.int64)
        one_bin_counts[0] = int(in_range_values.size)
        stats["counts_per_bin"] = one_bin_counts
        stats["selected_counts_per_bin"] = one_bin_counts.copy()
        return in_range_idx, stats

    edges = np.linspace(
        float(hist_min_eff),
        float(hist_max_eff),
        bins_eff + 1,
        dtype=np.float32,
    )
    bin_ids = np.searchsorted(edges, in_range_values, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, bins_eff - 1).astype(np.int64)
    counts = np.bincount(bin_ids, minlength=bins_eff).astype(np.int64)
    non_empty_bins = np.nonzero(counts > 0)[0].astype(np.int64)
    stats["counts_per_bin"] = counts
    stats["bins_non_empty"] = int(non_empty_bins.size)
    if non_empty_bins.size == 0:
        return np.zeros((0,), dtype=np.int64), stats

    non_empty_counts = counts[non_empty_bins].astype(np.int64)
    target_per_bin = int(np.floor(np.quantile(non_empty_counts, quantile_eff)))
    target_per_bin = max(1, target_per_bin)
    target_per_bin = max(min_target_eff, target_per_bin)
    if not upsample_eff:
        target_per_bin = min(target_per_bin, int(np.min(non_empty_counts)))
    stats["target_per_bin"] = max(0, target_per_bin)
    if target_per_bin <= 0:
        return np.zeros((0,), dtype=np.int64), stats

    rng = np.random.default_rng(int(seed))
    selected_chunks = []
    selected_counts = np.zeros((bins_eff,), dtype=np.int64)
    for bin_idx in non_empty_bins:
        bucket = in_range_idx[bin_ids == int(bin_idx)]
        if bucket.size == 0:
            continue
        if bucket.size > target_per_bin:
            picked = rng.choice(bucket, size=target_per_bin, replace=False)
            chosen = np.sort(picked.astype(np.int64))
            selected_chunks.append(chosen)
            selected_counts[int(bin_idx)] = int(chosen.size)
            continue
        if bucket.size < target_per_bin and upsample_eff:
            extra = rng.choice(bucket, size=(target_per_bin - bucket.size), replace=True).astype(np.int64)
            chosen = np.sort(np.concatenate([bucket.astype(np.int64), extra], axis=0))
            selected_chunks.append(chosen)
            selected_counts[int(bin_idx)] = int(chosen.size)
            continue
        chosen = np.sort(bucket.astype(np.int64))
        selected_chunks.append(chosen)
        selected_counts[int(bin_idx)] = int(chosen.size)

    if not selected_chunks:
        return np.zeros((0,), dtype=np.int64), stats

    selected = np.sort(np.concatenate(selected_chunks, axis=0).astype(np.int64))
    stats["selected_counts_per_bin"] = selected_counts
    stats["n_selected"] = int(selected.size)
    return selected, stats


def merge_balanced_indices(
    idx_primary: np.ndarray,
    idx_secondary: np.ndarray,
    *,
    strategy: str = "union_unique",
) -> np.ndarray:
    """
    Merge two index sets according to strategy:
    - union_unique: unique union (stable default for compact datasets),
    - component_concat: concatenation with duplicates (preserve component balancing weight),
    - intersection: common unique indices.
    """
    a = np.asarray(idx_primary, dtype=np.int64).reshape(-1)
    b = np.asarray(idx_secondary, dtype=np.int64).reshape(-1)
    mode = str(strategy).strip().lower()
    if mode == "component_concat":
        return np.concatenate([a, b], axis=0).astype(np.int64)
    if mode == "intersection":
        return np.intersect1d(a, b, assume_unique=False).astype(np.int64)
    return np.unique(np.concatenate([a, b], axis=0).astype(np.int64))
