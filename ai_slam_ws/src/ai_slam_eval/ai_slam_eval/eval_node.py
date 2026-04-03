import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, OccupancyGrid
    from nav_msgs.srv import GetMap
    from sensor_msgs.msg import LaserScan
except ModuleNotFoundError:
    # Pozwala importować moduł (testy funkcji pomocniczych) bez pełnego środowiska ROS2.
    rclpy = None

    class Node:  # type: ignore[override]
        pass

    class QoSProfile:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

    class ReliabilityPolicy:  # type: ignore[override]
        BEST_EFFORT = 0
        RELIABLE = 1

    class DurabilityPolicy:  # type: ignore[override]
        VOLATILE = 0
        TRANSIENT_LOCAL = 1

    class HistoryPolicy:  # type: ignore[override]
        KEEP_LAST = 0

    class PoseStamped:  # type: ignore[override]
        pass

    class Odometry:  # type: ignore[override]
        pass

    class OccupancyGrid:  # type: ignore[override]
        pass

    class LaserScan:  # type: ignore[override]
        pass

    class GetMap:  # type: ignore[override]
        class Request:
            pass
# Import loggera z pakietu ai_slam_ai
try:
    from ai_slam_ai.experiment_logger import ExperimentLogger, get_experiment_dir
except ImportError:
    # Fallback - próbuj znaleźć ścieżkę
    ExperimentLogger = None
    get_experiment_dir = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEBUG_LOG_PATH = "/home/matejko/SLAM_AI/.cursor/debug-a69755.log"
DEBUG_SESSION_ID = "a69755"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": str(run_id),
        "hypothesisId": str(hypothesis_id),
        "location": str(location),
        "message": str(message),
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def resolve_experiment_output_dir(base_out_dir: str, experiment_id: str | None) -> str:
    base_dir_abs = os.path.abspath(str(base_out_dir))
    exp_id = str(experiment_id or "").strip()
    if not exp_id:
        return base_dir_abs
    if get_experiment_dir is not None:
        return get_experiment_dir(base_dir_abs, exp_id)
    exp_folder = exp_id if exp_id.startswith("exp_") else f"exp_{exp_id}"
    return os.path.join(base_dir_abs, exp_folder)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def inverse_pose_transform_xy(x: float, y: float, tx: float, ty: float, yaw: float) -> tuple[float, float]:
    dx = float(x) - float(tx)
    dy = float(y) - float(ty)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return (
        c * dx + s * dy,
        -s * dx + c * dy,
    )


def frame_matches_hint(frame_id: str, hint: str) -> bool:
    frame = str(frame_id).strip().lower()
    h = str(hint).strip().lower()
    if not frame or not h:
        return False
    tokens = frame.replace("::", "/").replace(":", "/").split("/")
    tokens = [tok for tok in tokens if tok]
    return any((tok == h) or (h in tok) for tok in tokens)


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


def passes_motion_filter(
    prev_xyth,
    curr_xyth,
    *,
    dt_sec: float | None = None,
    min_translation: float = 0.0,
    min_rotation: float = 0.0,
    min_time_gap_sec: float = 0.0,
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

    if not checks:
        return True, delta

    filt_mode = parse_filter_mode(mode)
    return (all(checks) if filt_mode == "all" else any(checks)), delta


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def xytheta_from_pose(ps: PoseStamped):
    x = float(ps.pose.position.x)
    y = float(ps.pose.position.y)
    th = float(yaw_from_quat(ps.pose.orientation))
    return x, y, th


def xytheta_from_odom(od: Odometry):
    x = float(od.pose.pose.position.x)
    y = float(od.pose.pose.position.y)
    th = float(yaw_from_quat(od.pose.pose.orientation))
    return x, y, th


def load_yaml_simple(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def _read_token(f):
    """Czyta kolejny token z pliku PGM, pomija whitespace i komentarze #..."""
    token = b""
    while True:
        c = f.read(1)
        if not c:
            return None
        if c.isspace():
            continue
        if c == b"#":
            f.readline()
            continue
        token = c
        break

    while True:
        c = f.read(1)
        if not c or c.isspace():
            break
        token += c
    return token

def load_pgm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = _read_token(f)
        if magic not in (b"P2", b"P5"):
            raise ValueError(f"Unsupported PGM format: {magic!r}")

        w = int(_read_token(f))
        h = int(_read_token(f))
        maxval = int(_read_token(f))

        if magic == b"P2":
            vals = []
            while True:
                t = _read_token(f)
                if t is None:
                    break
                vals.append(int(t))
            arr = np.array(vals, dtype=np.uint16 if maxval > 255 else np.uint8)
            if arr.size != w * h:
                raise ValueError(f"PGM size mismatch: expected {w * h}, got {arr.size}")

        else:  # P5
            if maxval < 256:
                raw = f.read(w * h)
                if len(raw) < w * h:
                    raise ValueError("PGM size mismatch")
                arr = np.frombuffer(raw[: w * h], dtype=np.uint8)
            else:
                raw = f.read(w * h * 2)
                if len(raw) < w * h * 2:
                    raise ValueError("PGM size mismatch")
                arr = np.frombuffer(raw[: w * h * 2], dtype=">u2")

        return arr.reshape((h, w))



def occgrid_to_array(msg: OccupancyGrid) -> np.ndarray:
    w = msg.info.width
    h = msg.info.height
    data = np.array(msg.data, dtype=np.int16).reshape((h, w))
    return data


def project_map_to_ref_grid(
    ref_info: dict,
    ref_shape: tuple[int, int],
    slam_msg: OccupancyGrid,
    occ_threshold: int = 50,
):
    """Rzutuje OccupancyGrid SLAM do siatki mapy referencyjnej (z orientacją obu map)."""
    slam = occgrid_to_array(slam_msg)
    slam_res = float(slam_msg.info.resolution)
    slam_ox = float(slam_msg.info.origin.position.x)
    slam_oy = float(slam_msg.info.origin.position.y)
    slam_yaw = float(yaw_from_quat(slam_msg.info.origin.orientation))
    cs = math.cos(slam_yaw)
    ss = math.sin(slam_yaw)

    ref_res = float(ref_info["resolution"])
    origin = ref_info["origin"]
    ref_ox = float(origin[0])
    ref_oy = float(origin[1])
    ref_yaw = float(origin[2]) if len(origin) >= 3 else 0.0
    cr = math.cos(ref_yaw)
    sr = math.sin(ref_yaw)

    ref_h, ref_w = ref_shape
    occ = np.zeros((ref_h, ref_w), dtype=np.bool_)
    known = np.zeros((ref_h, ref_w), dtype=np.bool_)

    for i in range(ref_h):
        for j in range(ref_w):
            xr = (j + 0.5) * ref_res
            yr = (i + 0.5) * ref_res

            xw = ref_ox + cr * xr - sr * yr
            yw = ref_oy + sr * xr + cr * yr

            dx = xw - slam_ox
            dy = yw - slam_oy

            # world -> slam_local (R(-yaw))
            xs = cs * dx + ss * dy
            ys = -ss * dx + cs * dy

            sj = int(math.floor(xs / slam_res))
            si = int(math.floor(ys / slam_res))
            if si < 0 or sj < 0 or si >= slam.shape[0] or sj >= slam.shape[1]:
                continue

            v = int(slam[si, sj])
            if v == -1:
                continue

            known[i, j] = True
            occ[i, j] = (v >= int(occ_threshold))

    return occ, known


def map_iou(
    ref_occ: np.ndarray,
    ref_info: dict,
    slam_msg: OccupancyGrid,
    occ_threshold: int = 50,
) -> float:
    occ_s, known = project_map_to_ref_grid(ref_info, ref_occ.shape, slam_msg, occ_threshold=occ_threshold)
    return map_iou_binary(ref_occ, occ_s, known)


def map_iou_binary(ref_occ: np.ndarray, occ_s: np.ndarray, known: np.ndarray | None = None) -> float:
    occ_r = ref_occ.astype(np.bool_)
    occ_s = occ_s.astype(np.bool_)
    if known is None:
        known = np.ones_like(occ_r, dtype=np.bool_)
    else:
        known = known.astype(np.bool_)

    union_mask = known & (occ_r | occ_s)
    inter_mask = known & occ_r & occ_s

    union = int(np.count_nonzero(union_mask))
    if union == 0:
        return 1.0
    inter = int(np.count_nonzero(inter_mask))
    return float(inter) / float(union)


def best_iou_shift_only(
    ref_occ: np.ndarray,
    occ_s: np.ndarray,
    known: np.ndarray,
    max_shift_cells: int = 40,
    step: int = 4,
) -> tuple[float, int, int]:
    best_iou = map_iou_binary(ref_occ, occ_s, known)
    best_dx = 0
    best_dy = 0
    if max_shift_cells <= 0:
        return best_iou, best_dx, best_dy
    for dy in range(-max_shift_cells, max_shift_cells + 1, step):
        for dx in range(-max_shift_cells, max_shift_cells + 1, step):
            if dx == 0 and dy == 0:
                continue
            occ_shift = np.roll(occ_s, shift=(dy, dx), axis=(0, 1))
            known_shift = np.roll(known, shift=(dy, dx), axis=(0, 1))
            iou = map_iou_binary(ref_occ, occ_shift, known_shift)
            if iou > best_iou:
                best_iou = iou
                best_dx = dx
                best_dy = dy
    return float(best_iou), int(best_dx), int(best_dy)


def map_alignment_diagnostics(ref_occ: np.ndarray, ref_info: dict, slam_msg: OccupancyGrid) -> dict:
    occ50, known = project_map_to_ref_grid(ref_info, ref_occ.shape, slam_msg, occ_threshold=50)
    iou50 = map_iou_binary(ref_occ, occ50, known)
    iou35 = map_iou(ref_occ, ref_info, slam_msg, occ_threshold=35)
    iou65 = map_iou(ref_occ, ref_info, slam_msg, occ_threshold=65)
    best_iou, best_dx, best_dy = best_iou_shift_only(ref_occ, occ50, known, max_shift_cells=40, step=4)
    ref_occ_rot180 = np.rot90(ref_occ, 2).astype(np.bool_)
    iou_ref_rot180 = map_iou_binary(ref_occ_rot180, occ50, known)
    best_iou_rot180, best_dx_rot180, best_dy_rot180 = best_iou_shift_only(
        ref_occ_rot180,
        occ50,
        known,
        max_shift_cells=40,
        step=4,
    )
    world_origin = ref_info.get("origin_world", None)
    iou_world_origin = None
    if isinstance(world_origin, (list, tuple)) and len(world_origin) >= 3:
        ref_info_world = dict(ref_info)
        ref_info_world["origin"] = [float(world_origin[0]), float(world_origin[1]), float(world_origin[2])]
        try:
            iou_world_origin = map_iou(ref_occ, ref_info_world, slam_msg, occ_threshold=50)
        except Exception:
            iou_world_origin = None
    known_cells = int(np.count_nonzero(known))
    occ_cells = int(np.count_nonzero(occ50 & known))
    ref_occ_known_cells = int(np.count_nonzero(ref_occ & known))
    return {
        "iou_occ35": float(iou35),
        "iou_occ50": float(iou50),
        "iou_occ65": float(iou65),
        "known_cells": known_cells,
        "occ_cells": occ_cells,
        "ref_occ_cells_in_known": ref_occ_known_cells,
        "known_ratio_pct": (100.0 * known_cells / float(known.size)) if known.size else 0.0,
        "occ_ratio_within_known_pct": (100.0 * occ_cells / float(max(known_cells, 1))),
        "best_iou_shift_only": float(best_iou),
        "best_shift_dx_cells": int(best_dx),
        "best_shift_dy_cells": int(best_dy),
        "iou_ref_rot180_occ50": float(iou_ref_rot180),
        "best_iou_ref_rot180_shift_only": float(best_iou_rot180),
        "best_shift_ref_rot180_dx_cells": int(best_dx_rot180),
        "best_shift_ref_rot180_dy_cells": int(best_dy_rot180),
        "iou_world_origin_occ50": (None if iou_world_origin is None else float(iou_world_origin)),
        "ref_origin_local_x": float(ref_info["origin"][0]),
        "ref_origin_local_y": float(ref_info["origin"][1]),
        "ref_origin_local_yaw": float(ref_info["origin"][2]) if len(ref_info["origin"]) >= 3 else 0.0,
        "ref_origin_world_x": (float(world_origin[0]) if isinstance(world_origin, (list, tuple)) and len(world_origin) >= 1 else None),
        "ref_origin_world_y": (float(world_origin[1]) if isinstance(world_origin, (list, tuple)) and len(world_origin) >= 2 else None),
        "ref_origin_world_yaw": (float(world_origin[2]) if isinstance(world_origin, (list, tuple)) and len(world_origin) >= 3 else None),
        "map_origin_x": float(slam_msg.info.origin.position.x),
        "map_origin_y": float(slam_msg.info.origin.position.y),
        "map_origin_yaw": float(yaw_from_quat(slam_msg.info.origin.orientation)),
        "map_resolution": float(slam_msg.info.resolution),
        "map_width": int(slam_msg.info.width),
        "map_height": int(slam_msg.info.height),
    }


def rmse_from_error_components(
    err_xy: np.ndarray | list,
    err_theta: np.ndarray | list,
) -> tuple[float | None, float | None, int]:
    err_xy_arr = np.asarray(err_xy, dtype=np.float32)
    err_theta_arr = np.asarray(err_theta, dtype=np.float32)
    if err_xy_arr.size == 0 or err_theta_arr.size == 0:
        return None, None, 0
    rmse_xy = float(np.sqrt(np.mean(err_xy_arr[:, 0] ** 2 + err_xy_arr[:, 1] ** 2)))
    rmse_th = float(np.sqrt(np.mean(err_theta_arr ** 2)))
    return rmse_xy, rmse_th, int(err_theta_arr.shape[0])


def baseline_rmse_on_timeline(
    baseline_ts: np.ndarray | list,
    baseline_err_xy: np.ndarray | list,
    baseline_err_theta: np.ndarray | list,
    timeline_ts: np.ndarray | list,
) -> tuple[float | None, float | None, int]:
    base_t = np.asarray(baseline_ts, dtype=np.float64)
    target_t = np.asarray(timeline_ts, dtype=np.float64)
    if base_t.size == 0 or target_t.size == 0:
        return None, None, 0

    err_xy_arr = np.asarray(baseline_err_xy, dtype=np.float32)
    err_theta_arr = np.asarray(baseline_err_theta, dtype=np.float32)
    if err_xy_arr.size == 0 or err_theta_arr.size == 0:
        return None, None, 0

    time_key = lambda value: int(round(float(value) * 1_000_000.0))
    baseline_lookup = {time_key(ts): idx for idx, ts in enumerate(base_t)}
    matched_idx = [baseline_lookup[key] for key in (time_key(ts) for ts in target_t) if key in baseline_lookup]
    if not matched_idx:
        return None, None, 0

    matched_xy = err_xy_arr[matched_idx]
    matched_th = err_theta_arr[matched_idx]
    rmse_xy = float(np.sqrt(np.mean(matched_xy[:, 0] ** 2 + matched_xy[:, 1] ** 2)))
    rmse_th = float(np.sqrt(np.mean(matched_th ** 2)))
    return rmse_xy, rmse_th, int(len(matched_idx))


def error_trend_summary(ts: np.ndarray | list, err_xy: np.ndarray | list) -> dict:
    t = np.asarray(ts, dtype=np.float64)
    e = np.asarray(err_xy, dtype=np.float64)
    if t.size == 0 or e.size == 0 or e.ndim != 2 or e.shape[1] < 2:
        return {
            "count": 0,
            "duration_sec": 0.0,
            "mean_xy_error_m": None,
            "p95_xy_error_m": None,
            "start_window_mean_xy_error_m": None,
            "end_window_mean_xy_error_m": None,
            "drift_delta_end_minus_start_m": None,
            "end_xy_error_m": None,
        }
    err_norm = np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)
    n = int(err_norm.shape[0])
    win = max(1, int(max(10, n * 0.1)))
    start_mean = float(np.mean(err_norm[:win]))
    end_mean = float(np.mean(err_norm[-win:]))
    return {
        "count": n,
        "duration_sec": float(t[-1] - t[0]) if t.size >= 2 else 0.0,
        "mean_xy_error_m": float(np.mean(err_norm)),
        "p95_xy_error_m": float(np.percentile(err_norm, 95.0)),
        "start_window_mean_xy_error_m": start_mean,
        "end_window_mean_xy_error_m": end_mean,
        "drift_delta_end_minus_start_m": float(end_mean - start_mean),
        "end_xy_error_m": float(err_norm[-1]),
    }


def logodds_to_probability(logodds: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logodds, dtype=np.float32), -10.0, 10.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def bresenham_cells(i0: int, j0: int, i1: int, j1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    step_i = 1 if i0 < i1 else -1
    step_j = 1 if j0 < j1 else -1
    err = dj - di
    i = i0
    j = j0

    while True:
        cells.append((i, j))
        if i == i1 and j == j1:
            break
        err2 = 2 * err
        if err2 > -di:
            err -= di
            j += step_j
        if err2 < dj:
            err += dj
            i += step_i

    return cells


class EvalNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run EvalNode runtime.")
        super().__init__("eval_node")
        self.declare_parameter("seed", 123)
        self.declare_parameter("mode", "baseline")
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("duration_sec", 120.0)
        self.declare_parameter("reference_map_yaml", "")
        self.declare_parameter("spawn_x", 0.0)
        self.declare_parameter("spawn_y", 0.0)
        self.declare_parameter("spawn_yaw", 0.0)
        self.declare_parameter("gt_world_frame_hint", "world")
        self.declare_parameter("gt_jump_filter_enabled", True)
        self.declare_parameter("gt_jump_filter_max_step_m", 2.0)
        self.declare_parameter("config_snapshot_path", "")
        self.declare_parameter("world_name", "")
        self.declare_parameter("evaluation_label", "")
        self.declare_parameter("artifact_subdir", "")
        self.declare_parameter("finalize_experiment", True)
        self.declare_parameter("write_experiment_metadata", True)
        self.declare_parameter("finalize_wait_sec", -1.0)
        self.declare_parameter("finalize_wait_factor", 0.05)
        self.declare_parameter("finalize_wait_min_sec", 3.0)
        self.declare_parameter("finalize_wait_max_sec", 30.0)
        # --- tematy (żeby launch mógł przekazać inne nazwy bez edycji kodu)
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_topic_ai", "/pose_ai")
        self.declare_parameter("pose_topic_scanmatch", "/pose_scanmatch")
        self.declare_parameter("pose_topic_bruteforce", "/pose_bruteforce")
        self.declare_parameter("pose_topic_robak", "/pose_robak")
        self.declare_parameter("pose_topic_rywak", "/pose_rywak")
        self.declare_parameter("pose_topic_robak_no_slam", "/pose_robak_no_slam")
        self.declare_parameter("pose_topic_rywak_no_slam", "/pose_rywak_no_slam")
        self.declare_parameter("scan_topic_points", "/scan_slam")
        self.declare_parameter("points_max_range", 8.0)
        self.declare_parameter("points_beam_step", 6)
        self.declare_parameter("points_min_translation", 0.0)
        self.declare_parameter("points_min_rotation", 0.0)
        self.declare_parameter("points_min_time_gap_sec", 0.0)
        self.declare_parameter("points_filter_mode", "any")
        self.declare_parameter("points_use_probabilities", True)
        self.declare_parameter("points_occ_logodds_hit", 0.85)
        self.declare_parameter("points_free_logodds_miss", 0.40)
        self.declare_parameter("points_logodds_min", -4.0)
        self.declare_parameter("points_logodds_max", 4.0)
        self.declare_parameter("sync_tolerance_sec", 0.15)
        self.declare_parameter("maps_rotate_180", True)
        self.declare_parameter("maps_max_cols", 3)
        # --- nazwy artefaktów (żeby results.json wskazywał faktyczne pliki)
        self.declare_parameter("robak_dataset_name", "dataset_robak.npz")
        self.declare_parameter("robak_model_name", "model_robak.pt")
        self.declare_parameter("robak_history_name", "train_history_robak.json")
        self.declare_parameter("rywak_dataset_name", "dataset_rywak.npz")
        self.declare_parameter("rywak_model_name", "model_rywak.pt")
        self.declare_parameter("rywak_history_name", "train_history_rywak.json")
        self.seed = int(self.get_parameter("seed").value)
        self.mode = str(self.get_parameter("mode").value)
        base_out_dir = str(self.get_parameter("out_dir").value)
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.ref_yaml = str(self.get_parameter("reference_map_yaml").value)
        self.spawn_x = float(self.get_parameter("spawn_x").value)
        self.spawn_y = float(self.get_parameter("spawn_y").value)
        self.spawn_yaw = float(self.get_parameter("spawn_yaw").value)
        self.gt_world_frame_hint = str(self.get_parameter("gt_world_frame_hint").value).strip().lower()
        self.gt_jump_filter_enabled = bool(self.get_parameter("gt_jump_filter_enabled").value)
        self.gt_jump_filter_max_step_m = float(self.get_parameter("gt_jump_filter_max_step_m").value)
        self.config_snapshot_path = str(self.get_parameter("config_snapshot_path").value)
        self.world_name = str(self.get_parameter("world_name").value).strip()
        requested_label = str(self.get_parameter("evaluation_label").value).strip()
        self.evaluation_label = requested_label or self.world_name or "evaluation"
        self.artifact_subdir = str(self.get_parameter("artifact_subdir").value).strip()
        self.finalize_experiment = bool(self.get_parameter("finalize_experiment").value)
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.finalize_wait_sec = float(self.get_parameter("finalize_wait_sec").value)
        self.finalize_wait_factor = float(self.get_parameter("finalize_wait_factor").value)
        self.finalize_wait_min_sec = float(self.get_parameter("finalize_wait_min_sec").value)
        self.finalize_wait_max_sec = float(self.get_parameter("finalize_wait_max_sec").value)
        self.config_snapshot_out = ""

        # Inicjalizacja loggera eksperymentu (używa istniejącego podfolderu)
        self.exp_logger = None
        if self.write_experiment_metadata and ExperimentLogger is not None:
            self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
            self.out_dir = self.exp_logger.get_output_dir()
        else:
            self.out_dir = resolve_experiment_output_dir(base_out_dir, experiment_id)

        self.artifact_dir = (
            os.path.join(self.out_dir, self.artifact_subdir)
            if self.artifact_subdir
            else self.out_dir
        )
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.get_logger().info(f"Output directory: {self.out_dir}")
        self.get_logger().info(f"Evaluation artifact directory: {self.artifact_dir}")
        self.get_logger().info(f"Evaluation label: {self.evaluation_label}")
        if self.exp_logger:
            self.get_logger().info(f"Experiment ID: {self.exp_logger.experiment_id}")
        if self.config_snapshot_path and os.path.exists(self.config_snapshot_path):
            try:
                self.config_snapshot_out = os.path.join(self.artifact_dir, "config_snapshot.yaml")
                shutil.copyfile(self.config_snapshot_path, self.config_snapshot_out)
            except Exception as exc:
                self.config_snapshot_out = ""
                self.get_logger().warn(f"Could not copy config snapshot: {exc}")

        self.gt = None
        self.odom = None
        self.pose_ai = None
        self.pose_sm = None
        self.pose_bf = None
        self.pose_robak = None
        self.pose_rywak = None
        self.pose_robak_no_slam = None
        self.pose_rywak_no_slam = None

        self.sm_xy = []
        self.bf_xy = []
        self.robak_xy = []
        self.rywak_xy = []
        self.robak_no_slam_xy = []
        self.rywak_no_slam_xy = []

        self.err_xy_sm = []
        self.err_th_sm = []
        self.ts_sm = []
        self.err_xy_bf = []
        self.err_th_bf = []
        self.ts_bf = []
        self.err_xy_robak = []
        self.err_th_robak = []
        self.ts_robak = []
        self.err_xy_rywak = []
        self.err_th_rywak = []
        self.ts_rywak = []
        self.err_xy_robak_no_slam = []
        self.err_th_robak_no_slam = []
        self.ts_robak_no_slam = []
        self.err_xy_rywak_no_slam = []
        self.err_th_rywak_no_slam = []
        self.ts_rywak_no_slam = []

        self.map_baseline = None
        self.map_ai = None
        self.map_robak = None
        self.map_rywak = None
        # Ustal start czasu dopiero po ruszeniu zegara ROS/symulacji.
        # Gdy node startuje przed pojawieniem się /clock, zapisanie t0 tutaj
        # może dać błędny punkt odniesienia i ewaluacja nigdy nie dojdzie do duration_sec.
        self.t0 = None

        self.ts = []
        self.gt_xy = []
        self.odom_xy = []
        self.ai_xy = []

        self.err_xy = []
        self.err_th = []
        self.err_xy_ai = []
        self.err_th_ai = []
        self.ts_ai = []
        self.prev_gt_xy = None
        self.prev_odom_xy = None
        self.gt_jump_filter_dropped = 0
        
        # Śledzenie momentu startu inferencji AI
        self.ai_start_time = None  # Czas pierwszego otrzymania /pose_ai
        self.ai_start_idx = None   # Indeks w ts gdy AI wystartowało

        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pose_topic_ai = str(self.get_parameter("pose_topic_ai").value)
        self.pose_topic_scanmatch = str(self.get_parameter("pose_topic_scanmatch").value)
        self.pose_topic_bruteforce = str(self.get_parameter("pose_topic_bruteforce").value)
        self.pose_topic_robak = str(self.get_parameter("pose_topic_robak").value)
        self.pose_topic_rywak = str(self.get_parameter("pose_topic_rywak").value)
        self.pose_topic_robak_no_slam = str(self.get_parameter("pose_topic_robak_no_slam").value)
        self.pose_topic_rywak_no_slam = str(self.get_parameter("pose_topic_rywak_no_slam").value)
        self.scan_topic_points = str(self.get_parameter("scan_topic_points").value)
        self.points_max_range = float(self.get_parameter("points_max_range").value)
        self.points_beam_step = int(self.get_parameter("points_beam_step").value)
        self.points_min_translation = float(self.get_parameter("points_min_translation").value)
        self.points_min_rotation = float(self.get_parameter("points_min_rotation").value)
        self.points_min_time_gap_sec = float(self.get_parameter("points_min_time_gap_sec").value)
        self.points_filter_mode = parse_filter_mode(str(self.get_parameter("points_filter_mode").value))
        self.points_use_probabilities = bool(self.get_parameter("points_use_probabilities").value)
        self.points_occ_logodds_hit = float(self.get_parameter("points_occ_logodds_hit").value)
        self.points_free_logodds_miss = float(self.get_parameter("points_free_logodds_miss").value)
        self.points_logodds_min = float(self.get_parameter("points_logodds_min").value)
        self.points_logodds_max = float(self.get_parameter("points_logodds_max").value)
        self.sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self.maps_rotate_180 = bool(self.get_parameter("maps_rotate_180").value)
        self.maps_max_cols = max(1, int(self.get_parameter("maps_max_cols").value))
        self.create_subscription(PoseStamped, self.gt_topic, self.on_gt, 50)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.create_subscription(PoseStamped, self.pose_topic_ai, self.on_ai, 50)
        self.create_subscription(PoseStamped, self.pose_topic_scanmatch, self.on_sm, 50)
        self.create_subscription(PoseStamped, self.pose_topic_bruteforce, self.on_bf, 50)
        self.create_subscription(PoseStamped, self.pose_topic_robak, self.on_robak, 50)
        self.create_subscription(PoseStamped, self.pose_topic_rywak, self.on_rywak, 50)
        self.create_subscription(PoseStamped, self.pose_topic_robak_no_slam, self.on_robak_no_slam, 50)
        self.create_subscription(PoseStamped, self.pose_topic_rywak_no_slam, self.on_rywak_no_slam, 50)
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.create_subscription(LaserScan, self.scan_topic_points, self.on_scan_points, scan_qos)
        # QoS for map topics - slam_toolbox uses RELIABLE + TRANSIENT_LOCAL
        # Create both TRANSIENT_LOCAL and VOLATILE subscriptions for maximum compatibility
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Primary subscription with TRANSIENT_LOCAL (should get late-joining messages)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
        self.create_subscription(OccupancyGrid, "/map_ai", self.on_map_ai, map_qos)
        
        # Also create VOLATILE subscriptions as fallback (simpler QoS, always compatible)
        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(OccupancyGrid, "/map", self.on_map, volatile_qos)
        self.create_subscription(OccupancyGrid, "/map_ai", self.on_map_ai, volatile_qos)
        self.create_subscription(OccupancyGrid, "/map_robak", self.on_map_robak, map_qos)
        self.create_subscription(OccupancyGrid, "/map_rywak", self.on_map_rywak, map_qos)

        self.create_subscription(OccupancyGrid, "/map_robak", self.on_map_robak, volatile_qos)
        self.create_subscription(OccupancyGrid, "/map_rywak", self.on_map_rywak, volatile_qos)
        # Create service clients for requesting maps directly from slam_toolbox
        # This is more reliable than topic subscriptions when slam_toolbox doesn't have active subscribers
        # Node names: slam_toolbox_baseline and slam_toolbox_ai (from demo.launch.py)
        self.map_service_client = self.create_client(GetMap, '/slam_toolbox_baseline/dynamic_map')
        self.map_ai_service_client = self.create_client(GetMap, '/slam_toolbox_ai/dynamic_map')
        self.map_robak_service_client = self.create_client(GetMap, '/slam_toolbox_robak/dynamic_map')
        self.map_rywak_service_client = self.create_client(GetMap, '/slam_toolbox_rywak/dynamic_map')
        self.timer = self.create_timer(0.1, self.tick)

        self.ref_info = None
        self.ref_occ = None
        self.point_map_keys = (
            "odom",
            "gt",
            "robak",
            "rywak",
            "robak_no_slam",
            "rywak_no_slam",
        )
        self.points_logodds = {key: None for key in self.point_map_keys}
        self.points_known = {key: None for key in self.point_map_keys}
        self.points_stamp_state = {
            key: {
                "last_pose": None,
                "last_time": None,
                "stamped_scans": 0,
                "skipped_scans": 0,
                "stamped_points": 0,
                "free_cells_updated": 0,
            }
            for key in self.point_map_keys
        }

        if self.ref_yaml:
            self.ref_info = self._load_ref_info(self.ref_yaml)
            self.ref_occ = self._load_ref_occ(self.ref_yaml, self.ref_info)
        if self.ref_occ is not None:
            h, w = self.ref_occ.shape
            for key in self.point_map_keys:
                self.points_logodds[key] = np.zeros((h, w), dtype=np.float32)
                self.points_known[key] = np.zeros((h, w), dtype=np.uint8)

        self.get_logger().info(
            "[Eval] points filter: "
            f"min_translation={self.points_min_translation}, "
            f"min_rotation={self.points_min_rotation:.3f}, "
            f"min_dt={self.points_min_time_gap_sec:.3f}, "
            f"mode={self.points_filter_mode}, "
            f"use_probabilities={self.points_use_probabilities}, "
            f"hit={self.points_occ_logodds_hit:.2f}, miss={self.points_free_logodds_miss:.2f}"
        )
        if self.gt_jump_filter_enabled:
            self.get_logger().info(
                f"[Eval] GT jump filter enabled: max_step={self.gt_jump_filter_max_step_m:.2f}m"
            )
        
        # Logowanie startu ewaluacji
        if self.exp_logger is not None:
            self.exp_logger.start_evaluation(
                seed=self.seed,
                mode=self.mode,
                duration_sec=self.duration_sec,
                reference_map_yaml=self.ref_yaml
            )
        self.get_logger().info(
            "[Eval] Trajectory RMSE: 'baseline' keys = GT vs odom_topic "
            f"({self.odom_topic!r}), not slam_toolbox pose. "
            "Classical SLAM accuracy is reflected in IoU for /map and trajectory plots."
        )

    def _load_ref_info(self, yaml_path):
        y = load_yaml_simple(yaml_path)
        origin = y.get("origin", "[-3.0, -3.0, 0.0]").strip()
        origin = origin.strip("[]")
        origin_vals = [float(v.strip()) for v in origin.split(",")]
        while len(origin_vals) < 3:
            origin_vals.append(0.0)
        res = float(y.get("resolution", "0.1"))
        img = y.get("image", "reference_map.pgm").strip()
        local_ox, local_oy = inverse_pose_transform_xy(
            origin_vals[0],
            origin_vals[1],
            self.spawn_x,
            self.spawn_y,
            self.spawn_yaw,
        )
        local_yaw = wrap(float(origin_vals[2]) - self.spawn_yaw)
        return {
            "resolution": res,
            "origin": [local_ox, local_oy, local_yaw],
            "origin_world": origin_vals[:3],
            "spawn_pose": [self.spawn_x, self.spawn_y, self.spawn_yaw],
            "image": img,
        }

    def _load_ref_occ(self, yaml_path, info):
        base = os.path.dirname(yaml_path)
        img_path = os.path.join(base, info["image"])
        pgm = load_pgm(img_path)
        occ = (pgm < 128).astype(np.bool_)
        return occ

    def _ref_info_for_map_comparison(self) -> dict:
        """Projection for /map IoU should use world-frame map origin."""
        info = dict(self.ref_info) if self.ref_info is not None else {}
        world_origin = info.get("origin_world", None)
        if isinstance(world_origin, (list, tuple)) and len(world_origin) >= 3:
            info["origin"] = [float(world_origin[0]), float(world_origin[1]), float(world_origin[2])]
        return info

    def on_gt(self, msg: PoseStamped):
        self.gt = msg

    def on_odom(self, msg: Odometry):
        self.odom = msg

    def on_ai(self, msg: PoseStamped):
        self.pose_ai = msg
        # Zapisz moment pierwszego otrzymania danych z AI
        if self.ai_start_time is None:
            if self.t0 is None:
                t = 0.0
            else:
                t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
            self.ai_start_time = t
            self.ai_start_idx = len(self.ts)
            self.get_logger().info(f"AI inference started at t={t:.1f}s (idx={self.ai_start_idx})")
    def on_sm(self, msg: PoseStamped):
        self.pose_sm = msg

    def on_bf(self, msg: PoseStamped):
        self.pose_bf = msg
    def on_robak(self, msg: PoseStamped):
        self.pose_robak = msg

    def on_rywak(self, msg: PoseStamped):
        self.pose_rywak = msg

    def on_robak_no_slam(self, msg: PoseStamped):
        self.pose_robak_no_slam = msg

    def on_rywak_no_slam(self, msg: PoseStamped):
        self.pose_rywak_no_slam = msg
    def on_map(self, msg: OccupancyGrid):
        self.map_baseline = msg
        if self.map_baseline is not None and not hasattr(self, '_map_logged'):
            self._map_logged = True
            self.get_logger().info(f"Received /map: {msg.info.width}x{msg.info.height}, res={msg.info.resolution}")
    def on_map_robak(self, msg: OccupancyGrid):
        self.map_robak = msg

    def on_map_rywak(self, msg: OccupancyGrid):
        self.map_rywak = msg
    def on_map_ai(self, msg: OccupancyGrid):
        self.map_ai = msg
        if self.map_ai is not None and not hasattr(self, '_map_ai_logged'):
            self._map_ai_logged = True
            self.get_logger().info(f"Received /map_ai: {msg.info.width}x{msg.info.height}, res={msg.info.resolution}")

    def _points_grids(self, state_key: str):
        if state_key in self.points_logodds and state_key in self.points_known:
            return self.points_logodds[state_key], self.points_known[state_key]
        return None, None

    def _points_occ_known(self, state_key: str):
        logodds_grid, known_grid = self._points_grids(state_key)
        if logodds_grid is None or known_grid is None:
            return None, None, None
        known = known_grid.astype(np.bool_)
        prob = logodds_to_probability(logodds_grid)
        # strict threshold: neutral p=0.5 (brak informacji) nie jest traktowane jako occupied
        occ = prob > 0.5
        return occ, known, prob

    def _world_to_ref_indices(self, x: float, y: float) -> tuple[int, int] | None:
        if self.ref_info is None:
            return None
        res = float(self.ref_info["resolution"])
        ox = float(self.ref_info["origin"][0])
        oy = float(self.ref_info["origin"][1])
        h, w = self.ref_occ.shape
        j = int((x - ox) / res)
        i = int((y - oy) / res)
        if 0 <= i < h and 0 <= j < w:
            return i, j
        return None

    def _apply_logodds_update(
        self,
        logodds_grid: np.ndarray,
        known_grid: np.ndarray,
        i: int,
        j: int,
        delta: float,
    ) -> None:
        logodds_grid[i, j] = float(
            np.clip(logodds_grid[i, j] + delta, self.points_logodds_min, self.points_logodds_max)
        )
        known_grid[i, j] = 1

    def _stamp_points_to_ref_grid(
        self,
        pose_msg: PoseStamped,
        scan_msg: LaserScan,
        state_key: str,
    ):
        logodds_grid, known_grid = self._points_grids(state_key)
        if logodds_grid is None or known_grid is None or self.ref_info is None:
            return

        x, y, th = xytheta_from_pose(pose_msg)
        t_scan = self._stamp_to_sec(scan_msg.header.stamp)
        state = self.points_stamp_state.get(state_key)
        curr_pose = (x, y, th)

        if state is not None and state["last_pose"] is not None:
            keep_scan, _delta = passes_motion_filter(
                state["last_pose"],
                curr_pose,
                dt_sec=None if state["last_time"] is None else max(0.0, float(t_scan - state["last_time"])),
                min_translation=self.points_min_translation,
                min_rotation=self.points_min_rotation,
                min_time_gap_sec=self.points_min_time_gap_sec,
                mode=self.points_filter_mode,
            )
            if not keep_scan:
                state["skipped_scans"] += 1
                return

        h, w = logodds_grid.shape
        robot_idx = self._world_to_ref_indices(x, y)
        if robot_idx is None:
            return

        step = max(1, int(self.points_beam_step))
        rmax = float(self.points_max_range)

        a0 = float(scan_msg.angle_min)
        da = float(scan_msg.angle_increment)
        stamped_points = 0
        free_cells_updated = 0

        for k in range(0, len(scan_msg.ranges), step):
            r = float(scan_msg.ranges[k])
            if not math.isfinite(r):
                continue
            if r < float(scan_msg.range_min) or r > float(scan_msg.range_max):
                continue
            if r > rmax:
                continue

            ang = th + (a0 + k * da)
            px = x + r * math.cos(ang)
            py = y + r * math.sin(ang)

            endpoint_idx = self._world_to_ref_indices(px, py)
            if endpoint_idx is None:
                continue

            ray_cells = bresenham_cells(robot_idx[0], robot_idx[1], endpoint_idx[0], endpoint_idx[1])
            if len(ray_cells) > 1:
                for cell_i, cell_j in ray_cells[:-1]:
                    if 0 <= cell_i < h and 0 <= cell_j < w:
                        if self.points_use_probabilities:
                            self._apply_logodds_update(
                                logodds_grid,
                                known_grid,
                                cell_i,
                                cell_j,
                                -abs(self.points_free_logodds_miss),
                            )
                        else:
                            known_grid[cell_i, cell_j] = 1
                        free_cells_updated += 1

            cell_i, cell_j = ray_cells[-1]
            if 0 <= cell_i < h and 0 <= cell_j < w:
                prev_known = bool(known_grid[cell_i, cell_j])
                prev_occupied = bool(logodds_grid[cell_i, cell_j] >= 0.0)
                if self.points_use_probabilities:
                    self._apply_logodds_update(
                        logodds_grid,
                        known_grid,
                        cell_i,
                        cell_j,
                        abs(self.points_occ_logodds_hit),
                    )
                else:
                    known_grid[cell_i, cell_j] = 1
                    logodds_grid[cell_i, cell_j] = self.points_logodds_max
                if (not prev_known) or (not prev_occupied):
                    stamped_points += 1

        if state is not None:
            state["last_pose"] = curr_pose
            state["last_time"] = t_scan
            state["stamped_scans"] += 1
            state["stamped_points"] += stamped_points
            state["free_cells_updated"] += free_cells_updated

    @staticmethod
    def _pose_stamped_from_odom(msg: Odometry) -> PoseStamped:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        return pose

    def on_scan_points(self, msg: LaserScan):
        # Mapy punktowe dla torów bez SLAM i fallback dla torów modelowych.
        if self.ref_occ is None:
            return

        if self.odom is not None:
            self._stamp_points_to_ref_grid(self._pose_stamped_from_odom(self.odom), msg, "odom")

        if self.gt is not None:
            self._stamp_points_to_ref_grid(self.gt, msg, "gt")

        if self.pose_robak is not None:
            self._stamp_points_to_ref_grid(self.pose_robak, msg, "robak")

        if self.pose_rywak is not None:
            self._stamp_points_to_ref_grid(self.pose_rywak, msg, "rywak")

        if self.pose_robak_no_slam is not None:
            self._stamp_points_to_ref_grid(self.pose_robak_no_slam, msg, "robak_no_slam")

        if self.pose_rywak_no_slam is not None:
            self._stamp_points_to_ref_grid(self.pose_rywak_no_slam, msg, "rywak_no_slam")

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _is_time_synced(self, msg_a, msg_b) -> bool:
        if msg_a is None or msg_b is None:
            return False
        try:
            ta = self._stamp_to_sec(msg_a.header.stamp)
            tb = self._stamp_to_sec(msg_b.header.stamp)
            return abs(ta - tb) <= self.sync_tolerance_sec
        except Exception:
            return False

    def tick(self):
        now = self.get_clock().now()
        if self.t0 is None:
            # Przy use_sim_time początkowy czas bywa nieustalony zanim zacznie płynąć /clock.
            # Czekamy na pierwszy sensowny tick i dopiero wtedy zerujemy licznik ewaluacji.
            if now.nanoseconds <= 0:
                return
            self.t0 = now
            self.get_logger().info(
                f"Evaluation timer started at sim_time={now.nanoseconds * 1e-9:.3f}s"
            )

        t = (now - self.t0).nanoseconds * 1e-9
        if self.gt is None or self.odom is None:
            if t >= self.duration_sec:
                self.finish()
            return
        if not self._is_time_synced(self.odom, self.gt):
            if t >= self.duration_sec:
                self.finish()
            return

        gx, gy, gth = xytheta_from_pose(self.gt)
        gt_frame = str(self.gt.header.frame_id).strip().lower()
        if self.gt_world_frame_hint and frame_matches_hint(gt_frame, self.gt_world_frame_hint):
            gx, gy = inverse_pose_transform_xy(
                gx,
                gy,
                self.spawn_x,
                self.spawn_y,
                self.spawn_yaw,
            )
            gth = wrap(gth - self.spawn_yaw)
        ox, oy, oth = xytheta_from_odom(self.odom)

        if self.gt_jump_filter_enabled and self.prev_gt_xy is not None and self.prev_odom_xy is not None:
            gt_step = math.hypot(gx - self.prev_gt_xy[0], gy - self.prev_gt_xy[1])
            od_step = math.hypot(ox - self.prev_odom_xy[0], oy - self.prev_odom_xy[1])
            dynamic_limit = max(self.gt_jump_filter_max_step_m, 6.0 * od_step + 0.25)
            if gt_step > dynamic_limit:
                self.gt_jump_filter_dropped += 1
                self.prev_odom_xy = (ox, oy)
                if self.gt_jump_filter_dropped % 20 == 0:
                    self.get_logger().warn(
                        "[Eval] Dropped GT outlier sample "
                        f"(count={self.gt_jump_filter_dropped}, step={gt_step:.2f}m, "
                        f"odom_step={od_step:.2f}m, frame='{gt_frame}')"
                    )
                if t >= self.duration_sec:
                    self.finish()
                return

        self.ts.append(float(t))
        self.gt_xy.append([gx, gy, gth])
        self.odom_xy.append([ox, oy, oth])
        self.prev_gt_xy = (gx, gy)
        self.prev_odom_xy = (ox, oy)

        ex = gx - ox
        ey = gy - oy
        eth = wrap(gth - oth)
        self.err_xy.append([ex, ey])
        self.err_th.append(eth)

        if self.pose_ai is not None and self._is_time_synced(self.pose_ai, self.gt):
            ax, ay, ath = xytheta_from_pose(self.pose_ai)
            self.ai_xy.append([ax, ay, ath])
            exa = gx - ax
            eya = gy - ay
            etha = wrap(gth - ath)
            self.err_xy_ai.append([exa, eya])
            self.err_th_ai.append(etha)
            self.ts_ai.append(float(t))
            
        if self.pose_sm is not None and self._is_time_synced(self.pose_sm, self.gt):
            sx, sy, sth = xytheta_from_pose(self.pose_sm)
            self.sm_xy.append([sx, sy, sth])
            exs = gx - sx
            eys = gy - sy
            eths = wrap(gth - sth)
            self.err_xy_sm.append([exs, eys])
            self.err_th_sm.append(eths)
            self.ts_sm.append(float(t))

        if self.pose_bf is not None and self._is_time_synced(self.pose_bf, self.gt):
            bx, by, bth = xytheta_from_pose(self.pose_bf)
            self.bf_xy.append([bx, by, bth])
            exb = gx - bx
            eyb = gy - by
            ethb = wrap(gth - bth)
            self.err_xy_bf.append([exb, eyb])
            self.err_th_bf.append(ethb)
            self.ts_bf.append(float(t))
        if self.pose_robak is not None and self._is_time_synced(self.pose_robak, self.gt):
            rx, ry, rth = xytheta_from_pose(self.pose_robak)
            self.robak_xy.append([rx, ry, rth])
            exr = gx - rx
            eyr = gy - ry
            ethr = wrap(gth - rth)
            self.err_xy_robak.append([exr, eyr])
            self.err_th_robak.append(ethr)
            self.ts_robak.append(float(t))

        if self.pose_rywak is not None and self._is_time_synced(self.pose_rywak, self.gt):
            rx, ry, rth = xytheta_from_pose(self.pose_rywak)
            self.rywak_xy.append([rx, ry, rth])
            exr = gx - rx
            eyr = gy - ry
            ethr = wrap(gth - rth)
            self.err_xy_rywak.append([exr, eyr])
            self.err_th_rywak.append(ethr)
            self.ts_rywak.append(float(t))

        if self.pose_robak_no_slam is not None and self._is_time_synced(self.pose_robak_no_slam, self.gt):
            rx, ry, rth = xytheta_from_pose(self.pose_robak_no_slam)
            self.robak_no_slam_xy.append([rx, ry, rth])
            exr = gx - rx
            eyr = gy - ry
            ethr = wrap(gth - rth)
            self.err_xy_robak_no_slam.append([exr, eyr])
            self.err_th_robak_no_slam.append(ethr)
            self.ts_robak_no_slam.append(float(t))

        if self.pose_rywak_no_slam is not None and self._is_time_synced(self.pose_rywak_no_slam, self.gt):
            rx, ry, rth = xytheta_from_pose(self.pose_rywak_no_slam)
            self.rywak_no_slam_xy.append([rx, ry, rth])
            exr = gx - rx
            eyr = gy - ry
            ethr = wrap(gth - rth)
            self.err_xy_rywak_no_slam.append([exr, eyr])
            self.err_th_rywak_no_slam.append(ethr)
            self.ts_rywak_no_slam.append(float(t))
        if t >= self.duration_sec:
            # Poczekaj na mapy przed zakończeniem (max 10s dodatkowego czasu)
            if not hasattr(self, '_map_wait_deadline'):
                self._map_wait_deadline = t + 10.0
                self._last_service_request_time = 0
                self.get_logger().info(f"Evaluation duration reached at t={t:.1f}s")
                
            # Try to request maps via service every 2 seconds during wait period
            if (
                self.map_baseline is None
                or self.map_ai is None
                or self.map_robak is None
                or self.map_rywak is None
            ) and t - self._last_service_request_time >= 2.0:
                self._last_service_request_time = t
                self._request_maps_via_service()
                
            if (self.map_baseline is None or self.map_robak is None or self.map_rywak is None) and t < self._map_wait_deadline:
                if not hasattr(self, '_waiting_for_maps'):
                    self._waiting_for_maps = True
                    self.get_logger().info("Waiting for /map, /map_robak, /map_rywak from slam_toolbox (max 10s)...")
                return  # Keep waiting
            self.finish()
    
    def _request_maps_via_service(self):
        """Request maps via service call - more reliable than topic subscription."""
        # Request baseline map - use service_is_ready() which is non-blocking
        if self.map_baseline is None:
            if self.map_service_client.service_is_ready():
                self.get_logger().info("Requesting /map via /slam_toolbox_baseline/dynamic_map service...")
                request = GetMap.Request()
                future = self.map_service_client.call_async(request)
                future.add_done_callback(self._on_map_service_response)
            else:
                self.get_logger().warn("/slam_toolbox_baseline/dynamic_map service not ready (lifecycle node may not be active)")
        
        # Request AI map
        if self.map_ai is None:
            if self.map_ai_service_client.service_is_ready():
                self.get_logger().info("Requesting /map_ai via /slam_toolbox_ai/dynamic_map service...")
                request = GetMap.Request()
                future = self.map_ai_service_client.call_async(request)
                future.add_done_callback(self._on_map_ai_service_response)
            else:
                self.get_logger().warn("/slam_toolbox_ai/dynamic_map service not ready (lifecycle node may not be active)")

        # Request Robak map
        if self.map_robak is None:
            if self.map_robak_service_client.service_is_ready():
                self.get_logger().info("Requesting /map_robak via /slam_toolbox_robak/dynamic_map service...")
                request = GetMap.Request()
                future = self.map_robak_service_client.call_async(request)
                future.add_done_callback(self._on_map_robak_service_response)
            else:
                self.get_logger().warn("/slam_toolbox_robak/dynamic_map service not ready (lifecycle node may not be active)")

        # Request Rywak map
        if self.map_rywak is None:
            if self.map_rywak_service_client.service_is_ready():
                self.get_logger().info("Requesting /map_rywak via /slam_toolbox_rywak/dynamic_map service...")
                request = GetMap.Request()
                future = self.map_rywak_service_client.call_async(request)
                future.add_done_callback(self._on_map_rywak_service_response)
            else:
                self.get_logger().warn("/slam_toolbox_rywak/dynamic_map service not ready (lifecycle node may not be active)")
    
    def _on_map_service_response(self, future):
        """Handle response from /slam_toolbox/dynamic_map service."""
        try:
            response = future.result()
            if response.map.info.width > 0 and response.map.info.height > 0:
                self.map_baseline = response.map
                self.get_logger().info(f"Received /map via service: {response.map.info.width}x{response.map.info.height}")
            else:
                self.get_logger().warn("Received empty map from service")
        except Exception as e:
            self.get_logger().error(f"Failed to get map via service: {e}")
    
    def _on_map_ai_service_response(self, future):
        """Handle response from /slam_toolbox_ai/dynamic_map service."""
        try:
            response = future.result()
            if response.map.info.width > 0 and response.map.info.height > 0:
                self.map_ai = response.map
                self.get_logger().info(f"Received /map_ai via service: {response.map.info.width}x{response.map.info.height}")
            else:
                self.get_logger().warn("Received empty AI map from service")
        except Exception as e:
            self.get_logger().error(f"Failed to get AI map via service: {e}")

    def _on_map_robak_service_response(self, future):
        """Handle response from /slam_toolbox_robak/dynamic_map service."""
        try:
            response = future.result()
            if response.map.info.width > 0 and response.map.info.height > 0:
                self.map_robak = response.map
                self.get_logger().info(
                    f"Received /map_robak via service: {response.map.info.width}x{response.map.info.height}"
                )
            else:
                self.get_logger().warn("Received empty Robak map from service")
        except Exception as e:
            self.get_logger().error(f"Failed to get Robak map via service: {e}")

    def _on_map_rywak_service_response(self, future):
        """Handle response from /slam_toolbox_rywak/dynamic_map service."""
        try:
            response = future.result()
            if response.map.info.width > 0 and response.map.info.height > 0:
                self.map_rywak = response.map
                self.get_logger().info(
                    f"Received /map_rywak via service: {response.map.info.width}x{response.map.info.height}"
                )
            else:
                self.get_logger().warn("Received empty Rywak map from service")
        except Exception as e:
            self.get_logger().error(f"Failed to get Rywak map via service: {e}")

    def finish(self):
        if len(self.ts) == 0:
            rclpy.shutdown()
            return

        gt = np.asarray(self.gt_xy, dtype=np.float32)
        od = np.asarray(self.odom_xy, dtype=np.float32)
        rmse_xy = None
        rmse_th = None
        _, _, baseline_n_samples = rmse_from_error_components(self.err_xy, self.err_th)
        if baseline_n_samples > 0:
            rmse_xy, rmse_th, _ = rmse_from_error_components(self.err_xy, self.err_th)

        rmse_xy_ai, rmse_th_ai, _ = rmse_from_error_components(self.err_xy_ai, self.err_th_ai)
        rmse_xy_sm, rmse_th_sm, _ = rmse_from_error_components(self.err_xy_sm, self.err_th_sm)
        rmse_xy_bf, rmse_th_bf, _ = rmse_from_error_components(self.err_xy_bf, self.err_th_bf)
        rmse_xy_robak, rmse_th_robak, _ = rmse_from_error_components(self.err_xy_robak, self.err_th_robak)
        rmse_xy_rywak, rmse_th_rywak, _ = rmse_from_error_components(self.err_xy_rywak, self.err_th_rywak)
        rmse_xy_robak_no_slam, rmse_th_robak_no_slam, _ = rmse_from_error_components(
            self.err_xy_robak_no_slam,
            self.err_th_robak_no_slam,
        )
        rmse_xy_rywak_no_slam, rmse_th_rywak_no_slam, _ = rmse_from_error_components(
            self.err_xy_rywak_no_slam,
            self.err_th_rywak_no_slam,
        )

        rmse_xy_base_on_ai_t, rmse_th_base_on_ai_t, n_common_ai = baseline_rmse_on_timeline(
            self.ts, self.err_xy, self.err_th, self.ts_ai
        )
        rmse_xy_base_on_sm_t, rmse_th_base_on_sm_t, n_common_sm = baseline_rmse_on_timeline(
            self.ts, self.err_xy, self.err_th, self.ts_sm
        )
        rmse_xy_base_on_bf_t, rmse_th_base_on_bf_t, n_common_bf = baseline_rmse_on_timeline(
            self.ts, self.err_xy, self.err_th, self.ts_bf
        )
        rmse_xy_base_on_robak_t, rmse_th_base_on_robak_t, n_common_robak = baseline_rmse_on_timeline(
            self.ts, self.err_xy, self.err_th, self.ts_robak
        )
        rmse_xy_base_on_rywak_t, rmse_th_base_on_rywak_t, n_common_rywak = baseline_rmse_on_timeline(
            self.ts, self.err_xy, self.err_th, self.ts_rywak
        )
        rmse_xy_base_on_robak_no_slam_t, rmse_th_base_on_robak_no_slam_t, n_common_robak_no_slam = baseline_rmse_on_timeline(
            self.ts,
            self.err_xy,
            self.err_th,
            self.ts_robak_no_slam,
        )
        rmse_xy_base_on_rywak_no_slam_t, rmse_th_base_on_rywak_no_slam_t, n_common_rywak_no_slam = baseline_rmse_on_timeline(
            self.ts,
            self.err_xy,
            self.err_th,
            self.ts_rywak_no_slam,
        )
        iou_map = None
        iou_map_ai = None
        iou_map_robak = None
        iou_map_rywak = None
        iou_map_odom_points = None
        iou_map_gt_points = None
        iou_map_robak_no_slam = None
        iou_map_rywak_no_slam = None
        ref_info_cmp = self._ref_info_for_map_comparison()
        # Debug: sprawdź czy mamy dane do obliczenia IOU
        self.get_logger().info(
            "IOU calculation: "
            f"ref_occ={self.ref_occ is not None}, "
            f"map_baseline={self.map_baseline is not None}, "
            f"map_ai={self.map_ai is not None}, "
            f"map_robak={self.map_robak is not None}, "
            f"map_rywak={self.map_rywak is not None}"
        )
        # #region agent log H9 map comparison origin mode
        _debug_log(
            run_id="post-fix",
            hypothesis_id="H9",
            location="eval_node.py:map_comparison_ref_origin_mode",
            message="reference origin used for map comparison",
            data={
                "ref_origin_local": (
                    list(self.ref_info.get("origin", [])) if isinstance(self.ref_info, dict) else None
                ),
                "ref_origin_world": (
                    list(self.ref_info.get("origin_world", [])) if isinstance(self.ref_info, dict) else None
                ),
                "ref_origin_used": list(ref_info_cmp.get("origin", [])) if isinstance(ref_info_cmp, dict) else None,
            },
        )
        # #endregion
        
        if self.ref_occ is not None and self.map_baseline is not None:
            try:
                iou_map = map_iou(self.ref_occ, ref_info_cmp, self.map_baseline)
                self.get_logger().info(f"IOU baseline: {iou_map}")
                # #region agent log H6-H8 baseline map diagnostics
                _debug_log(
                    run_id="post-fix",
                    hypothesis_id="H6",
                    location="eval_node.py:map_alignment_diag_baseline",
                    message="baseline map alignment diagnostics",
                    data=map_alignment_diagnostics(self.ref_occ, self.ref_info, self.map_baseline),
                )
                # #endregion
            except Exception as e:
                self.get_logger().error(f"Failed to calculate IOU baseline: {e}")
        else:
            if self.ref_occ is None:
                self.get_logger().warn("No reference map loaded - IOU cannot be calculated")
            if self.map_baseline is None:
                self.get_logger().warn("No /map received - IOU baseline cannot be calculated")
                
        if self.ref_occ is not None and self.map_ai is not None:
            try:
                iou_map_ai = map_iou(self.ref_occ, ref_info_cmp, self.map_ai)
                self.get_logger().info(f"IOU AI: {iou_map_ai}")
                # #region agent log H6-H8 ai map diagnostics
                _debug_log(
                    run_id="post-fix",
                    hypothesis_id="H6",
                    location="eval_node.py:map_alignment_diag_ai",
                    message="ai map alignment diagnostics",
                    data=map_alignment_diagnostics(self.ref_occ, self.ref_info, self.map_ai),
                )
                # #endregion
            except Exception as e:
                self.get_logger().error(f"Failed to calculate IOU AI: {e}")
        else:
            if self.map_ai is None:
                self.get_logger().warn("No /map_ai received - IOU AI cannot be calculated")

        if self.ref_occ is not None:
            if self.map_robak is not None:
                try:
                    iou_map_robak = map_iou(self.ref_occ, ref_info_cmp, self.map_robak)
                    self.get_logger().info(f"IOU Robak: {iou_map_robak}")
                    # #region agent log H6-H8 robak map diagnostics
                    _debug_log(
                        run_id="post-fix",
                        hypothesis_id="H6",
                        location="eval_node.py:map_alignment_diag_robak",
                        message="robak map alignment diagnostics",
                        data=map_alignment_diagnostics(self.ref_occ, self.ref_info, self.map_robak),
                    )
                    # #endregion
                except Exception as e:
                    self.get_logger().error(f"Failed to calculate IOU Robak: {e}")
            else:
                occ_pts, known_pts, _prob_pts = self._points_occ_known("robak")
                if occ_pts is not None and known_pts is not None and np.count_nonzero(known_pts) > 0:
                    try:
                        iou_map_robak = map_iou_binary(self.ref_occ, occ_pts, known=known_pts)
                        self.get_logger().info(f"IOU Robak (points fallback): {iou_map_robak}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to calculate IOU Robak (points): {e}")
                else:
                    self.get_logger().warn("No /map_robak received - IOU Robak cannot be calculated")

            if self.map_rywak is not None:
                try:
                    iou_map_rywak = map_iou(self.ref_occ, ref_info_cmp, self.map_rywak)
                    self.get_logger().info(f"IOU Rywak: {iou_map_rywak}")
                    # #region agent log H6-H8 rywak map diagnostics
                    _debug_log(
                        run_id="post-fix",
                        hypothesis_id="H6",
                        location="eval_node.py:map_alignment_diag_rywak",
                        message="rywak map alignment diagnostics",
                        data=map_alignment_diagnostics(self.ref_occ, self.ref_info, self.map_rywak),
                    )
                    # #endregion
                except Exception as e:
                    self.get_logger().error(f"Failed to calculate IOU Rywak: {e}")
            else:
                occ_pts, known_pts, _prob_pts = self._points_occ_known("rywak")
                if occ_pts is not None and known_pts is not None and np.count_nonzero(known_pts) > 0:
                    try:
                        iou_map_rywak = map_iou_binary(self.ref_occ, occ_pts, known=known_pts)
                        self.get_logger().info(f"IOU Rywak (points fallback): {iou_map_rywak}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to calculate IOU Rywak (points): {e}")
                else:
                    self.get_logger().warn("No /map_rywak received - IOU Rywak cannot be calculated")

            def _iou_from_points(state_key: str, label: str):
                occ_pts, known_pts, _prob_pts = self._points_occ_known(state_key)
                if occ_pts is None or known_pts is None or np.count_nonzero(known_pts) <= 0:
                    self.get_logger().warn(f"No points map for {label} - IOU cannot be calculated")
                    return None
                try:
                    return map_iou_binary(self.ref_occ, occ_pts, known=known_pts)
                except Exception as exc:
                    self.get_logger().error(f"Failed to calculate IOU {label} (points): {exc}")
                    return None

            iou_map_odom_points = _iou_from_points("odom", "Odom")
            iou_map_gt_points = _iou_from_points("gt", "GT")
            iou_map_robak_no_slam = _iou_from_points("robak_no_slam", "Robak no SLAM")
            iou_map_rywak_no_slam = _iou_from_points("rywak_no_slam", "Rywak no SLAM")

        traj_path = os.path.join(self.artifact_dir, "eval_trajectory.png")
        err_path = os.path.join(self.artifact_dir, "eval_errors.png")
        err_hist_path = os.path.join(self.artifact_dir, "eval_error_histograms.png")
        maps_path = os.path.join(self.artifact_dir, "eval_maps.png")
        map_layers_path = os.path.join(self.artifact_dir, "eval_map_layers.npz")
        traj_data_path = os.path.join(self.artifact_dir, "eval_trajectory_data.npz")
        results_path = os.path.join(self.artifact_dir, "results.json")

        self._save_trajectory_data(traj_data_path)
        self._plot_trajectories(traj_path)
        self._plot_errors(err_path)
        self._plot_error_histograms(err_hist_path)
        err_hist_per_method_artifacts = self._plot_error_histograms_per_method()
        dataset_hist_artifacts = self._plot_dataset_histograms()
        self._save_map_layers(map_layers_path)
        self._plot_maps(maps_path)

        results = {
            "mode": self.mode,
            "seed": self.seed,
            "duration_sec": self.duration_sec,
            "world_name": self.world_name,
            "evaluation_label": self.evaluation_label,
            "artifact_subdir": self.artifact_subdir,
            "metrics_legend": {
                "rmse_xy_odom_topic": (
                    f"Kanon: RMSE ||p_gt - p_odom|| w XY; p_odom z odom_topic={self.odom_topic!r}. "
                    "Nie jest to pozycja z slam_toolbox; jakość mapy: iou_map_baseline."
                ),
                "rmse_xy_baseline": "Alias legace (wartość jak rmse_xy_odom_topic).",
                "rmse_theta_odom_topic": (
                    "Kanon: RMSE błędu yaw — GT vs odom_topic (ta sama referencja pozycji co RMSE XY odom)."
                ),
                "rmse_theta_baseline": "Alias legace (wartość jak rmse_theta_odom_topic).",
                "rmse_xy_ai": f"RMSE GT vs pose from pose_topic_ai={self.pose_topic_ai!r}.",
                "rmse_xy_robak": f"RMSE GT vs pose from pose_topic_robak={self.pose_topic_robak!r}.",
                "rmse_xy_rywak": f"RMSE GT vs pose from pose_topic_rywak={self.pose_topic_rywak!r}.",
                "rmse_xy_robak_no_slam": (
                    f"RMSE GT vs pose from pose_topic_robak_no_slam={self.pose_topic_robak_no_slam!r}."
                ),
                "rmse_xy_rywak_no_slam": (
                    f"RMSE GT vs pose from pose_topic_rywak_no_slam={self.pose_topic_rywak_no_slam!r}."
                ),
                "iou_map_baseline": "IoU vs reference occupancy for slam_toolbox /map.",
                "iou_map_odom_points": "IoU mapy punktowej budowanej z czystej odometrii.",
                "iou_map_gt_points": "IoU mapy punktowej budowanej z trajektorii GT.",
                "iou_map_robak_no_slam": "IoU mapy punktowej toru Robak bez SLAM.",
                "iou_map_rywak_no_slam": "IoU mapy punktowej toru Rywak bez SLAM.",
                "rmse_xy_odom_topic_on_ai_timeline": "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami AI.",
                "rmse_xy_odom_topic_on_scanmatch_timeline": "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami ScanMatcher.",
                "rmse_xy_odom_topic_on_bruteforce_timeline": "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami Bruteforce.",
                "rmse_xy_odom_topic_on_robak_timeline": "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami Robak.",
                "rmse_xy_odom_topic_on_rywak_timeline": "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami Rywak.",
                "rmse_xy_odom_topic_on_robak_no_slam_timeline": (
                    "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami Robak bez SLAM."
                ),
                "rmse_xy_odom_topic_on_rywak_no_slam_timeline": (
                    "Baseline RMSE XY liczony na wspólnej osi czasu z próbkami Rywak bez SLAM."
                ),
                "note_trajectory_alignment": (
                    "Global RMSE in a fixed frame compares integration conventions; "
                    "SLAM vs GT in *map* frame usually needs Sim(2) alignment (ATE) for a fair scalar."
                ),
            },
            "metrics": {
                "rmse_xy_odom_topic": rmse_xy,
                "rmse_theta_odom_topic": rmse_th,
                "rmse_xy_baseline": rmse_xy,
                "rmse_theta_baseline": rmse_th,
                "iou_map_baseline": iou_map,
                "rmse_xy_ai": rmse_xy_ai,
                "rmse_theta_ai": rmse_th_ai,
                "iou_map_ai": iou_map_ai,
                "rmse_xy_odom_topic_on_ai_timeline": rmse_xy_base_on_ai_t,
                "rmse_theta_odom_topic_on_ai_timeline": rmse_th_base_on_ai_t,
                "n_common_samples_ai": int(n_common_ai),
                "iou_map_robak": iou_map_robak,
                "iou_map_rywak": iou_map_rywak,
                "rmse_xy_scanmatch": rmse_xy_sm,
                "rmse_theta_scanmatch": rmse_th_sm,
                "rmse_xy_odom_topic_on_scanmatch_timeline": rmse_xy_base_on_sm_t,
                "rmse_theta_odom_topic_on_scanmatch_timeline": rmse_th_base_on_sm_t,
                "n_common_samples_scanmatch": int(n_common_sm),
                "rmse_xy_bruteforce": rmse_xy_bf,
                "rmse_theta_bruteforce": rmse_th_bf,
                "rmse_xy_odom_topic_on_bruteforce_timeline": rmse_xy_base_on_bf_t,
                "rmse_theta_odom_topic_on_bruteforce_timeline": rmse_th_base_on_bf_t,
                "n_common_samples_bruteforce": int(n_common_bf),
                "rmse_xy_robak": rmse_xy_robak,
                "rmse_theta_robak": rmse_th_robak,
                "rmse_xy_odom_topic_on_robak_timeline": rmse_xy_base_on_robak_t,
                "rmse_theta_odom_topic_on_robak_timeline": rmse_th_base_on_robak_t,
                "n_common_samples_robak": int(n_common_robak),
                "rmse_xy_rywak": rmse_xy_rywak,
                "rmse_theta_rywak": rmse_th_rywak,
                "rmse_xy_odom_topic_on_rywak_timeline": rmse_xy_base_on_rywak_t,
                "rmse_theta_odom_topic_on_rywak_timeline": rmse_th_base_on_rywak_t,
                "n_common_samples_rywak": int(n_common_rywak),
                "rmse_xy_robak_no_slam": rmse_xy_robak_no_slam,
                "rmse_theta_robak_no_slam": rmse_th_robak_no_slam,
                "rmse_xy_odom_topic_on_robak_no_slam_timeline": rmse_xy_base_on_robak_no_slam_t,
                "rmse_theta_odom_topic_on_robak_no_slam_timeline": rmse_th_base_on_robak_no_slam_t,
                "n_common_samples_robak_no_slam": int(n_common_robak_no_slam),
                "rmse_xy_rywak_no_slam": rmse_xy_rywak_no_slam,
                "rmse_theta_rywak_no_slam": rmse_th_rywak_no_slam,
                "rmse_xy_odom_topic_on_rywak_no_slam_timeline": rmse_xy_base_on_rywak_no_slam_t,
                "rmse_theta_odom_topic_on_rywak_no_slam_timeline": rmse_th_base_on_rywak_no_slam_t,
                "n_common_samples_rywak_no_slam": int(n_common_rywak_no_slam),
                "iou_map_odom_points": iou_map_odom_points,
                "iou_map_gt_points": iou_map_gt_points,
                "iou_map_robak_no_slam": iou_map_robak_no_slam,
                "iou_map_rywak_no_slam": iou_map_rywak_no_slam,
                "n_evaluation_samples": int(len(self.ts)),
            },
            "diagnostics": {
                "gt_jump_filter": {
                    "enabled": bool(self.gt_jump_filter_enabled),
                    "max_step_m": float(self.gt_jump_filter_max_step_m),
                    "dropped_samples": int(self.gt_jump_filter_dropped),
                },
                "point_map_filter": {
                    key: {
                        "stamped_scans": int(state["stamped_scans"]),
                        "skipped_scans": int(state["skipped_scans"]),
                        "stamped_points": int(state["stamped_points"]),
                        "free_cells_updated": int(state["free_cells_updated"]),
                        "known_cells": int(np.count_nonzero(self._points_grids(key)[1])) if self._points_grids(key)[1] is not None else 0,
                        "occupied_cells_prob_ge_0_5": (
                            int(np.count_nonzero(self._points_occ_known(key)[0])) if self._points_occ_known(key)[0] is not None else 0
                        ),
                    }
                    for key, state in self.points_stamp_state.items()
                }
            },
            "artifacts": {
                "results_json": results_path,
                "trajectory_png": traj_path,
                "trajectory_data_npz": traj_data_path,
                "errors_png": err_path,
                "errors_histogram_png": err_hist_path,
                "maps_png": maps_path,
                "map_layers_npz": map_layers_path,
                "reference_map_yaml": self.ref_yaml,
                "config_snapshot_yaml": self.config_snapshot_out,
                "map_topic_baseline": "/map",
                "map_topic_ai": "/map_ai",
                "map_topic_robak": "/map_robak",
                "map_topic_rywak": "/map_rywak",
                "pose_topic_robak_no_slam": self.pose_topic_robak_no_slam,
                "pose_topic_rywak_no_slam": self.pose_topic_rywak_no_slam,
                "dataset_npz": os.path.join(self.out_dir, "dataset.npz"),
                "model_pt": os.path.join(self.out_dir, "model.pt"),
                "train_history_json": os.path.join(self.out_dir, "train_history.json"),
                "experiment_metadata": os.path.join(self.out_dir, "experiment_metadata.json"),
                "robak_dataset_npz": os.path.join(self.out_dir, str(self.get_parameter("robak_dataset_name").value)),
                "robak_model_pt": os.path.join(self.out_dir, str(self.get_parameter("robak_model_name").value)),
                "robak_train_history_json": os.path.join(self.out_dir, str(self.get_parameter("robak_history_name").value)),
                "rywak_dataset_npz": os.path.join(self.out_dir, str(self.get_parameter("rywak_dataset_name").value)),
                "rywak_model_pt": os.path.join(self.out_dir, str(self.get_parameter("rywak_model_name").value)),
                "rywak_train_history_json": os.path.join(self.out_dir, str(self.get_parameter("rywak_history_name").value)),
            },
        }
        results["artifacts"].update(err_hist_per_method_artifacts)
        results["artifacts"].update(dataset_hist_artifacts)
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="eval_node.py:results_summary",
            message="trajectory tracks availability and baseline semantics",
            data={
                "n_eval_samples": int(len(self.ts)),
                "n_common_scanmatch": int(n_common_sm),
                "n_common_ai": int(n_common_ai),
                "n_common_robak": int(n_common_robak),
                "n_common_rywak": int(n_common_rywak),
                "n_common_robak_no_slam": int(n_common_robak_no_slam),
                "n_common_rywak_no_slam": int(n_common_rywak_no_slam),
                "rmse_xy_odom_topic": rmse_xy,
                "rmse_xy_scanmatch": rmse_xy_sm,
                "rmse_xy_ai": rmse_xy_ai,
                "rmse_xy_robak": rmse_xy_robak,
                "rmse_xy_rywak": rmse_xy_rywak,
                "rmse_xy_robak_no_slam": rmse_xy_robak_no_slam,
                "rmse_xy_rywak_no_slam": rmse_xy_rywak_no_slam,
            },
        )
        # endregion
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="eval_node.py:map_quality",
            message="map quality and known/occupied coverage",
            data={
                "iou_map_baseline": iou_map,
                "iou_map_ai": iou_map_ai,
                "iou_map_robak": iou_map_robak,
                "iou_map_rywak": iou_map_rywak,
                "iou_map_odom_points": iou_map_odom_points,
                "iou_map_gt_points": iou_map_gt_points,
                "iou_map_robak_no_slam": iou_map_robak_no_slam,
                "iou_map_rywak_no_slam": iou_map_rywak_no_slam,
                "robak_known_cells": int(np.count_nonzero(self._points_grids("robak")[1])) if self._points_grids("robak")[1] is not None else 0,
                "rywak_known_cells": int(np.count_nonzero(self._points_grids("rywak")[1])) if self._points_grids("rywak")[1] is not None else 0,
                "odom_known_cells": int(np.count_nonzero(self._points_grids("odom")[1])) if self._points_grids("odom")[1] is not None else 0,
                "gt_known_cells": int(np.count_nonzero(self._points_grids("gt")[1])) if self._points_grids("gt")[1] is not None else 0,
                "robak_no_slam_known_cells": int(np.count_nonzero(self._points_grids("robak_no_slam")[1])) if self._points_grids("robak_no_slam")[1] is not None else 0,
                "rywak_no_slam_known_cells": int(np.count_nonzero(self._points_grids("rywak_no_slam")[1])) if self._points_grids("rywak_no_slam")[1] is not None else 0,
            },
        )
        # endregion
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="eval_node.py:trajectory_extents",
            message="gt and baseline extents in eval frame",
            data={
                "gt_count": int(len(self.gt_xy)),
                "baseline_count": int(len(self.odom_xy)),
                "gt_x_min": float(min([p[0] for p in self.gt_xy])) if self.gt_xy else None,
                "gt_x_max": float(max([p[0] for p in self.gt_xy])) if self.gt_xy else None,
                "gt_y_min": float(min([p[1] for p in self.gt_xy])) if self.gt_xy else None,
                "gt_y_max": float(max([p[1] for p in self.gt_xy])) if self.gt_xy else None,
                "odom_x_min": float(min([p[0] for p in self.odom_xy])) if self.odom_xy else None,
                "odom_x_max": float(max([p[0] for p in self.odom_xy])) if self.odom_xy else None,
                "odom_y_min": float(min([p[1] for p in self.odom_xy])) if self.odom_xy else None,
                "odom_y_max": float(max([p[1] for p in self.odom_xy])) if self.odom_xy else None,
            },
        )
        # endregion
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H10",
            location="eval_node.py:rmse_drift_trends",
            message="RMSE/drift trend per method on its own timeline",
            data={
                "odom_topic": str(self.odom_topic),
                "world_name": str(self.world_name),
                "baseline": error_trend_summary(self.ts, self.err_xy),
                "scanmatch": error_trend_summary(self.ts_sm, self.err_xy_sm),
                "ai": error_trend_summary(self.ts_ai, self.err_xy_ai),
                "robak": error_trend_summary(self.ts_robak, self.err_xy_robak),
                "rywak": error_trend_summary(self.ts_rywak, self.err_xy_rywak),
                "robak_no_slam": error_trend_summary(self.ts_robak_no_slam, self.err_xy_robak_no_slam),
                "rywak_no_slam": error_trend_summary(self.ts_rywak_no_slam, self.err_xy_rywak_no_slam),
                "common_samples": {
                    "scanmatch": int(n_common_sm),
                    "ai": int(n_common_ai),
                    "robak": int(n_common_robak),
                    "rywak": int(n_common_rywak),
                    "robak_no_slam": int(n_common_robak_no_slam),
                    "rywak_no_slam": int(n_common_rywak_no_slam),
                },
            },
        )
        # endregion

        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, results_path)
        
        # Logowanie zakończenia ewaluacji
        if self.exp_logger is not None:
            self.exp_logger.end_evaluation(
                rmse_xy_odom_topic=rmse_xy,
                rmse_theta_odom_topic=rmse_th,
                rmse_xy_ai=rmse_xy_ai,
                rmse_theta_ai=rmse_th_ai,
                iou_map_baseline=iou_map,
                iou_map_ai=iou_map_ai,
                iou_map_robak=iou_map_robak,
                iou_map_rywak=iou_map_rywak,
                n_samples=len(self.ts),
                artifacts=results["artifacts"]
            )

            if self.finalize_experiment:
                if self.finalize_wait_sec >= 0.0:
                    wait_sec = float(self.finalize_wait_sec)
                else:
                    dynamic_wait = float(self.duration_sec) * max(0.0, float(self.finalize_wait_factor))
                    wait_sec = dynamic_wait
                wait_min = max(0.0, float(self.finalize_wait_min_sec))
                wait_max = max(wait_min, float(self.finalize_wait_max_sec))
                wait_sec = min(wait_max, max(wait_min, wait_sec))
                self.get_logger().info(
                    f"Waiting {wait_sec:.1f}s for infer_node metadata "
                    f"(duration={self.duration_sec:.1f}s, factor={self.finalize_wait_factor:.3f})..."
                )
                time.sleep(wait_sec)
                self.exp_logger.finalize()
                # Dodaj do pliku podsumowania wszystkich eksperymentów
                # Wczytuje najnowsze dane z metadata.json, więc inference data też będą uwzględnione
                summary_path = self.exp_logger.append_to_summary()
                self.get_logger().info(f"Experiment summary saved to: {summary_path}")

                # Wyświetl podsumowanie eksperymentu
                self.get_logger().info("\n" + self.exp_logger.get_summary())
            else:
                self.get_logger().info("Skipping experiment finalize/summary for intermediate evaluation run.")

        rclpy.shutdown()

    def _save_trajectory_data(self, path):
        def _as_xytheta_array(samples):
            arr = np.asarray(samples, dtype=np.float32)
            if arr.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return arr.reshape((-1, 3))

        def _as_err_xy_array(samples):
            arr = np.asarray(samples, dtype=np.float32)
            if arr.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
            return arr.reshape((-1, 2))

        def _as_err_theta_array(samples):
            arr = np.asarray(samples, dtype=np.float32)
            if arr.size == 0:
                return np.zeros((0,), dtype=np.float32)
            return arr.reshape((-1,))

        np.savez_compressed(
            path,
            time_s=np.asarray(self.ts, dtype=np.float32),
            gt_xytheta=_as_xytheta_array(self.gt_xy),
            baseline_xytheta=_as_xytheta_array(self.odom_xy),
            baseline_err_xy=_as_err_xy_array(self.err_xy),
            baseline_err_theta=_as_err_theta_array(self.err_th),
            ai_time_s=np.asarray(self.ts_ai, dtype=np.float32),
            ai_xytheta=_as_xytheta_array(self.ai_xy),
            ai_err_xy=_as_err_xy_array(self.err_xy_ai),
            ai_err_theta=_as_err_theta_array(self.err_th_ai),
            scanmatch_time_s=np.asarray(self.ts_sm, dtype=np.float32),
            scanmatch_xytheta=_as_xytheta_array(self.sm_xy),
            scanmatch_err_xy=_as_err_xy_array(self.err_xy_sm),
            scanmatch_err_theta=_as_err_theta_array(self.err_th_sm),
            bruteforce_time_s=np.asarray(self.ts_bf, dtype=np.float32),
            bruteforce_xytheta=_as_xytheta_array(self.bf_xy),
            bruteforce_err_xy=_as_err_xy_array(self.err_xy_bf),
            bruteforce_err_theta=_as_err_theta_array(self.err_th_bf),
            robak_time_s=np.asarray(self.ts_robak, dtype=np.float32),
            robak_xytheta=_as_xytheta_array(self.robak_xy),
            robak_err_xy=_as_err_xy_array(self.err_xy_robak),
            robak_err_theta=_as_err_theta_array(self.err_th_robak),
            rywak_time_s=np.asarray(self.ts_rywak, dtype=np.float32),
            rywak_xytheta=_as_xytheta_array(self.rywak_xy),
            rywak_err_xy=_as_err_xy_array(self.err_xy_rywak),
            rywak_err_theta=_as_err_theta_array(self.err_th_rywak),
            robak_no_slam_time_s=np.asarray(self.ts_robak_no_slam, dtype=np.float32),
            robak_no_slam_xytheta=_as_xytheta_array(self.robak_no_slam_xy),
            robak_no_slam_err_xy=_as_err_xy_array(self.err_xy_robak_no_slam),
            robak_no_slam_err_theta=_as_err_theta_array(self.err_th_robak_no_slam),
            rywak_no_slam_time_s=np.asarray(self.ts_rywak_no_slam, dtype=np.float32),
            rywak_no_slam_xytheta=_as_xytheta_array(self.rywak_no_slam_xy),
            rywak_no_slam_err_xy=_as_err_xy_array(self.err_xy_rywak_no_slam),
            rywak_no_slam_err_theta=_as_err_theta_array(self.err_th_rywak_no_slam),
            position_error_unit=np.asarray(["m"], dtype=object),
            orientation_error_unit=np.asarray(["rad"], dtype=object),
        )

    def _reference_bounds_polygon(self):
        if self.ref_info is None or self.ref_occ is None:
            corners = np.array(
                [
                    [-3.0, -3.0],
                    [3.0, -3.0],
                    [3.0, 3.0],
                    [-3.0, 3.0],
                ],
                dtype=np.float32,
            )
        else:
            res = float(self.ref_info["resolution"])
            ox = float(self.ref_info["origin"][0])
            oy = float(self.ref_info["origin"][1])
            oyaw = float(self.ref_info["origin"][2]) if len(self.ref_info["origin"]) >= 3 else 0.0
            h, w = self.ref_occ.shape

            local = np.array(
                [
                    [0.0, 0.0],
                    [w * res, 0.0],
                    [w * res, h * res],
                    [0.0, h * res],
                ],
                dtype=np.float32,
            )
            c = math.cos(oyaw)
            s = math.sin(oyaw)

            corners = np.zeros_like(local, dtype=np.float32)
            corners[:, 0] = ox + c * local[:, 0] - s * local[:, 1]
            corners[:, 1] = oy + s * local[:, 0] + c * local[:, 1]

        xmin = float(np.min(corners[:, 0]))
        xmax = float(np.max(corners[:, 0]))
        ymin = float(np.min(corners[:, 1]))
        ymax = float(np.max(corners[:, 1]))
        return corners, xmin, ymin, xmax, ymax

    def _reference_walls_world_points(self, max_points: int = 50000):
        """Zwraca punkty zajętych komórek mapy referencyjnej w lokalnym układzie od spawnu."""
        if self.ref_info is None or self.ref_occ is None:
            return None

        occ = self.ref_occ.astype(np.bool_)
        ii, jj = np.nonzero(occ)
        if ii.size == 0:
            return None

        if ii.size > max_points:
            step = int(math.ceil(float(ii.size) / float(max_points)))
            ii = ii[::step]
            jj = jj[::step]

        res = float(self.ref_info["resolution"])
        ox = float(self.ref_info["origin"][0])
        oy = float(self.ref_info["origin"][1])
        oyaw = float(self.ref_info["origin"][2]) if len(self.ref_info["origin"]) >= 3 else 0.0
        c = math.cos(oyaw)
        s = math.sin(oyaw)

        x_local = (jj.astype(np.float32) + 0.5) * res
        y_local = (ii.astype(np.float32) + 0.5) * res

        x_world = ox + c * x_local - s * y_local
        y_world = oy + s * x_local + c * y_local
        return np.column_stack([x_world, y_world]).astype(np.float32)

    def _plot_trajectories(self, path):
        gt = np.asarray(self.gt_xy, dtype=np.float32)
        od = np.asarray(self.odom_xy, dtype=np.float32)

        ref_poly, _, _, _, _ = self._reference_bounds_polygon()
        ref_walls = self._reference_walls_world_points()

        fig, (ax_focus, ax_full) = plt.subplots(1, 2, figsize=(14.5, 6.4))
        from matplotlib.patches import Polygon
        label_gt = "GT (trajektoria rzeczywista)"
        label_baseline = "baseline (SLAM)"
        label_robak = "robak"
        label_rywak = "rywak"
        label_robak_no_slam = "robak (no SLAM)"
        label_rywak_no_slam = "rywak (no SLAM)"
        label_scanmatch = "scanmatch"
        label_bruteforce = "bruteforce"

        series = [
            (gt, "tab:blue", label_gt, 1.8, 1.0),
            (od, "tab:orange", label_baseline, 1.0, 0.7),
        ]
        if len(self.robak_xy) > 0:
            series.append((np.asarray(self.robak_xy, dtype=np.float32), "tab:red", label_robak, 1.0, 0.85))
        if len(self.rywak_xy) > 0:
            series.append((np.asarray(self.rywak_xy, dtype=np.float32), "tab:purple", label_rywak, 1.0, 0.85))
        if len(self.robak_no_slam_xy) > 0:
            series.append((np.asarray(self.robak_no_slam_xy, dtype=np.float32), "#ef4444", label_robak_no_slam, 1.0, 0.85))
        if len(self.rywak_no_slam_xy) > 0:
            series.append((np.asarray(self.rywak_no_slam_xy, dtype=np.float32), "#84cc16", label_rywak_no_slam, 1.0, 0.85))
        if len(self.sm_xy) > 0:
            series.append((np.asarray(self.sm_xy, dtype=np.float32), "tab:brown", label_scanmatch, 1.0, 0.8))
        if len(self.bf_xy) > 0:
            series.append((np.asarray(self.bf_xy, dtype=np.float32), "tab:pink", label_bruteforce, 1.0, 0.8))

        def draw_reference(ax, label_bounds: bool):
            if ref_walls is not None and ref_walls.size > 0:
                ax.scatter(
                    ref_walls[:, 0],
                    ref_walls[:, 1],
                    s=0.8,
                    c="0.75",
                    alpha=0.35,
                    marker="s",
                    linewidths=0,
                    label="ref walls" if label_bounds else None,
                )
            poly = Polygon(
                ref_poly,
                fill=False,
                edgecolor="gray",
                linestyle="--",
                linewidth=1.5,
                label="ref map bounds" if label_bounds else None,
            )
            ax.add_patch(poly)

        for axis, title in (
            (ax_focus, "Trajektorie względem mapy referencyjnej"),
            (ax_full, "Trajektorie (pełny widok)"),
        ):
            draw_reference(axis, label_bounds=(axis is ax_full))
            for arr, color, label, linewidth, alpha in series:
                axis.plot(arr[:, 0], arr[:, 1], color=color, label=label if axis is ax_full else None, linewidth=linewidth, alpha=alpha)
            axis.set_aspect("equal")
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_title(title)
            axis.grid(True, alpha=0.3)

        if ref_poly is not None and len(ref_poly) >= 3:
            ref_x = ref_poly[:, 0]
            ref_y = ref_poly[:, 1]
            x_span = max(float(np.max(ref_x) - np.min(ref_x)), 1.0)
            y_span = max(float(np.max(ref_y) - np.min(ref_y)), 1.0)
            x_margin = max(1.0, 0.08 * x_span)
            y_margin = max(1.0, 0.08 * y_span)
            ax_focus.set_xlim(float(np.min(ref_x)) - x_margin, float(np.max(ref_x)) + x_margin)
            ax_focus.set_ylim(float(np.min(ref_y)) - y_margin, float(np.max(ref_y)) + y_margin)

        ax_full.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
        fig.suptitle("Trajektorie (układ lokalny od spawnu)", fontsize=13)
        fig.tight_layout(rect=[0.0, 0.0, 0.84, 0.96])
        fig.savefig(path, dpi=150)
        plt.close()

    def _plot_errors(self, path):
        t = np.asarray(self.ts, dtype=np.float32)
        err = np.asarray(self.err_xy, dtype=np.float32)
        eth = np.asarray(self.err_th, dtype=np.float32)

        if t.size == 0 or err.size == 0 or eth.size == 0:
            plt.figure()
            plt.text(0.5, 0.5, "No error data available", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            return

        fig, (ax_pos, ax_theta) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)

        ax_pos.plot(t, np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2), label="baseline", linewidth=1.6)
        ax_theta.plot(t, np.abs(eth), label="baseline", linewidth=1.6)

        def _plot_series(ts_list, err_xy_list, err_th_list, label_prefix, alpha=0.7):
            if len(ts_list) == 0 or len(err_xy_list) == 0:
                return
            tt = np.asarray(ts_list, dtype=np.float32)
            e_xy = np.asarray(err_xy_list, dtype=np.float32)
            e_th = np.asarray(err_th_list, dtype=np.float32)
            n = min(tt.shape[0], e_xy.shape[0], e_th.shape[0])
            if n <= 0:
                return
            ax_pos.plot(tt[:n], np.sqrt(e_xy[:n, 0] ** 2 + e_xy[:n, 1] ** 2), label=label_prefix, alpha=alpha)
            ax_theta.plot(tt[:n], np.abs(e_th[:n]), label=label_prefix, alpha=alpha)

        _plot_series(self.ts_sm, self.err_xy_sm, self.err_th_sm, "scanmatch")
        _plot_series(self.ts_bf, self.err_xy_bf, self.err_th_bf, "bruteforce")
        _plot_series(self.ts_robak, self.err_xy_robak, self.err_th_robak, "robak")
        _plot_series(self.ts_rywak, self.err_xy_rywak, self.err_th_rywak, "rywak")
        _plot_series(
            self.ts_robak_no_slam,
            self.err_xy_robak_no_slam,
            self.err_th_robak_no_slam,
            "robak_no_slam",
        )
        _plot_series(
            self.ts_rywak_no_slam,
            self.err_xy_rywak_no_slam,
            self.err_th_rywak_no_slam,
            "rywak_no_slam",
        )

        ax_pos.set_title("Błąd pozycji")
        ax_pos.set_ylabel("error [m]")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="best")

        ax_theta.set_title("Błąd orientacji")
        ax_theta.set_xlabel("t [s]")
        ax_theta.set_ylabel("|error| [rad]")
        ax_theta.grid(True, alpha=0.3)
        ax_theta.legend(loc="best")

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _error_series_specs(self):
        return [
            ("baseline", self.err_xy, self.err_th, "#c2410c"),
            ("ai", self.err_xy_ai, self.err_th_ai, "#0f766e"),
            ("scanmatch", self.err_xy_sm, self.err_th_sm, "#2563eb"),
            ("bruteforce", self.err_xy_bf, self.err_th_bf, "#7c3aed"),
            ("robak", self.err_xy_robak, self.err_th_robak, "#b91c1c"),
            ("rywak", self.err_xy_rywak, self.err_th_rywak, "#4d7c0f"),
            ("robak_no_slam", self.err_xy_robak_no_slam, self.err_th_robak_no_slam, "#ef4444"),
            ("rywak_no_slam", self.err_xy_rywak_no_slam, self.err_th_rywak_no_slam, "#84cc16"),
        ]

    def _plot_error_histogram_for_series(self, label, err_xy_list, err_th_list, color, path):
        if len(err_xy_list) == 0 or len(err_th_list) == 0:
            return False

        e_xy = np.asarray(err_xy_list, dtype=np.float32)
        e_th = np.asarray(err_th_list, dtype=np.float32)
        if e_xy.size == 0 or e_th.size == 0:
            return False
        e_xy = e_xy.reshape((-1, 2))
        e_th = e_th.reshape((-1,))
        n = min(e_xy.shape[0], e_th.shape[0])
        if n <= 0:
            return False

        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
        bins = 40
        axes[0].hist(e_xy[:n, 0], bins=bins, color=color, alpha=0.85)
        axes[1].hist(e_xy[:n, 1], bins=bins, color=color, alpha=0.85)
        axes[2].hist(e_th[:n], bins=bins, color=color, alpha=0.85)

        axes[0].set_title(f"{label}: ex")
        axes[0].set_xlabel("ex [m]")
        axes[0].set_ylabel("Liczność")
        axes[1].set_title(f"{label}: ey")
        axes[1].set_xlabel("ey [m]")
        axes[1].set_ylabel("Liczność")
        axes[2].set_title(f"{label}: eθ")
        axes[2].set_xlabel("eθ [rad]")
        axes[2].set_ylabel("Liczność")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        tor_hint = {
            "baseline": "Tor: odom (GT − /odom)",
            "ai": "Tor: AI SLAM",
            "scanmatch": "Tor: scan-matcher lokalny",
            "bruteforce": "Tor: scan-matcher bruteforce",
            "robak": "Tor: Robak + SLAM (wejście: /scan_slam_robak)",
            "rywak": "Tor: Rywak + SLAM (wejście: /scan_slam_rywak)",
            "robak_no_slam": "Tor: Robak bez SLAM (/scan_slam_robak, bez pętli SLAM)",
            "rywak_no_slam": "Tor: Rywak bez SLAM",
        }.get(label, "")
        fig.suptitle(
            f"Histogram błędu (ex, ey, eθ względem GT): {label}\n{tor_hint}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True

    def _plot_error_histograms(self, path):
        series = self._error_series_specs()

        has_any = False
        for _label, err_xy_list, err_th_list, _color in series:
            if len(err_xy_list) > 0 and len(err_th_list) > 0:
                has_any = True
                break

        if not has_any:
            plt.figure()
            plt.text(0.5, 0.5, "No error histogram data available", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            return

        fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
        bins = 40
        for label, err_xy_list, err_th_list, color in series:
            if len(err_xy_list) == 0 or len(err_th_list) == 0:
                continue
            e_xy = np.asarray(err_xy_list, dtype=np.float32)
            e_th = np.asarray(err_th_list, dtype=np.float32)
            n = min(e_xy.shape[0], e_th.shape[0])
            if n <= 0:
                continue
            axes[0].hist(e_xy[:n, 0], bins=bins, histtype="step", linewidth=1.2, color=color, label=label)
            axes[1].hist(e_xy[:n, 1], bins=bins, histtype="step", linewidth=1.2, color=color, label=label)
            axes[2].hist(e_th[:n], bins=bins, histtype="step", linewidth=1.2, color=color, label=label)

        axes[0].set_title("Histogram błędu: ex")
        axes[0].set_xlabel("ex [m]")
        axes[0].set_ylabel("Liczność")
        axes[1].set_title("Histogram błędu: ey")
        axes[1].set_xlabel("ey [m]")
        axes[1].set_ylabel("Liczność")
        axes[2].set_title("Histogram błędu: eθ")
        axes[2].set_xlabel("eθ [rad]")
        axes[2].set_ylabel("Liczność")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_error_histograms_per_method(self):
        out = {}
        for label, err_xy_list, err_th_list, color in self._error_series_specs():
            path = os.path.join(self.artifact_dir, f"eval_error_histogram_{label}.png")
            ok = self._plot_error_histogram_for_series(
                label=label,
                err_xy_list=err_xy_list,
                err_th_list=err_th_list,
                color=color,
                path=path,
            )
            if ok:
                out[f"errors_histogram_{label}_png"] = path
        return out

    def _plot_hist_components(self, values, labels, title, path):
        if values is None or len(values) == 0:
            return False
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return False
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return False

        n_comp = arr.shape[1]
        fig, axes = plt.subplots(1, n_comp, figsize=(4.8 * n_comp, 4.4))
        if n_comp == 1:
            axes = [axes]
        bins = 40
        for idx, ax in enumerate(axes):
            ax.hist(arr[:, idx], bins=bins, color="#0f766e", alpha=0.85)
            comp_label = labels[idx] if idx < len(labels) else f"c{idx}"
            ax.set_title(comp_label)
            ax.set_xlabel(comp_label)
            ax.set_ylabel("Liczność")
            ax.grid(True, alpha=0.25)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True

    def _load_npz_y_and_meta(self, npz_path):
        if not npz_path or not os.path.exists(npz_path):
            return None, None
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                y = data["Y"].astype(np.float32) if "Y" in data else None
                meta = None
                if "meta" in data:
                    meta_raw = data["meta"]
                    if isinstance(meta_raw, np.ndarray) and meta_raw.shape == ():
                        meta = meta_raw.item()
                return y, meta
        except Exception as exc:
            self.get_logger().warn(f"Failed to load dataset npz for histogram: {npz_path} ({exc})")
            return None, None

    def _extract_meta_path(self, meta, key):
        if not isinstance(meta, dict):
            return None
        value = meta.get(key)
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size <= 0:
                return None
            try:
                value = value.reshape((-1,))[0]
            except Exception:
                return None
        return str(value)

    def _plot_dataset_histograms(self):
        out = {}

        ai_path = os.path.join(self.out_dir, "dataset.npz")
        ai_y, _ai_meta = self._load_npz_y_and_meta(ai_path)
        ai_hist_path = os.path.join(self.artifact_dir, "dataset_hist_ai_targets.png")
        if self._plot_hist_components(
            values=ai_y,
            labels=["dx [m]", "dy [m]", "dθ [rad]"],
            title="Histogram targetów datasetu AI",
            path=ai_hist_path,
        ):
            out["dataset_hist_ai_targets_png"] = ai_hist_path

        robak_name = str(self.get_parameter("robak_dataset_name").value)
        robak_path = os.path.join(self.out_dir, robak_name)
        robak_y, _robak_meta = self._load_npz_y_and_meta(robak_path)
        robak_hist_path = os.path.join(self.artifact_dir, "dataset_hist_robak_targets.png")
        if self._plot_hist_components(
            values=robak_y,
            labels=["dx [m]", "dy [m]", "dθ [rad]"],
            title="Histogram targetów datasetu Robak",
            path=robak_hist_path,
        ):
            out["dataset_hist_robak_targets_png"] = robak_hist_path

        rywak_name = str(self.get_parameter("rywak_dataset_name").value)
        rywak_path = os.path.join(self.out_dir, rywak_name)
        rywak_y, rywak_meta = self._load_npz_y_and_meta(rywak_path)
        rywak_hist_path = os.path.join(self.artifact_dir, "dataset_hist_rywak_targets.png")
        if self._plot_hist_components(
            values=rywak_y,
            labels=["v [m/s]", "ω [rad/s]"],
            title="Histogram targetów datasetu Rywak",
            path=rywak_hist_path,
        ):
            out["dataset_hist_rywak_targets_png"] = rywak_hist_path

        linear_balanced = self._extract_meta_path(rywak_meta, "balanced_linear_dataset_path")
        if linear_balanced:
            y_linear, _meta_linear = self._load_npz_y_and_meta(linear_balanced)
            linear_hist_path = os.path.join(self.artifact_dir, "dataset_hist_rywak_linear_balanced.png")
            linear_values = None
            if y_linear is not None and y_linear.ndim == 2 and y_linear.shape[1] >= 1:
                linear_values = np.abs(y_linear[:, 0]).reshape((-1, 1))
            if self._plot_hist_components(
                values=linear_values,
                labels=["|v| [m/s]"],
                title="Histogram |v| (Rywak balanced-linear)",
                path=linear_hist_path,
            ):
                out["dataset_hist_rywak_linear_balanced_png"] = linear_hist_path

        angular_balanced = self._extract_meta_path(rywak_meta, "balanced_angular_dataset_path")
        if angular_balanced:
            y_angular, _meta_angular = self._load_npz_y_and_meta(angular_balanced)
            angular_hist_path = os.path.join(self.artifact_dir, "dataset_hist_rywak_angular_balanced.png")
            angular_values = None
            if y_angular is not None and y_angular.ndim == 2 and y_angular.shape[1] >= 2:
                angular_values = np.abs(y_angular[:, 1]).reshape((-1, 1))
            if self._plot_hist_components(
                values=angular_values,
                labels=["|ω| [rad/s]"],
                title="Histogram |ω| (Rywak balanced-angular)",
                path=angular_hist_path,
            ):
                out["dataset_hist_rywak_angular_balanced_png"] = angular_hist_path

        return out

    def _collect_map_layers(self):
        if self.ref_occ is None:
            return []

        ref_info_cmp = self._ref_info_for_map_comparison()
        layers = [{"key": "ref", "title": "GT reference", "data": self.ref_occ.astype(np.float32)}]

        def _append_occ(key, title, msg):
            if msg is None:
                return
            occ, _known = project_map_to_ref_grid(ref_info_cmp, self.ref_occ.shape, msg)
            layers.append({"key": key, "title": title, "data": occ.astype(np.float32)})

        def _append_points(key, title):
            occ, known, _prob = self._points_occ_known(key)
            if occ is not None and known is not None and np.count_nonzero(known) > 0:
                layers.append({"key": key, "title": title, "data": occ.astype(np.float32)})

        _append_points("gt", "GT points")
        _append_points("odom", "Odom points")
        _append_occ("baseline_slam", "baseline_slam", self.map_baseline)
        _append_occ("ai_slam", "ai_slam", self.map_ai)

        if getattr(self, "map_robak", None) is not None:
            mr, _known = project_map_to_ref_grid(ref_info_cmp, self.ref_occ.shape, self.map_robak)
            layers.append({"key": "robak_slam", "title": "robak_slam", "data": mr.astype(np.float32)})
        else:
            _append_points("robak", "robak_points_fallback")

        if getattr(self, "map_rywak", None) is not None:
            my, _known = project_map_to_ref_grid(ref_info_cmp, self.ref_occ.shape, self.map_rywak)
            layers.append({"key": "rywak_slam", "title": "rywak_slam", "data": my.astype(np.float32)})
        else:
            _append_points("rywak", "rywak_points_fallback")

        _append_points("robak_no_slam", "robak_no_slam")
        _append_points("rywak_no_slam", "rywak_no_slam")

        return layers

    def _save_map_layers(self, path):
        layers = self._collect_map_layers()
        if not layers:
            return

        payload = {
            "rotate_180": np.asarray([1 if self.maps_rotate_180 else 0], dtype=np.uint8),
        }
        for item in layers:
            payload[str(item["key"])] = np.asarray(item["data"], dtype=np.float32)
        np.savez_compressed(path, **payload)

    def _plot_maps(self, path):
        layers = self._collect_map_layers()
        if not layers:
            plt.figure()
            plt.text(0.5, 0.5, "No reference map loaded", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            return

        n_maps = len(layers)
        if n_maps <= self.maps_max_cols:
            nrows = 1
            ncols = n_maps
        else:
            nrows = 2
            ncols = int(math.ceil(n_maps / 2.0))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
        axes_flat = axes.ravel()

        for i, item in enumerate(layers):
            ax = axes_flat[i]
            disp = np.asarray(item["data"], dtype=np.float32)
            if self.maps_rotate_180:
                disp = np.rot90(disp, 2)
            ax.imshow(disp, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(str(item["title"]))
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n_maps, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle("Porownanie map", fontsize=12)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    if rclpy is None:
        raise RuntimeError("rclpy is required to run ai_slam_eval.eval_node.")
    rclpy.init()
    node = EvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if getattr(node, "exp_logger", None) is not None:
                node.exp_logger.finalize()
        except Exception as e:
            node.get_logger().error(f"Finalize failed: {e}")
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
