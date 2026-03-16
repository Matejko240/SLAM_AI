import math
import os
import random
from dataclasses import dataclass
import numpy as np


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def scan_delta_rms(scan_prev: np.ndarray, scan_curr: np.ndarray) -> float:
    if scan_prev is None or scan_curr is None:
        return 0.0
    delta = np.asarray(scan_curr, dtype=np.float32) - np.asarray(scan_prev, dtype=np.float32)
    if delta.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(delta * delta)))


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
