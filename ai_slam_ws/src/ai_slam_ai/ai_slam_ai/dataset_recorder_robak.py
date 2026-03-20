import io
import math
import os
import time
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .common import (
    atomic_write_bytes,
    ensure_dir,
    parse_filter_mode,
    passes_motion_filter,
    seed_all,
    wrap,
    xytheta_from_pose_stamped,
)
from .experiment_logger import ExperimentLogger


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _resample_to_360(ranges: np.ndarray) -> np.ndarray | None:
    """Resample arbitrary scan length to 360 beams."""
    n = int(ranges.size)
    if n == 360:
        return ranges.astype(np.float32)
    if n < 10:
        return None
    x_old = np.linspace(-math.pi, math.pi, n, endpoint=False)
    x_new = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    out = np.interp(x_new, x_old, ranges).astype(np.float32)
    return out


def _sanitize_scan(msg: LaserScan) -> np.ndarray | None:
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    rmax = float(msg.range_max) if msg.range_max > 0 else 10.0
    rmin = float(msg.range_min) if msg.range_min > 0 else 0.08

    ranges = np.where(np.isfinite(ranges), ranges, rmax).astype(np.float32)
    ranges = np.clip(ranges, rmin, rmax)

    if ranges.size != 360:
        ranges = _resample_to_360(ranges)
        if ranges is None:
            return None
        ranges = np.clip(ranges, rmin, rmax)

    return ranges.astype(np.float32)


def _interp_angle(th0: float, th1: float, alpha: float) -> float:
    d = wrap(float(th1) - float(th0))
    return wrap(float(th0) + float(alpha) * d)


def _augment_scan_pair(
    scan_prev: np.ndarray,
    scan_curr: np.ndarray,
    rng: np.random.Generator,
    noise_std_scale: float,
    cut_fraction: float,
    cut_max_points: int,
    range_min: float = 0.08,
    range_max: float = 10.0,
):
    """ALSAI-like augmentation: proportional Gaussian noise + random beam masking."""
    out_prev = scan_prev.copy().astype(np.float32)
    out_curr = scan_curr.copy().astype(np.float32)
    augmented = False

    if noise_std_scale > 0.0:
        sigma_prev = np.maximum(np.abs(out_prev) * noise_std_scale, 1e-4).astype(np.float32)
        sigma_curr = np.maximum(np.abs(out_curr) * noise_std_scale, 1e-4).astype(np.float32)
        out_prev += rng.normal(0.0, sigma_prev, size=out_prev.shape).astype(np.float32)
        out_curr += rng.normal(0.0, sigma_curr, size=out_curr.shape).astype(np.float32)
        augmented = True

    if cut_fraction > 0.0 and cut_max_points > 0 and rng.random() < cut_fraction:
        points = int(rng.integers(1, cut_max_points + 1))
        start = int(rng.integers(0, out_prev.size))
        idx = (start + np.arange(points)) % out_prev.size
        out_prev[idx] = range_min
        out_curr[idx] = range_min
        augmented = True

    if augmented:
        out_prev = np.clip(out_prev, range_min, range_max).astype(np.float32)
        out_curr = np.clip(out_curr, range_min, range_max).astype(np.float32)

    return out_prev, out_curr, augmented


def _delta_pose(
    prev_xyth: Tuple[float, float, float],
    curr_xyth: Tuple[float, float, float],
    label_frame: str,
) -> Tuple[float, float, float]:
    """Compute pose delta in either previous-local or world frame."""
    x1, y1, th1 = prev_xyth
    x2, y2, th2 = curr_xyth

    dx_w = x2 - x1
    dy_w = y2 - y1
    dth = wrap(th2 - th1)

    if label_frame.lower() == "world":
        return float(dx_w), float(dy_w), float(dth)

    c = math.cos(th1)
    s = math.sin(th1)
    dx_l = c * dx_w + s * dy_w
    dy_l = -s * dx_w + c * dy_w
    return float(dx_l), float(dy_l), float(dth)


class DatasetRecorderRobak(Node):
    def __init__(self):
        super().__init__("dataset_recorder_robak")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("max_samples", 8000)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("dataset_name", "dataset_robak.npz")
        self.declare_parameter("write_experiment_metadata", False)

        self.declare_parameter("offsets", [1, 2, 3, 4, 5, 8, 10])
        self.declare_parameter("min_pair_dist", 0.0)
        self.declare_parameter("min_pair_dyaw", 0.0)
        self.declare_parameter("min_pair_dt_sec", 0.0)
        self.declare_parameter("pair_filter_mode", "any")
        self.declare_parameter("max_pair_dist", 0.5)
        self.declare_parameter("max_pair_dyaw", float(math.pi))

        self.declare_parameter("label_frame", "local")
        self.declare_parameter("sync_tolerance_sec", 0.08)
        self.declare_parameter("sync_pair_gap_sec", 0.2)
        self.declare_parameter("interpolate_gt", True)
        self.declare_parameter("augment_noise_std_scale", 0.0)
        self.declare_parameter("augment_cut_fraction", 0.0)
        self.declare_parameter("augment_cut_max_points", 20)

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None

        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        ensure_dir(self.out_dir)

        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))

        self.offsets: List[int] = [int(x) for x in list(self.get_parameter("offsets").value)]
        self.offsets = sorted(list(set([o for o in self.offsets if o > 0])))
        self.min_pair_dist = float(self.get_parameter("min_pair_dist").value)
        self.min_pair_dyaw = float(self.get_parameter("min_pair_dyaw").value)
        self.min_pair_dt_sec = float(self.get_parameter("min_pair_dt_sec").value)
        self.pair_filter_mode = parse_filter_mode(str(self.get_parameter("pair_filter_mode").value))
        self.max_pair_dist = float(self.get_parameter("max_pair_dist").value)
        self.max_pair_dyaw = float(self.get_parameter("max_pair_dyaw").value)
        self.label_frame = str(self.get_parameter("label_frame").value).lower()
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self.sync_pair_gap_sec = float(self.get_parameter("sync_pair_gap_sec").value)
        self.interpolate_gt = bool(self.get_parameter("interpolate_gt").value)
        self.augment_noise_std_scale = float(self.get_parameter("augment_noise_std_scale").value)
        self.augment_cut_fraction = float(self.get_parameter("augment_cut_fraction").value)
        self.augment_cut_max_points = int(self.get_parameter("augment_cut_max_points").value)
        self.rng_aug = np.random.default_rng(self.seed + 1337)

        self.gt_count = 0
        self.gt_miss_count = 0
        self.gt_sync_interp_count = 0
        self.gt_sync_nearest_count = 0
        self.pair_accept_count = 0
        self.pair_reject_filter_count = 0
        self.aug_samples_count = 0
        self.scan_count = 0
        self.scan_rx_count = 0
        self.pending_drop_count = 0

        self.gt_buf: Deque[Tuple[float, Tuple[float, float, float]]] = deque(maxlen=2000)
        self.pending_scans: Deque[Tuple[float, np.ndarray]] = deque()
        self.max_pending_scans = 4000
        self.buf: Deque[Tuple[float, np.ndarray, Tuple[float, float, float]]] = deque(
            maxlen=max(self.offsets) + 1
        )

        self.X_pairs = []
        self.Y = []

        self.t0 = None
        self.topics_ready = False
        self.experiment_start = time.time()
        self.is_finishing = False

        self.sub_gt = self.create_subscription(PoseStamped, self.gt_topic, self.on_gt, 50)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        self.wait_timer = self.create_timer(1.0, self.wait_for_topics)
        self.timer = self.create_timer(0.5, self.check_done)

        self.get_logger().info(f"[Robak] out_dir={self.out_dir}")
        self.get_logger().info(f"[Robak] scan={self.scan_topic}, gt={self.gt_topic}, offsets={self.offsets}")
        self.get_logger().info(
            f"[Robak] label_frame={self.label_frame}, "
            f"min_dist={self.min_pair_dist}, min_dyaw={self.min_pair_dyaw:.3f}, "
            f"min_dt={self.min_pair_dt_sec:.3f}, filter_mode={self.pair_filter_mode}, "
            f"max_dist={self.max_pair_dist}, max_dyaw={self.max_pair_dyaw:.3f}"
        )
        self.get_logger().info(
            f"[Robak] sync: tol={self.sync_tolerance_sec:.3f}, gap={self.sync_pair_gap_sec:.3f}, "
            f"interp_gt={self.interpolate_gt}"
        )
        self.get_logger().info(
            f"[Robak] augment: noise_std_scale={self.augment_noise_std_scale}, "
            f"cut_fraction={self.augment_cut_fraction}, cut_max_points={self.augment_cut_max_points}"
        )

        if self.write_experiment_metadata:
            self.exp_logger.start_dataset_collection(
                seed=self.seed,
                duration_sec=self.duration_sec,
                max_samples=self.max_samples,
                scan_topic=self.scan_topic,
                odom_topic="(unused)",
                gt_topic=self.gt_topic,
            )

    def on_gt(self, msg: PoseStamped):
        self.gt_count += 1
        t = _stamp_to_sec(msg.header.stamp)
        self.gt_buf.append((t, xytheta_from_pose_stamped(msg)))
        self._process_pending_scans()

    def _trim_gt_buffer(self, t_ref: float):
        while len(self.gt_buf) > 2 and self.gt_buf[1][0] < (t_ref - 1.0):
            self.gt_buf.popleft()

    def _nearest_gt(self, t_scan: float):
        if not self.gt_buf:
            return None
        self._trim_gt_buffer(t_scan)
        t_best, pose_best = min(self.gt_buf, key=lambda item: abs(item[0] - t_scan))
        if abs(t_best - t_scan) > self.sync_tolerance_sec:
            return None
        return pose_best

    def _interpolated_gt(self, t_scan: float):
        if len(self.gt_buf) < 2:
            return None

        prev = None
        for cur in self.gt_buf:
            if cur[0] < t_scan:
                prev = cur
                continue

            if prev is None:
                return None

            t0, pose0 = prev
            t1, pose1 = cur
            gap = float(t1 - t0)
            if gap < 1e-6:
                if abs(t_scan - t0) <= self.sync_tolerance_sec:
                    return pose0
                return None
            if gap > self.sync_pair_gap_sec:
                return None
            if t_scan < t0 or t_scan > t1:
                return None

            x0, y0, th0 = pose0
            x1, y1, th1 = pose1
            alpha = (t_scan - t0) / gap
            x = float(x0 + alpha * (x1 - x0))
            y = float(y0 + alpha * (y1 - y0))
            th = _interp_angle(th0, th1, alpha)
            return x, y, th

        return None

    def _gt_at(self, t_scan: float):
        if self.interpolate_gt:
            pose = self._interpolated_gt(t_scan)
            if pose is not None:
                return pose, "interp"
        pose = self._nearest_gt(t_scan)
        if pose is not None:
            return pose, "nearest"
        return None, None

    def _can_attempt_gt(self, t_scan: float, flush: bool):
        if not self.gt_buf:
            return False
        if flush:
            return True

        self._trim_gt_buffer(t_scan)
        earliest_t = float(self.gt_buf[0][0])
        latest_t = float(self.gt_buf[-1][0])
        if earliest_t > t_scan or latest_t >= t_scan:
            return True

        t_best, _pose_best = min(self.gt_buf, key=lambda item: abs(item[0] - t_scan))
        return abs(t_best - t_scan) <= self.sync_tolerance_sec

    def _append_pending_scan(self, t_scan: float, scan: np.ndarray):
        self.pending_scans.append((t_scan, scan.copy()))
        while len(self.pending_scans) > self.max_pending_scans:
            self.pending_scans.popleft()
            self.pending_drop_count += 1

    def _process_gt_sample(self, t_scan: float, scan: np.ndarray, curr_gt: Tuple[float, float, float]):
        self.buf.append((t_scan, scan, curr_gt))
        self.scan_count += 1

        if len(self.buf) < (max(self.offsets) + 1):
            return

        t_curr, scan_curr, gt_curr = self.buf[-1]
        for off in self.offsets:
            t_prev, scan_prev, gt_prev = self.buf[-(off + 1)]

            keep_motion, delta = passes_motion_filter(
                gt_prev,
                gt_curr,
                dt_sec=max(0.0, float(t_curr - t_prev)),
                min_translation=self.min_pair_dist,
                min_rotation=self.min_pair_dyaw,
                min_time_gap_sec=self.min_pair_dt_sec,
                mode=self.pair_filter_mode,
            )

            if delta.distance > self.max_pair_dist:
                self.pair_reject_filter_count += 1
                continue
            if abs(delta.dtheta) > self.max_pair_dyaw:
                self.pair_reject_filter_count += 1
                continue
            if not keep_motion:
                self.pair_reject_filter_count += 1
                continue

            dx, dy, dth = _delta_pose(gt_prev, gt_curr, self.label_frame)
            self.pair_accept_count += 1

            self.X_pairs.append(np.stack([scan_prev, scan_curr], axis=0).astype(np.float32))
            self.Y.append(np.asarray([dx, dy, dth], dtype=np.float32))

            if len(self.Y) >= self.max_samples and not self.is_finishing:
                self.save_and_exit()
                return

            if self.augment_noise_std_scale > 0.0 or self.augment_cut_fraction > 0.0:
                aug_prev, aug_curr, did_aug = _augment_scan_pair(
                    scan_prev=scan_prev,
                    scan_curr=scan_curr,
                    rng=self.rng_aug,
                    noise_std_scale=self.augment_noise_std_scale,
                    cut_fraction=self.augment_cut_fraction,
                    cut_max_points=self.augment_cut_max_points,
                )
                if did_aug:
                    self.X_pairs.append(np.stack([aug_prev, aug_curr], axis=0).astype(np.float32))
                    self.Y.append(np.asarray([dx, dy, dth], dtype=np.float32))
                    self.aug_samples_count += 1
                    if len(self.Y) >= self.max_samples and not self.is_finishing:
                        self.save_and_exit()
                        return

    def _process_pending_scans(self, flush: bool = False):
        while self.pending_scans:
            t_scan, scan = self.pending_scans[0]
            if not self._can_attempt_gt(t_scan, flush):
                break

            self.pending_scans.popleft()
            curr_gt, gt_mode = self._gt_at(t_scan)
            if curr_gt is None:
                self.gt_miss_count += 1
                continue

            if gt_mode == "interp":
                self.gt_sync_interp_count += 1
            else:
                self.gt_sync_nearest_count += 1

            self._process_gt_sample(t_scan, scan, curr_gt)
            if self.is_finishing:
                return

    def wait_for_topics(self):
        if self.topics_ready:
            return
        if len(self.gt_buf) > 0:
            self.topics_ready = True
            self.t0 = self.get_clock().now()
            elapsed_exp = time.time() - self.experiment_start
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"[Robak][FAZA 1] DATASET START (t={elapsed_exp:.0f}s)")
            self.get_logger().info("=" * 60)
        else:
            self.get_logger().info(f"[Robak] Waiting for GT... (n_gt={self.gt_count})")

    def on_scan(self, msg: LaserScan):
        if not self.topics_ready:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        self.scan_rx_count += 1
        t_scan = _stamp_to_sec(msg.header.stamp)
        self._append_pending_scan(t_scan, scan)
        self._process_pending_scans()

    def check_done(self):
        if self.t0 is None:
            return
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self.save_and_exit()

    def save_and_exit(self):
        if self.is_finishing:
            return
        self.is_finishing = True
        self._process_pending_scans(flush=True)

        if len(self.Y) == 0:
            self.get_logger().error(
                f"[Robak] No samples collected. scan_rx={self.scan_rx_count}, "
                f"gt_miss={self.gt_miss_count}, pending_drops={self.pending_drop_count}"
            )
            rclpy.shutdown()
            return

        X = np.stack(self.X_pairs).astype(np.float32)
        Y = np.stack(self.Y).astype(np.float32)
        pair_distance = np.linalg.norm(Y[:, :2], axis=1).astype(np.float32)
        pair_rotation_abs_deg = np.rad2deg(np.abs(Y[:, 2])).astype(np.float32)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(Y.shape[0]),
            "x_shape": np.asarray(X.shape, dtype=np.int64),
            "label_frame": np.asarray([self.label_frame], dtype=object),
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            "min_pair_dist": np.float32(self.min_pair_dist),
            "min_pair_dyaw": np.float32(self.min_pair_dyaw),
            "min_pair_dt_sec": np.float32(self.min_pair_dt_sec),
            "pair_filter_mode": np.asarray([self.pair_filter_mode], dtype=object),
            "max_pair_dist": np.float32(self.max_pair_dist),
            "max_pair_dyaw": np.float32(self.max_pair_dyaw),
            "sync_tolerance_sec": np.float32(self.sync_tolerance_sec),
            "sync_pair_gap_sec": np.float32(self.sync_pair_gap_sec),
            "interpolate_gt": np.int64(1 if self.interpolate_gt else 0),
            "scan_rx": np.int64(self.scan_rx_count),
            "pending_drop_count": np.int64(self.pending_drop_count),
            "gt_sync_miss_count": np.int64(self.gt_miss_count),
            "gt_sync_interp_count": np.int64(self.gt_sync_interp_count),
            "gt_sync_nearest_count": np.int64(self.gt_sync_nearest_count),
            "pair_accept_count": np.int64(self.pair_accept_count),
            "pair_reject_filter_count": np.int64(self.pair_reject_filter_count),
            "augment_noise_std_scale": np.float32(self.augment_noise_std_scale),
            "augment_cut_fraction": np.float32(self.augment_cut_fraction),
            "augment_cut_max_points": np.int64(self.augment_cut_max_points),
            "augment_added_samples": np.int64(self.aug_samples_count),
            "label_translation_mean_m": np.float32(np.mean(pair_distance)),
            "label_translation_p95_m": np.float32(np.percentile(pair_distance, 95)),
            "label_translation_max_m": np.float32(np.max(pair_distance)),
            "label_rotation_abs_mean_deg": np.float32(np.mean(pair_rotation_abs_deg)),
            "label_rotation_abs_p95_deg": np.float32(np.percentile(pair_rotation_abs_deg, 95)),
            "label_rotation_abs_max_deg": np.float32(np.max(pair_rotation_abs_deg)),
        }

        ensure_dir(self.out_dir)
        try:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X_pairs=X, Y=Y, meta=meta)
            atomic_write_bytes(self.dataset_path, buffer.getvalue())
        except Exception as e:
            self.get_logger().error(f"[Robak] Failed to save dataset: {e}")
            rclpy.shutdown()
            return

        actual_duration = 0.0
        try:
            if self.t0 is not None:
                actual_duration = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        except Exception:
            pass

        if self.write_experiment_metadata:
            self.exp_logger.end_dataset_collection(
                n_samples=int(Y.shape[0]),
                scan_dim=720,
                actual_duration_sec=float(actual_duration),
                file_path=self.dataset_path,
            )

        self.get_logger().info(
            f"[Robak] Saved dataset: {self.dataset_path} "
            f"(n={Y.shape[0]}, pairs_ok={self.pair_accept_count}, "
            f"pairs_filtered={self.pair_reject_filter_count}, aug_added={self.aug_samples_count}, "
            f"gt_interp={self.gt_sync_interp_count}, gt_nearest={self.gt_sync_nearest_count}, "
            f"gt_sync_miss={self.gt_miss_count}, scan_rx={self.scan_rx_count})"
        )
        rclpy.shutdown()


def main():
    rclpy.init()
    node = DatasetRecorderRobak()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
