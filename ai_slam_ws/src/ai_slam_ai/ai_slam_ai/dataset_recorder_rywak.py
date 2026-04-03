import io
import math
import os
import time
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from .common import (
    atomic_write_bytes,
    ensure_dir,
    merge_balanced_indices,
    parse_filter_mode,
    passes_motion_filter,
    scan_delta_rms,
    seed_all,
    uniform_histogram_sample_indices,
    wrap,
    xytheta_from_odom,
)
from .experiment_logger import ExperimentLogger


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _resample_to_360(ranges: np.ndarray) -> np.ndarray:
    n = int(ranges.size)
    if n == 360:
        return ranges.astype(np.float32)
    if n < 10:
        return None
    x_old = np.linspace(-math.pi, math.pi, n, endpoint=False)
    x_new = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    return np.interp(x_new, x_old, ranges).astype(np.float32)


def _sanitize_scan(msg: LaserScan) -> np.ndarray:
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


def _normalize_trajectory_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"no_cycle", "nocycle", "acyclic", "without_cycles"}:
        return "no_cycle"
    if mode in {"cycle", "cyclic", "with_cycles"}:
        return "cycle"
    return "any"


def _normalize_balance_merge_strategy(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"component_concat", "concat", "components", "sum"}:
        return "component_concat"
    if mode in {"intersection", "intersect"}:
        return "intersection"
    return "union_unique"


class DatasetRecorderRywak(Node):
    def __init__(self):
        super().__init__("dataset_recorder_rywak")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("max_samples", 5000)

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom_raw")
        self.declare_parameter("dataset_name", "dataset_rywak.npz")
        self.declare_parameter("write_experiment_metadata", False)
        self.declare_parameter("sync_tolerance_sec", 0.08)
        self.declare_parameter("interpolate_odom", True)
        self.declare_parameter("sync_pair_gap_sec", 0.2)
        self.declare_parameter("delta_scan_clip", 2.0)
        self.declare_parameter("min_sample_dist", 0.0)
        self.declare_parameter("min_sample_dyaw", 0.0)
        self.declare_parameter("min_sample_dt_sec", 0.0)
        self.declare_parameter("min_delta_scan_rms", 0.0)
        self.declare_parameter("sample_filter_mode", "any")
        self.declare_parameter("balance_histograms", True)
        self.declare_parameter("balance_bins", 24)
        self.declare_parameter("balance_linear_use_abs", True)
        self.declare_parameter("balance_angular_use_abs", True)
        self.declare_parameter("balance_linear_hist_min_mps", 0.0)
        self.declare_parameter("balance_linear_hist_max_mps", 2.0)
        self.declare_parameter("balance_angular_hist_min_radps", 0.0)
        self.declare_parameter("balance_angular_hist_max_radps", 3.0)
        self.declare_parameter("balance_target_quantile", 0.35)
        self.declare_parameter("balance_target_min_per_bin", 8)
        self.declare_parameter("balance_upsample_sparse_bins", True)
        self.declare_parameter("balance_merge_strategy", "union_unique")
        self.declare_parameter("save_balanced_component_datasets", True)
        self.declare_parameter("balanced_linear_dataset_name", "dataset_rywak_linear_balanced.npz")
        self.declare_parameter("balanced_angular_dataset_name", "dataset_rywak_angular_balanced.npz")
        self.declare_parameter("trajectory_mode", "any")
        self.declare_parameter("trajectory_cell_size_m", 0.20)
        self.declare_parameter("cycle_min_repeat_hits", 1)
        self.declare_parameter("stop_on_planned_path_done", False)
        self.declare_parameter("planned_path_done_topic", "/planned_path_done")
        self.declare_parameter("planned_path_done_min_elapsed_sec", 0.0)

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None

        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        ensure_dir(self.out_dir)

        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.max_samples_enabled = self.max_samples > 0

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self.interpolate_odom = bool(self.get_parameter("interpolate_odom").value)
        self.sync_pair_gap_sec = float(self.get_parameter("sync_pair_gap_sec").value)
        self.delta_scan_clip = float(self.get_parameter("delta_scan_clip").value)
        self.min_sample_dist = float(self.get_parameter("min_sample_dist").value)
        self.min_sample_dyaw = float(self.get_parameter("min_sample_dyaw").value)
        self.min_sample_dt_sec = float(self.get_parameter("min_sample_dt_sec").value)
        self.min_delta_scan_rms = float(self.get_parameter("min_delta_scan_rms").value)
        self.sample_filter_mode = parse_filter_mode(str(self.get_parameter("sample_filter_mode").value))
        self.balance_histograms = bool(self.get_parameter("balance_histograms").value)
        self.balance_bins = int(self.get_parameter("balance_bins").value)
        self.balance_linear_use_abs = bool(self.get_parameter("balance_linear_use_abs").value)
        self.balance_angular_use_abs = bool(self.get_parameter("balance_angular_use_abs").value)
        self.balance_linear_hist_min_mps = float(self.get_parameter("balance_linear_hist_min_mps").value)
        self.balance_linear_hist_max_mps = float(self.get_parameter("balance_linear_hist_max_mps").value)
        self.balance_angular_hist_min_radps = float(self.get_parameter("balance_angular_hist_min_radps").value)
        self.balance_angular_hist_max_radps = float(self.get_parameter("balance_angular_hist_max_radps").value)
        self.balance_target_quantile = float(self.get_parameter("balance_target_quantile").value)
        self.balance_target_min_per_bin = max(1, int(self.get_parameter("balance_target_min_per_bin").value))
        self.balance_upsample_sparse_bins = bool(self.get_parameter("balance_upsample_sparse_bins").value)
        self.balance_merge_strategy = _normalize_balance_merge_strategy(
            str(self.get_parameter("balance_merge_strategy").value)
        )
        self.save_balanced_component_datasets = bool(
            self.get_parameter("save_balanced_component_datasets").value
        )
        self.balanced_linear_path = os.path.join(
            self.out_dir,
            str(self.get_parameter("balanced_linear_dataset_name").value),
        )
        self.balanced_angular_path = os.path.join(
            self.out_dir,
            str(self.get_parameter("balanced_angular_dataset_name").value),
        )
        self.trajectory_mode = _normalize_trajectory_mode(str(self.get_parameter("trajectory_mode").value))
        self.trajectory_cell_size_m = max(0.005, float(self.get_parameter("trajectory_cell_size_m").value))
        self.cycle_min_repeat_hits = max(0, int(self.get_parameter("cycle_min_repeat_hits").value))
        self.stop_on_planned_path_done = bool(self.get_parameter("stop_on_planned_path_done").value)
        self.planned_path_done_topic = str(self.get_parameter("planned_path_done_topic").value)
        self.planned_path_done_min_elapsed_sec = float(
            self.get_parameter("planned_path_done_min_elapsed_sec").value
        )

        # (stamp_sec, x, y, theta, v, w)
        self.odom_buf = deque(maxlen=2000)
        self.odom_count = 0
        self.odom_miss_count = 0
        self.odom_interp_count = 0
        self.odom_nearest_count = 0
        self.sample_accept_count = 0
        self.sample_filter_reject_count = 0
        self.trajectory_reject_count = 0
        self.trajectory_repeat_hits = 0
        self.trajectory_visited_cells = set()
        self.prev_scan = None
        self.prev_pose = None
        self.prev_scan_time_sec = None
        self.theta_hist = deque(maxlen=3)

        self.X = []
        self.Y = []
        self.P = []

        self.t0 = None
        self.topics_ready = False
        self.is_finishing = False
        self.experiment_start = time.time()
        self.stop_reason = "duration_sec_reached_or_max_samples"

        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.sub_path_done = None
        if self.stop_on_planned_path_done:
            self.sub_path_done = self.create_subscription(
                Bool, self.planned_path_done_topic, self.on_planned_path_done, 10
            )

        self.wait_timer = self.create_timer(1.0, self.wait_for_topics)
        self.timer = self.create_timer(0.5, self.check_done)

        if self.write_experiment_metadata:
            self.exp_logger.start_dataset_collection(
                seed=self.seed,
                duration_sec=self.duration_sec,
                max_samples=self.max_samples,
                scan_topic=self.scan_topic,
                odom_topic=self.odom_topic,
                gt_topic="(unused)",
            )

        self.get_logger().info(f"[Rywak] out_dir={self.out_dir}")
        self.get_logger().info(
            f"[Rywak] scan={self.scan_topic}, odom={self.odom_topic}, "
            f"interp={self.interpolate_odom}, tol={self.sync_tolerance_sec}, "
            f"gap={self.sync_pair_gap_sec}, delta_clip={self.delta_scan_clip}, "
            f"min_dist={self.min_sample_dist}, min_dyaw={self.min_sample_dyaw:.3f}, "
            f"min_dt={self.min_sample_dt_sec:.3f}, min_scan_rms={self.min_delta_scan_rms}, "
            f"filter_mode={self.sample_filter_mode}, "
            f"balance_hist={self.balance_histograms}, bins={self.balance_bins}, "
            f"balance_range_v=[{self.balance_linear_hist_min_mps:.2f},{self.balance_linear_hist_max_mps:.2f}]m/s, "
            f"balance_range_w=[{self.balance_angular_hist_min_radps:.2f},{self.balance_angular_hist_max_radps:.2f}]rad/s, "
            f"balance_q={self.balance_target_quantile:.2f}, "
            f"balance_min_bin={self.balance_target_min_per_bin}, "
            f"balance_upsample={self.balance_upsample_sparse_bins}, "
            f"balance_merge={self.balance_merge_strategy}, "
            f"trajectory_mode={self.trajectory_mode}, cell={self.trajectory_cell_size_m:.2f}m"
        )
        if self.max_samples_enabled:
            self.get_logger().info(f"[Rywak] stop mode: time or max_samples={self.max_samples}")
        else:
            self.get_logger().info("[Rywak] stop mode: time-driven (max_samples disabled)")

    def on_odom(self, msg: Odometry):
        self.odom_count += 1
        t = _stamp_to_sec(msg.header.stamp)
        x, y, th = xytheta_from_odom(msg)
        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)
        self.odom_buf.append((t, x, y, th, v, w))

    def _nearest_odom(self, t_scan: float):
        if not self.odom_buf:
            return None

        while len(self.odom_buf) > 2 and self.odom_buf[1][0] < (t_scan - 1.0):
            self.odom_buf.popleft()

        t_best, x_best, y_best, th_best, v_best, w_best = min(
            self.odom_buf, key=lambda x: abs(x[0] - t_scan)
        )
        if abs(t_best - t_scan) > self.sync_tolerance_sec:
            return None
        return x_best, y_best, th_best, v_best, w_best

    def _interpolated_odom(self, t_scan: float):
        if len(self.odom_buf) < 2:
            return None

        prev = None
        for cur in self.odom_buf:
            if cur[0] < t_scan:
                prev = cur
                continue

            if prev is None:
                return None

            t0, x0, y0, th0, v0, w0 = prev
            t1, x1, y1, th1, v1, w1 = cur
            gap = float(t1 - t0)
            if gap < 1e-6:
                if abs(t_scan - t0) <= self.sync_tolerance_sec:
                    return float(x0), float(y0), float(th0), float(v0), float(w0)
                return None

            if gap > self.sync_pair_gap_sec:
                return None
            if (t_scan - t0) > self.sync_tolerance_sec:
                return None
            if (t1 - t_scan) > self.sync_tolerance_sec:
                return None

            alpha = (t_scan - t0) / gap
            x = float(x0 + alpha * (x1 - x0))
            y = float(y0 + alpha * (y1 - y0))
            th = _interp_angle(th0, th1, alpha)
            v = float(v0 + alpha * (v1 - v0))
            w = float(w0 + alpha * (w1 - w0))
            return x, y, th, v, w

        return None

    def _odom_at(self, t_scan: float):
        if self.interpolate_odom:
            interp = self._interpolated_odom(t_scan)
            if interp is not None:
                self.odom_interp_count += 1
                return interp

        nearest = self._nearest_odom(t_scan)
        if nearest is not None:
            self.odom_nearest_count += 1
        return nearest

    def wait_for_topics(self):
        if self.topics_ready:
            return
        if len(self.odom_buf) > 0:
            self.topics_ready = True
            self.t0 = self.get_clock().now()
            self.get_logger().info("[Rywak][FAZA 1] DATASET START")
        else:
            self.get_logger().info("[Rywak] Waiting for odom...")

    def _pose_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cell_size = max(1e-6, self.trajectory_cell_size_m)
        return (
            int(math.floor(float(x) / cell_size)),
            int(math.floor(float(y) / cell_size)),
        )

    def on_scan(self, msg: LaserScan):
        if not self.topics_ready:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        t_scan = _stamp_to_sec(msg.header.stamp)
        odom_match = self._odom_at(t_scan)
        if odom_match is None:
            self.odom_miss_count += 1
            return
        x, y, th, v, w = odom_match
        curr_pose = (float(x), float(y), float(th))

        if self.prev_scan is None:
            self.prev_scan = scan
            self.prev_pose = curr_pose
            self.prev_scan_time_sec = t_scan
            self.theta_hist.append(th)
            return

        scan_rms = scan_delta_rms(self.prev_scan, scan)
        keep_sample, _delta = passes_motion_filter(
            self.prev_pose,
            curr_pose,
            dt_sec=None if self.prev_scan_time_sec is None else max(0.0, float(t_scan - self.prev_scan_time_sec)),
            min_translation=self.min_sample_dist,
            min_rotation=self.min_sample_dyaw,
            min_time_gap_sec=self.min_sample_dt_sec,
            min_scan_delta_rms=self.min_delta_scan_rms,
            scan_delta_rms_value=scan_rms,
            mode=self.sample_filter_mode,
        )
        if not keep_sample:
            self.sample_filter_reject_count += 1
            return

        curr_cell = self._pose_to_cell(curr_pose[0], curr_pose[1])
        repeated_cell = curr_cell in self.trajectory_visited_cells
        if repeated_cell:
            self.trajectory_repeat_hits += 1
        self.trajectory_visited_cells.add(curr_cell)

        if self.trajectory_mode == "no_cycle" and repeated_cell:
            self.trajectory_reject_count += 1
            return

        delta_scan = (scan - self.prev_scan).astype(np.float32)
        if self.delta_scan_clip > 0.0:
            delta_scan = np.clip(delta_scan, -self.delta_scan_clip, self.delta_scan_clip).astype(np.float32)
        self.theta_hist.append(th)
        if len(self.theta_hist) < 3:
            self.prev_scan = scan
            self.prev_pose = curr_pose
            self.prev_scan_time_sec = t_scan
            self.sample_accept_count += 1
            return

        d_theta1 = wrap(float(self.theta_hist[-1] - self.theta_hist[-2]))
        d_theta2 = wrap(float(self.theta_hist[-2] - self.theta_hist[-3]))

        # features: [d_theta1, d_theta2, delta_scan(360)] => 362
        x = np.concatenate(
            [np.asarray([d_theta1, d_theta2], dtype=np.float32), delta_scan], axis=0
        ).astype(np.float32)
        y = np.asarray([v, w], dtype=np.float32)
        p = np.asarray(
            [
                self.prev_pose[0],
                self.prev_pose[1],
                self.prev_pose[2],
                curr_pose[0],
                curr_pose[1],
                curr_pose[2],
            ],
            dtype=np.float32,
        )

        self.X.append(x)
        self.Y.append(y)
        self.P.append(p)

        self.prev_scan = scan
        self.prev_pose = curr_pose
        self.prev_scan_time_sec = t_scan
        self.sample_accept_count += 1

        if self.max_samples_enabled and len(self.Y) >= self.max_samples:
            self.stop_reason = "max_samples_reached"
            self.save_and_exit()

    def check_done(self):
        if self.is_finishing:
            return
        if self.t0 is None:
            return
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self.stop_reason = "duration_sec_reached"
            self.save_and_exit()

    def on_planned_path_done(self, msg: Bool):
        if self.is_finishing or not bool(msg.data):
            return
        if self.t0 is None:
            return
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if elapsed < self.planned_path_done_min_elapsed_sec:
            return
        self.stop_reason = "planned_path_completed"
        self.get_logger().info(
            f"[Rywak] planned path completed; stopping at t={elapsed:.2f}s"
        )
        self.save_and_exit()

    def save_and_exit(self):
        if self.is_finishing:
            return
        self.is_finishing = True
        if len(self.Y) == 0:
            self.get_logger().error("[Rywak] No samples collected.")
            rclpy.shutdown()
            return

        X = np.stack(self.X).astype(np.float32)  # (N,362)
        Y = np.stack(self.Y).astype(np.float32)  # (N,2)
        P = np.stack(self.P).astype(np.float32)  # (N,6) = prev(x,y,th), curr(x,y,th)
        raw_count = int(Y.shape[0])
        if self.trajectory_mode == "cycle" and self.trajectory_repeat_hits < self.cycle_min_repeat_hits:
            self.get_logger().error(
                "[Rywak] trajectory_mode=cycle requires repeated positions, "
                f"but only {self.trajectory_repeat_hits} repeats were detected "
                f"(min={self.cycle_min_repeat_hits})."
            )
            rclpy.shutdown()
            return

        linear_hist_stats = {}
        angular_hist_stats = {}
        selected_linear_idx = np.arange(raw_count, dtype=np.int64)
        selected_angular_idx = np.arange(raw_count, dtype=np.int64)
        selected_merged_idx = np.arange(raw_count, dtype=np.int64)

        if self.balance_histograms and raw_count > 0:
            selected_linear_idx, linear_hist_stats = uniform_histogram_sample_indices(
                Y[:, 0],
                bins=self.balance_bins,
                seed=self.seed + 101,
                use_abs=self.balance_linear_use_abs,
                hist_min=self.balance_linear_hist_min_mps,
                hist_max=self.balance_linear_hist_max_mps,
                target_quantile=self.balance_target_quantile,
                target_min_per_bin=self.balance_target_min_per_bin,
                upsample=self.balance_upsample_sparse_bins,
            )
            selected_angular_idx, angular_hist_stats = uniform_histogram_sample_indices(
                Y[:, 1],
                bins=self.balance_bins,
                seed=self.seed + 202,
                use_abs=self.balance_angular_use_abs,
                hist_min=self.balance_angular_hist_min_radps,
                hist_max=self.balance_angular_hist_max_radps,
                target_quantile=self.balance_target_quantile,
                target_min_per_bin=self.balance_target_min_per_bin,
                upsample=self.balance_upsample_sparse_bins,
            )
            selected_merged_idx = merge_balanced_indices(
                selected_linear_idx,
                selected_angular_idx,
                strategy=self.balance_merge_strategy,
            )
            if selected_merged_idx.size == 0:
                selected_merged_idx = np.arange(raw_count, dtype=np.int64)

            if self.save_balanced_component_datasets:
                try:
                    linear_buffer = io.BytesIO()
                    np.savez_compressed(
                        linear_buffer,
                        X=X[selected_linear_idx],
                        Y=Y[selected_linear_idx],
                        P=P[selected_linear_idx],
                        meta={
                            "source_dataset": np.asarray([self.dataset_path], dtype=object),
                            "component": np.asarray(["linear_velocity"], dtype=object),
                            "n_source": np.int64(raw_count),
                            "n_selected": np.int64(selected_linear_idx.size),
                            "balance_bins": np.int64(self.balance_bins),
                            "balance_use_abs": np.int64(1 if self.balance_linear_use_abs else 0),
                            "balance_hist_min_mps": np.float32(linear_hist_stats.get("hist_min", np.nan)),
                            "balance_hist_max_mps": np.float32(linear_hist_stats.get("hist_max", np.nan)),
                            "balance_target_per_bin": np.int64(linear_hist_stats.get("target_per_bin", 0)),
                            "balance_target_quantile": np.float32(
                                linear_hist_stats.get("target_quantile", self.balance_target_quantile)
                            ),
                            "balance_target_min_per_bin": np.int64(
                                linear_hist_stats.get("target_min_per_bin", self.balance_target_min_per_bin)
                            ),
                            "balance_upsample_sparse_bins": np.int64(
                                1 if linear_hist_stats.get("upsample", self.balance_upsample_sparse_bins) else 0
                            ),
                            "balance_counts_per_bin": np.asarray(
                                linear_hist_stats.get("counts_per_bin", np.zeros((0,), dtype=np.int64)),
                                dtype=np.int64,
                            ),
                            "balance_selected_counts_per_bin": np.asarray(
                                linear_hist_stats.get("selected_counts_per_bin", np.zeros((0,), dtype=np.int64)),
                                dtype=np.int64,
                            ),
                            "component_unit": np.asarray(["m/s"], dtype=object),
                            "pose_pair_key_included": np.int64(1),
                        },
                    )
                    atomic_write_bytes(self.balanced_linear_path, linear_buffer.getvalue())

                    angular_buffer = io.BytesIO()
                    np.savez_compressed(
                        angular_buffer,
                        X=X[selected_angular_idx],
                        Y=Y[selected_angular_idx],
                        P=P[selected_angular_idx],
                        meta={
                            "source_dataset": np.asarray([self.dataset_path], dtype=object),
                            "component": np.asarray(["angular_velocity"], dtype=object),
                            "n_source": np.int64(raw_count),
                            "n_selected": np.int64(selected_angular_idx.size),
                            "balance_bins": np.int64(self.balance_bins),
                            "balance_use_abs": np.int64(1 if self.balance_angular_use_abs else 0),
                            "balance_hist_min_radps": np.float32(angular_hist_stats.get("hist_min", np.nan)),
                            "balance_hist_max_radps": np.float32(angular_hist_stats.get("hist_max", np.nan)),
                            "balance_target_per_bin": np.int64(angular_hist_stats.get("target_per_bin", 0)),
                            "balance_target_quantile": np.float32(
                                angular_hist_stats.get("target_quantile", self.balance_target_quantile)
                            ),
                            "balance_target_min_per_bin": np.int64(
                                angular_hist_stats.get("target_min_per_bin", self.balance_target_min_per_bin)
                            ),
                            "balance_upsample_sparse_bins": np.int64(
                                1 if angular_hist_stats.get("upsample", self.balance_upsample_sparse_bins) else 0
                            ),
                            "balance_counts_per_bin": np.asarray(
                                angular_hist_stats.get("counts_per_bin", np.zeros((0,), dtype=np.int64)),
                                dtype=np.int64,
                            ),
                            "balance_selected_counts_per_bin": np.asarray(
                                angular_hist_stats.get("selected_counts_per_bin", np.zeros((0,), dtype=np.int64)),
                                dtype=np.int64,
                            ),
                            "component_unit": np.asarray(["rad/s"], dtype=object),
                            "pose_pair_key_included": np.int64(1),
                        },
                    )
                    atomic_write_bytes(self.balanced_angular_path, angular_buffer.getvalue())
                except Exception as exc:
                    self.get_logger().warn(f"[Rywak] Failed to save balanced component datasets: {exc}")

            X = X[selected_merged_idx]
            Y = Y[selected_merged_idx]
            P = P[selected_merged_idx]

        linear_speed_abs = np.abs(Y[:, 0]).astype(np.float32)
        angular_speed_abs = np.abs(Y[:, 1]).astype(np.float32)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(Y.shape[0]),
            "in_dim": np.int64(X.shape[1]),
            "out_dim": np.int64(2),
            "stop_reason": np.asarray([self.stop_reason], dtype=object),
            "feature_mode": np.asarray(["dtheta_dtheta_deltascan"], dtype=object),
            "sync_tolerance_sec": np.float32(self.sync_tolerance_sec),
            "interpolate_odom": np.int64(1 if self.interpolate_odom else 0),
            "sync_pair_gap_sec": np.float32(self.sync_pair_gap_sec),
            "delta_scan_clip": np.float32(self.delta_scan_clip),
            "min_sample_dist": np.float32(self.min_sample_dist),
            "min_sample_dyaw": np.float32(self.min_sample_dyaw),
            "min_sample_dt_sec": np.float32(self.min_sample_dt_sec),
            "min_delta_scan_rms": np.float32(self.min_delta_scan_rms),
            "sample_filter_mode": np.asarray([self.sample_filter_mode], dtype=object),
            "odom_sync_miss_count": np.int64(self.odom_miss_count),
            "odom_sync_interp_count": np.int64(self.odom_interp_count),
            "odom_sync_nearest_count": np.int64(self.odom_nearest_count),
            "sample_accept_count": np.int64(self.sample_accept_count),
            "sample_filter_reject_count": np.int64(self.sample_filter_reject_count),
            "trajectory_mode": np.asarray([self.trajectory_mode], dtype=object),
            "trajectory_cell_size_m": np.float32(self.trajectory_cell_size_m),
            "trajectory_repeat_hits": np.int64(self.trajectory_repeat_hits),
            "trajectory_reject_count": np.int64(self.trajectory_reject_count),
            "trajectory_unique_cells": np.int64(len(self.trajectory_visited_cells)),
            "cycle_min_repeat_hits": np.int64(self.cycle_min_repeat_hits),
            "balance_histograms_enabled": np.int64(1 if self.balance_histograms else 0),
            "balance_bins": np.int64(self.balance_bins),
            "balance_linear_use_abs": np.int64(1 if self.balance_linear_use_abs else 0),
            "balance_angular_use_abs": np.int64(1 if self.balance_angular_use_abs else 0),
            "balance_target_quantile": np.float32(self.balance_target_quantile),
            "balance_target_min_per_bin": np.int64(self.balance_target_min_per_bin),
            "balance_upsample_sparse_bins": np.int64(1 if self.balance_upsample_sparse_bins else 0),
            "balance_merge_strategy": np.asarray([self.balance_merge_strategy], dtype=object),
            "balance_linear_hist_min_mps": np.float32(linear_hist_stats.get("hist_min", self.balance_linear_hist_min_mps)),
            "balance_linear_hist_max_mps": np.float32(linear_hist_stats.get("hist_max", self.balance_linear_hist_max_mps)),
            "balance_angular_hist_min_radps": np.float32(
                angular_hist_stats.get("hist_min", self.balance_angular_hist_min_radps)
            ),
            "balance_angular_hist_max_radps": np.float32(
                angular_hist_stats.get("hist_max", self.balance_angular_hist_max_radps)
            ),
            "balance_raw_sample_count": np.int64(raw_count),
            "balance_linear_selected_count": np.int64(selected_linear_idx.size),
            "balance_angular_selected_count": np.int64(selected_angular_idx.size),
            "balance_merged_selected_count": np.int64(selected_merged_idx.size),
            "balance_linear_bins_non_empty": np.int64(linear_hist_stats.get("bins_non_empty", 0)),
            "balance_angular_bins_non_empty": np.int64(angular_hist_stats.get("bins_non_empty", 0)),
            "balance_linear_target_per_bin": np.int64(linear_hist_stats.get("target_per_bin", 0)),
            "balance_angular_target_per_bin": np.int64(angular_hist_stats.get("target_per_bin", 0)),
            "balance_linear_counts_per_bin": np.asarray(
                linear_hist_stats.get("counts_per_bin", np.zeros((0,), dtype=np.int64)),
                dtype=np.int64,
            ),
            "balance_angular_counts_per_bin": np.asarray(
                angular_hist_stats.get("counts_per_bin", np.zeros((0,), dtype=np.int64)),
                dtype=np.int64,
            ),
            "balance_linear_selected_counts_per_bin": np.asarray(
                linear_hist_stats.get("selected_counts_per_bin", np.zeros((0,), dtype=np.int64)),
                dtype=np.int64,
            ),
            "balance_angular_selected_counts_per_bin": np.asarray(
                angular_hist_stats.get("selected_counts_per_bin", np.zeros((0,), dtype=np.int64)),
                dtype=np.int64,
            ),
            "balance_linear_n_in_range": np.int64(linear_hist_stats.get("n_in_range", 0)),
            "balance_linear_n_out_of_range": np.int64(linear_hist_stats.get("n_out_of_range", 0)),
            "balance_angular_n_in_range": np.int64(angular_hist_stats.get("n_in_range", 0)),
            "balance_angular_n_out_of_range": np.int64(angular_hist_stats.get("n_out_of_range", 0)),
            "balanced_linear_dataset_path": np.asarray([self.balanced_linear_path], dtype=object),
            "balanced_angular_dataset_path": np.asarray([self.balanced_angular_path], dtype=object),
            "balance_linear_unit": np.asarray(["m/s"], dtype=object),
            "balance_angular_unit": np.asarray(["rad/s"], dtype=object),
            "label_linear_speed_abs_mean_mps": np.float32(np.mean(linear_speed_abs)),
            "label_linear_speed_abs_p95_mps": np.float32(np.percentile(linear_speed_abs, 95)),
            "label_linear_speed_abs_max_mps": np.float32(np.max(linear_speed_abs)),
            "label_angular_speed_abs_mean_radps": np.float32(np.mean(angular_speed_abs)),
            "label_angular_speed_abs_p95_radps": np.float32(np.percentile(angular_speed_abs, 95)),
            "label_angular_speed_abs_max_radps": np.float32(np.max(angular_speed_abs)),
            "pose_pair_key_included": np.int64(1),
            "pose_pair_dim": np.int64(P.shape[1]),
        }

        ensure_dir(self.out_dir)
        try:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X=X, Y=Y, P=P, meta=meta)
            atomic_write_bytes(self.dataset_path, buffer.getvalue())
        except Exception as e:
            self.get_logger().error(f"[Rywak] Failed to save dataset: {e}")
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
                scan_dim=int(X.shape[1]),
                actual_duration_sec=float(actual_duration),
                file_path=self.dataset_path,
            )

        self.get_logger().info(
            f"[Rywak] Saved dataset: {self.dataset_path} "
            f"(n={Y.shape[0]}, raw_n={raw_count}, samples_ok={self.sample_accept_count}, "
            f"samples_filtered={self.sample_filter_reject_count}, "
            f"trajectory_repeats={self.trajectory_repeat_hits}, "
            f"trajectory_rejects={self.trajectory_reject_count}, "
            f"odom_sync_interp={self.odom_interp_count}, "
            f"odom_sync_nearest={self.odom_nearest_count}, odom_sync_miss={self.odom_miss_count})"
        )
        rclpy.shutdown()


def main():
    rclpy.init()
    node = DatasetRecorderRywak()
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
