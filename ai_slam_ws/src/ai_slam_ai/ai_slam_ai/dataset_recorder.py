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
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from .common import atomic_write_bytes, ensure_dir, seed_all, wrap, xytheta_from_odom, xytheta_from_pose_stamped
from .experiment_logger import ExperimentLogger


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _resample_to_360(ranges: np.ndarray) -> np.ndarray | None:
    n = int(ranges.size)
    if n == 360:
        return ranges.astype(np.float32)
    if n < 10:
        return None
    x_old = np.linspace(-math.pi, math.pi, n, endpoint=False)
    x_new = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    return np.interp(x_new, x_old, ranges).astype(np.float32)


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


class DatasetRecorder(Node):
    def __init__(self):
        super().__init__("dataset_recorder")
        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("max_samples", 5000)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("dataset_name", "dataset.npz")
        self.declare_parameter("sync_tolerance_sec", 0.08)
        self.declare_parameter("sync_pair_gap_sec", 0.2)
        self.declare_parameter("interpolate_odom", True)
        self.declare_parameter("interpolate_gt", True)
        self.declare_parameter("stop_on_planned_path_done", False)
        self.declare_parameter("planned_path_done_topic", "/planned_path_done")
        self.declare_parameter("planned_path_done_min_elapsed_sec", 0.0)

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None

        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()

        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.max_samples_enabled = self.max_samples > 0
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))
        self.sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self.sync_pair_gap_sec = float(self.get_parameter("sync_pair_gap_sec").value)
        self.interpolate_odom = bool(self.get_parameter("interpolate_odom").value)
        self.interpolate_gt = bool(self.get_parameter("interpolate_gt").value)
        self.stop_on_planned_path_done = bool(self.get_parameter("stop_on_planned_path_done").value)
        self.planned_path_done_topic = str(self.get_parameter("planned_path_done_topic").value)
        self.planned_path_done_min_elapsed_sec = float(
            self.get_parameter("planned_path_done_min_elapsed_sec").value
        )

        ensure_dir(self.out_dir)
        self.get_logger().info(f"Output directory: {self.out_dir}")
        self.get_logger().info(f"Experiment ID: {self.exp_logger.experiment_id}")

        self.odom_buf = deque(maxlen=2000)
        self.gt_buf = deque(maxlen=2000)
        self.odom_count = 0
        self.gt_count = 0
        self.scan_count = 0
        self.scan_rx_count = 0
        self.odom_sync_miss_count = 0
        self.gt_sync_miss_count = 0
        self.odom_sync_interp_count = 0
        self.gt_sync_interp_count = 0
        self.odom_sync_nearest_count = 0
        self.gt_sync_nearest_count = 0
        self.pending_drop_count = 0
        self.pending_scans = deque()
        self.max_pending_scans = 4000

        self.x_scan = []
        self.x_odom = []
        self.y = []
        self.t0 = None
        self.topics_ready = False
        self.experiment_start = time.time()
        self.is_finishing = False
        self.stop_reason = "duration_sec_reached_or_max_samples"

        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data
        )
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, qos_profile_sensor_data
        )
        self.sub_gt = self.create_subscription(
            PoseStamped, self.gt_topic, self.on_gt, 50
        )
        self.sub_path_done = None
        if self.stop_on_planned_path_done:
            self.sub_path_done = self.create_subscription(
                Bool, self.planned_path_done_topic, self.on_planned_path_done, 10
            )

        self.get_logger().info(
            f"Subscriptions created: scan={self.scan_topic} (BEST_EFFORT QoS), "
            f"odom={self.odom_topic}, gt={self.gt_topic}"
        )
        self.get_logger().info(
            f"Sync: tol={self.sync_tolerance_sec:.3f}s, gap={self.sync_pair_gap_sec:.3f}s, "
            f"interp_odom={self.interpolate_odom}, interp_gt={self.interpolate_gt}"
        )
        if self.max_samples_enabled:
            self.get_logger().info(f"Dataset stop mode: time or max_samples={self.max_samples}")
        else:
            self.get_logger().info("Dataset stop mode: time-driven (max_samples disabled)")

        self.timer = self.create_timer(0.5, self.check_done)
        self.wait_timer = self.create_timer(1.0, self.wait_for_topics)

        self.exp_logger.start_dataset_collection(
            seed=self.seed,
            duration_sec=self.duration_sec,
            max_samples=self.max_samples,
            scan_topic=self.scan_topic,
            odom_topic=self.odom_topic,
            gt_topic=self.gt_topic,
        )

    def on_odom(self, msg: Odometry):
        self.odom_count += 1
        t = _stamp_to_sec(msg.header.stamp)
        self.odom_buf.append((t, *xytheta_from_odom(msg)))
        self._process_pending_scans()

    def on_gt(self, msg: PoseStamped):
        self.gt_count += 1
        t = _stamp_to_sec(msg.header.stamp)
        self.gt_buf.append((t, *xytheta_from_pose_stamped(msg)))
        self._process_pending_scans()

    def _trim_pose_buffer(self, buf, t_ref: float):
        while len(buf) > 2 and buf[1][0] < (t_ref - 1.0):
            buf.popleft()

    def _nearest_pose(self, buf, t_target: float):
        if not buf:
            return None
        self._trim_pose_buffer(buf, t_target)
        best = min(buf, key=lambda item: abs(item[0] - t_target))
        if abs(best[0] - t_target) > self.sync_tolerance_sec:
            return None
        return float(best[1]), float(best[2]), float(best[3])

    def _interpolated_pose(self, buf, t_target: float):
        if len(buf) < 2:
            return None

        prev = None
        for cur in buf:
            if cur[0] < t_target:
                prev = cur
                continue

            if prev is None:
                return None

            t0, x0, y0, th0 = prev
            t1, x1, y1, th1 = cur
            gap = float(t1 - t0)
            if gap < 1e-6:
                if abs(t_target - t0) <= self.sync_tolerance_sec:
                    return float(x0), float(y0), float(th0)
                return None
            if gap > self.sync_pair_gap_sec:
                return None
            if t_target < t0 or t_target > t1:
                return None

            alpha = (t_target - t0) / gap
            x = float(x0 + alpha * (x1 - x0))
            y = float(y0 + alpha * (y1 - y0))
            th = _interp_angle(th0, th1, alpha)
            return x, y, th

        return None

    def _pose_at(self, buf, t_target: float, interpolate: bool):
        if interpolate:
            pose = self._interpolated_pose(buf, t_target)
            if pose is not None:
                return pose, "interp"
        pose = self._nearest_pose(buf, t_target)
        if pose is not None:
            return pose, "nearest"
        return None, None

    @staticmethod
    def _latest_pose_time(buf):
        if not buf:
            return None
        return float(buf[-1][0])

    @staticmethod
    def _earliest_pose_time(buf):
        if not buf:
            return None
        return float(buf[0][0])

    def _can_attempt_pose(self, buf, t_target: float, flush: bool):
        if not buf:
            return False
        if flush:
            return True

        earliest_t = self._earliest_pose_time(buf)
        latest_t = self._latest_pose_time(buf)
        if earliest_t is not None and earliest_t > t_target:
            return True
        if latest_t is not None and latest_t >= t_target:
            return True

        best = min(buf, key=lambda item: abs(item[0] - t_target))
        return abs(best[0] - t_target) <= self.sync_tolerance_sec

    def _append_pending_scan(self, t_scan: float, ranges: np.ndarray):
        self.pending_scans.append((t_scan, ranges.copy()))
        while len(self.pending_scans) > self.max_pending_scans:
            self.pending_scans.popleft()
            self.pending_drop_count += 1

    def _process_pending_scans(self, flush: bool = False):
        while self.pending_scans:
            t_scan, ranges = self.pending_scans[0]
            if not self._can_attempt_pose(self.odom_buf, t_scan, flush):
                break
            if not self._can_attempt_pose(self.gt_buf, t_scan, flush):
                break

            self.pending_scans.popleft()

            odom_pose, odom_mode = self._pose_at(self.odom_buf, t_scan, interpolate=self.interpolate_odom)
            if odom_pose is None:
                self.odom_sync_miss_count += 1
                continue

            gt_pose, gt_mode = self._pose_at(self.gt_buf, t_scan, interpolate=self.interpolate_gt)
            if gt_pose is None:
                self.gt_sync_miss_count += 1
                continue

            if odom_mode == "interp":
                self.odom_sync_interp_count += 1
            else:
                self.odom_sync_nearest_count += 1
            if gt_mode == "interp":
                self.gt_sync_interp_count += 1
            else:
                self.gt_sync_nearest_count += 1

            ox, oy, oth = odom_pose
            gx, gy, gth = gt_pose
            dx = gx - ox
            dy = gy - oy
            dth = wrap(gth - oth)

            self.x_scan.append(ranges)
            self.x_odom.append([ox, oy, oth])
            self.y.append([dx, dy, dth])
            self.scan_count += 1

            if self.max_samples_enabled and len(self.y) >= self.max_samples and not self.is_finishing:
                self.stop_reason = "max_samples_reached"
                self.save_and_exit()
                return

    def wait_for_topics(self):
        if self.topics_ready:
            return
        if len(self.odom_buf) > 0 and len(self.gt_buf) > 0:
            elapsed_exp = time.time() - self.experiment_start
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"[FAZA 1] ZBIERANIE DANYCH - START (t={elapsed_exp:.0f}s)")
            self.get_logger().info(
                f"Planowany czas: {self.duration_sec}s | Zakończenie: ~t={elapsed_exp + self.duration_sec:.0f}s"
            )
            self.get_logger().info("=" * 60)
            self.topics_ready = True
            self.t0 = self.get_clock().now()
        else:
            self.get_logger().info(
                "Waiting for topics: odom={} (n={}), gt={} (n={})".format(
                    len(self.odom_buf) > 0,
                    self.odom_count,
                    len(self.gt_buf) > 0,
                    self.gt_count,
                )
            )

    def on_scan(self, msg: LaserScan):
        if not self.topics_ready:
            return

        ranges = _sanitize_scan(msg)
        if ranges is None:
            self.get_logger().warn(f"Scan size {len(msg.ranges)} too small, skipping")
            return

        self.scan_rx_count += 1
        t_scan = _stamp_to_sec(msg.header.stamp)
        self._append_pending_scan(t_scan, ranges)
        self._process_pending_scans()

    def check_done(self):
        if self.t0 is None:
            return
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if self.topics_ready and elapsed > 5.0 and self.scan_count == 0:
            if int(elapsed) % 10 == 0:
                self.get_logger().warn(
                    f"No synced scans yet! elapsed={elapsed:.1f}s, collected={len(self.y)}, "
                    f"odom_miss={self.odom_sync_miss_count}, gt_miss={self.gt_sync_miss_count}, "
                    f"pending={len(self.pending_scans)}"
                )
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
            f"[DatasetRecorder] planned path completed; stopping at t={elapsed:.2f}s"
        )
        self.save_and_exit()

    def save_and_exit(self):
        if self.is_finishing:
            return
        self.is_finishing = True
        self._process_pending_scans(flush=True)

        elapsed_str = "N/A" if self.t0 is None else f"{(self.get_clock().now() - self.t0).nanoseconds * 1e-9:.1f}s"
        if len(self.y) == 0:
            self.get_logger().error(
                f"No samples collected. Scans received: {self.scan_rx_count}, elapsed: {elapsed_str}, "
                f"odom_miss={self.odom_sync_miss_count}, gt_miss={self.gt_sync_miss_count}, "
                f"pending_drops={self.pending_drop_count}"
            )
            rclpy.shutdown()
            return

        elapsed_exp = time.time() - self.experiment_start
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[FAZA 1] ZBIERANIE DANYCH - KONIEC (t={elapsed_exp:.0f}s)")
        self.get_logger().info(f"Zebrano {len(self.y)} próbek w {elapsed_str}")
        self.get_logger().info(
            "Sync stats: "
            f"odom_interp={self.odom_sync_interp_count}, odom_nearest={self.odom_sync_nearest_count}, odom_miss={self.odom_sync_miss_count}, "
            f"gt_interp={self.gt_sync_interp_count}, gt_nearest={self.gt_sync_nearest_count}, gt_miss={self.gt_sync_miss_count}, "
            f"scan_rx={self.scan_rx_count}, pending_drops={self.pending_drop_count}"
        )
        self.get_logger().info("=" * 60)

        X_scan = np.stack(self.x_scan).astype(np.float32)
        X_odom = np.asarray(self.x_odom, dtype=np.float32)
        Y = np.asarray(self.y, dtype=np.float32)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(len(Y)),
            "scan_dim": np.int64(X_scan.shape[1]),
            "sync_tolerance_sec": np.float32(self.sync_tolerance_sec),
            "sync_pair_gap_sec": np.float32(self.sync_pair_gap_sec),
            "stop_reason": np.asarray([self.stop_reason], dtype=object),
            "scan_rx": np.int64(self.scan_rx_count),
            "pending_drop_count": np.int64(self.pending_drop_count),
            "odom_sync_interp": np.int64(self.odom_sync_interp_count),
            "odom_sync_nearest": np.int64(self.odom_sync_nearest_count),
            "odom_sync_miss": np.int64(self.odom_sync_miss_count),
            "gt_sync_interp": np.int64(self.gt_sync_interp_count),
            "gt_sync_nearest": np.int64(self.gt_sync_nearest_count),
            "gt_sync_miss": np.int64(self.gt_sync_miss_count),
        }

        ensure_dir(self.out_dir)
        self.get_logger().info(f"Output dir exists: {os.path.isdir(self.out_dir)}, path: {self.out_dir}")

        try:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X_scan=X_scan, X_odom=X_odom, Y=Y, meta=meta)
            atomic_write_bytes(self.dataset_path, buffer.getvalue())

            self.get_logger().info(
                f"Dataset file created: {os.path.exists(self.dataset_path)}, "
                f"size: {os.path.getsize(self.dataset_path) if os.path.exists(self.dataset_path) else 'N/A'}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to save dataset: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            rclpy.shutdown()
            return

        if not os.path.exists(self.dataset_path):
            self.get_logger().error(f"Dataset file was not created: {self.dataset_path}")
            rclpy.shutdown()
            return

        actual_duration = (self.get_clock().now() - self.t0).nanoseconds * 1e-9 if self.t0 else 0
        self.exp_logger.end_dataset_collection(
            n_samples=len(Y),
            scan_dim=int(X_scan.shape[1]),
            actual_duration_sec=actual_duration,
            file_path=self.dataset_path,
        )

        self.get_logger().info(f"Saved dataset: {self.dataset_path} (n={len(Y)})")
        self.get_logger().info(f"Metadata saved: {os.path.join(self.out_dir, 'experiment_metadata.json')}")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = DatasetRecorder()
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
