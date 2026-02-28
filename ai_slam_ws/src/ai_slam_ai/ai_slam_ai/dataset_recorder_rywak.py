import io
import math
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from .common import seed_all, ensure_dir
from .experiment_logger import ExperimentLogger


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
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))

        self.latest_odom: Odometry | None = None
        self.prev_scan = None

        self.X = []
        self.Y = []

        self.t0 = None
        self.topics_ready = False
        self.experiment_start = time.time()

        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        self.wait_timer = self.create_timer(1.0, self.wait_for_topics)
        self.timer = self.create_timer(0.5, self.check_done)

        self.exp_logger.start_dataset_collection(
            seed=self.seed,
            duration_sec=self.duration_sec,
            max_samples=self.max_samples,
            scan_topic=self.scan_topic,
            odom_topic=self.odom_topic,
            gt_topic="(unused)",
        )

        self.get_logger().info(f"[Rywak] out_dir={self.out_dir}")
        self.get_logger().info(f"[Rywak] scan={self.scan_topic}, odom={self.odom_topic}")

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def wait_for_topics(self):
        if self.topics_ready:
            return
        if self.latest_odom is not None:
            self.topics_ready = True
            self.t0 = self.get_clock().now()
            self.get_logger().info("[Rywak][FAZA 1] DATASET START")
        else:
            self.get_logger().info("[Rywak] Waiting for odom...")

    def on_scan(self, msg: LaserScan):
        if not self.topics_ready or self.latest_odom is None:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        if self.prev_scan is None:
            self.prev_scan = scan
            return

        diff = (scan - self.prev_scan).astype(np.float32)

        # features: [scan(360), diff(360)] => 720
        x = np.concatenate([scan, diff], axis=0).astype(np.float32)

        v = float(self.latest_odom.twist.twist.linear.x)
        w = float(self.latest_odom.twist.twist.angular.z)
        y = np.asarray([v, w], dtype=np.float32)

        self.X.append(x)
        self.Y.append(y)

        self.prev_scan = scan

        if len(self.Y) >= self.max_samples:
            self.save_and_exit()

    def check_done(self):
        if self.t0 is None:
            return
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self.save_and_exit()

    def save_and_exit(self):
        if len(self.Y) == 0:
            self.get_logger().error("[Rywak] No samples collected.")
            rclpy.shutdown()
            return

        X = np.stack(self.X).astype(np.float32)  # (N,720)
        Y = np.stack(self.Y).astype(np.float32)  # (N,2)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(Y.shape[0]),
            "in_dim": np.int64(X.shape[1]),
            "out_dim": np.int64(2),
        }

        ensure_dir(self.out_dir)
        try:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X=X, Y=Y, meta=meta)
            buffer.seek(0)
            with open(self.dataset_path, "wb") as f:
                f.write(buffer.read())
                f.flush()
                os.fsync(f.fileno())
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

        self.exp_logger.end_dataset_collection(
            n_samples=int(Y.shape[0]),
            scan_dim=int(X.shape[1]),
            actual_duration_sec=float(actual_duration),
            file_path=self.dataset_path,
        )

        self.get_logger().info(f"[Rywak] Saved dataset: {self.dataset_path} (n={Y.shape[0]})")
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