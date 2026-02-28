import io
import math
import os
import time
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

from .common import seed_all, ensure_dir, wrap, xytheta_from_pose_stamped
from .experiment_logger import ExperimentLogger


def _resample_to_360(ranges: np.ndarray) -> np.ndarray:
    """Resample dowolnego N do 360 przez interpolację po kącie."""
    n = int(ranges.size)
    if n == 360:
        return ranges.astype(np.float32)
    if n < 10:
        return None
    x_old = np.linspace(-math.pi, math.pi, n, endpoint=False)
    x_new = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    out = np.interp(x_new, x_old, ranges).astype(np.float32)
    return out


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


def _delta_pose(
    prev_xyth: Tuple[float, float, float],
    curr_xyth: Tuple[float, float, float],
    label_frame: str,
) -> Tuple[float, float, float]:
    """Delta między dwoma pozami GT.

    label_frame:
      - "local": dx,dy w układzie lokalnym prev (uczące się z par skanów jest sensowniejsze)
      - "world": dx=x2-x1, dy=y2-y1 (bliżej temu co jest w ALSAI utilities.is_data_near)
    """
    x1, y1, th1 = prev_xyth
    x2, y2, th2 = curr_xyth

    dx_w = x2 - x1
    dy_w = y2 - y1
    dth = wrap(th2 - th1)

    if label_frame.lower() == "world":
        return float(dx_w), float(dy_w), float(dth)

    # world -> local(prev)
    c = math.cos(th1)
    s = math.sin(th1)
    dx_l = c * dx_w + s * dy_w
    dy_l = -s * dx_w + c * dy_w
    return float(dx_l), float(dy_l), float(dth)


class DatasetRecorderRobak(Node):
    def __init__(self):
        super().__init__("dataset_recorder_robak")

        # --- IO / experiment
        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("max_samples", 8000)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("dataset_name", "dataset_robak.npz")

        # --- pairing like ALSAI (offsety w skanach)
        self.declare_parameter("offsets", [1, 2, 3, 4, 5, 8, 10])
        self.declare_parameter("max_pair_dist", 0.5)                 # [m]
        self.declare_parameter("max_pair_dyaw", float(math.pi))      # [rad] ~180deg

        # --- labels
        self.declare_parameter("label_frame", "local")  # local | world

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
        self.max_pair_dist = float(self.get_parameter("max_pair_dist").value)
        self.max_pair_dyaw = float(self.get_parameter("max_pair_dyaw").value)
        self.label_frame = str(self.get_parameter("label_frame").value).lower()

        self.latest_gt: PoseStamped | None = None
        self.gt_count = 0
        self.scan_count = 0

        # buf: (scan360, (x,y,th))
        self.buf: Deque[Tuple[np.ndarray, Tuple[float, float, float]]] = deque(
            maxlen=max(self.offsets) + 1
        )

        # dataset
        self.X_pairs = []  # list of (2,360)
        self.Y = []        # list of (3,)

        self.t0 = None
        self.topics_ready = False
        self.experiment_start = time.time()

        self.sub_gt = self.create_subscription(PoseStamped, self.gt_topic, self.on_gt, 50)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        self.wait_timer = self.create_timer(1.0, self.wait_for_topics)
        self.timer = self.create_timer(0.5, self.check_done)

        self.get_logger().info(f"[Robak] out_dir={self.out_dir}")
        self.get_logger().info(f"[Robak] scan={self.scan_topic}, gt={self.gt_topic}, offsets={self.offsets}")
        self.get_logger().info(f"[Robak] label_frame={self.label_frame}, max_dist={self.max_pair_dist}, max_dyaw={self.max_pair_dyaw:.3f}")

        self.exp_logger.start_dataset_collection(
            seed=self.seed,
            duration_sec=self.duration_sec,
            max_samples=self.max_samples,
            scan_topic=self.scan_topic,
            odom_topic="(unused)",
            gt_topic=self.gt_topic,
        )

    def on_gt(self, msg: PoseStamped):
        self.latest_gt = msg
        self.gt_count += 1

    def wait_for_topics(self):
        if self.topics_ready:
            return
        if self.latest_gt is not None:
            self.topics_ready = True
            self.t0 = self.get_clock().now()
            elapsed_exp = time.time() - self.experiment_start
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"[Robak][FAZA 1] DATASET START (t={elapsed_exp:.0f}s)")
            self.get_logger().info("=" * 60)
        else:
            self.get_logger().info(f"[Robak] Waiting for GT... (n_gt={self.gt_count})")

    def on_scan(self, msg: LaserScan):
        if not self.topics_ready or self.latest_gt is None:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        curr_gt = xytheta_from_pose_stamped(self.latest_gt)
        self.buf.append((scan, curr_gt))
        self.scan_count += 1

        # tworzymy próbki dla wszystkich offsetów (jeśli bufor ma dane)
        if len(self.buf) < (max(self.offsets) + 1):
            return

        scan_curr, gt_curr = self.buf[-1]

        for off in self.offsets:
            scan_prev, gt_prev = self.buf[-(off + 1)]

            # gating jak w ALSAI: max dystans i max dyaw
            dx_w = gt_curr[0] - gt_prev[0]
            dy_w = gt_curr[1] - gt_prev[1]
            dist = math.hypot(dx_w, dy_w)
            dyaw = wrap(gt_curr[2] - gt_prev[2])

            if dist > self.max_pair_dist:
                continue
            if abs(dyaw) > self.max_pair_dyaw:
                continue

            dx, dy, dth = _delta_pose(gt_prev, gt_curr, self.label_frame)

            self.X_pairs.append(np.stack([scan_prev, scan_curr], axis=0).astype(np.float32))
            self.Y.append(np.asarray([dx, dy, dth], dtype=np.float32))

            if len(self.Y) >= self.max_samples:
                self.save_and_exit()
                return

    def check_done(self):
        if self.t0 is None:
            return
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self.save_and_exit()

    def save_and_exit(self):
        if len(self.Y) == 0:
            self.get_logger().error("[Robak] No samples collected.")
            rclpy.shutdown()
            return

        X = np.stack(self.X_pairs).astype(np.float32)  # (N,2,360)
        Y = np.stack(self.Y).astype(np.float32)        # (N,3)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(Y.shape[0]),
            "x_shape": np.asarray(X.shape, dtype=np.int64),
            "label_frame": np.asarray([self.label_frame], dtype=object),
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            "max_pair_dist": np.float32(self.max_pair_dist),
            "max_pair_dyaw": np.float32(self.max_pair_dyaw),
        }

        ensure_dir(self.out_dir)
        try:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X_pairs=X, Y=Y, meta=meta)
            buffer.seek(0)
            with open(self.dataset_path, "wb") as f:
                f.write(buffer.read())
                f.flush()
                os.fsync(f.fileno())
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

        self.exp_logger.end_dataset_collection(
            n_samples=int(Y.shape[0]),
            scan_dim=720,
            actual_duration_sec=float(actual_duration),
            file_path=self.dataset_path,
        )

        self.get_logger().info(f"[Robak] Saved dataset: {self.dataset_path} (n={Y.shape[0]})")
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