import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from .common import seed_all, ensure_dir, wrap, xytheta_from_odom, xytheta_from_pose_stamped


class DatasetRecorder(Node):
    def __init__(self):
        super().__init__("dataset_recorder")
        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("max_samples", 5000)
        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("dataset_name", "dataset.npz")

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        self.out_dir = str(self.get_parameter("out_dir").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))

        ensure_dir(self.out_dir)

        self.latest_odom = None
        self.latest_gt = None

        self.x_scan = []
        self.x_odom = []
        self.y = []
        self.t0 = self.get_clock().now()

        self.sub_scan = self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value), self.on_scan, 20
        )
        self.sub_odom = self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self.on_odom, 50
        )
        self.sub_gt = self.create_subscription(
            PoseStamped, str(self.get_parameter("gt_topic").value), self.on_gt, 50
        )

        self.timer = self.create_timer(0.5, self.check_done)

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def on_gt(self, msg: PoseStamped):
        self.latest_gt = msg

    def on_scan(self, msg: LaserScan):
        if self.latest_odom is None or self.latest_gt is None:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size != 360:
            return
        if not np.all(np.isfinite(ranges)):
            rmax = float(msg.range_max) if msg.range_max > 0 else 6.0
            ranges = np.where(np.isfinite(ranges), ranges, rmax).astype(np.float32)

        ox, oy, oth = xytheta_from_odom(self.latest_odom)
        gx, gy, gth = xytheta_from_pose_stamped(self.latest_gt)

        dx = gx - ox
        dy = gy - oy
        dth = wrap(gth - oth)

        self.x_scan.append(ranges.copy())
        self.x_odom.append([ox, oy, oth])
        self.y.append([dx, dy, dth])

        if len(self.y) >= self.max_samples:
            self.save_and_exit()

    def check_done(self):
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self.save_and_exit()

    def save_and_exit(self):
        if len(self.y) == 0:
            self.get_logger().error("No samples collected.")
            rclpy.shutdown()
            return

        X_scan = np.stack(self.x_scan).astype(np.float32)
        X_odom = np.asarray(self.x_odom, dtype=np.float32)
        Y = np.asarray(self.y, dtype=np.float32)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(len(Y)),
            "scan_dim": np.int64(X_scan.shape[1]),
        }

        tmp = self.dataset_path + ".tmp"
        np.savez_compressed(tmp, X_scan=X_scan, X_odom=X_odom, Y=Y, meta=meta)
        os.replace(tmp, self.dataset_path)
        self.get_logger().info(f"Saved dataset: {self.dataset_path} (n={len(Y)})")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()
