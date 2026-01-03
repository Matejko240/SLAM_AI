import os
import time
import numpy as np
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from .common import seed_all, ensure_dir, wrap, xytheta_from_odom, xytheta_from_pose_stamped
from .experiment_logger import ExperimentLogger


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

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        
        # Inicjalizacja loggera eksperymentu (tworzy podfolder)
        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))

        ensure_dir(self.out_dir)
        self.get_logger().info(f"Output directory: {self.out_dir}")
        self.get_logger().info(f"Experiment ID: {self.exp_logger.experiment_id}")

        self.latest_odom = None
        self.latest_gt = None

        self.x_scan = []
        self.x_odom = []
        self.y = []
        self.t0 = None  # Will be set when topics are ready
        self.scan_count = 0

        self.sub_scan = self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value), self.on_scan, qos_profile_sensor_data
        )
        self.sub_odom = self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self.on_odom, qos_profile_sensor_data
        )
        self.sub_gt = self.create_subscription(
            PoseStamped, str(self.get_parameter("gt_topic").value), self.on_gt, 50
        )

        self.get_logger().info(f"Subscriptions created: scan={str(self.get_parameter('scan_topic').value)} (BEST_EFFORT QoS), odom={str(self.get_parameter('odom_topic').value)}, gt={str(self.get_parameter('gt_topic').value)}")

        self.timer = self.create_timer(0.5, self.check_done)
        self.wait_timer = self.create_timer(1.0, self.wait_for_topics)
        self.topics_ready = False
        self.odom_count = 0
        self.gt_count = 0
        self.experiment_start = time.time()  # Global experiment start time
        
        # Logowanie startu zbierania datasetu
        self.exp_logger.start_dataset_collection(
            seed=self.seed,
            duration_sec=self.duration_sec,
            max_samples=self.max_samples,
            scan_topic=str(self.get_parameter('scan_topic').value),
            odom_topic=str(self.get_parameter('odom_topic').value),
            gt_topic=str(self.get_parameter('gt_topic').value)
        )

    def wait_for_topics(self):
        if self.topics_ready:
            return
        if self.latest_odom is not None and self.latest_gt is not None:
            elapsed_exp = time.time() - self.experiment_start
            self.get_logger().info("="*60)
            self.get_logger().info(f"[FAZA 1] ZBIERANIE DANYCH - START (t={elapsed_exp:.0f}s)")
            self.get_logger().info(f"Planowany czas: {self.duration_sec}s | Zakończenie: ~t={elapsed_exp + self.duration_sec:.0f}s")
            self.get_logger().info("="*60)
            self.topics_ready = True
            # Reset t0 when topics are ready to start timing from now
            self.t0 = self.get_clock().now()
        else:
            self.get_logger().info("Waiting for topics: odom={} (n={}), gt={} (n={})".format(
                self.latest_odom is not None, self.odom_count,
                self.latest_gt is not None, self.gt_count))

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg
        self.odom_count += 1

    def on_gt(self, msg: PoseStamped):
        self.latest_gt = msg
        self.gt_count += 1

    def on_scan(self, msg: LaserScan):
        if not self.topics_ready or self.latest_odom is None or self.latest_gt is None:
            return
        
        # Resample to 360 if needed
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        rmax = float(msg.range_max) if msg.range_max > 0 else 6.0
        ranges = np.where(np.isfinite(ranges), ranges, rmax).astype(np.float32)
        
        if ranges.size != 360:
            if ranges.size < 10:
                self.get_logger().warn(f"Scan size {ranges.size} too small, skipping")
                return
            # Resample to 360
            x_old = np.linspace(-math.pi, math.pi, ranges.size, endpoint=False)
            x_new = np.linspace(-math.pi, math.pi, 360, endpoint=False)
            ranges = np.interp(x_new, x_old, ranges).astype(np.float32)
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
        self.scan_count += 1

        if len(self.y) >= self.max_samples:
            self.save_and_exit()

    def check_done(self):
        if self.t0 is None:
            return  # Wait for topics to be ready
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if self.topics_ready and elapsed > 5.0 and self.scan_count == 0:
            # Log diagnostic info periodically
            if int(elapsed) % 10 == 0:
                self.get_logger().warn(f"No scans yet! elapsed={elapsed:.1f}s, collected={len(self.y)}, scan_count={self.scan_count}")
        if elapsed >= self.duration_sec:
            self.save_and_exit()

    def save_and_exit(self):
        elapsed_str = "N/A" if self.t0 is None else f"{(self.get_clock().now() - self.t0).nanoseconds * 1e-9:.1f}s"
        if len(self.y) == 0:
            self.get_logger().error(f"No samples collected. Scans received: {self.scan_count}, elapsed: {elapsed_str}")
            rclpy.shutdown()
            return

        elapsed_exp = time.time() - self.experiment_start
        self.get_logger().info("="*60)
        self.get_logger().info(f"[FAZA 1] ZBIERANIE DANYCH - KONIEC (t={elapsed_exp:.0f}s)")
        self.get_logger().info(f"Zebrano {len(self.y)} próbek w {elapsed_str}")
        self.get_logger().info("="*60)
        
        X_scan = np.stack(self.x_scan).astype(np.float32)
        X_odom = np.asarray(self.x_odom, dtype=np.float32)
        Y = np.asarray(self.y, dtype=np.float32)

        meta = {
            "seed": np.int64(self.seed),
            "n": np.int64(len(Y)),
            "scan_dim": np.int64(X_scan.shape[1]),
        }

        # Ensure output directory exists before saving
        ensure_dir(self.out_dir)
        self.get_logger().info(f"Output dir exists: {os.path.isdir(self.out_dir)}, path: {self.out_dir}")
        
        # Save directly to the final path (skip temp file pattern due to WSL filesystem issues)
        try:
            # Use explicit file handle to ensure data is written
            import io
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X_scan=X_scan, X_odom=X_odom, Y=Y, meta=meta)
            buffer.seek(0)
            
            with open(self.dataset_path, 'wb') as f:
                f.write(buffer.read())
                f.flush()
                os.fsync(f.fileno())
            
            self.get_logger().info(f"Dataset file created: {os.path.exists(self.dataset_path)}, size: {os.path.getsize(self.dataset_path) if os.path.exists(self.dataset_path) else 'N/A'}")
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
        
        # Logowanie zakończenia zbierania datasetu
        actual_duration = (self.get_clock().now() - self.t0).nanoseconds * 1e-9 if self.t0 else 0
        self.exp_logger.end_dataset_collection(
            n_samples=len(Y),
            scan_dim=int(X_scan.shape[1]),
            actual_duration_sec=actual_duration,
            file_path=self.dataset_path
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
