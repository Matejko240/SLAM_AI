import math
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

import torch
import torch.nn as nn

from .common import seed_all, ensure_dir, wrap, quat_from_yaw, xytheta_from_odom
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


class MLP2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class InferRywakNode(Node):
    def __init__(self):
        super().__init__("infer_rywak_node")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("model_name", "model_rywak.pt")

        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("pose_topic", "/pose_rywak")
        self.declare_parameter("tf_parent", "odom_rywak")
        self.declare_parameter("tf_child", "base_link_rywak")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("init_from_odom_topic", "/odom_raw")  # ustaw start (x,y,th)

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        ensure_dir(self.out_dir)

        self.model_path = os.path.join(self.out_dir, str(self.get_parameter("model_name").value))
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.init_odom_topic = str(self.get_parameter("init_from_odom_topic").value)

        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.in_dim = None

        self.prev_scan = None
        self.prev_stamp = None

        self.pose_inited = False
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        self.infer_start = None
        self.inference_count = 0
        self.inference_times_ms = []

        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.sub_init_odom = self.create_subscription(Odometry, self.init_odom_topic, self.on_init_odom, 50)

        self.timer = self.create_timer(0.5, self.try_load_model)
        self.stats_timer = self.create_timer(10.0, self.periodic_save_stats)

    def on_init_odom(self, msg: Odometry):
        if self.pose_inited:
            return
        self.x, self.y, self.th = xytheta_from_odom(msg)
        self.pose_inited = True

    def try_load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            return

        payload = torch.load(self.model_path, map_location="cpu")
        self.in_dim = int(payload.get("in_dim", 720))
        self.model = MLP2(self.in_dim, 2)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.x_mean = payload["x_mean"].cpu().numpy().astype(np.float32)
        self.x_std = payload["x_std"].cpu().numpy().astype(np.float32)
        self.y_mean = payload["y_mean"].cpu().numpy().astype(np.float32)
        self.y_std = payload["y_std"].cpu().numpy().astype(np.float32)

        self.infer_start = time.time()

        self.exp_logger.start_inference(
            seed=self.seed,
            scan_topic=self.scan_topic,
            odom_topic="(unused)",
            pose_topic=self.pose_topic,
            tf_parent=self.tf_parent,
            tf_child=self.tf_child,
            model_path=self.model_path,
        )

        self.get_logger().info(f"[Rywak] Model loaded: {self.model_path}")

    def periodic_save_stats(self):
        if self.infer_start is None or self.inference_count == 0:
            return
        total = time.time() - self.infer_start
        avg_ms = float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0.0
        self.exp_logger.end_inference(
            n_predictions=int(self.inference_count),
            total_duration_sec=float(total),
            avg_inference_time_ms=float(avg_ms),
        )

    def on_scan(self, msg: LaserScan):
        if self.model is None or not self.pose_inited:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        if self.prev_scan is None:
            self.prev_scan = scan
            self.prev_stamp = msg.header.stamp
            return

        t_prev = float(self.prev_stamp.sec) + 1e-9 * float(self.prev_stamp.nanosec)
        t_cur = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        dt = max(1e-3, t_cur - t_prev)

        diff = (scan - self.prev_scan).astype(np.float32)
        x = np.concatenate([scan, diff], axis=0).astype(np.float32)
        if x.size != self.in_dim:
            return

        xn = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        xt = torch.from_numpy(xn[None, :]).float()

        t0 = time.perf_counter()
        with torch.no_grad():
            yn = self.model(xt).cpu().numpy().reshape(-1).astype(np.float32)
        t1 = time.perf_counter()

        self.inference_times_ms.append((t1 - t0) * 1000.0)
        self.inference_count += 1

        y = yn * self.y_std + self.y_mean
        v = float(y[0])
        w = float(y[1])

        # unicycle integration
        self.x += v * dt * math.cos(self.th)
        self.y += v * dt * math.sin(self.th)
        self.th = wrap(self.th + w * dt)

        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = "odom"
        qx, qy, qz, qw = quat_from_yaw(self.th)
        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pub_pose.publish(ps)

        if self.publish_tf and self.tf_br is not None:
            tfm = TransformStamped()
            tfm.header.stamp = msg.header.stamp
            tfm.header.frame_id = self.tf_parent
            tfm.child_frame_id = self.tf_child
            tfm.transform.translation.x = float(self.x)
            tfm.transform.translation.y = float(self.y)
            tfm.transform.translation.z = 0.0
            tfm.transform.rotation.x = qx
            tfm.transform.rotation.y = qy
            tfm.transform.rotation.z = qz
            tfm.transform.rotation.w = qw
            self.tf_br.sendTransform(tfm)

        self.prev_scan = scan
        self.prev_stamp = msg.header.stamp


def main():
    rclpy.init()
    node = InferRywakNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.periodic_save_stats()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass