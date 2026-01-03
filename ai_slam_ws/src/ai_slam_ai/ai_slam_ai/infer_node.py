import os
import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from .common import seed_all, ensure_dir, wrap, yaw_from_quat, quat_from_yaw, xytheta_from_odom
from .experiment_logger import ExperimentLogger

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
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


class InferNode(Node):
    def __init__(self):
        super().__init__("infer_node")
        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("model_name", "model.pt")
        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_topic", "/pose_ai")
        self.declare_parameter("odom_ai_topic", "/odom_ai")
        self.declare_parameter("tf_parent", "odom_ai")
        self.declare_parameter("tf_child", "base_link")

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        
        # Inicjalizacja loggera eksperymentu (używa istniejącego podfolderu)
        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        ensure_dir(self.out_dir)
        
        self.get_logger().info(f"Output directory: {self.out_dir}")
        self.get_logger().info(f"Experiment ID: {self.exp_logger.experiment_id}")
        
        self.model_path = os.path.join(self.out_dir, str(self.get_parameter("model_name").value))

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)

        self.latest_odom = None
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.in_dim = None

        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_odom_ai = self.create_publisher(Odometry, str(self.get_parameter("odom_ai_topic").value), 10)
        self.tf_br = TransformBroadcaster(self)

        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 20)

        self.timer = self.create_timer(0.5, self.try_load_model)
        
        # Timer do okresowego zapisywania statystyk (co 10s)
        self.stats_timer = self.create_timer(10.0, self.periodic_save_stats)
        
        # Statystyki inferencji
        self.inference_count = 0
        self.inference_times = []
        self.infer_start = None

    def periodic_save_stats(self):
        """Okresowo zapisuje statystyki inferencji do metadata.json."""
        if self.infer_start is not None and self.inference_count > 0:
            total_duration = time.time() - self.infer_start
            avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            
            # Zapisz aktualne statystyki (nadpisze poprzednie)
            self.exp_logger.end_inference(
                n_predictions=self.inference_count,
                total_duration_sec=total_duration,
                avg_inference_time_ms=avg_inference_time
            )

    def try_load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            return
        payload = torch.load(self.model_path, map_location="cpu")
        self.in_dim = int(payload.get("in_dim", 363))
        self.model = MLP(self.in_dim, 3)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.x_mean = payload["x_mean"].cpu().numpy().astype(np.float32)
        self.x_std = payload["x_std"].cpu().numpy().astype(np.float32)
        self.y_mean = payload["y_mean"].cpu().numpy().astype(np.float32)
        self.y_std = payload["y_std"].cpu().numpy().astype(np.float32)
        self.infer_start = time.time()
        
        # Logowanie startu inferencji
        self.exp_logger.start_inference(
            seed=self.seed,
            scan_topic=self.scan_topic,
            odom_topic=self.odom_topic,
            pose_topic=self.pose_topic,
            tf_parent=self.tf_parent,
            tf_child=self.tf_child,
            model_path=self.model_path
        )
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"[FAZA 3] INFERENCJA AI - START")
        self.get_logger().info(f"Model załadowany: {self.model_path}")
        self.get_logger().info("Korekcja pozycji publikowana na: /pose_ai")
        self.get_logger().info("="*60)

    def _publish_passthrough_odom(self, msg: Odometry):
        """Publikuje oryginalną odometrię jako odom_ai (passthrough mode przed załadowaniem modelu)."""
        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = self.tf_parent  # odom_ai
        od.child_frame_id = self.tf_child    # base_link
        od.pose = msg.pose
        od.twist = msg.twist
        self.pub_odom_ai.publish(od)
        
        # Publikuj TF odom_ai -> base_link
        tfm = TransformStamped()
        tfm.header.stamp = msg.header.stamp
        tfm.header.frame_id = self.tf_parent  # odom_ai
        tfm.child_frame_id = self.tf_child    # base_link
        tfm.transform.translation.x = msg.pose.pose.position.x
        tfm.transform.translation.y = msg.pose.pose.position.y
        tfm.transform.translation.z = msg.pose.pose.position.z
        tfm.transform.rotation = msg.pose.pose.orientation
        self.tf_br.sendTransform(tfm)

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg
        # Passthrough mode: gdy model nie jest jeszcze gotowy, przekazuj oryginalną odometrię jako odom_ai
        # To pozwala slam_toolbox_ai działać od początku eksperymentu
        if self.model is None:
            self._publish_passthrough_odom(msg)

    def on_scan(self, msg: LaserScan):
        if self.model is None or self.latest_odom is None:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size != 360:
            return
        rmax = float(msg.range_max) if msg.range_max > 0.0 else 6.0
        ranges = np.where(np.isfinite(ranges), ranges, rmax).astype(np.float32)
        ranges = np.clip(ranges, float(msg.range_min) if msg.range_min > 0 else 0.12, rmax)

        ox, oy, oth = xytheta_from_odom(self.latest_odom)

        x = np.concatenate([ranges, np.asarray([ox, oy, oth], dtype=np.float32)], axis=0)
        if x.size != self.in_dim:
            return
        xn = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        xt = torch.from_numpy(xn[None, :]).float()
        
        # Pomiar czasu inferencji
        t_start = time.perf_counter()
        with torch.no_grad():
            yn = self.model(xt).cpu().numpy().reshape(-1).astype(np.float32)
        t_end = time.perf_counter()
        self.inference_times.append((t_end - t_start) * 1000)  # w ms
        self.inference_count += 1
        
        y = yn * self.y_std + self.y_mean

        dx, dy, dth = float(y[0]), float(y[1]), float(y[2])
        cx = ox + dx
        cy = oy + dy
        cth = wrap(oth + dth)

        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = "odom"
        qx, qy, qz, qw = quat_from_yaw(cth)
        ps.pose.position.x = float(cx)
        ps.pose.position.y = float(cy)
        ps.pose.position.z = float(self.latest_odom.pose.pose.position.z)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pub_pose.publish(ps)

        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = self.tf_parent
        od.child_frame_id = self.tf_child
        od.pose.pose = ps.pose
        od.twist = self.latest_odom.twist
        self.pub_odom_ai.publish(od)

        tfm = TransformStamped()
        tfm.header.stamp = msg.header.stamp
        tfm.header.frame_id = self.tf_parent
        tfm.child_frame_id = self.tf_child
        tfm.transform.translation.x = float(cx)
        tfm.transform.translation.y = float(cy)
        tfm.transform.translation.z = float(self.latest_odom.pose.pose.position.z)
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw
        self.tf_br.sendTransform(tfm)
    
    def log_inference_stats(self):
        """Loguje statystyki inferencji przy zamykaniu node'a."""
        if self.infer_start is not None and self.inference_count > 0:
            total_duration = time.time() - self.infer_start
            avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            
            self.exp_logger.end_inference(
                n_predictions=self.inference_count,
                total_duration_sec=total_duration,
                avg_inference_time_ms=avg_inference_time
            )
            
            self.get_logger().info("="*60)
            self.get_logger().info(f"[FAZA 3] INFERENCJA AI - KONIEC")
            self.get_logger().info(f"Liczba predykcji: {self.inference_count}")
            self.get_logger().info(f"Całkowity czas: {total_duration:.1f}s")
            self.get_logger().info(f"Średni czas inferencji: {avg_inference_time:.3f}ms")
            self.get_logger().info("="*60)


def main():
    rclpy.init()
    node = InferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.log_inference_stats()
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
