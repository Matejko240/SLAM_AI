import os
import time
import math
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from .common import (
    seed_all,
    ensure_dir,
    quat_from_yaw,
    select_torch_device,
    synchronize_torch_device,
    wrap,
    xytheta_from_odom,
    xytheta_from_pose_stamped,
    yaw_from_quat,
)
from .experiment_logger import ExperimentLogger

import torch
import torch.nn as nn

DEBUG_LOG_PATH = "/home/matejko/SLAM_AI/.cursor/debug-a69755.log"
DEBUG_SESSION_ID = "a69755"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": str(run_id),
        "hypothesisId": str(hypothesis_id),
        "location": str(location),
        "message": str(message),
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
        self.declare_parameter(
            "model_source_experiment_id",
            "",
        )  # jeśli niepuste: ładuj model z out/<id>/ zamiast z bieżącego experiment_id
        self.declare_parameter("model_name", "model.pt")
        self.declare_parameter("model_wait_timeout", 300.0)
        self.declare_parameter("torch_device", "auto")
        self.declare_parameter("scan_topic", "/scan_slam_ai")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_topic", "/pose_ai")
        self.declare_parameter("odom_ai_topic", "/odom_ai")
        self.declare_parameter("tf_parent", "odom_ai")
        self.declare_parameter("tf_child", "base_link_ai")
        self.declare_parameter("max_correction_trans", 0.0)
        self.declare_parameter("max_correction_yaw", 0.0)

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

        model_src_id = str(self.get_parameter("model_source_experiment_id").value or "").strip()
        model_name = str(self.get_parameter("model_name").value)
        if model_src_id:
            cand = os.path.abspath(os.path.join(base_out_dir, model_src_id))
            base_abs = os.path.abspath(base_out_dir)
            try:
                if os.path.commonpath([base_abs, cand]) != base_abs:
                    raise ValueError(f"model_source_experiment_id resolves outside out_dir: {cand}")
            except ValueError as e:
                raise ValueError(f"invalid model_source_experiment_id path: {cand}") from e
            model_dir = cand
        else:
            model_dir = self.out_dir
        self.model_path = os.path.join(model_dir, model_name)
        self.get_logger().info(
            f"Model directory: {model_dir} (experiment output: {self.out_dir})"
        )
        self.torch_device_request = str(self.get_parameter("torch_device").value)
        self.torch_device_info = select_torch_device(self.torch_device_request)
        self.device = torch.device(self.torch_device_info.resolved)
        self.get_logger().info(
            f"Torch device: requested={self.torch_device_info.requested}, "
            f"using={self.torch_device_info.resolved} ({self.torch_device_info.reason})"
        )
        if self.torch_device_info.warning:
            self.get_logger().warn(self.torch_device_info.warning)
        self.exp_logger.add_note(
            f"infer_node torch device requested={self.torch_device_info.requested}, "
            f"resolved={self.torch_device_info.resolved}: {self.torch_device_info.reason}"
        )
        if self.torch_device_info.warning:
            self.exp_logger.add_note(self.torch_device_info.warning)
        self.exp_logger.save()

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)
        self.max_correction_trans = float(self.get_parameter("max_correction_trans").value)
        self.max_correction_yaw = float(self.get_parameter("max_correction_yaw").value)
        self.model_wait_timeout = float(self.get_parameter("model_wait_timeout").value)
        self.model_wait_start = time.time()
        self._model_wait_warned = False

        self.latest_odom = None
        self.model = None
        self.x_mean_t = None
        self.x_std_t = None
        self.y_mean_t = None
        self.y_std_t = None
        self.in_dim = None

        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_odom_ai = self.create_publisher(Odometry, str(self.get_parameter("odom_ai_topic").value), 10)
        self.tf_br = TransformBroadcaster(self)

        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        self.timer = self.create_timer(0.5, self.try_load_model)
        
        # Timer do okresowego zapisywania statystyk (co 10s)
        self.stats_timer = self.create_timer(10.0, self.periodic_save_stats)
        
        # Statystyki inferencji
        self.inference_count = 0
        self.inference_times = []
        self.infer_start = None
        self._debug_step = 0

    def periodic_save_stats(self):
        """Okresowo zapisuje statystyki inferencji do metadata.json."""
        if self.infer_start is not None and self.inference_count > 0:
            total_duration = time.time() - self.infer_start
            avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            
            # Zapisz aktualne statystyki bez kończenia etapu inferencji
            self.exp_logger.update_inference_statistics(
                n_predictions=self.inference_count,
                total_duration_sec=total_duration,
                avg_inference_time_ms=avg_inference_time
            )

    def try_load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            wait_elapsed = time.time() - self.model_wait_start
            if wait_elapsed >= self.model_wait_timeout:
                self.get_logger().error(
                    f"Model not found after {self.model_wait_timeout:.1f}s: {self.model_path}. "
                    "Shutting down infer_node."
                )
                rclpy.shutdown()
                return
            if (not self._model_wait_warned) and wait_elapsed >= 5.0:
                self.get_logger().warn(f"Waiting for model file: {self.model_path}")
                self._model_wait_warned = True
            return
        payload = torch.load(self.model_path, map_location="cpu")
        self.in_dim = int(payload.get("in_dim", 363))
        self.model = MLP(self.in_dim, 3)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.x_mean_t = payload["x_mean"].to(self.device, dtype=torch.float32).view(1, -1)
        self.x_std_t = payload["x_std"].to(self.device, dtype=torch.float32).view(1, -1)
        self.y_mean_t = payload["y_mean"].to(self.device, dtype=torch.float32)
        self.y_std_t = payload["y_std"].to(self.device, dtype=torch.float32)
        self.infer_start = time.time()
        
        # Logowanie startu inferencji
        self.exp_logger.start_inference(
            seed=self.seed,
            scan_topic=self.scan_topic,
            odom_topic=self.odom_topic,
            pose_topic=self.pose_topic,
            tf_parent=self.tf_parent,
            tf_child=self.tf_child,
            model_path=self.model_path,
            torch_device_requested=self.torch_device_info.requested,
            torch_device_used=self.torch_device_info.resolved,
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
        rmax = float(msg.range_max) if msg.range_max > 0.0 else 10.0
        ranges = np.where(np.isfinite(ranges), ranges, rmax).astype(np.float32)
        ranges = np.clip(ranges, float(msg.range_min) if msg.range_min > 0 else 0.08, rmax)

        ox, oy, oth = xytheta_from_odom(self.latest_odom)

        x = np.concatenate([ranges, np.asarray([ox, oy, oth], dtype=np.float32)], axis=0)
        if x.size != self.in_dim:
            return
        xt = torch.from_numpy(x[None, :]).to(self.device, dtype=torch.float32)
        
        # Pomiar czasu inferencji
        t_start = time.perf_counter()
        synchronize_torch_device(self.device)
        with torch.inference_mode():
            xn = (xt - self.x_mean_t) / torch.clamp(self.x_std_t, min=1e-6)
            yn = self.model(xn).reshape(-1)
            y = (yn * self.y_std_t + self.y_mean_t).detach().cpu().numpy().astype(np.float32)
        synchronize_torch_device(self.device)
        t_end = time.perf_counter()
        self.inference_times.append((t_end - t_start) * 1000)  # w ms
        self.inference_count += 1

        dx_raw, dy_raw, dth_raw = float(y[0]), float(y[1]), float(y[2])
        dx, dy, dth = float(dx_raw), float(dy_raw), float(dth_raw)
        if self.max_correction_trans > 0.0:
            corr_norm = math.hypot(dx, dy)
            if corr_norm > self.max_correction_trans and corr_norm > 1e-9:
                s = self.max_correction_trans / corr_norm
                dx *= s
                dy *= s
        if self.max_correction_yaw > 0.0:
            dth = float(np.clip(dth, -self.max_correction_yaw, self.max_correction_yaw))
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

        self._debug_step += 1
        if self._debug_step % 400 == 0:
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H12",
                location="infer_node.py:on_scan",
                message="ai correction snapshot",
                data={
                    "step": int(self._debug_step),
                    "dx_pred": float(dx),
                    "dy_pred": float(dy),
                    "dth_pred": float(dth),
                    "dx_pred_raw": float(dx_raw),
                    "dy_pred_raw": float(dy_raw),
                    "dth_pred_raw": float(dth_raw),
                    "pred_corr_norm_xy_m": float(math.hypot(dx, dy)),
                    "pred_corr_norm_xy_raw_m": float(math.hypot(dx_raw, dy_raw)),
                    "odom_x": float(ox),
                    "odom_y": float(oy),
                    "odom_th": float(oth),
                    "ai_x": float(cx),
                    "ai_y": float(cy),
                    "ai_th": float(cth),
                    "pose_frame_id": str(ps.header.frame_id),
                    "tf_parent": str(self.tf_parent),
                },
            )
            # endregion
    
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

            can_log = False
            try:
                can_log = bool(rclpy.ok())
            except Exception:
                can_log = False

            if can_log:
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
