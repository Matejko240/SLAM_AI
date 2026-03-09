import math
import os
import time
from collections import deque
from typing import List
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


def _parse_hidden_dims(value) -> List[int]:
    if value is None:
        return [256, 128, 64]
    dims = [int(v) for v in list(value) if int(v) > 0]
    return dims if dims else [256, 128, 64]


class MLP2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dims: List[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [192, 96, 48]

        layers = []
        prev = int(in_dim)
        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InferRywakNode(Node):
    def __init__(self):
        super().__init__("infer_rywak_node")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("model_name", "model_rywak.pt")
        self.declare_parameter("write_experiment_metadata", False)

        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("pose_topic", "/pose_rywak")
        self.declare_parameter("tf_parent", "odom_rywak")
        self.declare_parameter("tf_child", "base_link_rywak")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("odom_topic", "/odom_raw")
        self.declare_parameter("init_from_odom_topic", "/odom_raw")  # ustaw start (x,y,th)
        self.declare_parameter("sync_tolerance_sec", 0.08)
        self.declare_parameter("interpolate_odom", True)
        self.declare_parameter("sync_pair_gap_sec", 0.2)
        self.declare_parameter("delta_scan_clip", 2.0)
        self.declare_parameter("v_clip_abs", 0.45)
        self.declare_parameter("w_clip_abs", 1.20)
        self.declare_parameter("fuse_odom_v_weight", 0.25)
        self.declare_parameter("fuse_odom_w_weight", 0.55)
        self.declare_parameter("fuse_odom_v_gain", 0.45)
        self.declare_parameter("fuse_odom_w_gain", 0.35)
        self.declare_parameter("vel_ema_alpha", 0.60)
        self.declare_parameter("anchor_yaw_to_odom", 0.35)
        self.declare_parameter("heading_for_xy_odom_weight", 0.60)
        self.declare_parameter("xy_step_odom_weight", 0.35)
        self.declare_parameter("xy_step_odom_gain", 0.45)
        self.declare_parameter("max_integration_dt", 0.20)

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
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.init_odom_topic = str(self.get_parameter("init_from_odom_topic").value)
        self.sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self.interpolate_odom = bool(self.get_parameter("interpolate_odom").value)
        self.sync_pair_gap_sec = float(self.get_parameter("sync_pair_gap_sec").value)
        self.delta_scan_clip = float(self.get_parameter("delta_scan_clip").value)
        self.v_clip_abs = float(self.get_parameter("v_clip_abs").value)
        self.w_clip_abs = float(self.get_parameter("w_clip_abs").value)
        self.fuse_odom_v_weight = float(self.get_parameter("fuse_odom_v_weight").value)
        self.fuse_odom_w_weight = float(self.get_parameter("fuse_odom_w_weight").value)
        self.fuse_odom_v_gain = float(self.get_parameter("fuse_odom_v_gain").value)
        self.fuse_odom_w_gain = float(self.get_parameter("fuse_odom_w_gain").value)
        self.vel_ema_alpha = float(self.get_parameter("vel_ema_alpha").value)
        self.anchor_yaw_to_odom = float(self.get_parameter("anchor_yaw_to_odom").value)
        self.heading_for_xy_odom_weight = float(self.get_parameter("heading_for_xy_odom_weight").value)
        self.xy_step_odom_weight = float(self.get_parameter("xy_step_odom_weight").value)
        self.xy_step_odom_gain = float(self.get_parameter("xy_step_odom_gain").value)
        self.max_integration_dt = float(self.get_parameter("max_integration_dt").value)

        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.in_dim = None
        self._in_dim_warned = False

        self.prev_scan = None
        self.prev_stamp_sec = None
        self.theta_hist = deque(maxlen=3)
        # (stamp_sec, theta, v, w)
        self.odom_buf = deque(maxlen=2000)
        self.odom_miss_count = 0
        self.odom_interp_count = 0
        self.odom_nearest_count = 0
        self.v_filt = None
        self.w_filt = None

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
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_init_odom = None
        if self.init_odom_topic != self.odom_topic:
            self.sub_init_odom = self.create_subscription(Odometry, self.init_odom_topic, self.on_init_odom, 50)

        self.timer = self.create_timer(0.5, self.try_load_model)
        self.stats_timer = self.create_timer(10.0, self.periodic_save_stats)

        self.get_logger().info(
            f"[Rywak] sync: interp={self.interpolate_odom}, "
            f"tol={self.sync_tolerance_sec}, gap={self.sync_pair_gap_sec}, "
            f"delta_clip={self.delta_scan_clip}, v_clip={self.v_clip_abs}, w_clip={self.w_clip_abs}, "
            f"fuse(v={self.fuse_odom_v_weight}+{self.fuse_odom_v_gain}*err, "
            f"w={self.fuse_odom_w_weight}+{self.fuse_odom_w_gain}*err), "
            f"ema={self.vel_ema_alpha}, yaw_anchor={self.anchor_yaw_to_odom}, "
            f"xy_heading_odom_w={self.heading_for_xy_odom_weight}, "
            f"xy_step_odom_w={self.xy_step_odom_weight}+{self.xy_step_odom_gain}*err, "
            f"max_dt={self.max_integration_dt}"
        )

    def on_init_odom(self, msg: Odometry):
        if self.pose_inited:
            return
        self.x, self.y, self.th = xytheta_from_odom(msg)
        self.pose_inited = True

    def on_odom(self, msg: Odometry):
        t = _stamp_to_sec(msg.header.stamp)
        _, _, th = xytheta_from_odom(msg)
        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)
        self.odom_buf.append((t, th, v, w))
        if self.init_odom_topic == self.odom_topic and not self.pose_inited:
            self.x, self.y, self.th = xytheta_from_odom(msg)
            self.pose_inited = True

    def _nearest_odom(self, t_scan: float):
        if not self.odom_buf:
            return None

        while len(self.odom_buf) > 2 and self.odom_buf[1][0] < (t_scan - 1.0):
            self.odom_buf.popleft()

        t_best, th_best, v_best, w_best = min(self.odom_buf, key=lambda x: abs(x[0] - t_scan))
        if abs(t_best - t_scan) > self.sync_tolerance_sec:
            return None
        return float(th_best), float(v_best), float(w_best)

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

            t0, th0, v0, w0 = prev
            t1, th1, v1, w1 = cur
            gap = float(t1 - t0)
            if gap < 1e-6:
                if abs(t_scan - t0) <= self.sync_tolerance_sec:
                    return float(th0), float(v0), float(w0)
                return None

            if gap > self.sync_pair_gap_sec:
                return None
            if (t_scan - t0) > self.sync_tolerance_sec:
                return None
            if (t1 - t_scan) > self.sync_tolerance_sec:
                return None

            alpha = (t_scan - t0) / gap
            th = _interp_angle(th0, th1, alpha)
            v = float(v0 + alpha * (v1 - v0))
            w = float(w0 + alpha * (w1 - w0))
            return th, v, w

        return None

    def _odom_at(self, t_scan: float):
        if self.interpolate_odom:
            od = self._interpolated_odom(t_scan)
            if od is not None:
                self.odom_interp_count += 1
                return od
        od = self._nearest_odom(t_scan)
        if od is not None:
            self.odom_nearest_count += 1
        return od

    def try_load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            return

        payload = torch.load(self.model_path, map_location="cpu")
        self.in_dim = int(payload.get("in_dim", 362))
        hidden_dims = _parse_hidden_dims(payload.get("hidden_dims", [256, 128, 64]))
        dropout = float(payload.get("dropout", 0.0))
        self.model = MLP2(self.in_dim, 2, hidden_dims=hidden_dims, dropout=dropout)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.x_mean = payload["x_mean"].cpu().numpy().astype(np.float32)
        self.x_std = payload["x_std"].cpu().numpy().astype(np.float32)
        self.y_mean = payload["y_mean"].cpu().numpy().astype(np.float32)
        self.y_std = payload["y_std"].cpu().numpy().astype(np.float32)

        self.infer_start = time.time()

        if self.write_experiment_metadata:
            self.exp_logger.start_inference(
                seed=self.seed,
                scan_topic=self.scan_topic,
                odom_topic=self.odom_topic,
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
        if self.write_experiment_metadata:
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

        t_cur = _stamp_to_sec(msg.header.stamp)
        odom_match = self._odom_at(t_cur)
        if odom_match is None:
            self.odom_miss_count += 1
            return
        th_cur, v_odom, w_odom = odom_match

        if self.prev_scan is None:
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            self.theta_hist.append(th_cur)
            return

        dt = max(1e-3, t_cur - float(self.prev_stamp_sec))
        if self.max_integration_dt > 0.0:
            dt = min(dt, self.max_integration_dt)

        delta_scan = (scan - self.prev_scan).astype(np.float32)
        if self.delta_scan_clip > 0.0:
            delta_scan = np.clip(delta_scan, -self.delta_scan_clip, self.delta_scan_clip).astype(np.float32)
        self.theta_hist.append(th_cur)
        if len(self.theta_hist) < 3:
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            return

        d_theta1 = wrap(float(self.theta_hist[-1] - self.theta_hist[-2]))
        d_theta2 = wrap(float(self.theta_hist[-2] - self.theta_hist[-3]))

        x = np.concatenate(
            [np.asarray([d_theta1, d_theta2], dtype=np.float32), delta_scan], axis=0
        ).astype(np.float32)
        if x.size != self.in_dim:
            if not self._in_dim_warned:
                self.get_logger().warn(
                    f"[Rywak] Feature dim mismatch: got {x.size}, model expects {self.in_dim}. "
                    "Likely old model incompatible with current feature pipeline."
                )
                self._in_dim_warned = True
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
        v_pred = float(y[0])
        w_pred = float(y[1])

        v_base = min(max(self.fuse_odom_v_weight, 0.0), 1.0)
        w_base = min(max(self.fuse_odom_w_weight, 0.0), 1.0)
        v_gain = max(0.0, self.fuse_odom_v_gain)
        w_gain = max(0.0, self.fuse_odom_w_gain)
        v_scale = self.v_clip_abs if self.v_clip_abs > 0.0 else 0.5
        w_scale = self.w_clip_abs if self.w_clip_abs > 0.0 else 1.0
        v_err = abs(v_pred - float(v_odom)) / max(v_scale, 1e-3)
        w_err = abs(w_pred - float(w_odom)) / max(w_scale, 1e-3)
        wv = min(max(v_base + v_gain * v_err, 0.0), 1.0)
        ww = min(max(w_base + w_gain * w_err, 0.0), 1.0)
        v = (1.0 - wv) * v_pred + wv * float(v_odom)
        w = (1.0 - ww) * w_pred + ww * float(w_odom)

        if self.v_clip_abs > 0.0:
            v = float(np.clip(v, -self.v_clip_abs, self.v_clip_abs))
        if self.w_clip_abs > 0.0:
            w = float(np.clip(w, -self.w_clip_abs, self.w_clip_abs))

        alpha_ema = min(max(self.vel_ema_alpha, 0.0), 0.999)
        if self.v_filt is None:
            self.v_filt = v
            self.w_filt = w
        else:
            self.v_filt = alpha_ema * self.v_filt + (1.0 - alpha_ema) * v
            self.w_filt = alpha_ema * self.w_filt + (1.0 - alpha_ema) * w
        v = float(self.v_filt)
        w = float(self.w_filt)

        # unicycle integration
        th_pred = wrap(self.th + w * dt)
        yaw_anchor = min(max(self.anchor_yaw_to_odom, 0.0), 1.0)
        self.th = _interp_angle(th_pred, float(th_cur), yaw_anchor)

        heading_w = min(max(self.heading_for_xy_odom_weight, 0.0), 1.0)
        th_xy = _interp_angle(self.th, float(th_cur), heading_w)
        step_pred_x = v * dt * math.cos(th_xy)
        step_pred_y = v * dt * math.sin(th_xy)
        step_odom_x = float(v_odom) * dt * math.cos(float(th_cur))
        step_odom_y = float(v_odom) * dt * math.sin(float(th_cur))
        step_base = min(max(self.xy_step_odom_weight, 0.0), 1.0)
        step_gain = max(0.0, self.xy_step_odom_gain)
        step_err = abs(v - float(v_odom)) / max(v_scale, 1e-3)
        step_w = min(max(step_base + step_gain * step_err, 0.0), 1.0)
        self.x += (1.0 - step_w) * step_pred_x + step_w * step_odom_x
        self.y += (1.0 - step_w) * step_pred_y + step_w * step_odom_y

        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.tf_parent
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
        self.prev_stamp_sec = t_cur


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
