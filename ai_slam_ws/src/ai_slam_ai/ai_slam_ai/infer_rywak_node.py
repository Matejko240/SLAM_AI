import math
import json
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

from .common import (
    seed_all,
    ensure_dir,
    quat_from_yaw,
    select_torch_device,
    synchronize_torch_device,
    wrap,
    xytheta_from_odom,
    xytheta_from_pose_stamped,
)
from .experiment_logger import ExperimentLogger

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
        self.declare_parameter(
            "model_source_experiment_id",
            "",
        )  # jeśli niepuste: ładuj model z out/<id>/ zamiast z bieżącego experiment_id
        self.declare_parameter("model_name", "model_rywak.pt")
        self.declare_parameter("torch_device", "auto")
        self.declare_parameter("write_experiment_metadata", False)

        self.declare_parameter("scan_topic", "/scan_slam_rywak")
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
        self.declare_parameter("anchor_xy_to_odom", 0.0)
        self.declare_parameter("anchor_xy_to_odom_gain", 0.0)
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
            f"[Rywak] Model directory: {model_dir} (experiment output: {self.out_dir})"
        )
        self.torch_device_request = str(self.get_parameter("torch_device").value)
        self.torch_device_info = select_torch_device(self.torch_device_request)
        self.device = torch.device(self.torch_device_info.resolved)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.get_logger().info(
            f"[Rywak] Torch device: requested={self.torch_device_info.requested}, "
            f"using={self.torch_device_info.resolved} ({self.torch_device_info.reason})"
        )
        if self.torch_device_info.warning:
            self.get_logger().warn(self.torch_device_info.warning)
        self.exp_logger.add_note(
            f"infer_rywak_node torch device requested={self.torch_device_info.requested}, "
            f"resolved={self.torch_device_info.resolved}: {self.torch_device_info.reason}"
        )
        if self.torch_device_info.warning:
            self.exp_logger.add_note(self.torch_device_info.warning)
        self.exp_logger.save()

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
        self.anchor_xy_to_odom = float(self.get_parameter("anchor_xy_to_odom").value)
        self.anchor_xy_to_odom_gain = float(self.get_parameter("anchor_xy_to_odom_gain").value)
        self.heading_for_xy_odom_weight = float(self.get_parameter("heading_for_xy_odom_weight").value)
        self.xy_step_odom_weight = float(self.get_parameter("xy_step_odom_weight").value)
        self.xy_step_odom_gain = float(self.get_parameter("xy_step_odom_gain").value)
        self.max_integration_dt = float(self.get_parameter("max_integration_dt").value)

        self.model = None
        self.x_mean_t = None
        self.x_std_t = None
        self.y_mean_t = None
        self.y_std_t = None
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
        self._debug_step = 0

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
            f"xy_anchor={self.anchor_xy_to_odom}+{self.anchor_xy_to_odom_gain}*err, "
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
        x, y, th = xytheta_from_odom(msg)
        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)
        self.odom_buf.append((t, x, y, th, v, w))
        if self.init_odom_topic == self.odom_topic and not self.pose_inited:
            self.x, self.y, self.th = x, y, th
            self.pose_inited = True

    def _nearest_odom(self, t_scan: float):
        if not self.odom_buf:
            return None

        while len(self.odom_buf) > 2 and self.odom_buf[1][0] < (t_scan - 1.0):
            self.odom_buf.popleft()

        t_best, x_best, y_best, th_best, v_best, w_best = min(self.odom_buf, key=lambda x: abs(x[0] - t_scan))
        if abs(t_best - t_scan) > self.sync_tolerance_sec:
            return None
        return float(x_best), float(y_best), float(th_best), float(v_best), float(w_best)

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
        self.model.to(self.device)
        self.model.eval()

        self.x_mean_t = payload["x_mean"].to(self.device, dtype=torch.float32).view(1, -1)
        self.x_std_t = payload["x_std"].to(self.device, dtype=torch.float32).view(1, -1)
        self.y_mean_t = payload["y_mean"].to(self.device, dtype=torch.float32)
        self.y_std_t = payload["y_std"].to(self.device, dtype=torch.float32)

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
                torch_device_requested=self.torch_device_info.requested,
                torch_device_used=self.torch_device_info.resolved,
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
        x_odom, y_odom, th_cur, v_odom, w_odom = odom_match

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

        xt = torch.from_numpy(x[None, :]).to(self.device, dtype=torch.float32)

        t0 = time.perf_counter()
        synchronize_torch_device(self.device)
        with torch.inference_mode():
            xn = (xt - self.x_mean_t) / torch.clamp(self.x_std_t, min=1e-6)
            yn = self.model(xn).reshape(-1)
            y = (yn * self.y_std_t + self.y_mean_t).detach().cpu().numpy().astype(np.float32)
        synchronize_torch_device(self.device)
        t1 = time.perf_counter()

        self.inference_times_ms.append((t1 - t0) * 1000.0)
        self.inference_count += 1

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
        alpha_ema_default = float(alpha_ema)
        ema_trigger = "none"
        # Keep EMA adaptive thresholds independent from configured clip range.
        # Large clip values (e.g. 5 m/s) were masking meaningful sign/delta changes.
        v_flip_mag_th = 0.08
        w_flip_mag_th = 0.20
        v_jump_th = 0.35
        w_jump_th = 0.75
        v_filt_prev = float(self.v_filt) if self.v_filt is not None else None
        w_filt_prev = float(self.w_filt) if self.w_filt is not None else None
        if self.v_filt is not None:
            v_sign_flip = (self.v_filt * v < 0.0) and (abs(self.v_filt) > v_flip_mag_th or abs(v) > v_flip_mag_th)
            v_jump = abs(self.v_filt - v) > v_jump_th
            v_odom_conflict = (float(v_odom) * v < 0.0) and (abs(float(v_odom)) > v_flip_mag_th) and (abs(v) > 0.04)
            if v_sign_flip or v_jump or v_odom_conflict:
                alpha_ema = min(alpha_ema, 0.2)
                if v_sign_flip:
                    ema_trigger = "v_sign_flip"
                elif v_odom_conflict:
                    ema_trigger = "v_odom_conflict"
                else:
                    ema_trigger = "v_jump"
        if self.w_filt is not None:
            w_sign_flip = (self.w_filt * w < 0.0) and (abs(self.w_filt) > w_flip_mag_th or abs(w) > w_flip_mag_th)
            w_jump = abs(self.w_filt - w) > w_jump_th
            if w_sign_flip or w_jump:
                alpha_ema = min(alpha_ema, 0.2)
                if ema_trigger == "none":
                    ema_trigger = "w_sign_flip" if w_sign_flip else "w_jump"
        v_before_ema = float(v)
        w_before_ema = float(w)
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

        xy_anchor_base = min(max(self.anchor_xy_to_odom, 0.0), 1.0)
        xy_anchor_gain = max(0.0, self.anchor_xy_to_odom_gain)
        if xy_anchor_base > 0.0 or xy_anchor_gain > 0.0:
            err_xy = math.hypot(self.x - float(x_odom), self.y - float(y_odom))
            step_scale = max(abs(float(v_odom)) * dt, 0.05)
            anchor_w = min(max(xy_anchor_base + xy_anchor_gain * (err_xy / step_scale), 0.0), 1.0)
            self.x = (1.0 - anchor_w) * self.x + anchor_w * float(x_odom)
            self.y = (1.0 - anchor_w) * self.y + anchor_w * float(y_odom)

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

        self._debug_step += 1
        if self._debug_step % 400 == 0:
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="infer_rywak_node.py:on_scan",
                message="rywak fusion snapshot",
                data={
                    "step": int(self._debug_step),
                    "v_pred": float(v_pred),
                    "w_pred": float(w_pred),
                    "v_odom": float(v_odom),
                    "w_odom": float(w_odom),
                    "v_before_ema": float(v_before_ema),
                    "w_before_ema": float(w_before_ema),
                    "v_fused": float(v),
                    "w_fused": float(w),
                    "alpha_ema_used": float(alpha_ema),
                    "alpha_ema_default": float(alpha_ema_default),
                    "ema_trigger": str(ema_trigger),
                    "v_filt_prev": v_filt_prev,
                    "w_filt_prev": w_filt_prev,
                    "fuse_weight_v": float(wv),
                    "fuse_weight_w": float(ww),
                    "xy_step_weight": float(step_w),
                    "anchor_yaw_to_odom": float(self.anchor_yaw_to_odom),
                    "anchor_xy_to_odom": float(self.anchor_xy_to_odom),
                },
            )
            # endregion

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
