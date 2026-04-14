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


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _interp_angle(th0: float, th1: float, alpha: float) -> float:
    d = wrap(float(th1) - float(th0))
    return wrap(float(th0) + float(alpha) * d)


def _delta_local(prev_xyth, cur_xyth):
    x0, y0, th0 = prev_xyth
    x1, y1, th1 = cur_xyth
    dxw = float(x1) - float(x0)
    dyw = float(y1) - float(y0)
    c = math.cos(float(th0))
    s = math.sin(float(th0))
    dx = c * dxw + s * dyw
    dy = -s * dxw + c * dyw
    dth = wrap(float(th1) - float(th0))
    return float(dx), float(dy), float(dth)


def _rebase_pose(abs_xyth, origin_xyth):
    x, y, th = abs_xyth
    ox, oy, oth = origin_xyth
    dx_w = float(x) - float(ox)
    dy_w = float(y) - float(oy)
    c0 = math.cos(float(oth))
    s0 = math.sin(float(oth))
    x_l = c0 * dx_w + s0 * dy_w
    y_l = -s0 * dx_w + c0 * dy_w
    th_l = wrap(float(th) - float(oth))
    return float(x_l), float(y_l), float(th_l)


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


class RobakConv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.head(self.feat(x))


class InferRobakNode(Node):
    def __init__(self):
        super().__init__("infer_robak_node")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter(
            "model_source_experiment_id",
            "",
        )  # jeśli niepuste: ładuj model z out/<id>/ zamiast z bieżącego experiment_id
        self.declare_parameter("model_name", "model_robak.pt")
        self.declare_parameter("torch_device", "auto")
        self.declare_parameter("write_experiment_metadata", False)

        self.declare_parameter("scan_topic", "/scan_slam_robak")
        self.declare_parameter("relay_scan_topic", "")
        self.declare_parameter("pose_topic", "/pose_robak")
        self.declare_parameter("tf_parent", "odom_robak")
        self.declare_parameter("tf_child", "base_link_robak")
        self.declare_parameter("publish_tf", True)

        # init pose (żeby porównanie z GT miało sens)
        self.declare_parameter("init_from", "gt")  # gt | odom | none
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("odom_topic", "/odom_raw")
        self.declare_parameter("max_step_trans", 0.12)
        self.declare_parameter("max_step_yaw", 0.35)
        self.declare_parameter("delta_ema_alpha", 0.55)
        self.declare_parameter("odom_heading_alpha", 0.20)
        self.declare_parameter("odom_heading_gain", 0.75)
        self.declare_parameter("odom_sync_tolerance_sec", 0.08)
        self.declare_parameter("odom_delta_xy_alpha", 0.35)
        self.declare_parameter("odom_delta_xy_gain", 0.55)
        self.declare_parameter("odom_delta_yaw_alpha", 0.45)
        self.declare_parameter("odom_delta_yaw_gain", 0.45)
        self.declare_parameter("odom_pose_xy_alpha", 0.0)
        self.declare_parameter("odom_pose_xy_gain", 0.0)
        self.declare_parameter("odom_pose_xy_alpha_max", 1.0)
        self.declare_parameter("odom_guard_enabled", False)
        self.declare_parameter("odom_guard_xy_error_m", 0.0)
        self.declare_parameter("odom_guard_xy_anchor_base", 0.75)
        self.declare_parameter("odom_guard_xy_anchor_gain", 0.50)
        self.declare_parameter("odom_guard_yaw_error_rad", 0.0)
        self.declare_parameter("odom_guard_yaw_anchor_base", 0.70)
        self.declare_parameter("odom_guard_yaw_anchor_gain", 0.55)
        self.declare_parameter("odom_rebase_to_local_origin", False)
        self.declare_parameter("use_odom_corrections", True)
        self.declare_parameter("force_odom_pose", False)
        self.declare_parameter("odom_fallback_before_model_ready", True)

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
            f"[Robak] Model directory: {model_dir} (experiment output: {self.out_dir})"
        )
        self.torch_device_request = str(self.get_parameter("torch_device").value)
        self.torch_device_info = select_torch_device(self.torch_device_request)
        self.device = torch.device(self.torch_device_info.resolved)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.relay_scan_topic = str(self.get_parameter("relay_scan_topic").value).strip()
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.get_logger().info(
            f"[Robak] Torch device: requested={self.torch_device_info.requested}, "
            f"using={self.torch_device_info.resolved} ({self.torch_device_info.reason})"
        )
        if self.torch_device_info.warning:
            self.get_logger().warn(self.torch_device_info.warning)
        self.exp_logger.add_note(
            f"infer_robak_node torch device requested={self.torch_device_info.requested}, "
            f"resolved={self.torch_device_info.resolved}: {self.torch_device_info.reason}"
        )
        if self.torch_device_info.warning:
            self.exp_logger.add_note(self.torch_device_info.warning)
        self.exp_logger.save()

        self.init_from = str(self.get_parameter("init_from").value).lower()
        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.max_step_trans = float(self.get_parameter("max_step_trans").value)
        self.max_step_yaw = float(self.get_parameter("max_step_yaw").value)
        self.delta_ema_alpha = float(self.get_parameter("delta_ema_alpha").value)
        self.odom_heading_alpha = float(self.get_parameter("odom_heading_alpha").value)
        self.odom_heading_gain = float(self.get_parameter("odom_heading_gain").value)
        self.odom_sync_tolerance_sec = float(self.get_parameter("odom_sync_tolerance_sec").value)
        self.odom_delta_xy_alpha = float(self.get_parameter("odom_delta_xy_alpha").value)
        self.odom_delta_xy_gain = float(self.get_parameter("odom_delta_xy_gain").value)
        self.odom_delta_yaw_alpha = float(self.get_parameter("odom_delta_yaw_alpha").value)
        self.odom_delta_yaw_gain = float(self.get_parameter("odom_delta_yaw_gain").value)
        self.odom_pose_xy_alpha = float(self.get_parameter("odom_pose_xy_alpha").value)
        self.odom_pose_xy_gain = float(self.get_parameter("odom_pose_xy_gain").value)
        self.odom_pose_xy_alpha_max = float(self.get_parameter("odom_pose_xy_alpha_max").value)
        self.odom_guard_enabled = bool(self.get_parameter("odom_guard_enabled").value)
        self.odom_guard_xy_error_m = float(self.get_parameter("odom_guard_xy_error_m").value)
        self.odom_guard_xy_anchor_base = float(self.get_parameter("odom_guard_xy_anchor_base").value)
        self.odom_guard_xy_anchor_gain = float(self.get_parameter("odom_guard_xy_anchor_gain").value)
        self.odom_guard_yaw_error_rad = float(self.get_parameter("odom_guard_yaw_error_rad").value)
        self.odom_guard_yaw_anchor_base = float(self.get_parameter("odom_guard_yaw_anchor_base").value)
        self.odom_guard_yaw_anchor_gain = float(self.get_parameter("odom_guard_yaw_anchor_gain").value)
        self.odom_rebase_to_local_origin = bool(self.get_parameter("odom_rebase_to_local_origin").value)
        self.use_odom_corrections = bool(self.get_parameter("use_odom_corrections").value)
        self.force_odom_pose = bool(self.get_parameter("force_odom_pose").value)
        self.odom_fallback_before_model_ready = bool(
            self.get_parameter("odom_fallback_before_model_ready").value
        )

        self.prev_scan = None
        self.prev_stamp = None

        self.model = None
        self.x_mean_t = None
        self.x_std_t = None
        self.y_mean_t = None
        self.y_std_t = None

        self.infer_start = None
        self.inference_count = 0
        self.inference_times_ms = []

        # pose
        self.pose_inited = False
        self._bootstrap_pose_done = False
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        self.latest_gt = None
        self.latest_odom = None
        self.odom_buf = deque(maxlen=2000)
        self.odom_origin_xyth = None
        self.prev_odom_xyth = None
        self.dx_filt = None
        self.dy_filt = None
        self.dth_filt = None

        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_scan_relay = None
        if self.relay_scan_topic:
            self.pub_scan_relay = self.create_publisher(LaserScan, self.relay_scan_topic, qos_profile_sensor_data)
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.sub_gt = self.create_subscription(PoseStamped, self.gt_topic, self.on_gt, 50)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)

        self.timer = self.create_timer(0.05, self.try_load_model)
        self.stats_timer = self.create_timer(10.0, self.periodic_save_stats)

        self.get_logger().info(
            f"[Robak] infer stabilization: max_step_trans={self.max_step_trans}, "
            f"max_step_yaw={self.max_step_yaw}, ema={self.delta_ema_alpha}, "
            f"odom_heading_alpha={self.odom_heading_alpha}, odom_heading_gain={self.odom_heading_gain}, "
            f"odom_tol={self.odom_sync_tolerance_sec}, "
            f"odom_delta_xy_alpha={self.odom_delta_xy_alpha}, odom_delta_xy_gain={self.odom_delta_xy_gain}, "
            f"odom_delta_yaw_alpha={self.odom_delta_yaw_alpha}, odom_delta_yaw_gain={self.odom_delta_yaw_gain}, "
            f"odom_pose_xy_alpha={self.odom_pose_xy_alpha}, odom_pose_xy_gain={self.odom_pose_xy_gain}, "
            f"odom_pose_xy_alpha_max={self.odom_pose_xy_alpha_max}, "
            f"odom_guard(enabled={self.odom_guard_enabled}, "
            f"xy_err={self.odom_guard_xy_error_m}, yaw_err={self.odom_guard_yaw_error_rad}), "
            f"odom_rebase_to_local_origin={self.odom_rebase_to_local_origin}, "
            f"use_odom_corrections={self.use_odom_corrections}, "
            f"force_odom_pose={self.force_odom_pose}, "
            f"odom_fallback_before_model_ready={self.odom_fallback_before_model_ready}, "
            f"relay_scan_topic={self.relay_scan_topic or '(disabled)'}"
        )

    def on_gt(self, msg: PoseStamped):
        self.latest_gt = msg
        if not self.pose_inited and self.init_from == "gt":
            self.x, self.y, self.th = xytheta_from_pose_stamped(msg)
            self.pose_inited = True
        self._try_publish_bootstrap_pose()

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg
        t = _stamp_to_sec(msg.header.stamp)
        x, y, th = xytheta_from_odom(msg)
        if self.odom_rebase_to_local_origin:
            if self.odom_origin_xyth is None:
                self.odom_origin_xyth = (float(x), float(y), float(th))
            x, y, th = _rebase_pose((x, y, th), self.odom_origin_xyth)
        self.odom_buf.append((t, x, y, th))
        if not self.pose_inited and self.init_from == "odom":
            self.x, self.y, self.th = x, y, th
            self.pose_inited = True
        self._try_publish_bootstrap_pose()
        if self.model is None and self.odom_fallback_before_model_ready and self.pose_inited:
            self.x, self.y, self.th = x, y, th
            self._publish_pose_stamped(msg.header.stamp)

    def _publish_pose_stamped(self, stamp) -> None:
        """Publish PoseStamped + TF at current (x, y, th) with the given header stamp."""
        qx, qy, qz, qw = quat_from_yaw(self.th)
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.tf_parent
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
            tfm.header.stamp = stamp
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

    def _relay_scan(self, scan_msg: LaserScan) -> None:
        if self.pub_scan_relay is not None:
            self.pub_scan_relay.publish(scan_msg)

    def _try_publish_bootstrap_pose(self) -> None:
        """One-time pose/TF so eval can sync with GT before the second scan arrives."""
        if self._bootstrap_pose_done or (
            self.model is None
            and not self.force_odom_pose
            and not self.odom_fallback_before_model_ready
        ) or not self.pose_inited:
            return
        if self.init_from == "gt":
            if self.latest_gt is None:
                return
            self._publish_pose_stamped(self.latest_gt.header.stamp)
        elif self.init_from == "odom":
            if self.latest_odom is None:
                return
            self._publish_pose_stamped(self.latest_odom.header.stamp)
        else:
            return
        self._bootstrap_pose_done = True

    def _nearest_odom_xyth(self, t_scan: float):
        if not self.odom_buf:
            return None

        while len(self.odom_buf) > 2 and self.odom_buf[1][0] < (t_scan - 1.0):
            self.odom_buf.popleft()

        t_best, x_best, y_best, th_best = min(self.odom_buf, key=lambda x: abs(x[0] - t_scan))
        if abs(t_best - t_scan) > self.odom_sync_tolerance_sec:
            return None
        return float(x_best), float(y_best), float(th_best)

    def try_load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            return

        payload = torch.load(self.model_path, map_location="cpu")
        self.model = RobakConv1D()
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
                odom_topic="(unused)",
                pose_topic=self.pose_topic,
                tf_parent=self.tf_parent,
                tf_child=self.tf_child,
                model_path=self.model_path,
                torch_device_requested=self.torch_device_info.requested,
                torch_device_used=self.torch_device_info.resolved,
            )

        self.get_logger().info(f"[Robak] Model loaded: {self.model_path}")
        self._try_publish_bootstrap_pose()

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
        if self.model is None and not self.force_odom_pose and not self.odom_fallback_before_model_ready:
            return
        if self.init_from != "none" and not self.pose_inited:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        if self.prev_scan is None:
            self.prev_scan = scan
            self.prev_stamp = msg.header.stamp
            self._relay_scan(msg)
            return

        # dt
        t_prev = _stamp_to_sec(self.prev_stamp)
        t_cur = _stamp_to_sec(msg.header.stamp)
        dt = max(1e-3, t_cur - t_prev)
        need_odom = bool(
            self.force_odom_pose
            or (self.model is None and self.odom_fallback_before_model_ready)
            or self.use_odom_corrections
        )
        odom_xyth = self._nearest_odom_xyth(t_cur) if need_odom else None
        odom_th = float(odom_xyth[2]) if odom_xyth is not None else None

        if self.model is None and self.odom_fallback_before_model_ready and not self.force_odom_pose:
            if odom_xyth is None:
                self.prev_scan = scan
                self.prev_stamp = msg.header.stamp
                return
            self.x = float(odom_xyth[0])
            self.y = float(odom_xyth[1])
            self.th = float(odom_xyth[2])
            self._publish_pose_stamped(msg.header.stamp)
            self._relay_scan(msg)
            self.prev_scan = scan
            self.prev_stamp = msg.header.stamp
            self.prev_odom_xyth = odom_xyth
            return

        if self.force_odom_pose:
            if odom_xyth is None:
                self.prev_scan = scan
                self.prev_stamp = msg.header.stamp
                return
            self.x = float(odom_xyth[0])
            self.y = float(odom_xyth[1])
            self.th = float(odom_xyth[2])
            self._publish_pose_stamped(msg.header.stamp)
            self._relay_scan(msg)
            self.prev_scan = scan
            self.prev_stamp = msg.header.stamp
            self.prev_odom_xyth = odom_xyth
            return

        X_pair = np.stack([self.prev_scan, scan], axis=0).astype(np.float32)   # (2,360)
        x_flat = X_pair.reshape(-1)                                            # (720,)
        xt = torch.from_numpy(x_flat.reshape((1, -1))).to(self.device, dtype=torch.float32)

        t0 = time.perf_counter()
        synchronize_torch_device(self.device)
        with torch.inference_mode():
            xn = (xt - self.x_mean_t) / torch.clamp(self.x_std_t, min=1e-6)
            yn = self.model(xn.reshape((1, 2, 360))).reshape(-1)
            y = (yn * self.y_std_t + self.y_mean_t).detach().cpu().numpy().astype(np.float32)
        synchronize_torch_device(self.device)
        t1 = time.perf_counter()

        self.inference_times_ms.append((t1 - t0) * 1000.0)
        self.inference_count += 1
        dx, dy, dth = float(y[0]), float(y[1]), float(y[2])

        # Ogranicz pojedynczy krok i wygładź, żeby ograniczyć outliery.
        if self.max_step_trans > 0.0:
            trans = math.hypot(dx, dy)
            if trans > self.max_step_trans and trans > 1e-6:
                s = self.max_step_trans / trans
                dx *= s
                dy *= s
        if self.max_step_yaw > 0.0:
            dth = float(np.clip(dth, -self.max_step_yaw, self.max_step_yaw))

        dxo = dyo = dtho = 0.0
        if self.use_odom_corrections and self.prev_odom_xyth is not None and odom_xyth is not None:
            dxo, dyo, dtho = _delta_local(self.prev_odom_xyth, odom_xyth)
            w_xy_base = min(max(self.odom_delta_xy_alpha, 0.0), 1.0)
            w_yaw_base = min(max(self.odom_delta_yaw_alpha, 0.0), 1.0)
            w_xy_gain = max(0.0, self.odom_delta_xy_gain)
            w_yaw_gain = max(0.0, self.odom_delta_yaw_gain)
            xy_scale = self.max_step_trans if self.max_step_trans > 1e-3 else 0.10
            yaw_scale = self.max_step_yaw if self.max_step_yaw > 1e-3 else 0.25
            err_xy_step = math.hypot(dx - float(dxo), dy - float(dyo))
            err_yaw_step = abs(wrap(dth - float(dtho)))
            w_xy = min(max(w_xy_base + w_xy_gain * (err_xy_step / xy_scale), 0.0), 1.0)
            w_yaw = min(max(w_yaw_base + w_yaw_gain * (err_yaw_step / yaw_scale), 0.0), 1.0)
            dx = (1.0 - w_xy) * dx + w_xy * float(dxo)
            dy = (1.0 - w_xy) * dy + w_xy * float(dyo)
            dth = wrap((1.0 - w_yaw) * dth + w_yaw * float(dtho))

        if self.max_step_trans > 0.0:
            trans = math.hypot(dx, dy)
            if trans > self.max_step_trans and trans > 1e-6:
                s = self.max_step_trans / trans
                dx *= s
                dy *= s
        if self.max_step_yaw > 0.0:
            dth = float(np.clip(dth, -self.max_step_yaw, self.max_step_yaw))

        a = min(max(self.delta_ema_alpha, 0.0), 0.999)
        if self.dx_filt is None:
            self.dx_filt, self.dy_filt, self.dth_filt = dx, dy, dth
        else:
            self.dx_filt = a * self.dx_filt + (1.0 - a) * dx
            self.dy_filt = a * self.dy_filt + (1.0 - a) * dy
            self.dth_filt = a * self.dth_filt + (1.0 - a) * dth
        dx = float(self.dx_filt)
        dy = float(self.dy_filt)
        dth = float(self.dth_filt)

        # Integracja: delta jest „względem poprzedniego kroku” (local), więc składamy SE2
        c = math.cos(self.th)
        s = math.sin(self.th)
        self.x += c * dx - s * dy
        self.y += s * dx + c * dy
        self.th = wrap(self.th + dth)

        if self.use_odom_corrections and odom_th is not None and self.odom_heading_alpha > 0.0:
            heading_base = min(max(self.odom_heading_alpha, 0.0), 1.0)
            heading_gain = max(0.0, self.odom_heading_gain)
            yaw_err = abs(wrap(float(self.th) - float(odom_th)))
            yaw_scale = max(0.10, float(self.max_step_yaw) if self.max_step_yaw > 0.0 else 0.25)
            heading_w = min(max(heading_base + heading_gain * (yaw_err / yaw_scale), 0.0), 1.0)
            self.th = _interp_angle(self.th, odom_th, heading_w)
        if self.use_odom_corrections and self.odom_guard_enabled and odom_th is not None and self.odom_guard_yaw_error_rad > 0.0:
            yaw_err_guard = abs(wrap(float(self.th) - float(odom_th)))
            if yaw_err_guard > self.odom_guard_yaw_error_rad:
                yaw_guard_base = min(max(self.odom_guard_yaw_anchor_base, 0.0), 1.0)
                yaw_guard_gain = max(0.0, self.odom_guard_yaw_anchor_gain)
                yaw_guard_scale = max(1e-6, self.odom_guard_yaw_error_rad)
                yaw_guard_w = min(
                    max(yaw_guard_base + yaw_guard_gain * (yaw_err_guard / yaw_guard_scale), 0.0),
                    1.0,
                )
                self.th = _interp_angle(self.th, odom_th, yaw_guard_w)

        odom_pose_anchor_w_xy = 0.0
        err_to_odom_before_m = None
        err_to_odom_after_m = None
        if self.use_odom_corrections and odom_xyth is not None:
            ox, oy, _ = odom_xyth
            err_xy = math.hypot(self.x - float(ox), self.y - float(oy))
            err_to_odom_before_m = float(err_xy)
            if self.odom_pose_xy_alpha > 0.0 or self.odom_pose_xy_gain > 0.0:
                norm = self.max_step_trans if self.max_step_trans > 1e-3 else 0.10
                w_base = min(max(self.odom_pose_xy_alpha, 0.0), 1.0)
                w_gain = max(0.0, self.odom_pose_xy_gain)
                w_xy_max = min(max(self.odom_pose_xy_alpha_max, 0.0), 1.0)
                w_xy = min(max(w_base + w_gain * (err_xy / norm), 0.0), w_xy_max)
                odom_pose_anchor_w_xy = float(w_xy)
                self.x = (1.0 - w_xy) * self.x + w_xy * float(ox)
                self.y = (1.0 - w_xy) * self.y + w_xy * float(oy)
            if self.odom_guard_enabled and self.odom_guard_xy_error_m > 0.0:
                err_xy_guard = math.hypot(self.x - float(ox), self.y - float(oy))
                if err_xy_guard > self.odom_guard_xy_error_m:
                    guard_xy_base = min(max(self.odom_guard_xy_anchor_base, 0.0), 1.0)
                    guard_xy_gain = max(0.0, self.odom_guard_xy_anchor_gain)
                    guard_xy_scale = max(1e-6, self.odom_guard_xy_error_m)
                    guard_xy_w = min(
                        max(guard_xy_base + guard_xy_gain * (err_xy_guard / guard_xy_scale), 0.0),
                        1.0,
                    )
                    self.x = (1.0 - guard_xy_w) * self.x + guard_xy_w * float(ox)
                    self.y = (1.0 - guard_xy_w) * self.y + guard_xy_w * float(oy)
            err_to_odom_after_m = float(math.hypot(self.x - float(ox), self.y - float(oy)))

        self._publish_pose_stamped(msg.header.stamp)
        self._relay_scan(msg)

        self.prev_scan = scan
        self.prev_stamp = msg.header.stamp
        self.prev_odom_xyth = odom_xyth


def main():
    rclpy.init()
    node = InferRobakNode()
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
