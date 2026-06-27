import math
import os
import time
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

import torch

from .common import (
    seed_all,
    ensure_dir,
    quat_from_yaw,
    scan_motion_confidence,
    select_torch_device,
    synchronize_torch_device,
    wrap,
    xytheta_from_odom,
)
from .experiment_logger import ExperimentLogger
from .rywak_models import (
    build_rywak_model,
    model_type_from_payload,
    normalize_target_scaling,
    output_activation_for_target_scaling,
    parse_hidden_dims,
    unscale_targets_from_model_np,
)


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
        self.declare_parameter("relay_scan_topic", "")
        self.declare_parameter("relay_min_scan_confidence", 0.0)
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
        self.declare_parameter("anchor_yaw_to_odom_gain", 0.75)
        self.declare_parameter("anchor_xy_to_odom", 0.0)
        self.declare_parameter("anchor_xy_to_odom_gain", 0.0)
        self.declare_parameter("heading_for_xy_odom_weight", 0.60)
        self.declare_parameter("xy_step_odom_weight", 0.35)
        self.declare_parameter("xy_step_odom_gain", 0.45)
        self.declare_parameter("scan_confidence_low_rms", 0.02)
        self.declare_parameter("scan_confidence_high_rms", 0.18)
        self.declare_parameter("low_confidence_odom_gain", 0.30)
        self.declare_parameter("max_integration_dt", 0.20)
        self.declare_parameter("max_step_trans", 0.0)
        self.declare_parameter("max_step_yaw", 0.0)
        self.declare_parameter("v_bias_correction", 0.0)
        self.declare_parameter("w_bias_correction", 0.0)
        self.declare_parameter("odom_rebase_to_local_origin", False)
        self.declare_parameter("use_odom_corrections", True)
        self.declare_parameter("force_odom_pose", False)
        self.declare_parameter("odom_fallback_before_model_ready", True)
        self.declare_parameter("odom_guard_enabled", True)
        self.declare_parameter("odom_guard_fuse_weight", 0.95)
        self.declare_parameter("odom_guard_v_abs_diff", 0.35)
        self.declare_parameter("odom_guard_v_rel_diff", 0.60)
        self.declare_parameter("odom_guard_w_abs_diff", 0.80)
        self.declare_parameter("odom_guard_w_rel_diff", 0.80)
        self.declare_parameter("odom_guard_sign_conflict_speed", 0.08)
        self.declare_parameter("odom_guard_xy_error_m", 0.45)
        self.declare_parameter("odom_guard_xy_anchor_base", 0.80)
        self.declare_parameter("odom_guard_xy_anchor_gain", 0.70)
        self.declare_parameter("odom_guard_yaw_error_rad", 0.35)
        self.declare_parameter("odom_guard_yaw_anchor_base", 0.75)
        self.declare_parameter("odom_guard_yaw_anchor_gain", 0.50)
        self.declare_parameter("use_residual_odom_base", False)
        self.declare_parameter("residual_v_clip_abs", 0.0)
        self.declare_parameter("residual_w_clip_abs", 0.0)

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
        self.relay_scan_topic = str(self.get_parameter("relay_scan_topic").value).strip()
        self.relay_min_scan_confidence = float(
            self.get_parameter("relay_min_scan_confidence").value
        )
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
        self.init_odom_topic = str(self.get_parameter("init_from_odom_topic").value).strip()
        self.init_from_odom = bool(self.init_odom_topic)
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
        self.anchor_yaw_to_odom_gain = float(self.get_parameter("anchor_yaw_to_odom_gain").value)
        self.anchor_xy_to_odom = float(self.get_parameter("anchor_xy_to_odom").value)
        self.anchor_xy_to_odom_gain = float(self.get_parameter("anchor_xy_to_odom_gain").value)
        self.heading_for_xy_odom_weight = float(self.get_parameter("heading_for_xy_odom_weight").value)
        self.xy_step_odom_weight = float(self.get_parameter("xy_step_odom_weight").value)
        self.xy_step_odom_gain = float(self.get_parameter("xy_step_odom_gain").value)
        self.scan_confidence_low_rms = float(self.get_parameter("scan_confidence_low_rms").value)
        self.scan_confidence_high_rms = float(self.get_parameter("scan_confidence_high_rms").value)
        self.low_confidence_odom_gain = float(self.get_parameter("low_confidence_odom_gain").value)
        self.max_integration_dt = float(self.get_parameter("max_integration_dt").value)
        self.max_step_trans = float(self.get_parameter("max_step_trans").value)
        self.max_step_yaw = float(self.get_parameter("max_step_yaw").value)
        self.v_bias_correction = float(self.get_parameter("v_bias_correction").value)
        self.w_bias_correction = float(self.get_parameter("w_bias_correction").value)
        self.odom_rebase_to_local_origin = bool(self.get_parameter("odom_rebase_to_local_origin").value)
        self.use_odom_corrections = bool(self.get_parameter("use_odom_corrections").value)
        self.force_odom_pose = bool(self.get_parameter("force_odom_pose").value)
        self.odom_fallback_before_model_ready = bool(
            self.get_parameter("odom_fallback_before_model_ready").value
        )
        self.odom_guard_enabled = bool(self.get_parameter("odom_guard_enabled").value)
        self.odom_guard_fuse_weight = float(self.get_parameter("odom_guard_fuse_weight").value)
        self.odom_guard_v_abs_diff = float(self.get_parameter("odom_guard_v_abs_diff").value)
        self.odom_guard_v_rel_diff = float(self.get_parameter("odom_guard_v_rel_diff").value)
        self.odom_guard_w_abs_diff = float(self.get_parameter("odom_guard_w_abs_diff").value)
        self.odom_guard_w_rel_diff = float(self.get_parameter("odom_guard_w_rel_diff").value)
        self.odom_guard_sign_conflict_speed = float(
            self.get_parameter("odom_guard_sign_conflict_speed").value
        )
        self.odom_guard_xy_error_m = float(self.get_parameter("odom_guard_xy_error_m").value)
        self.odom_guard_xy_anchor_base = float(self.get_parameter("odom_guard_xy_anchor_base").value)
        self.odom_guard_xy_anchor_gain = float(self.get_parameter("odom_guard_xy_anchor_gain").value)
        self.odom_guard_yaw_error_rad = float(self.get_parameter("odom_guard_yaw_error_rad").value)
        self.odom_guard_yaw_anchor_base = float(self.get_parameter("odom_guard_yaw_anchor_base").value)
        self.odom_guard_yaw_anchor_gain = float(self.get_parameter("odom_guard_yaw_anchor_gain").value)
        self.use_residual_odom_base = bool(self.get_parameter("use_residual_odom_base").value)
        self.residual_v_clip_abs = float(self.get_parameter("residual_v_clip_abs").value)
        self.residual_w_clip_abs = float(self.get_parameter("residual_w_clip_abs").value)

        self.model = None
        self.model_type = "cnn"
        self.sequence_length = 1
        self.target_scaling = "zscore"
        self.target_tanh_meta = None
        self.x_mean_t = None
        self.x_std_t = None
        self.y_mean_t = None
        self.y_std_t = None
        self.in_dim = None
        self.out_dim = 2
        self._in_dim_warned = False

        self.prev_scan = None
        self.prev_stamp_sec = None
        self.last_pub_stamp_ns = None
        self.feature_history = deque(maxlen=1)
        # (stamp_sec, theta, v, w)
        self.odom_buf = deque(maxlen=2000)
        # (stamp_sec, vel_left, vel_right) — raw wheel angular velocities
        self.js_buf = deque(maxlen=2000)
        self.js_miss_count = 0
        self.odom_origin_xyth = None
        self.odom_miss_count = 0
        self.odom_interp_count = 0
        self.odom_nearest_count = 0
        self.latest_odom_msg = None
        self.latest_init_odom_msg = None
        self.v_filt = None
        self.w_filt = None
        self.dx_filt = None
        self.dy_filt = None
        self.dth_filt = None

        self.pose_inited = not self.init_from_odom
        self._bootstrap_pose_done = False
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        self.infer_start = None
        self._model_wait_start = time.time()
        self.inference_count = 0
        self.inference_times_ms = []

        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_scan_relay = None
        if self.relay_scan_topic:
            self.pub_scan_relay = self.create_publisher(LaserScan, self.relay_scan_topic, qos_profile_sensor_data)
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_js = self.create_subscription(JointState, '/joint_states', self.on_joint_state, 50)
        self.sub_init_odom = None
        if self.init_from_odom and self.init_odom_topic != self.odom_topic:
            self.sub_init_odom = self.create_subscription(Odometry, self.init_odom_topic, self.on_init_odom, 50)

        self.timer = self.create_timer(0.5, self.try_load_model)
        self.stats_timer = self.create_timer(10.0, self.periodic_save_stats)

        self.get_logger().info(
            f"[Rywak] sync: interp={self.interpolate_odom}, "
            f"tol={self.sync_tolerance_sec}, gap={self.sync_pair_gap_sec}, "
            f"delta_clip={self.delta_scan_clip}, v_clip={self.v_clip_abs}, w_clip={self.w_clip_abs}, "
            f"fuse(v={self.fuse_odom_v_weight}+{self.fuse_odom_v_gain}*err, "
            f"w={self.fuse_odom_w_weight}+{self.fuse_odom_w_gain}*err), "
            f"ema={self.vel_ema_alpha}, yaw_anchor={self.anchor_yaw_to_odom}+{self.anchor_yaw_to_odom_gain}*err, "
            f"xy_anchor={self.anchor_xy_to_odom}+{self.anchor_xy_to_odom_gain}*err, "
            f"xy_heading_odom_w={self.heading_for_xy_odom_weight}, "
            f"xy_step_odom_w={self.xy_step_odom_weight}+{self.xy_step_odom_gain}*err, "
            f"scan_conf=[{self.scan_confidence_low_rms},{self.scan_confidence_high_rms}], "
            f"low_conf_odom_gain={self.low_confidence_odom_gain}, "
            f"max_dt={self.max_integration_dt}, "
            f"max_step_trans={self.max_step_trans}, "
            f"max_step_yaw={self.max_step_yaw}, "
            f"bias_correction(v={self.v_bias_correction:.5f}, w={self.w_bias_correction:.5f}), "
            f"init_from_odom={self.init_from_odom}, "
            f"odom_rebase_to_local_origin={self.odom_rebase_to_local_origin}, "
            f"use_odom_corrections={self.use_odom_corrections}, "
            f"force_odom_pose={self.force_odom_pose}, "
            f"odom_fallback_before_model_ready={self.odom_fallback_before_model_ready}, "
            f"odom_guard(enabled={self.odom_guard_enabled}, "
            f"fuse={self.odom_guard_fuse_weight}, "
            f"xy_err={self.odom_guard_xy_error_m}, "
            f"yaw_err={self.odom_guard_yaw_error_rad}), "
            f"relay_scan_topic={self.relay_scan_topic or '(disabled)'}, "
            f"relay_min_scan_confidence={self.relay_min_scan_confidence:.2f}"
        )

    def _rebase_odom_pose(self, x: float, y: float, th: float):
        if not self.odom_rebase_to_local_origin:
            return float(x), float(y), float(th)
        if self.odom_origin_xyth is None:
            self.odom_origin_xyth = (float(x), float(y), float(th))
        return _rebase_pose((x, y, th), self.odom_origin_xyth)

    def on_init_odom(self, msg: Odometry):
        if not self.init_from_odom:
            return
        self.latest_init_odom_msg = msg
        if self.pose_inited:
            self._try_publish_bootstrap_pose()
            return
        x, y, th = xytheta_from_odom(msg)
        self.x, self.y, self.th = self._rebase_odom_pose(x, y, th)
        self.pose_inited = True
        self._try_publish_bootstrap_pose()

    def on_odom(self, msg: Odometry):
        self.latest_odom_msg = msg
        t = _stamp_to_sec(msg.header.stamp)
        x, y, th = xytheta_from_odom(msg)
        x, y, th = self._rebase_odom_pose(x, y, th)
        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)
        self.odom_buf.append((t, x, y, th, v, w))
        if self.init_from_odom and self.init_odom_topic == self.odom_topic and not self.pose_inited:
            self.x, self.y, self.th = x, y, th
            self.pose_inited = True
        self._try_publish_bootstrap_pose()
        if self.model is None and self.odom_fallback_before_model_ready and self.pose_inited:
            self.x, self.y, self.th = x, y, th
            self._publish_pose_stamped(msg.header.stamp)

    def on_joint_state(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp)
        vel_left = 0.0
        vel_right = 0.0
        for i, name in enumerate(msg.name):
            if name == 'left_wheel_joint' and i < len(msg.velocity):
                vel_left = float(msg.velocity[i])
            elif name == 'right_wheel_joint' and i < len(msg.velocity):
                vel_right = float(msg.velocity[i])
        self.js_buf.append((t, vel_left, vel_right))

    def _js_at(self, t_scan: float):
        """Find nearest joint state within sync tolerance."""
        if not self.js_buf:
            return None
        while len(self.js_buf) > 2 and self.js_buf[1][0] < (t_scan - 1.0):
            self.js_buf.popleft()
        t_best, vl_best, vr_best = min(self.js_buf, key=lambda x: abs(x[0] - t_scan))
        if abs(t_best - t_scan) > self.sync_tolerance_sec:
            return None
        return float(vl_best), float(vr_best)

    def _make_monotonic_stamp(self, stamp):
        stamp_ns = int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))
        if self.last_pub_stamp_ns is not None and stamp_ns <= self.last_pub_stamp_ns:
            stamp_ns = self.last_pub_stamp_ns + 1
        self.last_pub_stamp_ns = stamp_ns
        out = type(stamp)()
        out.sec = int(stamp_ns // 1_000_000_000)
        out.nanosec = int(stamp_ns % 1_000_000_000)
        return out

    def _publish_pose_stamped(self, stamp) -> None:
        stamp = self._make_monotonic_stamp(stamp)
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

    def _maybe_relay_scan(self, scan_msg: LaserScan, scan_confidence: float | None) -> None:
        if self.pub_scan_relay is None:
            return
        if scan_confidence is None:
            self.pub_scan_relay.publish(scan_msg)
            return
        if float(scan_confidence) >= self.relay_min_scan_confidence:
            self.pub_scan_relay.publish(scan_msg)

    def _try_publish_bootstrap_pose(self) -> None:
        if self._bootstrap_pose_done or (
            self.model is None
            and not self.force_odom_pose
            and not self.odom_fallback_before_model_ready
        ) or not self.pose_inited:
            return
        stamp_msg = self.latest_init_odom_msg or self.latest_odom_msg
        if stamp_msg is None:
            return
        self._publish_pose_stamped(stamp_msg.header.stamp)
        self._bootstrap_pose_done = True

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

    def _reset_feature_history(self) -> None:
        self.feature_history = deque(maxlen=max(1, int(self.sequence_length)))

    def _model_predict_raw(self, xt: torch.Tensor) -> np.ndarray:
        t0 = time.perf_counter()
        synchronize_torch_device(self.device)
        with torch.inference_mode():
            if xt.dim() == 2:
                xn = (xt - self.x_mean_t) / torch.clamp(self.x_std_t, min=1e-6)
            else:
                xn = (xt - self.x_mean_t.view(1, 1, -1)) / torch.clamp(
                    self.x_std_t.view(1, 1, -1), min=1e-6
                )
            yn = self.model(xn)
            y_model = yn.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
            y_raw = unscale_targets_from_model_np(
                y_model,
                target_scaling=self.target_scaling,
                y_mean=self.y_mean_t.detach().cpu().numpy().astype(np.float32),
                y_std=self.y_std_t.detach().cpu().numpy().astype(np.float32),
                target_tanh_meta=self.target_tanh_meta,
            ).reshape(-1)
        synchronize_torch_device(self.device)
        t1 = time.perf_counter()
        self.inference_times_ms.append((t1 - t0) * 1000.0)
        self.inference_count += 1
        return y_raw

    def try_load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            elapsed = time.time() - self._model_wait_start
            if elapsed > 60.0 and not self.odom_fallback_before_model_ready:
                self.get_logger().fatal(
                    f"[Rywak] Model not found after {elapsed:.0f}s: {self.model_path}. "
                    "odom_fallback_before_model_ready=false — no poses will ever be published. "
                    "Set rywak_model_source_experiment_id or run train phase first."
                )
                rclpy.shutdown()
            return

        payload = torch.load(self.model_path, map_location="cpu")
        self.in_dim = int(payload.get("in_dim", 720))
        self.out_dim = int(payload.get("out_dim", 2))
        self.model_type = model_type_from_payload(payload)
        self.sequence_length = max(1, int(payload.get("sequence_length", 1)))
        self.target_scaling = normalize_target_scaling(payload.get("target_scaling", "zscore"))
        self.target_tanh_meta = payload.get("target_tanh_meta")
        hidden_dims = parse_hidden_dims(payload.get("hidden_dims", [256, 128, 64]))
        dropout = float(payload.get("dropout", 0.0))
        self.model = build_rywak_model(
            model_type=self.model_type,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            sequence_length=self.sequence_length,
            output_activation=output_activation_for_target_scaling(self.target_scaling),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self._reset_feature_history()

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

        self.get_logger().info(
            f"[Rywak] Model loaded: {self.model_path} (type={self.model_type}, seq_len={self.sequence_length}, "
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, target_scaling={self.target_scaling})"
        )
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
        if (
            self.model is None
            and not self.force_odom_pose
            and not self.odom_fallback_before_model_ready
        ) or not self.pose_inited:
            return

        scan = _sanitize_scan(msg)
        if scan is None:
            return

        t_cur = _stamp_to_sec(msg.header.stamp)
        need_odom = bool(
            self.force_odom_pose
            or (self.model is None and self.odom_fallback_before_model_ready)
            or self.use_odom_corrections
            or self.use_residual_odom_base
        )
        odom_match = None
        if need_odom:
            odom_match = self._odom_at(t_cur)
            if odom_match is None:
                self.odom_miss_count += 1
                self._reset_feature_history()
                return
            x_odom, y_odom, th_cur, v_odom, w_odom = odom_match
        else:
            x_odom = float(self.x)
            y_odom = float(self.y)
            th_cur = float(self.th)
            v_odom = 0.0
            w_odom = 0.0

        if self.force_odom_pose:
            self.x = float(x_odom)
            self.y = float(y_odom)
            self.th = float(th_cur)
            self._reset_feature_history()
            self._publish_pose_stamped(msg.header.stamp)
            scan_conf = scan_motion_confidence(
                self.prev_scan,
                scan,
                low_rms=self.scan_confidence_low_rms,
                high_rms=self.scan_confidence_high_rms,
            )
            self._maybe_relay_scan(msg, scan_conf)
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            return

        if self.model is None and self.odom_fallback_before_model_ready:
            self.x = float(x_odom)
            self.y = float(y_odom)
            self.th = float(th_cur)
            self._reset_feature_history()
            self._publish_pose_stamped(msg.header.stamp)
            scan_conf = scan_motion_confidence(
                self.prev_scan,
                scan,
                low_rms=self.scan_confidence_low_rms,
                high_rms=self.scan_confidence_high_rms,
            )
            self._maybe_relay_scan(msg, scan_conf)
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            return

        if self.prev_scan is None:
            if odom_match is not None:
                self.x = float(x_odom)
                self.y = float(y_odom)
                self.th = float(th_cur)
            self._reset_feature_history()
            self._publish_pose_stamped(msg.header.stamp)
            self._maybe_relay_scan(msg, None)
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            return

        dt_raw = max(1e-3, t_cur - float(self.prev_stamp_sec))
        if (
            self.model_type in ("gru", "lstm")
            and self.sync_pair_gap_sec > 0.0
            and dt_raw > self.sync_pair_gap_sec
        ):
            self._reset_feature_history()
        dt = dt_raw
        if self.max_integration_dt > 0.0:
            dt = min(dt, self.max_integration_dt)

        if self.out_dim == 3:
            features = np.concatenate([self.prev_scan, scan], axis=0).astype(np.float32)
            if features.size != self.in_dim:
                if not self._in_dim_warned:
                    self.get_logger().warn(
                        f"[Rywak] Feature dim mismatch: got {features.size}, model expects {self.in_dim}. "
                        "Likely model/dataset mismatch for scan-pair mode."
                    )
                    self._in_dim_warned = True
                return
            xt = torch.from_numpy(features[None, :]).to(self.device, dtype=torch.float32)
            y = self._model_predict_raw(xt)

            dx = float(y[0])
            dy = float(y[1])
            dth = float(y[2])

            if self.max_step_trans > 0.0:
                trans = math.hypot(dx, dy)
                if trans > self.max_step_trans and trans > 1e-9:
                    scale = self.max_step_trans / trans
                    dx *= scale
                    dy *= scale
            if self.max_step_yaw > 0.0:
                dth = float(np.clip(dth, -self.max_step_yaw, self.max_step_yaw))

            alpha = min(max(self.vel_ema_alpha, 0.0), 0.999)
            if self.dx_filt is None:
                self.dx_filt = dx
                self.dy_filt = dy
                self.dth_filt = dth
            else:
                self.dx_filt = alpha * self.dx_filt + (1.0 - alpha) * dx
                self.dy_filt = alpha * self.dy_filt + (1.0 - alpha) * dy
                self.dth_filt = alpha * self.dth_filt + (1.0 - alpha) * dth
            dx = float(self.dx_filt)
            dy = float(self.dy_filt)
            dth = float(self.dth_filt)

            c = math.cos(self.th)
            s = math.sin(self.th)
            self.x += c * dx - s * dy
            self.y += s * dx + c * dy
            self.th = wrap(self.th + dth)

            self._publish_pose_stamped(msg.header.stamp)
            scan_conf = scan_motion_confidence(
                self.prev_scan,
                scan,
                low_rms=self.scan_confidence_low_rms,
                high_rms=self.scan_confidence_high_rms,
            )
            self._maybe_relay_scan(msg, scan_conf)
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            return
        if self.out_dim != 2:
            if not self._in_dim_warned:
                self.get_logger().warn(
                    f"[Rywak] Unsupported model out_dim={self.out_dim}. Expected 2 (v,w) or 3 (dx,dy,dtheta)."
                )
                self._in_dim_warned = True
            self._reset_feature_history()
            return

        delta_scan = (scan - self.prev_scan).astype(np.float32)
        if self.delta_scan_clip > 0.0:
            delta_scan = np.clip(delta_scan, -self.delta_scan_clip, self.delta_scan_clip).astype(np.float32)

        # Get raw wheel angular velocities from /joint_states (faithful to original thesis)
        js_match = self._js_at(t_cur)
        if js_match is None:
            self.js_miss_count += 1
            self._reset_feature_history()
            self.prev_scan = scan
            self.prev_stamp_sec = t_cur
            return
        vel_left, vel_right = js_match

        # Rywak features: [delta_scan(360), encoder_left_vel, encoder_right_vel] = 362
        x = np.concatenate(
            [delta_scan, np.asarray([vel_left, vel_right], dtype=np.float32)], axis=0
        ).astype(np.float32)
        if x.size != self.in_dim:
            if not self._in_dim_warned:
                self.get_logger().warn(
                    f"[Rywak] Feature dim mismatch: got {x.size}, model expects {self.in_dim}. "
                    "Likely old model incompatible with current feature pipeline."
                )
                self._in_dim_warned = True
            self._reset_feature_history()
            return
        if self.model_type in ("gru", "lstm"):
            self.feature_history.append(x)
            if len(self.feature_history) < self.sequence_length:
                self._publish_pose_stamped(msg.header.stamp)
                self._maybe_relay_scan(msg, None)
                self.prev_scan = scan
                self.prev_stamp_sec = t_cur
                return
            x_seq = np.stack(list(self.feature_history), axis=0).astype(np.float32)
            xt = torch.from_numpy(x_seq[None, :, :]).to(self.device, dtype=torch.float32)
        else:
            xt = torch.from_numpy(x[None, :]).to(self.device, dtype=torch.float32)

        y = self._model_predict_raw(xt)

        v_pred = float(y[0])
        w_pred = float(y[1])

        # Residual mode: model predicts correction over odometry (v_gt - v_odom, w_gt - w_odom).
        # v_odom is from wheel encoders — same signal that enters the feature vector.
        # residual_v/w_clip_abs optionally bounds the correction BEFORE adding odometry,
        # preventing the model from applying physically unreasonable corrections.
        if self.use_residual_odom_base:
            v_residual = v_pred
            w_residual = w_pred
            if self.residual_v_clip_abs > 0.0:
                v_residual = float(np.clip(v_residual, -self.residual_v_clip_abs, self.residual_v_clip_abs))
            if self.residual_w_clip_abs > 0.0:
                w_residual = float(np.clip(w_residual, -self.residual_w_clip_abs, self.residual_w_clip_abs))
            v_pred = float(v_odom) + v_residual
            w_pred = float(w_odom) + w_residual

        # Bias correction (subtract systematic model bias)
        v_pred -= self.v_bias_correction
        w_pred -= self.w_bias_correction

        v_scale = self.v_clip_abs if self.v_clip_abs > 0.0 else 0.5
        w_scale = self.w_clip_abs if self.w_clip_abs > 0.0 else 1.0
        scan_conf = scan_motion_confidence(
            self.prev_scan,
            scan,
            low_rms=self.scan_confidence_low_rms,
            high_rms=self.scan_confidence_high_rms,
        )
        low_conf_boost = max(0.0, self.low_confidence_odom_gain) * (1.0 - scan_conf)
        if self.use_odom_corrections:
            v_base = min(max(self.fuse_odom_v_weight, 0.0), 1.0)
            w_base = min(max(self.fuse_odom_w_weight, 0.0), 1.0)
            v_gain = max(0.0, self.fuse_odom_v_gain)
            w_gain = max(0.0, self.fuse_odom_w_gain)
            v_err = abs(v_pred - float(v_odom)) / max(v_scale, 1e-3)
            w_err = abs(w_pred - float(w_odom)) / max(w_scale, 1e-3)
            wv = min(max(v_base + low_conf_boost + v_gain * v_err, 0.0), 1.0)
            ww = min(max(w_base + low_conf_boost + w_gain * w_err, 0.0), 1.0)
            if self.odom_guard_enabled:
                odom_guard_fuse_w = min(max(self.odom_guard_fuse_weight, 0.0), 1.0)
                odom_guard_sign_v = max(0.01, self.odom_guard_sign_conflict_speed)
                v_sign_conflict = (
                    float(v_odom) * float(v_pred) < 0.0
                    and (abs(float(v_odom)) > odom_guard_sign_v or abs(float(v_pred)) > odom_guard_sign_v)
                )
                w_sign_conflict = (
                    float(w_odom) * float(w_pred) < 0.0
                    and (abs(float(w_odom)) > odom_guard_sign_v or abs(float(w_pred)) > odom_guard_sign_v)
                )
                v_guard_limit = max(0.0, self.odom_guard_v_abs_diff) + max(0.0, self.odom_guard_v_rel_diff) * max(
                    abs(float(v_odom)),
                    odom_guard_sign_v,
                )
                w_guard_limit = max(0.0, self.odom_guard_w_abs_diff) + max(0.0, self.odom_guard_w_rel_diff) * max(
                    abs(float(w_odom)),
                    odom_guard_sign_v,
                )
                if v_sign_conflict or abs(v_pred - float(v_odom)) > v_guard_limit:
                    wv = max(wv, odom_guard_fuse_w)
                if w_sign_conflict or abs(w_pred - float(w_odom)) > w_guard_limit:
                    ww = max(ww, odom_guard_fuse_w)
            v = (1.0 - wv) * v_pred + wv * float(v_odom)
            w = (1.0 - ww) * w_pred + ww * float(w_odom)
        else:
            v = v_pred
            w = w_pred

        if self.v_clip_abs > 0.0:
            v = float(np.clip(v, -self.v_clip_abs, self.v_clip_abs))
        if self.w_clip_abs > 0.0:
            w = float(np.clip(w, -self.w_clip_abs, self.w_clip_abs))

        alpha_ema = min(max(self.vel_ema_alpha + 0.30 * (1.0 - scan_conf), 0.0), 0.999)
        # Keep EMA adaptive thresholds independent from configured clip range.
        # Large clip values (e.g. 5 m/s) were masking meaningful sign/delta changes.
        v_flip_mag_th = 0.08
        w_flip_mag_th = 0.20
        v_jump_th = 0.35
        w_jump_th = 0.75
        if self.v_filt is not None:
            v_sign_flip = (self.v_filt * v < 0.0) and (abs(self.v_filt) > v_flip_mag_th or abs(v) > v_flip_mag_th)
            v_jump = abs(self.v_filt - v) > v_jump_th
            v_odom_conflict = self.use_odom_corrections and (float(v_odom) * v < 0.0) and (abs(float(v_odom)) > v_flip_mag_th) and (abs(v) > 0.04)
            v_odom_delta_conflict = False
            if self.use_odom_corrections and self.odom_guard_enabled:
                v_guard_limit = max(0.0, self.odom_guard_v_abs_diff) + max(0.0, self.odom_guard_v_rel_diff) * max(
                    abs(float(v_odom)),
                    max(0.01, self.odom_guard_sign_conflict_speed),
                )
                v_odom_delta_conflict = abs(v - float(v_odom)) > v_guard_limit
            if v_sign_flip or v_jump or v_odom_conflict or v_odom_delta_conflict:
                alpha_ema = min(alpha_ema, 0.2)
        if self.w_filt is not None:
            w_sign_flip = (self.w_filt * w < 0.0) and (abs(self.w_filt) > w_flip_mag_th or abs(w) > w_flip_mag_th)
            w_jump = abs(self.w_filt - w) > w_jump_th
            w_odom_delta_conflict = False
            if self.use_odom_corrections and self.odom_guard_enabled:
                w_guard_limit = max(0.0, self.odom_guard_w_abs_diff) + max(0.0, self.odom_guard_w_rel_diff) * max(
                    abs(float(w_odom)),
                    max(0.01, self.odom_guard_sign_conflict_speed),
                )
                w_odom_delta_conflict = abs(w - float(w_odom)) > w_guard_limit
            if w_sign_flip or w_jump or w_odom_delta_conflict:
                alpha_ema = min(alpha_ema, 0.2)
        if self.v_filt is None:
            self.v_filt = v
            self.w_filt = w
        else:
            self.v_filt = alpha_ema * self.v_filt + (1.0 - alpha_ema) * v
            self.w_filt = alpha_ema * self.w_filt + (1.0 - alpha_ema) * w
        v = float(self.v_filt)
        w = float(self.w_filt)

        # unicycle integration
        prev_x = float(self.x)
        prev_y = float(self.y)
        prev_th = float(self.th)
        th_pred = wrap(self.th + w * dt)
        if self.use_odom_corrections:
            yaw_anchor_base = min(max(self.anchor_yaw_to_odom, 0.0), 1.0)
            yaw_anchor_gain = max(0.0, self.anchor_yaw_to_odom_gain)
            yaw_err = abs(wrap(float(th_pred) - float(th_cur)))
            yaw_scale = max(0.10, abs(float(w_odom)) * dt)
            yaw_anchor = min(max(yaw_anchor_base + 0.5 * low_conf_boost + yaw_anchor_gain * (yaw_err / yaw_scale), 0.0), 1.0)
            self.th = _interp_angle(th_pred, float(th_cur), yaw_anchor)
        else:
            self.th = th_pred
        if self.max_step_yaw > 0.0:
            dth = wrap(self.th - prev_th)
            dth = float(np.clip(dth, -self.max_step_yaw, self.max_step_yaw))
            self.th = wrap(prev_th + dth)
        if self.use_odom_corrections and self.odom_guard_enabled and self.odom_guard_yaw_error_rad > 0.0:
            yaw_err_after = abs(wrap(float(self.th) - float(th_cur)))
            if yaw_err_after > self.odom_guard_yaw_error_rad:
                yaw_guard_base = min(max(self.odom_guard_yaw_anchor_base, 0.0), 1.0)
                yaw_guard_gain = max(0.0, self.odom_guard_yaw_anchor_gain)
                yaw_guard_scale = max(1e-6, self.odom_guard_yaw_error_rad)
                yaw_guard_w = min(max(yaw_guard_base + yaw_guard_gain * (yaw_err_after / yaw_guard_scale), 0.0), 1.0)
                self.th = _interp_angle(self.th, float(th_cur), yaw_guard_w)

        heading_w = min(max(self.heading_for_xy_odom_weight, 0.0), 1.0) if self.use_odom_corrections else 0.0
        th_xy = _interp_angle(self.th, float(th_cur), heading_w)
        step_pred_x = v * dt * math.cos(th_xy)
        step_pred_y = v * dt * math.sin(th_xy)
        step_odom_x = float(v_odom) * dt * math.cos(float(th_cur))
        step_odom_y = float(v_odom) * dt * math.sin(float(th_cur))
        if self.use_odom_corrections:
            step_base = min(max(self.xy_step_odom_weight, 0.0), 1.0)
            step_gain = max(0.0, self.xy_step_odom_gain)
            step_err = abs(v - float(v_odom)) / max(v_scale, 1e-3)
            step_w = min(max(step_base + low_conf_boost + step_gain * step_err, 0.0), 1.0)
        else:
            step_w = 0.0
        self.x += (1.0 - step_w) * step_pred_x + step_w * step_odom_x
        self.y += (1.0 - step_w) * step_pred_y + step_w * step_odom_y

        xy_anchor_base = min(max(self.anchor_xy_to_odom, 0.0), 1.0)
        xy_anchor_gain = max(0.0, self.anchor_xy_to_odom_gain)
        if self.use_odom_corrections and (xy_anchor_base > 0.0 or xy_anchor_gain > 0.0):
            err_xy = math.hypot(self.x - float(x_odom), self.y - float(y_odom))
            step_scale = max(abs(float(v_odom)) * dt, 0.05)
            anchor_w = min(max(xy_anchor_base + xy_anchor_gain * (err_xy / step_scale), 0.0), 1.0)
            self.x = (1.0 - anchor_w) * self.x + anchor_w * float(x_odom)
            self.y = (1.0 - anchor_w) * self.y + anchor_w * float(y_odom)
        if self.use_odom_corrections and self.odom_guard_enabled and self.odom_guard_xy_error_m > 0.0:
            err_xy_guard = math.hypot(self.x - float(x_odom), self.y - float(y_odom))
            if err_xy_guard > self.odom_guard_xy_error_m:
                guard_xy_base = min(max(self.odom_guard_xy_anchor_base, 0.0), 1.0)
                guard_xy_gain = max(0.0, self.odom_guard_xy_anchor_gain)
                guard_xy_scale = max(1e-6, self.odom_guard_xy_error_m)
                guard_xy_w = min(max(guard_xy_base + guard_xy_gain * (err_xy_guard / guard_xy_scale), 0.0), 1.0)
                self.x = (1.0 - guard_xy_w) * self.x + guard_xy_w * float(x_odom)
                self.y = (1.0 - guard_xy_w) * self.y + guard_xy_w * float(y_odom)
        if self.max_step_trans > 0.0:
            dx = self.x - prev_x
            dy = self.y - prev_y
            step = math.hypot(dx, dy)
            if step > self.max_step_trans and step > 1e-9:
                scale = self.max_step_trans / step
                self.x = prev_x + dx * scale
                self.y = prev_y + dy * scale

        self._publish_pose_stamped(msg.header.stamp)
        self._maybe_relay_scan(msg, scan_conf)

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
