import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float):
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


class GTPosePublisher(Node):
    def __init__(self):
        super().__init__("gt_pose_publisher")
        self.declare_parameter("in_topic", "/odom_raw")
        self.declare_parameter("tf_world_topic", "/tf_world")
        self.declare_parameter("out_topic", "/ground_truth_pose")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("use_tf_world", True)
        self.declare_parameter("tf_world_timeout_sec", 0.5)
        self.declare_parameter("model_name_hint", "diffbot")
        self.declare_parameter("base_link_hint", "base_link")
        self.declare_parameter("world_frame_hint", "world")
        self.declare_parameter("heuristic_max_score", 9.0)
        self.declare_parameter("heuristic_bootstrap_max_score", 64.0)
        self.declare_parameter("heuristic_max_step_m", 0.8)
        self.declare_parameter("publish_odom_fallback", False)
        self.declare_parameter("restamp_output_to_now", True)
        self.declare_parameter("propagate_tf_world_with_odom", True)
        self.declare_parameter("debug_every_n", 500)

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.tf_world_topic = str(self.get_parameter("tf_world_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.use_tf_world = bool(self.get_parameter("use_tf_world").value)
        self.tf_world_timeout_sec = float(self.get_parameter("tf_world_timeout_sec").value)
        self.model_name_hint = str(self.get_parameter("model_name_hint").value).lower().strip()
        self.base_link_hint = str(self.get_parameter("base_link_hint").value).lower().strip()
        self.world_frame_hint = str(self.get_parameter("world_frame_hint").value).lower().strip()
        self.heuristic_max_score = float(self.get_parameter("heuristic_max_score").value)
        self.heuristic_bootstrap_max_score = float(self.get_parameter("heuristic_bootstrap_max_score").value)
        self.heuristic_max_step_m = float(self.get_parameter("heuristic_max_step_m").value)
        self.publish_odom_fallback = bool(self.get_parameter("publish_odom_fallback").value)
        self.restamp_output_to_now = bool(self.get_parameter("restamp_output_to_now").value)
        self.propagate_tf_world_with_odom = bool(self.get_parameter("propagate_tf_world_with_odom").value)
        self.debug_every_n = int(self.get_parameter("debug_every_n").value)

        self.pub = self.create_publisher(PoseStamped, self.out_topic, 10)
        self.sub = self.create_subscription(Odometry, self.in_topic, self.on_odom, 50)
        self.sub_tf = None
        if self.use_tf_world:
            self.sub_tf = self.create_subscription(TFMessage, self.tf_world_topic, self.on_tf_world, 50)

        self.last_tf_world_stamp_sec = None
        self.n_tf_world = 0
        self.n_tf_world_propagated = 0
        self.n_odom_fallback = 0
        self.n_odom_fallback_suppressed = 0
        self.n_tf_world_no_match = 0
        self.n_tf_world_heuristic = 0
        self._last_tf_world_source = None
        self.latest_odom_xy = None
        self.latest_odom_pose = None
        self.last_gt_world_xyzt = None
        self.last_gt_world_pose = None
        self.last_gt_anchor_odom_pose = None

        self.get_logger().info(
            f"[GT] source: tf_world={self.use_tf_world} ({self.tf_world_topic}), "
            f"fallback odom={self.in_topic}, out={self.out_topic}, "
            f"publish_odom_fallback={self.publish_odom_fallback}, "
            f"restamp_output_to_now={self.restamp_output_to_now}, "
            f"propagate_tf_world_with_odom={self.propagate_tf_world_with_odom}"
        )

    @staticmethod
    def _frame_tokens(frame_id: str):
        frame = str(frame_id).strip().lower()
        if not frame:
            return []
        frame = frame.replace("::", "/").replace(":", "/")
        return [tok for tok in frame.split("/") if tok]

    @staticmethod
    def _tokens_match_hint(tokens, hint: str) -> bool:
        h = str(hint).strip().lower()
        if not h:
            return False
        return any((tok == h) or (h in tok) for tok in tokens)

    def _is_world_parent(self, parent_tokens) -> bool:
        if not self.world_frame_hint:
            return True
        if not parent_tokens:
            return True
        return self._tokens_match_hint(parent_tokens, self.world_frame_hint)

    def _match_score(self, tr) -> int:
        child_tokens = self._frame_tokens(tr.child_frame_id)
        parent_tokens = self._frame_tokens(tr.header.frame_id)
        if not child_tokens:
            return 0

        if not self._is_world_parent(parent_tokens):
            return 0

        score = 0
        child_base = child_tokens[-1]
        base_hint = self.base_link_hint or "base_link"

        if child_base == base_hint:
            score = 100
        if score == 0 and self.model_name_hint and child_base == self.model_name_hint:
            score = 70
        if score <= 0:
            return 0

        if self.model_name_hint and self._tokens_match_hint(child_tokens, self.model_name_hint):
            score += 5
        if self.world_frame_hint and parent_tokens and (
            parent_tokens[-1] == self.world_frame_hint or self.world_frame_hint in parent_tokens[-1]
        ):
            score += 2

        return score

    @staticmethod
    def _tf_brief(msg: TFMessage, max_items: int = 8) -> str:
        parts = []
        for tr in list(msg.transforms)[:max_items]:
            parent = str(tr.header.frame_id)
            child = str(tr.child_frame_id)
            parts.append(f"{parent}->{child}")
        return ", ".join(parts)

    def _select_unnamed_transform_by_odom(self, msg: TFMessage):
        """Fallback when the bridge does not preserve frame ids."""
        if self.latest_odom_xy is None and self.last_gt_world_xyzt is None:
            return None, None

        ox, oy = (None, None)
        if self.latest_odom_xy is not None:
            ox, oy = self.latest_odom_xy
        px, py, _pz, pt = self.last_gt_world_xyzt if self.last_gt_world_xyzt is not None else (None, None, None, None)
        best = None
        best_score = float("inf")
        best_d_prev2 = None
        best_d_odom2 = None

        dynamic_step_m = self.heuristic_max_step_m
        if pt is not None and len(msg.transforms) > 0:
            mt = _stamp_to_sec(msg.transforms[0].header.stamp)
            dt = max(0.0, mt - float(pt))
            dynamic_step_m = max(self.heuristic_max_step_m, 0.30 + 1.2 * dt)
        dynamic_step2 = dynamic_step_m * dynamic_step_m

        for tr in msg.transforms:
            parent = str(tr.header.frame_id).strip()
            child = str(tr.child_frame_id).strip()
            if parent or child:
                continue

            x = float(tr.transform.translation.x)
            y = float(tr.transform.translation.y)
            z = float(tr.transform.translation.z)
            if (not math.isfinite(x)) or (not math.isfinite(y)) or (not math.isfinite(z)):
                continue
            if z < -0.30 or z > 0.80:
                continue

            d_odom2 = None
            if self.latest_odom_xy is not None:
                d_odom2 = (x - ox) * (x - ox) + (y - oy) * (y - oy)

            d_prev2 = None
            if px is not None and py is not None:
                d_prev2 = (x - px) * (x - px) + (y - py) * (y - py)

            if d_prev2 is not None and d_prev2 <= dynamic_step2:
                score = d_prev2 + (0.10 * d_odom2 if d_odom2 is not None else 0.0) + 0.02 * abs(z - 0.10)
            else:
                if d_odom2 is None:
                    continue
                score = d_odom2 + 0.04 * abs(z - 0.10)

            if score < best_score:
                best_score = score
                best = tr
                best_d_prev2 = d_prev2
                best_d_odom2 = d_odom2

        if best is None:
            return None, None
        if best_d_prev2 is not None and best_d_prev2 <= dynamic_step2:
            return best, best_score
        if self.last_gt_world_xyzt is None and best_d_odom2 is not None and best_d_odom2 <= self.heuristic_bootstrap_max_score:
            return best, best_score
        if best_d_odom2 is not None and best_d_odom2 <= self.heuristic_max_score:
            return best, best_score
        return None, None

    def _remember_last_world_pose(self, stamp, pos, quat):
        try:
            t = _stamp_to_sec(stamp)
        except Exception:
            t = None
        if t is None:
            return

        yaw = _yaw_from_quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        self.last_gt_world_xyzt = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(t),
        )
        self.last_gt_world_pose = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(yaw),
            float(t),
        )
        if self.latest_odom_pose is not None:
            self.last_gt_anchor_odom_pose = self.latest_odom_pose

    def _publish_pose(self, stamp, frame_id: str, pos, quat, force_input_stamp: bool = False):
        ps = PoseStamped()
        if force_input_stamp or not self.restamp_output_to_now:
            ps.header.stamp = stamp
        else:
            ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        self.pub.publish(ps)

    def _publish_tf_world_pose(self, stamp, frame_id: str, pos, quat, source_tag):
        self._publish_pose(stamp, frame_id, pos, quat)
        self.last_tf_world_stamp_sec = _stamp_to_sec(stamp)
        self.n_tf_world += 1
        self._last_tf_world_source = source_tag
        self._remember_last_world_pose(stamp, pos, quat)

    def _publish_propagated_world_pose(self, msg: Odometry) -> bool:
        if not self.propagate_tf_world_with_odom:
            return False
        if self.last_gt_world_pose is None or self.last_gt_anchor_odom_pose is None or self.latest_odom_pose is None:
            return False

        wx0, wy0, wz0, wth0, _t_world = self.last_gt_world_pose
        ox0, oy0, oth0, _t_anchor = self.last_gt_anchor_odom_pose
        ox1, oy1, oth1, t_odom = self.latest_odom_pose

        if self.last_tf_world_stamp_sec is None:
            return False
        dt = t_odom - self.last_tf_world_stamp_sec
        if dt < 0.0 or dt > self.tf_world_timeout_sec:
            return False

        dx = ox1 - ox0
        dy = oy1 - oy0
        dth = _wrap_angle(oth1 - oth0)
        yaw = _wrap_angle(wth0 + dth)
        quat = _quat_from_yaw(yaw)
        self._publish_pose(
            msg.header.stamp,
            self.world_frame_hint or self.frame_id,
            (wx0 + dx, wy0 + dy, wz0),
            quat,
            force_input_stamp=True,
        )
        self.n_tf_world_propagated += 1
        if self.debug_every_n > 0 and self.n_tf_world_propagated % self.debug_every_n == 0:
            self.get_logger().info(
                f"[GT] propagated tf_world publishes={self.n_tf_world_propagated}, "
                f"last_tf={self._last_tf_world_source}"
            )
        return True

    def on_tf_world(self, msg: TFMessage):
        best = None
        best_score = -1

        for tr in msg.transforms:
            score = self._match_score(tr)
            if score > best_score:
                best_score = score
                best = tr

        if best is None or best_score <= 0:
            h_best, h_score = self._select_unnamed_transform_by_odom(msg)
            if h_best is not None:
                frame_id = self.world_frame_hint or self.frame_id
                pos = (
                    h_best.transform.translation.x,
                    h_best.transform.translation.y,
                    h_best.transform.translation.z,
                )
                quat = (
                    h_best.transform.rotation.x,
                    h_best.transform.rotation.y,
                    h_best.transform.rotation.z,
                    h_best.transform.rotation.w,
                )
                self._publish_tf_world_pose(h_best.header.stamp, frame_id, pos, quat, ("(heuristic)", "(unnamed)"))
                self.n_tf_world_heuristic += 1
                if self.debug_every_n > 0 and self.n_tf_world_heuristic % self.debug_every_n == 0:
                    self.get_logger().info(
                        f"[GT] heuristic tf_world fallback publishes={self.n_tf_world_heuristic}, "
                        f"score={h_score:.3f}"
                    )
                return

            self.n_tf_world_no_match += 1
            if self.debug_every_n > 0 and self.n_tf_world_no_match % self.debug_every_n == 0:
                self.get_logger().warn(
                    f"[GT] no matching tf_world transform "
                    f"(no_match={self.n_tf_world_no_match}, tf_msgs={self.n_tf_world + self.n_tf_world_no_match}); "
                    f"sample=[{self._tf_brief(msg)}]"
                )
            return

        frame_id = str(best.header.frame_id) if str(best.header.frame_id) else (self.world_frame_hint or self.frame_id)
        pos = (
            best.transform.translation.x,
            best.transform.translation.y,
            best.transform.translation.z,
        )
        quat = (
            best.transform.rotation.x,
            best.transform.rotation.y,
            best.transform.rotation.z,
            best.transform.rotation.w,
        )
        self._publish_tf_world_pose(best.header.stamp, frame_id, pos, quat, (str(best.header.frame_id), str(best.child_frame_id)))

        if self.debug_every_n > 0 and self.n_tf_world % self.debug_every_n == 0:
            self.get_logger().info(
                f"[GT] tf_world publishes={self.n_tf_world}, "
                f"tf_world_propagated={self.n_tf_world_propagated}, "
                f"odom fallback publishes={self.n_odom_fallback}, "
                f"last_tf={self._last_tf_world_source}"
            )

    def on_odom(self, msg: Odometry):
        ox = float(msg.pose.pose.position.x)
        oy = float(msg.pose.pose.position.y)
        oth = _yaw_from_quat(
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
            float(msg.pose.pose.orientation.w),
        )
        t_odom = _stamp_to_sec(msg.header.stamp)

        self.latest_odom_xy = (ox, oy)
        self.latest_odom_pose = (ox, oy, oth, t_odom)

        if self.use_tf_world and self.last_tf_world_stamp_sec is not None:
            dt = t_odom - self.last_tf_world_stamp_sec
            if 0.0 <= dt <= self.tf_world_timeout_sec:
                if self._publish_propagated_world_pose(msg):
                    return
                return

        if self.use_tf_world and not self.publish_odom_fallback:
            self.n_odom_fallback_suppressed += 1
            if self.debug_every_n > 0 and self.n_odom_fallback_suppressed % self.debug_every_n == 0:
                self.get_logger().warn(
                    f"[GT] suppressed odom fallback publishes={self.n_odom_fallback_suppressed}; "
                    "waiting for tf_world to avoid mixing GT and odometry frames."
                )
            return

        self._publish_pose(
            msg.header.stamp,
            self.frame_id,
            (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ),
            (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ),
            force_input_stamp=True,
        )
        self.n_odom_fallback += 1


def main():
    rclpy.init()
    node = GTPosePublisher()
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
