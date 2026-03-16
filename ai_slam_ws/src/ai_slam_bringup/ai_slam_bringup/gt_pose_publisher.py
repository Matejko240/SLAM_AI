import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


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
        self.declare_parameter("heuristic_max_step_m", 0.8)
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
        self.heuristic_max_step_m = float(self.get_parameter("heuristic_max_step_m").value)
        self.debug_every_n = int(self.get_parameter("debug_every_n").value)

        self.pub = self.create_publisher(PoseStamped, self.out_topic, 10)
        self.sub = self.create_subscription(Odometry, self.in_topic, self.on_odom, 50)
        self.sub_tf = None
        if self.use_tf_world:
            self.sub_tf = self.create_subscription(TFMessage, self.tf_world_topic, self.on_tf_world, 50)

        self.last_tf_world_stamp_sec = None
        self.n_tf_world = 0
        self.n_odom_fallback = 0
        self.n_tf_world_no_match = 0
        self.n_tf_world_heuristic = 0
        self._last_tf_world_source = None
        self.latest_odom_xy = None
        self.last_gt_world_xyzt = None

        self.get_logger().info(
            f"[GT] source: tf_world={self.use_tf_world} ({self.tf_world_topic}), "
            f"fallback odom={self.in_topic}, out={self.out_topic}"
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
            # Niektóre bridge potrafią zostawić pusty parent; dopuszczamy,
            # ale bez premii punktowej.
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

        # 1) Najlepiej: base_link.
        if child_base == base_hint:
            score = 100

        # 2) Fallback: model frame (np. world -> diffbot), gdy brak base_link.
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
        """Fallback gdy bridge nie niesie nazw ramek (puste parent/child)."""
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

        # Gdy mamy poprzedni GT w world, utrzymuj spójność ruchu robota.
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

            # Robot base_link w tym modelu jest blisko z~0.1m.
            if z < -0.30 or z > 0.80:
                continue

            d_odom2 = None
            if self.latest_odom_xy is not None:
                d_odom2 = (x - ox) * (x - ox) + (y - oy) * (y - oy)

            d_prev2 = None
            if px is not None and py is not None:
                d_prev2 = (x - px) * (x - px) + (y - py) * (y - py)

            if d_prev2 is not None and d_prev2 <= dynamic_step2:
                # Preferuj ciągłość toru; odometria tylko jako miękki tie-breaker.
                score = d_prev2 + (0.10 * d_odom2 if d_odom2 is not None else 0.0) + 0.02 * abs(z - 0.10)
            else:
                # Gdy ciągłość odpada, użyj odometrii.
                if d_odom2 is None:
                    continue
                score = d_odom2 + 0.04 * abs(z - 0.10)

            if score < best_score:
                best_score = score
                best = tr
                best_d_prev2 = d_prev2
                best_d_odom2 = d_odom2

        # Akceptacja: preferuj spójność toru, fallback do progu odometrii.
        if best is None:
            return None, None
        if best_d_prev2 is not None and best_d_prev2 <= dynamic_step2:
            return best, best_score
        if best_d_odom2 is not None and best_d_odom2 <= self.heuristic_max_score:
            return best, best_score
        return None, None

    def _remember_last_world_pose(self, stamp, pos):
        try:
            t = _stamp_to_sec(stamp)
        except Exception:
            t = None
        if t is None:
            return
        self.last_gt_world_xyzt = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(t),
        )

    def _publish_pose(self, stamp, frame_id: str, pos, quat):
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        self.pub.publish(ps)

    def on_tf_world(self, msg: TFMessage):
        best = None
        best_score = -1

        for tr in msg.transforms:
            score = self._match_score(tr)
            if score > best_score:
                best_score = score
                best = tr

        if best is None or best_score <= 0:
            # Fallback: gdy /tf_world ma puste frame ids (np. "->"), wybierz
            # najbardziej prawdopodobną pozycję robota po bliskości do odometrii.
            h_best, h_score = self._select_unnamed_transform_by_odom(msg)
            if h_best is not None:
                frame_id = self.world_frame_hint or self.frame_id
                self._publish_pose(
                    h_best.header.stamp,
                    frame_id,
                    (
                        h_best.transform.translation.x,
                        h_best.transform.translation.y,
                        h_best.transform.translation.z,
                    ),
                    (
                        h_best.transform.rotation.x,
                        h_best.transform.rotation.y,
                        h_best.transform.rotation.z,
                        h_best.transform.rotation.w,
                    ),
                )
                self.last_tf_world_stamp_sec = _stamp_to_sec(h_best.header.stamp)
                self.n_tf_world += 1
                self.n_tf_world_heuristic += 1
                self._last_tf_world_source = ("(heuristic)", "(unnamed)")
                self._remember_last_world_pose(
                    h_best.header.stamp,
                    (
                        h_best.transform.translation.x,
                        h_best.transform.translation.y,
                        h_best.transform.translation.z,
                    ),
                )
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
        self._publish_pose(
            best.header.stamp,
            frame_id,
            (
                best.transform.translation.x,
                best.transform.translation.y,
                best.transform.translation.z,
            ),
            (
                best.transform.rotation.x,
                best.transform.rotation.y,
                best.transform.rotation.z,
                best.transform.rotation.w,
            ),
        )

        self.last_tf_world_stamp_sec = _stamp_to_sec(best.header.stamp)
        self.n_tf_world += 1
        self._last_tf_world_source = (str(best.header.frame_id), str(best.child_frame_id))
        self._remember_last_world_pose(
            best.header.stamp,
            (
                best.transform.translation.x,
                best.transform.translation.y,
                best.transform.translation.z,
            ),
        )

        if self.debug_every_n > 0 and self.n_tf_world % self.debug_every_n == 0:
            self.get_logger().info(
                f"[GT] tf_world publishes={self.n_tf_world}, "
                f"odom fallback publishes={self.n_odom_fallback}, "
                f"last_tf={self._last_tf_world_source}"
            )

    def on_odom(self, msg: Odometry):
        self.latest_odom_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        if self.use_tf_world and self.last_tf_world_stamp_sec is not None:
            t_odom = _stamp_to_sec(msg.header.stamp)
            dt = t_odom - self.last_tf_world_stamp_sec
            if 0.0 <= dt <= self.tf_world_timeout_sec:
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
