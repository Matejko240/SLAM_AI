"""
Sterownik jazdy wzdłuż ścieżki z pliku YAML (kotwice + opcjonalnie A* na mapie referencyjnej).
Domyślnie feedback z /ground_truth_pose (te same współrzędne co świat Gazebo / mapa ref.).
"""
from __future__ import annotations

import math
import os

import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker

from ai_slam_bringup.occupancy_grid_plan import (
    densify_polyline,
    load_reference_map,
    plan_polyline_through_anchors,
)


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _yaw_from_quat(z: float, w: float) -> float:
    return math.atan2(2.0 * w * z, w * w - z * z)


def _chain_densify_straight(anchors: list[tuple[float, float]], step_m: float) -> list[tuple[float, float]]:
    if len(anchors) < 2:
        return list(anchors)
    out: list[tuple[float, float]] = [anchors[0]]
    for i in range(len(anchors) - 1):
        seg = densify_polyline([anchors[i], anchors[i + 1]], step_m)
        out.extend(seg[1:])
    return out


def _lookahead(
    path: list[tuple[float, float]],
    px: float,
    py: float,
    start_idx: int,
    lookahead_m: float,
    nearest_backtrack_points: int,
    nearest_horizon_m: float,
    ignore_passed_points: bool = False,
) -> tuple[float, float, int]:
    if not path:
        return px, py, 0
    n = len(path)
    start = max(0, min(int(start_idx), n - 1))
    backtrack = max(0, int(nearest_backtrack_points))
    horizon_m = max(float(lookahead_m), float(nearest_horizon_m))

    # Szukaj najbliższego punktu lokalnie: niewielki krok wstecz + ograniczony horyzont do przodu.
    # Chroni to przed "teleportacją indeksu" na dalszą część trasy przy jej samozbliżeniach.
    i0 = start if ignore_passed_points else max(0, start - backtrack)
    i1 = start
    acc_h = 0.0
    while i1 < n - 1 and acc_h < horizon_m:
        x0, y0 = path[i1]
        x1, y1 = path[i1 + 1]
        acc_h += math.hypot(x1 - x0, y1 - y0)
        i1 += 1
    i1 = max(i0, min(i1, n - 1))

    best_i = start
    best_d = float("inf")
    for i in range(i0, i1 + 1):
        d = (path[i][0] - px) ** 2 + (path[i][1] - py) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    if ignore_passed_points and best_i < start:
        best_i = start

    # Uwaga: zwracany indeks to "best_i" (najbliższy punkt referencyjny),
    # a nie indeks odcinka lookahead. Dzięki temu _path_idx nie "odpływa"
    # do przodu, gdy robot stoi lub ma chwilowy poślizg.
    acc = 0.0
    L = float(lookahead_m)
    for i in range(best_i, len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-9:
            continue
        if acc + seg >= L:
            t = (L - acc) / seg
            return x0 + t * dx, y0 + t * dy, best_i
        acc += seg
    return path[-1][0], path[-1][1], best_i


def _update_turn_in_place_mode(
    active: bool,
    err_abs: float,
    heading_stop_rad: float,
    heading_resume_rad: float,
) -> bool:
    """
    Histereza trybu obrotu w miejscu:
    - wejście gdy |err| >= heading_stop_rad
    - wyjście gdy |err| <= heading_resume_rad
    """
    if active:
        return err_abs > heading_resume_rad
    return err_abs >= heading_stop_rad


def _resolve_turn_direction_sign(
    err: float,
    err_abs: float,
    prev_sign: float,
    pi_ambiguity_guard_rad: float,
    preferred_sign: float = 1.0,
) -> float:
    """
    Przy błędzie bliskim +/-pi znak kąta może losowo przechodzić przez granicę wrapowania.
    Trzymaj poprzedni kierunek obrotu, dopóki nie wyjdziemy poza obszar niejednoznaczności.
    """
    pref = 1.0 if preferred_sign >= 0.0 else -1.0
    if prev_sign == 0.0:
        if err_abs >= (math.pi - pi_ambiguity_guard_rad):
            return pref
        return 1.0 if err >= 0.0 else -1.0
    if err_abs <= (math.pi - pi_ambiguity_guard_rad):
        return 1.0 if err >= 0.0 else -1.0
    return prev_sign


class PlannedPathDriver(Node):
    def __init__(self) -> None:
        super().__init__("planned_path_driver")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("pose_topic", "/ground_truth_pose")
        self.declare_parameter("path_spec_yaml", "")
        self.declare_parameter("reference_map_yaml", "")
        self.declare_parameter("use_astar", False)
        self.declare_parameter("map_flip_y", True)
        self.declare_parameter("inflate_robot_m", 0.35)
        self.declare_parameter("dense_step_m", 0.2)
        self.declare_parameter("lookahead_m", 0.35)
        self.declare_parameter("linear_vel_max", 1.2)
        self.declare_parameter("angular_vel_max", 2.4)
        self.declare_parameter("goal_tolerance_m", 0.22)
        self.declare_parameter("loop_path", True)
        self.declare_parameter("heading_gain", 2.2)
        self.declare_parameter("heading_stop_deg", 55.0)
        self.declare_parameter("heading_resume_deg", 35.0)
        self.declare_parameter("alignment_cos_power", 2.0)
        self.declare_parameter("turn_direction_guard_deg", 18.0)
        self.declare_parameter("turn_direction_preference", 1.0)
        self.declare_parameter("nearest_backtrack_points", 8)
        self.declare_parameter("nearest_horizon_m", 6.0)
        self.declare_parameter("ignore_passed_points", True)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("stop_on_path_error", False)
        # Optional dataset excitation: broaden v/omega histogram coverage on fixed trajectories.
        self.declare_parameter("dataset_excitation_enabled", False)
        self.declare_parameter("excitation_period_sec", 12.0)
        self.declare_parameter("excitation_v_min_scale", 0.25)
        self.declare_parameter("excitation_v_max_scale", 1.0)
        self.declare_parameter("excitation_heading_bias_deg", 12.0)
        self.declare_parameter("publish_reference_path_marker", True)
        self.declare_parameter("reference_path_marker_topic", "/planned_path_reference")
        self.declare_parameter("reference_path_dense_marker_topic", "/planned_path_dense")
        self.declare_parameter("reference_path_marker_frame", "world")
        self.declare_parameter("publish_dense_path_marker", True)
        self.declare_parameter("publish_completion_topic", True)
        self.declare_parameter("completion_topic", "/planned_path_done")

        spec_path = str(self.get_parameter("path_spec_yaml").value).strip()
        if not spec_path or not os.path.isfile(spec_path):
            raise RuntimeError(f"path_spec_yaml missing or not a file: {spec_path!r}")

        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}

        anchors_raw = spec.get("anchors") or spec.get("waypoints") or []
        anchors: list[tuple[float, float]] = []
        for a in anchors_raw:
            if not isinstance(a, dict):
                continue
            anchors.append((float(a["x"]), float(a["y"])))
        if len(anchors) < 2:
            raise RuntimeError("Planned path: need at least 2 anchors/waypoints")

        dense_step = float(spec.get("dense_step_m", float(self.get_parameter("dense_step_m").value)))
        use_astar = bool(spec.get("use_astar", bool(self.get_parameter("use_astar").value)))
        ref_map = str(self.get_parameter("reference_map_yaml").value).strip()
        map_flip_y = bool(spec.get("map_flip_y", bool(self.get_parameter("map_flip_y").value)))
        inflate_m = float(spec.get("inflate_robot_m", float(self.get_parameter("inflate_robot_m").value)))

        self._collision_poly: list[tuple[float, float]] = []
        if use_astar:
            if not ref_map or not os.path.isfile(ref_map):
                raise RuntimeError("use_astar=True wymaga istniejącego reference_map_yaml (abs lub share).")
            blocked, meta = load_reference_map(ref_map)
            res = float(meta["resolution"])
            inflate_cells = max(1, int(math.ceil(inflate_m / res)))
            try:
                poly = plan_polyline_through_anchors(
                    anchors,
                    blocked,
                    meta,
                    flip_y=map_flip_y,
                    inflate_cells=inflate_cells,
                )
            except ValueError as e:
                self.get_logger().error(f"A* / map planning failed: {e}")
                raise
            self._collision_poly = list(poly)
            self._path = densify_polyline(poly, dense_step)
        else:
            # Łamana przez kotwice — niegwarantowana wolna od kolizji z mapą; tylko podgląd w RViz.
            self._collision_poly = list(anchors)
            self._path = _chain_densify_straight(anchors, dense_step)

        self._lookahead_m = float(self.get_parameter("lookahead_m").value)
        self._v_max = float(self.get_parameter("linear_vel_max").value)
        self._w_max = float(self.get_parameter("angular_vel_max").value)
        self._goal_tol = float(self.get_parameter("goal_tolerance_m").value)
        spec_loop = spec.get("loop_path")
        if spec_loop is not None:
            self._loop = bool(spec_loop)
        else:
            self._loop = bool(self.get_parameter("loop_path").value)
        self._kh = float(self.get_parameter("heading_gain").value)
        heading_stop_deg = float(self.get_parameter("heading_stop_deg").value)
        heading_resume_deg = float(self.get_parameter("heading_resume_deg").value)
        if heading_resume_deg > heading_stop_deg - 1.0:
            heading_resume_deg = max(1.0, heading_stop_deg - 1.0)
            self.get_logger().warning(
                "heading_resume_deg > heading_stop_deg-1; clamped to "
                f"{heading_resume_deg:.1f} deg"
            )
        self._heading_stop_rad = math.radians(heading_stop_deg)
        self._heading_resume_rad = math.radians(heading_resume_deg)
        self._turn_direction_guard_rad = math.radians(
            max(0.0, min(45.0, float(self.get_parameter("turn_direction_guard_deg").value)))
        )
        self._turn_direction_preference = (
            1.0 if float(self.get_parameter("turn_direction_preference").value) >= 0.0 else -1.0
        )
        self._alignment_cos_power = max(1.0, float(self.get_parameter("alignment_cos_power").value))
        self._nearest_backtrack_points = max(0, int(self.get_parameter("nearest_backtrack_points").value))
        self._nearest_horizon_m = max(0.5, float(self.get_parameter("nearest_horizon_m").value))
        self._ignore_passed_points = bool(self.get_parameter("ignore_passed_points").value)
        self._excitation_enabled = bool(self.get_parameter("dataset_excitation_enabled").value)
        self._exc_period = max(1.0, float(self.get_parameter("excitation_period_sec").value))
        self._exc_v_min = float(self.get_parameter("excitation_v_min_scale").value)
        self._exc_v_max = float(self.get_parameter("excitation_v_max_scale").value)
        if self._exc_v_min > self._exc_v_max:
            self._exc_v_min, self._exc_v_max = self._exc_v_max, self._exc_v_min
        self._exc_v_min = max(0.02, min(1.0, self._exc_v_min))
        # Pozwalamy na szerszą modulację v w dataset mode, żeby skuteczniej pokryć
        # górne koszyki histogramu prędkości liniowej.
        self._exc_v_max = max(self._exc_v_min, min(2.2, self._exc_v_max))
        self._exc_heading_bias_rad = math.radians(
            float(self.get_parameter("excitation_heading_bias_deg").value)
        )
        self._t0 = float(self.get_clock().now().nanoseconds) * 1e-9
        self._completion_sent = False
        self._path_idx = 0
        self._turn_in_place = False
        self._turn_direction_sign = 0.0
        self._px = self._py = 0.0
        self._yaw = 0.0
        self._pose_ok = False

        self.get_logger().info(
            f"[planned_path] points={len(self._path)} astar={use_astar} loop={self._loop} "
            f"spec={spec_path} ignore_passed_points={self._ignore_passed_points} "
            f"backtrack={self._nearest_backtrack_points} horizon={self._nearest_horizon_m:.2f}m"
        )

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        self._pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 20)
        hz = max(5.0, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / hz, self._on_tick)

        self._marker_frame = str(self.get_parameter("reference_path_marker_frame").value).strip() or "world"
        self._marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        do_markers = bool(self.get_parameter("publish_reference_path_marker").value)
        self._pub_collision_marker = None
        self._pub_dense_marker = None
        if do_markers:
            mtopic = str(self.get_parameter("reference_path_marker_topic").value)
            self._pub_collision_marker = self.create_publisher(Marker, mtopic, self._marker_qos)
        if bool(self.get_parameter("publish_dense_path_marker").value):
            dtopic = str(self.get_parameter("reference_path_dense_marker_topic").value)
            self._pub_dense_marker = self.create_publisher(Marker, dtopic, self._marker_qos)
        if self._pub_collision_marker is not None or self._pub_dense_marker is not None:
            self.create_timer(2.0, self._publish_path_markers)
            self._publish_path_markers()
        self._pub_done = None
        if bool(self.get_parameter("publish_completion_topic").value):
            done_topic = str(self.get_parameter("completion_topic").value).strip() or "/planned_path_done"
            done_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            )
            self._pub_done = self.create_publisher(Bool, done_topic, done_qos)

    def _publish_path_markers(self) -> None:
        """RViz: polilinia bez kolizji (A*) + opcjonalnie zagęszczona ścieżka śledzenia."""
        now = self.get_clock().now().to_msg()
        if self._pub_collision_marker is not None and self._collision_poly:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = self._marker_frame
            m.ns = "planned_path_collision_free"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.07
            m.color = ColorRGBA(r=0.1, g=0.85, b=0.2, a=0.95)
            for x, y in self._collision_poly:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.05
                m.points.append(p)
            self._pub_collision_marker.publish(m)
        if self._pub_dense_marker is not None and self._path:
            m2 = Marker()
            m2.header.stamp = now
            m2.header.frame_id = self._marker_frame
            m2.ns = "planned_path_dense"
            m2.id = 1
            m2.type = Marker.LINE_STRIP
            m2.action = Marker.ADD
            m2.scale.x = 0.035
            m2.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.75)
            for x, y in self._path:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.03
                m2.points.append(p)
            self._pub_dense_marker.publish(m2)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._px = float(msg.pose.position.x)
        self._py = float(msg.pose.position.y)
        q = msg.pose.orientation
        self._yaw = _yaw_from_quat(float(q.z), float(q.w))
        self._pose_ok = True

    def _on_tick(self) -> None:
        twist = Twist()
        if not self._pose_ok:
            self._pub.publish(twist)
            return

        gx, gy, nearest_i = _lookahead(
            self._path,
            self._px,
            self._py,
            self._path_idx,
            self._lookahead_m,
            self._nearest_backtrack_points,
            self._nearest_horizon_m,
            self._ignore_passed_points,
        )
        self._path_idx = max(self._path_idx, nearest_i) if self._ignore_passed_points else nearest_i

        dx, dy = gx - self._px, gy - self._py
        target_h = math.atan2(dy, dx)
        err = _wrap_pi(target_h - self._yaw)
        v_max_eff = self._v_max
        if self._excitation_enabled:
            # Triangle waveform in [0,1], then sinusoidal heading bias.
            now_s = float(self.get_clock().now().nanoseconds) * 1e-9
            phase = ((now_s - self._t0) / self._exc_period) % 1.0
            tri = 1.0 - abs(2.0 * phase - 1.0)
            v_scale_cmd = self._exc_v_min + (self._exc_v_max - self._exc_v_min) * tri
            v_max_eff = self._v_max * v_scale_cmd
            err = _wrap_pi(err + self._exc_heading_bias_rad * math.sin(2.0 * math.pi * phase))

        err_abs = abs(err)
        self._turn_in_place = _update_turn_in_place_mode(
            self._turn_in_place,
            err_abs,
            self._heading_stop_rad,
            self._heading_resume_rad,
        )
        if self._turn_in_place:
            # Na ostrym skręcie najpierw dokończ obrót (stabilny kierunek), dopiero potem jedź do przodu.
            self._turn_direction_sign = _resolve_turn_direction_sign(
                err,
                err_abs,
                self._turn_direction_sign,
                self._turn_direction_guard_rad,
                self._turn_direction_preference,
            )
            w = float(
                max(
                    -self._w_max,
                    min(self._w_max, self._kh * self._turn_direction_sign * err_abs),
                )
            )
            v = 0.0
        else:
            self._turn_direction_sign = 0.0
            w = float(max(-self._w_max, min(self._w_max, self._kh * err)))
            v_scale = max(0.0, math.cos(err))
            v = float(max(0.0, min(v_max_eff, v_max_eff * (v_scale ** self._alignment_cos_power))))

        lx, ly = self._path[-1]
        dist_end = math.hypot(lx - self._px, ly - self._py)
        if dist_end < self._goal_tol:
            if self._loop:
                self._path_idx = 0
            else:
                v = 0.0
                w = 0.0
                if not self._completion_sent:
                    self._completion_sent = True
                    elapsed = float(self.get_clock().now().nanoseconds) * 1e-9 - self._t0
                    self.get_logger().info(
                        f"[planned_path] trajectory completed in {elapsed:.2f}s sim-time"
                    )
                    if self._pub_done is not None:
                        msg = Bool()
                        msg.data = True
                        self._pub_done.publish(msg)

        twist.linear.x = v
        twist.angular.z = w
        self._pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannedPathDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Launch system może już wykonać shutdown() globalnie.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
