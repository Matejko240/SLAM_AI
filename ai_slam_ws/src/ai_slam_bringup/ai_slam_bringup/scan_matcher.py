import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from tf2_ros import TransformBroadcaster


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_from_yaw(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return (0.0, 0.0, qz, qw)


def scan_to_points(scan: LaserScan, max_use_range: float, max_points: int):
    """LaserScan -> Nx2 points in laser frame."""
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    rmin = float(scan.range_min) if scan.range_min > 0.0 else 0.08
    rmax = float(scan.range_max) if scan.range_max > 0.0 else max_use_range
    rmax = min(rmax, max_use_range)

    # sanitize
    ranges = np.where(np.isfinite(ranges), ranges, rmax)
    ranges = np.clip(ranges, rmin, rmax)

    n = ranges.size
    angles = float(scan.angle_min) + np.arange(n, dtype=np.float32) * float(scan.angle_increment)

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    pts = np.stack([xs, ys], axis=1)

    # opcjonalne ograniczenie liczby punktów dla szybkości
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int32)
        pts = pts[idx]

    return pts.astype(np.float32)


class ScanMatcher(Node):
    """
    Scan-to-scan motion estimation:
      - metoda "local": wielopoziomowe przeszukiwanie małego okna (szybkie)
      - metoda "bruteforce": pełne przeszukiwanie zakresów (wolniejsze, referencja)

    Publikuje:
      - pose_topic: PoseStamped w frame_id="odom" (dla eval_node)
      - twist_topic: TwistStamped (v, omega)
      - tf: opcjonalnie odom_like -> base_link_like
    """

    def __init__(self):
        super().__init__("scan_matcher")

        # --- parametry IO
        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("pose_topic", "/pose_scanmatch")
        self.declare_parameter("twist_topic", "/twist_scanmatch")
        self.declare_parameter("frame_id", "odom")              # frame dla PoseStamped
        self.declare_parameter("tf_parent", "odom_scanmatch")   # TF parent (dla RViz)
        self.declare_parameter("tf_child", "base_link_scanmatch")
        self.declare_parameter("publish_tf", True)

        # --- algorytm
        self.declare_parameter("method", "local")  # "local" | "bruteforce"
        self.declare_parameter("publish_every_n", 1)  # bruteforce często daj 5/10
        self.declare_parameter("grid_res", 0.05)       # rozdzielczość siatki [m]
        self.declare_parameter("grid_extent", 6.0)     # zakres siatki +/- [m]
        self.declare_parameter("max_use_range", 10.0)  # max dystans punktów [m]
        self.declare_parameter("max_points", 240)      # ogranicz liczbę punktów (0=bez limitu)

        # --- search ranges (dx, dy w [m], dtheta w [rad])
        # Local: okna i kroki (3 poziomy)
        self.declare_parameter("local_lvl1_win_xy", 0.10)
        self.declare_parameter("local_lvl1_win_th", 0.20)
        self.declare_parameter("local_lvl1_step_xy", 0.02)
        self.declare_parameter("local_lvl1_step_th", 0.04)

        self.declare_parameter("local_lvl2_win_xy", 0.05)
        self.declare_parameter("local_lvl2_win_th", 0.10)
        self.declare_parameter("local_lvl2_step_xy", 0.01)
        self.declare_parameter("local_lvl2_step_th", 0.02)

        self.declare_parameter("local_lvl3_win_xy", 0.02)
        self.declare_parameter("local_lvl3_win_th", 0.05)
        self.declare_parameter("local_lvl3_step_xy", 0.005)
        self.declare_parameter("local_lvl3_step_th", 0.01)

        # Bruteforce: pełny zakres
        self.declare_parameter("bf_range_xy", 0.15)
        self.declare_parameter("bf_range_th", 0.25)
        self.declare_parameter("bf_step_xy", 0.01)
        self.declare_parameter("bf_step_th", 0.01)

        # --- regularizacja (żeby nie wybierało „dziwnych” skoków)
        self.declare_parameter("reg_lambda", 0.5)      # kara za ruch
        self.declare_parameter("reg_theta_scale", 1.0) # waga theta w karze

        # --- filtracja delty (wygładzenie)
        self.declare_parameter("lpf_alpha", 0.0)  # 0=off, np 0.3 = delikatny filtr

        # --- init
        self.declare_parameter("init_x", 0.0)
        self.declare_parameter("init_y", 0.0)
        self.declare_parameter("init_theta", 0.0)

        # ---- read params
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.twist_topic = str(self.get_parameter("twist_topic").value)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.method = str(self.get_parameter("method").value).lower()
        self.publish_every_n = int(self.get_parameter("publish_every_n").value)

        self.grid_res = float(self.get_parameter("grid_res").value)
        self.grid_extent = float(self.get_parameter("grid_extent").value)
        self.max_use_range = float(self.get_parameter("max_use_range").value)
        self.max_points = int(self.get_parameter("max_points").value)

        self.reg_lambda = float(self.get_parameter("reg_lambda").value)
        self.reg_theta_scale = float(self.get_parameter("reg_theta_scale").value)

        self.lpf_alpha = float(self.get_parameter("lpf_alpha").value)

        self.x = float(self.get_parameter("init_x").value)
        self.y = float(self.get_parameter("init_y").value)
        self.th = float(self.get_parameter("init_theta").value)

        # internal
        self.prev_scan_pts = None
        self.prev_stamp = None
        self.step_idx = 0

        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.prev_dth = 0.0

        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_twist = self.create_publisher(TwistStamped, self.twist_topic, 10)

        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        self.get_logger().info(
            f"ScanMatcher started: method={self.method}, scan={self.scan_topic} -> pose={self.pose_topic}"
        )

    def _make_grid(self, pts_prev: np.ndarray):
        """Buduje siatkę bool dla poprzedniego skanu."""
        res = self.grid_res
        ext = self.grid_extent
        size = int((2.0 * ext) / res) + 1
        grid = np.zeros((size, size), dtype=np.bool_)
        # map coords -> indices
        ix = ((pts_prev[:, 0] + ext) / res).astype(np.int32)
        iy = ((pts_prev[:, 1] + ext) / res).astype(np.int32)
        ok = (ix >= 0) & (iy >= 0) & (ix < size) & (iy < size)
        grid[iy[ok], ix[ok]] = True
        return grid, size

    def _score(self, grid, size, pts_curr, dx, dy, dth):
        """
        Score = hits - lambda * (dx^2 + dy^2 + (scale*dth)^2)
        Ruch (dx,dy,dth) jest rozumiany jako delta w układzie poprzedniej klatki.
        Transformujemy punkty z CURRENT -> PREV:
            p_prev_pred = [dx,dy] + Rot(dth) * p_curr
        """
        c = math.cos(dth)
        s = math.sin(dth)

        x = pts_curr[:, 0]
        y = pts_curr[:, 1]

        xp = dx + c * x - s * y
        yp = dy + s * x + c * y

        res = self.grid_res
        ext = self.grid_extent

        ix = ((xp + ext) / res).astype(np.int32)
        iy = ((yp + ext) / res).astype(np.int32)
        ok = (ix >= 0) & (iy >= 0) & (ix < size) & (iy < size)

        hits = int(np.count_nonzero(grid[iy[ok], ix[ok]]))

        reg = self.reg_lambda * (dx * dx + dy * dy + (self.reg_theta_scale * dth) ** 2)
        return float(hits) - float(reg)

    def _grid_search(self, grid, size, pts_curr, center, win_xy, win_th, step_xy, step_th):
        cx, cy, cth = center

        xs = np.arange(cx - win_xy, cx + win_xy + 1e-9, step_xy, dtype=np.float32)
        ys = np.arange(cy - win_xy, cy + win_xy + 1e-9, step_xy, dtype=np.float32)
        ths = np.arange(cth - win_th, cth + win_th + 1e-9, step_th, dtype=np.float32)

        best = (cx, cy, cth)
        best_score = -1e18

        # theta outer -> reuse cos/sin inside _score anyway, ale w tej wersji prosto
        for dth in ths:
            for dx in xs:
                for dy in ys:
                    sc = self._score(grid, size, pts_curr, float(dx), float(dy), float(dth))
                    if sc > best_score:
                        best_score = sc
                        best = (float(dx), float(dy), float(dth))

        return best, best_score

    def _estimate_delta_local(self, grid, size, pts_curr):
        """3-poziomowe przeszukiwanie wokół poprzedniej delty (albo 0)."""
        center = (self.prev_dx, self.prev_dy, self.prev_dth)

        lvl1 = (
            float(self.get_parameter("local_lvl1_win_xy").value),
            float(self.get_parameter("local_lvl1_win_th").value),
            float(self.get_parameter("local_lvl1_step_xy").value),
            float(self.get_parameter("local_lvl1_step_th").value),
        )
        lvl2 = (
            float(self.get_parameter("local_lvl2_win_xy").value),
            float(self.get_parameter("local_lvl2_win_th").value),
            float(self.get_parameter("local_lvl2_step_xy").value),
            float(self.get_parameter("local_lvl2_step_th").value),
        )
        lvl3 = (
            float(self.get_parameter("local_lvl3_win_xy").value),
            float(self.get_parameter("local_lvl3_win_th").value),
            float(self.get_parameter("local_lvl3_step_xy").value),
            float(self.get_parameter("local_lvl3_step_th").value),
        )

        best, _ = self._grid_search(grid, size, pts_curr, center, lvl1[0], lvl1[1], lvl1[2], lvl1[3])
        best, _ = self._grid_search(grid, size, pts_curr, best,   lvl2[0], lvl2[1], lvl2[2], lvl2[3])
        best, _ = self._grid_search(grid, size, pts_curr, best,   lvl3[0], lvl3[1], lvl3[2], lvl3[3])

        return best

    def _estimate_delta_bruteforce(self, grid, size, pts_curr):
        """Jedno duże przeszukanie (wolne)."""
        rxy = float(self.get_parameter("bf_range_xy").value)
        rth = float(self.get_parameter("bf_range_th").value)
        sxy = float(self.get_parameter("bf_step_xy").value)
        sth = float(self.get_parameter("bf_step_th").value)

        center = (0.0, 0.0, 0.0)  # brute-force jako „od zera”
        best, _ = self._grid_search(grid, size, pts_curr, center, rxy, rth, sxy, sth)
        return best

    def _integrate_pose(self, dx, dy, dth):
        """SE2 compose: world_pose = world_pose ⊕ delta(prev_frame)."""
        c = math.cos(self.th)
        s = math.sin(self.th)
        self.x += c * dx - s * dy
        self.y += s * dx + c * dy
        self.th = wrap(self.th + dth)

    def on_scan(self, msg: LaserScan):
        t_start = time.perf_counter()

        pts = scan_to_points(msg, self.max_use_range, self.max_points)

        if self.prev_scan_pts is None:
            self.prev_scan_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # publish throttling (np. bruteforce co 5 skanów)
        self.step_idx += 1
        if self.publish_every_n > 1 and (self.step_idx % self.publish_every_n) != 0:
            self.prev_scan_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # dt
        if self.prev_stamp is not None:
            t_prev = float(self.prev_stamp.sec) + 1e-9 * float(self.prev_stamp.nanosec)
            t_cur = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
            dt = max(1e-3, t_cur - t_prev)
        else:
            dt = 0.1

        # grid from prev
        grid, size = self._make_grid(self.prev_scan_pts)

        # estimate delta
        if self.method == "bruteforce":
            dx, dy, dth = self._estimate_delta_bruteforce(grid, size, pts)
        else:
            dx, dy, dth = self._estimate_delta_local(grid, size, pts)

        # optional low-pass filter on delta
        a = float(self.lpf_alpha)
        if a > 0.0:
            dx = a * dx + (1.0 - a) * self.prev_dx
            dy = a * dy + (1.0 - a) * self.prev_dy
            dth = a * dth + (1.0 - a) * self.prev_dth

        # store last delta as next center
        self.prev_dx, self.prev_dy, self.prev_dth = float(dx), float(dy), float(dth)

        # integrate
        self._integrate_pose(float(dx), float(dy), float(dth))

        # publish PoseStamped (dla eval_node ważne: frame_id="odom")
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.frame_id
        qx, qy, qz, qw = quat_from_yaw(self.th)
        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pub_pose.publish(ps)

        # publish twist (v, omega)
        v = float(math.hypot(dx, dy) / dt)
        w = float(dth / dt)
        tw = TwistStamped()
        tw.header.stamp = msg.header.stamp
        tw.header.frame_id = self.frame_id
        tw.twist.linear.x = v
        tw.twist.angular.z = w
        self.pub_twist.publish(tw)

        # publish TF for RViz (opcjonalnie)
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

        # update prev
        self.prev_scan_pts = pts
        self.prev_stamp = msg.header.stamp

        t_end = time.perf_counter()
        ms = (t_end - t_start) * 1000.0
        if ms > 60.0 and (self.step_idx % 10) == 0:
            self.get_logger().warn(f"scan_matcher step took {ms:.1f} ms (method={self.method})")


def main():
    rclpy.init()
    node = ScanMatcher()
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
