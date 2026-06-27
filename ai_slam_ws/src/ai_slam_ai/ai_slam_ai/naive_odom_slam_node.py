from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


# ── small math helpers ────────────────────────────────────────────────────────

def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _xytheta_from_odom(msg: Odometry) -> tuple[float, float, float]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    return float(p.x), float(p.y), float(yaw)


# ── Bresenham line rasteriser (no external library) ───────────────────────────

def _bresenham(x0: int, y0: int, x1: int, y1: int):
    """Yield integer (x, y) cells on the segment from (x0,y0) to (x1,y1).

    Standard integer Bresenham algorithm. Yields inclusive endpoints.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ── Main node ─────────────────────────────────────────────────────────────────

class NaiveOdomSlamNode(Node):
    """Scan-to-map SLAM node backed by odometry dead-reckoning + exhaustive candidate search."""

    def __init__(self) -> None:
        super().__init__("naive_odom_slam_node")

        # topic parameters
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom_raw")
        self.declare_parameter("pose_topic", "/pose_naive_odom_slam")
        self.declare_parameter("odom_out_topic", "/odom_naive_slam")
        self.declare_parameter("map_topic", "/map_naive_odom_slam")
        self.declare_parameter("tf_parent", "map_naive_odom_slam")
        self.declare_parameter("tf_child", "base_link_naive_odom_slam")
        self.declare_parameter("publish_tf", True)

        # map parameters
        self.declare_parameter("map_resolution", 0.05)
        self.declare_parameter("map_width_m", 60.0)
        self.declare_parameter("map_height_m", 60.0)
        # Map origin bottom-left corner in SLAM frame. The node will shift
        # these at startup so the robot begins at the map centre.
        self.declare_parameter("map_origin_x", -30.0)
        self.declare_parameter("map_origin_y", -30.0)

        # scan-matching parameters — thesis-readable names (preferred).
        # Search window is centred on the odometry-predicted pose:
        #   x_candidate = x_odom + dx,  dx ∈ [-range_m, +range_m]
        #   theta_candidate = theta_odom + dtheta
        # -1.0 sentinel means "fall back to legacy radian/metre param".
        self.declare_parameter("search_xy_range_m", -1.0)
        self.declare_parameter("search_xy_step_m", -1.0)
        self.declare_parameter("search_theta_range_deg", -1.0)
        self.declare_parameter("search_theta_step_deg", -1.0)
        self.declare_parameter("motion_prior", "odometry_centered")
        self.declare_parameter("use_exhaustive_search", True)
        # Legacy params (kept for backward compat; overridden by _m/_deg above).
        self.declare_parameter("search_xy_range", 0.10)
        self.declare_parameter("search_xy_step", 0.02)
        self.declare_parameter("search_theta_range", 0.15)
        self.declare_parameter("search_theta_step", 0.03)
        self.declare_parameter("max_scan_range", 8.0)
        self.declare_parameter("min_scan_range", 0.15)
        self.declare_parameter("update_every_n_scans", 1)
        # Use every Nth beam for map update (Bresenham loop — Python speed).
        self.declare_parameter("map_update_beam_subsample", 3)
        # Use every Nth beam for scoring (NumPy — fast).
        self.declare_parameter("score_beam_subsample", 1)

        # scoring parameters (penalty weights for candidates far from odom)
        self.declare_parameter("score_occupied_reward", 1.0)
        self.declare_parameter("score_free_penalty", -0.1)
        self.declare_parameter("score_unknown_weight", 0.0)
        self.declare_parameter("odom_prior_xy_weight", 0.0)
        self.declare_parameter("odom_prior_theta_weight", 0.0)

        # occupancy log-odds parameters
        self.declare_parameter("logodds_occ", 0.85)
        self.declare_parameter("logodds_free", -0.40)
        self.declare_parameter("logodds_min", -3.0)
        self.declare_parameter("logodds_max", 3.5)

        # misc
        self.declare_parameter("init_from_odom", True)
        # Publish the OccupancyGrid every N map updates to reduce CPU/bandwidth.
        self.declare_parameter("publish_map_every_n", 5)
        # External motion-prior pose topic (PoseStamped). When non-empty, the
        # node subscribes to this topic instead of odom_topic to drive the
        # candidate search window (tor11/tor12: Robak/Rywak prior instead of /odom_raw).
        self.declare_parameter("motion_prior_pose_topic", "")
        # Hybrid blending weight: 1.0 = pure AI prior (original tor11/tor12 behaviour),
        # 0.0 = pure odometry, 0 < alpha < 1 = weighted blend of AI and odom deltas.
        # Only effective when motion_prior_pose_topic is also set.
        self.declare_parameter("motion_prior_alpha", 1.0)

        # read back
        self.scan_topic      = str(self.get_parameter("scan_topic").value)
        self.odom_topic      = str(self.get_parameter("odom_topic").value)
        self.pose_topic      = str(self.get_parameter("pose_topic").value)
        self.odom_out_topic  = str(self.get_parameter("odom_out_topic").value)
        self.map_topic       = str(self.get_parameter("map_topic").value)
        self.tf_parent       = str(self.get_parameter("tf_parent").value)
        self.tf_child        = str(self.get_parameter("tf_child").value)
        self.publish_tf      = bool(self.get_parameter("publish_tf").value)

        self.resolution   = float(self.get_parameter("map_resolution").value)
        map_width_m       = float(self.get_parameter("map_width_m").value)
        map_height_m      = float(self.get_parameter("map_height_m").value)
        self.map_origin_x = float(self.get_parameter("map_origin_x").value)
        self.map_origin_y = float(self.get_parameter("map_origin_y").value)

        _xy_range_m   = float(self.get_parameter("search_xy_range_m").value)
        _xy_step_m    = float(self.get_parameter("search_xy_step_m").value)
        _th_range_deg = float(self.get_parameter("search_theta_range_deg").value)
        _th_step_deg  = float(self.get_parameter("search_theta_step_deg").value)
        self.search_xy_range    = _xy_range_m  if _xy_range_m  > 0 else float(self.get_parameter("search_xy_range").value)
        self.search_xy_step     = _xy_step_m   if _xy_step_m   > 0 else float(self.get_parameter("search_xy_step").value)
        self.search_theta_range = math.radians(_th_range_deg) if _th_range_deg > 0 else float(self.get_parameter("search_theta_range").value)
        self.search_theta_step  = math.radians(_th_step_deg)  if _th_step_deg  > 0 else float(self.get_parameter("search_theta_step").value)
        self.max_scan_range       = float(self.get_parameter("max_scan_range").value)
        self.min_scan_range       = float(self.get_parameter("min_scan_range").value)
        self.update_every_n_scans = max(1, int(self.get_parameter("update_every_n_scans").value))
        self.map_beam_sub  = max(1, int(self.get_parameter("map_update_beam_subsample").value))
        self.score_beam_sub = max(1, int(self.get_parameter("score_beam_subsample").value))

        self.odom_prior_xy_w  = float(self.get_parameter("odom_prior_xy_weight").value)
        self.odom_prior_th_w  = float(self.get_parameter("odom_prior_theta_weight").value)

        self.lo_occ  = float(self.get_parameter("logodds_occ").value)
        self.lo_free = float(self.get_parameter("logodds_free").value)
        self.lo_min  = float(self.get_parameter("logodds_min").value)
        self.lo_max  = float(self.get_parameter("logodds_max").value)

        self.init_from_odom     = bool(self.get_parameter("init_from_odom").value)
        self.publish_map_every_n = max(1, int(self.get_parameter("publish_map_every_n").value))
        self.motion_prior_pose_topic = str(self.get_parameter("motion_prior_pose_topic").value).strip()
        self.motion_prior_alpha = float(self.get_parameter("motion_prior_alpha").value)

        # occupancy grid (log-odds, float32)
        self.W = max(1, int(round(map_width_m / self.resolution)))
        self.H = max(1, int(round(map_height_m / self.resolution)))
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)

        # pre-compute search candidates: (K, 3) [dx, dy, dth]
        xy_vals  = np.arange(
            -self.search_xy_range,
             self.search_xy_range + 1e-9,
             self.search_xy_step,
        )
        th_vals  = np.arange(
            -self.search_theta_range,
             self.search_theta_range + 1e-9,
             self.search_theta_step,
        )
        dx_g, dy_g, dth_g = np.meshgrid(xy_vals, xy_vals, th_vals, indexing="ij")
        self.candidates = np.stack(
            [dx_g.ravel(), dy_g.ravel(), dth_g.ravel()], axis=-1
        ).astype(np.float64)

        # internal state
        self.pose_inited = False
        self.x   = 0.0
        self.y   = 0.0
        self.th  = 0.0
        self._prev_odom: tuple[float, float, float] | None = None
        self._scan_count   = 0
        self._update_count = 0
        self._last_pub_stamp_ns: int | None = None
        # Hybrid mode state: latest/previous raw odom position (used only when
        # 0 < motion_prior_alpha < 1 to blend odom delta and AI delta).
        self._hyb_odom_cur: tuple[float, float, float] | None = None
        self._hyb_odom_prev: tuple[float, float, float] | None = None

        # publishers
        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_odom = self.create_publisher(Odometry, self.odom_out_topic, 10)
        # Use TRANSIENT_LOCAL (latched) so eval_node's map_qos subscription is
        # compatible and the latest map is always available to late-joining subscribers.
        _map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_map  = self.create_publisher(OccupancyGridMsg, self.map_topic, _map_qos)
        self.tf_br    = TransformBroadcaster(self) if self.publish_tf else None

        # subscriptions
        if self.motion_prior_pose_topic:
            alpha = self.motion_prior_alpha
            if 0.0 < alpha < 1.0:
                # Hybrid mode: also subscribe to raw odometry (tracking only —
                # does NOT update SLAM estimate; only records _hyb_odom_cur).
                self._sub_odom_hyb = self.create_subscription(
                    Odometry, self.odom_topic, self._on_odom_hybrid, 50
                )
                self.get_logger().info(
                    f"[NaiveSlam] motion prior: hybrid alpha={alpha:.2f} "
                    f"(PoseStamped on {self.motion_prior_pose_topic} + odom on {self.odom_topic})"
                )
            else:
                self.get_logger().info(
                    f"[NaiveSlam] motion prior: PoseStamped on {self.motion_prior_pose_topic} (alpha=1.0)"
                )
            self.sub_prior = self.create_subscription(
                PoseStamped, self.motion_prior_pose_topic, self._on_prior_pose, 50
            )
        else:
            self.sub_prior = self.create_subscription(
                Odometry, self.odom_topic, self._on_odom, 50
            )
        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"[NaiveSlam] map={self.W}×{self.H} @ {self.resolution}m/cell  "
            f"candidates={len(self.candidates)}  "
            f"score_sub={self.score_beam_sub}  map_sub={self.map_beam_sub}"
        )

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _apply_prior_delta(self, ox: float, oy: float, oth: float) -> None:
        """Apply an absolute prior pose as a motion delta to the current SLAM estimate.

        Works identically whether the prior comes from raw odometry or an external
        model (Robak/Rywak). Tracks the previous prior pose and applies the SE(2)
        increment to self.x/y/th so that _on_scan searches around the predicted pose.
        """
        if not self.pose_inited:
            self.x, self.y, self.th = ox, oy, oth
            half_w = self.W * self.resolution * 0.5
            half_h = self.H * self.resolution * 0.5
            self.map_origin_x = ox - half_w
            self.map_origin_y = oy - half_h
            self._prev_odom   = (ox, oy, oth)
            self.pose_inited  = True
            return

        if self._prev_odom is None:
            self._prev_odom = (ox, oy, oth)
            return

        px, py, pth = self._prev_odom
        ddx_w = ox - px
        ddy_w = oy - py
        c = math.cos(pth)
        s = math.sin(pth)
        dx_r =  c * ddx_w + s * ddy_w
        dy_r = -s * ddx_w + c * ddy_w
        dth  = _wrap(oth - pth)

        c2 = math.cos(self.th)
        s2 = math.sin(self.th)
        self.x  += c2 * dx_r - s2 * dy_r
        self.y  += s2 * dx_r + c2 * dy_r
        self.th  = _wrap(self.th + dth)
        self._prev_odom = (ox, oy, oth)

    def _on_odom(self, msg: Odometry) -> None:
        ox, oy, oth = _xytheta_from_odom(msg)
        self._apply_prior_delta(ox, oy, oth)

    def _on_odom_hybrid(self, msg: Odometry) -> None:
        """Track raw odometry for hybrid blending; does NOT update SLAM estimate."""
        ox, oy, oth = _xytheta_from_odom(msg)
        self._hyb_odom_cur = (ox, oy, oth)
        # If SLAM not yet inited and AI prior hasn't arrived, bootstrap from odom.
        if not self.pose_inited:
            self.x, self.y, self.th = ox, oy, oth
            half_w = self.W * self.resolution * 0.5
            half_h = self.H * self.resolution * 0.5
            self.map_origin_x = ox - half_w
            self.map_origin_y = oy - half_h
            self._prev_odom     = (ox, oy, oth)
            self._hyb_odom_prev = (ox, oy, oth)
            self.pose_inited = True

    def _on_prior_pose(self, msg: PoseStamped) -> None:
        """Motion prior from an external PoseStamped topic (Robak or Rywak).

        alpha=1.0 (default): pure AI prior — identical to previous tor11/tor12 behaviour.
        0 < alpha < 1:       hybrid blend: (1-alpha)*odom_delta + alpha*ai_delta.
        """
        p = msg.pose.position
        q = msg.pose.orientation
        oth = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        ax, ay, ath = float(p.x), float(p.y), float(oth)

        alpha = self.motion_prior_alpha

        if alpha >= 1.0 or self._hyb_odom_cur is None:
            # Pure AI mode (original behaviour) or hybrid odom not yet available.
            self._apply_prior_delta(ax, ay, ath)
            return

        # ── Hybrid blend ─────────────────────────────────────────────────────
        # Initialize on first call.
        if not self.pose_inited or self._prev_odom is None:
            self.x, self.y, self.th = ax, ay, ath
            half_w = self.W * self.resolution * 0.5
            half_h = self.H * self.resolution * 0.5
            self.map_origin_x = ax - half_w
            self.map_origin_y = ay - half_h
            self._prev_odom     = (ax, ay, ath)   # AI anchor
            self._hyb_odom_prev = self._hyb_odom_cur
            self.pose_inited = True
            return

        # AI delta in world frame → local (previous AI heading) frame.
        pax, pay, path = self._prev_odom
        ai_ddx = ax - pax
        ai_ddy = ay - pay
        ai_dth = _wrap(ath - path)
        c_a =  math.cos(path)
        s_a =  math.sin(path)
        ai_dx_r =  c_a * ai_ddx + s_a * ai_ddy
        ai_dy_r = -s_a * ai_ddx + c_a * ai_ddy

        # Odom delta in world frame → local (previous odom heading) frame.
        ox, oy, oth_o = self._hyb_odom_cur
        if self._hyb_odom_prev is None:
            self._hyb_odom_prev = (ox, oy, oth_o)
        pox, poy, poth = self._hyb_odom_prev
        od_ddx = ox - pox
        od_ddy = oy - poy
        od_dth = _wrap(oth_o - poth)
        c_o =  math.cos(poth)
        s_o =  math.sin(poth)
        od_dx_r =  c_o * od_ddx + s_o * od_ddy
        od_dy_r = -s_o * od_ddx + c_o * od_ddy

        # Weighted blend: (1-alpha)*odom + alpha*ai.
        dx_r = (1.0 - alpha) * od_dx_r + alpha * ai_dx_r
        dy_r = (1.0 - alpha) * od_dy_r + alpha * ai_dy_r
        dth  = (1.0 - alpha) * od_dth  + alpha * ai_dth

        # Apply blended delta to SLAM estimate.
        c2 = math.cos(self.th)
        s2 = math.sin(self.th)
        self.x  += c2 * dx_r - s2 * dy_r
        self.y  += s2 * dx_r + c2 * dy_r
        self.th  = _wrap(self.th + dth)

        # Advance anchors for next step.
        self._prev_odom     = (ax, ay, ath)
        self._hyb_odom_prev = (ox, oy, oth_o)

    def _on_scan(self, msg: LaserScan) -> None:
        if not self.pose_inited:
            return
        self._scan_count += 1
        if (self._scan_count % self.update_every_n_scans) != 0:
            return

        stamp = msg.header.stamp

        # Parse valid scan endpoints into robot-frame (x, y) positions.
        n_rays    = len(msg.ranges)
        angles    = msg.angle_min + np.arange(n_rays, dtype=np.float32) * msg.angle_increment
        ranges    = np.asarray(msg.ranges, dtype=np.float32)
        valid     = (
            np.isfinite(ranges)
            & (ranges >= self.min_scan_range)
            & (ranges <= self.max_scan_range)
        )
        va = angles[valid]
        vr = ranges[valid]
        scan_pts = np.stack([vr * np.cos(va), vr * np.sin(va)], axis=-1)  # (N, 2)

        if len(scan_pts) < 10:
            self._publish_pose(stamp)
            return

        # Scan matching: score all candidates, pick the best.
        score_pts = scan_pts[:: self.score_beam_sub]
        scores    = self._score_candidates(score_pts)
        best      = self.candidates[int(np.argmax(scores))]
        self.x   += best[0]
        self.y   += best[1]
        self.th   = _wrap(self.th + best[2])

        # Map update with Bresenham traces (sub-sampled beams).
        self._update_map(scan_pts[:: self.map_beam_sub])

        self._update_count += 1
        self._publish_pose(stamp)
        if (self._update_count % self.publish_map_every_n) == 0:
            self._publish_map(stamp)

    # ── SLAM core ─────────────────────────────────────────────────────────────

    def _score_candidates(self, scan_pts: np.ndarray) -> np.ndarray:
        """Score all search candidates simultaneously.

        Vectorised NumPy — O(K × N) with no Python loop over candidates.

        Scoring: sum of log-odds values at the projected scan endpoint cells.
          - Occupied cells (log-odds > 0) reward the candidate.
          - Free cells (log-odds < 0) penalise it.
          - Unknown cells (log-odds ≈ 0) are neutral.

        Args:
            scan_pts: (N, 2) scan endpoint positions in robot frame.
        Returns:
            (K,) float score per candidate.
        """
        cx  = self.x  + self.candidates[:, 0]   # (K,)
        cy  = self.y  + self.candidates[:, 1]   # (K,)
        cth = self.th + self.candidates[:, 2]   # (K,)

        cos_c = np.cos(cth)   # (K,)
        sin_c = np.sin(cth)   # (K,)
        sx    = scan_pts[:, 0]  # (N,)
        sy    = scan_pts[:, 1]  # (N,)

        # World endpoint for every (candidate, beam) pair — shape (K, N).
        wx = cx[:, None] + cos_c[:, None] * sx[None, :] - sin_c[:, None] * sy[None, :]
        wy = cy[:, None] + sin_c[:, None] * sx[None, :] + cos_c[:, None] * sy[None, :]

        # Grid column (x-axis) and row (y-axis) indices.
        gi = np.floor((wx - self.map_origin_x) / self.resolution).astype(np.int32)  # (K, N)
        gj = np.floor((wy - self.map_origin_y) / self.resolution).astype(np.int32)  # (K, N)

        valid  = (gi >= 0) & (gi < self.W) & (gj >= 0) & (gj < self.H)
        gi_s   = np.clip(gi, 0, self.W - 1)
        gj_s   = np.clip(gj, 0, self.H - 1)

        # Look up log-odds and zero out out-of-bounds cells.
        lo_vals = self.log_odds[gj_s, gi_s]        # (K, N) — fancy indexing
        scores  = np.sum(lo_vals * valid, axis=1)   # (K,)

        if self.odom_prior_xy_w > 0.0:
            scores -= self.odom_prior_xy_w * (
                self.candidates[:, 0] ** 2 + self.candidates[:, 1] ** 2
            )
        if self.odom_prior_th_w > 0.0:
            scores -= self.odom_prior_th_w * (self.candidates[:, 2] ** 2)

        return scores

    def _update_map(self, scan_pts: np.ndarray) -> None:
        """Update the occupancy grid using Bresenham ray tracing.

        For every beam:
          - Cells along the beam: log-odds += logodds_free  (free space).
          - Endpoint cell:        log-odds += logodds_occ   (occupied surface).
        Log-odds are clamped to [lo_min, lo_max].

        Args:
            scan_pts: (N, 2) endpoint positions in robot frame (sub-sampled).
        """
        cos_th = math.cos(self.th)
        sin_th = math.sin(self.th)
        rx     = int((self.x - self.map_origin_x) / self.resolution)
        ry     = int((self.y - self.map_origin_y) / self.resolution)

        H      = self.H
        W      = self.W
        grid   = self.log_odds
        lo_occ = self.lo_occ
        lo_free = self.lo_free
        lo_min = self.lo_min
        lo_max = self.lo_max

        for sp in scan_pts:
            sx = float(sp[0])
            sy = float(sp[1])
            wx = self.x + cos_th * sx - sin_th * sy
            wy = self.y + sin_th * sx + cos_th * sy
            ex = int((wx - self.map_origin_x) / self.resolution)
            ey = int((wy - self.map_origin_y) / self.resolution)

            for bx, by in _bresenham(rx, ry, ex, ey):
                if bx < 0 or bx >= W or by < 0 or by >= H:
                    continue
                if bx == ex and by == ey:
                    v = grid[by, bx] + lo_occ
                else:
                    v = grid[by, bx] + lo_free
                if v < lo_min:
                    v = lo_min
                elif v > lo_max:
                    v = lo_max
                grid[by, bx] = v

    # ── publishers ────────────────────────────────────────────────────────────

    def _monotonic_stamp(self, stamp) -> tuple[int, int]:
        ns = (
            int(getattr(stamp, "sec", 0)) * 1_000_000_000
            + int(getattr(stamp, "nanosec", 0))
        )
        if self._last_pub_stamp_ns is not None and ns <= self._last_pub_stamp_ns:
            ns = self._last_pub_stamp_ns + 1
        self._last_pub_stamp_ns = ns
        return int(ns // 1_000_000_000), int(ns % 1_000_000_000)

    def _publish_pose(self, stamp) -> None:
        sec, nanosec = self._monotonic_stamp(stamp)
        qx, qy, qz, qw = _quat_from_yaw(self.th)

        ps = PoseStamped()
        ps.header.stamp.sec      = sec
        ps.header.stamp.nanosec  = nanosec
        ps.header.frame_id       = self.tf_parent
        ps.pose.position.x       = float(self.x)
        ps.pose.position.y       = float(self.y)
        ps.pose.position.z       = 0.0
        ps.pose.orientation.x    = qx
        ps.pose.orientation.y    = qy
        ps.pose.orientation.z    = qz
        ps.pose.orientation.w    = qw
        self.pub_pose.publish(ps)

        od = Odometry()
        od.header.stamp.sec     = sec
        od.header.stamp.nanosec = nanosec
        od.header.frame_id      = self.tf_parent
        od.child_frame_id       = self.tf_child
        od.pose.pose            = ps.pose
        self.pub_odom.publish(od)

        if self.tf_br is not None:
            tfm = TransformStamped()
            tfm.header.stamp.sec     = sec
            tfm.header.stamp.nanosec = nanosec
            tfm.header.frame_id      = self.tf_parent
            tfm.child_frame_id       = self.tf_child
            tfm.transform.translation.x = float(self.x)
            tfm.transform.translation.y = float(self.y)
            tfm.transform.translation.z = 0.0
            tfm.transform.rotation.x    = qx
            tfm.transform.rotation.y    = qy
            tfm.transform.rotation.z    = qz
            tfm.transform.rotation.w    = qw
            self.tf_br.sendTransform(tfm)

    def _publish_map(self, stamp) -> None:
        sec, nanosec = self._monotonic_stamp(stamp)

        # Convert log-odds to standard ROS occupancy values:
        #   -1  = unknown, 0 = free, 100 = occupied.
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        occ  = np.full((self.H, self.W), -1, dtype=np.int8)
        known = np.abs(self.log_odds) > 0.05   # cells that have been updated at least once
        occ[known & (prob > 0.5)]  = 100
        occ[known & (prob <= 0.5)] = 0

        grid_msg = OccupancyGridMsg()
        grid_msg.header.stamp.sec     = sec
        grid_msg.header.stamp.nanosec = nanosec
        grid_msg.header.frame_id      = self.tf_parent
        grid_msg.info.resolution      = float(self.resolution)
        grid_msg.info.width           = self.W
        grid_msg.info.height          = self.H
        grid_msg.info.origin.position.x  = float(self.map_origin_x)
        grid_msg.info.origin.position.y  = float(self.map_origin_y)
        grid_msg.info.origin.position.z  = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = occ.ravel().tolist()
        self.pub_map.publish(grid_msg)


def main() -> None:
    rclpy.init()
    node = NaiveOdomSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
