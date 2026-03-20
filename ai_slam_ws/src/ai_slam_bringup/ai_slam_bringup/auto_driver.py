import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def parse_bool(value, default=False):
    """Konwertuje parametr ROS (bool/str/int) do bool w przewidywalny sposób."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v in ("0", "false", "f", "no", "n", "off", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class AutoDriver(Node):
    def __init__(self):
        super().__init__("auto_driver")
        self.declare_parameter("debug", True)
        self.declare_parameter("debug_every_n", 10)  # log co N ticków timera (np. 10 => ~1/s przy 10Hz)
        self.declare_parameter("seed", 123)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("v_forward", 0.25)
        self.declare_parameter("w_turn", 1.0)
        self.declare_parameter("front_threshold", 0.45)   # react when < 45cm
        self.declare_parameter("side_threshold", 0.35)    # react when < 35cm
        self.declare_parameter("emergency_threshold", 0.25)  # very close = emergency < 25cm
        self.declare_parameter("linear_velocity", 0.25)
        self.declare_parameter("angular_velocity", 1.0)
        self.declare_parameter("obstacle_threshold", 0.45)
        self.declare_parameter("turn_probability", 0.02)
        # --- Exploration tuning (z demo.launch.py) ---
        self.declare_parameter("explore_interval_ticks", 30)         # co ile ticków rozważyć skręt eksploracyjny
        self.declare_parameter("explore_turn_probability", -1.0)     # <0 => użyj turn_probability
        self.declare_parameter("odom_topic", "/odom_raw")  # DO STUCK DETECTION 
        # --- Doorway / room-entering heuristic ---
        self.declare_parameter("doorway_turn_probability", 0.0)      # 0 => wyłączone
        self.declare_parameter("doorway_opening_threshold", 1.8)     # średnia odległość po stronie "otwartej"
        self.declare_parameter("doorway_wall_threshold", 0.9)        # średnia odległość po stronie "ściany"
        self.declare_parameter("doorway_turn_min_sec", 0.7)
        self.declare_parameter("doorway_turn_max_sec", 1.4)
        self.declare_parameter("motion_profile_enabled", False)
        self.declare_parameter("linear_velocity_min", 0.0)
        self.declare_parameter("linear_velocity_max", 0.0)
        self.declare_parameter("angular_velocity_min", 0.0)
        self.declare_parameter("angular_velocity_max", 0.0)
        self.declare_parameter("profile_change_interval_sec", 2.5)
        self.declare_parameter("profile_arc_probability", 0.35)
        self.declare_parameter("profile_arc_fraction_min", 0.12)
        self.declare_parameter("profile_arc_fraction_max", 0.45)
        self.declare_parameter("explore_spin_probability", 0.18)
        self.declare_parameter("explore_spin_min_sec", 1.0)
        self.declare_parameter("explore_spin_max_sec", 2.4)
        self.declare_parameter("forward_slowdown_min_factor", 0.45)
        self.declare_parameter("nav_sector_deg", 110.0)
        self.declare_parameter("nav_gap_half_window_deg", 16.0)
        self.declare_parameter("nav_safe_clearance", 0.52)
        self.declare_parameter("nav_lookahead_cap", 4.0)
        self.declare_parameter("nav_heading_gain", 1.8)
        self.declare_parameter("nav_avoid_gain", 0.7)
        self.declare_parameter("nav_min_linear_speed", 0.06)
        self.declare_parameter("nav_heading_bias_max_deg", 70.0)
        self.declare_parameter("nav_heading_bias_hold_sec", 5.0)
        self.declare_parameter("nav_heading_smooth_alpha", 0.55)
        self.declare_parameter("nav_novelty_lookahead_m", 1.4)
        self.declare_parameter("nav_novelty_bonus", 0.85)
        self.declare_parameter("nav_recent_cell_penalty", 1.15)
        self.declare_parameter("robot_front_extent", 0.15)
        self.declare_parameter("robot_rear_extent", 0.15)
        self.declare_parameter("robot_half_width", 0.10)
        self.declare_parameter("robot_safety_margin", 0.06)
        self.declare_parameter("repeat_cell_size_m", 0.9)
        self.declare_parameter("repeat_window_size", 40)
        self.declare_parameter("repeat_unique_ratio_threshold", 0.55)
        self.declare_parameter("repeat_escape_trigger", 6)
        self.declare_parameter("repeat_escape_turn_sec", 2.8)
        self.declare_parameter("repeat_escape_heading_deg", 85.0)
        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.v_forward = float(self.get_parameter("linear_velocity").value)
        self.w_turn = float(self.get_parameter("angular_velocity").value)
        self.turn_probability = float(self.get_parameter("turn_probability").value)
        self.front_threshold = float(self.get_parameter("obstacle_threshold").value)
        self.side_threshold = float(self.get_parameter("side_threshold").value)
        self.emergency_threshold = float(self.get_parameter("emergency_threshold").value)

        self.explore_interval = int(self.get_parameter("explore_interval_ticks").value)
        self.explore_turn_probability = float(self.get_parameter("explore_turn_probability").value)
        if self.explore_turn_probability < 0.0:
            self.explore_turn_probability = self.turn_probability

        self.doorway_turn_probability = float(self.get_parameter("doorway_turn_probability").value)
        self.doorway_opening_threshold = float(self.get_parameter("doorway_opening_threshold").value)
        self.doorway_wall_threshold = float(self.get_parameter("doorway_wall_threshold").value)
        self.doorway_turn_min_sec = float(self.get_parameter("doorway_turn_min_sec").value)
        self.doorway_turn_max_sec = float(self.get_parameter("doorway_turn_max_sec").value)
        self.motion_profile_enabled = parse_bool(
            self.get_parameter("motion_profile_enabled").value,
            default=False,
        )

        linear_velocity_min = float(self.get_parameter("linear_velocity_min").value)
        linear_velocity_max = float(self.get_parameter("linear_velocity_max").value)
        angular_velocity_min = float(self.get_parameter("angular_velocity_min").value)
        angular_velocity_max = float(self.get_parameter("angular_velocity_max").value)

        self.linear_velocity_min = max(0.0, linear_velocity_min if linear_velocity_min > 0.0 else self.v_forward)
        self.linear_velocity_max = max(self.linear_velocity_min, linear_velocity_max if linear_velocity_max > 0.0 else self.v_forward)
        self.angular_velocity_min = max(0.05, angular_velocity_min if angular_velocity_min > 0.0 else self.w_turn)
        self.angular_velocity_max = max(self.angular_velocity_min, angular_velocity_max if angular_velocity_max > 0.0 else self.w_turn)
        self.profile_change_interval_sec = max(0.5, float(self.get_parameter("profile_change_interval_sec").value))
        self.profile_arc_probability = float(np.clip(self.get_parameter("profile_arc_probability").value, 0.0, 1.0))
        self.profile_arc_fraction_min = max(0.0, float(self.get_parameter("profile_arc_fraction_min").value))
        self.profile_arc_fraction_max = max(
            self.profile_arc_fraction_min,
            float(self.get_parameter("profile_arc_fraction_max").value),
        )
        self.explore_spin_probability = float(np.clip(self.get_parameter("explore_spin_probability").value, 0.0, 1.0))
        self.explore_spin_min_sec = max(0.2, float(self.get_parameter("explore_spin_min_sec").value))
        self.explore_spin_max_sec = max(
            self.explore_spin_min_sec,
            float(self.get_parameter("explore_spin_max_sec").value),
        )
        self.forward_slowdown_min_factor = float(
            np.clip(self.get_parameter("forward_slowdown_min_factor").value, 0.1, 1.0)
        )
        self.nav_sector_rad = math.radians(float(self.get_parameter("nav_sector_deg").value))
        self.nav_gap_half_window_rad = math.radians(float(self.get_parameter("nav_gap_half_window_deg").value))
        self.nav_safe_clearance = float(self.get_parameter("nav_safe_clearance").value)
        self.nav_lookahead_cap = float(self.get_parameter("nav_lookahead_cap").value)
        self.nav_heading_gain = float(self.get_parameter("nav_heading_gain").value)
        self.nav_avoid_gain = float(self.get_parameter("nav_avoid_gain").value)
        self.nav_min_linear_speed = float(self.get_parameter("nav_min_linear_speed").value)
        self.nav_heading_bias_max_rad = math.radians(float(self.get_parameter("nav_heading_bias_max_deg").value))
        self.nav_heading_bias_hold_ticks = max(
            1,
            int(round(float(self.get_parameter("nav_heading_bias_hold_sec").value) * self.rate_hz)),
        )
        self.nav_heading_smooth_alpha = float(
            np.clip(self.get_parameter("nav_heading_smooth_alpha").value, 0.0, 0.95)
        )
        self.nav_novelty_lookahead_m = max(0.4, float(self.get_parameter("nav_novelty_lookahead_m").value))
        self.nav_novelty_bonus = max(0.0, float(self.get_parameter("nav_novelty_bonus").value))
        self.nav_recent_cell_penalty = max(0.0, float(self.get_parameter("nav_recent_cell_penalty").value))
        self.robot_front_extent = max(0.0, float(self.get_parameter("robot_front_extent").value))
        self.robot_rear_extent = max(0.0, float(self.get_parameter("robot_rear_extent").value))
        self.robot_half_width = max(0.0, float(self.get_parameter("robot_half_width").value))
        self.robot_safety_margin = max(0.0, float(self.get_parameter("robot_safety_margin").value))
        self.repeat_cell_size_m = max(0.2, float(self.get_parameter("repeat_cell_size_m").value))
        self.repeat_window_size = max(12, int(self.get_parameter("repeat_window_size").value))
        self.repeat_unique_ratio_threshold = float(
            np.clip(self.get_parameter("repeat_unique_ratio_threshold").value, 0.15, 0.95)
        )
        self.repeat_escape_trigger = max(1, int(self.get_parameter("repeat_escape_trigger").value))
        self.repeat_escape_turn_sec = max(0.8, float(self.get_parameter("repeat_escape_turn_sec").value))
        self.repeat_escape_heading_rad = math.radians(
            float(self.get_parameter("repeat_escape_heading_deg").value)
        )
        
        self.min_front = None
        self.min_left = None
        self.min_right = None
        self.min_front_left = None
        self.min_front_right = None
        self.avg_left = None
        self.avg_right = None
        self.turn_ticks = 0
        self.turn_dir = 1
        self.turn_linear_speed = 0.0
        self.backup_ticks = 0
        self._last_scan_time = None
        self._scan_rx_count = 0
        # Stuck detection - based on actual movement
        self.last_x = None
        self.last_y = None
        self.last_yaw = 0.0
        self.stuck_counter = 0
        self.stuck_threshold = 15  # ~1.5 seconds at 10Hz - faster reaction
        self.move_threshold = 0.01  # even smaller movement counts as stuck
        
        # Command tracking - detect when we're sending forward but not moving
        self.forward_cmd_counter = 0
        self.last_cmd_forward = False
        
        # Loop detection - check if robot returns to same area
        self.position_history = deque(maxlen=50)  # shorter history
        self.recent_cells = deque(maxlen=self.repeat_window_size)
        self.loop_counter = 0
        self.repetition_counter = 0
        self.total_distance = 0.0  # track total distance traveled
        
        # Exploration - random direction changes
        # Exploration - random direction changes
        self.explore_timer = 0
        # self.explore_interval ustawiane z parametru explore_interval_ticks
        
        # Consecutive obstacle counter
        self.obstacle_counter = 0
        
        # Cooldown after turning - don't react to obstacles immediately
        self.turn_cooldown = 0
        self.spin_counter = 0
        self.spin_threshold = int(2.5 * self.rate_hz)
        self.scan_ranges = None
        self.scan_angles = None
        self.scan_raw_ranges = None
        self.front_clearance = None
        self.preferred_heading = 0.0
        self.preferred_heading_ticks = 0
        self.filtered_heading = 0.0
        self.profile_refresh_ticks = max(1, int(round(self.profile_change_interval_sec * self.rate_hz)))
        self.profile_tick_counter = 0
        self.current_forward_speed = self.v_forward
        self.current_turn_speed = self.w_turn
        self.current_cruise_steering = 0.0

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_scan)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)
        
        self.debug = parse_bool(self.get_parameter("debug").value, default=True)
        self.debug_every_n = int(self.get_parameter("debug_every_n").value)
        self._dbg_tick = 0
        self._dbg_last_reason = ""
        self._refresh_motion_profile(force=True)

    def _choose_turn_dir(self) -> int:
        if self.avg_left is not None and self.avg_right is not None:
            if abs(float(self.avg_left) - float(self.avg_right)) < 0.15:
                return int(self.rng.choice([-1, 1]))
            return 1 if self.avg_left > self.avg_right else -1

        if self.min_front_left is not None and self.min_front_right is not None:
            if abs(float(self.min_front_left) - float(self.min_front_right)) < 0.10:
                return int(self.rng.choice([-1, 1]))
            return -1 if self.min_front_left < self.min_front_right else 1

        return int(self.rng.choice([-1, 1]))

    def _start_turn(
        self,
        turn_dir: int | None = None,
        turn_sec: float = 1.0,
        linear_speed: float = 0.0,
        min_fraction: float = 0.55,
        max_fraction: float = 0.85,
        cooldown_sec: float = 1.0,
    ) -> None:
        self.turn_dir = int(turn_dir) if turn_dir in (-1, 1) else self._choose_turn_dir()
        self.turn_ticks = max(1, int(round(float(turn_sec) * self.rate_hz)))
        self.turn_linear_speed = float(linear_speed)
        self.current_turn_speed = self._sample_turn_speed(
            min_fraction=min_fraction,
            max_fraction=max_fraction,
        )
        self.turn_cooldown = max(self.turn_cooldown, int(round(float(cooldown_sec) * self.rate_hz)))

    def _start_recovery(
        self,
        turn_dir: int | None = None,
        backup_sec: float = 0.9,
        turn_sec: float = 1.4,
        turn_linear_speed: float = 0.10,
        min_fraction: float = 0.70,
        max_fraction: float = 0.95,
        cooldown_sec: float = 1.2,
    ) -> None:
        self.turn_dir = int(turn_dir) if turn_dir in (-1, 1) else self._choose_turn_dir()
        self.backup_ticks = max(1, int(round(float(backup_sec) * self.rate_hz)))
        self.turn_ticks = max(1, int(round(float(turn_sec) * self.rate_hz)))
        self.turn_linear_speed = float(turn_linear_speed)
        self.current_turn_speed = self._sample_turn_speed(
            min_fraction=min_fraction,
            max_fraction=max_fraction,
        )
        self.turn_cooldown = max(self.turn_cooldown, int(round(float(cooldown_sec) * self.rate_hz)))
        self.last_cmd_forward = False
        self.forward_cmd_counter = 0
        self.stuck_counter = 0

    def _publish_cmd(self, twist: Twist, reason: str) -> None:
        cmd_is_forward = twist.linear.x > 0.02
        cmd_is_spin = (abs(float(twist.linear.x)) < 0.03) and (abs(float(twist.angular.z)) > 0.25)
        if cmd_is_spin:
            self.spin_counter += 1
        else:
            self.spin_counter = max(0, self.spin_counter - 2)

        self.last_cmd_forward = cmd_is_forward
        self.pub.publish(twist)

        if self.debug:
            if (self._dbg_tick % max(1, self.debug_every_n) == 0) or (reason != self._dbg_last_reason):
                self._dbg_last_reason = reason

                mf = -1.0 if self.min_front is None else float(self.min_front)
                mfl = -1.0 if self.min_front_left is None else float(self.min_front_left)
                mfr = -1.0 if self.min_front_right is None else float(self.min_front_right)
                ml = -1.0 if self.min_left is None else float(self.min_left)
                mr = -1.0 if self.min_right is None else float(self.min_right)
                al = -1.0 if self.avg_left is None else float(self.avg_left)
                ar = -1.0 if self.avg_right is None else float(self.avg_right)

                if hasattr(self, "_dbg_scan_stats"):
                    n, zeros, nans, clipped, amin, ainc = self._dbg_scan_stats
                else:
                    n, zeros, nans, clipped, amin, ainc = (0, 0, 0, 0, 0.0, 0.0)

                self.get_logger().info(
                    f"[DRV] reason={reason} cmd(v={twist.linear.x:.2f}, w={twist.angular.z:.2f}) "
                    f"front={mf:.2f} fl={mfl:.2f} fr={mfr:.2f} left(min/avg)={ml:.2f}/{al:.2f} right(min/avg)={mr:.2f}/{ar:.2f} "
                    f"stuck={self.stuck_counter:.1f} spin={self.spin_counter} fwd={int(self.last_cmd_forward)} "
                    f"scan(n={n}, zeros={zeros}, nans={nans}, maxed={clipped}, a_min={amin:.2f}, a_inc={ainc:.4f})"
                )

    def _update_preferred_heading(self, open_space: bool) -> None:
        if not open_space:
            self.preferred_heading *= 0.85
            self.preferred_heading_ticks = max(0, self.preferred_heading_ticks - 1)
            return

        if self.preferred_heading_ticks > 0:
            self.preferred_heading_ticks -= 1
            return

        choices = np.array([-1.0, -0.65, -0.35, 0.0, 0.35, 0.65, 1.0], dtype=np.float32)
        self.preferred_heading = float(self.rng.choice(choices)) * self.nav_heading_bias_max_rad
        self.preferred_heading_ticks = self.nav_heading_bias_hold_ticks

    def _choose_escape_dir(self) -> int:
        if self.avg_left is not None and self.avg_right is not None:
            if float(self.avg_left) > float(self.avg_right) + 0.08:
                return 1
            if float(self.avg_right) > float(self.avg_left) + 0.08:
                return -1
        return self._choose_turn_dir()

    def _register_recent_cell(self, x: float, y: float, *, force: bool = False) -> None:
        cell = (
            int(math.floor(float(x) / self.repeat_cell_size_m)),
            int(math.floor(float(y) / self.repeat_cell_size_m)),
        )
        if force or len(self.recent_cells) == 0 or self.recent_cells[-1] != cell:
            self.recent_cells.append(cell)

        min_window = max(8, int(round(0.5 * self.repeat_window_size)))
        if len(self.recent_cells) < min_window:
            self.repetition_counter = max(0, self.repetition_counter - 1)
            return

        unique_ratio = len(set(self.recent_cells)) / max(1, len(self.recent_cells))
        if unique_ratio < self.repeat_unique_ratio_threshold:
            self.repetition_counter += 1
        else:
            self.repetition_counter = max(0, self.repetition_counter - 2)

    def _heading_novelty_score(self, rel_heading: float, clearance: float) -> float:
        if self.last_x is None or self.last_y is None or len(self.recent_cells) == 0:
            return 0.0

        lookahead = min(
            self.nav_lookahead_cap,
            max(0.45, min(float(clearance), self.nav_novelty_lookahead_m)),
        )
        if lookahead <= 0.0:
            return 0.0

        recent_cells = list(self.recent_cells)
        scores = []
        for fraction in (0.55, 1.0):
            proj_dist = max(0.35, fraction * lookahead)
            world_heading = self.last_yaw + float(rel_heading)
            proj_x = self.last_x + proj_dist * math.cos(world_heading)
            proj_y = self.last_y + proj_dist * math.sin(world_heading)
            proj_cell = (
                int(math.floor(float(proj_x) / self.repeat_cell_size_m)),
                int(math.floor(float(proj_y) / self.repeat_cell_size_m)),
            )
            hits = recent_cells.count(proj_cell)
            if hits > 0:
                revisit_ratio = hits / max(1, len(recent_cells))
                scores.append(-self.nav_recent_cell_penalty * (0.35 + 0.65 * revisit_ratio))
            else:
                scores.append(self.nav_novelty_bonus)

        return float(np.mean(scores)) if scores else 0.0

    def _compute_gap_follow_command(self):
        if self.scan_ranges is None or self.scan_angles is None or self.scan_ranges.size == 0:
            return None

        mask = np.abs(self.scan_angles) <= self.nav_sector_rad
        if not np.any(mask):
            return None

        angles = self.scan_angles[mask]
        ranges = np.clip(self.scan_ranges[mask], 0.0, self.nav_lookahead_cap)
        if angles.size < 7:
            return None

        front_mask = np.abs(angles) <= math.radians(30.0)
        if np.any(front_mask):
            self.front_clearance = float(np.percentile(ranges[front_mask], 15))
        else:
            self.front_clearance = float(np.percentile(ranges, 15))

        wide_open = (
            self.front_clearance > max(self.front_threshold * 1.6, 0.9)
            and (self.avg_left is None or self.avg_left > self.side_threshold * 1.4)
            and (self.avg_right is None or self.avg_right > self.side_threshold * 1.4)
        )
        self._update_preferred_heading(wide_open)

        step = max(1, int(round(math.radians(3.0) / max(1e-4, float(np.median(np.diff(angles)))))))
        half_window = max(1, int(round(self.nav_gap_half_window_rad / max(1e-4, float(np.median(np.diff(angles)))))))
        best = None

        for idx in range(0, angles.size, step):
            lo = max(0, idx - half_window)
            hi = min(angles.size, idx + half_window + 1)
            local = ranges[lo:hi]
            if local.size == 0:
                continue

            local_min = float(np.percentile(local, 20))
            local_mean = float(np.mean(local))
            if local_min < self.nav_safe_clearance:
                continue

            angle = float(angles[idx])
            forward_pref = 1.0 - 0.75 * abs(angle) / max(self.nav_sector_rad, 1e-4)
            heading_pref = 1.0 - min(1.0, abs(angle - self.preferred_heading) / max(self.nav_sector_rad, 1e-4))
            score = (
                1.35 * min(local_mean, self.nav_lookahead_cap)
                + 0.55 * local_min
                + 0.45 * forward_pref
                + 0.25 * heading_pref
            )
            score += self._heading_novelty_score(angle, local_mean)
            candidate = (score, angle, local_min, local_mean)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            return None

        _, target_heading, local_min, _ = best
        alpha = self.nav_heading_smooth_alpha
        self.filtered_heading = alpha * self.filtered_heading + (1.0 - alpha) * target_heading

        side_balance = 0.0
        if self.avg_left is not None and self.avg_right is not None:
            denom = max(0.5, float(self.avg_left) + float(self.avg_right))
            side_balance = float(np.clip((float(self.avg_left) - float(self.avg_right)) / denom, -1.0, 1.0))

        angular = self.nav_heading_gain * self.filtered_heading + self.nav_avoid_gain * side_balance
        angular = float(np.clip(angular, -self.current_turn_speed, self.current_turn_speed))

        clearance_scale = np.clip(
            (self.front_clearance - self.emergency_threshold) / max(1e-3, self.front_threshold * 2.2 - self.emergency_threshold),
            0.0,
            1.0,
        )
        heading_scale = max(0.18, 1.0 - 0.82 * abs(self.filtered_heading) / max(self.nav_sector_rad, 1e-4))
        base_speed = max(0.0, float(self.current_forward_speed))
        linear = base_speed * clearance_scale * heading_scale

        if self.front_clearance > self.emergency_threshold * 1.1 and local_min > self.nav_safe_clearance:
            linear = max(self.nav_min_linear_speed, linear)
        else:
            linear = 0.0

        if abs(self.filtered_heading) > math.radians(65.0):
            linear = min(linear, max(self.nav_min_linear_speed, 0.08))

        return float(linear), float(angular), "gap_follow"

    def _sample_turn_speed(self, min_fraction: float = 0.55, max_fraction: float = 1.0) -> float:
        if not self.motion_profile_enabled:
            return max(0.05, self.w_turn)

        low = max(0.05, self.angular_velocity_min * float(min_fraction))
        high = max(low, self.angular_velocity_max * float(max_fraction))
        return float(self.rng.uniform(low, high))

    def _refresh_motion_profile(self, force: bool = False):
        if not self.motion_profile_enabled:
            self.current_forward_speed = self.v_forward
            self.current_turn_speed = self.w_turn
            self.current_cruise_steering = 0.0
            self.profile_tick_counter = 0
            return

        if not force and self.profile_tick_counter < self.profile_refresh_ticks:
            return

        self.profile_tick_counter = 0
        self.current_forward_speed = float(
            self.rng.uniform(self.linear_velocity_min, self.linear_velocity_max)
        )
        self.current_turn_speed = self._sample_turn_speed(min_fraction=0.75, max_fraction=1.0)
        self.current_cruise_steering = 0.0

        if self.rng.random() < self.profile_arc_probability:
            frac = float(self.rng.uniform(self.profile_arc_fraction_min, self.profile_arc_fraction_max))
            self.current_cruise_steering = float(self.rng.choice([-1.0, 1.0])) * frac * self.current_turn_speed

    def _scaled_forward_speed(self) -> float:
        base_speed = max(0.0, float(self.current_forward_speed))
        if self.min_front is None:
            return base_speed
        if self.min_front >= self.front_threshold:
            return base_speed

        denom = max(1e-3, self.front_threshold - self.emergency_threshold)
        margin = max(0.0, min(1.0, (float(self.min_front) - self.emergency_threshold) / denom))
        factor = self.forward_slowdown_min_factor + (1.0 - self.forward_slowdown_min_factor) * margin
        return base_speed * factor

    def on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        
        if self.last_x is not None:
            dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            self.total_distance += dist
            
            eps = self.move_threshold / self.rate_hz
            if self.last_cmd_forward:
                if dist < eps:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
            else:
                # nie jedziemy do przodu → nie eskaluj stucka
                self.stuck_counter = max(0, self.stuck_counter - 1)
        
        self.last_x = x
        self.last_y = y
        
        # Add to history - record position every 0.15m traveled
        if len(self.position_history) == 0:
            self.position_history.append((x, y, self.total_distance))
            self._register_recent_cell(x, y, force=True)
        else:
            last_hx, last_hy, _ = self.position_history[-1]
            if math.sqrt((x - last_hx)**2 + (y - last_hy)**2) > 0.15:
                self.position_history.append((x, y, self.total_distance))
                self._register_recent_cell(x, y)
        
        # Loop detection - if we're near an old position but traveled far
        if len(self.position_history) > 10:
            old_x, old_y, old_dist = self.position_history[0]
            dist_from_old = math.sqrt((x - old_x)**2 + (y - old_y)**2)
            dist_traveled = self.total_distance - old_dist
            
            # If traveled > 1.5m but back within 0.5m of start = loop!
            if dist_traveled > 1.5 and dist_from_old < 0.5:
                self.loop_counter += 1
            else:
                self.loop_counter = 0
        else:
            self.loop_counter = 0
    def _sector_stats(self, ranges: np.ndarray, angles: np.ndarray, a0: float, a1: float):
        """
        Zwraca (min_robust, mean) w sektorze [a0, a1] (radiany),
        angle=0 przód, +pi/2 lewo, -pi/2 prawo.
        """
        if a0 > a1:
            a0, a1 = a1, a0
        mask = (angles >= a0) & (angles <= a1)
        if not np.any(mask):
            return None, None

        rr = ranges[mask]
        if rr.size == 0:
            return None, None

        # Robust: ignoruj pojedyncze "glitche" (np. 0.0 / range_min)
        # zamiast czystego min bierz 10 percentyl
        min_robust = float(np.percentile(rr, 10))
        mean_v = float(np.mean(rr))
        return min_robust, mean_v

    def _compute_footprint_offsets(self, angles: np.ndarray) -> np.ndarray:
        if angles.size == 0:
            return np.empty_like(angles, dtype=np.float32)

        dx = np.cos(angles, dtype=np.float32)
        dy = np.sin(angles, dtype=np.float32)
        offsets = np.full(angles.shape, np.inf, dtype=np.float32)
        eps = np.float32(1e-5)

        if self.robot_front_extent > 0.0:
            mask = dx > eps
            t = np.divide(
                np.float32(self.robot_front_extent),
                dx,
                out=np.full_like(dx, np.inf),
                where=mask,
            )
            y = t * dy
            valid = mask & (np.abs(y) <= (self.robot_half_width + 1e-4))
            offsets = np.minimum(offsets, np.where(valid, t, np.inf).astype(np.float32))

        if self.robot_rear_extent > 0.0:
            mask = dx < -eps
            t = np.divide(
                np.float32(-self.robot_rear_extent),
                dx,
                out=np.full_like(dx, np.inf),
                where=mask,
            )
            y = t * dy
            valid = mask & (np.abs(y) <= (self.robot_half_width + 1e-4))
            offsets = np.minimum(offsets, np.where(valid, t, np.inf).astype(np.float32))

        if self.robot_half_width > 0.0:
            mask = dy > eps
            t = np.divide(
                np.float32(self.robot_half_width),
                dy,
                out=np.full_like(dy, np.inf),
                where=mask,
            )
            x = t * dx
            valid = mask & (x >= (-self.robot_rear_extent - 1e-4)) & (x <= (self.robot_front_extent + 1e-4))
            offsets = np.minimum(offsets, np.where(valid, t, np.inf).astype(np.float32))

            mask = dy < -eps
            t = np.divide(
                np.float32(-self.robot_half_width),
                dy,
                out=np.full_like(dy, np.inf),
                where=mask,
            )
            x = t * dx
            valid = mask & (x >= (-self.robot_rear_extent - 1e-4)) & (x <= (self.robot_front_extent + 1e-4))
            offsets = np.minimum(offsets, np.where(valid, t, np.inf).astype(np.float32))

        offsets = np.where(np.isfinite(offsets), offsets, 0.0).astype(np.float32)
        return np.maximum(offsets, 0.0)

    def on_scan(self, msg: LaserScan):
        self._last_scan_time = self.get_clock().now()
        self._scan_rx_count += 1
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = int(ranges.size)
        if n == 0:
            return

        rmax = float(msg.range_max) if msg.range_max > 0 else 10.0
        rmin = float(msg.range_min) if msg.range_min > 0 else 0.08

        # --- sanitize: INF/NAN oraz zera traktuj jako "brak pomiaru"
        raw = ranges.copy()
        invalid = (~np.isfinite(raw)) | (raw <= 1e-3)
        ranges = np.where(invalid, rmax, raw).astype(np.float32)
        ranges = np.clip(ranges, rmin, rmax)
        padded = np.pad(ranges, (2, 2), mode="edge")
        ranges = np.array([np.median(padded[i:i + 5]) for i in range(n)], dtype=np.float32)

        # kąty wiązek
        angles = float(msg.angle_min) + np.arange(n, dtype=np.float32) * float(msg.angle_increment)
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi  # normalize do [-pi, pi]
        self.scan_angles = angles
        self.scan_raw_ranges = ranges
        footprint_offsets = self._compute_footprint_offsets(angles)
        clearances = np.maximum(0.0, ranges - footprint_offsets - self.robot_safety_margin).astype(np.float32)
        self.scan_ranges = clearances

        # sektory
        front = math.radians(35.0)     # szerszy front
        diag0 = math.radians(25.0)
        diag1 = math.radians(75.0)
        side0 = math.radians(65.0)
        side1 = math.radians(115.0)

        self.min_front, _ = self._sector_stats(clearances, angles, -front, +front)
        self.min_front_left, _ = self._sector_stats(clearances, angles, +diag0, +diag1)
        self.min_front_right, _ = self._sector_stats(clearances, angles, -diag1, -diag0)

        self.min_left, self.avg_left = self._sector_stats(clearances, angles, +side0, +side1)
        self.min_right, self.avg_right = self._sector_stats(clearances, angles, -side1, -side0)
        if self.debug:
            raw = np.asarray(msg.ranges, dtype=np.float32)
            n = raw.size
            if n > 0:
                rmax = float(msg.range_max) if msg.range_max > 0 else 10.0
                zeros = int(np.sum(raw <= 1e-3))
                nans = int(np.sum(~np.isfinite(raw)))
                clipped = int(np.sum(np.isfinite(raw) & (raw >= rmax - 1e-3)))
                self._dbg_scan_stats = (n, zeros, nans, clipped, float(msg.angle_min), float(msg.angle_increment))
        


    def on_timer(self):
        twist = Twist()
        self._dbg_tick += 1
        self.profile_tick_counter += 1
        self._refresh_motion_profile()
        reason = "drive"

        if self._last_scan_time is None or self.scan_ranges is None or self.scan_angles is None:
            if self.debug and (self._dbg_tick % 20 == 0):
                self.get_logger().warn(f"[DRV] Waiting for LaserScan on topic: {self.scan_topic}")
            self._publish_cmd(twist, "waiting_for_scan")
            return

        if self.repetition_counter >= self.repeat_escape_trigger:
            escape_dir = self._choose_escape_dir()
            escape_heading = float(np.clip(
                float(escape_dir) * self.repeat_escape_heading_rad,
                -self.nav_sector_rad,
                self.nav_sector_rad,
            ))
            self.preferred_heading = escape_heading
            self.filtered_heading = escape_heading
            self.preferred_heading_ticks = max(
                self.preferred_heading_ticks,
                2 * self.nav_heading_bias_hold_ticks,
            )
            self.repetition_counter = 0
            self.recent_cells.clear()

            if self.min_front is not None and self.min_front < self.front_threshold * 1.15:
                self._start_recovery(
                    turn_dir=escape_dir,
                    backup_sec=0.7,
                    turn_sec=max(self.repeat_escape_turn_sec, 2.2),
                    turn_linear_speed=min(0.14, max(0.08, 0.24 * self.current_forward_speed)),
                    min_fraction=0.70,
                    max_fraction=0.95,
                    cooldown_sec=1.8,
                )
                twist.linear.x = -0.18
                twist.angular.z = min(1.0, 0.30 * max(0.05, self.current_turn_speed)) * self.turn_dir
                self._publish_cmd(twist, "repetition_escape_recovery")
                return

            self._start_turn(
                turn_dir=escape_dir,
                turn_sec=self.repeat_escape_turn_sec,
                linear_speed=min(0.16, max(0.10, 0.30 * self.current_forward_speed)),
                min_fraction=0.65,
                max_fraction=0.95,
                cooldown_sec=1.8,
            )
            twist.linear.x = self.turn_linear_speed
            twist.angular.z = float(self.turn_dir) * self.current_turn_speed
            self._publish_cmd(twist, "repetition_escape")
            return

        if self.spin_counter > self.spin_threshold:
            self._start_recovery(
                turn_dir=self._choose_turn_dir(),
                backup_sec=1.0,
                turn_sec=1.6,
                turn_linear_speed=min(0.14, max(0.08, 0.30 * self.current_forward_speed)),
                min_fraction=0.75,
                max_fraction=1.0,
                cooldown_sec=1.5,
            )
            self.spin_counter = 0
            reason = "spin_recovery"
        # Check for loop - robot circling around obstacle
        if self.loop_counter > 3:  # faster detection
            self._start_recovery(
                turn_dir=self._choose_turn_dir(),
                backup_sec=2.0,
                turn_sec=2.8,
                turn_linear_speed=min(0.14, max(0.08, 0.28 * self.current_forward_speed)),
                min_fraction=0.65,
                max_fraction=0.95,
                cooldown_sec=1.6,
            )
            self.loop_counter = 0
            self.position_history.clear()
            self.total_distance = 0.0
            reason = "loop_recovery"
        
        # Check if stuck - emergency maneuver (VERY IMPORTANT)
        if self.stuck_counter > self.stuck_threshold:
            self._start_recovery(
                turn_dir=self._choose_turn_dir(),
                backup_sec=1.5,
                turn_sec=float(self.rng.uniform(1.5, 3.0)),
                turn_linear_speed=min(0.14, max(0.08, 0.32 * self.current_forward_speed)),
                min_fraction=0.75,
                max_fraction=1.0,
                cooldown_sec=1.5,
            )
            reason = "stuck_recovery"
        
        # Backup phase (reverse)
        if self.backup_ticks > 0:
            twist.linear.x = -0.2  # faster reverse
            twist.angular.z = min(1.2, 0.35 * max(0.05, self.current_turn_speed)) * self.turn_dir
            self.backup_ticks -= 1
            self._publish_cmd(twist, reason if reason != "drive" else "backup")
            return
        
        # Turn phase
        if self.turn_ticks > 0:
            twist.linear.x = self.turn_linear_speed
            twist.angular.z = float(self.turn_dir) * max(0.05, self.current_turn_speed)
            self.turn_ticks -= 1
            self._publish_cmd(twist, reason if reason != "drive" else ("arc_turn" if self.turn_linear_speed > 0.0 else "turn"))
            return
        
        # Cooldown after turning - decrement and skip obstacle detection
        if self.turn_cooldown > 0:
            self.turn_cooldown -= 1

        front_obstacle = (self.min_front is not None) and (self.min_front < self.front_threshold)
        front_left_obstacle = (self.min_front_left is not None) and (self.min_front_left < self.front_threshold)
        front_right_obstacle = (self.min_front_right is not None) and (self.min_front_right < self.front_threshold)
        left_obstacle = (self.min_left is not None) and (self.min_left < self.side_threshold)
        right_obstacle = (self.min_right is not None) and (self.min_right < self.side_threshold)
        
        # EMERGENCY: Very close obstacle - immediate backup!
        emergency_front = (self.min_front is not None) and (self.min_front < self.emergency_threshold)
        emergency_fl = (self.min_front_left is not None) and (self.min_front_left < self.emergency_threshold)
        emergency_fr = (self.min_front_right is not None) and (self.min_front_right < self.emergency_threshold)
        
        # If LIDAR shows we're very close AND we were trying to go forward = we hit something!
        if (emergency_front or emergency_fl or emergency_fr) and self.last_cmd_forward:
            turn_dir = self._choose_turn_dir()
            if self.min_front_left is not None and self.min_front_right is not None:
                turn_dir = -1 if self.min_front_left < self.min_front_right else 1
            self._start_recovery(
                turn_dir=turn_dir,
                backup_sec=1.5,
                turn_sec=float(self.rng.uniform(1.5, 3.0)),
                turn_linear_speed=min(0.12, max(0.07, 0.25 * self.current_forward_speed)),
                min_fraction=0.75,
                max_fraction=1.0,
                cooldown_sec=1.5,
            )
            twist.linear.x = -0.25  # fast reverse
            twist.angular.z = min(1.2, 0.3 * max(0.05, self.current_turn_speed)) * self.turn_dir
            self._publish_cmd(twist, "emergency_recovery")
            return
        
        # Jeśli coś jest SUPER blisko, cofaj TYLKO gdy faktycznie pchaliśmy do przodu.
        super_close = (self.min_front is not None) and (self.min_front < 0.20)
        if super_close and self.last_cmd_forward:
            twist.linear.x = -0.2
            self.current_turn_speed = self._sample_turn_speed(min_fraction=0.55, max_fraction=0.8)
            twist.angular.z = float(self.rng.choice([-1.0, 1.0])) * min(1.0, 0.4 * self.current_turn_speed)
            reason = "super_close_backup"
            self._publish_cmd(twist, reason)
            return
        
        any_front_obstacle = front_obstacle or front_left_obstacle or front_right_obstacle

        # Track consecutive obstacles
        if any_front_obstacle:
            self.obstacle_counter += 1
        else:
            self.obstacle_counter = 0
        
        # If stuck on obstacle too long, do bigger turn
        if self.obstacle_counter > 20:  # 2 seconds of continuous obstacle
            self._start_recovery(
                turn_dir=self._choose_turn_dir(),
                backup_sec=0.7,
                turn_sec=2.2,
                turn_linear_speed=min(0.12, max(0.08, 0.25 * self.current_forward_speed)),
                min_fraction=0.70,
                max_fraction=0.95,
                cooldown_sec=1.4,
            )
            self.obstacle_counter = 0
            twist.linear.x = -0.2
            twist.angular.z = min(1.0, 0.30 * max(0.05, self.current_turn_speed)) * self.turn_dir
            self._publish_cmd(twist, "obstacle_recovery")
            return

        # Obstacle avoidance - skip if in cooldown
        if self.turn_cooldown == 0 and (front_obstacle or (left_obstacle and right_obstacle)):
            # Obstacle ahead or both sides blocked
            reason = "obstacle_turn"
            turn_dir = self._choose_turn_dir()
            if self.rng.random() < self.turn_probability:
                turn_dir = int(self.rng.choice([-1, 1]))

            hard_block = (
                (self.min_front is not None and self.min_front < max(self.emergency_threshold * 1.2, 0.30))
                or (front_left_obstacle and front_right_obstacle)
            )
            if hard_block:
                self._start_recovery(
                    turn_dir=turn_dir,
                    backup_sec=0.8,
                    turn_sec=1.4,
                    turn_linear_speed=min(0.12, max(0.08, 0.26 * self.current_forward_speed)),
                    min_fraction=0.65,
                    max_fraction=0.90,
                    cooldown_sec=1.2,
                )
                twist.linear.x = -0.18
                twist.angular.z = min(1.0, 0.30 * max(0.05, self.current_turn_speed)) * self.turn_dir
                self._publish_cmd(twist, "obstacle_recovery")
                return
            else:
                self._start_turn(
                    turn_dir=turn_dir,
                    turn_sec=float(self.rng.uniform(0.8, 1.3)),
                    linear_speed=min(0.12, max(0.06, 0.22 * self.current_forward_speed)),
                    min_fraction=0.50,
                    max_fraction=0.75,
                    cooldown_sec=1.0,
                )
            twist.linear.x = self.turn_linear_speed
            twist.angular.z = float(self.turn_dir) * self.current_turn_speed
            self._publish_cmd(twist, reason)
            return
        
        # Front-diagonal obstacles - steer away (skip if in cooldown)
        if self.turn_cooldown == 0 and front_left_obstacle and not front_right_obstacle:
            self._start_turn(
                turn_dir=-1,
                turn_sec=0.9,
                linear_speed=min(0.11, max(0.05, 0.20 * self.current_forward_speed)),
                min_fraction=0.55,
                max_fraction=0.80,
                cooldown_sec=1.0,
            )
            twist.linear.x = self.turn_linear_speed
            twist.angular.z = -self.current_turn_speed
            self._publish_cmd(twist, "front_left_escape")
            return
        elif self.turn_cooldown == 0 and front_right_obstacle and not front_left_obstacle:
            self._start_turn(
                turn_dir=1,
                turn_sec=0.9,
                linear_speed=min(0.11, max(0.05, 0.20 * self.current_forward_speed)),
                min_fraction=0.55,
                max_fraction=0.80,
                cooldown_sec=1.0,
            )
            twist.linear.x = self.turn_linear_speed
            twist.angular.z = self.current_turn_speed
            self._publish_cmd(twist, "front_right_escape")
            return
        
        nav_cmd = self._compute_gap_follow_command()
        if nav_cmd is None:
            self._start_recovery(
                turn_dir=self._choose_turn_dir(),
                backup_sec=0.8,
                turn_sec=1.8,
                turn_linear_speed=min(0.10, max(0.06, 0.18 * self.current_forward_speed)),
                min_fraction=0.70,
                max_fraction=0.95,
                cooldown_sec=1.4,
            )
            twist.linear.x = -0.16
            twist.angular.z = min(1.0, 0.25 * max(0.05, self.current_turn_speed)) * self.turn_dir
            self._publish_cmd(twist, "no_safe_gap")
            return

        twist.linear.x, twist.angular.z, reason = nav_cmd
        self._publish_cmd(twist, reason)

def main():
    rclpy.init()
    node = AutoDriver()
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
