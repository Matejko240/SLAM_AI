import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

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
        
        self.min_front = None
        self.min_left = None
        self.min_right = None
        self.min_front_left = None
        self.min_front_right = None
        self.avg_left = None
        self.avg_right = None
        self.turn_ticks = 0
        self.turn_dir = 1
        self.backup_ticks = 0
        self._last_scan_time = None
        self._scan_rx_count = 0
        # Stuck detection - based on actual movement
        self.last_x = None
        self.last_y = None
        self.stuck_counter = 0
        self.stuck_threshold = 15  # ~1.5 seconds at 10Hz - faster reaction
        self.move_threshold = 0.01  # even smaller movement counts as stuck
        
        # Command tracking - detect when we're sending forward but not moving
        self.forward_cmd_counter = 0
        self.last_cmd_forward = False
        
        # Loop detection - check if robot returns to same area
        self.position_history = deque(maxlen=50)  # shorter history
        self.loop_counter = 0
        self.total_distance = 0.0  # track total distance traveled
        
        # Exploration - random direction changes
        # Exploration - random direction changes
        self.explore_timer = 0
        # self.explore_interval ustawiane z parametru explore_interval_ticks
        
        # Consecutive obstacle counter
        self.obstacle_counter = 0
        
        # Cooldown after turning - don't react to obstacles immediately
        self.turn_cooldown = 0

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
        
        self.debug = bool(self.get_parameter("debug").value)
        self.debug_every_n = int(self.get_parameter("debug_every_n").value)
        self._dbg_tick = 0
        self._dbg_last_reason = ""

    def on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
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
        else:
            last_hx, last_hy, _ = self.position_history[-1]
            if math.sqrt((x - last_hx)**2 + (y - last_hy)**2) > 0.15:
                self.position_history.append((x, y, self.total_distance))
        
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

        # kąty wiązek
        angles = float(msg.angle_min) + np.arange(n, dtype=np.float32) * float(msg.angle_increment)
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi  # normalize do [-pi, pi]

        # sektory
        front = math.radians(25.0)     # +/-25°
        diag0 = math.radians(25.0)
        diag1 = math.radians(65.0)
        side0 = math.radians(70.0)
        side1 = math.radians(110.0)

        self.min_front, _ = self._sector_stats(ranges, angles, -front, +front)
        self.min_front_left, _ = self._sector_stats(ranges, angles, +diag0, +diag1)
        self.min_front_right, _ = self._sector_stats(ranges, angles, -diag1, -diag0)

        self.min_left, self.avg_left = self._sector_stats(ranges, angles, +side0, +side1)
        self.min_right, self.avg_right = self._sector_stats(ranges, angles, -side1, -side0)
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
        if self.debug and self._last_scan_time is None and (self._dbg_tick % 20 == 0):
            self.get_logger().warn(f"[DRV] No LaserScan received yet on topic: {self.scan_topic}")
        self._dbg_tick += 1
        reason = "drive"
        # Check for loop - robot circling around obstacle
        if self.loop_counter > 3:  # faster detection
            self.backup_ticks = int(2.0 * self.rate_hz)  # longer backup
            self.turn_ticks = int(3.5 * self.rate_hz)   # longer turn to escape
            self.turn_dir = int(self.rng.choice([-1, 1]))
            self.loop_counter = 0
            self.position_history.clear()
            self.total_distance = 0.0
        
        # Check if stuck - emergency maneuver (VERY IMPORTANT)
        if self.stuck_counter > self.stuck_threshold:
            self.backup_ticks = int(1.5 * self.rate_hz)  # 1.5 seconds backup
            self.turn_ticks = int(self.rng.uniform(1.5, 3) * self.rate_hz)  # big random turn
            # Random direction with bias away from closest obstacle
            if self.avg_left is not None and self.avg_right is not None:
                if abs(self.avg_left - self.avg_right) < 0.2:
                    self.turn_dir = int(self.rng.choice([-1, 1]))
                else:
                    self.turn_dir = 1 if self.avg_left > self.avg_right else -1
            else:
                self.turn_dir = int(self.rng.choice([-1, 1]))
            self.stuck_counter = 0
            self.forward_cmd_counter = 0
            self.last_cmd_forward = False
        
        # Backup phase (reverse)
        if self.backup_ticks > 0:
            twist.linear.x = -0.2  # faster reverse
            twist.angular.z = 0.5 * self.turn_dir  # turn while backing
            self.backup_ticks -= 1
            self.last_cmd_forward = False
            self.pub.publish(twist)
            return
        
        # Turn phase
        if self.turn_ticks > 0:
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.turn_ticks -= 1
            self.last_cmd_forward = False
            self.pub.publish(twist)
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
            self.backup_ticks = int(1.5 * self.rate_hz)
            self.turn_ticks = int(self.rng.uniform(1.5, 3) * self.rate_hz)
            # Turn AWAY from the closest obstacle
            if self.min_front_left is not None and self.min_front_right is not None:
                if self.min_front_left < self.min_front_right:
                    self.turn_dir = -1  # obstacle on left, turn right
                else:
                    self.turn_dir = 1   # obstacle on right, turn left
            else:
                self.turn_dir = int(self.rng.choice([-1, 1]))
            self.last_cmd_forward = False
            self.stuck_counter = 0
            twist.linear.x = -0.25  # fast reverse
            twist.angular.z = 0.3 * self.turn_dir
            self.pub.publish(twist)
            return
        
        # Jeśli coś jest SUPER blisko, cofaj TYLKO gdy faktycznie pchaliśmy do przodu.
        super_close = (self.min_front is not None) and (self.min_front < 0.20)
        if super_close and self.last_cmd_forward:
            twist.linear.x = -0.2
            twist.angular.z = float(self.rng.choice([-1.0, 1.0]))
            self.last_cmd_forward = False
            self.pub.publish(twist)
            reason = "super_close_backup"
            return
        
        any_front_obstacle = front_obstacle or front_left_obstacle or front_right_obstacle

        # Track consecutive obstacles
        if any_front_obstacle:
            self.obstacle_counter += 1
        else:
            self.obstacle_counter = 0
        
        # If stuck on obstacle too long, do bigger turn
        if self.obstacle_counter > 20:  # 2 seconds of continuous obstacle
            self.backup_ticks = int(0.5 * self.rate_hz)
            self.turn_dir = int(self.rng.choice([-1, 1]))
            self.turn_ticks = int(2.5 * self.rate_hz)
            self.obstacle_counter = 0
            self.last_cmd_forward = False
            return

        # Obstacle avoidance - skip if in cooldown
        if self.turn_cooldown == 0 and (front_obstacle or (left_obstacle and right_obstacle)):
            # Obstacle ahead or both sides blocked
            if self.avg_left is not None and self.avg_right is not None:
                reason = "obstacle_turn"
                # Add randomness to avoid getting stuck in patterns
                if self.rng.random() < self.turn_probability:
                    self.turn_dir = int(self.rng.choice([-1, 1]))
                else:
                    self.turn_dir = 1 if self.avg_left > self.avg_right else -1
            else:
                self.turn_dir = int(self.rng.choice([-1, 1]))
            self.turn_ticks = int(self.rng.uniform(0.7, 1.2) * self.rate_hz)  # 0.7-1.2s (~40-70°)
            self.turn_cooldown = int(1.0 * self.rate_hz)  # 1s cooldown
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.last_cmd_forward = False
            self.pub.publish(twist)
            return
        
        # Front-diagonal obstacles - steer away (skip if in cooldown)
        if self.turn_cooldown == 0 and front_left_obstacle and not front_right_obstacle:
            self.turn_dir = -1
            self.turn_ticks = int(0.9 * self.rate_hz)  # 1.5s (~90°)
            self.turn_cooldown = int(1.0 * self.rate_hz)
            twist.linear.x = 0.0
            twist.angular.z = -self.w_turn
            self.last_cmd_forward = False
            self.pub.publish(twist)
            return
        elif self.turn_cooldown == 0 and front_right_obstacle and not front_left_obstacle:
            self.turn_dir = 1
            self.turn_ticks = int(0.9 * self.rate_hz)  # 1.5s (~90°)
            self.turn_cooldown = int(1.0 * self.rate_hz)
            twist.linear.x = 0.0
            twist.angular.z = self.w_turn
            self.last_cmd_forward = False
            self.pub.publish(twist)
            return
        
        # Wall following - gentle steering away from close walls
        steering = 0.0
        if left_obstacle and not right_obstacle:
            steering = -0.6 * self.w_turn
        elif right_obstacle and not left_obstacle:
            steering = 0.6 * self.w_turn
        # Doorway / room-entering heuristic:
        # Jeśli przód jest wolny, a jedna strona wygląda jak "otwarcie" (dużo przestrzeni),
        # a druga jak "ściana", to czasem skręcamy w stronę otwarcia.
        if (
            self.turn_cooldown == 0
            and self.doorway_turn_probability > 0.0
            and not any_front_obstacle
            and self.avg_left is not None
            and self.avg_right is not None
        ):
            left_open = (self.avg_left > self.doorway_opening_threshold) and (self.avg_right < self.doorway_wall_threshold)
            right_open = (self.avg_right > self.doorway_opening_threshold) and (self.avg_left < self.doorway_wall_threshold)

            if (left_open or right_open) and (self.rng.random() < self.doorway_turn_probability):
                # +1 = skręt w lewo (CCW), -1 = skręt w prawo (CW) – zgodnie z resztą Twojego kodu
                self.turn_dir = 1 if left_open else -1
                turn_sec = float(self.rng.uniform(self.doorway_turn_min_sec, self.doorway_turn_max_sec))
                self.turn_ticks = int(turn_sec * self.rate_hz)
                self.turn_cooldown = int(1.0 * self.rate_hz)
                twist.linear.x = 0.0
                twist.angular.z = float(self.turn_dir) * self.w_turn
                self.last_cmd_forward = False
                reason = "doorway_turn"
                self.pub.publish(twist)
                return
        # Random exploration turns - less frequent, rely more on straight driving
        self.explore_timer += 1
        if self.explore_timer >= self.explore_interval:
            self.explore_timer = 0
            if self.rng.random() < self.explore_turn_probability:
                self.turn_dir = int(self.rng.choice([-1, 1]))
                self.turn_ticks = int(self.rng.uniform(0.8, 1.5) * self.rate_hz)  # longer turns
                twist.linear.x = 0.0
                twist.angular.z = float(self.turn_dir) * self.w_turn
                self.last_cmd_forward = False
                self.pub.publish(twist)
                reason = "explore_turn"
                return

        # Normal forward motion - STRAIGHT, minimal wander
        twist.linear.x = self.v_forward
        # Only add steering correction, NO sine wave wandering
        twist.angular.z = steering
        self.last_cmd_forward = True
        self.pub.publish(twist)
        # --- DEBUG: log co N ticków lub przy zmianie powodu (DZIAŁA ZAWSZE w tej ścieżce)
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
                    f"stuck={self.stuck_counter:.1f} fwd={int(self.last_cmd_forward)} "
                    f"scan(n={n}, zeros={zeros}, nans={nans}, maxed={clipped}, a_min={amin:.2f}, a_inc={ainc:.4f})"
                )

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
