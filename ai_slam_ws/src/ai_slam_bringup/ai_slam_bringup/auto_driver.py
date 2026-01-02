import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class AutoDriver(Node):
    def __init__(self):
        super().__init__("auto_driver")
        self.declare_parameter("seed", 123)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan_slam")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("v_forward", 0.25)
        self.declare_parameter("w_turn", 1.0)
        self.declare_parameter("front_threshold", 0.55)
        self.declare_parameter("side_threshold", 0.35)

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.v_forward = float(self.get_parameter("v_forward").value)
        self.w_turn = float(self.get_parameter("w_turn").value)
        self.front_threshold = float(self.get_parameter("front_threshold").value)
        self.side_threshold = float(self.get_parameter("side_threshold").value)

        self.min_front = None
        self.min_left = None
        self.min_right = None
        self.avg_left = None
        self.avg_right = None
        self.turn_ticks = 0
        self.turn_dir = 1
        self.backup_ticks = 0
        
        # Stuck detection
        self.last_x = None
        self.last_y = None
        self.stuck_counter = 0
        self.stuck_threshold = 30  # ~3 seconds at 10Hz
        self.move_threshold = 0.02  # minimum movement in 1 second
        
        # Exploration - random direction changes
        self.explore_timer = 0
        self.explore_interval = 50  # ticks between random turns

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

    def on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        if self.last_x is not None:
            dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            if dist < self.move_threshold / self.rate_hz:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        self.last_x = x
        self.last_y = y

    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return
        
        # Replace inf/nan with max range
        max_range = msg.range_max
        ranges = np.where(np.isfinite(ranges), ranges, max_range)
        
        # Front: indices around 0 (±15 degrees)
        front_window = max(1, int(n * 15 / 360))
        front_indices = [(i % n) for i in range(-front_window, front_window + 1)]
        front = ranges[front_indices]
        self.min_front = float(np.min(front))
        
        # Left: indices around 90 degrees (±30 degrees)
        left_center = n // 4  # 90 degrees
        left_window = max(1, int(n * 30 / 360))
        left_indices = [(left_center + i) % n for i in range(-left_window, left_window + 1)]
        left = ranges[left_indices]
        self.min_left = float(np.min(left))
        self.avg_left = float(np.mean(left))
        
        # Right: indices around 270 degrees (±30 degrees)
        right_center = 3 * n // 4  # 270 degrees
        right_window = max(1, int(n * 30 / 360))
        right_indices = [(right_center + i) % n for i in range(-right_window, right_window + 1)]
        right = ranges[right_indices]
        self.min_right = float(np.min(right))
        self.avg_right = float(np.mean(right))

    def on_timer(self):
        twist = Twist()
        
        # Check if stuck - emergency maneuver
        if self.stuck_counter > self.stuck_threshold:
            self.get_logger().warn("Robot stuck! Executing escape maneuver...")
            self.backup_ticks = int(1.0 * self.rate_hz)  # backup for 1 second
            self.turn_ticks = int(2.0 * self.rate_hz)   # then turn for 2 seconds
            # Turn towards more open space
            if self.avg_left is not None and self.avg_right is not None:
                self.turn_dir = 1 if self.avg_left > self.avg_right else -1
            else:
                self.turn_dir = int(self.rng.choice([-1, 1]))
            self.stuck_counter = 0
        
        # Backup phase (reverse)
        if self.backup_ticks > 0:
            twist.linear.x = -0.15  # reverse slowly
            twist.angular.z = 0.0
            self.backup_ticks -= 1
            self.pub.publish(twist)
            return
        
        # Turn phase
        if self.turn_ticks > 0:
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.turn_ticks -= 1
            self.pub.publish(twist)
            return

        front_obstacle = (self.min_front is not None) and (self.min_front < self.front_threshold)
        left_obstacle = (self.min_left is not None) and (self.min_left < self.side_threshold)
        right_obstacle = (self.min_right is not None) and (self.min_right < self.side_threshold)

        # Obstacle avoidance
        if front_obstacle or (left_obstacle and right_obstacle):
            # Obstacle ahead or both sides blocked - turn towards more open space
            if self.avg_left is not None and self.avg_right is not None:
                self.turn_dir = 1 if self.avg_left > self.avg_right else -1
            else:
                self.turn_dir = int(self.rng.choice([-1, 1]))
            self.turn_ticks = int(self.rng.uniform(1.0, 2.0) * self.rate_hz)
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.pub.publish(twist)
            return
        
        # Wall following - gentle steering away from close walls
        steering = 0.0
        if left_obstacle and not right_obstacle:
            steering = -0.5 * self.w_turn  # steer right
        elif right_obstacle and not left_obstacle:
            steering = 0.5 * self.w_turn   # steer left
        
        # Random exploration turns
        self.explore_timer += 1
        if self.explore_timer >= self.explore_interval:
            self.explore_timer = 0
            # Random chance to change direction
            if self.rng.random() < 0.3:
                self.turn_dir = int(self.rng.choice([-1, 1]))
                self.turn_ticks = int(self.rng.uniform(0.5, 1.5) * self.rate_hz)
                twist.linear.x = 0.0
                twist.angular.z = float(self.turn_dir) * self.w_turn
                self.pub.publish(twist)
                return

        # Normal forward motion with slight wander
        t = self.get_clock().now().nanoseconds * 1e-9
        twist.linear.x = self.v_forward
        twist.angular.z = steering + 0.3 * self.w_turn * math.sin(0.3 * t)
        self.pub.publish(twist)


def main():
    rclpy.init()
    node = AutoDriver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
