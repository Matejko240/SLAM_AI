import math
import numpy as np
from collections import deque

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
        self.declare_parameter("front_threshold", 0.45)   # react when < 45cm
        self.declare_parameter("side_threshold", 0.35)    # react when < 35cm
        self.declare_parameter("emergency_threshold", 0.25)  # very close = emergency < 25cm

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.v_forward = float(self.get_parameter("v_forward").value)
        self.w_turn = float(self.get_parameter("w_turn").value)
        self.front_threshold = float(self.get_parameter("front_threshold").value)
        self.side_threshold = float(self.get_parameter("side_threshold").value)
        self.emergency_threshold = float(self.get_parameter("emergency_threshold").value)

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
        self.explore_timer = 0
        self.explore_interval = 30  # more frequent random turns
        
        # Consecutive obstacle counter
        self.obstacle_counter = 0
        
        # Cooldown after turning - don't react to obstacles immediately
        self.turn_cooldown = 0

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

    def on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        if self.last_x is not None:
            dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            self.total_distance += dist
            
            # If we're trying to go forward but not moving - we're stuck!
            if self.last_cmd_forward and dist < self.move_threshold / self.rate_hz:
                self.stuck_counter += 1
            elif dist < self.move_threshold / self.rate_hz:
                self.stuck_counter += 0.5
            else:
                self.stuck_counter = 0
                self.forward_cmd_counter = 0
        
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

    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return
        
        # Replace inf/nan with max range
        max_range = msg.range_max
        ranges = np.where(np.isfinite(ranges), ranges, max_range)
        
        # Log raw min for debugging
        raw_min = float(np.min(ranges))
        
        # NOTE: Index 0 = BACK of robot, Index n/2 = FRONT of robot
        # Front: indices around n/2 (180 degrees) ±20 degrees
        front_center = n // 2
        front_window = max(1, int(n * 20 / 360))
        front_indices = [(front_center + i) % n for i in range(-front_window, front_window + 1)]
        front = ranges[front_indices]
        self.min_front = float(np.min(front))
        
        # Front-left: around 135 degrees (±15 degrees)
        fl_center = 3 * n // 8  # 135 degrees
        fl_window = max(1, int(n * 15 / 360))
        fl_indices = [(fl_center + i) % n for i in range(-fl_window, fl_window + 1)]
        self.min_front_left = float(np.min(ranges[fl_indices]))
        
        # Front-right: around 225 degrees (±15 degrees)
        fr_center = 5 * n // 8  # 225 degrees
        fr_window = max(1, int(n * 15 / 360))
        fr_indices = [(fr_center + i) % n for i in range(-fr_window, fr_window + 1)]
        self.min_front_right = float(np.min(ranges[fr_indices]))
        
        # Right: indices around 90 degrees (±30 degrees) - swapped!
        right_center = n // 4  # 90 degrees
        right_window = max(1, int(n * 30 / 360))
        right_indices = [(right_center + i) % n for i in range(-right_window, right_window + 1)]
        right = ranges[right_indices]
        self.min_right = float(np.min(right))
        self.avg_right = float(np.mean(right))
        
        # Left: indices around 270 degrees (±30 degrees) - swapped!
        left_center = 3 * n // 4  # 270 degrees
        left_window = max(1, int(n * 30 / 360))
        left_indices = [(left_center + i) % n for i in range(-left_window, left_window + 1)]
        left = ranges[left_indices]
        self.min_left = float(np.min(left))
        self.avg_left = float(np.mean(left))
        


    def on_timer(self):
        twist = Twist()
        
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
            self.turn_ticks = int(self.rng.uniform(2.0, 4.0) * self.rate_hz)  # big random turn
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
            self.turn_ticks = int(self.rng.uniform(2.0, 3.5) * self.rate_hz)
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
        
        # Even if not trying to go forward, if something is SUPER close, backup
        super_close = (self.min_front is not None) and (self.min_front < 0.20)
        if super_close:
            twist.linear.x = -0.2
            twist.angular.z = self.rng.choice([-1.0, 1.0])
            self.last_cmd_forward = False
            self.pub.publish(twist)
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
                # Add randomness to avoid getting stuck in patterns
                if self.rng.random() < 0.2:
                    self.turn_dir = int(self.rng.choice([-1, 1]))
                else:
                    self.turn_dir = 1 if self.avg_left > self.avg_right else -1
            else:
                self.turn_dir = int(self.rng.choice([-1, 1]))
            self.turn_ticks = int(self.rng.uniform(1.5, 2.5) * self.rate_hz)  # 1.5-2.5s (~90-150°)
            self.turn_cooldown = int(1.0 * self.rate_hz)  # 1s cooldown
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.last_cmd_forward = False
            self.pub.publish(twist)
            return
        
        # Front-diagonal obstacles - steer away (skip if in cooldown)
        if self.turn_cooldown == 0 and front_left_obstacle and not front_right_obstacle:
            self.turn_dir = -1
            self.turn_ticks = int(1.5 * self.rate_hz)  # 1.5s (~90°)
            self.turn_cooldown = int(1.0 * self.rate_hz)
            twist.linear.x = 0.0
            twist.angular.z = -self.w_turn
            self.last_cmd_forward = False
            self.pub.publish(twist)
            return
        elif self.turn_cooldown == 0 and front_right_obstacle and not front_left_obstacle:
            self.turn_dir = 1
            self.turn_ticks = int(1.5 * self.rate_hz)  # 1.5s (~90°)
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
        
        # Random exploration turns - less frequent, rely more on straight driving
        self.explore_timer += 1
        if self.explore_timer >= self.explore_interval:
            self.explore_timer = 0
            if self.rng.random() < 0.20:  # 20% chance (was 25%)
                self.turn_dir = int(self.rng.choice([-1, 1]))
                self.turn_ticks = int(self.rng.uniform(0.8, 1.5) * self.rate_hz)  # longer turns
                twist.linear.x = 0.0
                twist.angular.z = float(self.turn_dir) * self.w_turn
                self.last_cmd_forward = False
                self.pub.publish(twist)
                return

        # Normal forward motion - STRAIGHT, minimal wander
        twist.linear.x = self.v_forward
        # Only add steering correction, NO sine wave wandering
        twist.angular.z = steering
        self.last_cmd_forward = True
        self.pub.publish(twist)


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
