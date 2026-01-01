import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


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

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.v_forward = float(self.get_parameter("v_forward").value)
        self.w_turn = float(self.get_parameter("w_turn").value)
        self.front_threshold = float(self.get_parameter("front_threshold").value)

        self.min_front = None
        self.turn_ticks = 0
        self.turn_dir = 1

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = ranges[np.isfinite(ranges)]
        if ranges.size == 0:
            self.min_front = None
            return
        n = len(msg.ranges)
        if n == 0:
            self.min_front = None
            return
        half_window = max(1, int(0.05 * n))
        idx0 = 0
        front = []
        for k in range(-half_window, half_window + 1):
            front.append(msg.ranges[(idx0 + k) % n])
        front = np.asarray(front, dtype=np.float32)
        front = front[np.isfinite(front)]
        if front.size == 0:
            self.min_front = None
        else:
            self.min_front = float(np.min(front))

    def on_timer(self):
        twist = Twist()
        obstacle = (self.min_front is not None) and (self.min_front < self.front_threshold)

        if self.turn_ticks > 0:
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.turn_ticks -= 1
            self.pub.publish(twist)
            return

        if obstacle:
            self.turn_dir = int(self.rng.choice([-1, 1]))
            self.turn_ticks = int(1.2 * self.rate_hz)
            twist.linear.x = 0.0
            twist.angular.z = float(self.turn_dir) * self.w_turn
            self.pub.publish(twist)
            return

        t = self.get_clock().now().nanoseconds * 1e-9
        twist.linear.x = self.v_forward
        twist.angular.z = 0.25 * self.w_turn * math.sin(0.5 * t)
        self.pub.publish(twist)


def main():
    rclpy.init()
    node = AutoDriver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
