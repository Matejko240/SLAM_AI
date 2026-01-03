import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanFix(Node):
    def __init__(self):
        super().__init__("scan_fix")
        self.declare_parameter("in_topic", "/scan")
        self.declare_parameter("out_topic", "/scan_slam")
        self.declare_parameter("frame_id", "laser_link")
        self.declare_parameter("target_samples", 360)

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.target_samples = int(self.get_parameter("target_samples").value)

        self.pub = self.create_publisher(LaserScan, self.out_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.in_topic, self.on_scan, qos_profile_sensor_data)

    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        rmax = float(msg.range_max) if msg.range_max > 0.0 else 6.0
        ranges = np.where(np.isfinite(ranges), ranges, rmax)
        ranges = np.clip(ranges, float(msg.range_min), rmax)

        if ranges.size != self.target_samples:
            x_old = np.linspace(-math.pi, math.pi, ranges.size, endpoint=False)
            x_new = np.linspace(-math.pi, math.pi, self.target_samples, endpoint=False)
            ranges = np.interp(x_new, x_old, ranges).astype(np.float32)

        out = LaserScan()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.angle_min = -math.pi
        out.angle_max = math.pi
        out.angle_increment = (2.0 * math.pi) / float(self.target_samples)
        out.time_increment = 0.0
        out.scan_time = msg.scan_time if msg.scan_time > 0.0 else 0.1
        out.range_min = float(msg.range_min) if msg.range_min > 0.0 else 0.12
        out.range_max = rmax
        out.ranges = ranges.tolist()
        out.intensities = []
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanFix()
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
