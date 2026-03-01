import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanFix(Node):
    def __init__(self):
        super().__init__("scan_fix")

        self.declare_parameter("in_topic", "/scan")
        self.declare_parameter("out_topic", "/scan_slam")
        self.declare_parameter("frame_id", "base_link")

        self.declare_parameter("range_min_override", -1.0)  # <0 => nie zmieniaj
        self.declare_parameter("range_max_override", -1.0)  # <0 => nie zmieniaj

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.rmin_ovr = float(self.get_parameter("range_min_override").value)
        self.rmax_ovr = float(self.get_parameter("range_max_override").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        self.pub = self.create_publisher(LaserScan, self.out_topic, qos)
        self.sub = self.create_subscription(LaserScan, self.in_topic, self.on_scan, qos)

        self.get_logger().info(f"in={self.in_topic} -> out={self.out_topic}, frame_id={self.frame_id}")

    def on_scan(self, msg: LaserScan):
        out = LaserScan()
        out.header = msg.header
        out.header.frame_id = self.frame_id

        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time

        out.range_min = msg.range_min if self.rmin_ovr < 0.0 else self.rmin_ovr
        out.range_max = msg.range_max if self.rmax_ovr < 0.0 else self.rmax_ovr

        out.ranges = msg.ranges
        out.intensities = msg.intensities

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
        if rclpy.ok():
            rclpy.shutdown()