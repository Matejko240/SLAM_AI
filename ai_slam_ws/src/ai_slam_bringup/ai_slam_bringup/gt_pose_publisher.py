import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class GTPosePublisher(Node):
    def __init__(self):
        super().__init__("gt_pose_publisher")
        self.declare_parameter("in_topic", "/odom_raw")
        self.declare_parameter("out_topic", "/ground_truth_pose")
        self.declare_parameter("frame_id", "odom")

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.pub = self.create_publisher(PoseStamped, self.out_topic, 10)
        self.sub = self.create_subscription(Odometry, self.in_topic, self.on_odom, 50)

    def on_odom(self, msg: Odometry):
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.frame_id
        ps.pose = msg.pose.pose
        self.pub.publish(ps)


def main():
    rclpy.init()
    node = GTPosePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
