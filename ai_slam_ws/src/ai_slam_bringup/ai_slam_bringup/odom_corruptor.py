import math
import os
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return (0.0, 0.0, qz, qw)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class OdomCorruptor(Node):
    def __init__(self):
        super().__init__("odom_corruptor")
        self.declare_parameter("seed", 123)
        self.declare_parameter("rw_sigma_xy", 0.02)
        self.declare_parameter("rw_sigma_theta", 0.01)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("in_topic", "/odom_raw")
        self.declare_parameter("out_topic", "/odom")

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)

        self.rw_sigma_xy = float(self.get_parameter("rw_sigma_xy").value)
        self.rw_sigma_theta = float(self.get_parameter("rw_sigma_theta").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)

        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0
        self.last_stamp = None

        self.pub = self.create_publisher(Odometry, str(self.get_parameter("out_topic").value), 10)
        self.sub = self.create_subscription(Odometry, str(self.get_parameter("in_topic").value), self.on_odom, 50)
        self.tf_br = TransformBroadcaster(self)

    def on_odom(self, msg: Odometry):
        stamp = msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.last_stamp is None:
            self.last_stamp = t

        dt = max(0.0, t - self.last_stamp)
        self.last_stamp = t

        if dt > 0.0:
            s = math.sqrt(dt)
            self.dx += float(self.rng.normal(0.0, self.rw_sigma_xy * s))
            self.dy += float(self.rng.normal(0.0, self.rw_sigma_xy * s))
            self.dtheta += float(self.rng.normal(0.0, self.rw_sigma_theta * s))
            self.dtheta = wrap(self.dtheta)

        x_raw = msg.pose.pose.position.x
        y_raw = msg.pose.pose.position.y
        yaw_raw = yaw_from_quat(msg.pose.pose.orientation)

        x = float(x_raw + self.dx)
        y = float(y_raw + self.dy)
        yaw = wrap(float(yaw_raw + self.dtheta))

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = msg.pose.pose.position.z
        qx, qy, qz, qw = quat_from_yaw(yaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist = msg.twist
        self.pub.publish(out)

        tfm = TransformStamped()
        tfm.header.stamp = msg.header.stamp
        tfm.header.frame_id = self.frame_id
        tfm.child_frame_id = self.child_frame_id
        tfm.transform.translation.x = x
        tfm.transform.translation.y = y
        tfm.transform.translation.z = msg.pose.pose.position.z
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw
        self.tf_br.sendTransform(tfm)


def main():
    rclpy.init()
    node = OdomCorruptor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
