"""Odometry-only custom SLAM track node.

Reuses the same node-level pipeline shape as ``infer_robak_node`` /
``infer_rywak_node`` (scan input → integrated pose → relayed scan + TF +
PoseStamped) but the motion source is exclusively odometry. No neural model is
loaded and no ground-truth subscription is created — the integrated pose is a
pure passthrough/integration of ``/odom_raw`` (or whatever odometry topic the
launch file points at), which is exactly what slam_toolbox needs as a TF
prediction in order to perform an isolated, comparable SLAM run.

Intended use cases (controlled from ``experiment_config.yaml``):
  - When ``tor9_odom_only`` is enabled, this node feeds a dedicated
    ``slam_toolbox`` instance (``slam_toolbox_odom_only``) via a relayed scan
    topic, producing a fully independent map / TF / pose track so it can be
    compared head-to-head with the classical SLAM baseline (tor1) and the
    model-based tracks (tor5/6/7/8).
  - The node also publishes a "no-SLAM" pose topic (raw integrated odom) for
    symmetry with the model-based ``no_slam`` evaluation tracks (tor7/8).

Methodological notes for the thesis:
  - This node never reads ``/ground_truth_pose``. The class is intentionally
    free of any GT subscription, so even an accidental misconfiguration cannot
    leak GT into the integrated trajectory.
  - The integrated pose published on ``pose_topic`` and ``pose_topic_no_slam``
    is identical (modulo the optional rebase to the local origin) to the
    /odom_raw signal — i.e. exactly what an honest pure-odom dead-reckoner
    would deliver. slam_toolbox then layers its scan-matching correction on top
    via the ``map_odom_only -> odom_odom_only`` TF.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from .common import (
    quat_from_yaw,
    wrap,
    xytheta_from_odom,
)


def _rebase_pose(abs_xyth, origin_xyth):
    x, y, th = abs_xyth
    ox, oy, oth = origin_xyth
    dx_w = float(x) - float(ox)
    dy_w = float(y) - float(oy)
    c0 = math.cos(float(oth))
    s0 = math.sin(float(oth))
    x_l = c0 * dx_w + s0 * dy_w
    y_l = -s0 * dx_w + c0 * dy_w
    th_l = wrap(float(th) - float(oth))
    return float(x_l), float(y_l), float(th_l)


class OdomOnlyNode(Node):
    """Publishes a pose / TF / scan-relay track driven purely by odometry."""

    def __init__(self):
        super().__init__("odom_only_node")

        # === topics ===
        self.declare_parameter("scan_topic", "/scan_slam_odom_only")
        self.declare_parameter("relay_scan_topic", "/scan_slam_odom_only_relay")
        self.declare_parameter("odom_topic", "/odom_raw")
        self.declare_parameter("pose_topic", "/pose_odom_only")
        self.declare_parameter("pose_topic_no_slam", "/pose_odom_only_no_slam")

        # === frames ===
        self.declare_parameter("tf_parent", "odom_odom_only")
        self.declare_parameter("tf_child", "base_link_odom_only")
        self.declare_parameter("publish_tf", True)

        # === rebasing / publishing controls ===
        # If True, the first odom message defines the local origin and every
        # subsequent pose is expressed relative to it. Useful when /odom_raw
        # carries an arbitrary initial pose; off by default to mirror Robak's
        # ``odom_rebase_to_local_origin`` default (False).
        self.declare_parameter("odom_rebase_to_local_origin", False)
        # Always publish on the no_slam pose topic (mirror behaviour of
        # robak_no_slam / rywak_no_slam tracks even when the SLAM-fed track is
        # also active).
        self.declare_parameter("publish_no_slam_pose", True)
        # Heartbeat: when no scan arrives we still tick TF/pose at this rate so
        # slam_toolbox sees a fresh transform chain. 0 disables the timer.
        self.declare_parameter("publish_rate_hz", 20.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.relay_scan_topic = str(self.get_parameter("relay_scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.pose_topic_no_slam = str(self.get_parameter("pose_topic_no_slam").value)
        self.tf_parent = str(self.get_parameter("tf_parent").value)
        self.tf_child = str(self.get_parameter("tf_child").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.odom_rebase_to_local_origin = bool(
            self.get_parameter("odom_rebase_to_local_origin").value
        )
        self.publish_no_slam_pose = bool(self.get_parameter("publish_no_slam_pose").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        # state
        self.pose_inited = False
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.latest_odom = None
        self.odom_origin_xyth = None
        self.last_pub_stamp_ns = None

        # publishers / subs
        self.pub_pose = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pub_pose_no_slam = None
        if self.publish_no_slam_pose and self.pose_topic_no_slam:
            self.pub_pose_no_slam = self.create_publisher(
                PoseStamped, self.pose_topic_no_slam, 10
            )
        self.pub_scan_relay = None
        if self.relay_scan_topic:
            self.pub_scan_relay = self.create_publisher(
                LaserScan, self.relay_scan_topic, qos_profile_sensor_data
            )
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data
        )
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, 50
        )
        # NOTE: this node intentionally does NOT subscribe to /ground_truth_pose.
        # Removing the subscription completely (rather than gating it on a flag)
        # makes it impossible for the integrated trajectory to depend on GT in
        # any way, which is the methodological invariant the thesis comparison
        # relies on.

        if self.publish_rate_hz > 0.0:
            self._heartbeat_timer = self.create_timer(
                1.0 / float(self.publish_rate_hz), self._heartbeat_publish
            )
        else:
            self._heartbeat_timer = None

        self.get_logger().info(
            "[OdomOnly] motion_source=odom_only "
            f"odom_topic={self.odom_topic!r} scan_topic={self.scan_topic!r} "
            f"relay_scan_topic={self.relay_scan_topic or '(disabled)'} "
            f"pose_topic={self.pose_topic!r} pose_topic_no_slam={self.pose_topic_no_slam!r} "
            f"tf={self.tf_parent}->{self.tf_child} "
            f"rebase={self.odom_rebase_to_local_origin}"
        )

    # ----- ROS callbacks ----------------------------------------------------

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg
        x, y, th = xytheta_from_odom(msg)
        if self.odom_rebase_to_local_origin:
            if self.odom_origin_xyth is None:
                self.odom_origin_xyth = (float(x), float(y), float(th))
            x, y, th = _rebase_pose((x, y, th), self.odom_origin_xyth)

        # In odom-only mode the integrated pose IS the (possibly rebased) odom
        # pose. There is no internal state to maintain beyond pose_inited.
        self.x, self.y, self.th = x, y, th
        self.pose_inited = True
        self._publish_pose_stamped(msg.header.stamp)

    def on_scan(self, msg: LaserScan):
        # The scan path mirrors infer_robak_node: relay scans 1:1 to slam_toolbox.
        # No model inference here; pose is integrated purely from odometry.
        if self.pub_scan_relay is not None:
            self.pub_scan_relay.publish(msg)
        # Republish the current pose stamped at the scan time so slam_toolbox
        # sees a synchronized TF chain even when odom messages arrive at a
        # lower rate than scans.
        if self.pose_inited:
            self._publish_pose_stamped(msg.header.stamp)

    def _heartbeat_publish(self):
        if not self.pose_inited:
            return
        # Use the latest odom stamp when available so /tf timestamps align with
        # other simulation traffic; fall back to the current ROS time otherwise.
        if self.latest_odom is not None:
            stamp = self.latest_odom.header.stamp
        else:
            stamp = self.get_clock().now().to_msg()
        self._publish_pose_stamped(stamp)

    # ----- helpers ----------------------------------------------------------

    def _make_monotonic_stamp(self, stamp):
        stamp_ns = int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))
        if self.last_pub_stamp_ns is not None and stamp_ns <= self.last_pub_stamp_ns:
            stamp_ns = self.last_pub_stamp_ns + 1
        self.last_pub_stamp_ns = stamp_ns
        out = type(stamp)()
        out.sec = int(stamp_ns // 1_000_000_000)
        out.nanosec = int(stamp_ns % 1_000_000_000)
        return out

    def _publish_pose_stamped(self, stamp) -> None:
        stamp = self._make_monotonic_stamp(stamp)
        qx, qy, qz, qw = quat_from_yaw(self.th)
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.tf_parent
        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        self.pub_pose.publish(ps)
        if self.pub_pose_no_slam is not None:
            self.pub_pose_no_slam.publish(ps)

        if self.publish_tf and self.tf_br is not None:
            tfm = TransformStamped()
            tfm.header.stamp = stamp
            tfm.header.frame_id = self.tf_parent
            tfm.child_frame_id = self.tf_child
            tfm.transform.translation.x = float(self.x)
            tfm.transform.translation.y = float(self.y)
            tfm.transform.translation.z = 0.0
            tfm.transform.rotation.x = qx
            tfm.transform.rotation.y = qy
            tfm.transform.rotation.z = qz
            tfm.transform.rotation.w = qw
            self.tf_br.sendTransform(tfm)


def main():
    rclpy.init()
    node = OdomOnlyNode()
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
