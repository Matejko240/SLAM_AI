import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def _resample_scan(ranges: np.ndarray, target_beams: int) -> np.ndarray | None:
    n = int(ranges.size)
    if n == target_beams:
        return ranges.astype(np.float32)
    if n < 10:
        return None
    x_old = np.linspace(-math.pi, math.pi, n, endpoint=False)
    x_new = np.linspace(-math.pi, math.pi, target_beams, endpoint=False)
    return np.interp(x_new, x_old, ranges).astype(np.float32)


def _apply_orientation_fix(ranges: np.ndarray, reverse: bool, shift_deg: float) -> np.ndarray:
    out = np.asarray(ranges, dtype=np.float32)
    if out.size == 0:
        return out
    if reverse:
        out = out[::-1]
    shift_bins = int(round((float(shift_deg) / 360.0) * float(out.size)))
    if shift_bins != 0:
        out = np.roll(out, shift_bins)
    return out.astype(np.float32)


class JetRacerScanAdapter(Node):
    def __init__(self):
        super().__init__("jetracer_scan_adapter")

        self.declare_parameter("in_topic", "/scan")
        self.declare_parameter("out_topic", "/scan_jetracer_ai")
        self.declare_parameter("target_beams", 360)
        self.declare_parameter("range_min_override", -1.0)
        self.declare_parameter("range_max_override", -1.0)
        self.declare_parameter("clip_min", 0.08)
        self.declare_parameter("clip_max", 10.0)
        self.declare_parameter("frame_id_override", "")
        self.declare_parameter("scan_reverse", False)
        self.declare_parameter("scan_shift_deg", 0.0)
        self.declare_parameter("report_every_sec", 5.0)

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.target_beams = max(10, int(self.get_parameter("target_beams").value))
        self.range_min_override = float(self.get_parameter("range_min_override").value)
        self.range_max_override = float(self.get_parameter("range_max_override").value)
        self.clip_min = float(self.get_parameter("clip_min").value)
        self.clip_max = float(self.get_parameter("clip_max").value)
        self.frame_id_override = str(self.get_parameter("frame_id_override").value)
        self.scan_reverse = bool(self.get_parameter("scan_reverse").value)
        self.scan_shift_deg = float(self.get_parameter("scan_shift_deg").value)
        self.report_every_sec = max(1.0, float(self.get_parameter("report_every_sec").value))

        self.pub = self.create_publisher(LaserScan, self.out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(LaserScan, self.in_topic, self._on_scan, qos_profile_sensor_data)

        self.msg_count = 0
        self.nan_inf_count = 0
        self.last_report_t = time.time()
        self.last_msg_t = None
        self.dt_samples: list[float] = []
        self.last_len = None

        self.get_logger().info(
            f"JetRacer scan adapter: {self.in_topic} -> {self.out_topic}, "
            f"target_beams={self.target_beams}, scan_reverse={self.scan_reverse}, "
            f"scan_shift_deg={self.scan_shift_deg:.1f}"
        )

    def _on_scan(self, msg: LaserScan):
        now = time.time()
        if self.last_msg_t is not None:
            dt = now - self.last_msg_t
            if dt > 0:
                self.dt_samples.append(dt)
                if len(self.dt_samples) > 200:
                    self.dt_samples.pop(0)
        self.last_msg_t = now

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        self.last_len = int(ranges.size)
        finite_mask = np.isfinite(ranges)
        self.nan_inf_count += int(np.count_nonzero(~finite_mask))
        ranges = np.where(finite_mask, ranges, float(msg.range_max) if msg.range_max > 0 else self.clip_max)

        out = LaserScan()
        out.header = msg.header
        if self.frame_id_override:
            out.header.frame_id = self.frame_id_override

        out.range_min = (
            self.range_min_override
            if self.range_min_override >= 0.0
            else max(float(msg.range_min) if msg.range_min > 0 else 0.0, self.clip_min)
        )
        out.range_max = (
            self.range_max_override
            if self.range_max_override >= 0.0
            else min(float(msg.range_max) if msg.range_max > 0 else self.clip_max, self.clip_max)
        )

        scan = np.clip(ranges, out.range_min, out.range_max).astype(np.float32)
        scan = _apply_orientation_fix(scan, reverse=self.scan_reverse, shift_deg=self.scan_shift_deg)
        scan = _resample_scan(scan, self.target_beams)
        if scan is None:
            return
        scan = np.clip(scan, out.range_min, out.range_max).astype(np.float32)

        out.angle_min = -math.pi
        out.angle_max = math.pi
        out.angle_increment = (out.angle_max - out.angle_min) / float(self.target_beams)
        out.time_increment = 0.0
        out.scan_time = msg.scan_time
        out.ranges = scan.tolist()
        out.intensities = []
        self.pub.publish(out)
        self.msg_count += 1

        if now - self.last_report_t >= self.report_every_sec:
            hz = None
            if self.dt_samples:
                hz = 1.0 / (sum(self.dt_samples) / len(self.dt_samples))
            latency_ms = 1000.0 * (now - (float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)))
            hz_txt = f"{hz:.2f}" if hz is not None else "n/a"
            self.get_logger().info(
                f"[adapter] hz={hz_txt}, latency_ms={latency_ms:.1f}, "
                f"input_len={self.last_len}, output_len={self.target_beams}, "
                f"nan_inf_total={self.nan_inf_count}"
            )
            self.last_report_t = now


def main():
    rclpy.init()
    node = JetRacerScanAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
