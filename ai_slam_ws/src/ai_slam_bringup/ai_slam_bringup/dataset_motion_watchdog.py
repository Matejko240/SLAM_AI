from __future__ import annotations

from collections import deque
import math
from typing import Deque, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

PoseTXY = Tuple[float, float, float]


def _recent_motion_metrics(
    samples: Sequence[PoseTXY],
    *,
    now_sec: float,
    window_sec: float,
) -> tuple[float, float, float, float, int]:
    """
    Returns metrics over recent samples in [now-window, now]:
    - duration covered by recent samples [s]
    - net displacement between first and last sample [m]
    - spatial span (diag of bbox) [m]
    - total path length inside recent samples [m]
    - number of recent samples
    """
    if window_sec <= 0.0 or not math.isfinite(now_sec):
        return 0.0, 0.0, 0.0, 0.0, 0

    cutoff = now_sec - window_sec
    recent = [s for s in samples if s[0] >= cutoff]
    if len(recent) < 2:
        return 0.0, 0.0, 0.0, 0.0, len(recent)

    first = recent[0]
    last = recent[-1]
    duration = max(0.0, float(last[0] - first[0]))
    net = math.hypot(last[1] - first[1], last[2] - first[2])
    xs = [p[1] for p in recent]
    ys = [p[2] for p in recent]
    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    path_len = 0.0
    for i in range(1, len(recent)):
        path_len += math.hypot(recent[i][1] - recent[i - 1][1], recent[i][2] - recent[i - 1][2])
    return duration, net, span, path_len, len(recent)


def _is_low_progress_stall(
    samples: Sequence[PoseTXY],
    *,
    now_sec: float,
    window_sec: float,
    min_progress_m: float,
    span_ratio: float,
) -> tuple[bool, float, float, float, int]:
    duration, net, span, _path_len, count = _recent_motion_metrics(samples, now_sec=now_sec, window_sec=window_sec)
    has_full_window = duration >= max(1.0, window_sec - 1.0)
    low_progress = net < min_progress_m and span < (min_progress_m * span_ratio)
    return bool(has_full_window and low_progress), duration, net, span, count


def _is_circling_stall(
    samples: Sequence[PoseTXY],
    *,
    now_sec: float,
    window_sec: float,
    min_progress_m: float,
    min_path_m: float,
    max_net_path_ratio: float,
    max_net_m: float,
    max_span_m: float,
) -> tuple[bool, float, float, float, float, float, int]:
    duration, net, span, path_len, count = _recent_motion_metrics(samples, now_sec=now_sec, window_sec=window_sec)
    if duration < max(1.0, window_sec - 1.0):
        return False, duration, net, span, path_len, float("inf"), count
    if path_len < max(0.1, min_path_m):
        return False, duration, net, span, path_len, float("inf"), count
    if span > max(0.2, max_span_m):
        return False, duration, net, span, path_len, float("inf"), count
    if net > max(0.05, max_net_m):
        return False, duration, net, span, path_len, float("inf"), count
    ratio = net / max(path_len, 1e-9)
    # Circling/looping in-place: long traveled path but weak global displacement.
    # We do not require net < min_progress_m (too strict), because pathological loops
    # often drift by 0.2-1.0m while still staying in one room.
    circling = ratio <= max_net_path_ratio
    return bool(circling), duration, net, span, path_len, ratio, count


class DatasetMotionWatchdog(Node):
    """
    Failsafe for dataset collection:
    if robot position does not change for too long, exit node with abort flag.
    """

    def __init__(self) -> None:
        super().__init__("dataset_motion_watchdog")

        self.declare_parameter("pose_topic", "/ground_truth_pose")
        self.declare_parameter("min_motion_delta_m", 0.035)
        self.declare_parameter("stall_timeout_sec", 35.0)
        self.declare_parameter("startup_grace_sec", 18.0)
        self.declare_parameter("no_pose_timeout_sec", 20.0)
        self.declare_parameter("check_hz", 4.0)
        self.declare_parameter("enable_window_progress_guard", True)
        self.declare_parameter("stall_min_window_progress_m", 0.12)
        self.declare_parameter("stall_window_span_ratio", 1.8)
        self.declare_parameter("enable_circling_guard", True)
        self.declare_parameter("stall_circling_min_window_path_m", 1.6)
        self.declare_parameter("stall_circling_max_net_path_ratio", 0.25)
        self.declare_parameter("stall_circling_max_net_m", 1.2)
        self.declare_parameter("stall_circling_max_span_m", 2.5)
        self.declare_parameter("log_alive_heartbeat", False)

        self._pose_topic = str(self.get_parameter("pose_topic").value).strip() or "/ground_truth_pose"
        self._min_motion_delta_m = max(1e-4, float(self.get_parameter("min_motion_delta_m").value))
        self._stall_timeout_sec = max(1.0, float(self.get_parameter("stall_timeout_sec").value))
        self._startup_grace_sec = max(0.0, float(self.get_parameter("startup_grace_sec").value))
        self._no_pose_timeout_sec = max(0.0, float(self.get_parameter("no_pose_timeout_sec").value))
        self._check_hz = max(1.0, float(self.get_parameter("check_hz").value))
        self._enable_window_progress_guard = bool(self.get_parameter("enable_window_progress_guard").value)
        self._stall_min_window_progress_m = max(
            0.02, float(self.get_parameter("stall_min_window_progress_m").value)
        )
        self._stall_window_span_ratio = max(1.0, float(self.get_parameter("stall_window_span_ratio").value))
        self._enable_circling_guard = bool(self.get_parameter("enable_circling_guard").value)
        self._stall_circling_min_window_path_m = max(
            0.1, float(self.get_parameter("stall_circling_min_window_path_m").value)
        )
        self._stall_circling_max_net_path_ratio = min(
            1.0,
            max(0.01, float(self.get_parameter("stall_circling_max_net_path_ratio").value)),
        )
        self._stall_circling_max_net_m = max(0.05, float(self.get_parameter("stall_circling_max_net_m").value))
        self._stall_circling_max_span_m = max(0.2, float(self.get_parameter("stall_circling_max_span_m").value))
        self._log_alive_heartbeat = bool(self.get_parameter("log_alive_heartbeat").value)
        self._pose_history_horizon_sec = max(
            self._stall_timeout_sec + self._startup_grace_sec + 5.0,
            self._stall_timeout_sec + 5.0,
        )

        self._start_time_sec: Optional[float] = None
        self._last_motion_time_sec: Optional[float] = None
        self._anchor_x: Optional[float] = None
        self._anchor_y: Optional[float] = None
        self._pose_history: Deque[PoseTXY] = deque()
        self._pose_count = 0
        self._abort_reason = ""
        self._last_info_log_sec = -1e9

        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, 30)
        self.create_timer(1.0 / self._check_hz, self._on_tick)

        self.get_logger().info(
            "[motion_watchdog] enabled: "
            f"topic={self._pose_topic}, min_delta={self._min_motion_delta_m:.3f}m, "
            f"stall_timeout={self._stall_timeout_sec:.1f}s, startup_grace={self._startup_grace_sec:.1f}s, "
            f"window_guard={self._enable_window_progress_guard}, "
            f"window_min_progress={self._stall_min_window_progress_m:.3f}m/{self._stall_timeout_sec:.1f}s, "
            f"circling_guard={self._enable_circling_guard}, "
            f"circling_min_path={self._stall_circling_min_window_path_m:.2f}m, "
            f"circling_max_ratio={self._stall_circling_max_net_path_ratio:.3f}, "
            f"circling_max_net={self._stall_circling_max_net_m:.2f}m, "
            f"circling_max_span={self._stall_circling_max_span_m:.2f}m"
        )

    @property
    def should_abort(self) -> bool:
        return bool(self._abort_reason)

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_pose(self, msg: PoseStamped) -> None:
        now = self._now_sec()
        if not math.isfinite(now) or now <= 0.0:
            return
        if self._start_time_sec is None:
            self._start_time_sec = now

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return

        self._pose_count += 1
        self._pose_history.append((now, x, y))
        cutoff = now - self._pose_history_horizon_sec
        while self._pose_history and self._pose_history[0][0] < cutoff:
            self._pose_history.popleft()

        if self._anchor_x is None or self._anchor_y is None:
            self._anchor_x = x
            self._anchor_y = y
            self._last_motion_time_sec = now
            return

        dist = math.hypot(x - self._anchor_x, y - self._anchor_y)
        if dist >= self._min_motion_delta_m:
            self._anchor_x = x
            self._anchor_y = y
            self._last_motion_time_sec = now

    def _on_tick(self) -> None:
        if self.should_abort:
            return

        now = self._now_sec()
        if not math.isfinite(now) or now <= 0.0:
            return
        if self._start_time_sec is None:
            self._start_time_sec = now

        elapsed = now - self._start_time_sec
        if elapsed < self._startup_grace_sec:
            return

        if self._pose_count <= 0:
            if elapsed >= (self._startup_grace_sec + self._no_pose_timeout_sec):
                self._abort_reason = (
                    "No pose updates after startup grace "
                    f"({elapsed:.1f}s >= {self._startup_grace_sec + self._no_pose_timeout_sec:.1f}s)."
                )
            return

        if self._last_motion_time_sec is None:
            self._last_motion_time_sec = now

        stall_elapsed = now - self._last_motion_time_sec
        if stall_elapsed >= self._stall_timeout_sec:
            self._abort_reason = (
                "Robot pose stalled for "
                f"{stall_elapsed:.1f}s (>= {self._stall_timeout_sec:.1f}s), aborting dataset run."
            )
            return

        if self._enable_window_progress_guard:
            stalled_window, win_dur, win_net, win_span, win_count = _is_low_progress_stall(
                self._pose_history,
                now_sec=now,
                window_sec=self._stall_timeout_sec,
                min_progress_m=self._stall_min_window_progress_m,
                span_ratio=self._stall_window_span_ratio,
            )
            if stalled_window:
                self._abort_reason = (
                    "Robot low-progress stall detected: "
                    f"window={win_dur:.1f}s, net={win_net:.3f}m, span={win_span:.3f}m, "
                    f"samples={win_count}, threshold_net={self._stall_min_window_progress_m:.3f}m."
                )
                return

        if self._enable_circling_guard:
            circling, c_dur, c_net, c_span, c_path, c_ratio, c_count = _is_circling_stall(
                self._pose_history,
                now_sec=now,
                window_sec=self._stall_timeout_sec,
                min_progress_m=self._stall_min_window_progress_m,
                min_path_m=self._stall_circling_min_window_path_m,
                max_net_path_ratio=self._stall_circling_max_net_path_ratio,
                max_net_m=self._stall_circling_max_net_m,
                max_span_m=self._stall_circling_max_span_m,
            )
            if circling:
                self._abort_reason = (
                    "Robot circling/looping with low net progress: "
                    f"window={c_dur:.1f}s, net={c_net:.3f}m, path={c_path:.3f}m, "
                    f"net/path={c_ratio:.3f}, span={c_span:.3f}m, samples={c_count}."
                )
                return

        if self._log_alive_heartbeat and (now - self._last_info_log_sec) >= 5.0:
            self._last_info_log_sec = now
            self.get_logger().info(
                "[motion_watchdog] alive: "
                f"pose_count={self._pose_count}, time_since_motion={stall_elapsed:.1f}s"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetMotionWatchdog()
    rc = 0
    try:
        while rclpy.ok() and not node.should_abort:
            try:
                rclpy.spin_once(node, timeout_sec=0.2)
            except RuntimeError as exc:
                # W trakcie zamykania launcha potrafi pojawić się błąd konwersji
                # z warstwy C-extension; traktujemy to jako shutdown, nie crash.
                if not rclpy.ok() or "Unable to convert call argument" in str(exc):
                    break
                raise
        if node.should_abort:
            node.get_logger().error(f"[motion_watchdog] {node.abort_reason}")
            rc = 42
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
