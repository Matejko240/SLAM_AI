import json
import math
import os
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan
# Import loggera z pakietu ai_slam_ai
try:
    from ai_slam_ai.experiment_logger import ExperimentLogger
except ImportError:
    # Fallback - próbuj znaleźć ścieżkę
    ExperimentLogger = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def xytheta_from_pose(ps: PoseStamped):
    x = float(ps.pose.position.x)
    y = float(ps.pose.position.y)
    th = float(yaw_from_quat(ps.pose.orientation))
    return x, y, th


def xytheta_from_odom(od: Odometry):
    x = float(od.pose.pose.position.x)
    y = float(od.pose.pose.position.y)
    th = float(yaw_from_quat(od.pose.pose.orientation))
    return x, y, th


def load_yaml_simple(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def _read_token(f):
    """Czyta kolejny token z pliku PGM, pomija whitespace i komentarze #..."""
    token = b""
    while True:
        c = f.read(1)
        if not c:
            return None
        if c.isspace():
            continue
        if c == b"#":
            f.readline()
            continue
        token = c
        break

    while True:
        c = f.read(1)
        if not c or c.isspace():
            break
        token += c
    return token

def load_pgm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = _read_token(f)
        if magic not in (b"P2", b"P5"):
            raise ValueError(f"Unsupported PGM format: {magic!r}")

        w = int(_read_token(f))
        h = int(_read_token(f))
        maxval = int(_read_token(f))

        if magic == b"P2":
            vals = []
            while True:
                t = _read_token(f)
                if t is None:
                    break
                vals.append(int(t))
            arr = np.array(vals, dtype=np.uint16 if maxval > 255 else np.uint8)

            # mniej = błąd, więcej = utnij
            if arr.size < w * h:
                raise ValueError("PGM size mismatch")
            arr = arr[: w * h]

        else:  # P5
            if maxval < 256:
                raw = f.read(w * h)
                if len(raw) < w * h:
                    raise ValueError("PGM size mismatch")
                arr = np.frombuffer(raw[: w * h], dtype=np.uint8)
            else:
                raw = f.read(w * h * 2)
                if len(raw) < w * h * 2:
                    raise ValueError("PGM size mismatch")
                arr = np.frombuffer(raw[: w * h * 2], dtype=">u2")

        return arr.reshape((h, w))



def occgrid_to_array(msg: OccupancyGrid) -> np.ndarray:
    w = msg.info.width
    h = msg.info.height
    data = np.array(msg.data, dtype=np.int16).reshape((h, w))
    return data


def map_iou(ref_occ: np.ndarray, ref_info: dict, slam_msg: OccupancyGrid) -> float:
    slam = occgrid_to_array(slam_msg)
    slam_res = float(slam_msg.info.resolution)
    slam_ox = float(slam_msg.info.origin.position.x)
    slam_oy = float(slam_msg.info.origin.position.y)

    ref_res = float(ref_info["resolution"])
    origin = ref_info["origin"]
    ref_ox = float(origin[0])
    ref_oy = float(origin[1])

    ref_h, ref_w = ref_occ.shape
    occ_ref = ref_occ

    union = 0
    inter = 0

    for i in range(ref_h):
        for j in range(ref_w):
            x = ref_ox + (j + 0.5) * ref_res
            y = ref_oy + (i + 0.5) * ref_res
            sj = int(math.floor((x - slam_ox) / slam_res))
            si = int(math.floor((y - slam_oy) / slam_res))
            if si < 0 or sj < 0 or si >= slam.shape[0] or sj >= slam.shape[1]:
                continue
            v = int(slam[si, sj])
            if v == -1:
                continue
            occ_s = v >= 50
            occ_r = bool(occ_ref[i, j])
            u = occ_r or occ_s
            if u:
                union += 1
                if occ_r and occ_s:
                    inter += 1
    if union == 0:
        return 1.0
    return float(inter) / float(union)


class EvalNode(Node):
    def __init__(self):
        super().__init__("eval_node")
        self.declare_parameter("seed", 123)
        self.declare_parameter("mode", "baseline")
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("duration_sec", 120.0)
        self.declare_parameter("reference_map_yaml", "")
        # --- tematy (żeby launch mógł przekazać inne nazwy bez edycji kodu)
        self.declare_parameter("gt_topic", "/ground_truth_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_topic_ai", "/pose_ai")
        self.declare_parameter("pose_topic_scanmatch", "/pose_scanmatch")
        self.declare_parameter("pose_topic_bruteforce", "/pose_bruteforce")
        self.declare_parameter("pose_topic_robak", "/pose_robak")
        self.declare_parameter("pose_topic_rywak", "/pose_rywak")
        self.declare_parameter("scan_topic_points", "/scan_slam")
        self.declare_parameter("points_max_range", 8.0)
        self.declare_parameter("points_beam_step", 6)
        # --- nazwy artefaktów (żeby results.json wskazywał faktyczne pliki)
        self.declare_parameter("robak_dataset_name", "dataset_robak.npz")
        self.declare_parameter("robak_model_name", "model_robak.pt")
        self.declare_parameter("robak_history_name", "train_history_robak.json")
        self.declare_parameter("rywak_dataset_name", "dataset_rywak.npz")
        self.declare_parameter("rywak_model_name", "model_rywak.pt")
        self.declare_parameter("rywak_history_name", "train_history_rywak.json")
        self.seed = int(self.get_parameter("seed").value)
        self.mode = str(self.get_parameter("mode").value)
        base_out_dir = str(self.get_parameter("out_dir").value)
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.ref_yaml = str(self.get_parameter("reference_map_yaml").value)

        # Inicjalizacja loggera eksperymentu (używa istniejącego podfolderu)
        self.exp_logger = None
        if ExperimentLogger is not None:
            self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
            self.out_dir = self.exp_logger.get_output_dir()
        else:
            self.out_dir = os.path.abspath(base_out_dir)
        
        os.makedirs(self.out_dir, exist_ok=True)
        self.get_logger().info(f"Output directory: {self.out_dir}")
        if self.exp_logger:
            self.get_logger().info(f"Experiment ID: {self.exp_logger.experiment_id}")

        self.gt = None
        self.odom = None
        self.pose_ai = None
        self.pose_sm = None
        self.pose_bf = None
        self.pose_robak = None
        self.pose_rywak = None

        self.sm_xy = []
        self.bf_xy = []
        self.robak_xy = []
        self.rywak_xy = []

        self.err_xy_sm = []
        self.err_th_sm = []
        self.err_xy_bf = []
        self.err_th_bf = []
        self.err_xy_robak = []
        self.err_th_robak = []
        self.err_xy_rywak = []
        self.err_th_rywak = []

        self.map_baseline = None
        self.map_ai = None
        self.map_robak = None
        self.map_rywak = None
        self.t0 = self.get_clock().now()

        self.ts = []
        self.gt_xy = []
        self.odom_xy = []
        self.ai_xy = []

        self.err_xy = []
        self.err_th = []
        self.err_xy_ai = []
        self.err_th_ai = []
        
        # Śledzenie momentu startu inferencji AI
        self.ai_start_time = None  # Czas pierwszego otrzymania /pose_ai
        self.ai_start_idx = None   # Indeks w ts gdy AI wystartowało

        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pose_topic_ai = str(self.get_parameter("pose_topic_ai").value)
        self.pose_topic_scanmatch = str(self.get_parameter("pose_topic_scanmatch").value)
        self.pose_topic_bruteforce = str(self.get_parameter("pose_topic_bruteforce").value)
        self.pose_topic_robak = str(self.get_parameter("pose_topic_robak").value)
        self.pose_topic_rywak = str(self.get_parameter("pose_topic_rywak").value)
        self.scan_topic_points = str(self.get_parameter("scan_topic_points").value)
        self.points_max_range = float(self.get_parameter("points_max_range").value)
        self.points_beam_step = int(self.get_parameter("points_beam_step").value)
        self.create_subscription(PoseStamped, self.gt_topic, self.on_gt, 50)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.create_subscription(PoseStamped, self.pose_topic_ai, self.on_ai, 50)
        self.create_subscription(PoseStamped, self.pose_topic_scanmatch, self.on_sm, 50)
        self.create_subscription(PoseStamped, self.pose_topic_bruteforce, self.on_bf, 50)
        self.create_subscription(PoseStamped, self.pose_topic_robak, self.on_robak, 50)
        self.create_subscription(PoseStamped, self.pose_topic_rywak, self.on_rywak, 50)
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.create_subscription(LaserScan, self.scan_topic_points, self.on_scan_points, scan_qos)
        # QoS for map topics - slam_toolbox uses RELIABLE + TRANSIENT_LOCAL
        # Create both TRANSIENT_LOCAL and VOLATILE subscriptions for maximum compatibility
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Primary subscription with TRANSIENT_LOCAL (should get late-joining messages)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
        self.create_subscription(OccupancyGrid, "/map_ai", self.on_map_ai, map_qos)
        
        # Also create VOLATILE subscriptions as fallback (simpler QoS, always compatible)
        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(OccupancyGrid, "/map", self.on_map, volatile_qos)
        self.create_subscription(OccupancyGrid, "/map_ai", self.on_map_ai, volatile_qos)
        self.create_subscription(OccupancyGrid, "/map_robak", self.on_map_robak, map_qos)
        self.create_subscription(OccupancyGrid, "/map_rywak", self.on_map_rywak, map_qos)

        self.create_subscription(OccupancyGrid, "/map_robak", self.on_map_robak, volatile_qos)
        self.create_subscription(OccupancyGrid, "/map_rywak", self.on_map_rywak, volatile_qos)
        # Create service clients for requesting maps directly from slam_toolbox
        # This is more reliable than topic subscriptions when slam_toolbox doesn't have active subscribers
        # Node names: slam_toolbox_baseline and slam_toolbox_ai (from demo.launch.py)
        self.map_service_client = self.create_client(GetMap, '/slam_toolbox_baseline/dynamic_map')
        self.map_ai_service_client = self.create_client(GetMap, '/slam_toolbox_ai/dynamic_map')
        self.map_robak_service_client = self.create_client(GetMap, '/slam_toolbox_robak/dynamic_map')
        self.map_rywak_service_client = self.create_client(GetMap, '/slam_toolbox_rywak/dynamic_map')
        self.timer = self.create_timer(0.1, self.tick)

        self.ref_info = None
        self.ref_occ = None
        self.points_map_robak = None
        self.points_map_rywak = None

        if self.ref_occ is not None:
            h, w = self.ref_occ.shape
            self.points_map_robak = np.zeros((h, w), dtype=np.uint8)
            self.points_map_rywak = np.zeros((h, w), dtype=np.uint8)
        if self.ref_yaml:
            self.ref_info = self._load_ref_info(self.ref_yaml)
            self.ref_occ = self._load_ref_occ(self.ref_yaml, self.ref_info)
        
        # Logowanie startu ewaluacji
        if self.exp_logger is not None:
            self.exp_logger.start_evaluation(
                seed=self.seed,
                mode=self.mode,
                duration_sec=self.duration_sec,
                reference_map_yaml=self.ref_yaml
            )

    def _load_ref_info(self, yaml_path):
        y = load_yaml_simple(yaml_path)
        origin = y.get("origin", "[-3.0, -3.0, 0.0]").strip()
        origin = origin.strip("[]")
        origin_vals = [float(v.strip()) for v in origin.split(",")]
        res = float(y.get("resolution", "0.1"))
        img = y.get("image", "reference_map.pgm").strip()
        return {"resolution": res, "origin": origin_vals, "image": img}

    def _load_ref_occ(self, yaml_path, info):
        base = os.path.dirname(yaml_path)
        img_path = os.path.join(base, info["image"])
        pgm = load_pgm(img_path)
        occ = (pgm < 128).astype(np.bool_)
        return occ

    def on_gt(self, msg: PoseStamped):
        self.gt = msg

    def on_odom(self, msg: Odometry):
        self.odom = msg

    def on_ai(self, msg: PoseStamped):
        self.pose_ai = msg
        # Zapisz moment pierwszego otrzymania danych z AI
        if self.ai_start_time is None:
            t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
            self.ai_start_time = t
            self.ai_start_idx = len(self.ts)
            self.get_logger().info(f"AI inference started at t={t:.1f}s (idx={self.ai_start_idx})")
    def on_sm(self, msg: PoseStamped):
        self.pose_sm = msg

    def on_bf(self, msg: PoseStamped):
        self.pose_bf = msg
    def on_robak(self, msg: PoseStamped):
        self.pose_robak = msg

    def on_rywak(self, msg: PoseStamped):
        self.pose_rywak = msg
    def on_map(self, msg: OccupancyGrid):
        self.map_baseline = msg
        if self.map_baseline is not None and not hasattr(self, '_map_logged'):
            self._map_logged = True
            self.get_logger().info(f"Received /map: {msg.info.width}x{msg.info.height}, res={msg.info.resolution}")
    def on_map_robak(self, msg: OccupancyGrid):
        self.map_robak = msg

    def on_map_rywak(self, msg: OccupancyGrid):
        self.map_rywak = msg
    def on_map_ai(self, msg: OccupancyGrid):
        self.map_ai = msg
        if self.map_ai is not None and not hasattr(self, '_map_ai_logged'):
            self._map_ai_logged = True
            self.get_logger().info(f"Received /map_ai: {msg.info.width}x{msg.info.height}, res={msg.info.resolution}")
    def _stamp_points_to_ref_grid(self, out_grid: np.ndarray, pose_msg: PoseStamped, scan_msg: LaserScan):
        if out_grid is None or self.ref_info is None:
            return

        x, y, th = xytheta_from_pose(pose_msg)

        res = float(self.ref_info["resolution"])
        ox = float(self.ref_info["origin"][0])
        oy = float(self.ref_info["origin"][1])
        h, w = out_grid.shape

        step = max(1, int(self.points_beam_step))
        rmax = float(self.points_max_range)

        a0 = float(scan_msg.angle_min)
        da = float(scan_msg.angle_increment)

        for k in range(0, len(scan_msg.ranges), step):
            r = float(scan_msg.ranges[k])
            if not math.isfinite(r):
                continue
            if r < float(scan_msg.range_min) or r > float(scan_msg.range_max):
                continue
            if r > rmax:
                continue

            ang = th + (a0 + k * da)
            px = x + r * math.cos(ang)
            py = y + r * math.sin(ang)

            j = int((px - ox) / res)
            i = int((py - oy) / res)
            if 0 <= i < h and 0 <= j < w:
                out_grid[i, j] = 1


    def on_scan_points(self, msg: LaserScan):
        # robak / rywak mapy punktowe
        if self.ref_occ is None:
            return

        if self.pose_robak is not None and self.points_map_robak is not None:
            self._stamp_points_to_ref_grid(self.points_map_robak, self.pose_robak, msg)

        if self.pose_rywak is not None and self.points_map_rywak is not None:
            self._stamp_points_to_ref_grid(self.points_map_rywak, self.pose_rywak, msg)
    def tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if self.gt is None or self.odom is None:
            if t >= self.duration_sec:
                self.finish()
            return

        gx, gy, gth = xytheta_from_pose(self.gt)
        ox, oy, oth = xytheta_from_odom(self.odom)

        self.ts.append(float(t))
        self.gt_xy.append([gx, gy, gth])
        self.odom_xy.append([ox, oy, oth])

        ex = gx - ox
        ey = gy - oy
        eth = wrap(gth - oth)
        self.err_xy.append([ex, ey])
        self.err_th.append(eth)

        if self.pose_ai is not None:
            ax, ay, ath = xytheta_from_pose(self.pose_ai)
            self.ai_xy.append([ax, ay, ath])
            exa = gx - ax
            eya = gy - ay
            etha = wrap(gth - ath)
            self.err_xy_ai.append([exa, eya])
            self.err_th_ai.append(etha)
            
        if self.pose_sm is not None:
            sx, sy, sth = xytheta_from_pose(self.pose_sm)
            self.sm_xy.append([sx, sy, sth])
            exs = gx - sx
            eys = gy - sy
            eths = wrap(gth - sth)
            self.err_xy_sm.append([exs, eys])
            self.err_th_sm.append(eths)

        if self.pose_bf is not None:
            bx, by, bth = xytheta_from_pose(self.pose_bf)
            self.bf_xy.append([bx, by, bth])
            exb = gx - bx
            eyb = gy - by
            ethb = wrap(gth - bth)
            self.err_xy_bf.append([exb, eyb])
            self.err_th_bf.append(ethb)
        if self.pose_robak is not None:
            rx, ry, rth = xytheta_from_pose(self.pose_robak)
            self.robak_xy.append([rx, ry, rth])
            exr = gx - rx
            eyr = gy - ry
            ethr = wrap(gth - rth)
            self.err_xy_robak.append([exr, eyr])
            self.err_th_robak.append(ethr)

        if self.pose_rywak is not None:
            rx, ry, rth = xytheta_from_pose(self.pose_rywak)
            self.rywak_xy.append([rx, ry, rth])
            exr = gx - rx
            eyr = gy - ry
            ethr = wrap(gth - rth)
            self.err_xy_rywak.append([exr, eyr])
            self.err_th_rywak.append(ethr)
        if t >= self.duration_sec:
            # Poczekaj na mapy przed zakończeniem (max 10s dodatkowego czasu)
            if not hasattr(self, '_map_wait_deadline'):
                self._map_wait_deadline = t + 10.0
                self._last_service_request_time = 0
                self.get_logger().info(f"Evaluation duration reached at t={t:.1f}s")
                
            # Try to request maps via service every 2 seconds during wait period
            if (self.map_baseline is None or self.map_ai is None) and t - self._last_service_request_time >= 2.0:
                self._last_service_request_time = t
                self._request_maps_via_service()
                
            if self.map_baseline is None and t < self._map_wait_deadline:
                if not hasattr(self, '_waiting_for_maps'):
                    self._waiting_for_maps = True
                    self.get_logger().info("Waiting for /map from slam_toolbox (max 10s)...")
                return  # Keep waiting
            self.finish()
    
    def _request_maps_via_service(self):
        """Request maps via service call - more reliable than topic subscription."""
        # Request baseline map - use service_is_ready() which is non-blocking
        if self.map_baseline is None:
            if self.map_service_client.service_is_ready():
                self.get_logger().info("Requesting /map via /slam_toolbox_baseline/dynamic_map service...")
                request = GetMap.Request()
                future = self.map_service_client.call_async(request)
                future.add_done_callback(self._on_map_service_response)
            else:
                self.get_logger().warn("/slam_toolbox_baseline/dynamic_map service not ready (lifecycle node may not be active)")
        
        # Request AI map
        if self.map_ai is None:
            if self.map_ai_service_client.service_is_ready():
                self.get_logger().info("Requesting /map_ai via /slam_toolbox_ai/dynamic_map service...")
                request = GetMap.Request()
                future = self.map_ai_service_client.call_async(request)
                future.add_done_callback(self._on_map_ai_service_response)
            else:
                self.get_logger().warn("/slam_toolbox_ai/dynamic_map service not ready (lifecycle node may not be active)")
    
    def _on_map_service_response(self, future):
        """Handle response from /slam_toolbox/dynamic_map service."""
        try:
            response = future.result()
            if response.map.info.width > 0 and response.map.info.height > 0:
                self.map_baseline = response.map
                self.get_logger().info(f"Received /map via service: {response.map.info.width}x{response.map.info.height}")
            else:
                self.get_logger().warn("Received empty map from service")
        except Exception as e:
            self.get_logger().error(f"Failed to get map via service: {e}")
    
    def _on_map_ai_service_response(self, future):
        """Handle response from /slam_toolbox_ai/dynamic_map service."""
        try:
            response = future.result()
            if response.map.info.width > 0 and response.map.info.height > 0:
                self.map_ai = response.map
                self.get_logger().info(f"Received /map_ai via service: {response.map.info.width}x{response.map.info.height}")
            else:
                self.get_logger().warn("Received empty AI map from service")
        except Exception as e:
            self.get_logger().error(f"Failed to get AI map via service: {e}")

    def finish(self):
        if len(self.ts) == 0:
            rclpy.shutdown()
            return

        gt = np.asarray(self.gt_xy, dtype=np.float32)
        od = np.asarray(self.odom_xy, dtype=np.float32)
        err = np.asarray(self.err_xy, dtype=np.float32)
        err_th = np.asarray(self.err_th, dtype=np.float32)

        rmse_xy = float(np.sqrt(np.mean(err[:, 0] ** 2 + err[:, 1] ** 2)))
        rmse_th = float(np.sqrt(np.mean(err_th ** 2)))

        rmse_xy_ai = None
        rmse_th_ai = None
        if len(self.err_xy_ai) > 0:
            err_ai = np.asarray(self.err_xy_ai, dtype=np.float32)
            err_th_ai = np.asarray(self.err_th_ai, dtype=np.float32)
            rmse_xy_ai = float(np.sqrt(np.mean(err_ai[:, 0] ** 2 + err_ai[:, 1] ** 2)))
            rmse_th_ai = float(np.sqrt(np.mean(err_th_ai ** 2)))

        rmse_xy_sm = None
        rmse_th_sm = None
        if len(self.err_xy_sm) > 0:
            err_sm = np.asarray(self.err_xy_sm, dtype=np.float32)
            err_th_sm = np.asarray(self.err_th_sm, dtype=np.float32)
            rmse_xy_sm = float(np.sqrt(np.mean(err_sm[:, 0] ** 2 + err_sm[:, 1] ** 2)))
            rmse_th_sm = float(np.sqrt(np.mean(err_th_sm ** 2)))

        rmse_xy_bf = None
        rmse_th_bf = None
        if len(self.err_xy_bf) > 0:
            err_bf = np.asarray(self.err_xy_bf, dtype=np.float32)
            err_th_bf = np.asarray(self.err_th_bf, dtype=np.float32)
            rmse_xy_bf = float(np.sqrt(np.mean(err_bf[:, 0] ** 2 + err_bf[:, 1] ** 2)))
            rmse_th_bf = float(np.sqrt(np.mean(err_th_bf ** 2)))

        rmse_xy_robak = None
        rmse_th_robak = None
        if len(self.err_xy_robak) > 0:
            err_r = np.asarray(self.err_xy_robak, dtype=np.float32)
            err_th_r = np.asarray(self.err_th_robak, dtype=np.float32)
            rmse_xy_robak = float(np.sqrt(np.mean(err_r[:, 0] ** 2 + err_r[:, 1] ** 2)))
            rmse_th_robak = float(np.sqrt(np.mean(err_th_r ** 2)))

        rmse_xy_rywak = None
        rmse_th_rywak = None
        if len(self.err_xy_rywak) > 0:
            err_r = np.asarray(self.err_xy_rywak, dtype=np.float32)
            err_th_r = np.asarray(self.err_th_rywak, dtype=np.float32)
            rmse_xy_rywak = float(np.sqrt(np.mean(err_r[:, 0] ** 2 + err_r[:, 1] ** 2)))
            rmse_th_rywak = float(np.sqrt(np.mean(err_th_r ** 2)))
        iou_map = None
        iou_map_ai = None
        # Debug: sprawdź czy mamy dane do obliczenia IOU
        self.get_logger().info(f"IOU calculation: ref_occ={self.ref_occ is not None}, map_baseline={self.map_baseline is not None}, map_ai={self.map_ai is not None}")
        
        if self.ref_occ is not None and self.map_baseline is not None:
            try:
                iou_map = map_iou(self.ref_occ, self.ref_info, self.map_baseline)
                self.get_logger().info(f"IOU baseline: {iou_map}")
            except Exception as e:
                self.get_logger().error(f"Failed to calculate IOU baseline: {e}")
        else:
            if self.ref_occ is None:
                self.get_logger().warn("No reference map loaded - IOU cannot be calculated")
            if self.map_baseline is None:
                self.get_logger().warn("No /map received - IOU baseline cannot be calculated")
                
        if self.ref_occ is not None and self.map_ai is not None:
            try:
                iou_map_ai = map_iou(self.ref_occ, self.ref_info, self.map_ai)
                self.get_logger().info(f"IOU AI: {iou_map_ai}")
            except Exception as e:
                self.get_logger().error(f"Failed to calculate IOU AI: {e}")
        else:
            if self.map_ai is None:
                self.get_logger().warn("No /map_ai received - IOU AI cannot be calculated")

        traj_path = os.path.join(self.out_dir, "trajectory.png")
        err_path = os.path.join(self.out_dir, "errors.png")
        maps_path = os.path.join(self.out_dir, "maps.png")
        results_path = os.path.join(self.out_dir, "results.json")

        self._plot_trajectories(traj_path)
        self._plot_errors(err_path)
        self._plot_maps(maps_path)

        results = {
            "mode": self.mode,
            "seed": self.seed,
            "duration_sec": self.duration_sec,
            "metrics": {
                "rmse_xy_baseline": rmse_xy,
                "rmse_theta_baseline": rmse_th,
                "rmse_xy_ai": rmse_xy_ai,
                "rmse_theta_ai": rmse_th_ai,
                "iou_map_baseline": iou_map,
                "iou_map_ai": iou_map_ai,
                "rmse_xy_scanmatch": rmse_xy_sm,
                "rmse_theta_scanmatch": rmse_th_sm,
                "rmse_xy_bruteforce": rmse_xy_bf,
                "rmse_theta_bruteforce": rmse_th_bf,
                "rmse_xy_robak": rmse_xy_robak,
                "rmse_theta_robak": rmse_th_robak,
                "rmse_xy_rywak": rmse_xy_rywak,
                "rmse_theta_rywak": rmse_th_rywak,                
            },
            "artifacts": {
                "trajectory_png": traj_path,
                "errors_png": err_path,
                "maps_png": maps_path,
                "reference_map_yaml": self.ref_yaml,
                "map_topic_baseline": "/map",
                "map_topic_ai": "/map_ai",
                "dataset_npz": os.path.join(self.out_dir, "dataset.npz"),
                "model_pt": os.path.join(self.out_dir, "model.pt"),
                "train_history_json": os.path.join(self.out_dir, "train_history.json"),
                "experiment_metadata": os.path.join(self.out_dir, "experiment_metadata.json"),
                "robak_dataset_npz": os.path.join(self.out_dir, str(self.get_parameter("robak_dataset_name").value)),
                "robak_model_pt": os.path.join(self.out_dir, str(self.get_parameter("robak_model_name").value)),
                "robak_train_history_json": os.path.join(self.out_dir, str(self.get_parameter("robak_history_name").value)),
                "rywak_dataset_npz": os.path.join(self.out_dir, str(self.get_parameter("rywak_dataset_name").value)),
                "rywak_model_pt": os.path.join(self.out_dir, str(self.get_parameter("rywak_model_name").value)),
                "rywak_train_history_json": os.path.join(self.out_dir, str(self.get_parameter("rywak_history_name").value)),
            },
        }

        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, results_path)
        
        # Logowanie zakończenia ewaluacji
        if self.exp_logger is not None:
            self.exp_logger.end_evaluation(
                rmse_xy_baseline=rmse_xy,
                rmse_theta_baseline=rmse_th,
                rmse_xy_ai=rmse_xy_ai,
                rmse_theta_ai=rmse_th_ai,
                iou_map_baseline=iou_map,
                iou_map_ai=iou_map_ai,
                n_samples=len(self.ts),
                artifacts=results["artifacts"]
            )

            
            # Poczekaj, aby infer_node zdążył zapisać swoje dane do metadata.json
            # Infer_node zazwyczaj kończy się kilka sekund po evaluation,
            # więc czekamy, aby CSV miał wszystkie dane
            self.get_logger().info("Waiting 15s for infer_node to save metadata...")
            time.sleep(15.0)
            self.exp_logger.finalize()
            # Dodaj do pliku podsumowania wszystkich eksperymentów
            # Wczytuje najnowsze dane z metadata.json, więc inference data też będą uwzględnione
            summary_path = self.exp_logger.append_to_summary()
            self.get_logger().info(f"Experiment summary saved to: {summary_path}")
            
            # Wyświetl podsumowanie eksperymentu
            self.get_logger().info("\n" + self.exp_logger.get_summary())

        rclpy.shutdown()

    def _plot_trajectories(self, path):
        gt = np.asarray(self.gt_xy, dtype=np.float32)
        od = np.asarray(self.odom_xy, dtype=np.float32)

        # --- bounds z mapy referencyjnej (zamiast -3..3)
        if self.ref_info is not None and self.ref_occ is not None:
            res = float(self.ref_info["resolution"])
            ox = float(self.ref_info["origin"][0])
            oy = float(self.ref_info["origin"][1])
            h, w = self.ref_occ.shape
            xmin, ymin = ox, oy
            xmax, ymax = ox + w * res, oy + h * res
        else:
            xmin, ymin, xmax, ymax = -3.0, -3.0, 3.0, 3.0

        margin = 0.5

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        from matplotlib.patches import Rectangle

        # Left: full view with all trajectories
        ax1 = axes[0]
        ax1.plot(gt[:, 0], gt[:, 1], color="tab:blue", label="GT", linewidth=1.5)
        ax1.plot(od[:, 0], od[:, 1], color="tab:orange", label="baseline", linewidth=1.0, alpha=0.7)

        if len(self.ai_xy) > 0:
            ai = np.asarray(self.ai_xy, dtype=np.float32)
            ax1.plot(ai[:, 0], ai[:, 1], color="tab:green", label="AI", linewidth=1.5)

        if len(self.robak_xy) > 0:
            rb = np.asarray(self.robak_xy, dtype=np.float32)
            ax1.plot(rb[:, 0], rb[:, 1], color="tab:red", label="robak", linewidth=1.0, alpha=0.8)

        if len(self.rywak_xy) > 0:
            ry = np.asarray(self.rywak_xy, dtype=np.float32)
            ax1.plot(ry[:, 0], ry[:, 1], color="tab:purple", label="rywak", linewidth=1.0, alpha=0.8)

        if len(self.sm_xy) > 0:
            sm = np.asarray(self.sm_xy, dtype=np.float32)
            ax1.plot(sm[:, 0], sm[:, 1], label="scanmatch", linewidth=1.0, alpha=0.8)

        if len(self.bf_xy) > 0:
            bf = np.asarray(self.bf_xy, dtype=np.float32)
            ax1.plot(bf[:, 0], bf[:, 1], label="bruteforce", linewidth=1.0, alpha=0.8)

        rect1 = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, edgecolor="gray", linestyle="--", linewidth=1.5, label="ref map bounds")
        ax1.add_patch(rect1)

        ax1.set_aspect("equal")
        ax1.legend(loc="best")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_title("Trajektorie (pełny widok)")
        ax1.grid(True, alpha=0.3)

        # Right: zoom to reference map bounds
        ax2 = axes[1]
        ax2.plot(gt[:, 0], gt[:, 1], color="tab:blue", label="GT", linewidth=1.5)

        if len(self.ai_xy) > 0:
            ai = np.asarray(self.ai_xy, dtype=np.float32)
            ax2.plot(ai[:, 0], ai[:, 1], color="tab:green", label="AI", linewidth=1.5)

        if len(self.robak_xy) > 0:
            rb = np.asarray(self.robak_xy, dtype=np.float32)
            ax2.plot(rb[:, 0], rb[:, 1], color="tab:red", label="robak", linewidth=1.0, alpha=0.8)

        if len(self.rywak_xy) > 0:
            ry = np.asarray(self.rywak_xy, dtype=np.float32)
            ax2.plot(ry[:, 0], ry[:, 1], color="tab:purple", label="rywak", linewidth=1.0, alpha=0.8)

        ax2.set_xlim(xmin - margin, xmax + margin)
        ax2.set_ylim(ymin - margin, ymax + margin)
        ax2.set_aspect("equal")
        ax2.legend(loc="best")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_title("GT vs metody (widok wg mapy referencyjnej)")
        ax2.grid(True, alpha=0.3)

        rect2 = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, edgecolor="gray", linestyle="--", linewidth=2)
        ax2.add_patch(rect2)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_errors(self, path):
        t = np.asarray(self.ts, dtype=np.float32)
        err = np.asarray(self.err_xy, dtype=np.float32)
        eth = np.asarray(self.err_th, dtype=np.float32)

        plt.figure(figsize=(12, 5))

        # baseline (odom)
        plt.plot(t, np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2), label="pos err baseline")
        plt.plot(t, np.abs(eth), label="|theta| baseline")

        def _plot_series(err_xy_list, err_th_list, label_prefix, alpha=0.7):
            if len(err_xy_list) == 0:
                return
            e_xy = np.asarray(err_xy_list, dtype=np.float32)
            e_th = np.asarray(err_th_list, dtype=np.float32)
            n = min(len(t), e_xy.shape[0], e_th.shape[0])
            if n <= 0:
                return
            plt.plot(t[:n], np.sqrt(e_xy[:n, 0] ** 2 + e_xy[:n, 1] ** 2), label=f"pos err {label_prefix}", alpha=alpha)
            plt.plot(t[:n], np.abs(e_th[:n]), label=f"|theta| {label_prefix}", alpha=alpha)

        _plot_series(self.err_xy_sm, self.err_th_sm, "scanmatch")
        _plot_series(self.err_xy_bf, self.err_th_bf, "bruteforce")
        _plot_series(self.err_xy_robak, self.err_th_robak, "robak")
        _plot_series(self.err_xy_rywak, self.err_th_rywak, "rywak")

        # AI: rysujemy od momentu, kiedy AI zaczęło publikować /pose_ai
        if len(self.err_xy_ai) > 0:
            e_xy = np.asarray(self.err_xy_ai, dtype=np.float32)
            e_th = np.asarray(self.err_th_ai, dtype=np.float32)
            if self.ai_start_idx is not None and self.ai_start_idx < len(t):
                t_ai = t[self.ai_start_idx:self.ai_start_idx + len(e_xy)]
            else:
                t_ai = t[-len(e_xy):] if len(e_xy) <= len(t) else t
            n = min(len(t_ai), e_xy.shape[0], e_th.shape[0])
            if n > 0:
                plt.plot(t_ai[:n], np.sqrt(e_xy[:n, 0] ** 2 + e_xy[:n, 1] ** 2), label="pos err AI")
                plt.plot(t_ai[:n], np.abs(e_th[:n]), label="|theta| AI", alpha=0.7)
            if self.ai_start_time is not None:
                plt.axvline(x=self.ai_start_time, color='gray', linestyle='--', alpha=0.7, label=f"AI start (t={self.ai_start_time:.1f}s)")

        plt.legend(loc='best')
        plt.xlabel("t [s]")
        plt.ylabel("error [m / rad]")
        plt.title("Błędy pozycji i orientacji")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_maps(self, path):
        if self.ref_occ is None:
            plt.figure()
            plt.text(0.5, 0.5, "No reference map loaded", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            return

        ref = self.ref_occ.astype(np.uint8)
        maps = [("ref", ref)]

        def _append_occ(name, msg):
            if msg is None:
                return
            m = occgrid_to_array(msg)
            occ = (m >= 50).astype(np.uint8)
            maps.append((name, occ))

        _append_occ("baseline", self.map_baseline)
        _append_occ("ai", self.map_ai)

        # --- Robak map (prefer occupancy grid from slam_toolbox, fallback to points) ---
        if getattr(self, "map_robak", None) is not None:
            m = occgrid_to_array(self.map_robak)
            mr = (m >= 50).astype(np.uint8)
            maps.append(("robak", mr))
        elif getattr(self, "points_map_robak", None) is not None:
            maps.append(("robak (points)", self.points_map_robak.astype(np.uint8)))

        # --- Rywak map (prefer occupancy grid from slam_toolbox, fallback to points) ---
        if getattr(self, "map_rywak", None) is not None:
            m = occgrid_to_array(self.map_rywak)
            my = (m >= 50).astype(np.uint8)
            maps.append(("rywak", my))
        elif getattr(self, "points_map_rywak", None) is not None:
            maps.append(("rywak (points)", self.points_map_rywak.astype(np.uint8)))

        plt.figure(figsize=(4 * len(maps), 4))
        for i, (name, arr) in enumerate(maps, start=1):
            ax = plt.subplot(1, len(maps), i)
            ax.imshow(arr, origin="lower")
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def main():
    rclpy.init()
    node = EvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if getattr(node, "exp_logger", None) is not None:
                node.exp_logger.finalize()
        except Exception as e:
            node.get_logger().error(f"Finalize failed: {e}")
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass