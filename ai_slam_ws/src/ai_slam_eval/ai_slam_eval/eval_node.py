import json
import math
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid

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
        self.declare_parameter("duration_sec", 120)
        self.declare_parameter("reference_map_yaml", "")

        self.seed = int(self.get_parameter("seed").value)
        self.mode = str(self.get_parameter("mode").value)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.ref_yaml = str(self.get_parameter("reference_map_yaml").value)

        os.makedirs(self.out_dir, exist_ok=True)

        self.gt = None
        self.odom = None
        self.pose_ai = None

        self.map_baseline = None
        self.map_ai = None

        self.t0 = self.get_clock().now()

        self.ts = []
        self.gt_xy = []
        self.odom_xy = []
        self.ai_xy = []

        self.err_xy = []
        self.err_th = []
        self.err_xy_ai = []
        self.err_th_ai = []

        self.create_subscription(PoseStamped, "/ground_truth_pose", self.on_gt, 50)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(PoseStamped, "/pose_ai", self.on_ai, 50)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, 10)
        self.create_subscription(OccupancyGrid, "/map_ai", self.on_map_ai, 10)

        self.timer = self.create_timer(0.1, self.tick)

        self.ref_info = None
        self.ref_occ = None
        if self.ref_yaml:
            self.ref_info = self._load_ref_info(self.ref_yaml)
            self.ref_occ = self._load_ref_occ(self.ref_yaml, self.ref_info)

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

    def on_map(self, msg: OccupancyGrid):
        self.map_baseline = msg

    def on_map_ai(self, msg: OccupancyGrid):
        self.map_ai = msg

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

        if t >= self.duration_sec:
            self.finish()

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

        iou_map = None
        iou_map_ai = None
        if self.ref_occ is not None and self.map_baseline is not None:
            iou_map = map_iou(self.ref_occ, self.ref_info, self.map_baseline)
        if self.ref_occ is not None and self.map_ai is not None:
            iou_map_ai = map_iou(self.ref_occ, self.ref_info, self.map_ai)

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
            },
        }

        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, results_path)

        rclpy.shutdown()

    def _plot_trajectories(self, path):
        gt = np.asarray(self.gt_xy, dtype=np.float32)
        od = np.asarray(self.odom_xy, dtype=np.float32)
        plt.figure()
        plt.plot(gt[:, 0], gt[:, 1], label="GT")
        plt.plot(od[:, 0], od[:, 1], label="baseline")
        if len(self.ai_xy) > 0:
            ai = np.asarray(self.ai_xy, dtype=np.float32)
            plt.plot(ai[:, 0], ai[:, 1], label="AI")
        plt.axis("equal")
        plt.legend()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_errors(self, path):
        t = np.asarray(self.ts, dtype=np.float32)
        err = np.asarray(self.err_xy, dtype=np.float32)
        eth = np.asarray(self.err_th, dtype=np.float32)

        plt.figure()
        plt.plot(t, np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2), label="pos err baseline")
        plt.plot(t, np.abs(eth), label="|theta| baseline")
        if len(self.err_xy_ai) > 0:
            err_ai = np.asarray(self.err_xy_ai, dtype=np.float32)
            eth_ai = np.asarray(self.err_th_ai, dtype=np.float32)
            plt.plot(t[: err_ai.shape[0]], np.sqrt(err_ai[:, 0] ** 2 + err_ai[:, 1] ** 2), label="pos err AI")
            plt.plot(t[: eth_ai.shape[0]], np.abs(eth_ai), label="|theta| AI")
        plt.legend()
        plt.xlabel("t [s]")
        plt.ylabel("error")
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
        if self.map_baseline is not None:
            m = occgrid_to_array(self.map_baseline)
            mb = (m >= 50).astype(np.uint8)
            maps.append(("baseline", mb))
        if self.map_ai is not None:
            m = occgrid_to_array(self.map_ai)
            ma = (m >= 50).astype(np.uint8)
            maps.append(("ai", ma))

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
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
