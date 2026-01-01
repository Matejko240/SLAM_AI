import math
import os
import random
import numpy as np


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return (0.0, 0.0, qz, qw)


def xytheta_from_odom(odom_msg):
    x = float(odom_msg.pose.pose.position.x)
    y = float(odom_msg.pose.pose.position.y)
    th = float(yaw_from_quat(odom_msg.pose.pose.orientation))
    return x, y, th


def xytheta_from_pose_stamped(ps):
    x = float(ps.pose.position.x)
    y = float(ps.pose.position.y)
    th = float(yaw_from_quat(ps.pose.orientation))
    return x, y, th


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
