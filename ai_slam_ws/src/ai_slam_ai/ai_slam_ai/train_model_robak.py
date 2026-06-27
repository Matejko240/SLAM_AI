import json
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node

import torch
import torch.nn as nn
import torch.optim as optim

from .common import (
    collect_rollout_window_starts,
    Normalizer,
    compute_rollout_metrics_from_local_deltas,
    ensure_dir,
    seed_all,
    select_torch_device,
    split_time_coverage_stats,
    split_train_val_indices,
    torch_state_dict_to_cpu,
    wait_for_npz_dataset,
)
from .experiment_logger import ExperimentLogger


class RobakConv1D(nn.Module):
    """Conv1D scan matcher faithful to original thesis: 3xConv1D + LeakyReLU + MaxPool."""

    def __init__(self, out_dim: int = 3):
        super().__init__()
        self.out_dim = int(out_dim)
        self.feat = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 32, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 16, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, self.out_dim),
        )

    def forward(self, x):
        return self.head(self.feat(x))


def _normalize_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"", "zscore", "standard", "standardize"}:
        return "zscore"
    if mode in {"none", "identity", "raw"}:
        return "none"
    return "zscore"


def _normalize_loss_type(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"huber", "smooth_l1", "smoothl1"}:
        return "huber"
    return "mse"


def _normalize_lr_schedule(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"cosine", "cos", "cosine_annealing"}:
        return "cosine"
    return "none"


def _normalize_target_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"forward_yaw", "forward_dtheta", "forward+theta", "forward_theta"}:
        return "forward_yaw"
    return "se2_local"


def _normalize_selection_metric(value: str) -> str:
    metric = str(value or "").strip().lower()
    return metric or "val_loss"


def _forward_yaw_targets_from_pose_pairs(pose_pairs: np.ndarray) -> np.ndarray:
    prev_x = pose_pairs[:, 0].astype(np.float32)
    prev_y = pose_pairs[:, 1].astype(np.float32)
    prev_th = pose_pairs[:, 2].astype(np.float32)
    curr_x = pose_pairs[:, 3].astype(np.float32)
    curr_y = pose_pairs[:, 4].astype(np.float32)
    curr_th = pose_pairs[:, 5].astype(np.float32)

    dx_w = curr_x - prev_x
    dy_w = curr_y - prev_y
    c = np.cos(prev_th).astype(np.float32)
    s = np.sin(prev_th).astype(np.float32)
    forward = (c * dx_w + s * dy_w).astype(np.float32)
    dth = ((curr_th - prev_th + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    return np.stack([forward, dth], axis=1).astype(np.float32)


def _weighted_loss(loss_fn, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss_elem = loss_fn(pred, target)
    if loss_elem.ndim == 0:
        return loss_elem
    if loss_elem.ndim == 1:
        return loss_elem.mean()
    return (loss_elem * weights.view(1, -1)).mean()


def _wrap_torch_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _compose_local_pose_sequence_torch(
    start_pose: torch.Tensor,
    local_deltas: torch.Tensor,
) -> torch.Tensor:
    x = start_pose[:, 0]
    y = start_pose[:, 1]
    th = start_pose[:, 2]
    for step_idx in range(local_deltas.shape[1]):
        dx = local_deltas[:, step_idx, 0]
        dy = local_deltas[:, step_idx, 1]
        dth = local_deltas[:, step_idx, 2]
        c = torch.cos(th)
        s = torch.sin(th)
        x = x + c * dx - s * dy
        y = y + s * dx + c * dy
        th = _wrap_torch_angle(th + dth)
    return torch.stack([x, y, th], dim=1)


def _apply_train_cutout(
    xb: torch.Tensor,
    *,
    enabled: bool,
    prob: float,
    min_len: int,
    max_len: int,
    fill_value: float,
) -> torch.Tensor:
    if not enabled or prob <= 0.0:
        return xb
    if xb.ndim != 3 or xb.shape[1] != 2 or xb.shape[2] != 360:
        return xb

    prob_eff = float(np.clip(prob, 0.0, 1.0))
    min_len_eff = max(1, int(min_len))
    max_len_eff = max(min_len_eff, int(max_len))
    if max_len_eff <= 0:
        return xb

    batch = xb.shape[0]
    apply_mask = torch.rand((batch,), device=xb.device) < prob_eff
    if not torch.any(apply_mask):
        return xb

    starts = torch.randint(0, 360, (batch,), device=xb.device)
    lengths = torch.randint(min_len_eff, max_len_eff + 1, (batch,), device=xb.device)
    beam_idx = torch.arange(360, device=xb.device).view(1, 1, 360)
    relative = (beam_idx - starts.view(-1, 1, 1)) % 360
    mask = relative < lengths.view(-1, 1, 1)
    mask = mask.expand(-1, xb.shape[1], -1) & apply_mask.view(-1, 1, 1)

    xb_aug = xb.clone()
    xb_aug.masked_fill_(mask, float(fill_value))
    return xb_aug


class TrainModelRobak(Node):
    def __init__(self):
        super().__init__("train_model_robak")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("dataset_name", "dataset_robak.npz")
        self.declare_parameter("model_name", "model_robak.pt")
        self.declare_parameter("history_name", "train_history_robak.json")
        self.declare_parameter("skip_if_model_exists", True)
        self.declare_parameter("write_experiment_metadata", False)

        self.declare_parameter("max_epochs", 200)
        self.declare_parameter("patience", 20)
        self.declare_parameter("min_delta", 1e-5)
        self.declare_parameter("lr", 1e-3)
        self.declare_parameter("weight_decay", 1e-4)
        self.declare_parameter("val_ratio", 0.2)
        self.declare_parameter("split_strategy", "tail_holdout_no_shuffle")
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("torch_device", "auto")
        self.declare_parameter("torch_deterministic", False)
        self.declare_parameter("dataset_wait_timeout", 600.0)
        self.declare_parameter("normalization", "zscore")
        self.declare_parameter("target_mode", "se2_local")
        self.declare_parameter("label_source", "gt_delta")
        self.declare_parameter("loss_type", "mse")
        self.declare_parameter("huber_delta", 1.0)
        self.declare_parameter("lr_schedule", "none")
        self.declare_parameter("loss_dx_weight", 1.0)
        self.declare_parameter("loss_dy_weight", 1.0)
        self.declare_parameter("loss_dtheta_weight", 1.0)
        self.declare_parameter("input_noise_std", 0.01)
        self.declare_parameter("clip_grad_norm", 1.0)
        self.declare_parameter("train_repeat_factor", 1)
        self.declare_parameter("train_cutout_enabled", False)
        self.declare_parameter("train_cutout_prob", 0.0)
        self.declare_parameter("train_cutout_min_len", 20)
        self.declare_parameter("train_cutout_max_len", 80)
        self.declare_parameter("train_cutout_fill_value", 0.0)
        self.declare_parameter("train_filter_max_step_trans", 0.0)
        self.declare_parameter("train_filter_max_step_yaw", 0.0)
        self.declare_parameter("train_filter_scan_offset", 0)
        self.declare_parameter("train_filter_scan_offsets", [-1])
        self.declare_parameter("selection_metric", "val_loss")
        self.declare_parameter("selection_min_delta", -1.0)
        self.declare_parameter("val_rollout_horizons", [2, 3, 4])
        self.declare_parameter("rollout_eval_scan_offset", 0)
        self.declare_parameter("rollout_eval_position_tol_m", 1e-3)
        self.declare_parameter("rollout_eval_yaw_tol_rad", 1e-3)
        self.declare_parameter("train_rollout_weight", 0.0)
        self.declare_parameter("train_rollout_horizon", 0)
        self.declare_parameter("train_rollout_windows_per_epoch", 0)
        self.declare_parameter("train_rollout_batch_size", 64)
        self.declare_parameter("train_rollout_xy_weight", 1.0)
        self.declare_parameter("train_rollout_yaw_weight", 0.25)

        self.seed = int(self.get_parameter("seed").value)
        self.torch_deterministic = bool(self.get_parameter("torch_deterministic").value)
        seed_all(self.seed, deterministic=self.torch_deterministic)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        ensure_dir(self.out_dir)

        self.dataset_path = os.path.join(self.out_dir, str(self.get_parameter("dataset_name").value))
        self.model_path = os.path.join(self.out_dir, str(self.get_parameter("model_name").value))
        self.history_path = os.path.join(self.out_dir, str(self.get_parameter("history_name").value))

        self.skip_if_model_exists = bool(self.get_parameter("skip_if_model_exists").value)
        self.max_epochs = int(self.get_parameter("max_epochs").value)
        self.patience = int(self.get_parameter("patience").value)
        self.min_delta = float(self.get_parameter("min_delta").value)
        self.lr = float(self.get_parameter("lr").value)
        self.weight_decay = float(self.get_parameter("weight_decay").value)
        self.val_ratio = float(self.get_parameter("val_ratio").value)
        self.split_strategy = str(self.get_parameter("split_strategy").value)
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.torch_device_request = str(self.get_parameter("torch_device").value)
        self.dataset_wait_timeout = float(self.get_parameter("dataset_wait_timeout").value)
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.normalization_mode = _normalize_mode(self.get_parameter("normalization").value)
        self.target_mode = _normalize_target_mode(self.get_parameter("target_mode").value)
        self.label_source = str(self.get_parameter("label_source").value).strip().lower()
        self.loss_type = _normalize_loss_type(self.get_parameter("loss_type").value)
        self.huber_delta = float(self.get_parameter("huber_delta").value)
        self.lr_schedule = _normalize_lr_schedule(self.get_parameter("lr_schedule").value)
        self.loss_dx_weight = float(self.get_parameter("loss_dx_weight").value)
        self.loss_dy_weight = float(self.get_parameter("loss_dy_weight").value)
        self.loss_dtheta_weight = float(self.get_parameter("loss_dtheta_weight").value)
        self.input_noise_std = float(self.get_parameter("input_noise_std").value)
        self.clip_grad_norm = float(self.get_parameter("clip_grad_norm").value)
        self.train_repeat_factor = max(1, int(self.get_parameter("train_repeat_factor").value))
        self.train_cutout_enabled = bool(self.get_parameter("train_cutout_enabled").value)
        self.train_cutout_prob = float(self.get_parameter("train_cutout_prob").value)
        self.train_cutout_min_len = int(self.get_parameter("train_cutout_min_len").value)
        self.train_cutout_max_len = int(self.get_parameter("train_cutout_max_len").value)
        self.train_cutout_fill_value = float(self.get_parameter("train_cutout_fill_value").value)
        self.train_filter_max_step_trans = float(self.get_parameter("train_filter_max_step_trans").value)
        self.train_filter_max_step_yaw = float(self.get_parameter("train_filter_max_step_yaw").value)
        self.train_filter_scan_offset = int(self.get_parameter("train_filter_scan_offset").value)
        raw_multi_offsets = self.get_parameter("train_filter_scan_offsets").value
        self.train_filter_scan_offsets = sorted(
            {int(v) for v in raw_multi_offsets if int(v) > 0}
        )
        self.selection_metric = _normalize_selection_metric(self.get_parameter("selection_metric").value)
        self.selection_min_delta = float(self.get_parameter("selection_min_delta").value)
        if self.selection_min_delta < 0.0:
            self.selection_min_delta = self.min_delta
        raw_rollout_horizons = self.get_parameter("val_rollout_horizons").value
        self.val_rollout_horizons = sorted({int(v) for v in raw_rollout_horizons if int(v) >= 1})
        self.rollout_eval_scan_offset = int(self.get_parameter("rollout_eval_scan_offset").value)
        self.rollout_eval_position_tol_m = float(self.get_parameter("rollout_eval_position_tol_m").value)
        self.rollout_eval_yaw_tol_rad = float(self.get_parameter("rollout_eval_yaw_tol_rad").value)
        self.train_rollout_weight = max(0.0, float(self.get_parameter("train_rollout_weight").value))
        self.train_rollout_horizon = max(0, int(self.get_parameter("train_rollout_horizon").value))
        self.train_rollout_windows_per_epoch = max(
            0, int(self.get_parameter("train_rollout_windows_per_epoch").value)
        )
        self.train_rollout_batch_size = max(1, int(self.get_parameter("train_rollout_batch_size").value))
        self.train_rollout_xy_weight = max(0.0, float(self.get_parameter("train_rollout_xy_weight").value))
        self.train_rollout_yaw_weight = max(0.0, float(self.get_parameter("train_rollout_yaw_weight").value))

        self.torch_device_info = select_torch_device(self.torch_device_request)
        self.device = torch.device(self.torch_device_info.resolved)
        self.exp_logger.add_note(
            f"train_model_robak torch device requested={self.torch_device_info.requested}, "
            f"resolved={self.torch_device_info.resolved}: {self.torch_device_info.reason}"
        )
        if self.torch_device_info.warning:
            self.exp_logger.add_note(self.torch_device_info.warning)
        self.exp_logger.save()
        self.get_logger().info(
            f"[Robak] Torch device: requested={self.torch_device_info.requested}, "
            f"using={self.torch_device_info.resolved} ({self.torch_device_info.reason})"
        )
        self.get_logger().info(f"[Robak] Torch deterministic mode: {self.torch_deterministic}")
        self.get_logger().info(
            f"[Robak] training config: normalization={self.normalization_mode}, target_mode={self.target_mode}, "
            f"loss_type={self.loss_type}, "
            f"weight_decay={self.weight_decay}, lr_schedule={self.lr_schedule}, "
            f"selection_metric={self.selection_metric}, "
            f"repeat_factor={self.train_repeat_factor}, cutout(enabled={self.train_cutout_enabled}, "
            f"prob={self.train_cutout_prob}, len=[{self.train_cutout_min_len},{self.train_cutout_max_len}], "
            f"fill={self.train_cutout_fill_value}), train_rollout_weight={self.train_rollout_weight}, "
            f"train_rollout_horizon={self.train_rollout_horizon}"
        )
        if self.torch_device_info.warning:
            self.get_logger().warn(self.torch_device_info.warning)

        self.timer = self.create_timer(0.5, self.run_once)
        self.did = False

    def run_once(self):
        if self.did:
            return
        self.did = True

        if self.skip_if_model_exists and os.path.exists(self.model_path):
            self.get_logger().info(f"[Robak] Model exists, skipping: {self.model_path}")
            rclpy.shutdown()
            return

        self.get_logger().info(
            f"[Robak] Waiting for dataset (timeout {self.dataset_wait_timeout:.0f}s): {self.dataset_path}"
        )
        dataset_ready, dataset_error = wait_for_npz_dataset(
            self.dataset_path,
            self.dataset_wait_timeout,
            required_keys=("X_pairs", "Y"),
        )
        if not dataset_ready:
            if dataset_error is not None and os.path.exists(self.dataset_path):
                self.get_logger().error(
                    f"[Robak] Dataset is incomplete or unreadable: {self.dataset_path} ({dataset_error})"
                )
            else:
                self.get_logger().error(f"[Robak] Dataset not found: {self.dataset_path}")
            rclpy.shutdown()
            return

        if self.write_experiment_metadata:
            self.exp_logger.start_training(
                seed=self.seed,
                max_epochs=self.max_epochs,
                patience=self.patience,
                min_delta=self.min_delta,
                lr=self.lr,
                val_ratio=self.val_ratio,
                batch_size=self.batch_size,
                torch_device_requested=self.torch_device_info.requested,
                torch_device_used=self.torch_device_info.resolved,
            )

        with np.load(self.dataset_path, allow_pickle=True) as data:
            X = data["X_pairs"].astype(np.float32)
            Y = data["Y"].astype(np.float32)
            Y_odom = data["Y_odom"].astype(np.float32) if "Y_odom" in data else None
            P = data["P"].astype(np.float32) if "P" in data else None
            N_off = data["N"].astype(np.int32) if "N" in data else None

        n_before = int(X.shape[0])
        requested_offsets = self.train_filter_scan_offsets
        if not requested_offsets and self.train_filter_scan_offset > 0:
            requested_offsets = [self.train_filter_scan_offset]
        if requested_offsets:
            if N_off is None:
                self.get_logger().error(
                    f"[Robak] Offset-filtered training requires per-sample 'N' in dataset, "
                    f"but {self.dataset_path} has no 'N'. Requested scan_offset(s)={requested_offsets}."
                )
                rclpy.shutdown()
                return
            offset_mask = np.isin(N_off, np.asarray(requested_offsets, dtype=np.int32))
            n_offset_match = int(offset_mask.sum())
            if n_offset_match <= 0:
                self.get_logger().error(
                    f"[Robak] No samples with scan_offset(s)={requested_offsets} found in dataset "
                    f"{self.dataset_path} (n_total={n_before})."
                )
                rclpy.shutdown()
                return
            X = X[offset_mask]
            Y = Y[offset_mask]
            if Y_odom is not None:
                Y_odom = Y_odom[offset_mask]
            if P is not None:
                P = P[offset_mask]
            N_off = N_off[offset_mask]
            self.get_logger().info(
                f"[Robak] Offset filter: {n_before} -> {n_offset_match} samples "
                f"({n_offset_match / max(n_before, 1) * 100.0:.1f}%) "
                f"[scan_offsets={requested_offsets}]"
            )
            n_before = n_offset_match

        if self.train_filter_max_step_trans > 0.0 or self.train_filter_max_step_yaw > 0.0:
            step_norm = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
            abs_dth = np.abs(Y[:, 2])
            mask = np.ones(int(X.shape[0]), dtype=bool)
            if self.train_filter_max_step_trans > 0.0:
                mask &= step_norm <= self.train_filter_max_step_trans
            if self.train_filter_max_step_yaw > 0.0:
                mask &= abs_dth <= self.train_filter_max_step_yaw
            X = X[mask]
            Y = Y[mask]
            if Y_odom is not None:
                Y_odom = Y_odom[mask]
            if P is not None:
                P = P[mask]
            if N_off is not None:
                N_off = N_off[mask]
            n_after = int(X.shape[0])
            self.get_logger().info(
                f"[Robak] Dataset filter: {n_before} -> {n_after} samples "
                f"({n_after / max(n_before, 1) * 100.0:.1f}%) "
                f"[max_step_trans={self.train_filter_max_step_trans:.3f}, "
                f"max_step_yaw={self.train_filter_max_step_yaw:.3f}]"
            )

        if self.label_source == "residual_delta":
            if Y_odom is None:
                self.get_logger().error(
                    "[Robak] label_source=residual_delta requires Y_odom in dataset, "
                    "but it was not found. Re-record dataset with odom_topic set."
                )
                rclpy.shutdown()
                return
            valid = np.all(np.isfinite(Y_odom), axis=1)
            n_invalid = int((~valid).sum())
            if n_invalid > 0:
                self.get_logger().warn(
                    f"[Robak] label_source=residual_delta: dropping {n_invalid} samples with NaN Y_odom"
                )
            X = X[valid]; Y = Y[valid]; Y_odom = Y_odom[valid]
            if P is not None: P = P[valid]
            if N_off is not None: N_off = N_off[valid]
            Y_residual = (Y - Y_odom).astype(np.float32)
            res_mean = Y_residual.mean(axis=0)
            res_std = Y_residual.std(axis=0)
            odom_mean = Y_odom.mean(axis=0)
            gt_mean = Y.mean(axis=0)
            self.get_logger().info(
                f"[Robak] residual_delta: gt_mean={gt_mean}, odom_mean={odom_mean}, "
                f"res_mean={res_mean}, res_std={res_std}"
            )
            abs_res = np.abs(Y_residual)
            for i, dim in enumerate(["dx", "dy", "dtheta"]):
                self.get_logger().info(
                    f"[Robak] |{dim}| residual p50={np.percentile(abs_res[:,i],50):.4f} "
                    f"p75={np.percentile(abs_res[:,i],75):.4f} "
                    f"p90={np.percentile(abs_res[:,i],90):.4f}"
                )
            Y = Y_residual

        n = int(X.shape[0])
        if n < 100:
            self.get_logger().error(f"[Robak] Dataset too small after filtering: n={n}")
            rclpy.shutdown()
            return

        if self.target_mode == "forward_yaw":
            if P is not None:
                Y_target = _forward_yaw_targets_from_pose_pairs(P)
            else:
                self.get_logger().warn(
                    "[Robak] target_mode=forward_yaw requested but dataset has no pose pairs 'P'; "
                    "falling back to Y[:, [0, 2]]."
                )
                Y_target = Y[:, [0, 2]].astype(np.float32)
            target_names = ("forward", "dtheta")
        else:
            Y_target = Y
            target_names = ("dx", "dy", "dtheta")

        train_idx, val_idx, split_strategy_used = split_train_val_indices(
            n,
            self.val_ratio,
            seed=self.seed,
            split_strategy=self.split_strategy,
        )
        X_tr, Y_tr = X[train_idx], Y_target[train_idx]
        X_val, Y_val = X[val_idx], Y_target[val_idx]
        P_tr = P[train_idx] if P is not None else None
        P_val = P[val_idx] if P is not None else None
        N_val = N_off[val_idx] if N_off is not None else None
        n_val = int(val_idx.size)
        split_stats = split_time_coverage_stats(
            n,
            train_idx,
            val_idx,
            split_strategy=split_strategy_used,
        )

        x_mean = x_std = y_mean = y_std = None
        Y_val_raw_t = torch.from_numpy(Y_val).float().to(self.device)
        if self.normalization_mode == "zscore":
            Xtr_flat = X_tr.reshape((X_tr.shape[0], -1))
            Xva_flat = X_val.reshape((X_val.shape[0], -1))
            x_mean = Xtr_flat.mean(axis=0).astype(np.float32)
            x_std = (Xtr_flat.std(axis=0) + 1e-6).astype(np.float32)
            y_mean = Y_tr.mean(axis=0).astype(np.float32)
            y_std = (Y_tr.std(axis=0) + 1e-6).astype(np.float32)
            x_norm = Normalizer(x_mean, x_std)
            y_norm = Normalizer(y_mean, y_std)
            X_tr_t = torch.from_numpy(x_norm.apply(Xtr_flat).reshape((-1, 2, 360))).float().to(self.device)
            Y_tr_t = torch.from_numpy(y_norm.apply(Y_tr)).float().to(self.device)
            X_val_t = torch.from_numpy(x_norm.apply(Xva_flat).reshape((-1, 2, 360))).float().to(self.device)
            Y_val_loss_t = torch.from_numpy(y_norm.apply(Y_val)).float().to(self.device)
            y_mean_t = torch.from_numpy(y_mean).float().to(self.device)
            y_std_t = torch.from_numpy(y_std).float().to(self.device)
        else:
            X_tr_t = torch.from_numpy(X_tr).float().to(self.device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(self.device)
            X_val_t = torch.from_numpy(X_val).float().to(self.device)
            Y_val_loss_t = torch.from_numpy(Y_val).float().to(self.device)
            y_mean_t = None
            y_std_t = None
        P_tr_t = None
        train_rollout_starts = np.zeros((0,), dtype=np.int64)
        rollout_offsets_t = None
        if self.train_rollout_weight > 0.0 and self.train_rollout_horizon > 0 and P_tr is not None:
            train_rollout_starts = collect_rollout_window_starts(
                P_tr,
                self.train_rollout_horizon,
                continuity_pos_tol_m=self.rollout_eval_position_tol_m,
                continuity_yaw_tol_rad=self.rollout_eval_yaw_tol_rad,
            )
            if train_rollout_starts.size > 0:
                P_tr_t = torch.from_numpy(P_tr.astype(np.float32)).to(self.device)
                rollout_offsets_t = torch.arange(
                    self.train_rollout_horizon,
                    device=self.device,
                    dtype=torch.long,
                ).view(1, -1)
            else:
                self.get_logger().warn(
                    "[Robak] train_rollout_weight>0 but no contiguous training rollout windows were found; "
                    "rollout loss will stay disabled."
                )

        model = RobakConv1D(out_dim=Y_target.shape[1]).to(self.device)
        if self.write_experiment_metadata:
            self.exp_logger.set_training_dataset_info(
                n_total=n,
                n_train=X_tr.shape[0],
                n_val=n_val,
                input_dim=720,
                output_dim=int(Y_target.shape[1]),
            )
            self.exp_logger.set_training_model_info(
                architecture=f"RobakConv1D(2x360->{int(Y_target.shape[1])})",
                model=model,
            )

        opt = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        if self.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-6)

        if self.loss_type == "huber":
            loss_fn = nn.HuberLoss(delta=self.huber_delta, reduction="none")
        else:
            loss_fn = nn.MSELoss(reduction="none")
        if self.target_mode == "forward_yaw":
            loss_weight_values = [
                max(self.loss_dx_weight, 1e-6),
                max(self.loss_dtheta_weight, 1e-6),
            ]
        else:
            loss_weight_values = [
                max(self.loss_dx_weight, 1e-6),
                max(self.loss_dy_weight, 1e-6),
                max(self.loss_dtheta_weight, 1e-6),
            ]
        loss_weights_t = torch.tensor(loss_weight_values, dtype=torch.float32, device=self.device)

        best_val = float("inf")
        best_selection_value = float("inf")
        best_state = None
        wait = 0
        history = {
            "seed": self.seed,
            "architecture": f"RobakConv1D(2x360->{int(Y_target.shape[1])})",
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "normalization": self.normalization_mode,
            "target_mode": self.target_mode,
            "label_source": self.label_source,
            "target_names": list(target_names),
            "loss_type": self.loss_type,
            "huber_delta": self.huber_delta,
            "lr_schedule": self.lr_schedule,
            "batch_size": self.batch_size,
            "val_ratio": self.val_ratio,
            "split_strategy": split_strategy_used,
            "split": split_stats,
            "patience": self.patience,
            "input_noise_std": self.input_noise_std,
            "clip_grad_norm": self.clip_grad_norm,
            "train_repeat_factor": self.train_repeat_factor,
            "train_cutout_enabled": self.train_cutout_enabled,
            "train_cutout_prob": self.train_cutout_prob,
            "train_cutout_min_len": self.train_cutout_min_len,
            "train_cutout_max_len": self.train_cutout_max_len,
            "train_cutout_fill_value": self.train_cutout_fill_value,
            "train_filter_max_step_trans": self.train_filter_max_step_trans,
            "train_filter_max_step_yaw": self.train_filter_max_step_yaw,
            "train_filter_scan_offset": self.train_filter_scan_offset,
            "train_filter_scan_offsets": list(self.train_filter_scan_offsets),
            "selection_metric": self.selection_metric,
            "selection_min_delta": self.selection_min_delta,
            "val_rollout_horizons": list(self.val_rollout_horizons),
            "rollout_eval_scan_offset": self.rollout_eval_scan_offset,
            "rollout_eval_position_tol_m": self.rollout_eval_position_tol_m,
            "rollout_eval_yaw_tol_rad": self.rollout_eval_yaw_tol_rad,
            "train_rollout_weight": self.train_rollout_weight,
            "train_rollout_horizon": self.train_rollout_horizon,
            "train_rollout_windows_per_epoch": self.train_rollout_windows_per_epoch,
            "train_rollout_batch_size": self.train_rollout_batch_size,
            "train_rollout_xy_weight": self.train_rollout_xy_weight,
            "train_rollout_yaw_weight": self.train_rollout_yaw_weight,
            "n_train_rollout_windows": int(train_rollout_starts.size),
            "loss_dx_weight": self.loss_dx_weight,
            "loss_dy_weight": self.loss_dy_weight,
            "loss_dtheta_weight": self.loss_dtheta_weight,
            "loss_weights": loss_weight_values,
            "torch_device_requested": self.torch_device_info.requested,
            "torch_device_used": self.torch_device_info.resolved,
            "torch_device_reason": self.torch_device_info.reason,
            "epochs": [],
        }
        if self.torch_device_info.warning:
            history["torch_device_warning"] = self.torch_device_info.warning

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            train_losses = []
            for _repeat_idx in range(self.train_repeat_factor):
                perm = torch.randperm(X_tr_t.shape[0], device=self.device)
                for i in range(0, X_tr_t.shape[0], self.batch_size):
                    idx = perm[i : i + self.batch_size]
                    xb = X_tr_t[idx]
                    yb = Y_tr_t[idx]
                    if self.input_noise_std > 0.0:
                        xb = xb + self.input_noise_std * torch.randn_like(xb)
                    xb = _apply_train_cutout(
                        xb,
                        enabled=self.train_cutout_enabled,
                        prob=self.train_cutout_prob,
                        min_len=self.train_cutout_min_len,
                        max_len=self.train_cutout_max_len,
                        fill_value=self.train_cutout_fill_value,
                    )
                    opt.zero_grad(set_to_none=True)
                    pred = model(xb)
                    loss = _weighted_loss(loss_fn, pred, yb, loss_weights_t)
                    loss.backward()
                    if self.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    opt.step()
                    train_losses.append(float(loss.detach().cpu().item()))

            train_rollout_loss = None
            train_rollout_windows_used = 0
            if (
                self.train_rollout_weight > 0.0
                and rollout_offsets_t is not None
                and train_rollout_starts.size > 0
            ):
                model.train()
                available_windows = int(train_rollout_starts.size)
                requested_windows = self.train_rollout_windows_per_epoch or available_windows
                n_selected_windows = min(available_windows, max(1, requested_windows))
                rng = np.random.default_rng(self.seed + epoch)
                selected_starts = train_rollout_starts[
                    rng.choice(available_windows, size=n_selected_windows, replace=False)
                ]
                rollout_losses_epoch = []
                for start_idx in range(0, n_selected_windows, self.train_rollout_batch_size):
                    start_batch = selected_starts[start_idx : start_idx + self.train_rollout_batch_size]
                    if start_batch.size == 0:
                        continue
                    start_batch_t = torch.from_numpy(start_batch.astype(np.int64)).to(self.device)
                    idx_mat = start_batch_t.view(-1, 1) + rollout_offsets_t
                    xb_roll = X_tr_t[idx_mat.reshape(-1)]

                    opt.zero_grad(set_to_none=True)
                    pred_roll = model(xb_roll).view(-1, self.train_rollout_horizon, Y_target.shape[1])
                    if self.normalization_mode == "zscore":
                        pred_roll_raw = pred_roll * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
                    else:
                        pred_roll_raw = pred_roll
                    if self.target_mode == "forward_yaw":
                        local_deltas = torch.stack(
                            [
                                pred_roll_raw[:, :, 0],
                                torch.zeros_like(pred_roll_raw[:, :, 0]),
                                pred_roll_raw[:, :, 1],
                            ],
                            dim=2,
                        )
                    else:
                        local_deltas = pred_roll_raw[:, :, :3]
                    start_pose = P_tr_t[start_batch_t, 0:3]
                    gt_end = P_tr_t[start_batch_t + (self.train_rollout_horizon - 1), 3:6]
                    pred_end = _compose_local_pose_sequence_torch(start_pose, local_deltas)
                    xy_sq = torch.sum((pred_end[:, 0:2] - gt_end[:, 0:2]) ** 2, dim=1)
                    yaw_err = _wrap_torch_angle(pred_end[:, 2] - gt_end[:, 2])
                    rollout_loss_raw = (
                        self.train_rollout_xy_weight * xy_sq.mean()
                        + self.train_rollout_yaw_weight * torch.mean(yaw_err * yaw_err)
                    )
                    rollout_loss = self.train_rollout_weight * rollout_loss_raw
                    rollout_loss.backward()
                    if self.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    opt.step()
                    rollout_losses_epoch.append(float(rollout_loss_raw.detach().cpu().item()))
                    train_rollout_windows_used += int(start_batch.size)
                if rollout_losses_epoch:
                    train_rollout_loss = float(np.mean(rollout_losses_epoch))

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_loss = float(_weighted_loss(loss_fn, pred_val, Y_val_loss_t, loss_weights_t).detach().cpu().item())
                if self.normalization_mode == "zscore":
                    pred_val_raw = pred_val * y_std_t + y_mean_t
                else:
                    pred_val_raw = pred_val
                val_err = torch.abs(pred_val_raw - Y_val_raw_t)
                val_mae_target = [
                    float(val_err[:, idx].mean().detach().cpu().item())
                    for idx in range(val_err.shape[1])
                ]
                val_rmse_raw = float(torch.sqrt(torch.mean((pred_val_raw - Y_val_raw_t) ** 2)).detach().cpu().item())
                pred_val_raw_np = pred_val_raw.detach().cpu().numpy().astype(np.float32)

            rollout_metrics = {}
            if P_val is not None and self.val_rollout_horizons:
                if self.target_mode == "forward_yaw":
                    pred_rollout_local = np.stack(
                        [
                            pred_val_raw_np[:, 0],
                            np.zeros(pred_val_raw_np.shape[0], dtype=np.float32),
                            pred_val_raw_np[:, 1],
                        ],
                        axis=1,
                    ).astype(np.float32)
                else:
                    pred_rollout_local = pred_val_raw_np[:, :3].astype(np.float32)

                rollout_pose_pairs = P_val
                if self.rollout_eval_scan_offset > 0 and N_val is not None:
                    rollout_mask = N_val == int(self.rollout_eval_scan_offset)
                    pred_rollout_local = pred_rollout_local[rollout_mask]
                    rollout_pose_pairs = P_val[rollout_mask]

                rollout_metrics = compute_rollout_metrics_from_local_deltas(
                    pred_rollout_local,
                    rollout_pose_pairs,
                    horizons=self.val_rollout_horizons,
                    continuity_pos_tol_m=self.rollout_eval_position_tol_m,
                    continuity_yaw_tol_rad=self.rollout_eval_yaw_tol_rad,
                )

            tr_loss = float(np.mean(train_losses)) if train_losses else val_loss
            current_lr = float(opt.param_groups[0]["lr"])
            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_rmse_raw": val_rmse_raw,
                "train_rollout_loss": train_rollout_loss,
                "train_rollout_windows_used": train_rollout_windows_used,
            }
            for idx, name in enumerate(target_names):
                epoch_record[f"val_mae_{name}"] = val_mae_target[idx]
            epoch_record.update(rollout_metrics)

            selection_value = float(val_loss)
            if self.selection_metric != "val_loss":
                alt_value = epoch_record.get(self.selection_metric)
                if alt_value is not None and np.isfinite(float(alt_value)):
                    selection_value = float(alt_value)
            epoch_record["selection_value"] = selection_value
            history["epochs"].append(epoch_record)

            if epoch == 1 or epoch % 10 == 0 or epoch == self.max_epochs:
                mae_str = ", ".join(
                    f"{name}={value:.4f}" for name, value in zip(target_names, val_mae_target)
                )
                rollout_log = ""
                if self.val_rollout_horizons:
                    h_main = self.val_rollout_horizons[0]
                    rollout_main = epoch_record.get(f"rollout_xy_rmse_h{h_main}")
                    if rollout_main is not None:
                        rollout_log = f" rollout_xy_h{h_main}={float(rollout_main):.6f}"
                self.get_logger().info(
                    f"[Robak] epoch {epoch}/{self.max_epochs} "
                    f"train={tr_loss:.6f} val={val_loss:.6f} raw_rmse={val_rmse_raw:.6f} "
                    f"mae({mae_str}) "
                    f"best_sel={best_selection_value:.6f} sel={selection_value:.6f}{rollout_log} "
                    f"wait={wait} lr={current_lr:.2e}"
                )

            improved = (best_selection_value - selection_value) > self.selection_min_delta
            if improved:
                best_val = val_loss
                best_selection_value = selection_value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

            if scheduler is not None:
                scheduler.step()

        if best_state is not None:
            model.load_state_dict(best_state)

        payload = {
            "state_dict": torch_state_dict_to_cpu(model.state_dict()),
            "in_shape": (2, 360),
            "out_dim": int(Y_target.shape[1]),
            "seed": self.seed,
            "normalization": self.normalization_mode,
            "target_mode": self.target_mode,
            "label_source": self.label_source,
        }
        if self.normalization_mode == "zscore" and x_mean is not None:
            payload["x_mean"] = torch.from_numpy(x_mean)
            payload["x_std"] = torch.from_numpy(x_std)
            payload["y_mean"] = torch.from_numpy(y_mean)
            payload["y_std"] = torch.from_numpy(y_std)

        tmp = self.model_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.model_path)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        final_train_loss = history["epochs"][-1]["train_loss"] if history["epochs"] else 0.0
        early_stopped = wait >= self.patience
        best_epoch_idx = max(
            (i for i, e in enumerate(history["epochs"]) if e.get("selection_value", float("inf")) <= best_selection_value + self.selection_min_delta),
            default=0,
        ) + 1

        if self.write_experiment_metadata:
            self.exp_logger.end_training(
                epochs_run=len(history["epochs"]),
                best_epoch=best_epoch_idx,
                best_val_loss=best_val,
                final_train_loss=final_train_loss,
                early_stopped=early_stopped,
                model_path=self.model_path,
                history_path=self.history_path,
            )

        self.get_logger().info(
            f"[Robak] Saved model: {self.model_path} | best_val={best_val:.6f} "
            f"best_selection={best_selection_value:.6f} "
            f"normalization={self.normalization_mode} target_mode={self.target_mode}"
        )
        rclpy.shutdown()



def main():
    rclpy.init()
    node = TrainModelRobak()
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
