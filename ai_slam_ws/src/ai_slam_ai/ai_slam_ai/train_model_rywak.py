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
    compute_rollout_metrics_from_local_deltas,
    estimate_dt_from_pose_pairs_and_velocity_labels,
    ensure_dir,
    Normalizer,
    select_torch_device,
    seed_all,
    split_time_coverage_stats,
    split_train_val_indices,
    torch_state_dict_to_cpu,
    velocity_predictions_to_local_deltas,
    wait_for_npz_dataset,
)
from .experiment_logger import ExperimentLogger
from .rywak_models import (
    build_rywak_model,
    build_tanh_target_meta,
    normalize_model_type,
    normalize_target_scaling,
    output_activation_for_target_scaling,
    parse_hidden_dims,
    scale_targets_for_model_np,
    unscale_targets_from_model_np,
    unscale_targets_from_model_torch,
)


def _normalize_selection_metric(value: str) -> str:
    metric = str(value or "").strip().lower()
    return metric or "val_loss"


def _wrap_torch_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _compose_local_pose_sequence_torch(start_pose: torch.Tensor, local_deltas: torch.Tensor) -> torch.Tensor:
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


def _build_sequence_dataset(
    X_model: np.ndarray,
    Y_model: np.ndarray,
    Y_raw: np.ndarray,
    P: np.ndarray | None,
    dt_est: np.ndarray | None,
    *,
    sequence_length: int,
    continuity_pos_tol_m: float,
    continuity_yaw_tol_rad: float,
):
    if sequence_length <= 1:
        return {
            "X": X_model,
            "Y_model": Y_model,
            "Y_raw": Y_raw,
            "P": P,
            "dt_est": dt_est,
            "end_indices": np.arange(X_model.shape[0], dtype=np.int64),
            "window_starts": np.arange(X_model.shape[0], dtype=np.int64),
        }
    if P is None:
        return None
    starts = collect_rollout_window_starts(
        P,
        sequence_length,
        continuity_pos_tol_m=continuity_pos_tol_m,
        continuity_yaw_tol_rad=continuity_yaw_tol_rad,
    )
    if starts.size == 0:
        return None
    offsets = np.arange(sequence_length, dtype=np.int64).reshape(1, -1)
    idx = starts.reshape(-1, 1) + offsets
    end_idx = starts + (sequence_length - 1)
    return {
        "X": X_model[idx],
        "Y_model": Y_model[end_idx],
        "Y_raw": Y_raw[end_idx],
        "P": P[end_idx],
        "dt_est": dt_est[end_idx] if dt_est is not None else None,
        "end_indices": end_idx,
        "window_starts": starts,
    }


class TrainModelRywak(Node):
    def __init__(self):
        super().__init__("train_model_rywak")

        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("dataset_name", "dataset_rywak.npz")
        self.declare_parameter("model_name", "model_rywak.pt")
        self.declare_parameter("history_name", "train_history_rywak.json")
        self.declare_parameter("skip_if_model_exists", True)
        self.declare_parameter("write_experiment_metadata", False)

        self.declare_parameter("max_epochs", 200)
        self.declare_parameter("patience", 12)
        self.declare_parameter("min_delta", 1e-5)
        self.declare_parameter("lr", 1e-3)
        self.declare_parameter("val_ratio", 0.2)
        self.declare_parameter("split_strategy", "tail_holdout_no_shuffle")
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("torch_device", "auto")
        self.declare_parameter("torch_deterministic", False)
        self.declare_parameter("dataset_wait_timeout", 600.0)
        self.declare_parameter("model_type", "cnn")
        self.declare_parameter("sequence_length", 1)
        self.declare_parameter("hidden_dims", [192, 96, 48])
        self.declare_parameter("dropout", 0.1)
        self.declare_parameter("weight_decay", 1e-4)
        self.declare_parameter("huber_delta", 1.0)
        self.declare_parameter("input_noise_std", 0.02)
        self.declare_parameter("clip_grad_norm", 1.0)
        self.declare_parameter("loss_dx_weight", 1.0)
        self.declare_parameter("loss_dy_weight", 1.0)
        self.declare_parameter("loss_dtheta_weight", 1.5)
        self.declare_parameter("loss_v_weight", 1.0)
        self.declare_parameter("loss_w_weight", 1.5)
        self.declare_parameter("lambda_residual_reg", 0.0)
        self.declare_parameter("alpha_w_residual_reg", 1.0)
        self.declare_parameter("v_clip_abs", 0.0)
        self.declare_parameter("w_clip_abs", 0.0)
        self.declare_parameter("lr_schedule", "cosine")
        self.declare_parameter("selection_metric", "val_loss")
        self.declare_parameter("selection_min_delta", -1.0)
        self.declare_parameter("val_rollout_horizons", [2, 3, 5])
        self.declare_parameter("rollout_eval_position_tol_m", 1e-3)
        self.declare_parameter("rollout_eval_yaw_tol_rad", 1e-3)
        self.declare_parameter("target_scaling", "zscore")
        self.declare_parameter("target_tanh_gamma", 0.6)
        self.declare_parameter("target_tanh_v_min", -2.0)
        self.declare_parameter("target_tanh_v_max", 2.0)
        self.declare_parameter("target_tanh_w_min", -3.0)
        self.declare_parameter("target_tanh_w_max", 3.0)
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
        self.val_ratio = float(self.get_parameter("val_ratio").value)
        self.split_strategy = str(self.get_parameter("split_strategy").value)
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.torch_device_request = str(self.get_parameter("torch_device").value)
        self.dataset_wait_timeout = float(self.get_parameter("dataset_wait_timeout").value)
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)
        self.model_type = normalize_model_type(self.get_parameter("model_type").value)
        self.sequence_length = max(1, int(self.get_parameter("sequence_length").value))
        self.hidden_dims = parse_hidden_dims(self.get_parameter("hidden_dims").value)
        self.dropout = float(self.get_parameter("dropout").value)
        self.weight_decay = float(self.get_parameter("weight_decay").value)
        self.huber_delta = float(self.get_parameter("huber_delta").value)
        self.input_noise_std = float(self.get_parameter("input_noise_std").value)
        self.clip_grad_norm = float(self.get_parameter("clip_grad_norm").value)
        self.loss_dx_weight = float(self.get_parameter("loss_dx_weight").value)
        self.loss_dy_weight = float(self.get_parameter("loss_dy_weight").value)
        self.loss_dtheta_weight = float(self.get_parameter("loss_dtheta_weight").value)
        self.loss_v_weight = float(self.get_parameter("loss_v_weight").value)
        self.loss_w_weight = float(self.get_parameter("loss_w_weight").value)
        self.lambda_residual_reg = float(self.get_parameter("lambda_residual_reg").value)
        self.alpha_w_residual_reg = float(self.get_parameter("alpha_w_residual_reg").value)
        self.v_clip_abs = float(self.get_parameter("v_clip_abs").value)
        self.w_clip_abs = float(self.get_parameter("w_clip_abs").value)
        self.lr_schedule = str(self.get_parameter("lr_schedule").value)
        self.selection_metric = _normalize_selection_metric(self.get_parameter("selection_metric").value)
        self.selection_min_delta = float(self.get_parameter("selection_min_delta").value)
        if self.selection_min_delta < 0.0:
            self.selection_min_delta = self.min_delta
        raw_rollout_horizons = self.get_parameter("val_rollout_horizons").value
        self.val_rollout_horizons = sorted({int(v) for v in raw_rollout_horizons if int(v) >= 1})
        self.rollout_eval_position_tol_m = float(self.get_parameter("rollout_eval_position_tol_m").value)
        self.rollout_eval_yaw_tol_rad = float(self.get_parameter("rollout_eval_yaw_tol_rad").value)
        self.target_scaling = normalize_target_scaling(self.get_parameter("target_scaling").value)
        self.target_tanh_gamma = float(self.get_parameter("target_tanh_gamma").value)
        self.target_tanh_v_min = float(self.get_parameter("target_tanh_v_min").value)
        self.target_tanh_v_max = float(self.get_parameter("target_tanh_v_max").value)
        self.target_tanh_w_min = float(self.get_parameter("target_tanh_w_min").value)
        self.target_tanh_w_max = float(self.get_parameter("target_tanh_w_max").value)
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
            f"train_model_rywak torch device requested={self.torch_device_info.requested}, "
            f"resolved={self.torch_device_info.resolved}: {self.torch_device_info.reason}"
        )
        if self.torch_device_info.warning:
            self.exp_logger.add_note(self.torch_device_info.warning)
        self.exp_logger.save()
        self.get_logger().info(
            f"[Rywak] Torch device: requested={self.torch_device_info.requested}, "
            f"using={self.torch_device_info.resolved} ({self.torch_device_info.reason})"
        )
        self.get_logger().info(f"[Rywak] Torch deterministic mode: {self.torch_deterministic}")
        self.get_logger().info(
            f"[Rywak] training config: model_type={self.model_type}, seq_len={self.sequence_length}, "
            f"target_scaling={self.target_scaling}, dropout={self.dropout}, weight_decay={self.weight_decay}, "
            f"lr_schedule={self.lr_schedule}, selection_metric={self.selection_metric}, "
            f"input_noise_std={self.input_noise_std}, v_clip_abs={self.v_clip_abs}, "
            f"w_clip_abs={self.w_clip_abs}, train_rollout_weight={self.train_rollout_weight}, "
            f"train_rollout_horizon={self.train_rollout_horizon}"
        )
        if self.torch_device_info.warning:
            self.get_logger().warn(self.torch_device_info.warning)

        self.timer = self.create_timer(0.5, self.run_once)
        self.did = False
        self.node_start = time.time()

    def run_once(self):
        if self.did:
            return
        self.did = True

        if self.skip_if_model_exists and os.path.exists(self.model_path):
            self.get_logger().info(f"[Rywak] Model exists, skipping: {self.model_path}")
            rclpy.shutdown()
            return

        self.get_logger().info(f"[Rywak] Waiting for dataset (timeout {self.dataset_wait_timeout:.0f}s): {self.dataset_path}")
        dataset_ready, dataset_error = wait_for_npz_dataset(
            self.dataset_path,
            self.dataset_wait_timeout,
            required_keys=("X", "Y"),
        )

        if not dataset_ready:
            if dataset_error is not None and os.path.exists(self.dataset_path):
                self.get_logger().error(f"[Rywak] Dataset is incomplete or unreadable: {self.dataset_path} ({dataset_error})")
            else:
                self.get_logger().error(f"[Rywak] Dataset not found: {self.dataset_path}")
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
            X = data["X"].astype(np.float32)  # (N,in_dim)
            Y = data["Y"].astype(np.float32)  # (N,out_dim)
            P = data["P"].astype(np.float32) if "P" in data else None

        # Fix D: clip training labels to match inference clip range
        if Y.shape[1] >= 2:
            if self.v_clip_abs > 0.0:
                before_v = int(np.sum(np.abs(Y[:, 0]) > self.v_clip_abs))
                Y[:, 0] = np.clip(Y[:, 0], -self.v_clip_abs, self.v_clip_abs)
                if before_v > 0:
                    self.get_logger().info(f"[Rywak] Clipped {before_v} v labels to +/-{self.v_clip_abs}")
            if self.w_clip_abs > 0.0:
                before_w = int(np.sum(np.abs(Y[:, 1]) > self.w_clip_abs))
                Y[:, 1] = np.clip(Y[:, 1], -self.w_clip_abs, self.w_clip_abs)
                if before_w > 0:
                    self.get_logger().info(f"[Rywak] Clipped {before_w} w labels to +/-{self.w_clip_abs}")

        n = int(X.shape[0])
        if n < 100:
            self.get_logger().error("[Rywak] Dataset too small.")
            rclpy.shutdown()
            return

        out_dim = int(Y.shape[1]) if Y.ndim == 2 else 0
        if out_dim < 1:
            self.get_logger().error(f"[Rywak] Invalid output dimension in dataset Y: shape={Y.shape}")
            rclpy.shutdown()
            return

        train_idx, val_idx, split_strategy_used = split_train_val_indices(
            n,
            self.val_ratio,
            seed=self.seed,
            split_strategy=self.split_strategy,
        )
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        P_tr = P[train_idx] if P is not None else None
        P_val = P[val_idx] if P is not None else None
        n_val = int(val_idx.size)
        split_stats = split_time_coverage_stats(
            n,
            train_idx,
            val_idx,
            split_strategy=split_strategy_used,
        )
        dt_est_all = None
        dt_est_tr = None
        dt_est_val = None
        if P is not None and Y.shape[1] >= 2:
            dt_est_all = estimate_dt_from_pose_pairs_and_velocity_labels(P, Y)
            dt_est_tr = dt_est_all[train_idx]
            dt_est_val = dt_est_all[val_idx]

        x_mean = X_tr.mean(axis=0)
        x_std = X_tr.std(axis=0) + 1e-6
        y_mean = Y_tr.mean(axis=0)
        y_std = Y_tr.std(axis=0) + 1e-6

        effective_sequence_length = self.sequence_length if self.model_type in ("gru", "lstm") else 1
        if effective_sequence_length != self.sequence_length:
            self.get_logger().info(
                f"[Rywak] model_type={self.model_type} ignores sequence_length={self.sequence_length}; using 1."
            )
        effective_target_scaling = self.target_scaling
        if effective_target_scaling == "tanh" and out_dim != 2:
            self.get_logger().warn(
                f"[Rywak] target_scaling=tanh supports only out_dim=2; falling back to zscore for out_dim={out_dim}."
            )
            effective_target_scaling = "zscore"

        x_norm = Normalizer(x_mean, x_std)
        X_tr_norm = x_norm.apply(X_tr)
        X_val_norm = x_norm.apply(X_val)

        target_tanh_meta = None
        if effective_target_scaling == "tanh":
            target_tanh_meta = build_tanh_target_meta(
                Y_tr,
                gamma=self.target_tanh_gamma,
                v_min=self.target_tanh_v_min,
                v_max=self.target_tanh_v_max,
                w_min=self.target_tanh_w_min,
                w_max=self.target_tanh_w_max,
            )
            Y_tr_model = scale_targets_for_model_np(
                Y_tr,
                target_scaling=effective_target_scaling,
                y_mean=None,
                y_std=None,
                target_tanh_meta=target_tanh_meta,
            )
            Y_val_model = scale_targets_for_model_np(
                Y_val,
                target_scaling=effective_target_scaling,
                y_mean=None,
                y_std=None,
                target_tanh_meta=target_tanh_meta,
            )
        else:
            y_norm = Normalizer(y_mean, y_std)
            Y_tr_model = y_norm.apply(Y_tr)
            Y_val_model = y_norm.apply(Y_val)

        train_seq = _build_sequence_dataset(
            X_tr_norm,
            Y_tr_model,
            Y_tr,
            P_tr,
            dt_est_tr,
            sequence_length=effective_sequence_length,
            continuity_pos_tol_m=self.rollout_eval_position_tol_m,
            continuity_yaw_tol_rad=self.rollout_eval_yaw_tol_rad,
        )
        val_seq = _build_sequence_dataset(
            X_val_norm,
            Y_val_model,
            Y_val,
            P_val,
            dt_est_val,
            sequence_length=effective_sequence_length,
            continuity_pos_tol_m=self.rollout_eval_position_tol_m,
            continuity_yaw_tol_rad=self.rollout_eval_yaw_tol_rad,
        )
        if train_seq is None or val_seq is None:
            self.get_logger().error(
                f"[Rywak] sequence_length={effective_sequence_length} produced no contiguous train/val windows. "
                "Use a shorter sequence_length or rebuild the dataset."
            )
            rclpy.shutdown()
            return

        X_tr_t = torch.from_numpy(train_seq["X"]).float().to(self.device)
        Y_tr_t = torch.from_numpy(train_seq["Y_model"]).float().to(self.device)
        X_val_t = torch.from_numpy(val_seq["X"]).float().to(self.device)
        Y_val_t = torch.from_numpy(val_seq["Y_model"]).float().to(self.device)
        Y_val_raw = val_seq["Y_raw"].astype(np.float32)
        y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(self.device)
        y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
        P_tr_seq = train_seq["P"]
        P_val_seq = val_seq["P"]
        dt_est_tr_seq = train_seq["dt_est"]
        dt_est_val_seq = val_seq["dt_est"]

        P_tr_t = None
        dt_est_tr_t = None
        train_rollout_starts = np.zeros((0,), dtype=np.int64)
        rollout_offsets_t = None
        if (
            self.train_rollout_weight > 0.0
            and self.train_rollout_horizon > 0
            and P_tr_seq is not None
            and dt_est_tr_seq is not None
        ):
            train_rollout_starts = collect_rollout_window_starts(
                P_tr_seq,
                self.train_rollout_horizon,
                continuity_pos_tol_m=self.rollout_eval_position_tol_m,
                continuity_yaw_tol_rad=self.rollout_eval_yaw_tol_rad,
            )
            if train_rollout_starts.size > 0:
                P_tr_t = torch.from_numpy(P_tr_seq.astype(np.float32)).to(self.device)
                dt_est_tr_t = torch.from_numpy(dt_est_tr_seq.astype(np.float32)).to(self.device)
                rollout_offsets_t = torch.arange(
                    self.train_rollout_horizon,
                    device=self.device,
                    dtype=torch.long,
                ).view(1, -1)
            else:
                self.get_logger().warn(
                    "[Rywak] train_rollout_weight>0 but no contiguous training rollout windows were found; "
                    "rollout loss will stay disabled."
                )

        output_activation = output_activation_for_target_scaling(effective_target_scaling)
        model = build_rywak_model(
            model_type=self.model_type,
            in_dim=int(X.shape[1]),
            out_dim=out_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            sequence_length=effective_sequence_length,
            output_activation=output_activation,
        )
        model.to(self.device)
        self.get_logger().info(
            f"[Rywak] Prepared model dataset: train={tuple(X_tr_t.shape)}, val={tuple(X_val_t.shape)}, "
            f"model_type={self.model_type}, seq_len={effective_sequence_length}, "
            f"target_scaling={effective_target_scaling}"
        )

        if self.write_experiment_metadata:
            self.exp_logger.set_training_dataset_info(
                n_total=n,
                n_train=X_tr_t.shape[0],
                n_val=X_val_t.shape[0],
                input_dim=int(X.shape[1]), output_dim=out_dim
            )
            self.exp_logger.set_training_model_info(
                architecture=(
                    f"Rywak{self.model_type.upper()}(seq={effective_sequence_length},"
                    f"in={int(X.shape[1])}->{out_dim},scaling={effective_target_scaling})"
                ),
                model=model
            )

        opt = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.HuberLoss(delta=self.huber_delta, reduction="none")

        scheduler = None
        if self.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-6)
        if out_dim == 2:
            target_weight_values = [max(self.loss_v_weight, 1e-6), max(self.loss_w_weight, 1e-6)]
        elif out_dim == 3:
            target_weight_values = [
                max(self.loss_dx_weight, 1e-6),
                max(self.loss_dy_weight, 1e-6),
                max(self.loss_dtheta_weight, 1e-6),
            ]
        else:
            target_weight_values = [1.0] * out_dim
        target_weights = torch.tensor(
            target_weight_values,
            dtype=torch.float32,
            device=self.device,
        ).view(1, out_dim)

        best_val = float("inf")
        best_selection_value = float("inf")
        best_state = None
        wait = 0
        history = {
            "seed": self.seed,
            "architecture": f"Rywak{self.model_type.upper()}",
            "model_type": self.model_type,
            "sequence_length": effective_sequence_length,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "huber_delta": self.huber_delta,
            "input_noise_std": self.input_noise_std,
            "clip_grad_norm": self.clip_grad_norm,
            "lr_schedule": self.lr_schedule,
            "split_strategy": split_strategy_used,
            "split": split_stats,
            "out_dim": out_dim,
            "loss_v_weight": self.loss_v_weight,
            "loss_w_weight": self.loss_w_weight,
            "lambda_residual_reg": self.lambda_residual_reg,
            "alpha_w_residual_reg": self.alpha_w_residual_reg,
            "loss_dx_weight": self.loss_dx_weight,
            "loss_dy_weight": self.loss_dy_weight,
            "loss_dtheta_weight": self.loss_dtheta_weight,
            "target_weights": target_weight_values,
            "selection_metric": self.selection_metric,
            "selection_min_delta": self.selection_min_delta,
            "val_rollout_horizons": list(self.val_rollout_horizons),
            "rollout_eval_position_tol_m": self.rollout_eval_position_tol_m,
            "rollout_eval_yaw_tol_rad": self.rollout_eval_yaw_tol_rad,
            "target_scaling": effective_target_scaling,
            "target_tanh_meta": target_tanh_meta,
            "train_rollout_weight": self.train_rollout_weight,
            "train_rollout_horizon": self.train_rollout_horizon,
            "train_rollout_windows_per_epoch": self.train_rollout_windows_per_epoch,
            "train_rollout_batch_size": self.train_rollout_batch_size,
            "train_rollout_xy_weight": self.train_rollout_xy_weight,
            "train_rollout_yaw_weight": self.train_rollout_yaw_weight,
            "n_train_rollout_windows": int(train_rollout_starts.size),
            "n_train_model_samples": int(X_tr_t.shape[0]),
            "n_val_model_samples": int(X_val_t.shape[0]),
            "torch_device_requested": self.torch_device_info.requested,
            "torch_device_used": self.torch_device_info.resolved,
            "torch_device_reason": self.torch_device_info.reason,
            "epochs": [],
        }
        if self.torch_device_info.warning:
            history["torch_device_warning"] = self.torch_device_info.warning

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            perm = torch.randperm(X_tr_t.shape[0], device=self.device)

            train_losses = []
            for i in range(0, X_tr_t.shape[0], self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = X_tr_t[idx]
                yb = Y_tr_t[idx]
                if self.input_noise_std > 0.0:
                    xb = xb + self.input_noise_std * torch.randn_like(xb)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = (loss_fn(pred, yb) * target_weights).mean()
                if self.lambda_residual_reg > 0.0 and out_dim == 2:
                    # Penalize residual magnitude in physical space to discourage
                    # large corrections when the model is uncertain.
                    pred_phys = pred * y_std_t + y_mean_t
                    reg = (pred_phys[:, 0] ** 2 + self.alpha_w_residual_reg * pred_phys[:, 1] ** 2).mean()
                    loss = loss + self.lambda_residual_reg * reg
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
                    pred_roll = model(xb_roll).view(-1, self.train_rollout_horizon, out_dim)
                    pred_roll_raw = unscale_targets_from_model_torch(
                        pred_roll,
                        target_scaling=effective_target_scaling,
                        y_mean_t=y_mean_t.view(1, 1, -1),
                        y_std_t=y_std_t.view(1, 1, -1),
                        target_tanh_meta=target_tanh_meta,
                    )
                    dt_roll = dt_est_tr_t[idx_mat]
                    if out_dim >= 3:
                        local_deltas = pred_roll_raw[:, :, :3]
                    else:
                        local_deltas = torch.stack(
                            [
                                pred_roll_raw[:, :, 0] * dt_roll,
                                torch.zeros_like(pred_roll_raw[:, :, 0]),
                                pred_roll_raw[:, :, 1] * dt_roll,
                            ],
                            dim=2,
                        )
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

            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_loss = float((loss_fn(pred_val, Y_val_t) * target_weights).mean().detach().cpu().item())
                pred_val_model_np = pred_val.detach().cpu().numpy().astype(np.float32)
                pred_val_raw_np = unscale_targets_from_model_np(
                    pred_val_model_np,
                    target_scaling=effective_target_scaling,
                    y_mean=y_mean.astype(np.float32),
                    y_std=y_std.astype(np.float32),
                    target_tanh_meta=target_tanh_meta,
                )
                val_err_np = np.abs(pred_val_raw_np - Y_val_raw)
                val_mae_target = [
                    float(np.mean(val_err_np[:, idx]))
                    for idx in range(val_err_np.shape[1])
                ]
                val_rmse_raw = float(np.sqrt(np.mean((pred_val_raw_np - Y_val_raw) ** 2)))

            tr_loss = float(np.mean(train_losses)) if train_losses else val_loss
            epoch_record = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_rmse_raw": val_rmse_raw,
                "val_mae_v": val_mae_target[0] if len(val_mae_target) > 0 else None,
                "val_mae_w": val_mae_target[1] if len(val_mae_target) > 1 else None,
                "train_rollout_loss": train_rollout_loss,
                "train_rollout_windows_used": train_rollout_windows_used,
            }
            if P_val_seq is not None and dt_est_val_seq is not None and self.val_rollout_horizons:
                pred_rollout_local = velocity_predictions_to_local_deltas(pred_val_raw_np[:, :2], dt_est_val_seq)
                rollout_metrics = compute_rollout_metrics_from_local_deltas(
                    pred_rollout_local,
                    P_val_seq,
                    horizons=self.val_rollout_horizons,
                    continuity_pos_tol_m=self.rollout_eval_position_tol_m,
                    continuity_yaw_tol_rad=self.rollout_eval_yaw_tol_rad,
                )
                epoch_record.update(rollout_metrics)

            selection_value = float(val_loss)
            if self.selection_metric != "val_loss":
                alt_value = epoch_record.get(self.selection_metric)
                if alt_value is not None and np.isfinite(float(alt_value)):
                    selection_value = float(alt_value)
            epoch_record["selection_value"] = selection_value
            history["epochs"].append(epoch_record)

            if epoch == 1 or epoch % 10 == 0 or epoch == self.max_epochs:
                lr_now = scheduler.get_last_lr()[0] if scheduler is not None else self.lr
                rollout_log = ""
                if self.val_rollout_horizons:
                    h_main = self.val_rollout_horizons[0]
                    rollout_main = epoch_record.get(f"rollout_xy_rmse_h{h_main}")
                    if rollout_main is not None:
                        rollout_log = f" rollout_xy_h{h_main}={float(rollout_main):.6f}"
                self.get_logger().info(
                    f"[Rywak] epoch {epoch}/{self.max_epochs}  "
                    f"train={tr_loss:.6f}  val={val_loss:.6f}  "
                    f"raw_rmse={val_rmse_raw:.6f}  best_sel={best_selection_value:.6f}  "
                    f"sel={selection_value:.6f}{rollout_log}  wait={wait}  lr={lr_now:.2e}"
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

        if best_state is not None:
            model.load_state_dict(best_state)

        payload = {
            "state_dict": torch_state_dict_to_cpu(model.state_dict()),
            "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
            "x_std": torch.from_numpy(x_std.astype(np.float32)),
            "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
            "y_std": torch.from_numpy(y_std.astype(np.float32)),
            "in_dim": int(X.shape[1]),
            "out_dim": out_dim,
            "architecture": f"Rywak{self.model_type.upper()}",
            "model_type": self.model_type,
            "sequence_length": effective_sequence_length,
            "hidden_dims": list(self.hidden_dims),
            "dropout": float(self.dropout),
            "target_scaling": effective_target_scaling,
            "target_tanh_meta": target_tanh_meta,
            "seed": self.seed,
        }

        tmp = self.model_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.model_path)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        final_train_loss = history["epochs"][-1]["train_loss"] if history["epochs"] else 0.0
        early_stopped = wait >= self.patience
        best_epoch_idx = max(
            (i for i, e in enumerate(history["epochs"]) if e.get("selection_value", float("inf")) <= best_selection_value + self.selection_min_delta),
            default=0
        ) + 1

        if self.write_experiment_metadata:
            self.exp_logger.end_training(
                epochs_run=len(history["epochs"]),
                best_epoch=best_epoch_idx,
                best_val_loss=best_val,
                final_train_loss=final_train_loss,
                early_stopped=early_stopped,
                model_path=self.model_path,
                history_path=self.history_path
            )

        self.get_logger().info(
            f"[Rywak] Saved model: {self.model_path} | best_val={best_val:.6f} "
            f"best_selection={best_selection_value:.6f}"
        )
        rclpy.shutdown()


def main():
    rclpy.init()
    node = TrainModelRywak()
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
