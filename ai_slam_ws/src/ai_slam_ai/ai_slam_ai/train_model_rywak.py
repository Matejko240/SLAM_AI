import json
import os
import time
import numpy as np
from typing import List

import rclpy
from rclpy.node import Node

import torch
import torch.nn as nn
import torch.optim as optim

from .common import (
    seed_all,
    ensure_dir,
    Normalizer,
    select_torch_device,
    split_time_coverage_stats,
    split_train_val_indices,
    torch_state_dict_to_cpu,
    wait_for_npz_dataset,
)
from .experiment_logger import ExperimentLogger


def _parse_hidden_dims(value) -> List[int]:
    if value is None:
        return [192, 96, 48]
    dims = [int(v) for v in list(value) if int(v) > 0]
    return dims if dims else [192, 96, 48]


class MLP2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dims: List[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [192, 96, 48]

        layers = []
        prev = int(in_dim)
        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
        self.declare_parameter("patience", 20)
        self.declare_parameter("min_delta", 1e-5)
        self.declare_parameter("lr", 1e-3)
        self.declare_parameter("val_ratio", 0.2)
        self.declare_parameter("split_strategy", "tail_holdout_no_shuffle")
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("torch_device", "auto")
        self.declare_parameter("torch_deterministic", True)
        self.declare_parameter("dataset_wait_timeout", 600.0)
        self.declare_parameter("hidden_dims", [192, 96, 48])
        self.declare_parameter("dropout", 0.1)
        self.declare_parameter("weight_decay", 1e-4)
        self.declare_parameter("huber_delta", 1.0)
        self.declare_parameter("input_noise_std", 0.02)
        self.declare_parameter("clip_grad_norm", 1.0)
        self.declare_parameter("loss_v_weight", 1.0)
        self.declare_parameter("loss_w_weight", 1.5)

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
        self.hidden_dims = _parse_hidden_dims(self.get_parameter("hidden_dims").value)
        self.dropout = float(self.get_parameter("dropout").value)
        self.weight_decay = float(self.get_parameter("weight_decay").value)
        self.huber_delta = float(self.get_parameter("huber_delta").value)
        self.input_noise_std = float(self.get_parameter("input_noise_std").value)
        self.clip_grad_norm = float(self.get_parameter("clip_grad_norm").value)
        self.loss_v_weight = float(self.get_parameter("loss_v_weight").value)
        self.loss_w_weight = float(self.get_parameter("loss_w_weight").value)
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
            Y = data["Y"].astype(np.float32)  # (N,2)

        n = int(X.shape[0])
        if n < 100:
            self.get_logger().error("[Rywak] Dataset too small.")
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
        n_val = int(val_idx.size)
        split_stats = split_time_coverage_stats(
            n,
            train_idx,
            val_idx,
            split_strategy=split_strategy_used,
        )

        x_mean = X_tr.mean(axis=0)
        x_std = X_tr.std(axis=0) + 1e-6
        y_mean = Y_tr.mean(axis=0)
        y_std = Y_tr.std(axis=0) + 1e-6

        x_norm = Normalizer(x_mean, x_std)
        y_norm = Normalizer(y_mean, y_std)

        X_tr_t = torch.from_numpy(x_norm.apply(X_tr)).float().to(self.device)
        Y_tr_t = torch.from_numpy(y_norm.apply(Y_tr)).float().to(self.device)
        X_val_t = torch.from_numpy(x_norm.apply(X_val)).float().to(self.device)
        Y_val_t = torch.from_numpy(y_norm.apply(Y_val)).float().to(self.device)

        model = MLP2(
            in_dim=int(X.shape[1]),
            out_dim=2,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        model.to(self.device)

        if self.write_experiment_metadata:
            self.exp_logger.set_training_dataset_info(
                n_total=n, n_train=X_tr.shape[0], n_val=n_val,
                input_dim=int(X.shape[1]), output_dim=2
            )
            self.exp_logger.set_training_model_info(
                architecture=f"MLP2(in={int(X.shape[1])}->{self.hidden_dims}->2, dropout={self.dropout})",
                model=model
            )

        opt = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.HuberLoss(delta=self.huber_delta, reduction="none")
        target_weights = torch.tensor(
            [max(self.loss_v_weight, 1e-6), max(self.loss_w_weight, 1e-6)],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 2)

        best_val = float("inf")
        best_state = None
        wait = 0
        history = {
            "seed": self.seed,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "huber_delta": self.huber_delta,
            "input_noise_std": self.input_noise_std,
            "clip_grad_norm": self.clip_grad_norm,
            "split_strategy": split_strategy_used,
            "split": split_stats,
            "loss_v_weight": self.loss_v_weight,
            "loss_w_weight": self.loss_w_weight,
            "torch_device_requested": self.torch_device_info.requested,
            "torch_device_used": self.torch_device_info.resolved,
            "torch_device_reason": self.torch_device_info.reason,
            "epochs": [],
        }
        if self.torch_device_info.warning:
            history["torch_device_warning"] = self.torch_device_info.warning

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            rng = np.random.default_rng(self.seed + epoch)
            perm = torch.from_numpy(rng.permutation(X_tr_t.shape[0]).astype(np.int64)).to(self.device)
            Xb = X_tr_t.index_select(0, perm)
            Yb = Y_tr_t.index_select(0, perm)

            train_losses = []
            for i in range(0, Xb.shape[0], self.batch_size):
                xb = Xb[i:i + self.batch_size]
                yb = Yb[i:i + self.batch_size]
                if self.input_noise_std > 0.0:
                    xb = xb + self.input_noise_std * torch.randn_like(xb)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = (loss_fn(pred, yb) * target_weights).mean()
                loss.backward()
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                opt.step()
                train_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_loss = float((loss_fn(pred_val, Y_val_t) * target_weights).mean().detach().cpu().item())

            tr_loss = float(np.mean(train_losses)) if train_losses else val_loss
            history["epochs"].append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})

            improved = (best_val - val_loss) > self.min_delta
            if improved:
                best_val = val_loss
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
            "hidden_dims": list(self.hidden_dims),
            "dropout": float(self.dropout),
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
            (i for i, e in enumerate(history["epochs"]) if e["val_loss"] <= best_val + self.min_delta),
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

        self.get_logger().info(f"[Rywak] Saved model: {self.model_path} | best_val={best_val:.6f}")
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
