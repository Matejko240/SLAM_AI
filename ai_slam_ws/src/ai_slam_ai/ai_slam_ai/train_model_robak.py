import json
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node

import torch
import torch.nn as nn
import torch.optim as optim

from .common import seed_all, ensure_dir, Normalizer
from .experiment_logger import ExperimentLogger


class RobakConv1D(nn.Module):
    """Prosty Conv1D: input (B,2,360) -> (B,3)."""
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x: (B,2,360)
        z = self.feat(x)
        return self.head(z)


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
        self.declare_parameter("val_ratio", 0.2)
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("dataset_wait_timeout", 600.0)

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

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
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.dataset_wait_timeout = float(self.get_parameter("dataset_wait_timeout").value)
        self.write_experiment_metadata = bool(self.get_parameter("write_experiment_metadata").value)

        self.timer = self.create_timer(0.5, self.run_once)
        self.did = False
        self.node_start = time.time()

    def run_once(self):
        if self.did:
            return
        self.did = True

        if self.skip_if_model_exists and os.path.exists(self.model_path):
            self.get_logger().info(f"[Robak] Model exists, skipping: {self.model_path}")
            rclpy.shutdown()
            return

        t0 = time.time()
        self.get_logger().info(f"[Robak] Waiting for dataset (timeout {self.dataset_wait_timeout:.0f}s): {self.dataset_path}")
        while not os.path.exists(self.dataset_path) and (time.time() - t0) < self.dataset_wait_timeout:
            time.sleep(0.5)

        if not os.path.exists(self.dataset_path):
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
            )

        data = np.load(self.dataset_path, allow_pickle=True)
        X = data["X_pairs"].astype(np.float32)  # (N,2,360)
        Y = data["Y"].astype(np.float32)        # (N,3)

        n = int(X.shape[0])
        if n < 100:
            self.get_logger().error("[Robak] Dataset too small.")
            rclpy.shutdown()
            return

        idx = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

        n_val = int(max(1, round(self.val_ratio * n)))
        X_val, Y_val = X[:n_val], Y[:n_val]
        X_tr, Y_tr = X[n_val:], Y[n_val:]

        # Normalizacja: liczymy mean/std na spłaszczonym wejściu (720)
        Xtr_flat = X_tr.reshape((X_tr.shape[0], -1))
        Xva_flat = X_val.reshape((X_val.shape[0], -1))

        x_mean = Xtr_flat.mean(axis=0)
        x_std = Xtr_flat.std(axis=0) + 1e-6
        y_mean = Y_tr.mean(axis=0)
        y_std = Y_tr.std(axis=0) + 1e-6

        x_norm = Normalizer(x_mean, x_std)
        y_norm = Normalizer(y_mean, y_std)

        X_tr_t = torch.from_numpy(x_norm.apply(Xtr_flat).reshape((-1, 2, 360))).float()
        Y_tr_t = torch.from_numpy(y_norm.apply(Y_tr)).float()
        X_val_t = torch.from_numpy(x_norm.apply(Xva_flat).reshape((-1, 2, 360))).float()
        Y_val_t = torch.from_numpy(y_norm.apply(Y_val)).float()

        model = RobakConv1D()
        device = torch.device("cpu")
        model.to(device)

        if self.write_experiment_metadata:
            self.exp_logger.set_training_dataset_info(
                n_total=n, n_train=X_tr.shape[0], n_val=n_val,
                input_dim=720, output_dim=3
            )
            self.exp_logger.set_training_model_info(
                architecture="RobakConv1D(2x360)",
                model=model
            )

        opt = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        wait = 0
        history = {"seed": self.seed, "epochs": []}

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            perm = rng.permutation(X_tr_t.shape[0])
            Xb = X_tr_t[perm]
            Yb = Y_tr_t[perm]

            train_losses = []
            for i in range(0, Xb.shape[0], self.batch_size):
                xb = Xb[i:i + self.batch_size].to(device)
                yb = Yb[i:i + self.batch_size].to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                train_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t.to(device))
                val_loss = float(loss_fn(pred_val, Y_val_t.to(device)).detach().cpu().item())

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
            "state_dict": model.state_dict(),
            "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
            "x_std": torch.from_numpy(x_std.astype(np.float32)),
            "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
            "y_std": torch.from_numpy(y_std.astype(np.float32)),
            "in_shape": (2, 360),
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

        self.get_logger().info(f"[Robak] Saved model: {self.model_path} | best_val={best_val:.6f}")
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
