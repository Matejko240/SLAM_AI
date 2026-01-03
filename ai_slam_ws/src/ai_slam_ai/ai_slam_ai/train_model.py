import json
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node

from .common import seed_all, ensure_dir, Normalizer
from .experiment_logger import ExperimentLogger

import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TrainModel(Node):
    def __init__(self):
        super().__init__("train_model")
        self.declare_parameter("seed", 123)
        self.declare_parameter("out_dir", "out")
        self.declare_parameter("experiment_id", "")
        self.declare_parameter("dataset_name", "dataset.npz")
        self.declare_parameter("model_name", "model.pt")
        self.declare_parameter("history_name", "train_history.json")
        self.declare_parameter("skip_if_model_exists", True)
        self.declare_parameter("max_epochs", 200)
        self.declare_parameter("patience", 20)
        self.declare_parameter("min_delta", 1e-5)
        self.declare_parameter("lr", 1e-3)
        self.declare_parameter("val_ratio", 0.2)
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("dataset_wait_timeout", 600.0)  # Max wait for dataset (seconds)

        self.seed = int(self.get_parameter("seed").value)
        seed_all(self.seed)

        base_out_dir = os.path.abspath(str(self.get_parameter("out_dir").value))
        experiment_id = str(self.get_parameter("experiment_id").value) or None
        
        # Inicjalizacja loggera eksperymentu (używa istniejącego podfolderu)
        self.exp_logger = ExperimentLogger(base_out_dir, experiment_id)
        self.out_dir = self.exp_logger.get_output_dir()
        ensure_dir(self.out_dir)
        
        self.get_logger().info(f"Output directory: {self.out_dir}")
        self.get_logger().info(f"Experiment ID: {self.exp_logger.experiment_id}")

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

        self.timer = self.create_timer(0.5, self.run_once)
        self.did = False
        self.node_start = time.time()  # Time when node started

    def run_once(self):
        if self.did:
            return
        self.did = True

        if self.skip_if_model_exists and os.path.exists(self.model_path):
            self.get_logger().info(f"Model exists, skipping: {self.model_path}")
            rclpy.shutdown()
            return

        t0 = time.time()
        self.get_logger().info(f"Czekam na dataset (timeout: {self.dataset_wait_timeout:.0f}s)...")
        while not os.path.exists(self.dataset_path) and time.time() - t0 < self.dataset_wait_timeout:
            time.sleep(0.5)

        if not os.path.exists(self.dataset_path):
            self.get_logger().error(f"Dataset not found: {self.dataset_path}")
            self._save_zero_model()
            rclpy.shutdown()
            return

        self.train_start = time.time()
        wait_time = self.train_start - self.node_start
        self.get_logger().info("="*60)
        self.get_logger().info(f"[FAZA 2] TRENING MODELU AI - START (czekano {wait_time:.0f}s na dataset)")
        self.get_logger().info(f"Dataset: {self.dataset_path}")
        self.get_logger().info("="*60)
        
        # Logowanie startu treningu
        self.exp_logger.start_training(
            seed=self.seed,
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_delta=self.min_delta,
            lr=self.lr,
            val_ratio=self.val_ratio,
            batch_size=self.batch_size
        )

        data = np.load(self.dataset_path, allow_pickle=True)
        X_scan = data["X_scan"].astype(np.float32)
        X_odom = data["X_odom"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

        X = np.concatenate([X_scan, X_odom], axis=1)
        n = X.shape[0]
        if n < 50:
            self.get_logger().error("Dataset too small, saving zero model.")
            self._save_zero_model()
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

        x_mean = X_tr.mean(axis=0)
        x_std = X_tr.std(axis=0) + 1e-6
        y_mean = Y_tr.mean(axis=0)
        y_std = Y_tr.std(axis=0) + 1e-6

        x_norm = Normalizer(x_mean, x_std)
        y_norm = Normalizer(y_mean, y_std)

        X_tr_t = torch.from_numpy(x_norm.apply(X_tr))
        Y_tr_t = torch.from_numpy(y_norm.apply(Y_tr))
        X_val_t = torch.from_numpy(x_norm.apply(X_val))
        Y_val_t = torch.from_numpy(y_norm.apply(Y_val))

        model = MLP(in_dim=X.shape[1], out_dim=3)
        device = torch.device("cpu")
        model.to(device)
        
        # Logowanie informacji o datasecie i modelu
        self.exp_logger.set_training_dataset_info(
            n_total=n, n_train=X_tr.shape[0], n_val=n_val,
            input_dim=X.shape[1], output_dim=3
        )
        self.exp_logger.set_training_model_info(
            architecture="MLP(in->256->128->64->3)",
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
                xb = Xb[i : i + self.batch_size].to(device)
                yb = Yb[i : i + self.batch_size].to(device)
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
            "x_mean": torch.from_numpy(x_mean),
            "x_std": torch.from_numpy(x_std),
            "y_mean": torch.from_numpy(y_mean),
            "y_std": torch.from_numpy(y_std),
            "in_dim": X.shape[1],
            "seed": self.seed,
        }
        tmp = self.model_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.model_path)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        train_elapsed = time.time() - self.train_start
        
        # Logowanie zakończenia treningu
        final_train_loss = history["epochs"][-1]["train_loss"] if history["epochs"] else 0.0
        early_stopped = wait >= self.patience
        best_epoch_idx = max((i for i, e in enumerate(history["epochs"]) if e["val_loss"] <= best_val + self.min_delta), default=0) + 1
        
        self.exp_logger.end_training(
            epochs_run=len(history["epochs"]),
            best_epoch=best_epoch_idx,
            best_val_loss=best_val,
            final_train_loss=final_train_loss,
            early_stopped=early_stopped,
            model_path=self.model_path,
            history_path=self.history_path
        )
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"[FAZA 2] TRENING MODELU AI - KONIEC (trening: {train_elapsed:.1f}s)")
        self.get_logger().info(f"Model zapisany: {self.model_path}")
        self.get_logger().info(f"Najlepszy val_loss: {best_val:.6f} | Epoki: {len(history['epochs'])}")
        self.get_logger().info(f"Metadata: {os.path.join(self.out_dir, 'experiment_metadata.json')}")
        self.get_logger().info("="*60)
        rclpy.shutdown()

    def _save_zero_model(self):
        in_dim = 363
        model = MLP(in_dim=in_dim, out_dim=3)
        x_mean = np.zeros((in_dim,), dtype=np.float32)
        x_std = np.ones((in_dim,), dtype=np.float32)
        y_mean = np.zeros((3,), dtype=np.float32)
        y_std = np.ones((3,), dtype=np.float32)
        payload = {
            "state_dict": model.state_dict(),
            "x_mean": torch.from_numpy(x_mean),
            "x_std": torch.from_numpy(x_std),
            "y_mean": torch.from_numpy(y_mean),
            "y_std": torch.from_numpy(y_std),
            "in_dim": in_dim,
            "seed": self.seed,
        }
        tmp = self.model_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.model_path)


def main():
    rclpy.init()
    node = TrainModel()
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
