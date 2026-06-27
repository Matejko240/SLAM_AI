from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def parse_hidden_dims(value) -> List[int]:
    if value is None:
        return [192, 96, 48]
    dims = [int(v) for v in list(value) if int(v) > 0]
    return dims if dims else [192, 96, 48]


def normalize_model_type(value: str) -> str:
    model_type = str(value or "").strip().lower()
    return model_type or "cnn"


def normalize_target_scaling(value: str) -> str:
    scaling = str(value or "").strip().lower()
    return scaling or "zscore"


def model_type_from_payload(payload: dict) -> str:
    raw_model_type = str(payload.get("model_type", "")).strip().lower()
    if raw_model_type:
        return normalize_model_type(raw_model_type)
    architecture = str(payload.get("architecture", ""))
    if architecture.startswith("RywakGRU"):
        return "gru"
    if architecture.startswith("RywakLSTM"):
        return "lstm"
    if architecture.startswith("MLP2"):
        return "mlp2"
    return "cnn"


def output_activation_for_target_scaling(target_scaling: str) -> str:
    return "tanh" if normalize_target_scaling(target_scaling) == "tanh" else "linear"


def _make_output_activation(output_activation: str) -> nn.Module:
    if str(output_activation).strip().lower() == "tanh":
        return nn.Tanh()
    return nn.Identity()


class RywakCNN(nn.Module):
    """Single-frame CNN variant used in the current Rywak pipeline."""

    def __init__(
        self,
        in_dim: int = 362,
        out_dim: int = 2,
        dropout: float = 0.0,
        output_activation: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.in_dim)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel()
        head_layers = [nn.Flatten(), nn.Linear(flat_dim, 32), nn.ReLU()]
        if dropout > 0.0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(32, int(out_dim)))
        head_layers.append(_make_output_activation(output_activation))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.conv(x))


class RywakGRU(nn.Module):
    """Faithful stacked-GRU variant inspired by the original Rywak repository."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        sequence_length: int = 4,
        hidden_size: int = 40,
        dropout: float = 0.0,
        output_activation: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.sequence_length = max(1, int(sequence_length))
        self.hidden_size = int(hidden_size)
        self.gru1 = nn.GRU(self.in_dim, self.hidden_size, batch_first=True)
        self.gru2 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        head_layers = [nn.Linear(self.hidden_size * self.sequence_length, int(out_dim))]
        head_layers.append(_make_output_activation(output_activation))
        self.head = nn.Sequential(*head_layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru1(x)
        out = self.dropout(out)
        out, _ = self.gru2(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        return self.head(out)


class RywakLSTM(nn.Module):
    """Faithful stacked-LSTM variant inspired by the original Rywak repository."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        sequence_length: int = 4,
        hidden_size: int = 40,
        dropout: float = 0.0,
        output_activation: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.sequence_length = max(1, int(sequence_length))
        self.hidden_size = int(hidden_size)
        self.lstm1 = nn.LSTM(self.in_dim, self.hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        head_layers = [nn.Linear(self.hidden_size, int(out_dim))]
        head_layers.append(_make_output_activation(output_activation))
        self.head = nn.Sequential(*head_layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        return self.head(out)


class MLP2(nn.Module):
    """Legacy MLP kept for backward compatibility with older checkpoints."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        hidden_dims: List[int] = None,
        dropout: float = 0.0,
        output_activation: str = "linear",
        **kwargs,
    ):
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
        layers.append(_make_output_activation(output_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_rywak_model(
    *,
    model_type: str,
    in_dim: int,
    out_dim: int,
    hidden_dims: List[int],
    dropout: float,
    sequence_length: int,
    output_activation: str,
):
    model_type = normalize_model_type(model_type)
    common_kwargs = dict(
        in_dim=int(in_dim),
        out_dim=int(out_dim),
        dropout=float(dropout),
        sequence_length=max(1, int(sequence_length)),
        output_activation=output_activation,
    )
    if model_type == "cnn":
        return RywakCNN(**common_kwargs)
    if model_type == "gru":
        return RywakGRU(**common_kwargs)
    if model_type == "lstm":
        return RywakLSTM(**common_kwargs)
    if model_type == "mlp2":
        return MLP2(hidden_dims=hidden_dims, **common_kwargs)
    raise ValueError(f"Unsupported Rywak model_type={model_type}")


def build_tanh_target_meta(
    y_train: np.ndarray,
    *,
    gamma: float,
    v_min: float,
    v_max: float,
    w_min: float,
    w_max: float,
) -> Dict[str, float]:
    y_train = np.asarray(y_train, dtype=np.float32)
    if y_train.ndim != 2 or y_train.shape[1] < 2:
        raise ValueError("Rywak tanh target scaling expects Y shape (N,2+) with velocity labels.")

    def _resolve_range(values: np.ndarray, cfg_min: float, cfg_max: float) -> tuple[float, float]:
        lo = float(cfg_min)
        hi = float(cfg_max)
        if hi > lo:
            return lo, hi
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        pad = max(0.05 * (hi - lo), 1e-3)
        return lo - pad, hi + pad

    v_lo, v_hi = _resolve_range(y_train[:, 0], v_min, v_max)
    w_lo, w_hi = _resolve_range(y_train[:, 1], w_min, w_max)
    gamma = max(1e-3, float(gamma))
    return {
        "gamma": gamma,
        "v_min": v_lo,
        "v_max": v_hi,
        "w_min": w_lo,
        "w_max": w_hi,
    }


def _scale_range_np(values: np.ndarray, vmin: float, vmax: float, gamma: float) -> np.ndarray:
    denom = max(float(vmax) - float(vmin), 1e-6)
    scaled = (2.0 * gamma * (values - float(vmin)) / denom) - gamma
    return np.clip(scaled, -1.0, 1.0)


def _scale_range_torch(values: torch.Tensor, vmin: float, vmax: float, gamma: float) -> torch.Tensor:
    denom = max(float(vmax) - float(vmin), 1e-6)
    scaled = (2.0 * float(gamma) * (values - float(vmin)) / denom) - float(gamma)
    return torch.clamp(scaled, min=-1.0, max=1.0)


def _unscale_range_np(values: np.ndarray, vmin: float, vmax: float, gamma: float) -> np.ndarray:
    denom = max(2.0 * float(gamma), 1e-6)
    unit = (values + float(gamma)) / denom
    return unit * (float(vmax) - float(vmin)) + float(vmin)


def _unscale_range_torch(values: torch.Tensor, vmin: float, vmax: float, gamma: float) -> torch.Tensor:
    denom = max(2.0 * float(gamma), 1e-6)
    unit = (values + float(gamma)) / denom
    return unit * (float(vmax) - float(vmin)) + float(vmin)


def scale_targets_for_model_np(
    y: np.ndarray,
    *,
    target_scaling: str,
    y_mean: Optional[np.ndarray],
    y_std: Optional[np.ndarray],
    target_tanh_meta: Optional[Dict[str, float]],
) -> np.ndarray:
    target_scaling = normalize_target_scaling(target_scaling)
    y = np.asarray(y, dtype=np.float32)
    if target_scaling == "tanh":
        if target_tanh_meta is None:
            raise ValueError("target_tanh_meta is required for tanh target scaling.")
        scaled = y.copy()
        gamma = float(target_tanh_meta["gamma"])
        scaled[:, 0] = _scale_range_np(
            y[:, 0],
            float(target_tanh_meta["v_min"]),
            float(target_tanh_meta["v_max"]),
            gamma,
        )
        scaled[:, 1] = _scale_range_np(
            y[:, 1],
            float(target_tanh_meta["w_min"]),
            float(target_tanh_meta["w_max"]),
            gamma,
        )
        return scaled.astype(np.float32)
    if y_mean is None or y_std is None:
        raise ValueError("y_mean and y_std are required for zscore target scaling.")
    return ((y - y_mean) / y_std).astype(np.float32)


def unscale_targets_from_model_np(
    y_model: np.ndarray,
    *,
    target_scaling: str,
    y_mean: Optional[np.ndarray],
    y_std: Optional[np.ndarray],
    target_tanh_meta: Optional[Dict[str, float]],
) -> np.ndarray:
    target_scaling = normalize_target_scaling(target_scaling)
    y_model = np.asarray(y_model, dtype=np.float32)
    if target_scaling == "tanh":
        if target_tanh_meta is None:
            raise ValueError("target_tanh_meta is required for tanh target scaling.")
        raw = y_model.copy()
        gamma = float(target_tanh_meta["gamma"])
        raw[:, 0] = _unscale_range_np(
            y_model[:, 0],
            float(target_tanh_meta["v_min"]),
            float(target_tanh_meta["v_max"]),
            gamma,
        )
        raw[:, 1] = _unscale_range_np(
            y_model[:, 1],
            float(target_tanh_meta["w_min"]),
            float(target_tanh_meta["w_max"]),
            gamma,
        )
        return raw.astype(np.float32)
    if y_mean is None or y_std is None:
        raise ValueError("y_mean and y_std are required for zscore target scaling.")
    return (y_model * y_std + y_mean).astype(np.float32)


def unscale_targets_from_model_torch(
    y_model: torch.Tensor,
    *,
    target_scaling: str,
    y_mean_t: Optional[torch.Tensor],
    y_std_t: Optional[torch.Tensor],
    target_tanh_meta: Optional[Dict[str, float]],
) -> torch.Tensor:
    target_scaling = normalize_target_scaling(target_scaling)
    if target_scaling == "tanh":
        if target_tanh_meta is None:
            raise ValueError("target_tanh_meta is required for tanh target scaling.")
        raw = y_model.clone()
        gamma = float(target_tanh_meta["gamma"])
        raw[..., 0] = _unscale_range_torch(
            y_model[..., 0],
            float(target_tanh_meta["v_min"]),
            float(target_tanh_meta["v_max"]),
            gamma,
        )
        raw[..., 1] = _unscale_range_torch(
            y_model[..., 1],
            float(target_tanh_meta["w_min"]),
            float(target_tanh_meta["w_max"]),
            gamma,
        )
        return raw
    if y_mean_t is None or y_std_t is None:
        raise ValueError("y_mean_t and y_std_t are required for zscore target scaling.")
    return y_model * y_std_t + y_mean_t
