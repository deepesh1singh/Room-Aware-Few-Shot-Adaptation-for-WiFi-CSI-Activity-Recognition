"""1-D temporal CNN encoder for (T, 256) CSI windows (subcarriers = channels, time = sequence).

The embedding is [mean, std] pooling over time of the last conv block followed by a linear layer, so
it accepts windows of any length >= 1 (needed for the window-length study) and always yields embed_dim values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CSIEncoder(nn.Module):
    def __init__(self, in_features: int = 256, widths=(128, 128, 192, 256),
                 kernels=(9, 7, 5, 3), strides=(2, 2, 2, 1), embed_dim: int = 128):
        super().__init__()
        layers, c_in = [], in_features
        for w, k, s in zip(widths, kernels, strides):
            layers += [nn.Conv1d(c_in, w, kernel_size=k, stride=s, padding=k // 2, bias=False),
                       nn.BatchNorm1d(w), nn.GELU()]
            c_in = w
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(2 * c_in, embed_dim)
        self.config = {"in_features": in_features, "widths": list(widths), "kernels": list(kernels),
                       "strides": list(strides), "embed_dim": embed_dim}

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        h = self.backbone(x.transpose(1, 2))  # (B, C, T')
        mean = h.mean(dim=2)
        std = torch.sqrt(h.var(dim=2, unbiased=False) + 1e-6)
        return self.proj(torch.cat([mean, std], dim=1))


class ProjectionHead(nn.Module):
    """Used only during contrastive pre-training and discarded afterwards."""

    def __init__(self, embed_dim: int = 128, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_encoder(cfg: dict) -> CSIEncoder:
    ec = cfg.get("encoder", {}) or {}
    return CSIEncoder(
        in_features=int((cfg.get("data", {}) or {}).get("n_subcarriers", 256)),
        widths=tuple(ec.get("widths", (128, 128, 192, 256))),
        kernels=tuple(ec.get("kernels", (9, 7, 5, 3))),
        strides=tuple(ec.get("strides", (2, 2, 2, 1))),
        embed_dim=int(ec.get("embed_dim", 128)),
    )


def build_random_encoder(cfg: dict, seed: int) -> CSIEncoder:
    """Same architecture, random (untrained) weights - the Task-2 control."""
    torch.manual_seed(int(seed))
    return build_encoder(cfg).eval()


def save_encoder(encoder: CSIEncoder, path, history=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder.state_dict(), "config": encoder.config,
                "history": list(history or [])}, path)


def load_encoder(path, device: str = "cpu") -> CSIEncoder:
    ckpt = torch.load(path, map_location="cpu")
    c = ckpt["config"]
    enc = CSIEncoder(in_features=c["in_features"], widths=tuple(c["widths"]), kernels=tuple(c["kernels"]),
                     strides=tuple(c["strides"]), embed_dim=c["embed_dim"])
    enc.load_state_dict(ckpt["state_dict"])
    return enc.to(device).eval()


@torch.no_grad()
def encode(encoder: CSIEncoder, X: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Embed normalised windows (N, T, F) -> float32 (N, embed_dim) with the encoder frozen (eval mode)."""
    encoder = encoder.to(device).eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(encoder(xb).cpu().numpy())
    if not out:
        return np.zeros((0, encoder.config["embed_dim"]), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)
