"""Self-supervised contrastive pre-training of the CSI encoder (Task 2).

SimCLR-style NT-Xent on two stochastic views of the same window. Only development windows of BOTH rooms
are used, pooled, with NO activity labels and NO room labels (the loop below never sees them).

Augmentations (all label-free, applied on normalised windows):
  random temporal crop | static-component removal (subtract the temporal mean -> invariance to the static,
  room-specific multipath) | global gain offset + slow spectral tilt | contiguous subcarrier-band masking |
  contiguous time masking | Gaussian noise.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .config import cget
from .encoder import ProjectionHead, build_encoder
from .utils import set_seed


def augment_batch(x: torch.Tensor, aug: dict, crop_len: int) -> torch.Tensor:
    """x: (B, T, F) normalised windows -> augmented view (B, crop_len, F)."""
    B, T, F = x.shape
    dev = x.device
    crop_len = min(int(crop_len), T)

    starts = torch.randint(0, T - crop_len + 1, (B,), device=dev)
    idx = starts[:, None] + torch.arange(crop_len, device=dev)[None, :]
    x = x[torch.arange(B, device=dev)[:, None], idx]  # (B, crop_len, F)

    p_static = float(aug.get("static_removal_prob", 0.5))
    if p_static > 0:
        m = (torch.rand(B, 1, 1, device=dev) < p_static).float()
        x = x - m * x.mean(dim=1, keepdim=True)

    off = torch.randn(B, 1, 1, device=dev) * float(aug.get("offset_std", 0.3))
    tilt = (torch.randn(B, 1, 1, device=dev) * float(aug.get("tilt_std", 0.3))
            * torch.linspace(-1.0, 1.0, F, device=dev)[None, None, :])
    x = x + off + tilt

    p_band = float(aug.get("band_mask_prob", 0.6))
    if p_band > 0:
        wmin, wmax = int(aug.get("band_mask_min", 8)), int(aug.get("band_mask_max", 48))
        width = torch.randint(wmin, wmax + 1, (B, 1), device=dev)
        start = (torch.rand(B, 1, device=dev) * (F - width)).long()
        fidx = torch.arange(F, device=dev)[None, :]
        band = (fidx >= start) & (fidx < start + width)
        active = torch.rand(B, 1, device=dev) < p_band
        x = x.masked_fill((band & active)[:, None, :], 0.0)

    p_time = float(aug.get("time_mask_prob", 0.5))
    if p_time > 0:
        tmax = max(1, int(aug.get("time_mask_max", 24)))
        tw = torch.randint(1, tmax + 1, (B, 1), device=dev)
        ts = (torch.rand(B, 1, device=dev) * (crop_len - tw).clamp(min=1)).long()
        tidx = torch.arange(crop_len, device=dev)[None, :]
        tband = (tidx >= ts) & (tidx < ts + tw)
        tactive = torch.rand(B, 1, device=dev) < p_time
        x = x.masked_fill((tband & tactive)[:, :, None], 0.0)

    return x + torch.randn_like(x) * float(aug.get("noise_std", 0.05))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = torch.cat([Fn.normalize(z1, dim=1), Fn.normalize(z2, dim=1)], dim=0)
    n = z1.shape[0]
    sim = (z @ z.t()) / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return Fn.cross_entropy(sim, targets)


def train_ssl(X: np.ndarray, cfg: dict, device: str, logger):
    """Train on normalised development windows X (N, T, F). Returns (encoder.eval() on CPU, loss history)."""
    sc = cfg.get("ssl", {}) or {}
    epochs = int(sc.get("epochs", 60))
    bs = min(int(sc.get("batch_size", 128)), X.shape[0])
    lr = float(sc.get("lr", 1e-3))
    wd = float(sc.get("weight_decay", 1e-4))
    temp = float(sc.get("temperature", 0.2))
    crop_len = int(sc.get("crop_len", 120))
    aug = sc.get("augment", {}) or {}
    seed = int(cget(cfg, "seed", 42))

    set_seed(seed)
    enc = build_encoder(cfg).to(device)
    head = ProjectionHead(embed_dim=enc.config["embed_dim"]).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, X.shape[0] // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))

    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    gen = torch.Generator().manual_seed(seed)
    history = []
    logger.info("SSL pre-training: %d windows, %d epochs, batch %d, device %s", X.shape[0], epochs, bs, device)
    for epoch in range(1, epochs + 1):
        enc.train()
        head.train()
        perm = torch.randperm(X.shape[0], generator=gen)
        losses = []
        for step in range(steps_per_epoch):
            xb = Xt[perm[step * bs:(step + 1) * bs]].to(device)
            v1, v2 = augment_batch(xb, aug, crop_len), augment_batch(xb, aug, crop_len)
            loss = nt_xent(head(enc(v1)), head(enc(v2)), temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info("epoch %3d/%d  NT-Xent loss %.4f  (chance level %.2f)", epoch, epochs, history[-1],
                        math.log(2 * bs - 1))
    return enc.cpu().eval(), history
