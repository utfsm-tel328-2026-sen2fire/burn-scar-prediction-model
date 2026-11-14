"""Tests unitarios para el modelo de propagación y su pérdida enmascarada."""

from __future__ import annotations

import torch

from burn_scar.models import BurnScarPropagationModel, corrupt_scar, masked_bce_dice_loss


def test_corrupt_scar_bounds_and_shape() -> None:
    scar = torch.zeros(2, 1, 64, 64)
    scar[:, :, 20:40, 20:40] = 1.0
    corrupted = corrupt_scar(scar, drop_prob=0.15)
    assert corrupted.shape == scar.shape
    assert corrupted.min() >= 0.0
    assert corrupted.max() <= 1.0


def test_propagation_model_forward_and_backward() -> None:
    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=16, scar_corruption_prob=0.15)
    model.train()

    B, T, H, W = 2, 7, 64, 64
    x_olci = torch.randn(B, T, 5, H, W)
    x_dem = torch.rand(B, 1, H, W)
    x_wc = torch.rand(B, 1, H, W)
    x_scar = torch.zeros(B, 1, H, W)
    x_scar[:, :, 24:36, 24:36] = 1.0
    horizon_days = torch.tensor([[1.0], [7.0]])

    logits = model(x_olci, x_dem, x_wc, x_scar, horizon_days)
    assert logits.shape == (B, 1, H, W)

    target = torch.zeros(B, 1, H, W)
    target[:, :, 20:40, 20:40] = 1.0
    mask = torch.ones(B, 1, H, W, dtype=torch.uint8)
    mask[:, :, :5, :5] = 255

    loss = masked_bce_dice_loss(logits, target, mask)
    assert torch.isfinite(loss)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
