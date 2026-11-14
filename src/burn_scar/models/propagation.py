"""
Modelo de predicción de propagación de cicatrices de incendio (ConvLSTM + SegFormer).

Implementa la arquitectura descrita en docs/operacion.md §4:
- Encoder temporal ConvLSTM sobre la secuencia de 7 frames Sentinel-3 OLCI (5 bandas).
- Fusión con capas estáticas (DEM, WorldCover), cicatriz conocida generada por TMB-S3
  congelado y horizonte explícito h (en días).
- Decoder espacial MiniSegFormer con cabeza de propagación binaria (logits por píxel).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from burn_scar.models.convlstm import ConvLSTMCell
from burn_scar.models.segformer import MiniSegFormer


def corrupt_scar(x_scar: Tensor, drop_prob: float = 0.15) -> Tensor:
    """
    Aplica ruido de corrupción sobre la cicatriz de entrada durante el entrenamiento.

    Borra aleatoriamente ~`drop_prob` de los píxeles quemados y aplica dilatación o
    erosión morfológica de 1 píxel en el borde (docs/operacion.md §4).
    """
    if drop_prob <= 0.0:
        return x_scar

    out = x_scar.clone()
    # 1. Dropout aleatorio de píxeles
    keep_mask = (torch.rand_like(out) >= drop_prob).to(out.dtype)
    out = out * keep_mask

    # 2. Dilatación (50%) o erosión (50%) de 1 píxel por muestra en el batch
    B = out.shape[0]
    for i in range(B):
        patch = out[i : i + 1]
        if torch.rand(1, device=out.device).item() < 0.5:
            out[i : i + 1] = F.max_pool2d(patch, kernel_size=3, stride=1, padding=1)
        else:
            out[i : i + 1] = -F.max_pool2d(-patch, kernel_size=3, stride=1, padding=1)

    return out.clamp(0.0, 1.0)


class BurnScarPropagationModel(nn.Module):
    """
    Predice el mapa de probabilidad de nueva quema en (f, f + h].

    Entradas:
        x_olci:       (B, T, 5, H, W)
        x_dem:        (B, 1, H, W)
        x_worldcover: (B, 1, H, W)
        x_scar:       (B, 1, H, W)
        horizon_days: (B, 1) o (B,)

    Salida:
        logits:       (B, 1, H, W)
    """

    def __init__(
        self,
        olci_channels: int = 5,
        hidden_ch: int = 32,
        scar_corruption_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.scar_corruption_prob = scar_corruption_prob

        self.temporal_encoder = ConvLSTMCell(
            in_channels=olci_channels,
            hidden_ch=hidden_ch,
            kernel_size=3,
        )

        # Canales espaciales fusionados:
        # hidden_ch (OLCI temporal) + 1 (DEM) + 1 (WorldCover) + 1 (scar) + 1 (horizon)
        fused_channels = hidden_ch + 4
        self.spatial_head = MiniSegFormer(
            in_channels=fused_channels,
            num_classes=1,
            embed_dims=(32, 64, 128),
            num_heads=(1, 2, 4),
            sr_ratios=(4, 2, 1),
            depths=(2, 2, 2),
            decoder_dim=64,
        )

    def forward(
        self,
        x_olci: Tensor,
        x_dem: Tensor,
        x_worldcover: Tensor,
        x_scar: Tensor,
        horizon_days: Tensor,
    ) -> Tensor:
        B, T, _, H, W = x_olci.shape

        if self.training and self.scar_corruption_prob > 0.0:
            x_scar = corrupt_scar(x_scar, drop_prob=self.scar_corruption_prob)

        # 1. Codificar secuencia temporal OLCI con ConvLSTM
        h = torch.zeros(B, self.hidden_ch, H, W, device=x_olci.device, dtype=x_olci.dtype)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.temporal_encoder(x_olci[:, t], h, c)

        # 2. Proyectar horizonte explícito h (log1p normalizado a [0, ~1]) como canal espacial
        if horizon_days.ndim == 1:
            horizon_days = horizon_days.unsqueeze(1)
        h_scaled = torch.log1p(horizon_days.clamp(min=0.0)) / 2.5
        h_map = h_scaled.view(B, 1, 1, 1).expand(B, 1, H, W)

        # 3. Fusionar contexto temporal + estáticas + cicatriz + horizonte -> logits (B, 1, H, W)
        fused = torch.cat([h, x_dem, x_worldcover, x_scar, h_map], dim=1)
        return self.spatial_head(fused)


def masked_bce_dice_loss(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    pos_weight: float = 4.0,
    dice_weight: float = 0.5,
) -> Tensor:
    """
    Pérdida BCEWithLogits + Soft Dice calculada solo en píxeles válidos (mask == 1).
    """
    valid = (mask == 1).float()
    valid_count = valid.sum().clamp(min=1.0)

    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce_per_px = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none"
    )
    bce = (bce_per_px * valid).sum() / valid_count

    probs = torch.sigmoid(logits) * valid
    tgt_valid = target * valid
    intersection = (probs * tgt_valid).sum()
    union = probs.sum() + tgt_valid.sum()
    dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

    return bce + dice_weight * dice_loss
