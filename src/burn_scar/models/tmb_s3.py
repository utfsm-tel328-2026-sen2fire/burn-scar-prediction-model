"""
Generador de cicatriz congelado TMB-S3 (ConvLSTM + SegFormer, 5 bandas).

Implementa la arquitectura oficial de TMB-S3 (links-ads/tmb-s3) para producir
el canal de cicatriz conocida `scar` en el frame f:
- Entrada: (B, T, 6, 64, 64) -> 5 bandas OLCI (Oa04, Oa06, Oa08, Oa17, Oa21) + landcover
- Salida:  (B, T, 2, 64, 64) -> logits de no quemado (0) y quemado (1)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch import Tensor

from burn_scar.models.convlstm import ConvLSTMCell, TemporalContextModel
from burn_scar.models.segformer import MiniSegFormer

DEFAULT_WEIGHTS_PATH = Path("models/tmb_s3/convlstm_segformer_prepost_subset/model.safetensors")

__all__ = [
    "ConvLSTMCell",
    "MiniSegFormer",
    "TemporalContextModel",
    "load_tmb_s3",
    "predict_burn_scar",
]


def load_tmb_s3(
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> TemporalContextModel:
    """Carga e instancia el modelo TMB-S3 congelado con pesos preentrenados."""
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS_PATH

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No se encontraron los pesos de TMB-S3 en: {weights_path}. "
            "Asegúrate de haber descargado el checkpoint desde links-ads/tmb-s3."
        )

    # MiniSegFormer configurado para 5 bandas + landcover (in_channels = 6 + hidden_ch 64 = 70)
    backbone = MiniSegFormer(
        in_channels=70,
        num_classes=2,
        embed_dims=(32, 64, 160),
        num_heads=(1, 2, 5),
        sr_ratios=(4, 2, 1),
        depths=(2, 2, 2),
        decoder_dim=128,
    )
    model = TemporalContextModel(
        backbone=backbone,
        in_channels=6,
        hidden_ch=64,
        kernel_size=3,
    )

    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Congelar parámetros (el generador de cicatriz no se reentrena en propagación)
    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def predict_burn_scar(model: nn.Module, x: Tensor) -> Tensor:
    """
    Produce el mapa de probabilidad de quemado [0..1] para cada timestep.

    Args:
        model: modelo TMB-S3 cargado
        x: tensor (B, T, 6, H, W) o (T, 6, H, W)
    Retorna:
        Tensor de probabilidades con forma (B, T, H, W) o (T, H, W)
    """
    is_batched = (x.ndim == 5)
    if not is_batched:
        x = x.unsqueeze(0)

    logits = model(x)  # (B, T, 2, H, W)
    # Canal 0 = logit de quemado (upstream: _FIRE_CH=0). El canal 1 es la cabeza de nubes.
    probs = torch.sigmoid(logits[:, :, 0])

    return probs if is_batched else probs.squeeze(0)
