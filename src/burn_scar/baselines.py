"""
Modelos de Referencia (Baselines B0 y B1) y Métricas de Evaluación.

Implementa los baselines sin aprendizaje automático estipulados en docs/dataset.md §9:
- B0 (Persistencia): el fuego no crece (delta = 0).
- B1 (Dilatación radial): el frente avanza de forma isótropa.
- Métricas oficiales de evaluación: F1 score, IoU, Precision, Recall calculados
  exclusivamente en píxeles válidos (mask == 1) y promediados por evento (D3, docs/dataset.md §10).
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import scipy.ndimage as ndi
import torch
from torch import Tensor


def compute_metrics_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula métricas binarias sobre los píxeles válidos (mask == 1).
    
    pred:   array continuo o de probabilidades en [0..1]
    target: array de valores en [0..1] (lo nuevo quemado)
    mask:   array de validez (1 = válido, 255 = ignorar)
    """
    # Filtrar solo píxeles válidos
    valid_idx = (mask == 1)
    if not np.any(valid_idx):
        return {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "valid_pixels": 0}

    y_pred = (pred[valid_idx] >= threshold).astype(bool)
    y_true = (target[valid_idx] >= threshold).astype(bool)

    tp = np.logical_and(y_pred, y_true).sum()
    fp = np.logical_and(y_pred, ~y_true).sum()
    fn = np.logical_and(~y_pred, y_true).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if y_true.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if y_true.sum() == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if y_true.sum() == 0 else 0.0)

    return {
        "f1": float(f1),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "valid_pixels": int(valid_idx.sum()),
    }


class BaselineB0Persistence:
    """B0: El fuego no crece (predice delta = 0 en todo el mapa)."""

    def predict(self, x_scar: Tensor | np.ndarray, horizon_days: float | Tensor = 1.0) -> np.ndarray:
        if isinstance(x_scar, Tensor):
            x_scar = x_scar.cpu().numpy()
        # Devuelve ceros con la forma de la cicatriz
        return np.zeros_like(x_scar, dtype=np.float32)


class BaselineB1RadialDilation:
    """B1: Dilatación radial isótropa de la cicatriz previa según el horizonte h."""

    def __init__(self, growth_rate_km_per_day: float = 0.6) -> None:
        self.rate_px_per_day = (growth_rate_km_per_day / 0.3)  # a 300 m/px -> 2 px/día

    def predict(self, x_scar: Tensor | np.ndarray, horizon_days: float | Tensor = 1.0) -> np.ndarray:
        if isinstance(x_scar, Tensor):
            x_scar = x_scar.cpu().numpy()
        if isinstance(horizon_days, Tensor):
            horizon_days = horizon_days.item()

        # Binarizar cicatriz previa a 0.5
        scar_binary = (x_scar >= 0.5)
        if not np.any(scar_binary):
            return np.zeros_like(x_scar, dtype=np.float32)

        # Radio de dilatación en píxeles proporcional a h
        radius_px = max(1, int(round(self.rate_px_per_day * horizon_days)))
        struct = ndi.generate_binary_structure(2, 2)  # 8-conectividad
        dilated = ndi.binary_dilation(scar_binary, structure=struct, iterations=radius_px)

        # Delta predicho = lo nuevo en la dilatación
        pred_delta = np.logical_and(dilated, ~scar_binary).astype(np.float32)
        return pred_delta
