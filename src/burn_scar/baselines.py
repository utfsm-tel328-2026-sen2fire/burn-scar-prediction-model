"""
Modelos de Referencia (baselines) y Métricas de Evaluación.

Implementa los baselines sin aprendizaje automático estipulados en docs/dataset.md §9:
- Persistencia (BASE-0): el fuego no crece (delta = 0).
- Dilatación radial (BASE-1): el frente avanza de forma isótropa.
- Todo quemado: la escala mínima de cada etiqueta.

Los códigos BASE-n son los de la documentación. Las clases de aquí no los llevan:
- las decisiones de diseño se numeran B1-B13 (docs/decisiones.md §1.1) y son otra
  serie, que no debe confundirse con esta.
- Métricas oficiales de evaluación: F1 score, IoU, Precision, Recall calculados
  exclusivamente en píxeles válidos (mask == 1) y promediados por evento (D3, docs/dataset.md §10).
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
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

    Si la verdad no contiene ningún píxel quemado (target >= threshold), las
    métricas binarias no son informativas: un modelo que predice "nada" obtendría
    un 1.0 artificial. En ese caso se devuelven NaN.
    """
    # Filtrar solo píxeles válidos
    valid_idx = (mask == 1)
    n_valid = int(valid_idx.sum())
    if n_valid == 0:
        return _nan_metrics(0, 0)

    y_pred = (pred[valid_idx] >= threshold).astype(bool)
    y_true = (target[valid_idx] >= threshold).astype(bool)
    n_pos = int(y_true.sum())

    if n_pos == 0:
        return _nan_metrics(n_valid, 0)

    tp = np.logical_and(y_pred, y_true).sum()
    fp = np.logical_and(y_pred, ~y_true).sum()
    fn = np.logical_and(~y_pred, y_true).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "f1": float(f1),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "valid_pixels": n_valid,
        "n_positives": n_pos,
    }


def _nan_metrics(n_valid: int, n_pos: int) -> dict[str, float]:
    """Métricas no informativas (la verdad no tiene quemado o no hay píxeles válidos)."""
    nan = float("nan")
    return {
        "f1": nan,
        "iou": nan,
        "precision": nan,
        "recall": nan,
        "valid_pixels": int(n_valid),
        "n_positives": int(n_pos),
    }


def compact_series_metrics(aggregate: dict[str, float]) -> dict[str, float]:
    """Extrae el resumen público de métricas de un agregado por serie."""
    return {
        "f1": aggregate["f1_event_mean"],
        "iou": aggregate["iou_event_mean"],
        "precision": aggregate["precision_event_mean"],
        "recall": aggregate["recall_event_mean"],
        "n_series_evaluated": aggregate["n_series_evaluated"],
        "n_series_total": aggregate["n_series_total"],
        "n_samples_evaluated": aggregate["n_samples_evaluated"],
        "n_samples_total": aggregate["n_samples_total"],
    }


def aggregate_series_metrics(series_metrics: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    """
    Promedio por serie/evento (D10) ignorando las métricas no informativas (NaN).

    Devuelve además cuántas series y cuántas muestras han podido evaluarse, para
    detectar el caso degenerado en el que el target no contiene quemado (métricas
    sin sentido). Una muestra es evaluable cuando su métrica es informativa.
    """
    keys = ("f1", "iou", "precision", "recall")
    per_series: dict[str, list[float]] = {k: [] for k in keys}

    n_samples_total = 0
    n_samples_evaluated = 0
    for m_list in series_metrics.values():
        n_samples_total += len(m_list)
        for k in keys:
            vals = np.array([m[k] for m in m_list], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size:
                per_series[k].append(float(vals.mean()))
        # Una muestra es evaluable si su F1 es informativo (no NaN).
        f1_vals = np.array([m["f1"] for m in m_list], dtype=float)
        n_samples_evaluated += int((~np.isnan(f1_vals)).sum())

    out = {
        f"{k}_event_mean": (float(np.mean(v)) if v else float("nan"))
        for k, v in per_series.items()
    }
    out["f1_event_std"] = (
        float(np.std(per_series["f1"])) if per_series["f1"] else float("nan")
    )
    out["n_series_evaluated"] = len(per_series["f1"])
    out["n_series_total"] = len(series_metrics)
    out["n_samples_evaluated"] = n_samples_evaluated
    out["n_samples_total"] = n_samples_total
    return out



class BaselinePersistence:
    """BASE-0: El fuego no crece (predice delta = 0 en todo el mapa)."""

    def predict(
        self, x_scar: Tensor | np.ndarray, horizon_days: float | Tensor = 1.0
    ) -> np.ndarray:
        if isinstance(x_scar, Tensor):
            x_scar = x_scar.cpu().numpy()
        # Devuelve ceros con la forma de la cicatriz
        return np.zeros_like(x_scar, dtype=np.float32)


class BaselineAllBurned:
    """
    Trivial: predice que todo el mapa está quemado.

    No es un modelo, sino la **escala mínima** de cada etiqueta: un F1 por
    debajo de este valor no demuestra que nada se haya aprendido. Como el
    target de ensamblaje tiene mucha más superficie que el crecimiento real,
    este trivial puntúa distinto contra cada etiqueta y sirve de referencia
    para no comparar F1 de etiquetas distintas entre sí.
    """

    def predict(
        self, x_scar: Tensor | np.ndarray, horizon_days: float | Tensor = 1.0
    ) -> np.ndarray:
        if isinstance(x_scar, Tensor):
            x_scar = x_scar.cpu().numpy()
        return np.ones_like(x_scar, dtype=np.float32)


class BaselineRadialDilation:
    """BASE-1: Dilatación radial isótropa de la cicatriz previa según el horizonte h."""
    def __init__(self, growth_rate_km_per_day: float = 0.6) -> None:
        self.rate_px_per_day = (growth_rate_km_per_day / 0.3)  # a 300 m/px -> 2 px/día

    def predict(
        self, x_scar: Tensor | np.ndarray, horizon_days: float | Tensor = 1.0
    ) -> np.ndarray:
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
