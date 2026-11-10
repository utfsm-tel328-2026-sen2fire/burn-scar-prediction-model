"""Paquete principal para predicción de crecimiento de cicatrices de incendio."""

from burn_scar.dataset import BurnScarDataset, get_cv_dataloaders
from burn_scar.baselines import (
    BaselineB0Persistence,
    BaselineB1RadialDilation,
    compute_metrics_numpy,
)

__all__ = [
    "BurnScarDataset",
    "get_cv_dataloaders",
    "BaselineB0Persistence",
    "BaselineB1RadialDilation",
    "compute_metrics_numpy",
]
