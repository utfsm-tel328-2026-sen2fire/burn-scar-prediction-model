"""Paquete principal para predicción de crecimiento de cicatrices de incendio."""

from burn_scar.baselines import (
    BaselineB0Persistence,
    BaselineB1RadialDilation,
    compute_metrics_numpy,
)
from burn_scar.dataset import (
    BurnScarDataset,
    build_temporal_window,
    get_cv_dataloaders,
    match_frame_key,
)

__all__ = [
    "BurnScarDataset",
    "build_temporal_window",
    "get_cv_dataloaders",
    "match_frame_key",
    "BaselineB0Persistence",
    "BaselineB1RadialDilation",
    "compute_metrics_numpy",
]
