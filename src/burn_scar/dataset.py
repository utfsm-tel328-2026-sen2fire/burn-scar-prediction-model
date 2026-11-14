"""
Dataset PyTorch para Predicción de Propagación de Cicatrices (BurnScarDataset).

Carga las muestras empaquetadas en data/processed/samples/ con soporte
para particionado por Folds de Validación Cruzada (0..4 y -1).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_INDEX_PATH = Path("data/processed/dataset_index.json")


def resolve_loader_options(
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[int, bool]:
    """
    Decide (num_workers, pin_memory) según el backend de GPU.

    En ROCm (AMD) `pin_memory=True` con workers es patológico: cada lote
    reserva memoria fijada con hipHostMalloc (cientos de ms por tensor) y el
    fork de un proceso con contexto HIP inicializado cuesta segundos. Con este
    dataset (≈2 ms por muestra) los workers no aportan nada, así que en AMD
    se desactivan. En CUDA se mantienen los valores habituales.
    """
    is_rocm = bool(getattr(torch.version, "hip", None))
    if num_workers is None:
        num_workers = 0 if is_rocm else 2
    if pin_memory is None:
        pin_memory = False if is_rocm else torch.cuda.is_available()
    return int(num_workers), bool(pin_memory)


class BurnScarDataset(Dataset):
    """
    Dataset PyTorch para entrenamiento y evaluación multi-horizonte.
    
    Retorna por muestra:
        x_olci:       (7, 5, 64, 64) float32 (secuencia de 7 frames con 5 bandas espectrales)
        x_dem:        (1, 64, 64)    float32 (elevación normalizada)
        x_worldcover: (1, 64, 64)    float32 (clases de cobertura de suelo normalizadas)
        x_scar:       (1, 64, 64)    float32 (cicatriz conocida producida por TMB-S3 en el frame actual f)
        horizon_days: (1,)           float32 (horizonte h en días hacia el objetivo)
        target:       (1, 64, 64)    float32 (lo NUEVO quemado en [f, f+h], valores en [0..1])
        mask:         (1, 64, 64)    uint8   (1 = píxel válido observado, 255 = ignorar/no analizado)
        meta:         dict           (sample_id, series_id, fold, etc.)
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        folds: Sequence[int] | int | None = None,
        transform=None,
    ) -> None:
        super().__init__()
        if index_path is None:
            index_path = DEFAULT_INDEX_PATH
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"No se encontró el índice del dataset en: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        all_samples = index_data.get("samples", [])
        if folds is not None:
            if isinstance(folds, int):
                folds = [folds]
            allowed_folds = set(folds)
            self.samples = [s for s in all_samples if s["fold"] in allowed_folds]
        else:
            self.samples = all_samples

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        info = self.samples[idx]
        file_path = info["file"]

        data = np.load(file_path)

        # 1. OLCI: normalizar radiancia (dividir entre 1000.0) -> shape (7, 5, 64, 64)
        x_olci = torch.from_numpy(data["x_olci"].astype(np.float32) / 1000.0)

        # 2. DEM: normalizar elevación en km (dividir entre 1000.0) -> shape (1, 64, 64)
        x_dem = torch.from_numpy(data["x_dem"].astype(np.float32) / 1000.0).unsqueeze(0)

        # 3. WorldCover: normalizar categorías -> shape (1, 64, 64)
        x_wc = torch.from_numpy(data["x_worldcover"].astype(np.float32) / 100.0).unsqueeze(0)

        # 4. Cicatriz TMB-S3 en f -> shape (1, 64, 64)
        x_scar = torch.from_numpy(data["x_scar"].astype(np.float32)).unsqueeze(0)

        # 5. Horizonte h en días -> shape (1,)
        h_days = torch.tensor([float(data["horizon_days"])], dtype=torch.float32)

        # 6. Target: incremento de quemado -> shape (1, 64, 64)
        target = torch.from_numpy(data["target"].astype(np.float32)).unsqueeze(0)

        # 7. Máscara de validez -> shape (1, 64, 64)
        mask = torch.from_numpy(data["mask"].astype(np.uint8)).unsqueeze(0)

        sample = {
            "x_olci": x_olci,
            "x_dem": x_dem,
            "x_worldcover": x_wc,
            "x_scar": x_scar,
            "horizon_days": h_days,
            "target": target,
            "mask": mask,
            "meta": info,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_cv_dataloaders(
    val_fold: int,
    batch_size: int = 8,
    num_workers: int | None = None,
    index_path: str | Path | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Retorna (train_loader, val_loader) para un fold de validación específico (0..4).
    Entrena en los 4 folds restantes y valida en val_fold (D14).

    `num_workers` y `pin_memory` en None se resuelven según el backend
    (ver `resolve_loader_options`).
    """
    if val_fold not in range(5):
        raise ValueError(f"val_fold debe estar en [0..4], recibido: {val_fold}")

    nw, pin = resolve_loader_options(num_workers, pin_memory)

    train_folds = [f for f in range(5) if f != val_fold]
    train_dataset = BurnScarDataset(index_path=index_path, folds=train_folds)
    val_dataset = BurnScarDataset(index_path=index_path, folds=[val_fold])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )

    return train_loader, val_loader
