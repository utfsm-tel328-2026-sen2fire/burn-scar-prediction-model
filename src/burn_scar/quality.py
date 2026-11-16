"""Controles de calidad compartidos por el dataset y la evaluación."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

DATASET_INDEX_SCHEMA_VERSION = 2

# Clases ESA WorldCover v200 aceptadas. El valor 0 no es una clase: representa
# ausencia de dato y nunca debe entrar en el tensor del modelo.
WORLDCOVER_VALID_CLASSES = frozenset({10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100})

HORIZON_BAND_SPECS = (
    ("0-1d", 0.0, 1.0),
    ("1-3d", 1.0, 3.0),
    ("3-7d", 3.0, 7.0),
    ("7-14d", 7.0, 14.0),
    ("14d+", 14.0, float("inf")),
)


class StaticLayerValidationError(ValueError):
    """Las capas estáticas requeridas no son utilizables."""


def file_sha256(path: str | Path) -> str:
    """SHA-256 del contenido de un fichero."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """
    SHA-256 estable de un payload JSON.

    La serialización es determinista (claves ordenadas, separadores fijos) y
    no depende de marcas de tiempo, de modo que dos regeneraciones del mismo
    dataset producen la misma huella de contenido aunque el fichero cambie en
    su cabecera.
    """
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def historical_status(run_dir: str | Path) -> dict[str, Any] | None:
    """
    Estado histórico de un directorio de checkpoints.

    Un run anterior a la corrección del dataset se marca con `historico.json`.
    Sus métricas no son comparables con las del run canónico y no deben
    publicarse como resultado actual.
    """
    marker = Path(run_dir) / "historico.json"
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def describe_evaluated_set(dataset: Any) -> dict[str, int]:
    """
    Descriptivo único del conjunto evaluado.

    Separa los denominadores que antes se confundían: entradas del índice
    completo frente a entradas del split evaluado, y, en cada nivel, el
    recuento de entradas frente al de muestras y ficheros únicos. El modelo y
    los baselines usan este mismo descriptivo, de modo que sus números se
    comparan sobre el mismo conjunto y no sobre agregados equivalentes por
    casualidad.
    """
    samples = list(getattr(dataset, "samples", []) or [])

    def _unique(key: str) -> int:
        return len({
            str(sample.get(key))
            for sample in samples
            if sample.get(key) is not None
        })

    def _attribute(name: str, fallback: int) -> int:
        value = getattr(dataset, name, None)
        return int(value) if value is not None else int(fallback)

    return {
        "index_entries": _attribute("total_entries", len(samples)),
        "index_unique_samples": _attribute("total_unique_samples", len(samples)),
        "index_unique_files": _attribute("total_unique_files", len(samples)),
        "index_series": _attribute("total_series", _unique("series_id")),
        "split_samples": len(samples),
        "split_unique_samples": _unique("sample_id"),
        "split_unique_files": _unique("file"),
        "split_series": _unique("series_id"),
    }


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """SHA-256 estable de los tensores de un ``state_dict``."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        if hasattr(tensor, "detach"):
            array = tensor.detach().cpu().numpy()
        else:
            array = np.asarray(tensor)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_save_npz(path: str | Path, **arrays: Any) -> None:
    """Guarda un NPZ de forma atómica: no publica un destino a medio escribir."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def horizon_band(horizon_days: float) -> str:
    """Banda canónica de un horizonte continuo en días."""
    value = float(horizon_days)
    for name, lower, upper in HORIZON_BAND_SPECS:
        if lower < value <= upper:
            return name
    if value <= 0:
        raise ValueError(f"horizon_days debe ser positivo, recibido {horizon_days}")
    return HORIZON_BAND_SPECS[-1][0]


def summarize_horizons(values: Iterable[float]) -> dict[str, Any]:
    """Distribución empírica y recuentos por banda de horizonte."""
    horizons = np.asarray(list(values), dtype=np.float64)
    if horizons.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "bands": {name: 0 for name, _, _ in HORIZON_BAND_SPECS},
        }

    invalid = horizons[horizons <= 0]
    if invalid.size:
        raise ValueError(f"hay horizontes no positivos: {invalid.tolist()}")

    return {
        "count": int(horizons.size),
        "min": float(np.min(horizons)),
        "p25": float(np.quantile(horizons, 0.25)),
        "median": float(np.quantile(horizons, 0.50)),
        "p75": float(np.quantile(horizons, 0.75)),
        "p90": float(np.quantile(horizons, 0.90)),
        "max": float(np.max(horizons)),
        "bands": {
            name: int(sum(horizon_band(value) == name for value in horizons))
            for name, _, _ in HORIZON_BAND_SPECS
        },
    }


def scar_coverage(scar: np.ndarray) -> dict[str, Any]:
    """Cobertura y estado de la cicatriz conocida de una muestra."""
    values = np.asarray(scar, dtype=np.float64)
    if values.size == 0:
        raise ValueError("x_scar no puede estar vacío")
    fraction_gt_zero = float(np.mean(values > 0))
    fraction_ge_half = float(np.mean(values >= 0.5))
    return {
        "fraction_gt_zero": fraction_gt_zero,
        "fraction_ge_0_5": fraction_ge_half,
        "state": "cold_start" if fraction_gt_zero == 0 else "detected",
    }


def summarize_scar_coverage(
    coverages: Iterable[Mapping[str, Any]],
    series_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resumen de cobertura de cicatriz y número de muestras de arranque en frío."""
    values = list(coverages)
    ids = list(series_ids) if series_ids is not None else []
    cold_samples = sum(v.get("state") == "cold_start" for v in values)
    cold_series = (
        len({sid for sid, v in zip(ids, values) if v.get("state") == "cold_start"})
        if ids
        else 0
    )
    return {
        "samples_total": len(values),
        "cold_start_samples": int(cold_samples),
        "cold_start_series": int(cold_series),
        "cold_start_represented": bool(cold_samples),
    }


def static_layer_validation_errors(
    dem: np.ndarray,
    worldcover: np.ndarray,
    expected_shape: tuple[int, int] = (64, 64),
) -> list[str]:
    """Motivos por los que las capas estáticas requeridas no son utilizables."""
    errors: list[str] = []
    dem_array = np.asarray(dem)
    worldcover_array = np.asarray(worldcover)

    if dem_array.shape != expected_shape:
        errors.append(f"DEM tiene forma {dem_array.shape}, se esperaba {expected_shape}")
    elif not np.issubdtype(dem_array.dtype, np.number):
        errors.append(f"DEM tiene dtype no numérico {dem_array.dtype}")
    elif not np.all(np.isfinite(dem_array)):
        errors.append("DEM contiene valores no finitos")
    elif not np.any(dem_array != 0):
        errors.append("DEM no contiene ninguna observación no nula")

    if worldcover_array.shape != expected_shape:
        errors.append(
            f"WorldCover tiene forma {worldcover_array.shape}, se esperaba {expected_shape}"
        )
    elif not np.issubdtype(worldcover_array.dtype, np.number):
        errors.append(f"WorldCover tiene dtype no numérico {worldcover_array.dtype}")
    elif not np.all(np.isfinite(worldcover_array)):
        errors.append("WorldCover contiene valores no finitos")
    else:
        classes = {int(value) for value in np.unique(worldcover_array)}
        unknown = sorted(classes - WORLDCOVER_VALID_CLASSES)
        if unknown:
            errors.append(f"WorldCover contiene clases no admitidas: {unknown}")
        if not classes or classes == {0}:
            errors.append("WorldCover está completamente vacío")

    return errors


def validate_static_layers(
    dem: np.ndarray,
    worldcover: np.ndarray,
    expected_shape: tuple[int, int] = (64, 64),
) -> None:
    """Valida DEM y WorldCover o lanza un error con todos los motivos."""
    errors = static_layer_validation_errors(dem, worldcover, expected_shape)
    if errors:
        raise StaticLayerValidationError("; ".join(errors))
