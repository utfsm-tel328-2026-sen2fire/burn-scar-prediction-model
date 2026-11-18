"""Helpers para construir datasets sintéticos en disco durante los tests."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from burn_scar.dataset import compute_generator_version
from burn_scar.quality import horizon_band, scar_coverage


def _sample_arrays(
    cold_start: bool,
    assembly_burn: bool,
    reference_burn: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scar = np.zeros((64, 64), dtype=np.float32)
    if not cold_start:
        scar[10:20, 10:20] = 1.0
    target = np.zeros((64, 64), dtype=np.float32)
    if assembly_burn:
        target[20:30, 20:30] = 1.0
    target_cems = np.zeros((64, 64), dtype=np.float32)
    if reference_burn:
        target_cems[20:30, 20:30] = 1.0
    return {
        "x_olci": rng.random((7, 5, 64, 64), dtype=np.float32) * 1000.0,
        "x_dem": np.full((64, 64), 100.0, dtype=np.float32),
        "x_worldcover": np.full((64, 64), 40, dtype=np.uint8),
        "x_scar": scar,
        "horizon_days": np.float32(1.0),
        "target": target,
        "mask": np.ones((64, 64), dtype=np.uint8),
        "target_cems": target_cems,
        "mask_cems": np.ones((64, 64), dtype=np.uint8),
    }


def build_synthetic_dataset(
    root: Path,
    specs: Sequence[dict[str, Any]],
    target_mode: str = "scar",
) -> Path:
    """
    Escribe un dataset sintético canónico y devuelve la ruta de su índice.

    Cada entrada de `specs` describe una muestra: `sample_id`, `series_id`,
    `fold`, `horizon_days` y las banderas `cold_start`, `assembly_burn` y
    `reference_burn`.
    """
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for position, spec in enumerate(specs):
        sample_id = spec["sample_id"]
        file_path = samples_dir / f"{sample_id}.npz"
        arrays = _sample_arrays(
            cold_start=bool(spec.get("cold_start", False)),
            assembly_burn=bool(spec.get("assembly_burn", True)),
            reference_burn=bool(spec.get("reference_burn", True)),
            seed=position,
        )
        arrays["horizon_days"] = np.float32(spec["horizon_days"])
        np.savez_compressed(file_path, **arrays)

        coverage = scar_coverage(arrays["x_scar"])
        samples.append({
            "sample_id": sample_id,
            "series_id": spec["series_id"],
            "file": str(file_path),
            "fold": spec.get("fold", 0),
            "horizon_days": float(spec["horizon_days"]),
            "horizon_band": horizon_band(float(spec["horizon_days"])),
            "scar_state": coverage["state"],
            "scar_coverage": {
                "state": coverage["state"],
                "fraction_gt_zero": coverage["fraction_gt_zero"],
                "fraction_ge_0_5": coverage["fraction_ge_0_5"],
            },
        })

    index_path = root / "dataset_index.json"
    index_path.write_text(json.dumps({
        "generated_at": "2026-01-01T00:00:00Z",
        "dataset_index_schema_version": 2,
        "target_mode": target_mode,
        "generator_version": compute_generator_version(),
        "total_samples": len(samples),
        "samples": samples,
    }, indent=2), encoding="utf-8")
    return index_path
