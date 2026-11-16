#!/usr/bin/env python3
"""Audita la integridad de un dataset de propagación y publica un informe JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burn_scar.dataset import validate_unique_sample_index
from burn_scar.quality import (
    WORLDCOVER_VALID_CLASSES,
    file_sha256,
    horizon_band,
    scar_coverage,
    summarize_horizons,
    summarize_scar_coverage,
    validate_static_layers,
)


def audit_dataset(index_path: str | Path) -> dict:
    """Audita índice, paquetes, geometría estática y metadatos de cobertura."""
    index_path = Path(index_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    errors: list[str] = []

    try:
        validate_unique_sample_index(samples)
    except ValueError as exc:
        errors.append(str(exc))

    sample_paths = {Path(sample["file"]) for sample in samples}
    sample_dirs = {path.parent for path in sample_paths}
    files_on_disk = {
        path
        for directory in sample_dirs
        for path in directory.glob("*.npz")
    }
    missing = sorted(str(path) for path in sample_paths - files_on_disk)
    orphan = sorted(str(path) for path in files_on_disk - sample_paths)
    if missing:
        errors.append(f"paquetes ausentes: {missing[:5]}")
    if orphan:
        errors.append(f"paquetes huérfanos: {orphan[:5]}")

    horizons = []
    coverages = []
    series_ids = []
    for sample in samples:
        path = Path(sample["file"])
        if not path.exists():
            continue
        with np.load(path) as data:
            expected_shapes = {
                "x_olci": (7, 5, 64, 64),
                "x_dem": (64, 64),
                "x_worldcover": (64, 64),
                "x_scar": (64, 64),
            }
            for key, shape in expected_shapes.items():
                if key not in data.files or data[key].shape != shape:
                    actual = None if key not in data.files else data[key].shape
                    errors.append(f"{sample['sample_id']}: {key} tiene forma {actual}")
            if "x_dem" in data.files and "x_worldcover" in data.files:
                try:
                    validate_static_layers(data["x_dem"], data["x_worldcover"])
                except ValueError as exc:
                    errors.append(f"{sample['sample_id']}: {exc}")
                unknown = sorted(
                    set(np.unique(data["x_worldcover"]).astype(int).tolist())
                    - WORLDCOVER_VALID_CLASSES
                )
                if unknown:
                    errors.append(f"{sample['sample_id']}: clases WorldCover {unknown}")

            coverage = scar_coverage(data["x_scar"])
            if sample.get("scar_state") != coverage["state"]:
                errors.append(
                    f"{sample['sample_id']}: scar_state {sample.get('scar_state')} "
                    f"no coincide con {coverage['state']}"
                )
            if sample.get("horizon_band") != horizon_band(float(sample["horizon_days"])):
                errors.append(f"{sample['sample_id']}: banda de horizonte incorrecta")

            horizons.append(float(sample["horizon_days"]))
            coverages.append(coverage)
            series_ids.append(sample["series_id"])

    inventory = payload.get("inventory", {})
    expected_unique_files = len({sample.get("file") for sample in samples})
    if inventory.get("total_unique_files") != expected_unique_files:
        errors.append("inventory.total_unique_files no coincide con el índice")
    if inventory.get("total_entries") != len(samples):
        errors.append("inventory.total_entries no coincide con el índice")

    report = {
        "index": str(index_path),
        "index_sha256": file_sha256(index_path),
        "schema_version": payload.get("dataset_index_schema_version"),
        "generator_version": payload.get("generator_version"),
        "target_mode": payload.get("target_mode"),
        "ok": not errors,
        "errors": errors,
        "counts": {
            "entries": len(samples),
            "unique_samples": len({sample.get("sample_id") for sample in samples}),
            "unique_files": expected_unique_files,
            "series": len(set(series_ids)),
            "files_on_disk": len(files_on_disk),
        },
        "horizon_distribution": summarize_horizons(horizons),
        "scar_coverage": summarize_scar_coverage(coverages, series_ids),
        "excluded_series": inventory.get("excluded_series", []),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Auditoría de integridad del dataset de propagación")
    parser.add_argument("--index", default="data/processed/dataset_index.json")
    parser.add_argument("--output", default=None, help="Fichero JSON opcional para el informe")
    args = parser.parse_args()

    report = audit_dataset(args.index)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if not report["ok"]:
        print("Auditoría FALLIDA", file=sys.stderr)
        return 1
    print("Auditoría correcta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
