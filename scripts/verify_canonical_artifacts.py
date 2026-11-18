#!/usr/bin/env python3
"""
Verifica las sumas de verificación de los artefactos canónicos del proyecto.

El manifiesto `artefactos/canonicos.json` registra la huella de cada índice de
dataset, de cada checkpoint del run canónico y del fichero de umbral
operativo. Este script recalcula esas huellas y las contrasta con el
manifiesto, de modo que se pueda comprobar que los artefactos publicados
corresponden al dataset y a la ejecución canónicos.

Uso:
    python3 scripts/verify_canonical_artifacts.py            # verifica
    python3 scripts/verify_canonical_artifacts.py --write    # regenera el manifiesto
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burn_scar.quality import canonical_json_sha256, file_sha256

MANIFEST_PATH = Path("artefactos/canonicos.json")

# Índices de dataset que forman parte del estado canónico.
DATASET_INDICES = (
    ("production", "data/processed/dataset_index.json"),
    ("reference_cems", "data/processed_target_cems/dataset_index.json"),
)

# Runs canónicos: etiqueta lógica. El directorio se descubre en disco, de
# forma que el manifiesto no depende de la marca temporal del run.
CANONICAL_RUNS = (
    "prop_v4_integrity",
    "prop_cems_diag_v4_integrity",
)


def find_run_dir(label: str) -> Path | None:
    """Directorio del run canónico más reciente con la etiqueta dada."""
    candidates = sorted(Path("checkpoints").glob(f"*_{label}_*"))
    return candidates[-1] if candidates else None


def dataset_content_sha256(index_path: Path) -> str | None:
    """Huella de contenido del índice, independiente de su cabecera temporal."""
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return canonical_json_sha256({
        "inventory": payload.get("inventory", {}),
        "samples": payload.get("samples", []),
        "target_mode": payload.get("target_mode"),
    })


def collect() -> dict[str, Any]:
    """Construye el manifiesto a partir del estado actual del disco."""
    datasets: dict[str, Any] = {}
    for label, relative in DATASET_INDICES:
        index_path = Path(relative)
        if not index_path.is_file():
            datasets[label] = {"index": relative, "available": False}
            continue
        datasets[label] = {
            "index": relative,
            "available": True,
            "file_sha256": file_sha256(index_path),
            "content_sha256": dataset_content_sha256(index_path),
            "generator_version": json.loads(
                index_path.read_text(encoding="utf-8")
            ).get("generator_version"),
            "schema_version": json.loads(
                index_path.read_text(encoding="utf-8")
            ).get("dataset_index_schema_version"),
        }

    runs: dict[str, Any] = {}
    for label in CANONICAL_RUNS:
        run_dir = find_run_dir(label)
        if run_dir is None:
            runs[label] = {"checkpoint_dir": None, "available": False}
            continue
        entry: dict[str, Any] = {
            "checkpoint_dir": str(run_dir),
            "available": True,
            "checkpoints": [],
        }
        threshold_file = run_dir / "operational_threshold.json"
        if threshold_file.is_file():
            payload = json.loads(threshold_file.read_text(encoding="utf-8"))
            entry["operational_threshold_file"] = str(threshold_file)
            entry["operational_threshold_sha256"] = file_sha256(threshold_file)
            entry["operational_threshold_value"] = payload.get("operational_threshold")
            entry["threshold_spread"] = payload.get("threshold_spread")
            entry["seed"] = payload.get("seed")
            entry["dataset_index_sha256"] = payload.get("dataset_index_sha256")
            entry["dataset_content_sha256"] = payload.get("dataset_content_sha256")
            entry["dataset_generator_version"] = payload.get(
                "dataset_generator_version"
            )
        for checkpoint in sorted(run_dir.glob("best_fold_*.pt")):
            entry["checkpoints"].append({
                "path": str(checkpoint),
                "file_sha256": file_sha256(checkpoint),
            })
        runs[label] = entry

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": datasets,
        "runs": runs,
    }


def verify(manifest: dict[str, Any]) -> tuple[int, int, int]:
    """Contrasta el manifiesto con el disco. Devuelve (ok, mismatch, missing)."""
    ok = 0
    mismatch = 0
    missing = 0

    for label, entry in manifest.get("datasets", {}).items():
        index_path = Path(entry["index"])
        if not index_path.is_file():
            print(f"[AUSENTE] dataset {label}: {index_path}")
            missing += 1
            continue
        current_file = file_sha256(index_path)
        if entry.get("file_sha256") and current_file != entry["file_sha256"]:
            print(
                f"[DIFIERE] dataset {label}: el fichero del índice cambió "
                f"({current_file[:12]}… frente a {entry['file_sha256'][:12]}…)"
            )
            mismatch += 1
            continue
        current_content = dataset_content_sha256(index_path)
        if entry.get("content_sha256") and current_content != entry["content_sha256"]:
            print(
                f"[DIFIERE] dataset {label}: el contenido del índice cambió "
                f"({current_content[:12]}… frente a {entry['content_sha256'][:12]}…)"
            )
            mismatch += 1
            continue
        print(f"[OK] dataset {label}: {index_path} ({current_content[:12]}…)")
        ok += 1

    for label, entry in manifest.get("runs", {}).items():
        relative = entry.get("checkpoint_dir")
        if not relative:
            print(f"[AUSENTE] run {label}: sin directorio registrado")
            missing += 1
            continue
        run_dir = Path(relative)
        if not run_dir.is_dir():
            print(f"[AUSENTE] run {label}: {run_dir}")
            missing += 1
            continue
        threshold_file = run_dir / "operational_threshold.json"
        if threshold_file.is_file() and entry.get("operational_threshold_sha256"):
            current = file_sha256(threshold_file)
            if current != entry["operational_threshold_sha256"]:
                print(f"[DIFIERE] run {label}: el umbral operativo cambió")
                mismatch += 1
                continue
        for checkpoint in entry.get("checkpoints", []):
            path = Path(checkpoint["path"])
            if not path.is_file():
                print(f"[AUSENTE] run {label}: {path}")
                missing += 1
                continue
            current = file_sha256(path)
            if current != checkpoint["file_sha256"]:
                print(f"[DIFIERE] run {label}: el checkpoint {path.name} cambió")
                mismatch += 1
                continue
            print(f"[OK] run {label}: {path.name} ({current[:12]}…)")
            ok += 1

    return ok, mismatch, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica las sumas de verificación de los artefactos canónicos"
    )
    parser.add_argument(
        "--manifest", default=str(MANIFEST_PATH), help="Ruta del manifiesto JSON"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenera el manifiesto a partir del estado actual del disco",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    if args.write:
        manifest = collect()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Manifiesto escrito en {manifest_path}")
        return 0

    if not manifest_path.is_file():
        print(
            f"No se encuentra el manifiesto {manifest_path}. "
            "Géneralo con `--write` tras produce los artefactos canónicos.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok, mismatch, missing = verify(manifest)
    print("-" * 72)
    print(f"Artefactos verificados: {ok} | divergentes: {mismatch} | ausentes: {missing}")
    if mismatch:
        print("Los artefactos divergen del manifiesto canónico.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
