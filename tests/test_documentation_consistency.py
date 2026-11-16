"""Tests de consistencia entre la documentación y los artefactos canónicos.

Las cifras de la documentación no se mantienen como constantes manuales: se
contrastan contra el índice canónico y contra el código que define las bandas
de horizonte. Si el dataset cambia, el test falla.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from burn_scar.quality import HORIZON_BAND_SPECS

CANONICAL_INDEX = Path("data/processed/dataset_index.json")
CANONICAL_INDEX_CEMS = Path("data/processed_target_cems/dataset_index.json")
OPERACION = Path("docs/operacion.md")
DATASET_DOC = Path("docs/dataset.md")
DECISIONES = Path("docs/decisiones.md")
README = Path("README.md")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _band_intervals() -> list[str]:
    intervals = []
    for _, lower, upper in HORIZON_BAND_SPECS:
        low = f"{lower:g}"
        if upper == float("inf"):
            intervals.append(f"({low},+inf)")
        else:
            intervals.append(f"({low},{upper:g}]")
    return intervals


def _canonical_inventory() -> dict:
    if not CANONICAL_INDEX.exists():
        pytest.skip("Índice canónico no generado")
    payload = json.loads(CANONICAL_INDEX.read_text(encoding="utf-8"))
    return payload.get("inventory", {})


def test_las_bandas_de_la_documentacion_coinciden_con_el_codigo():
    for document in (OPERACION, DATASET_DOC):
        if not document.exists():
            pytest.skip(f"{document} no encontrado")
        text = _normalized(document.read_text(encoding="utf-8"))
        for interval in _band_intervals():
            assert interval in text, f"{document} no documenta la banda {interval}"


def test_la_composicion_documentada_cuadra_con_el_indice_canonico():
    inventory = _canonical_inventory()
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    muestras = inventory["total_unique_samples"]
    series = inventory["total_series"]
    assert f"{muestras} muestras" in text
    assert f"{series} series" in text


def test_la_distribucion_de_horizontes_documentada_cuadra_con_el_indice():
    inventory = _canonical_inventory()
    bands = inventory["horizon_distribution"]["bands"]
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    expected = (
        f"{bands['0-1d']} muestras en `(0,1]`, "
        f"{bands['1-3d']} en `(1,3]`, "
        f"{bands['3-7d']} en `(3,7]`, "
        f"{bands['7-14d']} en `(7,14]`"
    )
    assert expected in text


def test_la_cobertura_de_arranque_en_frio_documentada_cuadra_con_el_indice():
    inventory = _canonical_inventory()
    coverage = inventory["scar_coverage"]
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    assert coverage["cold_start_samples"] == 0
    assert coverage["cold_start_represented"] is False
    assert "arranque en frío" in text
    assert "cold_start_represented: false" in text


def test_la_exclusion_de_series_documentada_cuadra_con_el_indice():
    inventory = _canonical_inventory()
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    for exclusion in inventory["excluded_series"]:
        assert exclusion["series_id"] in text
        assert exclusion["reason"] in text


def test_ambos_indices_canonicos_comparten_composicion():
    if not (CANONICAL_INDEX.exists() and CANONICAL_INDEX_CEMS.exists()):
        pytest.skip("Índices canónicos no generados")
    produccion = json.loads(CANONICAL_INDEX.read_text(encoding="utf-8"))
    cems = json.loads(CANONICAL_INDEX_CEMS.read_text(encoding="utf-8"))

    assert produccion["inventory"]["total_unique_samples"] == (
        cems["inventory"]["total_unique_samples"]
    )
    assert produccion["inventory"]["total_series"] == (
        cems["inventory"]["total_series"]
    )
    assert produccion["inventory"]["horizon_distribution"] == (
        cems["inventory"]["horizon_distribution"]
    )
    assert produccion["target_mode"] != cems["target_mode"]


def test_la_documentacion_no_declara_arranque_en_frio_como_representado():
    for document in (OPERACION, DATASET_DOC, DECISIONES, README):
        if not document.exists():
            continue
        text = _normalized(document.read_text(encoding="utf-8")).lower()
        assert "arranque en frío incluido" not in text


def test_el_run_historico_prop_v4_esta_marcado_como_no_publicable():
    text = _normalized(OPERACION.read_text(encoding="utf-8")).lower()
    assert "prop_v4_integrity" in text
    assert "histórico" in text
    assert "no publicable" in text
    # El run canónico es el único que puede Appecer como resultado publicable.
    assert "prop_v4` queda marcado" in text


def test_la_documentacion_deja_claro_que_no_hay_maximo_de_horizonte():
    negaciones = ("sin máximo", "no hay máximo", "no hay horizonte máximo",
                  "ningún horizonte máximo", "ningun horizonte maximo",
                  "no imponen ningún horizonte máximo",
                  "no fija un horizonte máximo", "no impone un horizonte máximo")
    for document in (OPERACION, DATASET_DOC):
        text = _normalized(document.read_text(encoding="utf-8")).lower()
        assert any(negacion in text for negacion in negaciones), (
            f"{document} no deja claro que no hay horizonte máximo"
        )


def test_las_huellas_documentadas_coinciden_con_el_manifiesto_de_artefactos():
    manifest_path = Path("artefactos/canonicos.json")
    if not manifest_path.exists():
        pytest.skip("Manifiesto de artefactos no generado")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    for label, entry in manifest["datasets"].items():
        if not entry.get("available"):
            continue
        assert entry["content_sha256"] in text, (
            f"la huella de contenido de {label} no aparece en operacion.md"
        )


def test_el_manifiesto_de_artefactos_declara_la_procedencia_de_los_runs():
    manifest_path = Path("artefactos/canonicos.json")
    if not manifest_path.exists():
        pytest.skip("Manifiesto de artefactos no generado")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for label, entry in manifest["runs"].items():
        if not entry.get("available"):
            continue
        assert entry["checkpoint_dir"], f"{label} no declara su directorio"
        assert entry["operational_threshold_sha256"], f"{label} no declara el hash del umbral"
        assert entry["dataset_index_sha256"], f"{label} no declara el hash del índice"
        assert entry["dataset_content_sha256"], f"{label} no declara la huella de contenido"
        assert entry["dataset_generator_version"], f"{label} no declara el generador"
        assert entry["seed"] is not None, f"{label} no declara la semilla"
        assert len(entry["checkpoints"]) == 5, f"{label} no tiene cinco checkpoints"
        for checkpoint in entry["checkpoints"]:
            assert len(checkpoint["file_sha256"]) == 64
