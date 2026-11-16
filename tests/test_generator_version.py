"""Tests de la codificación canónica de las máscaras y de la versión del generador."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from burn_scar.dataset import (
    GENERATOR_SOURCES,
    BurnScarDataset,
    compute_generator_version,
    generator_source_names,
)

INDEX_PATH = Path("data/processed/dataset_index.json")
SAMPLES_DIR = Path("data/processed/samples")


# ── 2.2 El paquete almacenado usa la codificación canónica ─────────────────

@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Dataset index no generado")
def test_el_paquete_almacenado_usa_la_codificacion_canonica():
    """Las máscaras del .npz contienen solo 1 y 255, sin pasar por el cargador."""
    if not SAMPLES_DIR.exists():
        pytest.skip("directorio de muestras no generado")

    ficheros = sorted(SAMPLES_DIR.glob("*.npz"))[:50]
    if not ficheros:
        pytest.skip("no hay paquetes generados")

    for path in ficheros:
        with np.load(path) as data:
            mask_vals = set(np.unique(data["mask"]).tolist())
            mask_cems_vals = set(np.unique(data["mask_cems"]).tolist())
        assert mask_vals <= {1, 255}, f"{path.name}: mask contiene {sorted(mask_vals)}"
        assert mask_cems_vals <= {1, 255}, f"{path.name}: mask_cems contiene {sorted(mask_cems_vals)}"


# ── 3.1-3.4 Versión del generador ─────────────────────────────────────────

def test_la_version_del_generador_es_estable_entre_llamadas():
    assert compute_generator_version() == compute_generator_version()


def test_la_version_del_generador_cambia_con_el_contenido(tmp_path):
    """La huella depende del contenido de los ficheros, no de su nombre."""
    fuente = tmp_path / "generador.py"
    fuente.write_text("VERSION = 1\n", encoding="utf-8")

    v1 = compute_generator_version([fuente])
    fuente.write_text("VERSION = 2\n", encoding="utf-8")
    v2 = compute_generator_version([fuente])

    assert v1 != v2


def test_la_version_del_generador_tiene_longitud_estable():
    assert len(compute_generator_version()) == 12


def test_los_ficheros_del_generador_son_los_esperados():
    """Si cambia el conjunto de ficheros que determina el dataset, se actualiza."""
    assert generator_source_names() == [
        "scripts/build_pytorch_dataset.py",
        "src/burn_scar/dataset.py",
    ]
    assert len(GENERATOR_SOURCES) == 2


def _escribir_indice(tmp_path, extra: dict) -> Path:
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"samples": [], **extra}), encoding="utf-8")
    return index_file


def test_el_cargador_avisa_si_el_dataset_no_registra_versio(tmp_path):
    """Un dataset sin versión registrada se avisa aparte."""
    index_file = _escribir_indice(tmp_path, {})

    with pytest.warns(RuntimeWarning, match="no registra la versión"):
        BurnScarDataset(index_path=index_file)


def test_el_cargador_avisa_si_el_dataset_es_de_otra_version(tmp_path):
    """Un dataset generado por otro generador se avisa al usarlo."""
    index_file = _escribir_indice(tmp_path, {"generator_version": "000000000000"})

    with pytest.warns(RuntimeWarning, match="Regenera el dataset"):
        BurnScarDataset(index_path=index_file)


def test_el_cargador_no_avisa_si_el_dataset_es_de_la_version_actual(tmp_path):
    import warnings

    index_file = _escribir_indice(
        tmp_path, {"generator_version": compute_generator_version()}
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        BurnScarDataset(index_path=index_file)

    assert not [r for r in records if "versión del generador" in str(r.message)]
