"""Tests de los controles de calidad compartidos del dataset."""
from __future__ import annotations

import numpy as np
import pytest

from burn_scar.quality import (
    StaticLayerValidationError,
    horizon_band,
    scar_coverage,
    static_layer_validation_errors,
    summarize_horizons,
    validate_static_layers,
)


def _valid_layers() -> tuple[np.ndarray, np.ndarray]:
    dem = np.full((64, 64), 100.0, dtype=np.float32)
    worldcover = np.full((64, 64), 40, dtype=np.uint8)
    return dem, worldcover


def test_las_clases_worldcover_documentadas_coinciden_con_el_codigo():
    from pathlib import Path

    from burn_scar.quality import WORLDCOVER_VALID_CLASSES

    documentation = Path("docs/datos.md").read_text(encoding="utf-8")
    for class_code in WORLDCOVER_VALID_CLASSES:
        assert str(class_code) in documentation


def test_hash_de_pesos_es_estable_y_cambia_con_los_pesos():
    import torch

    from burn_scar.quality import state_dict_sha256

    first = {"weight": torch.tensor([1.0, 2.0])}
    second = {"weight": torch.tensor([1.0, 2.0])}
    changed = {"weight": torch.tensor([1.0, 3.0])}

    assert state_dict_sha256(first) == state_dict_sha256(second)
    assert state_dict_sha256(first) != state_dict_sha256(changed)


def test_la_huella_de_contenido_ignora_la_cabecera_temporal():
    from burn_scar.quality import canonical_json_sha256

    contenido = {"inventory": {"total_entries": 3}, "samples": [{"sample_id": "a"}]}

    primera = canonical_json_sha256(contenido)
    segunda = canonical_json_sha256(contenido)

    assert primera == segunda
    assert len(primera) == 64
    assert canonical_json_sha256({"inventory": {}, "samples": []}) != primera


def test_un_run_historico_declara_su_incompatibilidad(tmp_path):
    import json

    from burn_scar.quality import historical_status

    run_dir = tmp_path / "khaen_prop_v4"
    run_dir.mkdir()
    assert historical_status(run_dir) is None

    (run_dir / "historico.json").write_text(json.dumps({
        "status": "historico_no_publicable",
        "incompatible_with_canonical_index": True,
        "canonical_run": "prop_v4_integrity",
    }), encoding="utf-8")

    status = historical_status(run_dir)
    assert status["incompatible_with_canonical_index"] is True
    assert status["canonical_run"] == "prop_v4_integrity"


def test_capas_estaticas_validas_se_aceptan():
    dem, worldcover = _valid_layers()
    assert static_layer_validation_errors(dem, worldcover) == []
    validate_static_layers(dem, worldcover)


def test_worldcover_completamente_vacio_se_rechaza():
    dem, worldcover = _valid_layers()
    worldcover[:] = 0
    errors = static_layer_validation_errors(dem, worldcover)
    assert any("WorldCover está completamente vacío" in error for error in errors)


def test_clase_worldcover_no_admitida_se_rechaza():
    dem, worldcover = _valid_layers()
    worldcover = worldcover.astype(np.uint16)
    worldcover[0, 0] = 999
    errors = static_layer_validation_errors(dem, worldcover)
    assert any("clases no admitidas" in error for error in errors)


def test_dem_con_nan_o_forma_incorrecta_se_rechaza():
    dem, worldcover = _valid_layers()
    dem[0, 0] = np.nan
    assert any("no finitos" in error for error in static_layer_validation_errors(dem, worldcover))

    dem, worldcover = _valid_layers()
    assert any("forma" in error for error in static_layer_validation_errors(dem[:32], worldcover))


def test_validate_static_layers_incluye_todos_los_motivos():
    dem, worldcover = _valid_layers()
    worldcover[:] = 0
    with pytest.raises(StaticLayerValidationError, match="WorldCover"):
        validate_static_layers(dem, worldcover)


def test_las_bandas_de_horizonte_son_exhaustivas_y_excluyentes():
    assert horizon_band(0.001) == "0-1d"
    assert horizon_band(1.0) == "0-1d"
    assert horizon_band(1.0001) == "1-3d"
    assert horizon_band(3.0) == "1-3d"
    assert horizon_band(3.0001) == "3-7d"
    assert horizon_band(7.0) == "3-7d"
    assert horizon_band(7.0001) == "7-14d"
    assert horizon_band(14.0) == "7-14d"
    assert horizon_band(14.0001) == "14d+"

    with pytest.raises(ValueError, match="positivo"):
        horizon_band(0.0)


def test_resumen_de_horizontes_conserva_percentiles_y_bandas():
    summary = summarize_horizons([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    assert summary["count"] == 6
    assert summary["min"] == 0.5
    assert summary["max"] == 16.0
    assert summary["bands"] == {
        "0-1d": 2,
        "1-3d": 1,
        "3-7d": 1,
        "7-14d": 1,
        "14d+": 1,
    }


def test_cobertura_de_cicatriz_identifica_arranque_en_frio():
    assert scar_coverage(np.zeros((4, 4), dtype=np.float32))["state"] == "cold_start"
    detected = scar_coverage(np.array([[0.0, 0.5]], dtype=np.float32))
    assert detected["state"] == "detected"
    assert detected["fraction_gt_zero"] == 0.5
    assert detected["fraction_ge_0_5"] == 0.5
