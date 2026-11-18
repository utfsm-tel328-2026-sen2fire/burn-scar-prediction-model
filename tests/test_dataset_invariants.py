"""Invariantes del dataset de propagación sobre el índice generado."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from burn_scar.dataset import is_chronological

INDEX_PATH = Path("data/processed/dataset_index.json")
SAMPLES_DIR = Path("data/processed/samples")


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        pytest.skip("Dataset index no generado")
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_todas_las_ventanas_terminan_en_el_frame_f_de_su_muestra():
    """La ventana temporal acaba en la adquisición que corresponde a f."""
    index = _load_index()
    samples = index["samples"]
    assert samples, "el índice no contiene muestras"

    for s in samples:
        f_key = s.get("frame_key") or s["sample_id"].split("_F_")[-1]
        assert s["frame_keys"][-1] == f_key, (
            f"{s['sample_id']}: la ventana termina en {s['frame_keys'][-1]}, "
            f"no en {f_key}"
        )


def test_todas_las_ventanas_estan_ordenadas_en_el_tiempo():
    index = _load_index()

    desordenadas = [
        s["sample_id"] for s in index["samples"] if not is_chronological(s["frame_keys"])
    ]
    assert not desordenadas, f"ventanas no ordenadas: {desordenadas[:5]}"


def test_todas_las_ventanas_tienen_siete_frames():
    index = _load_index()

    for s in index["samples"]:
        assert len(s["frame_keys"]) == 7, s["sample_id"]


def test_no_hay_paquetes_sin_entrada_en_el_indice():
    """Las muestras descartadas no dejan ficheros .npz huérfanos en disco."""
    index = _load_index()
    if not SAMPLES_DIR.exists():
        pytest.skip("directorio de muestras no generado")

    en_indice = {Path(s["file"]).name for s in index["samples"]}
    en_disco = {p.name for p in SAMPLES_DIR.glob("*.npz")}

    huerfanos = en_disco - en_indice
    assert not huerfanos, f"ficheros .npz sin entrada en el índice: {sorted(huerfanos)[:5]}"


def test_el_indice_registra_el_modo_de_target():
    index = _load_index()
    assert index.get("target_mode") in ("scar", "cems")
