"""Tests de identidad canónica de adquisiciones OLCI del manifiesto CEMS."""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts/build_cems_manifest.py")


def _load_manifest_script():
    spec = importlib.util.spec_from_file_location("build_cems_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(product_id: str, timestamp: str, satellite: str = "S3B") -> dict:
    return {
        "id": product_id,
        "name": f"{satellite}_OL_1_EFR",
        "satellite": satellite,
        "acquisition_time": timestamp,
        "dt": dt.datetime.fromisoformat(timestamp),
    }


def test_productos_del_mismo_satelite_y_segundo_se_colapsan():
    module = _load_manifest_script()
    scenes = [
        _scene("z-product", "2026-07-11T09:52:40.993084"),
        _scene("a-product", "2026-07-11T09:52:40.993084"),
    ]

    canonical = module.deduplicate_olci_scenes(scenes)

    assert len(canonical) == 1
    assert canonical[0]["id"] == "a-product"
    assert canonical[0]["product_ids"] == ["a-product", "z-product"]
    assert canonical[0]["collapsed_product_ids"] == ["z-product"]
    assert canonical[0]["acquisition_key"] == "S3B:20260711T095240"


def test_las_tres_colisiones_conocidas_se_colapsan():
    module = _load_manifest_script()
    collisions = [
        (
            "2026-07-11T09:52:40.993084",
            "53bffbc8-3b7f-46ed-9007-d9fa143a7829",
            "9c7d1fe4-21ba-4afa-87cb-a14b7cd02cfc",
        ),
        (
            "2026-07-10T10:15:51.823541",
            "3e4df8c3-0ead-4377-9000-134e54ae1c7e",
            "f0cf9ed6-0638-445d-9a6b-70db23d84e76",
        ),
        (
            "2025-01-27T04:10:27.818311",
            "eeeeb81b-3565-3168-a8b6-020197d01f13",
            "b837f9fe-4b8c-4628-8c87-874592db672b",
        ),
    ]

    for timestamp, first_id, second_id in collisions:
        canonical = module.deduplicate_olci_scenes(
            [_scene(first_id, timestamp), _scene(second_id, timestamp)]
        )
        assert len(canonical) == 1
        assert canonical[0]["product_ids"] == sorted([first_id, second_id])


def test_manifiesto_almacenado_no_contiene_muestras_duplicadas():
    import json

    path = Path("data/cems_manifest_46.json")
    if not path.exists():
        pytest.skip("manifiesto CEMS no generado")
    payload = json.loads(path.read_text(encoding="utf-8"))
    catalog = payload["summary"]["olci_catalog"]
    assert catalog["products_collapsed"] == 3
    assert catalog["acquisitions_unique"] == 411
    for series in payload["series"]:
        ids = [sample["sample_id"] for sample in series.get("samples", [])]
        assert len(ids) == len(set(ids))


def test_adquisiciones_distintas_no_se_colapsan():
    module = _load_manifest_script()
    scenes = [
        _scene("a", "2026-07-11T09:52:40"),
        _scene("b", "2026-07-11T09:52:41"),
        _scene("c", "2026-07-11T09:52:40", satellite="S3A"),
    ]

    canonical = module.deduplicate_olci_scenes(scenes)

    assert len(canonical) == 3
    assert len({scene["acquisition_key"] for scene in canonical}) == 3


def test_deduplicacion_del_manifiesto_conserva_una_muestra_por_id():
    module = _load_manifest_script()
    series = [
        {
            "series_id": "serie",
            "samples": [
                {"sample_id": "a", "frame_scene_id": "1"},
                {"sample_id": "a", "frame_scene_id": "2"},
                {"sample_id": "b", "frame_scene_id": "3"},
            ],
        }
    ]

    summary = module.deduplicate_manifest_samples(series)

    assert [sample["sample_id"] for sample in series[0]["samples"]] == ["a", "b"]
    assert series[0]["samples"][0]["frame_scene_id"] == "1"
    assert summary == {
        "products_total": 3,
        "acquisitions_unique": 2,
        "acquisitions_collapsed": 1,
        "products_collapsed": 1,
    }


def test_resumen_catalogo_reporta_productos_colapsados():
    module = _load_manifest_script()
    scenes = module.deduplicate_olci_scenes(
        [
            _scene("z", "2026-07-11T09:52:40.993084"),
            _scene("a", "2026-07-11T09:52:40.993084"),
            _scene("b", "2026-07-12T09:52:40"),
        ]
    )

    summary = module.summarize_olci_catalog(scenes)

    assert summary == {
        "products_total": 3,
        "acquisitions_unique": 2,
        "acquisitions_collapsed": 1,
        "products_collapsed": 1,
    }
