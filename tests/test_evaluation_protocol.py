"""Tests del protocolo de evaluación canónico: denominadores, bandas y arranque en frío."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from burn_scar.baselines import BaselineB0Persistence
from burn_scar.dataset import BurnScarDataset
from burn_scar.quality import describe_evaluated_set

from _fixtures import build_synthetic_dataset


def _load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_specs() -> list[dict]:
    return [
        {
            "sample_id": "a_0",
            "series_id": "serie_a",
            "fold": 0,
            "horizon_days": 0.5,
        },
        {
            "sample_id": "a_1",
            "series_id": "serie_a",
            "fold": 0,
            "horizon_days": 2.0,
        },
        {
            "sample_id": "b_0",
            "series_id": "serie_b",
            "fold": 1,
            "horizon_days": 4.0,
        },
    ]


# ── 5.1 Denominadores separados ────────────────────────────────────────────


def test_el_descriptivo_separa_indice_y_split(tmp_path):
    index_path = build_synthetic_dataset(tmp_path, _canonical_specs())
    dataset = BurnScarDataset(index_path=index_path)

    described = describe_evaluated_set(dataset)

    assert described["index_entries"] == 3
    assert described["index_unique_samples"] == 3
    assert described["index_unique_files"] == 3
    assert described["index_series"] == 2
    assert described["split_samples"] == 3
    assert described["split_unique_samples"] == 3
    assert described["split_unique_files"] == 3
    assert described["split_series"] == 2


def test_el_descriptivo_no_colapsa_split_e_indice(tmp_path):
    index_path = build_synthetic_dataset(tmp_path, _canonical_specs())
    dataset = BurnScarDataset(index_path=index_path, folds=[0])

    described = describe_evaluated_set(dataset)

    assert described["index_entries"] == 3
    assert described["index_unique_samples"] == 3
    assert described["split_samples"] == 2
    assert described["split_unique_samples"] == 2
    assert described["split_series"] == 1
    assert described["index_series"] == 2


def test_un_indice_duplicado_se_rechaza_antes_de_evaluar(tmp_path):
    index_path = build_synthetic_dataset(tmp_path, _canonical_specs())
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    duplicated = dict(payload["samples"][0])
    payload["samples"].append(duplicated)
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_id duplicados"):
        BurnScarDataset(index_path=index_path)


def test_un_indice_con_ruta_duplicada_se_rechaza(tmp_path):
    index_path = build_synthetic_dataset(tmp_path, _canonical_specs())
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["samples"][1]["file"] = payload["samples"][0]["file"]
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rutas de fichero duplicadas"):
        BurnScarDataset(index_path=index_path)


# ── 5.2 Bandas de horizonte ────────────────────────────────────────────────


def _band_specs() -> list[dict]:
    return [
        {"sample_id": "s1", "series_id": "a", "fold": 0, "horizon_days": 0.5},
        {"sample_id": "s2", "series_id": "a", "fold": 0, "horizon_days": 2.0},
        {"sample_id": "s3", "series_id": "b", "fold": 0, "horizon_days": 5.0},
        {"sample_id": "s4", "series_id": "b", "fold": 0, "horizon_days": 10.0},
        {"sample_id": "s5", "series_id": "c", "fold": 0, "horizon_days": 20.0},
    ]


def test_las_bandas_evaluan_sin_filtrar_las_metricas_globales(tmp_path):
    module = _load_script("evaluate_baselines", "evaluate_baselines.py")
    index_path = build_synthetic_dataset(tmp_path, _band_specs())
    dataset = BurnScarDataset(index_path=index_path)

    result = module.evaluate_model(
        dataset, BaselineB0Persistence(), "B0", threshold=0.5
    )

    assert set(result["horizon_strata"]) == {"0-1d", "1-3d", "3-7d", "7-14d", "14d+"}
    total_por_banda = sum(
        strata["assembly"]["n_samples_total"]
        for strata in result["horizon_strata"].values()
    )
    assert total_por_banda == result["n_samples"]
    assert result["n_samples"] == 5
    assert result["evaluated_set"]["split_unique_samples"] == 5


def test_una_banda_no_informativa_devuelve_nan_y_no_cero(tmp_path):
    module = _load_script("evaluate_baselines", "evaluate_baselines.py")
    specs = [
        {
            "sample_id": "informativa",
            "series_id": "a",
            "fold": 0,
            "horizon_days": 0.5,
            "assembly_burn": True,
        },
        {
            "sample_id": "vacia",
            "series_id": "b",
            "fold": 0,
            "horizon_days": 5.0,
            "assembly_burn": False,
        },
    ]
    index_path = build_synthetic_dataset(tmp_path, specs)
    dataset = BurnScarDataset(index_path=index_path)

    result = module.evaluate_model(
        dataset, BaselineB0Persistence(), "B0", threshold=0.5
    )

    banda_vacia = result["horizon_strata"]["3-7d"]["assembly"]
    assert math.isnan(banda_vacia["f1"])
    assert banda_vacia["n_samples_total"] == 1
    assert banda_vacia["n_samples_evaluated"] == 0
    assert not math.isnan(result["horizon_strata"]["0-1d"]["assembly"]["f1"])


# ── 5.3 Arranque en frío ───────────────────────────────────────────────────


def test_el_arranque_en_frio_no_representado_no_publica_desempeno(tmp_path):
    module = _load_script("evaluate_baselines", "evaluate_baselines.py")
    index_path = build_synthetic_dataset(tmp_path, _canonical_specs())
    dataset = BurnScarDataset(index_path=index_path)

    result = module.evaluate_model(
        dataset, BaselineB0Persistence(), "B0", threshold=0.5
    )

    assert result["cold_start"]["represented"] is False
    assert result["cold_start"]["n_samples_total"] == 0
    assert result["cold_start"]["assembly"] is None
    assert "no está representado" in result["cold_start"]["warning"]


def test_el_arranque_en_frio_representado_se_mide_sin_fabricarlo(tmp_path):
    module = _load_script("evaluate_baselines", "evaluate_baselines.py")
    specs = [
        {
            "sample_id": "frio",
            "series_id": "a",
            "fold": 0,
            "horizon_days": 0.5,
            "cold_start": True,
            "assembly_burn": True,
        },
        {
            "sample_id": "caliente",
            "series_id": "b",
            "fold": 0,
            "horizon_days": 2.0,
        },
    ]
    index_path = build_synthetic_dataset(tmp_path, specs)
    dataset = BurnScarDataset(index_path=index_path)

    result = module.evaluate_model(
        dataset, BaselineB0Persistence(), "B0", threshold=0.5
    )

    assert result["cold_start"]["represented"] is True
    assert result["cold_start"]["n_samples_total"] == 1
    assert result["cold_start"]["n_series_total"] == 1
    assert result["cold_start"]["warning"] is None
    assert result["cold_start"]["assembly"]["n_samples_total"] == 1


# ── 5.4 Modelo y baselines sobre el mismo conjunto ─────────────────────────


def test_modelo_y_baselines_comparten_el_mismo_conjunto_evaluado(tmp_path):
    train_module = _load_script("train_propagation", "train_propagation.py")
    baselines_module = _load_script("evaluate_baselines", "evaluate_baselines.py")
    index_path = build_synthetic_dataset(tmp_path, _band_specs())
    dataset = BurnScarDataset(index_path=index_path)

    model = train_module.BurnScarPropagationModel(
        olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    model_result = train_module.evaluate_fold(model, loader, device="cpu")
    baseline_result = baselines_module.evaluate_model(
        dataset, BaselineB0Persistence(), "B0", threshold=0.1
    )

    model_metrics = model_result["per_threshold"][0.1]
    assert model_metrics["n_samples_total"] == baseline_result["n_samples"]
    assert model_metrics["n_samples_evaluated"] == baseline_result["n_samples_evaluated"]
    assert model_result["evaluated_set"] == baseline_result["evaluated_set"]


def test_el_umbral_operativo_se_aplica_a_modelo_y_baselines(tmp_path):
    train_module = _load_script("train_propagation", "train_propagation.py")
    baselines_module = _load_script("evaluate_baselines", "evaluate_baselines.py")
    index_path = build_synthetic_dataset(tmp_path, _band_specs())
    dataset = BurnScarDataset(index_path=index_path)

    model = train_module.BurnScarPropagationModel(
        olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    for threshold in (0.1, 0.5):
        model_result = train_module.evaluate_fold(model, loader, device="cpu")
        baseline_result = baselines_module.evaluate_model(
            dataset, BaselineB0Persistence(), "B0", threshold=threshold
        )
        model_metrics = model_result["per_threshold"][threshold]
        assert (
            model_metrics["n_samples_evaluated"]
            == baseline_result["n_samples_evaluated"]
        )


# ── 5.5 Procedencia persistida ────────────────────────────────────────────


def test_la_procedencia_declara_el_umbral_y_su_dispersion(tmp_path):
    train_module = _load_script("train_propagation", "train_propagation.py")
    index_path = build_synthetic_dataset(tmp_path, _canonical_specs())

    provenance = train_module.build_dataset_provenance(
        index_path,
        json.loads(index_path.read_text(encoding="utf-8")),
        seed=42,
        tag="prop_test",
    )

    assert provenance["seed"] == 42
    assert provenance["run_tag"] == "prop_test"
    assert provenance["dataset_index_sha256"]
    assert provenance["dataset_schema_version"] == 2
    assert provenance["dataset_target_mode"] == "scar"
    assert "horizon_support" in provenance
    assert "cold_start_coverage" in provenance


def test_la_procedencia_de_checkpoints_registra_sumas_de_verificacion(tmp_path):
    train_module = _load_script("train_propagation", "train_propagation.py")
    checkpoint = tmp_path / "best_fold_0.pt"
    torch.save(
        {
            "epoch": 7,
            "seed": 42,
            "weights_sha256": "pesos",
            "dataset_index_sha256": "indice",
        },
        checkpoint,
    )

    provenance = train_module._collect_checkpoint_provenance([checkpoint], [0])

    assert len(provenance) == 1
    entry = provenance[0]
    assert entry["fold"] == 0
    assert entry["epoch"] == 7
    assert entry["seed"] == 42
    assert entry["weights_sha256"] == "pesos"
    assert entry["dataset_index_sha256"] == "indice"
    assert len(entry["file_sha256"]) == 64
