"""Tests unitarios para el modelo de propagación y su pérdida enmascarada."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from burn_scar.models import BurnScarPropagationModel, corrupt_scar, masked_bce_dice_loss


def _load_train_script():
    """Carga `scripts/train_propagation.py` como módulo para probar su evaluación."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_propagation.py"
    spec = importlib.util.spec_from_file_location("train_propagation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrupt_scar_bounds_and_shape() -> None:
    scar = torch.zeros(2, 1, 64, 64)
    scar[:, :, 20:40, 20:40] = 1.0
    corrupted = corrupt_scar(scar, drop_prob=0.15)
    assert corrupted.shape == scar.shape
    assert corrupted.min() >= 0.0
    assert corrupted.max() <= 1.0


def test_corrupt_scar_es_vectorizado_por_muestra() -> None:
    """La dilatación/erosión se decide por muestra sin sincronizar la GPU."""
    scar = torch.zeros(4, 1, 32, 32)
    scar[:, :, 10:20, 10:20] = 1.0
    torch.manual_seed(0)
    corrupted = corrupt_scar(scar, drop_prob=0.0)
    # Cada muestra se dilata o se erosiona, pero nunca se mezclan entre muestras.
    for i in range(4):
        sample = corrupted[i]
        assert torch.isin(sample, torch.tensor([0.0, 1.0])).all()
        assert sample.sum() > 0


def test_propagation_model_forward_and_backward() -> None:
    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=16, scar_corruption_prob=0.15)
    model.train()

    B, T, H, W = 2, 7, 64, 64
    x_olci = torch.randn(B, T, 5, H, W)
    x_dem = torch.rand(B, 1, H, W)
    x_wc = torch.rand(B, 1, H, W)
    x_scar = torch.zeros(B, 1, H, W)
    x_scar[:, :, 24:36, 24:36] = 1.0
    horizon_days = torch.tensor([[1.0], [7.0]])

    logits = model(x_olci, x_dem, x_wc, x_scar, horizon_days)
    assert logits.shape == (B, 1, H, W)

    target = torch.zeros(B, 1, H, W)
    target[:, :, 20:40, 20:40] = 1.0
    mask = torch.ones(B, 1, H, W, dtype=torch.uint8)
    mask[:, :, :5, :5] = 255

    loss = masked_bce_dice_loss(logits, target, mask)
    assert torch.isfinite(loss)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_masked_loss_derives_pos_weight_from_batch_imbalance() -> None:
    """El peso positivo se deriva del desbalance real del lote."""
    logits = torch.zeros(1, 1, 10, 10)
    target = torch.zeros(1, 1, 10, 10)
    mask = torch.ones(1, 1, 10, 10, dtype=torch.uint8)

    # 10% de píxeles quemados (una columna de 10 sobre 100) -> pos_weight ~ 9
    target[:, :, :, 0] = 1.0
    loss_auto = masked_bce_dice_loss(logits, target, mask)
    loss_manual = masked_bce_dice_loss(logits, target, mask, pos_weight=9.0)
    assert torch.isfinite(loss_auto)
    assert float(loss_auto) == pytest.approx(float(loss_manual), abs=1e-5)

    # Una sobrescritura explícita se respeta
    loss_w = masked_bce_dice_loss(logits, target, mask, pos_weight=1.0)
    assert float(loss_w) != pytest.approx(float(loss_auto), abs=1e-5)


def _fake_batch(*, with_burn: bool, with_reference: bool) -> dict:
    """Lote sintético con una muestra cuya verdad tiene quemado o está vacía."""
    B, T, H, W = 1, 7, 64, 64
    target = torch.zeros(B, 1, H, W)
    if with_burn:
        target[:, :, 20:40, 20:40] = 1.0
    mask = torch.ones(B, 1, H, W, dtype=torch.uint8)
    return {
        "x_olci": torch.zeros(B, T, 5, H, W),
        "x_dem": torch.zeros(B, 1, H, W),
        "x_worldcover": torch.zeros(B, 1, H, W),
        "x_scar": torch.zeros(B, 1, H, W),
        "horizon_days": torch.ones(B, 1),
        "target": target,
        "mask": mask,
        "target_cems": target.clone() if with_reference else torch.zeros(B, 1, H, W),
        "mask_cems": mask.clone(),
        "has_reference": torch.ones(B, 1) if with_reference else torch.zeros(B, 1),
        "meta": {"series_id": ["serie_a"], "sample_id": ["m1"], "fold": [0]},
    }


def _evaluation_batch_with_horizons() -> dict:
    horizons = torch.tensor([[0.5], [2.0], [2.0]])
    target = torch.zeros((3, 1, 64, 64))
    target[0, 0, 10:20, 10:20] = 1.0
    mask = torch.ones((3, 1, 64, 64), dtype=torch.uint8)
    scar = torch.zeros((3, 1, 64, 64))
    scar[0, 0, 10:20, 10:20] = 1.0
    return {
        "x_olci": torch.zeros((3, 7, 5, 64, 64)),
        "x_dem": torch.zeros((3, 1, 64, 64)),
        "x_worldcover": torch.zeros((3, 1, 64, 64)),
        "x_scar": scar,
        "horizon_days": horizons,
        "target": target,
        "mask": mask,
        "target_cems": target.clone(),
        "mask_cems": mask.clone(),
        "has_reference": torch.ones((3, 1)),
        "meta": {
            "sample_id": ["m0", "m1", "m2"],
            "series_id": ["a", "b", "c"],
            "fold": [0, 0, 0],
            "scar_state": ["detected", "cold_start", "detected"],
        },
    }


def test_la_procedencia_del_dataset_incluye_hash_y_campos_requeridos(tmp_path):
    module = _load_train_script()
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({
        "generator_version": "generator",
        "dataset_index_schema_version": 2,
        "target_mode": "scar",
        "inventory": {
            "total_unique_samples": 3,
            "horizon_distribution": {"bands": {"0-1d": 3}},
            "scar_coverage": {"cold_start_samples": 0},
        },
    }), encoding="utf-8")

    provenance = module.build_dataset_provenance(
        index_path,
        json.loads(index_path.read_text(encoding="utf-8")),
        seed=42,
        tag="run",
    )

    assert provenance["dataset_index_sha256"]
    assert provenance["dataset_generator_version"] == "generator"
    assert provenance["dataset_schema_version"] == 2
    assert provenance["dataset_target_mode"] == "scar"
    assert provenance["seed"] == 42
    assert provenance["run_tag"] == "run"
    assert provenance["dataset_counts"]["total_unique_samples"] == 3
    assert "horizon_support" in provenance
    assert "cold_start_coverage" in provenance


def test_la_semilla_fija_el_modelo_y_el_generador_de_aleatoriedad() -> None:
    module = _load_train_script()
    from burn_scar.quality import state_dict_sha256

    module.set_reproducibility_seed(123)
    first_model = BurnScarPropagationModel(
        olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0
    )
    first_random = torch.rand(8)

    module.set_reproducibility_seed(123)
    second_model = BurnScarPropagationModel(
        olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0
    )
    second_random = torch.rand(8)

    assert torch.equal(first_random, second_random)
    assert state_dict_sha256(first_model.state_dict()) == state_dict_sha256(
        second_model.state_dict()
    )


def test_evaluate_fold_estratifica_horizontes_sin_filtrar_el_global() -> None:
    module = _load_train_script()
    model = BurnScarPropagationModel(
        olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0
    )

    result = module.evaluate_fold(model, [_evaluation_batch_with_horizons()], device="cpu")

    assert set(result["horizon_strata"]) == {"0-1d", "1-3d"}
    assert result["n_samples"] == 3
    assert result["horizon_strata"]["1-3d"]["assembly"]["n_samples_total"] == 2
    assert result["horizon_strata"]["1-3d"]["assembly"]["n_samples_evaluated"] == 0
    assert math.isnan(result["horizon_strata"]["1-3d"]["assembly"]["f1"])
    assert result["cold_start"]["represented"] is True
    assert result["cold_start"]["n_samples_total"] == 1


def test_evaluate_fold_declara_arranque_en_frio_no_representado() -> None:
    module = _load_train_script()
    model = BurnScarPropagationModel(
        olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0
    )
    batch = _evaluation_batch_with_horizons()
    batch["x_scar"][:] = 1.0
    batch["meta"]["scar_state"] = ["detected", "detected", "detected"]

    result = module.evaluate_fold(model, [batch], device="cpu")

    assert result["cold_start"]["represented"] is False
    assert "no está representado" in result["cold_start"]["warning"]


def test_evaluate_fold_informa_de_muestras_evaluables() -> None:
    """El informe de fold distingue muestras informativas de las que no lo son."""
    module = _load_train_script()
    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0)
    loader = [
        _fake_batch(with_burn=True, with_reference=True),
        _fake_batch(with_burn=False, with_reference=True),
    ]

    res = module.evaluate_fold(model, loader, device="cpu")

    # Dos muestras: solo la que tiene quemado en la verdad es evaluable.
    assert res["n_samples"] == 2
    assert res["n_samples_evaluated"] == 1
    assert res["per_threshold"][0.5]["n_samples_total"] == 2
    assert res["per_threshold"][0.5]["n_samples_evaluated"] == 1
    assert res["ref_n_samples_evaluated"] == 1
    assert res["ref_n_samples_total"] == 2


def test_evaluate_fold_sin_target_de_referencia() -> None:
    """Sin target de referencia, el informe lo marca como no disponible."""
    module = _load_train_script()
    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0)
    loader = [_fake_batch(with_burn=True, with_reference=False)]

    res = module.evaluate_fold(model, loader, device="cpu")

    assert res["ref_available"] is False
    assert res["ref_per_threshold"] == {}
    assert res["n_samples_evaluated"] == 1

