"""Tests unitarios para el modelo de propagación y su pérdida enmascarada."""

from __future__ import annotations

import importlib.util
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

