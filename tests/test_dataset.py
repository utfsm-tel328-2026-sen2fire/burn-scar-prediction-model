import json

import pytest
import torch
from pathlib import Path
from burn_scar.dataset import BurnScarDataset, get_cv_dataloaders, resolve_loader_options

INDEX_PATH = Path("data/processed/dataset_index.json")


def test_resolve_loader_options_desactiva_workers_en_rocm():
    """En AMD los workers con pin_memory son patológicos: se desactivan."""
    num_workers, pin_memory = resolve_loader_options()
    if torch.version.hip:
        assert num_workers == 0
        assert pin_memory is False
    else:
        assert num_workers == 2


def test_resolve_loader_options_respeta_valores_explicitos():
    assert resolve_loader_options(num_workers=4, pin_memory=True) == (4, True)
    assert resolve_loader_options(num_workers=0, pin_memory=False) == (0, False)


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Dataset index no generado")
def test_burn_scar_dataset_loading():
    dataset = BurnScarDataset(index_path=INDEX_PATH, folds=[0])
    assert len(dataset) > 0

    sample = dataset[0]
    assert "x_olci" in sample
    assert "x_dem" in sample
    assert "x_worldcover" in sample
    assert "x_scar" in sample
    assert "horizon_days" in sample
    assert "target" in sample
    assert "mask" in sample

    assert sample["x_olci"].shape == (7, 5, 64, 64)
    assert sample["x_dem"].shape == (1, 64, 64)
    assert sample["x_worldcover"].shape == (1, 64, 64)
    assert sample["x_scar"].shape == (1, 64, 64)
    assert sample["target"].shape == (1, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Dataset index no generado")
def test_cv_dataloaders_split():
    train_loader, val_loader = get_cv_dataloaders(val_fold=0, batch_size=4, num_workers=0, index_path=INDEX_PATH)

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    # Los datasets deben ser disjuntos
    assert len(train_loader.dataset) + len(val_loader.dataset) <= 390

    # Test batch iteration
    batch = next(iter(train_loader))
    assert batch["x_olci"].shape == (4, 7, 5, 64, 64)
    assert batch["target"].shape == (4, 1, 64, 64)


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Dataset index no generado")
def test_dataset_expone_target_de_referencia():
    """Cada muestra trae el crecimiento de referencia y su validez común."""
    dataset = BurnScarDataset(index_path=INDEX_PATH, folds=[0])
    sample = dataset[0]

    assert sample["target_cems"].shape == (1, 64, 64)
    assert sample["mask_cems"].shape == (1, 64, 64)
    assert float(sample["has_reference"].item()) == 1.0
    assert float(sample["target_cems"].min()) >= 0.0
    assert float(sample["target_cems"].max()) <= 1.0
    assert set(torch.unique(sample["mask_cems"]).tolist()) <= {0, 1}


def test_dataset_tolera_muestras_sin_target_de_referencia(tmp_path):
    """Un dataset antiguo (sin target_cems) se carga marcando la referencia como no disponible."""
    import numpy as np

    sample_file = tmp_path / "muestra.npz"
    np.savez_compressed(
        sample_file,
        x_olci=np.zeros((7, 5, 64, 64), dtype=np.float16),
        x_dem=np.zeros((64, 64), dtype=np.float32),
        x_worldcover=np.zeros((64, 64), dtype=np.uint8),
        x_scar=np.zeros((64, 64), dtype=np.float16),
        horizon_days=np.float32(1.0),
        target=np.zeros((64, 64), dtype=np.float16),
        mask=np.ones((64, 64), dtype=np.uint8),
    )
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({
        "samples": [{
            "sample_id": "muestra",
            "series_id": "serie",
            "file": str(sample_file),
            "fold": 0,
            "horizon_days": 1.0,
        }]
    }))

    sample = BurnScarDataset(index_path=index_file)[0]
    assert float(sample["has_reference"].item()) == 0.0
    assert float(sample["target_cems"].abs().sum()) == 0.0
