import pytest
import torch
from pathlib import Path
from burn_scar.dataset import BurnScarDataset, get_cv_dataloaders

INDEX_PATH = Path("data/processed/dataset_index.json")


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
