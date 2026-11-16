import json

import pytest
import torch
from pathlib import Path
from burn_scar.dataset import (
    BurnScarDataset,
    compute_generator_version,
    get_cv_dataloaders,
    resolve_loader_options,
)

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
    # Codificación canónica: 1 = evaluable, 255 = ignorado
    assert set(torch.unique(sample["mask_cems"]).tolist()) <= {1, 255}


def test_mascaras_usan_la_codificacion_canonica(tmp_path):
    """Ambas máscaras solo contienen 1 (evaluable) y 255 (ignorado)."""
    import numpy as np

    mask = np.array([[1, 255], [1, 1]], dtype=np.uint8)
    mask_cems = np.array([[255, 255], [1, 1]], dtype=np.uint8)

    sample = _write_sample(tmp_path, mask=mask, mask_cems=mask_cems)

    assert set(torch.unique(sample["mask"]).tolist()) <= {1, 255}
    assert set(torch.unique(sample["mask_cems"]).tolist()) <= {1, 255}
    assert sample["mask_cems"][0, 0, 0].item() == 255
    assert sample["mask_cems"][0, 1, 1].item() == 1


def test_cargador_normaliza_mascaras_antiguas(tmp_path):
    """Un paquete antiguo con 0 = no evaluable se carga como 255 = ignorado."""
    import numpy as np

    mask = np.array([[1, 0], [255, 1]], dtype=np.uint8)
    mask_cems = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    sample = _write_sample(tmp_path, mask=mask, mask_cems=mask_cems)

    assert set(torch.unique(sample["mask"]).tolist()) <= {1, 255}
    assert set(torch.unique(sample["mask_cems"]).tolist()) <= {1, 255}
    assert sample["mask"][0, 0, 1].item() == 255
    assert sample["mask_cems"][0, 1, 1].item() == 255


def test_muestra_sin_ronda_previa_marca_la_validez_como_no_evaluable(tmp_path):
    """Sin ronda CEMS previa, la validez común es íntegramente no evaluable."""
    import numpy as np

    sample = _write_sample(
        tmp_path,
        mask=np.ones((64, 64), dtype=np.uint8),
        mask_cems=np.full((64, 64), 255, dtype=np.uint8),
    )

    assert set(torch.unique(sample["mask_cems"]).tolist()) == {255}


def _write_sample(tmp_path, *, mask, mask_cems=None):
    """Escribe una muestra y su índice, y la devuelve cargada por el dataset."""
    import numpy as np

    payload = {
        "x_olci": np.zeros((7, 5, 64, 64), dtype=np.float16),
        "x_dem": np.zeros((64, 64), dtype=np.float32),
        "x_worldcover": np.zeros((64, 64), dtype=np.uint8),
        "x_scar": np.zeros((64, 64), dtype=np.float16),
        "horizon_days": np.float32(1.0),
        "target": np.zeros((64, 64), dtype=np.float16),
        "mask": mask,
    }
    if mask_cems is not None:
        payload["target_cems"] = np.zeros((64, 64), dtype=np.float16)
        payload["mask_cems"] = mask_cems

    sample_file = tmp_path / "muestra.npz"
    np.savez_compressed(sample_file, **payload)
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({
        "generator_version": compute_generator_version(),
        "samples": [{
            "sample_id": "muestra",
            "series_id": "serie",
            "file": str(sample_file),
            "fold": 0,
            "horizon_days": 1.0,
        }]
    }))
    return BurnScarDataset(index_path=index_file)[0]


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
        "generator_version": compute_generator_version(),
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
