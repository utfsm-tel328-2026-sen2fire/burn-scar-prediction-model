from pathlib import Path

import pytest
import torch

from burn_scar.models import load_tmb_s3, predict_burn_scar

WEIGHTS_PATH = Path("models/tmb_s3/convlstm_segformer_prepost_subset/model.safetensors")


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pesos de TMB-S3 no descargados")
def test_load_tmb_s3_weights():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_tmb_s3(WEIGHTS_PATH, device=device)
    assert model is not None
    # Verificar que los pesos están congelados
    for p in model.parameters():
        assert not p.requires_grad


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pesos de TMB-S3 no descargados")
def test_tmb_s3_forward_batched():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_tmb_s3(WEIGHTS_PATH, device=device)

    # Batch de 2 secuencias de 7 frames con 6 canales (5 bandas + 1 landcover)
    x = torch.randn(2, 7, 6, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 7, 2, 64, 64)


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pesos de TMB-S3 no descargados")
def test_predict_burn_scar_probabilities():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_tmb_s3(WEIGHTS_PATH, device=device)

    # Entrada unbatched (T=7, C=6, 64, 64)
    x = torch.randn(7, 6, 64, 64, device=device)
    probs = predict_burn_scar(model, x)

    assert probs.shape == (7, 64, 64)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0
