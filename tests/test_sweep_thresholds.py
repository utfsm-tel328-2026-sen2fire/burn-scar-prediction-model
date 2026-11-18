"""Pruebas unitarias para scripts/sweep_thresholds.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sweep_thresholds as st


def test_sweep_samples_removes_low_confidence_false_positives():
    """Al subir el umbral sobre falsos positivos de baja confianza, sube la precisión."""
    mask = np.ones((8, 8), dtype=np.uint8)
    target = np.zeros((8, 8), dtype=np.float32)
    target[0:2, 0:2] = 1.0  # 4 píxeles quemados reales

    # Verdaderos positivos con confianza alta (0.85) y falsos positivos con confianza baja (0.20)
    probs = np.zeros((8, 8), dtype=np.float32)
    probs[0:2, 0:2] = 0.85
    probs[4:8, 4:8] = 0.20  # 16 píxeles FP con baja confianza

    samples = [
        {"series_id": "S1", "probs": probs, "target": target, "mask": mask},
    ]

    res = st.sweep_samples_over_thresholds(samples, [0.1, 0.5, 0.8])

    # Con umbral 0.1 entran los 16 FP de baja confianza -> precisión baja (4/20 = 0.20)
    assert np.isclose(res[0.1]["precision"], 0.20)
    assert np.isclose(res[0.1]["recall"], 1.0)
    assert np.isclose(res[0.1]["area_ratio"], 5.0)

    # Con umbral 0.5 desaparecen los FP -> F1 = 1.0, área = 1.0x
    assert np.isclose(res[0.5]["precision"], 1.0)
    assert np.isclose(res[0.5]["recall"], 1.0)
    assert np.isclose(res[0.5]["f1"], 1.0)
    assert np.isclose(res[0.5]["area_ratio"], 1.0)


def test_aggregate_across_checkpoints_averages_folds():
    """Promedia correctamente las métricas entre múltiples folds/checkpoints."""
    fold0 = {
        0.5: {"f1": 0.4, "iou": 0.25, "precision": 0.5, "recall": 0.3, "area_ratio": 1.2},
    }
    fold1 = {
        0.5: {"f1": 0.6, "iou": 0.45, "precision": 0.7, "recall": 0.5, "area_ratio": 0.8},
    }
    summary = st.aggregate_across_checkpoints([fold0, fold1], [0.5])
    assert np.isclose(summary[0.5]["f1"], 0.5)
    assert np.isclose(summary[0.5]["iou"], 0.35)
    assert np.isclose(summary[0.5]["precision"], 0.6)
    assert np.isclose(summary[0.5]["recall"], 0.4)
    assert np.isclose(summary[0.5]["area_ratio"], 1.0)
