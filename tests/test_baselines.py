import numpy as np
import pytest
from burn_scar.baselines import (
    BaselineB0Persistence,
    BaselineB1RadialDilation,
    aggregate_series_metrics,
    compute_metrics_numpy,
)


def test_b0_persistence():
    b0 = BaselineB0Persistence()
    scar = np.zeros((64, 64), dtype=np.float32)
    scar[20:30, 20:30] = 1.0

    pred = b0.predict(scar, horizon_days=3.0)
    assert pred.shape == (64, 64)
    # B0 predice cero crecimiento
    assert np.all(pred == 0.0)


def test_b1_radial_dilation():
    b1 = BaselineB1RadialDilation(growth_rate_km_per_day=0.6)
    scar = np.zeros((64, 64), dtype=np.float32)
    scar[30:34, 30:34] = 1.0

    pred = b1.predict(scar, horizon_days=1.0)
    assert pred.shape == (64, 64)
    # Debe predecir crecimiento alrededor de la cicatriz previa
    assert (pred > 0).sum() > 0
    # No debe predecir dentro de la cicatriz vieja
    assert np.all(pred[scar == 1.0] == 0.0)


def test_compute_metrics():
    pred = np.zeros((64, 64), dtype=np.float32)
    target = np.zeros((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=np.uint8)

    # Coincidencia perfecta
    pred[10:20, 10:20] = 1.0
    target[10:20, 10:20] = 1.0

    metrics = compute_metrics_numpy(pred, target, mask, threshold=0.5)
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_metrics_are_nan_when_target_has_no_burn():
    """Sin quemado en la verdad las métricas no son informativas: NaN, no 1.0."""
    pred = np.zeros((64, 64), dtype=np.float32)
    target = np.zeros((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=np.uint8)

    metrics = compute_metrics_numpy(pred, target, mask, threshold=0.5)

    assert np.isnan(metrics["f1"])
    assert np.isnan(metrics["iou"])
    assert np.isnan(metrics["precision"])
    assert np.isnan(metrics["recall"])
    assert metrics["n_positives"] == 0


def test_aggregate_series_metrics_ignores_nan():
    """El agregado por evento ignora las muestras no informativas y las cuenta."""
    vacio = compute_metrics_numpy(
        np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32),
        np.ones((8, 8), np.uint8),
    )
    con_quemado = compute_metrics_numpy(
        np.ones((8, 8), np.float32), np.ones((8, 8), np.float32),
        np.ones((8, 8), np.uint8),
    )

    agg = aggregate_series_metrics({"serie_a": [vacio, con_quemado], "serie_b": [vacio]})

    assert agg["n_series_evaluated"] == 1
    assert agg["n_series_total"] == 2
    assert agg["f1_event_mean"] == 1.0

