import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from burn_scar.baselines import (
    BaselineAllBurned,
    BaselineB0Persistence,
    BaselineB1RadialDilation,
    aggregate_series_metrics,
    compute_metrics_numpy,
)


def _load_baselines_script():
    """Carga `scripts/evaluate_baselines.py` como módulo para probar sus utilidades."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_eval_threshold_por_defecto() -> None:
    """Sin opciones, el umbral histórico 0,50."""
    mod = _load_baselines_script()

    threshold, source = mod.resolve_eval_threshold(None, None)

    assert threshold == 0.5
    assert "por defecto" in source


def test_resolve_eval_threshold_explicito() -> None:
    """Un umbral explícito se usa tal cual."""
    mod = _load_baselines_script()

    threshold, source = mod.resolve_eval_threshold(0.1, None)

    assert threshold == 0.1
    assert source == "explícito"


def test_resolve_eval_threshold_desde_el_run(tmp_path: Path) -> None:
    """El umbral operativo congelado del run se lee de su JSON."""
    mod = _load_baselines_script()
    (tmp_path / "operational_threshold.json").write_text(
        json.dumps({"operational_threshold": 0.1}), encoding="utf-8"
    )

    threshold, source = mod.resolve_eval_threshold(None, tmp_path)

    assert threshold == 0.1
    assert "umbral operativo del run" in source


def test_resolve_eval_threshold_rechaza_conflicto(tmp_path: Path) -> None:
    """Indicar las dos opciones a la vez es un error explícito."""
    mod = _load_baselines_script()
    (tmp_path / "operational_threshold.json").write_text(
        json.dumps({"operational_threshold": 0.1}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="una sola forma"):
        mod.resolve_eval_threshold(0.5, tmp_path)


def test_resolve_eval_threshold_error_si_falta_el_json(tmp_path: Path) -> None:
    """Un directorio sin `operational_threshold.json` es un error, no un 0,50 silencioso."""
    mod = _load_baselines_script()

    with pytest.raises(FileNotFoundError, match="operational_threshold.json"):
        mod.resolve_eval_threshold(None, tmp_path)


def test_b0_persistence():
    b0 = BaselineB0Persistence()
    scar = np.zeros((64, 64), dtype=np.float32)
    scar[20:30, 20:30] = 1.0

    pred = b0.predict(scar, horizon_days=3.0)
    assert pred.shape == (64, 64)
    # B0 predice cero crecimiento
    assert np.all(pred == 0.0)


def test_trivial_todo_quemado():
    """El trivial es la escala mínima de cada etiqueta: predice todo quemado."""
    trivial = BaselineAllBurned()
    scar = np.zeros((64, 64), dtype=np.float32)
    scar[20:30, 20:30] = 1.0

    pred = trivial.predict(scar, horizon_days=1.0)
    assert pred.shape == (64, 64)
    assert np.all(pred == 1.0)


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


def test_aggregate_series_metrics_cuenta_muestras_evaluables():
    """El agregado informa de cuántas muestras son evaluables y cuántas hay en total."""
    vacio = compute_metrics_numpy(
        np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32),
        np.ones((8, 8), np.uint8),
    )
    con_quemado = compute_metrics_numpy(
        np.ones((8, 8), np.float32), np.ones((8, 8), np.float32),
        np.ones((8, 8), np.uint8),
    )

    agg = aggregate_series_metrics({"serie_a": [vacio, con_quemado], "serie_b": [vacio]})

    # 3 muestras en total; solo la de "serie_a" con quemado es informativa.
    assert agg["n_samples_total"] == 3
    assert agg["n_samples_evaluated"] == 1


def test_aggregate_series_metrics_sin_muestras_evaluables():
    """Un conjunto entero no informativo no inventa recuentos."""
    vacio = compute_metrics_numpy(
        np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32),
        np.ones((8, 8), np.uint8),
    )

    agg = aggregate_series_metrics({"serie_a": [vacio, vacio]})

    assert agg["n_samples_total"] == 2
    assert agg["n_samples_evaluated"] == 0
    assert agg["n_series_evaluated"] == 0
    assert np.isnan(agg["f1_event_mean"])

