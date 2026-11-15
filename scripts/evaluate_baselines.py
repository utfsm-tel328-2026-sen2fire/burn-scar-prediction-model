#!/usr/bin/env python3
"""
Evaluación de Modelos de Referencia (Baselines B0 y B1) sobre los 5 Folds de CV.

Evalúa los baselines B0 (Persistencia) y B1 (Dilatación Radial) sobre las muestras
reales del dataset en píxeles válidos (mask == 1) promediando por evento (docs/dataset.md §10).

Informa además del trivial "todo quemado" y de las métricas contra el target de
referencia (crecimiento real entre rondas CEMS), para que cada F1 se lea con la
escala de su propia etiqueta y no se comparen F1 de etiquetas distintas.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Asegurar import de burn_scar desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from burn_scar.dataset import BurnScarDataset
from burn_scar.baselines import (
    BaselineAllBurned,
    BaselineB0Persistence,
    BaselineB1RadialDilation,
    aggregate_series_metrics,
    compute_metrics_numpy,
)

# Umbral por defecto: el histórico de esta herramienta (predicciones binarias).
DEFAULT_THRESHOLD = 0.5


def resolve_eval_threshold(
    threshold: float | None,
    run_dir: str | Path | None,
) -> tuple[float, str]:
    """
    Resuelve el umbral de evaluación y declara su origen.

    El umbral se aplica a la vez a la predicción y a la verdad. En los baselines
    la predicción es binaria, así que el umbral no la cambia: determina qué
    píxeles de la verdad cuentan como quemado y, con ello, qué muestras son
    evaluables. Para que el F1 del modelo y el de los baselines se comparen
    sobre la misma discretización, ambos deben medirse al mismo umbral: el
    operativo congelado de un run, leído de `operational_threshold.json`.

    Devuelve `(umbral, origen)`.
    """
    if threshold is not None and run_dir is not None:
        raise ValueError(
            "Indica el umbral de una sola forma: o `--threshold` explícito, o "
            "`--operational-threshold-from` con el directorio de checkpoints de un run."
        )

    if threshold is not None:
        return float(threshold), "explícito"

    if run_dir is not None:
        thr_file = Path(run_dir) / "operational_threshold.json"
        if not thr_file.is_file():
            raise FileNotFoundError(
                f"No se encuentra el umbral operativo en {thr_file}. El directorio debe "
                "contener el `operational_threshold.json` del run entrenado."
            )
        payload = json.loads(thr_file.read_text(encoding="utf-8"))
        tau = float(payload["operational_threshold"])
        return tau, f"umbral operativo del run {run_dir}"

    return DEFAULT_THRESHOLD, "por defecto (0,50)"


def evaluate_model(
    dataset: BurnScarDataset,
    model,
    model_name: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evalúa un baseline promediando métricas por serie/evento (docs/dataset.md §10)."""
    series_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    series_metrics_ref: dict[str, list[dict[str, float]]] = defaultdict(list)

    for i in range(len(dataset)):
        sample = dataset[i]
        meta = sample["meta"]
        series_id = meta["series_id"]

        x_scar = sample["x_scar"].squeeze(0).numpy()
        h_days = sample["horizon_days"].item()
        target = sample["target"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()

        pred = model.predict(x_scar, horizon_days=h_days)
        m = compute_metrics_numpy(pred, target, mask, threshold=threshold)
        series_metrics[series_id].append(m)

        # Target de referencia (crecimiento real entre rondas CEMS), si existe
        if float(sample["has_reference"].item()) > 0:
            ref_target = sample["target_cems"].squeeze(0).numpy()
            ref_mask = sample["mask_cems"].squeeze(0).numpy()
            mr = compute_metrics_numpy(pred, ref_target, ref_mask, threshold=threshold)
            series_metrics_ref[series_id].append(mr)

    # Promedio por serie primero (D10: un evento = una serie), ignorando NaN
    agg = aggregate_series_metrics(series_metrics)
    ref_agg = (
        aggregate_series_metrics(series_metrics_ref) if series_metrics_ref else None
    )

    out = {
        "model": model_name,
        "n_series": agg["n_series_total"],
        "n_series_evaluated": agg["n_series_evaluated"],
        "n_samples": len(dataset),
        "n_samples_evaluated": agg["n_samples_evaluated"],
        "f1_event_mean": agg["f1_event_mean"],
        "f1_event_std": agg["f1_event_std"],
        "iou_event_mean": agg["iou_event_mean"],
        "precision_mean": agg["precision_event_mean"],
        "recall_mean": agg["recall_event_mean"],
        "ref_available": ref_agg is not None,
        "ref_f1_event_mean": ref_agg["f1_event_mean"] if ref_agg else float("nan"),
        "ref_iou_event_mean": ref_agg["iou_event_mean"] if ref_agg else float("nan"),
        "ref_precision_mean": ref_agg["precision_event_mean"] if ref_agg else float("nan"),
        "ref_recall_mean": ref_agg["recall_event_mean"] if ref_agg else float("nan"),
        "ref_n_series_evaluated": ref_agg["n_series_evaluated"] if ref_agg else 0,
        "ref_n_samples_evaluated": ref_agg["n_samples_evaluated"] if ref_agg else 0,
    }
    return out


def _pct(value: float) -> str:
    return "n/d" if np.isnan(value) else f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de Baselines B0/B1")
    parser.add_argument("--index", default="data/processed/dataset_index.json")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Umbral de evaluación (se aplica a la predicción y a la verdad). "
            "Por defecto 0,50."
        ),
    )
    parser.add_argument(
        "--operational-threshold-from",
        default=None,
        metavar="CHECKPOINT_DIR",
        help=(
            "Directorio de checkpoints de un run entrenado: usa su umbral operativo "
            "congelado (operational_threshold.json) para comparar con el modelo."
        ),
    )
    args = parser.parse_args()

    try:
        threshold, threshold_source = resolve_eval_threshold(
            args.threshold, args.operational_threshold_from
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=" * 72)
    print("EVALUACIÓN DE BASELINES DE REFERENCIA (docs/dataset.md §9-§10)")
    print(f"Métrica: Promedio por evento sobre píxeles válidos (umbral {threshold:.2f})")
    print(f"Umbral: {threshold:.2f} ({threshold_source})")
    print(
        "Ojo: los baselines predicen de forma binaria, así que el umbral no cambia su "
        "predicción;"
    )
    print(
        "  determina qué píxeles de la verdad cuentan como quemado y qué muestras son "
        "evaluables."
    )
    print("=" * 72)

    # Evaluar sobre el conjunto mediterráneo (folds 0..4) y sobre el test
    dataset_med = BurnScarDataset(index_path=args.index, folds=list(range(5)))
    dataset_test = BurnScarDataset(index_path=args.index, folds=[-1])

    models = [
        (BaselineB0Persistence(), "B0: Persistencia (Cero Crecimiento)"),
        (BaselineB1RadialDilation(growth_rate_km_per_day=0.6), "B1: Dilatación Radial Isótropa"),
        (BaselineAllBurned(), "TRIVIAL: Todo quemado (escala mínima, no es un modelo)"),
    ]

    for dataset, nombre_conjunto in (
        (dataset_med, "CONJUNTO DE VALIDACIÓN (folds 0-4)"),
        (dataset_test, "CONJUNTO DE TEST (fold -1, fuera de dominio)"),
    ):
        print("\n" + "#" * 72)
        print(f"# {nombre_conjunto}")
        print("#" * 72)
        for model, name in models:
            res = evaluate_model(dataset, model, name, threshold=threshold)
            print(f"\nResultados: {name}")
            print(f"  Series evaluadas:  {res['n_series_evaluated']}/{res['n_series']} "
                  f"({res['n_samples']} muestras)")
            print(f"  Muestras evaluables: {res['n_samples_evaluated']}/{res['n_samples']}")
            print(f"  F1 (Macro-evento): {_pct(res['f1_event_mean'])} ± "
                  f"{_pct(res['f1_event_std'])}")
            print(f"  IoU (Jaccard):     {_pct(res['iou_event_mean'])}")
            print(f"  Precisión:         {_pct(res['precision_mean'])}")
            print(f"  Recall (Sensib.):  {_pct(res['recall_mean'])}")
            if res["ref_available"]:
                print("  --- contra el target de referencia (crecimiento real CEMS) ---")
                print(f"  Series evaluadas:  {res['ref_n_series_evaluated']}/{res['n_series']}")
                print(f"  Muestras evaluables: {res['ref_n_samples_evaluated']}/{res['n_samples']}")
                print(f"  F1 (Macro-evento): {_pct(res['ref_f1_event_mean'])}")
                print(f"  IoU (Jaccard):     {_pct(res['ref_iou_event_mean'])}")
                print(f"  Precisión:         {_pct(res['ref_precision_mean'])}")
                print(f"  Recall (Sensib.):  {_pct(res['ref_recall_mean'])}")
            else:
                print("  --- target de referencia: no disponible ---")

    print("\n" + "=" * 72)
    print("✓ Baselines calculados. Cualquier red neuronal (M-A o M-B) debe superar B1.")
    print("  Los F1 de etiquetas distintas NO son comparables entre sí: cada uno se")
    print("  lee contra el trivial 'todo quemado' de su propia etiqueta.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
