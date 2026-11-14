#!/usr/bin/env python3
"""
Evaluación de Modelos de Referencia (Baselines B0 y B1) sobre los 5 Folds de CV.

Evalúa los baselines B0 (Persistencia) y B1 (Dilatación Radial) sobre las muestras
reales del dataset en píxeles válidos (mask == 1) promediando por evento (docs/dataset.md §10).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Asegurar import de burn_scar desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from burn_scar.dataset import BurnScarDataset
from burn_scar.baselines import (
    BaselineB0Persistence,
    BaselineB1RadialDilation,
    compute_metrics_numpy,
)


def evaluate_model(dataset: BurnScarDataset, model, model_name: str) -> dict[str, float]:
    """Evalúa un baseline promediando métricas por serie/evento (docs/dataset.md §10)."""
    series_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)

    for i in range(len(dataset)):
        sample = dataset[i]
        meta = sample["meta"]
        series_id = meta["series_id"]

        x_scar = sample["x_scar"].squeeze(0).numpy()
        h_days = sample["horizon_days"].item()
        target = sample["target"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()

        pred = model.predict(x_scar, horizon_days=h_days)
        m = compute_metrics_numpy(pred, target, mask, threshold=0.5)
        series_metrics[series_id].append(m)

    # Promedio por serie primero (D10: un evento = una serie)
    f1_per_series = []
    iou_per_series = []
    prec_per_series = []
    rec_per_series = []

    for sid, m_list in series_metrics.items():
        f1_per_series.append(np.mean([m["f1"] for m in m_list]))
        iou_per_series.append(np.mean([m["iou"] for m in m_list]))
        prec_per_series.append(np.mean([m["precision"] for m in m_list]))
        rec_per_series.append(np.mean([m["recall"] for m in m_list]))

    return {
        "model": model_name,
        "n_series": len(series_metrics),
        "n_samples": len(dataset),
        "f1_event_mean": float(np.mean(f1_per_series)),
        "f1_event_std": float(np.std(f1_per_series)),
        "iou_event_mean": float(np.mean(iou_per_series)),
        "precision_mean": float(np.mean(prec_per_series)),
        "recall_mean": float(np.mean(rec_per_series)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de Baselines B0/B1")
    parser.add_argument("--index", default="data/processed/dataset_index.json")
    args = parser.parse_args()

    print("=" * 72)
    print("EVALUACIÓN DE BASELINES DE REFERENCIA (docs/dataset.md §9-§10)")
    print("Métrica: Promedio por evento sobre píxeles válidos (umbral 0.5)")
    print("=" * 72)

    # Evaluar sobre el conjunto mediterráneo (folds 0..4)
    dataset_med = BurnScarDataset(index_path=args.index, folds=list(range(5)))

    models = [
        (BaselineB0Persistence(), "B0: Persistencia (Cero Crecimiento)"),
        (BaselineB1RadialDilation(growth_rate_km_per_day=0.6), "B1: Dilatación Radial Isótropa"),
    ]

    for model, name in models:
        res = evaluate_model(dataset_med, model, name)
        print(f"\nResultados: {name}")
        print(f"  Series evaluadas:  {res['n_series']} ({res['n_samples']} muestras)")
        print(f"  F1 (Macro-evento): {res['f1_event_mean'] * 100:.2f}% ± {res['f1_event_std'] * 100:.2f}%")
        print(f"  IoU (Jaccard):     {res['iou_event_mean'] * 100:.2f}%")
        print(f"  Precisión:         {res['precision_mean'] * 100:.2f}%")
        print(f"  Recall (Sensib.):  {res['recall_mean'] * 100:.2f}%")

    print("\n" + "=" * 72)
    print("✓ Baselines calculados. Cualquier red neuronal (M-A o M-B) debe superar B1/B2.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
