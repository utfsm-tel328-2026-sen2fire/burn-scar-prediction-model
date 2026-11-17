#!/usr/bin/env python3
"""
Barrido rápido de umbrales de probabilidad sobre checkpoints de propagación
(ConvLSTM + SegFormer) sin reentrenar.

Ejecuta una sola pasada de inferencia por checkpoint y evalúa múltiples umbrales
para comparar Precision, Recall, F1, IoU y razón de área predicha vs. real.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from burn_scar.baselines import aggregate_series_metrics, compute_metrics_numpy
from burn_scar.dataset import BurnScarDataset, get_cv_dataloaders
from burn_scar.visualization import load_propagation_checkpoint

TEST_FOLD = -1
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa uno o varios checkpoints de propagación a través de múltiples "
            "umbrales de probabilidad en val o test."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta a un checkpoint individual (ej. best_fold_0.pt)",
    )
    src.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directorio de run con checkpoints best_fold_*.pt",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Split a evaluar: 'val' (fold de cada checkpoint) o 'test' (fold -1). Default: val",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Lista de umbrales en (0, 1] a evaluar (default: 0.1 .. 0.9)",
    )
    parser.add_argument(
        "--index",
        default="data/processed/dataset_index.json",
        help="Ruta al índice JSON del dataset procesado",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold explícito (solo cuando se usa --checkpoint y --split=val)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta opcional para guardar los resultados en JSON",
    )
    parser.add_argument(
        "--dedup-index",
        action="store_true",
        help=(
            "Deduplica temporalmente sample_id repetidos en el índice JSON "
            "(útil para evaluar runs antiguos con índices previos al pulido del dataset)"
        ),
    )
    return parser


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> list[dict[str, object]]:
    """Ejecuta una sola pasada de inferencia y guarda probs, target, mask y series_id."""
    model.eval()
    samples: list[dict[str, object]] = []
    for batch in loader:
        x_olci = batch["x_olci"].to(device)
        x_dem = batch["x_dem"].to(device)
        x_wc = batch["x_worldcover"].to(device)
        x_scar = batch["x_scar"].to(device)
        h_days = batch["horizon_days"].to(device)

        logits = model(x_olci, x_dem, x_wc, x_scar, h_days)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        tgt_np = batch["target"].squeeze(1).cpu().numpy()
        msk_np = batch["mask"].squeeze(1).cpu().numpy()
        series_ids = batch["meta"]["series_id"]

        for i, sid in enumerate(series_ids):
            samples.append(
                {
                    "series_id": str(sid),
                    "probs": probs[i],
                    "target": tgt_np[i],
                    "mask": msk_np[i],
                }
            )
    return samples


def sweep_samples_over_thresholds(
    samples: Sequence[dict[str, object]],
    thresholds: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Calcula métricas macro-evento y razón de área predicha/real para cada umbral."""
    results: dict[float, dict[str, float]] = {}
    for th in thresholds:
        series_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        pred_px_total = 0
        true_px_total = 0
        for s in samples:
            probs = s["probs"]
            target = s["target"]
            mask = s["mask"]
            m = compute_metrics_numpy(probs, target, mask, threshold=th)
            series_metrics[str(s["series_id"])].append(m)

            valid = mask == 1
            pred_px_total += int((probs[valid] >= th).sum())
            true_px_total += int((target[valid] >= th).sum())

        agg = aggregate_series_metrics(series_metrics)
        area_ratio = (
            float(pred_px_total / true_px_total)
            if true_px_total > 0
            else float("nan")
        )
        results[float(th)] = {
            "f1": agg["f1_event_mean"],
            "iou": agg["iou_event_mean"],
            "precision": agg["precision_event_mean"],
            "recall": agg["recall_event_mean"],
            "area_ratio": area_ratio,
            "pred_px": pred_px_total,
            "true_px": true_px_total,
            "n_series_evaluated": agg["n_series_evaluated"],
            "n_samples_evaluated": agg["n_samples_evaluated"],
            "n_samples_total": agg["n_samples_total"],
        }
    return results


def _nanmean(vals: Sequence[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _fmt_pct(val: float) -> str:
    return "   n/d" if np.isnan(val) else f"{val * 100:5.2f}%"


def _fmt_ratio(val: float) -> str:
    return "  n/d" if np.isnan(val) else f"{val:5.2f}x"


def aggregate_across_checkpoints(
    per_ckpt: Sequence[dict[float, dict[str, float]]],
    thresholds: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Promedia las métricas por umbral entre múltiples checkpoints/folds."""
    summary: dict[float, dict[str, float]] = {}
    for th in thresholds:
        f_th = float(th)
        summary[f_th] = {
            "f1": _nanmean([c[f_th]["f1"] for c in per_ckpt]),
            "iou": _nanmean([c[f_th]["iou"] for c in per_ckpt]),
            "precision": _nanmean([c[f_th]["precision"] for c in per_ckpt]),
            "recall": _nanmean([c[f_th]["recall"] for c in per_ckpt]),
            "area_ratio": _nanmean([c[f_th]["area_ratio"] for c in per_ckpt]),
        }
    return summary


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint: {path}")
        return [path]

    ckpt_dir = Path(args.checkpoint_dir)
    paths = sorted(ckpt_dir.glob("best_fold_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No hay checkpoints best_fold_*.pt en {ckpt_dir}")
    return paths


def prepare_index_path(index_path: str, dedup: bool) -> tuple[str, Path | None]:
    """Si --dedup-index está activo y hay duplicados, crea un índice temporal deduplicado."""
    if not dedup:
        return index_path, None

    src = Path(index_path)
    payload = json.loads(src.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    seen: set[str] = set()
    deduped = []
    removed = 0
    for s in samples:
        sid = str(s.get("sample_id"))
        if sid in seen:
            removed += 1
            continue
        seen.add(sid)
        deduped.append(s)

    if removed == 0:
        return index_path, None

    print(f"⚠ Aviso: deduplicadas {removed} entradas repetidas en {index_path} (solo en memoria temporal).")
    payload["samples"] = deduped
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="dedup_index_", delete=False, encoding="utf-8"
    )
    with tmp:
        json.dump(payload, tmp)
    return tmp.name, Path(tmp.name)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for th in args.thresholds:
        if not (0.0 < th <= 1.0):
            raise ValueError(f"Umbral inválido ({th}): debe estar en (0, 1].")

    ckpt_paths = resolve_checkpoints(args)
    thresholds = [float(t) for t in args.thresholds]
    resolved_index, tmp_index_path = prepare_index_path(args.index, args.dedup_index)

    print("=" * 74)
    print(f"BARRIDO DE UMBRALES | split={args.split} | checkpoints={len(ckpt_paths)}")
    print(f"Índice: {args.index} | Dispositivo: {args.device}")
    print("=" * 74)

    per_ckpt_results: list[dict[float, dict[str, float]]] = []
    ckpt_details: list[dict[str, object]] = []

    try:
        for path in ckpt_paths:
            model, resolved_fold, _ = load_propagation_checkpoint(
                path,
                fold=args.fold if args.checkpoint else None,
                device=args.device,
            )
            if args.split == "val":
                _, loader = get_cv_dataloaders(
                    val_fold=resolved_fold,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    index_path=resolved_index,
                )
            else:
                test_ds = BurnScarDataset(index_path=resolved_index, folds=[TEST_FOLD])
                if len(test_ds) == 0:
                    raise RuntimeError("El split de test (fold -1) está vacío.")
                loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )

            samples = collect_predictions(model, loader, args.device)
            res = sweep_samples_over_thresholds(samples, thresholds)
            per_ckpt_results.append(res)
            ckpt_details.append({"checkpoint": str(path), "fold": resolved_fold, "metrics": res})
            print(f"  Evaluado {path.name} (fold={resolved_fold}, muestras={len(samples)})")
    finally:
        if tmp_index_path is not None and tmp_index_path.exists():
            tmp_index_path.unlink()

    summary = aggregate_across_checkpoints(per_ckpt_results, thresholds)
    best_th = max(
        thresholds,
        key=lambda t: summary[t]["f1"] if not np.isnan(summary[t]["f1"]) else -1.0,
    )

    print("\n" + "-" * 74)
    print(f"{'Umbral':>8} | {'F1':>7} | {'IoU':>7} | {'Precisión':>9} | {'Recall':>7} | {'Área Pred/Real':>14}")
    print("-" * 74)
    for th in thresholds:
        row = summary[th]
        marker = "  <-- mejor F1" if th == best_th else ""
        print(
            f"{th:8.2f} | {_fmt_pct(row['f1']):>7} | {_fmt_pct(row['iou']):>7} | "
            f"{_fmt_pct(row['precision']):>9} | {_fmt_pct(row['recall']):>7} | "
            f"{_fmt_ratio(row['area_ratio']):>14}{marker}"
        )
    print("-" * 74)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "index": args.index,
            "thresholds": thresholds,
            "best_threshold_by_f1": best_th,
            "summary": summary,
            "per_checkpoint": ckpt_details,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Resultados guardados en: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
