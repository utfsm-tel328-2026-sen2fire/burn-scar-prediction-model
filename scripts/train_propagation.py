#!/usr/bin/env python3
"""
Entrenamiento y evaluación en 5 folds (CV agrupada por evento) del modelo de
propagación ConvLSTM + SegFormer (docs/operacion.md §4, docs/dataset.md §10).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from burn_scar.baselines import aggregate_series_metrics, compute_metrics_numpy
from burn_scar.dataset import BurnScarDataset, get_cv_dataloaders, resolve_loader_options
from burn_scar.models import BurnScarPropagationModel, masked_bce_dice_loss
from burn_scar.tracking import ExperimentTracker

# Umbrales evaluados en validación: un modelo probabilístico desbalanceado
# raramente opera en 0,5, y los baselines B0/B1 no tienen umbral (son binarios).
EVAL_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_THRESHOLD = 0.5

# Fold de test: series fuera de dominio (no mediterráneas), nunca visto en
# validación ni en la elección del umbral.
TEST_FOLD = -1


@torch.no_grad()
def evaluate_fold(
    model: BurnScarPropagationModel,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evalúa un fold promediando métricas por serie/evento (D10)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # Métricas por serie y por umbral (ignora NaN: targets sin quemado)
    series_metrics: dict[str, dict[float, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for batch in val_loader:
        x_olci = batch["x_olci"].to(device)
        x_dem = batch["x_dem"].to(device)
        x_wc = batch["x_worldcover"].to(device)
        x_scar = batch["x_scar"].to(device)
        h_days = batch["horizon_days"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        logits = model(x_olci, x_dem, x_wc, x_scar, h_days)
        loss = masked_bce_dice_loss(logits, target, mask)
        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        tgt_np = target.squeeze(1).cpu().numpy()
        msk_np = mask.squeeze(1).cpu().numpy()
        series_ids = batch["meta"]["series_id"]

        for i, sid in enumerate(series_ids):
            for th in EVAL_THRESHOLDS:
                m = compute_metrics_numpy(probs[i], tgt_np[i], msk_np[i], threshold=th)
                series_metrics[sid][th].append(m)

    # Agregado por serie para cada umbral
    per_threshold = {
        th: aggregate_series_metrics({sid: ms[th] for sid, ms in series_metrics.items()})
        for th in EVAL_THRESHOLDS
    }
    at_default = per_threshold[threshold]
    best_th = max(
        EVAL_THRESHOLDS,
        key=lambda th: (
            per_threshold[th]["f1_event_mean"]
            if not np.isnan(per_threshold[th]["f1_event_mean"])
            else -1.0
        ),
    )
    at_best = per_threshold[best_th]

    return {
        "val_loss": total_loss / max(1, n_batches),
        "f1_event": at_default["f1_event_mean"],
        "iou_event": at_default["iou_event_mean"],
        "precision": at_default["precision_event_mean"],
        "recall": at_default["recall_event_mean"],
        "f1_event_best": at_best["f1_event_mean"],
        "best_threshold": float(best_th),
        "n_series": at_default["n_series_total"],
        "n_series_evaluated": at_default["n_series_evaluated"],
        # Métricas por umbral, para poder evaluar con un umbral ya congelado
        "per_threshold": {
            th: {
                "f1": per_threshold[th]["f1_event_mean"],
                "iou": per_threshold[th]["iou_event_mean"],
                "precision": per_threshold[th]["precision_event_mean"],
                "recall": per_threshold[th]["recall_event_mean"],
                "n_series_evaluated": per_threshold[th]["n_series_evaluated"],
            }
            for th in EVAL_THRESHOLDS
        },
    }


def evaluate_checkpoint_on_test(
    checkpoint_path: Path,
    index_path: str,
    device: str,
    threshold: float,
    batch_size: int = 8,
    num_workers: int | None = None,
) -> dict[str, float]:
    """
    Evalúa un checkpoint sobre el conjunto de test (fold -1) con un umbral ya
    congelado. El test no ha participado en la elección del checkpoint ni del
    umbral, por lo que sus métricas son el número publicable.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BurnScarPropagationModel(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = BurnScarDataset(index_path=index_path, folds=[TEST_FOLD])
    if len(dataset) == 0:
        return {}

    nw, pin = resolve_loader_options(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    res = evaluate_fold(model, loader, device=device)
    metrics = res["per_threshold"].get(threshold)
    if metrics is None:
        # El umbral congelado puede no estar en la rejilla: se evalúa aparte
        metrics = _metrics_at_threshold(model, loader, device, threshold)
    return metrics


@torch.no_grad()
def _metrics_at_threshold(
    model: BurnScarPropagationModel,
    loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float,
) -> dict[str, float]:
    """Métricas agregadas por evento con un umbral arbitrario."""
    model.eval()
    series_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)

    for batch in loader:
        logits = model(
            batch["x_olci"].to(device),
            batch["x_dem"].to(device),
            batch["x_worldcover"].to(device),
            batch["x_scar"].to(device),
            batch["horizon_days"].to(device),
        )
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        tgt_np = batch["target"].squeeze(1).cpu().numpy()
        msk_np = batch["mask"].squeeze(1).cpu().numpy()
        for i, sid in enumerate(batch["meta"]["series_id"]):
            m = compute_metrics_numpy(probs[i], tgt_np[i], msk_np[i], threshold=threshold)
            series_metrics[sid].append(m)

    agg = aggregate_series_metrics(series_metrics)
    return {
        "f1": agg["f1_event_mean"],
        "iou": agg["iou_event_mean"],
        "precision": agg["precision_event_mean"],
        "recall": agg["recall_event_mean"],
        "n_series_evaluated": agg["n_series_evaluated"],
    }


def train_fold(
    fold: int,
    args: argparse.Namespace,
    tracker: ExperimentTracker,
    device: str,
) -> dict[str, float]:
    """Entrena y evalúa un fold específico de validación cruzada."""
    train_loader, val_loader = get_cv_dataloaders(
        val_fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        index_path=args.index,
        pin_memory=args.pin_memory,
    )

    model = BurnScarPropagationModel(
        olci_channels=5,
        hidden_ch=args.hidden_ch,
        scar_corruption_prob=args.scar_corruption,
    ).to(device)
    model_config = {
        "olci_channels": 5,
        "hidden_ch": args.hidden_ch,
        "scar_corruption_prob": args.scar_corruption,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -float("inf")
    best_metrics: dict[str, float] = {}
    checkpoint_path: Path | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x_olci = batch["x_olci"].to(device)
            x_dem = batch["x_dem"].to(device)
            x_wc = batch["x_worldcover"].to(device)
            x_scar = batch["x_scar"].to(device)
            h_days = batch["horizon_days"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_olci, x_dem, x_wc, x_scar, h_days)
            loss = masked_bce_dice_loss(logits, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_batches)
        val_res = evaluate_fold(model, val_loader, device=device)

        tracker.log_metrics(
            {
                f"fold_{fold}/train_loss": avg_train_loss,
                f"fold_{fold}/val_loss": val_res["val_loss"],
                f"fold_{fold}/val_f1_event": val_res["f1_event"],
                f"fold_{fold}/val_iou_event": val_res["iou_event"],
                f"fold_{fold}/val_f1_event_best": val_res["f1_event_best"],
                f"fold_{fold}/val_recall": val_res["recall"],
            },
            global_step=epoch,
        )

        # Selección del checkpoint por F1 óptimo; si no hay métricas informativas
        # (target sin quemado) se recurre a la pérdida de validación.
        score = val_res["f1_event_best"]
        if np.isnan(score):
            score = -val_res["val_loss"]
        if score > best_score:
            best_score = score
            best_metrics = dict(val_res)
            checkpoint_path = tracker.save_checkpoint(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "metrics": best_metrics,
                },
                filename=f"best_fold_{fold}.pt",
            )

        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            f1_txt = (
                f"F1={val_res['f1_event'] * 100:.2f}%"
                if not np.isnan(val_res["f1_event"])
                else "F1=n/d"
            )
            best_txt = (
                f" F1*={val_res['f1_event_best'] * 100:.2f}%@umbral{val_res['best_threshold']:.1f}"
                if not np.isnan(val_res["f1_event_best"])
                else ""
            )
            print(
                f"  [Fold {fold} | Época {epoch:02d}/{args.epochs:02d}] "
                f"train_loss={avg_train_loss:.4f} val_loss={val_res['val_loss']:.4f} "
                f"{f1_txt}{best_txt}"
            )

    return best_metrics, checkpoint_path


def _nanmean(values: list[float]) -> float:
    """Media ignorando NaN (métricas no informativas)."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento 5-fold del modelo de propagación")
    parser.add_argument("--index", default="data/processed/dataset_index.json")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Workers del DataLoader (None = automático: 0 en ROCm, 2 en CUDA)",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Memoria fijada del DataLoader (None = automático)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-ch", type=int, default=32)
    parser.add_argument("--scar-corruption", type=float, default=0.15)
    parser.add_argument("--tag", default="propagation_convlstm_segformer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 72)
    print("ENTRENAMIENTO DE MODELO DE PROPAGACIÓN (ConvLSTM + SegFormer)")
    print(f"Dispositivo: {args.device} | Folds: {args.folds} | Épocas: {args.epochs}")
    print("=" * 72)

    t0 = time.time()
    fold_results: list[dict[str, float]] = []
    checkpoint_paths: list[Path] = []

    with ExperimentTracker(args.tag) as tracker:
        print(f"TensorBoard run: {tracker.run_dir}")
        print(f"Checkpoints:     {tracker.checkpoint_dir}\n")

        for fold in args.folds:
            print(f"--- Entrenando Fold {fold} ---")
            res, ckpt_path = train_fold(fold, args, tracker, device=args.device)
            fold_results.append(res)
            if ckpt_path is not None:
                checkpoint_paths.append(ckpt_path)

        mean_f1 = _nanmean([r["f1_event"] for r in fold_results])
        std_f1 = float(np.std([r["f1_event"] for r in fold_results]))
        mean_iou = _nanmean([r["iou_event"] for r in fold_results])
        mean_f1_best = _nanmean([r["f1_event_best"] for r in fold_results])
        mean_recall = _nanmean([r["recall"] for r in fold_results])
        tracker.log_metrics(
            {
                "cv/f1_event_mean": mean_f1,
                "cv/iou_event_mean": mean_iou,
                "cv/f1_event_best_mean": mean_f1_best,
                "cv/recall_mean": mean_recall,
            },
            global_step=1,
        )

        # ── Umbral operativo ────────────────────────────────────────────────
        # Los umbrales óptimos se eligen SOLO con los folds de validación. Se
        # consolidan en uno solo (mediana) y se congelan; el conjunto de test
        # (fold -1) se evalúa después con ese valor, sin haberlo visto nunca.
        opt_thresholds = [
            r["best_threshold"] for r in fold_results if not np.isnan(r["best_threshold"])
        ]
        if opt_thresholds:
            tau = float(np.median(opt_thresholds))
            spread = float(max(opt_thresholds) - min(opt_thresholds))
        else:
            tau, spread = DEFAULT_THRESHOLD, 0.0

        print(f"\nUmbral operativo (mediana de los óptimos por fold): {tau:.2f}")
        print(f"  Rango entre folds: {min(opt_thresholds):.2f}–{max(opt_thresholds):.2f}"
              if opt_thresholds else "  (sin umbrales disponibles)")
        if spread > 0.1:
            print(
                "  ⚠ AVISO: los umbrales óptimos difieren en más de 0,1 entre folds. "
                "El modelo no es estable; el umbral congelado es poco fiable."
            )

        # ── Test con el umbral congelado ────────────────────────────────────
        test_metrics: list[dict[str, float]] = []
        for path in checkpoint_paths:
            m = evaluate_checkpoint_on_test(
                path,
                index_path=args.index,
                device=args.device,
                threshold=tau,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            if m:
                test_metrics.append(m)

        test_summary = {
            "f1": _nanmean([m["f1"] for m in test_metrics]),
            "iou": _nanmean([m["iou"] for m in test_metrics]),
            "precision": _nanmean([m["precision"] for m in test_metrics]),
            "recall": _nanmean([m["recall"] for m in test_metrics]),
            "n_series_evaluated": int(
                np.mean([m["n_series_evaluated"] for m in test_metrics])
            ) if test_metrics else 0,
        }
        tracker.log_metrics(
            {f"test/f1@{tau:.2f}": test_summary["f1"], f"test/iou@{tau:.2f}": test_summary["iou"]},
            global_step=1,
        )

        # Persistir el umbral junto a los checkpoints para la inferencia
        payload = {
            "operational_threshold": tau,
            "per_fold_best_threshold": opt_thresholds,
            "threshold_spread": spread,
            "test_fold": TEST_FOLD,
            "test_metrics": test_summary,
            "note": (
                "El umbral se elige con los folds de validación y se congela aquí. "
                "Para estimar área quemada NO hace falta umbral: suma las probabilidades."
            ),
        }
        threshold_file = tracker.checkpoint_dir / "operational_threshold.json"
        with open(threshold_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nUmbral y métricas de test guardados en: {threshold_file}")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"RESUMEN CV ({len(fold_results)} folds) en {elapsed:.1f} s:")
    print(f"  F1 (Macro-evento): {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print(f"  IoU (Jaccard):     {mean_iou * 100:.2f}%")
    print(f"  F1 (mejor umbral): {mean_f1_best * 100:.2f}%")
    print(f"  Recall:            {mean_recall * 100:.2f}%")
    print("-" * 72)
    print(
        f"EVALUACIÓN EN TEST (fold {TEST_FOLD}, fuera de dominio) "
        f"con el umbral congelado {tau:.2f}:"
    )
    if test_metrics:
        print(f"  F1:        {test_summary['f1'] * 100:.2f}%")
        print(f"  IoU:       {test_summary['iou'] * 100:.2f}%")
        print(f"  Precisión: {test_summary['precision'] * 100:.2f}%")
        print(f"  Recall:    {test_summary['recall'] * 100:.2f}%")
        print(f"  Series evaluadas: {test_summary['n_series_evaluated']}")
    else:
        print("  (sin muestras en el fold de test)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
