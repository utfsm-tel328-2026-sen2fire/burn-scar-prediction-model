#!/usr/bin/env python3
"""
Entrenamiento y evaluación en 5 folds (CV agrupada por evento) del modelo de
propagación ConvLSTM + SegFormer (docs/operacion.md §4, docs/dataset.md §10).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from burn_scar.baselines import compute_metrics_numpy
from burn_scar.dataset import get_cv_dataloaders
from burn_scar.models import BurnScarPropagationModel, masked_bce_dice_loss
from burn_scar.tracking import ExperimentTracker


@torch.no_grad()
def evaluate_fold(
    model: BurnScarPropagationModel,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evalúa un fold promediando métricas por serie/evento (D10)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    series_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)

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
            m = compute_metrics_numpy(probs[i], tgt_np[i], msk_np[i], threshold=threshold)
            series_metrics[sid].append(m)

    f1_series = [np.mean([m["f1"] for m in ms]) for ms in series_metrics.values()]
    iou_series = [np.mean([m["iou"] for m in ms]) for ms in series_metrics.values()]
    prec_series = [np.mean([m["precision"] for m in ms]) for ms in series_metrics.values()]
    rec_series = [np.mean([m["recall"] for m in ms]) for ms in series_metrics.values()]

    return {
        "val_loss": total_loss / max(1, n_batches),
        "f1_event": float(np.mean(f1_series)) if f1_series else 0.0,
        "iou_event": float(np.mean(iou_series)) if iou_series else 0.0,
        "precision": float(np.mean(prec_series)) if prec_series else 0.0,
        "recall": float(np.mean(rec_series)) if rec_series else 0.0,
        "n_series": len(series_metrics),
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
    )

    model = BurnScarPropagationModel(
        olci_channels=5,
        hidden_ch=args.hidden_ch,
        scar_corruption_prob=args.scar_corruption,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

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
            },
            global_step=epoch,
        )

        if val_res["f1_event"] > best_f1:
            best_f1 = val_res["f1_event"]
            best_metrics = dict(val_res)
            tracker.save_checkpoint(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": best_metrics,
                },
                filename=f"best_fold_{fold}.pt",
            )

        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(
                f"  [Fold {fold} | Época {epoch:02d}/{args.epochs:02d}] "
                f"train_loss={avg_train_loss:.4f} val_loss={val_res['val_loss']:.4f} "
                f"F1={val_res['f1_event'] * 100:.2f}% IoU={val_res['iou_event'] * 100:.2f}%"
            )

    return best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento 5-fold del modelo de propagación")
    parser.add_argument("--index", default="data/processed/dataset_index.json")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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

    with ExperimentTracker(args.tag) as tracker:
        print(f"TensorBoard run: {tracker.run_dir}")
        print(f"Checkpoints:     {tracker.checkpoint_dir}\n")

        for fold in args.folds:
            print(f"--- Entrenando Fold {fold} ---")
            res = train_fold(fold, args, tracker, device=args.device)
            fold_results.append(res)

        mean_f1 = float(np.mean([r["f1_event"] for r in fold_results]))
        std_f1 = float(np.std([r["f1_event"] for r in fold_results]))
        mean_iou = float(np.mean([r["iou_event"] for r in fold_results]))
        tracker.log_metrics(
            {"cv/f1_event_mean": mean_f1, "cv/iou_event_mean": mean_iou},
            global_step=1,
        )

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"RESUMEN CV ({len(fold_results)} folds) en {elapsed:.1f} s:")
    print(f"  F1 (Macro-evento): {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print(f"  IoU (Jaccard):     {mean_iou * 100:.2f}%")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
