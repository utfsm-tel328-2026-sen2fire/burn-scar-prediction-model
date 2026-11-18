#!/usr/bin/env python3
"""
Etapa 1 (M-A): Preentrenamiento del modelo de propagación sobre MCD64A1
y evaluación en cero-disparo sobre los 5 folds de CEMS (B10, D16).

1. Entrena `BurnScarPropagationModel` usando `data/processed_mcd/dataset_index.json`
   con partición interna train/val sin fuga espacial (fold 0..3 train, fold 4 val).
2. Guarda el checkpoint preentrenado en `--output-checkpoint` (`checkpoints/mcd_pretrained.pt`)
   y en el directorio del `ExperimentTracker`.
3. Evalúa el modelo **M-A** directamente sobre el índice de CEMS
   (`data/processed/dataset_index.json`) sin haber visto nunca etiquetas CEMS.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from burn_scar.dataset import BurnScarDataset
from burn_scar.models import BurnScarPropagationModel, masked_bce_dice_loss
from burn_scar.tracking import ExperimentTracker
from scripts.train_propagation import evaluate_fold


def _build_pretrain_loaders(
    index_path: str | Path,
    val_fold: int = 4,
    batch_size: int = 8,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    with open(index_path, encoding="utf-8") as f:
        idx_data = json.load(f)
    samples = idx_data.get("samples", [])
    if not samples:
        raise ValueError(f"El índice MCD64A1 no contiene muestras: {index_path}")

    present_folds = sorted({int(s.get("fold", 0)) for s in samples})
    if len(present_folds) > 1 and val_fold in present_folds:
        train_folds = [f for f in present_folds if f != val_fold]
        train_ds = BurnScarDataset(index_path=index_path, folds=train_folds)
        val_ds = BurnScarDataset(index_path=index_path, folds=[val_fold])
    elif len(present_folds) > 1:
        v_f = present_folds[-1]
        train_folds = [f for f in present_folds if f != v_f]
        train_ds = BurnScarDataset(index_path=index_path, folds=train_folds)
        val_ds = BurnScarDataset(index_path=index_path, folds=[v_f])
    else:
        # Si el lote de prueba cae en un solo clúster/fold, usa todas para entrenar y validar
        train_ds = BurnScarDataset(index_path=index_path)
        val_ds = BurnScarDataset(index_path=index_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etapa 1 (M-A): Preentrenamiento con MCD64A1 y evaluación cero-disparo en CEMS"
    )
    parser.add_argument("--mcd-index", default="data/processed_mcd/dataset_index.json")
    parser.add_argument("--cems-index", default="data/processed/dataset_index.json")
    parser.add_argument("--output-checkpoint", default="checkpoints/mcd_pretrained.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-ch", type=int, default=32)
    parser.add_argument("--scar-corruption", type=float, default=0.15)
    parser.add_argument("--tag", default="pretrain_mcd64a1_M_A")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 72)
    print("ETAPA 1 (M-A): PREENTRENAMIENTO CON MCD64A1")
    print(f"Dispositivo: {args.device} | Épocas: {args.epochs} | Índice MCD: {args.mcd_index}")
    print("=" * 72)

    train_loader, val_loader = _build_pretrain_loaders(
        args.mcd_index, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = BurnScarPropagationModel(
        olci_channels=5,
        hidden_ch=args.hidden_ch,
        scar_corruption_prob=args.scar_corruption,
    ).to(args.device)
    model_config = {
        "olci_channels": 5,
        "hidden_ch": args.hidden_ch,
        "scar_corruption_prob": args.scar_corruption,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    t0 = time.time()
    best_score = -float("inf")
    best_ckpt_path: Path | None = None

    with ExperimentTracker(args.tag) as tracker:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_olci = batch["x_olci"].to(args.device)
                x_dem = batch["x_dem"].to(args.device)
                x_wc = batch["x_worldcover"].to(args.device)
                x_scar = batch["x_scar"].to(args.device)
                h_days = batch["horizon_days"].to(args.device)
                target = batch["target"].to(args.device)
                mask = batch["mask"].to(args.device)

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
            val_res = evaluate_fold(model, val_loader, device=args.device)

            tracker.log_metrics(
                {
                    "pretrain/train_loss": avg_train_loss,
                    "pretrain/val_loss": val_res["val_loss"],
                    "pretrain/val_f1_event": val_res["f1_event"],
                    "pretrain/val_f1_event_best": val_res["f1_event_best"],
                },
                global_step=epoch,
            )

            score = val_res["f1_event_best"]
            if np.isnan(score):
                score = -val_res["val_loss"]
            if score >= best_score or best_ckpt_path is None:
                best_score = score
                best_ckpt_path = tracker.save_checkpoint(
                    {
                        "stage": "M-A",
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config,
                        "mcd_val_metrics": val_res,
                    },
                    filename="mcd_pretrained.pt",
                )

            print(
                f"  [M-A Época {epoch:02d}/{args.epochs:02d}] "
                f"train_loss={avg_train_loss:.4f} val_loss={val_res['val_loss']:.4f}"
            )

        assert best_ckpt_path is not None
        out_ckpt = Path(args.output_checkpoint)
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt_path, out_ckpt)

        # Evaluación cero-disparo de M-A sobre CEMS (D16, B10)
        cems_zero_shot: dict[str, float] = {}
        if Path(args.cems_index).exists():
            ckpt_data = torch.load(out_ckpt, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            cems_ds = BurnScarDataset(index_path=args.cems_index, folds=[0, 1, 2, 3, 4])
            cems_loader = DataLoader(
                cems_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            cems_zero_shot = evaluate_fold(model, cems_loader, device=args.device)
            tracker.log_metrics(
                {
                    "m_a_cems_zeroshot/f1_event": cems_zero_shot["f1_event"],
                    "m_a_cems_zeroshot/f1_event_best": cems_zero_shot["f1_event_best"],
                    "m_a_cems_zeroshot/iou_event": cems_zero_shot["iou_event"],
                },
                global_step=1,
            )

    print("\n" + "=" * 72)
    print(f"PREENTRENAMIENTO M-A COMPLETADO en {time.time() - t0:.1f} s")
    print(f"Checkpoint base guardado en: {args.output_checkpoint}")
    if cems_zero_shot:
        print(
            f"Evaluación M-A cero-disparo sobre CEMS (folds 0..4): "
            f"F1={cems_zero_shot['f1_event'] * 100:.2f}% | "
            f"F1*={cems_zero_shot['f1_event_best'] * 100:.2f}% | "
            f"IoU={cems_zero_shot['iou_event'] * 100:.2f}%"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
