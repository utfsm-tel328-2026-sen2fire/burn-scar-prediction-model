#!/usr/bin/env python3
"""
Entrenamiento y evaluación en 5 folds (CV agrupada por evento) del modelo de
propagación ConvLSTM + SegFormer (docs/operacion.md §4, docs/dataset.md §10).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from burn_scar.baselines import (
    aggregate_series_metrics,
    compact_series_metrics,
    compute_metrics_numpy,
)
from burn_scar.dataset import BurnScarDataset, get_cv_dataloaders, resolve_loader_options
from burn_scar.models import BurnScarPropagationModel, masked_bce_dice_loss
from burn_scar.quality import (
    describe_evaluated_set,
    file_sha256,
    historical_status,
    horizon_band,
    state_dict_sha256,
)
from burn_scar.tracking import ExperimentTracker

# Umbrales evaluados en validación: un modelo probabilístico desbalanceado
# raramente opera en 0,5, y los baselines B0/B1 no tienen umbral (son binarios).
EVAL_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_THRESHOLD = 0.5

# Fold de test: series fuera de dominio (no mediterráneas), nunca visto en
# validación ni en la elección del umbral.
TEST_FOLD = -1
DEFAULT_SEED = 42


def set_reproducibility_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad usadas por entrenamiento y carga."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_provenance(
    index_path: str | Path,
    index_payload: dict,
    seed: int,
    tag: str,
) -> dict:
    """Procedencia común de dataset, checkpoint y métricas."""
    inventory = index_payload.get("inventory", {})
    return {
        "run_tag": tag,
        "seed": seed,
        "dataset_index": str(index_path),
        "dataset_index_sha256": file_sha256(index_path),
        "dataset_content_sha256": index_payload.get("dataset_content_sha256"),
        "dataset_generator_version": index_payload.get("generator_version"),
        "dataset_schema_version": index_payload.get("dataset_index_schema_version"),
        "dataset_target_mode": index_payload.get("target_mode"),
        "dataset_counts": inventory,
        "horizon_support": inventory.get("horizon_distribution", {}),
        "cold_start_coverage": inventory.get("scar_coverage", {}),
    }


@torch.no_grad()
def evaluate_fold(
    model: BurnScarPropagationModel,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """
    Evalúa un fold promediando métricas por serie/evento (D10).

    Calcula las métricas contra dos etiquetas: el target de ensamblaje del
    dataset y el target de referencia (crecimiento real entre rondas CEMS).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # Métricas por serie y por umbral (ignora NaN: targets sin quemado)
    series_metrics: dict[str, dict[float, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    series_metrics_ref: dict[str, dict[float, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    horizon_metrics: dict[str, dict[float, dict[str, dict[str, list]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    cold_metrics: dict[float, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    cold_sample_count = 0
    cold_series: set[str] = set()

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
        ref_tgt_np = batch["target_cems"].squeeze(1).cpu().numpy()
        ref_msk_np = batch["mask_cems"].squeeze(1).cpu().numpy()
        has_ref = batch["has_reference"].cpu().numpy().reshape(-1)
        series_ids = batch["meta"]["series_id"]
        horizon_values = h_days.view(-1).cpu().numpy()
        scar_values = x_scar.detach().cpu().numpy()
        scar_states = batch["meta"].get("scar_state")

        for i, sid in enumerate(series_ids):
            band = horizon_band(float(horizon_values[i]))
            if scar_states is not None:
                is_cold_start = str(scar_states[i]) == "cold_start"
            else:
                is_cold_start = bool(np.all(scar_values[i] == 0))
            if is_cold_start:
                cold_sample_count += 1
                cold_series.add(str(sid))

            for th in EVAL_THRESHOLDS:
                m = compute_metrics_numpy(probs[i], tgt_np[i], msk_np[i], threshold=th)
                series_metrics[sid][th].append(m)
                horizon_metrics[band][th]["assembly"][sid].append(m)
                if is_cold_start:
                    cold_metrics[th]["assembly"][sid].append(m)
                if has_ref[i] > 0:
                    mr = compute_metrics_numpy(
                        probs[i], ref_tgt_np[i], ref_msk_np[i], threshold=th
                    )
                    series_metrics_ref[sid][th].append(mr)
                    horizon_metrics[band][th]["reference"][sid].append(mr)
                    if is_cold_start:
                        cold_metrics[th]["reference"][sid].append(mr)

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

    # Target de referencia (crecimiento real entre rondas CEMS)
    ref_available = bool(series_metrics_ref)
    if ref_available:
        per_threshold_ref = {
            th: aggregate_series_metrics({sid: ms[th] for sid, ms in series_metrics_ref.items()})
            for th in EVAL_THRESHOLDS
        }
        ref_at_default = per_threshold_ref[threshold]
        ref_best_th = max(
            EVAL_THRESHOLDS,
            key=lambda th: (
                per_threshold_ref[th]["f1_event_mean"]
                if not np.isnan(per_threshold_ref[th]["f1_event_mean"])
                else -1.0
            ),
        )
        ref_at_best = per_threshold_ref[ref_best_th]
    else:
        per_threshold_ref = {}
        ref_at_default = {}
        ref_at_best = {}
        ref_best_th = float("nan")

    horizon_per_threshold: dict[str, dict[float, dict[str, object]]] = {}
    for band, threshold_groups in horizon_metrics.items():
        horizon_per_threshold[band] = {}
        for th, target_groups in threshold_groups.items():
            assembly = compact_series_metrics(
                aggregate_series_metrics(target_groups.get("assembly", {}))
            )
            reference = (
                compact_series_metrics(
                    aggregate_series_metrics(target_groups.get("reference", {}))
                )
                if target_groups.get("reference")
                else None
            )
            horizon_per_threshold[band][th] = {
                "assembly": assembly,
                "reference": reference,
            }

    cold_assembly = (
        compact_series_metrics(aggregate_series_metrics(cold_metrics[threshold]["assembly"]))
        if cold_metrics[threshold]["assembly"]
        else None
    )
    cold_reference = (
        compact_series_metrics(aggregate_series_metrics(cold_metrics[threshold]["reference"]))
        if cold_metrics[threshold]["reference"]
        else None
    )
    cold_start = {
        "represented": bool(cold_sample_count),
        "n_samples_total": cold_sample_count,
        "n_series_total": len(cold_series),
        "assembly": cold_assembly,
        "reference": cold_reference,
        "warning": (
            None
            if cold_sample_count
            else "El arranque en frío no está representado en el dataset evaluado."
        ),
    }

    dataset = getattr(val_loader, "dataset", None)
    if dataset is None:
        evaluated_set = {}
    else:
        evaluated_set = describe_evaluated_set(dataset)

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
        "n_samples": at_default["n_samples_total"],
        "n_samples_evaluated": at_default["n_samples_evaluated"],
        # Métricas por umbral, para poder evaluar con un umbral ya congelado
        "per_threshold": {
            th: {
                "f1": per_threshold[th]["f1_event_mean"],
                "iou": per_threshold[th]["iou_event_mean"],
                "precision": per_threshold[th]["precision_event_mean"],
                "recall": per_threshold[th]["recall_event_mean"],
                "n_series_evaluated": per_threshold[th]["n_series_evaluated"],
                "n_samples_evaluated": per_threshold[th]["n_samples_evaluated"],
                "n_samples_total": per_threshold[th]["n_samples_total"],
            }
            for th in EVAL_THRESHOLDS
        },
        # Target de referencia: el desempeño contra el crecimiento real CEMS
        "ref_available": ref_available,
        "ref_f1_event": ref_at_default.get("f1_event_mean", float("nan")),
        "ref_iou_event": ref_at_default.get("iou_event_mean", float("nan")),
        "ref_precision": ref_at_default.get("precision_event_mean", float("nan")),
        "ref_recall": ref_at_default.get("recall_event_mean", float("nan")),
        "ref_n_series_evaluated": ref_at_default.get("n_series_evaluated", 0),
        "ref_n_samples_evaluated": ref_at_default.get("n_samples_evaluated", 0),
        "ref_n_samples_total": ref_at_default.get("n_samples_total", 0),
        "ref_f1_event_best": ref_at_best.get("f1_event_mean", float("nan")),
        "ref_best_threshold": float(ref_best_th),
        "ref_per_threshold": {
            th: {
                "f1": per_threshold_ref[th]["f1_event_mean"],
                "iou": per_threshold_ref[th]["iou_event_mean"],
                "precision": per_threshold_ref[th]["precision_event_mean"],
                "recall": per_threshold_ref[th]["recall_event_mean"],
                "n_series_evaluated": per_threshold_ref[th]["n_series_evaluated"],
                "n_samples_evaluated": per_threshold_ref[th]["n_samples_evaluated"],
                "n_samples_total": per_threshold_ref[th]["n_samples_total"],
            }
            for th in EVAL_THRESHOLDS
        } if ref_available else {},
        "evaluated_set": evaluated_set,
        "horizon_per_threshold": horizon_per_threshold,
        "horizon_strata": {
            band: values.get(threshold)
            for band, values in horizon_per_threshold.items()
        },
        "cold_start": cold_start,
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
    ref_metrics = res["ref_per_threshold"].get(threshold) if res.get("ref_available") else None
    if metrics is None or (res.get("ref_available") and ref_metrics is None):
        # El umbral congelado puede no estar en la rejilla: se evalúa aparte
        return _metrics_at_threshold(model, loader, device, threshold)
    return {
        "assembly": {
            "f1": metrics["f1"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "n_series_evaluated": metrics["n_series_evaluated"],
            "n_samples_evaluated": metrics["n_samples_evaluated"],
            "n_samples_total": metrics["n_samples_total"],
        },
        "reference": None if ref_metrics is None else {
            "f1": ref_metrics["f1"],
            "iou": ref_metrics["iou"],
            "precision": ref_metrics["precision"],
            "recall": ref_metrics["recall"],
            "n_series_evaluated": ref_metrics["n_series_evaluated"],
            "n_samples_evaluated": ref_metrics["n_samples_evaluated"],
            "n_samples_total": ref_metrics["n_samples_total"],
        },
        "evaluated_set": res["evaluated_set"],
        "horizon_strata": {
            band: values[threshold]
            for band, values in res["horizon_per_threshold"].items()
            if threshold in values
        },
        "cold_start": res["cold_start"],
    }


@torch.no_grad()
def _metrics_at_threshold(
    model: BurnScarPropagationModel,
    loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float,
) -> dict[str, dict[str, float] | None]:
    """Métricas agregadas por evento con un umbral arbitrario, contra ambas etiquetas."""
    model.eval()
    series_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    series_metrics_ref: dict[str, list[dict[str, float]]] = defaultdict(list)

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
        ref_tgt_np = batch["target_cems"].squeeze(1).cpu().numpy()
        ref_msk_np = batch["mask_cems"].squeeze(1).cpu().numpy()
        has_ref = batch["has_reference"].cpu().numpy().reshape(-1)
        for i, sid in enumerate(batch["meta"]["series_id"]):
            series_metrics[sid].append(
                compute_metrics_numpy(probs[i], tgt_np[i], msk_np[i], threshold=threshold)
            )
            if has_ref[i] > 0:
                series_metrics_ref[sid].append(
                    compute_metrics_numpy(
                        probs[i], ref_tgt_np[i], ref_msk_np[i], threshold=threshold
                    )
                )

    agg = aggregate_series_metrics(series_metrics)
    out: dict[str, dict[str, float] | None] = {
        "assembly": {
            "f1": agg["f1_event_mean"],
            "iou": agg["iou_event_mean"],
            "precision": agg["precision_event_mean"],
            "recall": agg["recall_event_mean"],
            "n_series_evaluated": agg["n_series_evaluated"],
            "n_samples_evaluated": agg["n_samples_evaluated"],
            "n_samples_total": agg["n_samples_total"],
        },
        "reference": None,
    }
    if series_metrics_ref:
        agg_ref = aggregate_series_metrics(series_metrics_ref)
        out["reference"] = {
            "f1": agg_ref["f1_event_mean"],
            "iou": agg_ref["iou_event_mean"],
            "precision": agg_ref["precision_event_mean"],
            "recall": agg_ref["recall_event_mean"],
            "n_series_evaluated": agg_ref["n_series_evaluated"],
            "n_samples_evaluated": agg_ref["n_samples_evaluated"],
            "n_samples_total": agg_ref["n_samples_total"],
        }
    return out


def train_fold(
    fold: int,
    args: argparse.Namespace,
    tracker: ExperimentTracker,
    device: str,
) -> dict[str, float]:
    """Entrena y evalúa un fold específico de validación cruzada."""
    set_reproducibility_seed(args.seed + fold)
    index_payload = json.loads(Path(args.index).read_text(encoding="utf-8"))

    train_loader, val_loader = get_cv_dataloaders(
        val_fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        index_path=args.index,
        pin_memory=args.pin_memory,
        seed=args.seed + fold,
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
                f"fold_{fold}/val_ref_f1_event": val_res["ref_f1_event"],
                f"fold_{fold}/val_ref_iou_event": val_res["ref_iou_event"],
                f"fold_{fold}/val_ref_recall": val_res["ref_recall"],
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
            best_metrics["n_train_samples"] = len(train_loader.dataset)
            best_metrics["n_val_samples"] = len(val_loader.dataset)
            checkpoint_path = tracker.save_checkpoint(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "seed": args.seed + fold,
                    "dataset_index_sha256": file_sha256(args.index),
                    "dataset_content_sha256": index_payload.get(
                        "dataset_content_sha256"
                    ),
                    "weights_sha256": state_dict_sha256(model.state_dict()),
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
            ref_txt = ""
            if val_res["ref_available"] and not np.isnan(val_res["ref_f1_event"]):
                ref_txt = f" | ref F1={val_res['ref_f1_event'] * 100:.2f}%"
            print(
                f"  [Fold {fold} | Época {epoch:02d}/{args.epochs:02d}] "
                f"train_loss={avg_train_loss:.4f} val_loss={val_res['val_loss']:.4f} "
                f"{f1_txt}{best_txt}{ref_txt}"
            )

    return best_metrics, checkpoint_path


def _nanmean(values: list[float]) -> float:
    """Media ignorando NaN (métricas no informativas)."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def evaluate_only(args: argparse.Namespace) -> int:
    """
    Reevalúa los checkpoints de un run existente sobre el conjunto de test, con
    el umbral operativo ya congelado, y reporta las métricas contra las dos
    etiquetas (target de ensamblaje y target de referencia).
    """
    ckpt_dir = Path(args.eval_only)
    thr_file = ckpt_dir / "operational_threshold.json"
    tau = DEFAULT_THRESHOLD
    if thr_file.exists():
        with open(thr_file, "r", encoding="utf-8") as f:
            tau = float(json.load(f).get("operational_threshold", DEFAULT_THRESHOLD))

    paths = sorted(ckpt_dir.glob("best_fold_*.pt"))
    if not paths:
        print(f"No hay checkpoints en {ckpt_dir}")
        return 1

    print("=" * 72)
    print("REEVALUACIÓN DE CHECKPOINTS (sin reentrenar)")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Umbral operativo: {tau:.2f}")
    print("=" * 72)

    status = historical_status(ckpt_dir)
    if status and status.get("incompatible_with_canonical_index"):
        print("⚠ AVISO: este run es histórico y NO compatible con el índice canónico.")
        print(f"  Run canónico: {status.get('canonical_run')}.")
        for reason in status.get("reasons", []):
            print(f"  - {reason}")
        print("  Sus métricas no se publican como resultado actual.")
        print("=" * 72)

    test_metrics: list[dict[str, dict[str, float] | None]] = []
    for path in paths:
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

    def _agg(key: str, field: str) -> float:
        vals = [m[key][field] for m in test_metrics if m.get(key) is not None]
        return _nanmean(vals) if vals else float("nan")

    has_ref = any(m.get("reference") is not None for m in test_metrics)
    print("\nTEST (fold -1) con el umbral congelado "
          f"{tau:.2f} | {len(test_metrics)} checkpoints")
    print("  Contra el target de ensamblaje:")
    print(f"    F1:        {_pct(_agg('assembly', 'f1'))}")
    print(f"    IoU:       {_pct(_agg('assembly', 'iou'))}")
    print(f"    Precisión: {_pct(_agg('assembly', 'precision'))}")
    print(f"    Recall:    {_pct(_agg('assembly', 'recall'))}")
    print(
        f"    Muestras evaluables: {_agg('assembly', 'n_samples_evaluated'):.0f}"
        f"/{_agg('assembly', 'n_samples_total'):.0f}"
    )
    if has_ref:
        print("  Contra el target de referencia (crecimiento real entre rondas CEMS):")
        print(f"    F1:        {_pct(_agg('reference', 'f1'))}")
        print(f"    IoU:       {_pct(_agg('reference', 'iou'))}")
        print(f"    Precisión: {_pct(_agg('reference', 'precision'))}")
        print(f"    Recall:    {_pct(_agg('reference', 'recall'))}")
        print(
            f"    Muestras evaluables: {_agg('reference', 'n_samples_evaluated'):.0f}"
            f"/{_agg('reference', 'n_samples_total'):.0f}"
        )
    else:
        print("  Contra el target de referencia: no disponible")
    print("=" * 72)
    return 0


def _aggregate_horizon_strata(
    test_metrics: list[dict],
    label: str,
) -> dict[str, dict[str, float | int | None]]:
    """Promedia los estratos de horizonte de los checkpoints evaluados."""
    bands = sorted({
        band
        for metrics in test_metrics
        for band in metrics.get("horizon_strata", {})
    })
    output: dict[str, dict[str, float | int | None]] = {}
    for band in bands:
        summaries = [
            metrics.get("horizon_strata", {}).get(band, {}).get(label)
            for metrics in test_metrics
        ]
        summaries = [summary for summary in summaries if summary is not None]
        if not summaries:
            output[band] = None
            continue
        output[band] = {
            "f1": float(np.mean([summary["f1"] for summary in summaries])),
            "iou": float(np.mean([summary["iou"] for summary in summaries])),
            "precision": float(np.mean([summary["precision"] for summary in summaries])),
            "recall": float(np.mean([summary["recall"] for summary in summaries])),
            "n_series_evaluated": int(
                np.mean([summary["n_series_evaluated"] for summary in summaries])
            ),
            "n_samples_evaluated": int(
                np.mean([summary["n_samples_evaluated"] for summary in summaries])
            ),
            "n_samples_total": int(
                np.mean([summary["n_samples_total"] for summary in summaries])
            ),
        }
    return output


def _aggregate_cold_start(test_metrics: list[dict]) -> dict[str, object]:
    """Resume el estrato de arranque en frío sin inventar métricas."""
    if not test_metrics:
        return {"represented": False, "n_samples_total": 0, "n_series_total": 0}
    represented = all(metrics.get("cold_start", {}).get("represented", False) for metrics in test_metrics)
    return {
        "represented": represented,
        "n_samples_total": int(
            np.mean([metrics.get("cold_start", {}).get("n_samples_total", 0) for metrics in test_metrics])
        ),
        "n_series_total": int(
            np.mean([metrics.get("cold_start", {}).get("n_series_total", 0) for metrics in test_metrics])
        ),
        "warning": (
            None
            if represented
            else "El arranque en frío no está representado en el conjunto de test."
        ),
    }


def _collect_checkpoint_provenance(
    checkpoint_paths: list[Path],
    folds: list[int],
) -> list[dict[str, object]]:
    """
    Procedencia de cada checkpoint del run.

    Registra la huella del fichero, la huella de los pesos, la época, la
    semilla y el hash del índice con el que se entrenó, de modo que el
    manifiesto permita comprobar que los artefactos publicados corresponden al
    dataset y a la ejecución canónicos.
    """
    provenance: list[dict[str, object]] = []
    for path, fold in zip(checkpoint_paths, folds):
        entry: dict[str, object] = {
            "fold": int(fold),
            "path": str(path),
            "file_sha256": file_sha256(path),
        }
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # pragma: no cover - solo informative
            provenance.append(entry)
            continue
        entry.update({
            "epoch": checkpoint.get("epoch"),
            "seed": checkpoint.get("seed"),
            "weights_sha256": checkpoint.get("weights_sha256"),
            "dataset_index_sha256": checkpoint.get("dataset_index_sha256"),
        })
        provenance.append(entry)
    return provenance


def _pct(value: float) -> str:
    """Porcentaje legible, o 'n/d' si la métrica no es informativa."""
    return "n/d" if np.isnan(value) else f"{value * 100:.2f}%"


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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--eval-only",
        metavar="CHECKPOINT_DIR",
        default=None,
        help=(
            "No entrena: reevalúa los checkpoints de un run existente sobre el "
            "conjunto de test, con el umbral operativo ya congelado"
        ),
    )
    args = parser.parse_args()
    index_payload = json.loads(Path(args.index).read_text(encoding="utf-8"))
    set_reproducibility_seed(args.seed)

    if args.eval_only:
        return evaluate_only(args)

    print("=" * 72)
    print("ENTRENAMIENTO DE MODELO DE PROPAGACIÓN (ConvLSTM + SegFormer)")
    print(f"Dispositivo: {args.device} | Folds: {args.folds} | Épocas: {args.epochs}")
    print("=" * 72)

    t0 = time.time()
    fold_results: list[dict[str, float]] = []
    checkpoint_paths: list[Path] = []
    checkpoint_folds: list[int] = []

    with ExperimentTracker(args.tag) as tracker:
        print(f"TensorBoard run: {tracker.run_dir}")
        print(f"Checkpoints:     {tracker.checkpoint_dir}\n")

        for fold in args.folds:
            print(f"--- Entrenando Fold {fold} ---")
            res, ckpt_path = train_fold(fold, args, tracker, device=args.device)
            fold_results.append(res)
            if ckpt_path is not None:
                checkpoint_paths.append(ckpt_path)
                checkpoint_folds.append(fold)

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
        test_metrics: list[dict[str, dict[str, float] | None]] = []
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

        def _agg(key: str, field: str) -> float:
            vals = [m[key][field] for m in test_metrics if m.get(key) is not None]
            return _nanmean(vals) if vals else float("nan")

        test_summary = {
            "f1": _agg("assembly", "f1"),
            "iou": _agg("assembly", "iou"),
            "precision": _agg("assembly", "precision"),
            "recall": _agg("assembly", "recall"),
            "n_series_evaluated": int(
                np.mean([m["assembly"]["n_series_evaluated"] for m in test_metrics])
            ) if test_metrics else 0,
            "n_samples_evaluated": int(
                np.mean([m["assembly"]["n_samples_evaluated"] for m in test_metrics])
            ) if test_metrics else 0,
            "n_samples_total": int(
                np.mean([m["assembly"]["n_samples_total"] for m in test_metrics])
            ) if test_metrics else 0,
            "evaluated_set": test_metrics[0]["evaluated_set"] if test_metrics else {},
            "horizon_strata": _aggregate_horizon_strata(test_metrics, "assembly"),
            "cold_start": _aggregate_cold_start(test_metrics),
        }
        has_ref = any(m.get("reference") is not None for m in test_metrics)
        ref_summary = (
            {
                "f1": _agg("reference", "f1"),
                "iou": _agg("reference", "iou"),
                "precision": _agg("reference", "precision"),
                "recall": _agg("reference", "recall"),
                "n_series_evaluated": int(
                    np.mean([
                        m["reference"]["n_series_evaluated"]
                        for m in test_metrics
                        if m.get("reference") is not None
                    ])
                ),
                "n_samples_evaluated": int(
                    np.mean([
                        m["reference"]["n_samples_evaluated"]
                        for m in test_metrics
                        if m.get("reference") is not None
                    ])
                ),
                "n_samples_total": int(
                    np.mean([
                        m["reference"]["n_samples_total"]
                        for m in test_metrics
                        if m.get("reference") is not None
                    ])
                ),
                "horizon_strata": _aggregate_horizon_strata(test_metrics, "reference"),
            }
            if has_ref else None
        )
        tracker.log_metrics(
            {f"test/f1@{tau:.2f}": test_summary["f1"], f"test/iou@{tau:.2f}": test_summary["iou"]},
            global_step=1,
        )
        if ref_summary is not None:
            tracker.log_metrics(
                {
                    f"test/ref_f1@{tau:.2f}": ref_summary["f1"],
                    f"test/ref_iou@{tau:.2f}": ref_summary["iou"],
                },
                global_step=1,
            )

        # Persistir el umbral junto a los checkpoints para la inferencia
        payload = {
            "operational_threshold": tau,
            "per_fold_best_threshold": opt_thresholds,
            "threshold_spread": spread,
            **build_dataset_provenance(args.index, index_payload, args.seed, args.tag),
            "checkpoints": _collect_checkpoint_provenance(checkpoint_paths, checkpoint_folds),
            "test_fold": TEST_FOLD,
            "test_metrics": test_summary,
            "test_metrics_reference_target": ref_summary,
            "note": (
                "El umbral se elige con los folds de validación y se congela aquí. "
                "Para estimar área quemada NO hace falta umbral: suma las probabilidades. "
                "Las métricas 'reference' son contra el crecimiento real entre rondas CEMS: "
                "son el desempeño en propagación; las de ensamblaje son contra el target "
                "que incluye quemado no detectado por TMB-S3."
            ),
        }
        threshold_file = tracker.checkpoint_dir / "operational_threshold.json"
        with open(threshold_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nUmbral y métricas de test guardados en: {threshold_file}")

    elapsed = time.time() - t0
    mean_samples_eval = _nanmean(
        [r["n_samples_evaluated"] for r in fold_results if "n_samples_evaluated" in r]
    )
    mean_samples_total = _nanmean(
        [r["n_samples"] for r in fold_results if "n_samples" in r]
    )
    print("\n" + "=" * 72)
    print(f"RESUMEN CV ({len(fold_results)} folds) en {elapsed:.1f} s:")
    print(f"  F1 (Macro-evento): {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print(f"  IoU (Jaccard):     {mean_iou * 100:.2f}%")
    print(f"  F1 (mejor umbral): {mean_f1_best * 100:.2f}%")
    print(f"  Recall:            {mean_recall * 100:.2f}%")
    if not np.isnan(mean_samples_eval):
        print(
            f"  Muestras evaluables (media por fold): {mean_samples_eval:.0f}"
            f"/{mean_samples_total:.0f}"
        )
    print("-" * 72)
    print(
        f"EVALUACIÓN EN TEST (fold {TEST_FOLD}, fuera de dominio) "
        f"con el umbral congelado {tau:.2f}:"
    )
    if test_metrics:
        print("  Contra el target de ensamblaje:")
        print(f"    F1:        {test_summary['f1'] * 100:.2f}%")
        print(f"    IoU:       {test_summary['iou'] * 100:.2f}%")
        print(f"    Precisión: {test_summary['precision'] * 100:.2f}%")
        print(f"    Recall:    {test_summary['recall'] * 100:.2f}%")
        print(f"    Series evaluadas: {test_summary['n_series_evaluated']}")
        print(
            f"    Muestras evaluables: {test_summary['n_samples_evaluated']}"
            f"/{test_summary['n_samples_total']}"
        )
        print("    Por banda de horizonte (ensamblaje):")
        for band, strata in test_summary["horizon_strata"].items():
            if strata is None:
                print(f"      {band}: n/d")
            else:
                print(
                    f"      {band}: F1 {_pct(strata['f1'])}, "
                    f"{strata['n_samples_evaluated']}/{strata['n_samples_total']} evaluables"
                )
        if not test_summary["cold_start"]["represented"]:
            print("    Aviso: el arranque en frío no está representado en el test.")
        if ref_summary is not None:
            print("  Contra el target de referencia (crecimiento real entre rondas CEMS):")
            print(f"    F1:        {ref_summary['f1'] * 100:.2f}%")
            print(f"    IoU:       {ref_summary['iou'] * 100:.2f}%")
            print(f"    Precisión: {ref_summary['precision'] * 100:.2f}%")
            print(f"    Recall:    {ref_summary['recall'] * 100:.2f}%")
            print(f"    Series evaluadas: {ref_summary['n_series_evaluated']}")
            print(
                f"    Muestras evaluables: {ref_summary['n_samples_evaluated']}"
                f"/{ref_summary['n_samples_total']}"
            )
        else:
            print("  Contra el target de referencia: no disponible")
    else:
        print("  (sin muestras en el fold de test)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
