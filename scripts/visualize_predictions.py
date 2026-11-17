#!/usr/bin/env python3
"""
Visualización post-entrenamiento de checkpoints del modelo de propagación
(ConvLSTM + SegFormer) en la pestaña IMAGES de TensorBoard y exportación PNG opcional.
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

import torch

from burn_scar.dataset import get_cv_dataloaders
from burn_scar.tracking import ExperimentTracker, StoragePaths
from burn_scar.visualization import (
    VALID_SELECTION_MODES,
    build_comparison_panel,
    evaluate_validation_samples,
    export_panel_png,
    informative_metric_mean,
    load_propagation_checkpoint,
    select_samples,
    split_informative_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa un checkpoint best_fold_*.pt sobre su split de validación y "
            "registra paneles comparativos en TensorBoard (pestaña IMAGES)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Ruta al archivo de checkpoint (.pt), p. ej. checkpoints/<run>/best_fold_0.pt",
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
        help="Fold de validación [0..4] (por defecto se lee automáticamente del checkpoint)",
    )
    parser.add_argument(
        "--hidden-ch",
        type=int,
        default=None,
        help="Canales ocultos del ConvLSTM (por defecto se infiere de model_state_dict)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_SELECTION_MODES,
        default="both",
        help="Modo de selección de muestras: best, worst, both, series, indices, random",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=6,
        help="Cantidad de muestras a seleccionar por grupo (best/worst/random/series)",
    )
    parser.add_argument(
        "--series-id",
        default=None,
        help="Identificador de serie/evento específico cuando --mode=series",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Lista de índices del split de validación cuando --mode=indices",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de probabilidad para binarizar TP/FP/FN y métricas (default: 0.5)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Factor entero de escalado nearest-neighbor para mejor legibilidad (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamaño de batch durante la inferencia de validación",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers del DataLoader de validación",
    )
    parser.add_argument(
        "--tag",
        default="vis_propagation",
        help="Tag del experimento en TensorBoard (crea <user>_<tag>_<timestamp>)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directorio opcional donde exportar también los paneles en archivos .png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo aleatorio (--mode=random)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo de inferencia (cuda o cpu)",
    )
    parser.add_argument(
        "--dedup-index",
        action="store_true",
        help="Deduplica temporalmente sample_id repetidos en el índice JSON si existen",
    )
    return parser


def run_visualization(
    argv: Sequence[str] | None = None,
    *,
    storage: StoragePaths | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    model, resolved_fold, ckpt = load_propagation_checkpoint(
        args.checkpoint,
        fold=args.fold,
        hidden_ch=args.hidden_ch,
        device=args.device,
    )

    resolved_index = args.index
    tmp_index_path: Path | None = None
    if args.dedup_index:
        raw_payload = json.loads(Path(args.index).read_text(encoding="utf-8"))
        seen_ids: set[str] = set()
        deduped = []
        for s in raw_payload.get("samples", []):
            sid = str(s.get("sample_id"))
            if sid not in seen_ids:
                seen_ids.add(sid)
                deduped.append(s)
        if len(deduped) < len(raw_payload.get("samples", [])):
            raw_payload["samples"] = deduped
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="dedup_index_", delete=False, encoding="utf-8"
            )
            with tmp:
                json.dump(raw_payload, tmp)
            tmp_index_path = Path(tmp.name)
            resolved_index = tmp.name

    try:
        _, val_loader = get_cv_dataloaders(
            val_fold=resolved_fold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            index_path=resolved_index,
        )
    finally:
        if tmp_index_path is not None and tmp_index_path.exists():
            tmp_index_path.unlink()

    evaluated = evaluate_validation_samples(
        model,
        val_loader,
        device=args.device,
        threshold=args.threshold,
    )
    if not evaluated:
        raise RuntimeError(f"El split de validación para fold={resolved_fold} está vacío.")

    selected = select_samples(
        evaluated,
        mode=args.mode,
        n_samples=args.n_samples,
        series_id=args.series_id,
        indices=args.indices,
        seed=args.seed,
    )

    informative, n_no_informativas = split_informative_samples(evaluated)
    mean_f1 = informative_metric_mean(evaluated, "f1")
    mean_iou = informative_metric_mean(evaluated, "iou")
    ckpt_epoch = ckpt.get("epoch", "?")

    print("=" * 72)
    print("VISUALIZACIÓN DE PREDICCIONES (ConvLSTM + SegFormer)")
    print(f"Checkpoint: {args.checkpoint} (Fold {resolved_fold}, Época {ckpt_epoch})")
    print(
        f"Muestras eval: {len(evaluated)} | Seleccionadas: {len(selected)} (modo={args.mode}) | "
        f"Umbral: {args.threshold:.2f}"
    )
    print(
        f"Muestras informativas: {len(informative)}/{len(evaluated)} "
        f"(no informativas excluidas: {n_no_informativas})"
    )
    print(
        "Orden de bloques en cada panel: "
        "[OLCI pseudo-RGB | x_scar | Target | Probabilidad | Overlay TP/FP/FN]"
    )
    print("=" * 72)

    group_Step: dict[str, int] = defaultdict(int)
    exported_paths: list[Path] = []

    with ExperimentTracker(args.tag, storage=storage) as tracker:
        print(f"TensorBoard run: {tracker.run_dir}")
        tracker.log_metrics(
            {
                f"fold_{resolved_fold}/val_f1_sample_mean": mean_f1,
                f"fold_{resolved_fold}/val_iou_sample_mean": mean_iou,
            },
            global_step=0,
        )

        for group, sample in selected:
            group_Step[group] += 1
            rank = group_Step[group]

            panel = build_comparison_panel(
                sample.x_olci,
                sample.x_scar,
                sample.target,
                sample.probs,
                sample.mask,
                threshold=args.threshold,
                scale=args.scale,
            )

            tb_tag = (
                f"fold_{resolved_fold}/{group}/"
                f"{rank:02d}_{sample.series_id}_idx{sample.dataset_idx}_h{int(round(sample.horizon_days))}d"
            )
            tracker.log_image(tb_tag, panel, global_step=rank)

            print(
                f"  [{group.upper():6s} #{rank:02d}] idx={sample.dataset_idx:03d} "
                f"serie={sample.series_id} h={sample.horizon_days:.0f}d "
                f"F1={sample.f1 * 100:.1f}% IoU={sample.iou * 100:.1f}% "
                f"(quema_real={sample.target_burned_px} px)"
            )

            if args.save_dir:
                png_path = export_panel_png(
                    panel,
                    args.save_dir,
                    group=group,
                    rank=rank,
                    sample=sample,
                )
                exported_paths.append(png_path)

    if exported_paths:
        print(f"\nExportados {len(exported_paths)} paneles PNG en: {Path(args.save_dir).resolve()}")
    print("Listo. Abre TensorBoard en la pestaña IMAGES para inspeccionar o guardar las imágenes.")
    return 0


def main() -> int:
    return run_visualization()


if __name__ == "__main__":
    raise SystemExit(main())
