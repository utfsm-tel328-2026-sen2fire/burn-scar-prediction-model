#!/usr/bin/env python3
"""
Ensamblaje del dataset PyTorch de preentrenamiento MCD64A1 (Etapa M-A).

Para cada evento en `data/mcd_manifest.json` con parches descargados en `data/raw_mcd/`:
1. Construye secuencias de 7 frames Sentinel-3 OLCI (1 pre-incendio + 6 post-incendio, D18).
2. Calcula `x_scar` con el modelo TMB-S3 congelado en GPU/CPU (1 vez por frame `f`).
3. Para cada horizonte `h` en `{1, 2, 3, 5, 7, 10, 14}` días:
   - `target`: píxeles con `BurnDate ∈ (f, f + h]` (aplicando `--burndate-shift-days` si se indica),
     descontando lo ya quemado en `<= f`.
   - `mask`: `1` en píxeles terrestres válidos y `255` donde `BurnDate < 0` (`-1` sin dato,
     `-2` agua) o `QA` indica agua (`bit 0 == 0`) u observaciones insuficientes (`bits 5..7 != 0`).
4. Guarda las muestras `.npz` en `data/processed_mcd/samples/` y el índice en
   `data/processed_mcd/dataset_index.json`, 100% compatible con `BurnScarDataset`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from burn_scar.models import load_tmb_s3, predict_burn_scar

MAX_NODATA_FRAC = 0.20
WINDOW_LEN = 7
DEFAULT_HORIZONS = [1, 2, 3, 5, 7, 10, 14]


def _frame_key_to_date(frame_key: str) -> dt.date:
    return dt.datetime.strptime(frame_key[:8], "%Y%m%d").date()


def compute_mcd_target_and_mask(
    burn_ordinal: np.ndarray,
    qa: np.ndarray,
    f_date: dt.date,
    horizon_days: float,
    burndate_shift_days: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye `(target, mask)` a partir de `burn_ordinal` (64, 64) y `qa` (64, 64):
    - `target` float32 en {0.0, 1.0}: 1.0 si `(burn_ordinal + shift) ∈ (f_ord, f_ord + h]`.
    - `mask` uint8: `1` válido, `255` ignorar si `burn_ordinal < 0`, `bit 0 de QA == 0` (agua)
      o `bits 5..7 de QA != 0` (observaciones demasiado espaciadas / sin mapeo fiable).
    """
    f_ord = f_date.toordinal()
    end_ord = f_ord + int(round(horizon_days))

    has_burn = burn_ordinal > 0
    shifted_ord = np.where(has_burn, burn_ordinal + int(burndate_shift_days), 0)

    # Píxeles que arden en la ventana futura (f, f + h] (excluye automáticamente <= f_ord)
    in_window = has_burn & (shifted_ord > f_ord) & (shifted_ord <= end_ord)
    target = in_window.astype(np.float32)

    # Máscara de validez según QA de MODIS/061/MCD64A1 (docs/datos.md §4.4):
    # - burn_ordinal < 0 (-1 sin dato, -2 agua) -> inválido
    # - QA bit 0 == 0 (agua) -> inválido
    # - QA bits 5..7 != 0 (periodo de mapeo inválido / observaciones espaciadas) -> inválido
    is_land = (qa & 0b1) == 1
    qa_special = (qa >> 5) & 0b111
    valid_px = (burn_ordinal >= 0) & is_land & (qa_special == 0)

    mask = np.where(valid_px, np.uint8(1), np.uint8(255))
    return target, mask


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensambla el dataset PyTorch de preentrenamiento MCD64A1 (data/processed_mcd/)"
    )
    parser.add_argument("--manifest", default="data/mcd_manifest.json")
    parser.add_argument("--raw-root", default="data/raw_mcd")
    parser.add_argument("--samples-out", default="data/processed_mcd/samples")
    parser.add_argument("--index-out", default="data/processed_mcd/dataset_index.json")
    parser.add_argument(
        "--burndate-shift-days",
        type=int,
        default=0,
        help="Shift en días sobre BurnDate si el diagnóstico R19 lo requiere (por defecto 0).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Límite de eventos a ensamblar")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("ENSAMBLAJE DEL DATASET DE PREENTRENAMIENTO MCD64A1 (ETAPA M-A)")
    print(f"Dispositivo TMB-S3: {device} | Shift R19: {args.burndate_shift_days:+d} d")
    print("=" * 72)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    series_list = manifest.get("series", [])
    if args.limit:
        series_list = series_list[: args.limit]

    tmb_model = load_tmb_s3(device=device)
    raw_root = Path(args.raw_root)
    samples_out = Path(args.samples_out)
    samples_out.mkdir(parents=True, exist_ok=True)

    sample_index: list[dict] = []
    t0 = time.time()

    for idx, ev in enumerate(series_list, 1):
        sid = ev["series_id"]
        fold_id = int(ev.get("fold", idx % 5))
        horizons = ev.get("candidate_horizons_days", DEFAULT_HORIZONS)

        mcd_path = raw_root / "mcd64a1" / sid / "mcd64a1_layers.npz"
        static_path = raw_root / "static" / sid / "static_layers.npz"
        olci_files = sorted(glob.glob(str(raw_root / "s3_olci" / sid / "*.npz")))

        if not mcd_path.exists() or not static_path.exists() or not olci_files:
            continue

        mcd_npz = np.load(mcd_path)
        burn_ordinal = mcd_npz["burn_ordinal"].astype(np.int32)
        qa = mcd_npz["qa"].astype(np.uint8)

        # Si el parche no tiene ningún píxel quemado válido, saltar
        if not np.any(burn_ordinal > 0):
            continue

        static_npz = np.load(static_path)
        dem = static_npz["dem"].astype(np.float32)
        worldcover = static_npz["worldcover"].astype(np.float32)

        frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for fpath in olci_files:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            raw_bands = np.load(fpath)["olci_bands"]
            b5 = raw_bands[[3, 5, 7, 16, 20]] if raw_bands.shape[0] == 22 else raw_bands[:5]
            b5 = b5.astype(np.float32)
            valid = (b5 > 0) & (b5 < 32768)
            fill_frac = float((~valid).mean())
            has_signal = bool(valid.any()) and float(b5[valid].max()) > 0.0
            if fill_frac > MAX_NODATA_FRAC or not has_signal:
                continue
            frame_cache[fname] = (b5, valid)

        sorted_keys = sorted(frame_cache.keys())
        if not sorted_keys:
            continue

        burn_start_date = dt.date.fromisoformat(ev["burn_start_date"])
        burn_end_date = dt.date.fromisoformat(ev["burn_end_date"])
        pre_cut_key = burn_start_date.strftime("%Y%m%dT000000")
        pre_keys = [k for k in sorted_keys if k < pre_cut_key]

        # Elegir frames f ancla durante o justo antes del intervalo activo del incendio
        anchor_keys = [
            k
            for k in sorted_keys
            if (burn_start_date - dt.timedelta(days=2))
            <= _frame_key_to_date(k)
            <= burn_end_date
        ]
        if not anchor_keys:
            anchor_keys = sorted_keys[-min(len(sorted_keys), 3) :]

        series_samples = 0
        for f_key in anchor_keys:
            f_date = _frame_key_to_date(f_key)
            past_keys = [k for k in sorted_keys if k <= f_key]
            if not past_keys:
                continue

            post_keys = [k for k in past_keys if k >= pre_cut_key] or past_keys
            pre_key = pre_keys[-1] if (pre_keys and pre_keys[-1] <= f_key) else None

            if pre_key:
                selected_keys = [pre_key] + post_keys[-(WINDOW_LEN - 1) :]
            else:
                selected_keys = past_keys[-WINDOW_LEN:]
            while len(selected_keys) < WINDOW_LEN:
                selected_keys.insert(0, selected_keys[0])

            olci_tensor = np.stack([frame_cache[k][0] for k in selected_keys], axis=0)
            olci_valid_mask = np.stack([frame_cache[k][1] for k in selected_keys], axis=0)
            olci_clean = np.where(olci_valid_mask, olci_tensor, np.nan)

            # Inferencia TMB-S3 una sola vez por ventana f_key
            olci_tmb_order = olci_clean[:, [4, 3, 2, 1, 0], :, :]
            olci_tmb_norm = np.zeros_like(olci_tmb_order, dtype=np.float32)
            for b_idx in range(5):
                band_vals = olci_tmb_order[:, b_idx]
                valid_b = ~np.isnan(band_vals)
                if np.any(valid_b):
                    mean_b = float(np.mean(band_vals[valid_b]))
                    std_b = float(np.std(band_vals[valid_b]))
                    if std_b < 1e-6:
                        std_b = 1.0
                    olci_tmb_norm[:, b_idx] = np.where(
                        valid_b, (band_vals - mean_b) / std_b, 0.0
                    )

            wc_norm = (worldcover / 100.0).astype(np.float32)
            wc_seq = np.broadcast_to(wc_norm[np.newaxis, np.newaxis, :, :], (7, 1, 64, 64))
            tmb_in_np = np.concatenate([olci_tmb_norm, wc_seq], axis=1)

            tmb_tensor = torch.from_numpy(tmb_in_np).unsqueeze(0).to(device)
            with torch.no_grad():
                scar_probs = predict_burn_scar(tmb_model, tmb_tensor).cpu().numpy()[0]
            scar_f = scar_probs[-1]

            olci_stored = olci_tensor.copy()
            for t in range(olci_stored.shape[0]):
                for b in range(olci_stored.shape[1]):
                    band = olci_stored[t, b]
                    band_valid = olci_valid_mask[t, b]
                    if band_valid.any():
                        band[~band_valid] = float(band[band_valid].mean())
                    else:
                        band[:] = 0.0

            for h in horizons:
                target, mask = compute_mcd_target_and_mask(
                    burn_ordinal,
                    qa,
                    f_date=f_date,
                    horizon_days=float(h),
                    burndate_shift_days=args.burndate_shift_days,
                )
                # Descontar señal ya presente en scar_f si se desea mantener D9 consistente
                # manteniendo la supervisión de crecimiento futuro en (f, f+h]
                if not np.any(mask == 1):
                    continue

                sample_id = f"{sid}_{f_key}_h{int(h):02d}d"
                sample_file = str(samples_out / f"{sample_id}.npz")
                np.savez_compressed(
                    sample_file,
                    x_olci=olci_stored.astype(np.float16),
                    x_dem=dem.astype(np.float32),
                    x_worldcover=worldcover.astype(np.uint8),
                    x_scar=scar_f.astype(np.float16),
                    horizon_days=np.float32(h),
                    target=target.astype(np.float16),
                    mask=mask.astype(np.uint8),
                )

                sample_index.append(
                    {
                        "sample_id": sample_id,
                        "series_id": sid,
                        "file": sample_file,
                        "fold": fold_id,
                        "year": int(ev.get("year", f_date.year)),
                        "horizon_days": float(h),
                        "frame_keys": selected_keys,
                        "sensor_changed": False,
                        "geo_mediterranean": True,
                        "in_tmb_test": False,
                    }
                )
                series_samples += 1

        print(
            f"[{idx:>2}/{len(series_list)}] {sid:<20} -> "
            f"{series_samples} muestras (Fold {fold_id})"
        )

    os.makedirs(os.path.dirname(args.index_out) or ".", exist_ok=True)
    with open(args.index_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "MODIS/061/MCD64A1",
                "burndate_shift_days": args.burndate_shift_days,
                "total_samples": len(sample_index),
                "folds_distribution": {
                    f"fold_{f}": sum(1 for s in sample_index if s["fold"] == f)
                    for f in range(5)
                },
                "samples": sample_index,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print(
        f"Total muestras MCD64A1: {len(sample_index)} | "
        f"Índice: {args.index_out} ({time.time() - t0:.1f} s)"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
