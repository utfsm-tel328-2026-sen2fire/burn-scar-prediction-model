#!/usr/bin/env python3
"""
Paso 5: Ensamblaje del Dataset de Propagación de Cicatrices en PyTorch.

1. Precalcula el canal de cicatriz conocida `scar` usando TMB-S3 en GPU.
2. Agrupa las 41 series en 5 folds de validación cruzada geográfica sin fuga (D8, D14).
3. Empaqueta las muestras alineadas (X: OLCI 7 frames + estáticas + scar + h, Y: nuevo quemado CEMS).

Salida:
    data/processed/dataset_index.json
    data/processed/samples/{sample_id}.npz
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

# Asegurar import de burn_scar desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from burn_scar.models import load_tmb_s3, predict_burn_scar

# Una adquisición con más de este porcentaje de píxeles sin dato se descarta
# (mismo criterio que la ficha del dataset TMB-S3 upstream).
MAX_NODATA_FRAC = 0.20

# Longitud de la ventana temporal: 1 frame pre-incendio + 6 post-incendio (BL1)
WINDOW_LEN = 7


def _frame_key(iso_timestamp: str) -> str:
    """'2026-09-12T15:47:00' -> '20260912T154700' (mismo formato que las claves de frame)."""
    return iso_timestamp.replace("-", "").replace(":", "")[:15]


def assign_geographic_folds(series_list: list[dict]) -> dict[str, int]:
    """Asigna cada serie a uno de los 5 folds geográficos (D8, D14, BL6)."""
    # 1. Agrupar series por código de activación EMSR para evitar que distintos AOIs del mismo fuego se separen
    by_code: dict[str, list[dict]] = {}
    for s in series_list:
        if s.get("geo_mediterranean", True):
            by_code.setdefault(s["code"], []).append(s)

    # 2. Ordenar activaciones por longitud geográfica (Oeste a Este)
    sorted_codes = sorted(by_code.keys(), key=lambda c: by_code[c][0]["center_wgs84"][0])

    n_acts = len(sorted_codes)
    chunk_size = n_acts / 5.0
    series_fold_map: dict[str, int] = {}

    for i, code in enumerate(sorted_codes):
        fold_idx = min(int(i / chunk_size), 4)
        for s in by_code[code]:
            series_fold_map[s["series_id"]] = fold_idx

    # 3. Series de ultramar / atlánticas quedan como test complementario fuera de dominio (fold -1, BL6)
    for s in series_list:
        if not s.get("geo_mediterranean", True):
            series_fold_map[s["series_id"]] = -1

    return series_fold_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Paso 5: Ensamblaje del Dataset PyTorch")
    parser.add_argument("--labels-manifest", default="data/processed/cems_labels/manifest_rasterized.json")
    parser.add_argument("--cems-manifest", default="data/cems_manifest_46.json")
    parser.add_argument("--samples-out", default="data/processed/samples")
    parser.add_argument("--index-out", default="data/processed/dataset_index.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("PASO 5: ENSAMBLAJE DEL DATASET PYTORCH (TMB-S3 + OLCI + CEMS)")
    print(f"Dispositivo activo: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    with open(args.labels_manifest, "r", encoding="utf-8") as f:
        labels_manifest = json.load(f)

    with open(args.cems_manifest, "r", encoding="utf-8") as f:
        cems_manifest = json.load(f)

    # 1. Asignar los 5 folds geográficos
    series_meta_list = labels_manifest["series"]
    series_fold_map = assign_geographic_folds(series_meta_list)

    print("\nAsignación de Folds Geográficos (CV 5 Folds, D14):")
    fold_counts = {f: 0 for f in range(5)}
    fold_counts[-1] = 0
    for sid, f in series_fold_map.items():
        fold_counts[f] += 1
    for f in range(5):
        print(f"   - Fold {f}: {fold_counts[f]} series mediterráneas")
    print(f"   - Fuera de dominio (ultramar/atlántico): {fold_counts[-1]} series")

    # 2. Cargar modelo TMB-S3 congelado
    print("\nCargando generador de cicatriz congelado TMB-S3 en GPU...")
    tmb_model = load_tmb_s3(device=device)

    cems_series_map = {s["series_id"]: s for s in cems_manifest["series"]}
    os.makedirs(args.samples_out, exist_ok=True)

    sample_index = []
    total_samples = 0
    start_time = time.time()

    print("\nProcesando muestras por serie:")
    for idx, s_meta in enumerate(series_meta_list, 1):
        sid = s_meta["series_id"]
        if sid not in cems_series_map:
            continue

        cems_s = cems_series_map[sid]
        samples = cems_s.get("samples", [])
        if not samples:
            continue

        # Cargar etiquetas CEMS (.npz)
        label_file = f"data/processed/cems_labels/{sid}.npz"
        if not os.path.exists(label_file):
            continue
        cems_npz = np.load(label_file)
        cems_quemado = cems_npz["quemado"]  # (T_rondas, 64, 64)
        cems_valido = cems_npz["valido"]    # (T_rondas, 64, 64)
        round_numbers = list(cems_npz["round_numbers"])

        # Cargar estáticas (.npz)
        static_file = f"data/raw/static/{sid}/static_layers.npz"
        if not os.path.exists(static_file):
            continue
        static_npz = np.load(static_file)
        dem = static_npz["dem"].astype(np.float32)  # (64, 64)
        worldcover = static_npz["worldcover"].astype(np.float32)  # (64, 64)

        # Cargar los frames OLCI de la serie y descartar los no utilizables:
        # - más de un 20% de píxeles sin dato (borde del swath)
        # - sin ninguna señal válida (adquisiciones nocturnas)
        olci_frame_files = sorted(glob.glob(f"data/raw/s3_olci/{sid}/*.npz"))
        if not olci_frame_files:
            continue

        frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for fpath in olci_frame_files:
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

        sorted_frame_keys = sorted(frame_cache.keys())
        if not sorted_frame_keys:
            continue

        # Corte pre-incendio: hora de adquisición de la primera ronda CEMS de la serie
        rounds_meta = cems_s.get("rounds", [])
        acq_times = [
            str(r.get("acquisition_time")) for r in rounds_meta if r.get("acquisition_time")
        ]
        pre_fire_cut = min((_frame_key(t) for t in acq_times), default="")
        pre_fire_keys = [k for k in sorted_frame_keys if k < pre_fire_cut]

        series_samples_count = 0
        for smp in samples:
            frame_iso = smp["frame_time"]
            target_round = smp["target_round"]
            horizon_days = float(smp["horizon_days"])
            sample_id = smp["sample_id"]

            if target_round not in round_numbers:
                continue
            r_idx = round_numbers.index(target_round)
            gt_target_burnt = cems_quemado[r_idx]
            gt_target_valid = cems_valido[r_idx]

            # Frame f actual
            f_dt = dt.datetime.fromisoformat(frame_iso.replace("Z", ""))
            f_key = f_dt.strftime("%Y%m%dT%H%M%S")

            # Ventana temporal: 1 frame pre-incendio + hasta 6 post-incendio (BL1).
            # El frame pre-incendio es la adquisición utilizable más reciente anterior
            # a la primera ronda CEMS; el resto son post-incendios que terminan en f.
            past_keys = [k for k in sorted_frame_keys if k <= f_key]
            if not past_keys:
                continue

            post_keys = [k for k in past_keys if k >= pre_fire_cut] or past_keys
            pre_key = pre_fire_keys[-1] if (pre_fire_keys and pre_fire_keys[-1] <= f_key) else None

            if pre_key:
                selected_keys = [pre_key] + post_keys[-(WINDOW_LEN - 1):]
            else:
                selected_keys = past_keys[-WINDOW_LEN:]
            while len(selected_keys) < WINDOW_LEN:
                selected_keys.insert(0, selected_keys[0])

            # Cargar los 7 frames (5 bandas cada uno)
            olci_tensor = np.stack([frame_cache[k][0] for k in selected_keys], axis=0)
            olci_valid_mask = np.stack([frame_cache[k][1] for k in selected_keys], axis=0)

            # Tensor OLCI shape: (7, 5, 64, 64) en orden [Oa04, Oa06, Oa08, Oa17, Oa21]
            # Limpiar nodata/fill de GEE OLCI (>= 32768 o <= 0)
            olci_clean = np.where(olci_valid_mask, olci_tensor, np.nan)

            # 3. Inferencia de TMB-S3 para obtener la cicatriz actual scar(f)
            # TMB-S3 espera bandas en orden [Oa21, Oa17, Oa08, Oa06, Oa04] (config.yaml: [20, 16, 7, 5, 3])
            # normalizadas (z-score por banda sobre píxeles válidos) + landcover en [0..1]
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
            tmb_in_np = np.concatenate([olci_tmb_norm, wc_seq], axis=1)  # (7, 6, 64, 64)

            tmb_tensor = torch.from_numpy(tmb_in_np).unsqueeze(0).to(device)
            with torch.no_grad():
                scar_probs = predict_burn_scar(tmb_model, tmb_tensor).cpu().numpy()[0]  # (7, 64, 64)

            # La cicatriz de entrada en el frame actual f es el último frame de la secuencia
            scar_f = scar_probs[-1]  # (64, 64) en [0..1]

            # Guardar OLCI sin el centinela de nodata: cada píxel sin dato se
            # sustituye por la media de su banda en el frame (valor neutro; el 0
            # introduciría un sesgo hacia "quemado" en las bandas visibles).
            olci_stored = olci_tensor.copy()
            for t in range(olci_stored.shape[0]):
                for b in range(olci_stored.shape[1]):
                    band = olci_stored[t, b]
                    band_valid = olci_valid_mask[t, b]
                    if band_valid.any():
                        band[~band_valid] = float(band[band_valid].mean())
                    else:
                        band[:] = 0.0

            # 4. Target de propagación: lo NUEVO quemado en (f, f+h] (D9)
            # Target = clip(poligono(r+1) - scar_TMB3(f), 0, 1)
            target_delta = np.clip(gt_target_burnt.astype(np.float32) - scar_f.astype(np.float32), 0.0, 1.0)

            # Guardar muestra en disco
            sample_file = os.path.join(args.samples_out, f"{sample_id}.npz")
            np.savez_compressed(
                sample_file,
                x_olci=olci_stored.astype(np.float16),
                x_dem=dem.astype(np.float32),
                x_worldcover=worldcover.astype(np.uint8),
                x_scar=scar_f.astype(np.float16),
                horizon_days=np.float32(horizon_days),
                target=target_delta.astype(np.float16),
                mask=gt_target_valid.astype(np.uint8),
            )

            fold_id = series_fold_map.get(sid, 0)
            sample_index.append({
                "sample_id": sample_id,
                "series_id": sid,
                "file": sample_file,
                "fold": fold_id,
                "horizon_days": horizon_days,
                "frame_keys": selected_keys,
                "sensor_changed": smp.get("sensor_changed", False),
                "geo_mediterranean": s_meta.get("geo_mediterranean", True),
                "in_tmb_test": s_meta.get("in_tmb_test", False),
            })

            series_samples_count += 1
            total_samples += 1

        print(f"[{idx:>2}/{len(series_meta_list)}] {sid:<18} -> {series_samples_count} muestras ensambladas (Fold {series_fold_map.get(sid, 0)})")

    # Guardar índice final del dataset
    with open(args.index_out, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_samples": len(sample_index),
            "folds_distribution": {
                f"fold_{f}": sum(1 for s in sample_index if s["fold"] == f)
                for f in range(5)
            },
            "out_of_domain_samples": sum(1 for s in sample_index if s["fold"] == -1),
            "samples": sample_index,
        }, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("RESUMEN DE ENSAMBLAJE (PASO 5)")
    print("=" * 72)
    print(f"Total de muestras ensambladas:  {total_samples}")
    print(f"Archivo de índice guardado en: {args.index_out}")
    print(f"Archivos de muestra (.npz) en:  {args.samples_out}")
    print(f"Tiempo total en GPU:           {elapsed:.2f} s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
