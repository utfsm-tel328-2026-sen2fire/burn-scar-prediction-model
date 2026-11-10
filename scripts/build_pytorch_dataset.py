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

        # Cargar todos los frames OLCI disponibles de la serie
        olci_frame_files = sorted(glob.glob(f"data/raw/s3_olci/{sid}/*.npz"))
        if not olci_frame_files:
            continue

        olci_frames_map = {}
        for fpath in olci_frame_files:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            olci_frames_map[fname] = fpath

        sorted_frame_keys = sorted(olci_frames_map.keys())

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

            # Identificar hasta 6 frames pasados + 1 frame pre-incendio (BL1)
            past_keys = [k for k in sorted_frame_keys if k <= f_key]
            if not past_keys:
                continue

            # Tomar los últimos 7 frames (o rellenar el inicio si hay menos)
            selected_keys = past_keys[-7:]
            while len(selected_keys) < 7:
                selected_keys.insert(0, selected_keys[0])

            # Cargar los 7 frames (5 bandas cada uno)
            olci_seq = []
            for k in selected_keys:
                frame_data = np.load(olci_frames_map[k])
                raw_bands = frame_data["olci_bands"]  # (22, 64, 64) o (5, 64, 64)
                if raw_bands.shape[0] == 22:
                    b5 = raw_bands[[3, 5, 7, 16, 20]]
                else:
                    b5 = raw_bands[:5]
                olci_seq.append(b5.astype(np.float32))

            # Tensor OLCI shape: (7, 5, 64, 64)
            olci_tensor = np.stack(olci_seq, axis=0)

            # 3. Inferencia de TMB-S3 para obtener la cicatriz actual scar(f)
            # Normalización de entrada para TMB-S3 (5 bandas + landcover normalizado)
            tmb_in_seq = []
            wc_norm = (worldcover / 100.0).astype(np.float32)
            for t_idx in range(7):
                b5_norm = olci_tensor[t_idx] / 100.0
                combined_6ch = np.concatenate([b5_norm, wc_norm[np.newaxis, :, :]], axis=0)
                tmb_in_seq.append(combined_6ch)

            tmb_tensor = torch.from_numpy(np.stack(tmb_in_seq, axis=0)).unsqueeze(0).to(device)
            with torch.no_grad():
                scar_probs = predict_burn_scar(tmb_model, tmb_tensor).cpu().numpy()[0]  # (7, 64, 64)

            # La cicatriz de entrada en el frame actual f es el último frame de la secuencia
            scar_f = scar_probs[-1]  # (64, 64) en [0..1]

            # 4. Target de propagación: lo NUEVO quemado en (f, f+h] (D9)
            # Target = clip(poligono(r+1) - scar_TMB3(f), 0, 1)
            target_delta = np.clip(gt_target_burnt.astype(np.float32) - scar_f.astype(np.float32), 0.0, 1.0)

            # Guardar muestra en disco
            sample_file = os.path.join(args.samples_out, f"{sample_id}.npz")
            np.savez_compressed(
                sample_file,
                x_olci=olci_tensor.astype(np.float16),
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
