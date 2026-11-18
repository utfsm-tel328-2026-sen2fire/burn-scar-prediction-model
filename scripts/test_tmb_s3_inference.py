#!/usr/bin/env python3
"""
Demostración y verificación del generador de cicatriz congelado TMB-S3.

Carga el checkpoint de 5 bandas (convlstm_segformer_prepost_subset)
y ejecuta una pasada de inferencia sobre un tensor temporal dummy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Asegurar import de burn_scar desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from burn_scar.models import load_tmb_s3, predict_burn_scar


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("SMOKE TEST DE INFERENCIA TMB-S3 (PASO 3)")
    print(f"Dispositivo activo: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 68)

    t0 = time.time()
    print("Cargando modelo TMB-S3 (ConvLSTM + SegFormer, 5 bandas)...")
    model = load_tmb_s3(device=device)
    load_time = time.time() - t0
    print(f"Modelo cargado en {load_time:.2f} s. Todos los parámetros congelados.")

    # Tensor de prueba: 1 muestra, 7 frames temporales, 6 canales (5 OLCI + landcover), 64x64 px
    B, T, C, H, W = 1, 7, 6, 64, 64
    x = torch.randn(B, T, C, H, W, device=device)

    print(f"\nEjecutando inferencia sobre tensor de entrada: {list(x.shape)}...")
    t1 = time.time()
    probs = predict_burn_scar(model, x)
    infer_time = time.time() - t1

    print(f"Inferencia completada en {infer_time * 1000:.1f} ms.")
    print(f"Tensor de salida (probabilidad de quemado): {list(probs.shape)}")
    print(f"Rango de valores: min = {probs.min().item():.4f}, max = {probs.max().item():.4f}")
    print("=" * 68)
    print("✓ Generador de cicatriz TMB-S3 operativo para generar el canal 'scar'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
