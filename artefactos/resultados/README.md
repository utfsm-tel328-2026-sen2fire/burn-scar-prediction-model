# Resultados y auditorías publicables

Este directorio guarda los **JSON de resultados que sí se versionan**, para que las cifras
publicadas en la documentación se puedan comprobar sin depender de la máquina de nadie.

No sustituye a [`../canonicos.json`](../canonicos.json), que es el manifiesto de sumas de
verificación (índices, umbral operativo y checkpoints del run publicado). Aquí están los
**números**, no solo sus huellas.

## Qué hay aquí

| Fichero | Qué es | Cómo se regenera |
|---|---|---|
| `baselines_canonicos_umbral_010.json` | B0, B1 y el trivial "todo quemado" sobre el dataset canónico (296 muestras), **umbral 0,10**, en validación y test | `uv run --extra rocm python scripts/evaluate_baselines.py --threshold 0.10 --output artefactos/resultados/baselines_canonicos_umbral_010.json` |
| `baselines_canonicos_umbral_020.json` | Lo mismo al **umbral operativo 0,20** del run canónico | `uv run --extra rocm python scripts/evaluate_baselines.py --threshold 0.20 --output artefactos/resultados/baselines_canonicos_umbral_020.json` |
| `integridad_dataset_produccion.json` | Auditoría del índice de producción: recuentos, horizontes, cobertura de cicatriz, errores | `uv run --extra rocm python scripts/audit_dataset_integrity.py --index data/processed/dataset_index.json --output artefactos/resultados/integridad_dataset_produccion.json` |
| `integridad_dataset_referencia.json` | La misma auditoría sobre el dataset de etiqueta de referencia (CEMS) | `uv run --extra rocm python scripts/audit_dataset_integrity.py --index data/processed_target_cems/dataset_index.json --output artefactos/resultados/integridad_dataset_referencia.json` |
| `prop_v4_integrity_reproduccion_local.json` | **Reproducción local** del run `prop_v4_integrity`: umbral operativo congelado, umbrales por fold, dispersión, huellas del dataset y de cada checkpoint, y métricas de test | `uv run --extra rocm python scripts/train_propagation.py --epochs 15 --seed 42 --tag prop_v4_integrity` |
| `prop_v4_integrity_reproduccion_local_barrido_umbrales.json` | Barrido de umbrales (0,10 y 0,20) sobre el test del run local, con una sola pasada de inferencia | `uv run --extra rocm python scripts/sweep_thresholds.py --checkpoint-dir checkpoints/<run> --split test --thresholds 0.1 0.2` |

## Canónico frente a reproducción

- **Los baselines son canónicos y reproducibles**: los dos JSON de baselines salen idénticos
  en cualquier máquina con el mismo dataset, y sus cifras coinciden con la tabla de
  [`docs/operacion.md`](../../docs/operacion.md) §9 (B1 15,94 % / 7,30 % y trivial 16,29 % /
  8,42 % en el test).
- **Las filas de los modelos (M-A, M-B, M-B + Tversky y producción)** salen del run
  publicado, cuyo JSON de métricas vive en la máquina de origen y **no se versiona**: lo que
  sí queda aquí son sus sumas de verificación, en [`../canonicos.json`](../canonicos.json).
- **La reproducción local es local**: el fichero `prop_v4_integrity_reproduccion_local*.json`
  viene de una GPU AMD (ROCm) y da **decimales distintos** de los del run publicado (NVIDIA):
  41,93 % / 7,92 % frente a 39,34 % / 10,59 % a umbral 0,20. El **umbral operativo se
  reproduce** (0,20) y el dataset sale con la misma composición (296 muestras, mismos folds,
  mismas bandas), pero la cicatriz de TMB-S3 y el entrenamiento se calculan en GPU y por eso
  los decimales no viajan entre máquinas. Está explicado en
  [`docs/operacion.md`](../../docs/operacion.md) §9.

> Por eso `dataset_content_sha256` **no es estable entre máquinas**: el índice incorpora la
> cobertura de la cicatriz de TMB-S3, que depende del hardware. Sí es estable entre
> regeneraciones en la misma máquina. Para comparar dos índices, el criterio es el
> inventario, no el hash.
