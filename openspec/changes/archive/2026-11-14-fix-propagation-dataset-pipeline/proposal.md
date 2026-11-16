# Proposal: Corrección del pipeline de dataset y evaluación de propagación

## Why

El dataset de propagación generado por `scripts/build_pytorch_dataset.py` es **degradado de forma silenciosa**: el 99,7% de sus 390 muestras tiene el target (nuevo quemado) vacío, y la métrica de evaluación devuelve F1/IoU = 1,0 (100%) para cualquier predicción sobre esas muestras, lo que hace que un entrenamiento sobre datos vacíos parezca perfecto. La causa raíz es que `predict_burn_scar` lee el canal equivocado del checkpoint TMB-S3 (la cabeza de nubes, entrenada con peso 0) en lugar del logit de quemado, por lo que la cicatriz de entrada vale ~1,0 en todos los píxeles y la resta `clip(CEMS − cicatriz, 0, 1)` colapsa a cero. A ello se suman tres defectos de construcción de la ventana temporal y de las entradas que impiden que el modelo aprenda física de propagación: ausencia del frame pre-incendio (BL1 no implementado), inclusión de frames nocturnos (44,1% de las ventanas) y valores de relleno 32768 almacenados como si fueran radiancia.

Aparte de la corrección del dataset, la carga de las muestras imponía un coste **43× superior al del cómputo del modelo** en el backend ROCm: el entrenamiento completo de 15 épocas × 5 folds tardaba ~2 h en lugar de ~2 min, por un cuello de botella que no estaba en la GPU sino en el cargador de datos.

## What Changes

- **BREAKING** — `predict_burn_scar` pasa a devolver `sigmoid(logits[:, :, 0])` en lugar de `softmax(logits, dim=2)[:, :, 1]`. El canal 0 es el logit de quemado (upstream: `_FIRE_CH = 0`, entrenado con BCE sobre un único logit); el canal 1 es la cabeza de nubes (`_CLOUD_CH = 1`, peso `lambda_cloud = 0`) y sus valores no son fiables. Esto invalida el dataset ya generado: `x_scar` y `target` están horneados en los `.npz` y deben regenerarse.
- La selección de la ventana temporal pasa a componer **1 frame pre-incendio + 6 frames post-incendio** (BL1), usando la fecha de la primera ronda CEMS como corte de "pre-incendio".
- Se añade un **filtro de calidad de adquisición** que descarta los frames con más del 20% de nodata (misma regla que upstream) y los frames nocturnos (todos los píxeles válidos a cero), antes de componer la ventana.
- Los **valores de relleno (32768) se eliminan** de las entradas: cada píxel nodata se sustituye por la media de su banda en el frame, en lugar de almacenarse como un outlier de radiancia.
- `compute_metrics_numpy` devuelve **NaN** cuando la verdad no contiene ningún píxel quemado, en lugar de 1,0; se añade un agregador por serie que ignora los NaN e informa de cuántas series han podido evaluarse.
- La evaluación informa del **F1 en el mejor umbral** (barrido 0,1–0,7) además del F1 a 0,5, para que los números de un modelo probabilístico sean comparables con los baselines binarios B0/B1.
- El umbral operativo se **consolida en un único valor** (mediana de los umbrales óptimos por fold de validación), se **congela** y se usa para evaluar el **conjunto de test** (fold -1), de modo que el número publicado no se haya visto durante la elección del checkpoint ni del umbral. El umbral queda guardado junto a los checkpoints para la inferencia.
- `masked_bce_dice_loss` calcula el `pos_weight` **a partir del desbalance real del batch** (acotado) en lugar del valor fijo 4,0.
- La carga de las muestras se configura **según el backend de GPU**: en ROCm (AMD) se desactiva el proceso auxiliar y la memoria fijada, porque bifurcarse desde un proceso con el contexto HIP inicializado y reservar memoria fijada multiplicaban por 43 el tiempo de cada época (74 s → 1,7 s); en CUDA se mantienen los valores habituales. Ambos se pueden forzar desde la línea de comandos.
- La corrupción de la cicatriz de entrada se vectoriza, eliminando una sincronización de GPU por muestra del lote sin cambiar el resultado observado.

## Capabilities

### New Capabilities
- `propagation-dataset`: Generación del dataset de propagación: producción de la cicatriz con TMB-S3 congelado, composición de la ventana temporal multi-frame, tratamiento de nodata y de frames nocturnos, y semántica del target de nuevo quemado.
- `propagation-evaluation`: Métricas de evaluación del modelo de propagación (F1/IoU/precisión/recall por evento, umbral óptimo, detección de targets vacíos) y ponderación de la pérdida enmascarada.

### Modified Capabilities
*(Ninguna; las capacidades existentes `execution-environment` y `shared-experiment-tracking` no cambian de comportamiento.)*

## Impact

- **Archivos modificados**: `src/burn_scar/models/tmb_s3.py` (canal de salida), `scripts/build_pytorch_dataset.py` (ventana temporal, filtro de calidad, limpieza de nodata), `src/burn_scar/baselines.py` (métricas NaN + agregador por serie), `scripts/evaluate_baselines.py` (agregación NaN-aware), `src/burn_scar/models/propagation.py` (`pos_weight` adaptativo y corrupción vectorizada), `src/burn_scar/dataset.py` (configuración de la carga según backend), `scripts/train_propagation.py` (barrido de umbral, uso del agregador, opciones de carga).
- **Artefactos de datos**: el dataset en `data/processed/` debe regenerarse por completo (≈20 s con GPU) después de aplicar el cambio; no requiere re-descargar datos ni pesos. El ajuste de rendimiento de la carga no exige regenerarlo.
- **Resultados del primer entrenamiento completo** (15 épocas × 5 folds, 131 s): F1 en CV 23,18% ± 8,33%, F1 en el conjunto de test (fold -1, umbral congelado 0,10) 33,80%, IoU 23,27%, precisión 25,26%, recall 86,25%. Supera al baseline B1 (10,80% de F1) pero queda por debajo de TMB-S3 upstream (F1 69,45 / IoU 59,81).
- **Compatibilidad**: los checkpoints y datasets generados antes de este cambio quedan obsoletos y no son comparables con los nuevos. Los `.npz` de muestras son binarios, no versionables en Git, por lo que la regeneración es local.
- **Riesgo residual aceptado**: la normalización de la entrada OLCI sigue sin coincidir con la de upstream (z-score por imagen sobre radiancia en lugar de medias fijas sobre reflectancia TOA); se deja fuera de este cambio por requerir la conversión radiancia→reflectancia del sensor OLCI.
