# Proposal: Diagnóstico de la definición del target de propagación

## Why

El target de propagación actual, `clip(CEMS(r+h) − scar_TMB3(f), 0, 1)`, **no es una medida del crecimiento**: es la diferencia entre el polígono CEMS de la ronda objetivo y una fotografía de TMB-S3 que, en la zona del frente de fuego, se queda corta. Medido sobre las 390 muestras, ese target tiene una **IoU de 0,112** con el crecimiento real entre rondas CEMS (`clip(CEMS(r+h) − CEMS(f), 0, 1)`) y una superficie mediana de **77 píxeles frente a 10** (7,7 veces más): la etiqueta mezcla quemado nuevo con quemado que ya existía y que el generador de cicatriz no detectó. Todo el desempeño publicado hasta ahora (F1 33,80% en el conjunto de test) está medido contra esa etiqueta contaminada, y su comparación con el F1 de TMB-S3 upstream (69,45, que evalúa **detección** de quemado con etiquetas MCD64A1 y no **pronóstico** de crecimiento) no es una comparación válida.

Con un coste de pocos minutos se puede medir si el cuello de botella es el modelo o la definición del enunciado, sin tocar el modelo ni las entradas: es lo que hace este cambio.

## What Changes

- El dataset empaquetado guarda **dos targets**: el target de propagación de siempre y, junto a él, un **target de referencia** con el crecimiento real entre la última ronda CEMS conocida en el instante *f* y la ronda objetivo, más la máscara de validez común a ambas rondas. El target de referencia es el instrumento de diagnóstico: no se usa para entrenar salvo que se pida explícitamente.
- El script de ensamblaje acepta un modo de target (`--target-mode scar`, por defecto, y `--target-mode cems`), de modo que se pueda generar un dataset con el crecimiento real como objetivo **sin cambiar ninguna entrada** (misma secuencia OLCI, mismo DEM, misma cobertura del suelo, misma cicatriz y mismo horizonte).
- La evaluación informa de las métricas **contra los dos targets** y del número de muestras y series evaluables en cada caso, de modo que el número de desempeño nunca pueda leerse fuera de contexto.
- Se ejecuta el experimento: reevaluar el modelo actual contra el target de referencia (sin reentrenar) y entrenar un modelo con el target de referencia (mismo protocolo), para separar la contribución del modelo y la de la etiqueta.
- Se corrige la comparación con TMB-S3 upstream: pasa de presentarse como una brecha a cerrar a lo que es, una tarea distinta; la vara de comparación es el baseline B1, que usa la misma información que el modelo.

## Capabilities

### New Capabilities
*(Ninguna.)*

### Modified Capabilities
- `propagation-dataset`: se añade el target de referencia y el modo de ensamblaje alternativo, dejando intacto el target de producción por defecto.
- `propagation-evaluation`: se añade el reporte doble (target de ensamblaje y target de referencia) con el recuento de evaluables de cada uno.

## Impact

- **Archivos modificados**: `scripts/build_pytorch_dataset.py` (target de referencia, modo de ensamblaje, máscara común), `scripts/train_propagation.py` (reporte de métricas contra ambos targets), `scripts/evaluate_baselines.py` (ídem), `src/burn_scar/dataset.py` (carga del target de referencia), `docs/operacion.md` y `docs/refinamiento_documentacion.md` (qué mide cada número).
- **Artefactos de datos**: los `.npz` de muestra pasan a llevar un array `target_cems` adicional; el dataset en `data/processed/` se regenera (≈20 s con GPU) y el dataset de diagnóstico se escribe en un directorio aparte, sin tocar el de producción.
- **Sin cambios en el modelo**: ni la arquitectura, ni las entradas, ni el protocolo de entrenamiento, ni los hiperparámetros.
- **Riesgo residual aceptado**: el target de referencia mide el crecimiento entre dos rondas CEMS, no exactamente en la ventana `[f, f+h]`; la ronda CEMS de *f* está a ~1 día del frame *f* de mediana y el horizonte mediano es 0,92 días, así que los dos intervalos son comparables pero no idénticos. Se acepta porque el instrumento es de diagnóstico.
- **Decisión que se aplaza**: redefinir o no el target de producción a la vista de los números. Este change solo aporta el instrumento y la evidencia.
