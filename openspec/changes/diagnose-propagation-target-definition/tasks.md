# Tasks

## 1. Target de referencia en el ensamblaje

- [ ] 1.1 En `scripts/build_pytorch_dataset.py`, calcular para cada muestra la ronda CEMS `r_f` (última ronda con tiempo de adquisición anterior o igual al frame actual) y el crecimiento de referencia `clip(CEMS(r_target) − CEMS(r_f), 0, 1)`. Verificar que las 390 muestras actuales lo obtienen.
- [ ] 1.2 Calcular la validez común a las dos rondas (intersección de `valido[r_f]` y `valido[r_target]`) y usarla como máscara del paquete.
- [ ] 1.3 Añadir el conmutador `--target-mode {scar,cems}`: en modo `scar` el target sigue siendo `clip(CEMS(r_target) − scar_TMB3(f), 0, 1)` con la validez de la ronda objetivo; en modo `cems` es el crecimiento de referencia con la validez común. Verificar que el modo por defecto produce un dataset equivalente al actual.
- [ ] 1.4 Guardar `target_cems` en cada `.npz` junto al `target` principal, y registrar el modo de ensamblaje en `dataset_index.json`. Verificar que el cargador tolera la ausencia de `target_cems` en muestras antiguas.
- [ ] 1.5 Ensamblar el dataset de diagnóstico en un directorio propio (por ejemplo `data/processed_target_cems/`) sin tocar `data/processed/`, y comprobar que las entradas (OLCI, DEM, worldcover, cicatriz, horizonte) son idénticas bit a bit a las del dataset de producción.

## 2. Reporte de métricas contra las dos etiquetas

- [ ] 2.1 En `src/burn_scar/dataset.py`, exponer el target de referencia de cada muestra junto al target principal.
- [ ] 2.2 En `scripts/train_propagation.py`, reportar en el informe de cada fold y en el del conjunto de test las métricas contra el target de ensamblaje y contra el target de referencia, con el número de muestras y series evaluables de cada uno, y marcar la referencia como no disponible cuando falte.
- [ ] 2.3 En `scripts/evaluate_baselines.py`, reportar lo mismo para B0 y B1, y añadir el predictor trivial "todo quemado" como escala mínima de cada etiqueta. Verificar que el trivial queda claramente identificado como tal en el informe.
- [ ] 2.4 Verificar que el número ya publicado contra el target de ensamblaje no cambia al añadir el reporte doble.

## 3. Experimento de diagnóstico

- [ ] 3.1 Reevaluar el checkpoint del run existente contra el target de referencia, sin reentrenar, y anotar el F1, IoU, precisión y recall con el recuento de evaluables. Esta es la medición (A) de la Decisión 4.
- [ ] 3.2 Entrenar con el dataset de diagnóstico (`--target-mode cems`) usando el mismo protocolo que el run de producción (15 épocas, 5 folds, mismos hiperparámetros) y anotar sus métricas contra el target de referencia. Esta es la medición (B).
- [ ] 3.3 Redactar la tabla comparativa: para cada modelo, F1 contra su propia etiqueta, con el trivial "todo quemado" de esa etiqueta al lado; y la comparación (A) del modelo de producción entre las dos etiquetas.
- [ ] 3.4 Redactar la conclusión en términos de cuánto del desempeño actual es propagación real y cuánto es acertar la etiqueta contaminada, dejando explícito qué decisión queda abierta.

## 4. Documentación

- [ ] 4.1 En `docs/operacion.md`, documentar qué etiqueta usa el dataset de producción, cuál es el target de referencia y cómo se leen los números de desempeño; añadir el apartado del diagnóstico con su tabla y su conclusión.
- [ ] 4.2 En `docs/operacion.md`, corregir la comparación con TMB-S3 upstream: dejar de presentarlo como meta o brecha y explicar que evalúa detección con etiquetas MCD64A1 mientras que el proyecto evalúa pronóstico con incremento CEMS; señalar a B1 como la vara de comparación.
- [ ] 4.3 En `docs/refinamiento_documentacion.md`, actualizar la descripción del target del dataset y el procedimiento de ensamblaje con el conmutador de modo.
- [ ] 4.4 Ejecutar la suite completa de pruebas y verificar que ninguna línea añadida supera el límite de 100 caracteres.

## Workflow follow-up

- Con la tabla del grupo 3, decidir si se abre un change de producto para redefinir el target de propagación, y en ese caso cómo se define el crecimiento en operación, donde no existe CEMS.
- En paralelo, medir la calidad de TMB-S3 contra CEMS sobre nuestras series (P5 pendiente) para saber si el 0,112 de IoU proviene de la cicatriz o de la definición del crecimiento.
