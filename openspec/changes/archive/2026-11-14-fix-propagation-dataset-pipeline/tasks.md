# Tasks

## 1. Generador de cicatriz (TMB-S3)

- [x] 1.1 Cambiar `predict_burn_scar` en `src/burn_scar/models/tmb_s3.py` para devolver `sigmoid(logits[:, :, 0])` (logit de quemado) en lugar de `softmax(logits, dim=2)[:, :, 1]` (cabeza de nubes), y eliminar el import de `torch.nn.functional` si queda sin uso. Verificar que `tests/test_tmb_s3.py` sigue pasando.
- [x] 1.2 Verificar la coherencia del generador de cicatriz frente a los polígonos CEMS: la IoU media sobre las muestras del dataset debe superar 0,3 y la cicatriz no debe ser 1,0 en toda la patch. Verificar con un script de inspección sobre `data/processed/samples/`.

## 2. Construcción de la ventana temporal

- [x] 2.1 En `scripts/build_pytorch_dataset.py`, calcular la calidad de cada adquisición OLCI de la serie (fracción de píxeles sin dato y presencia de señal válida) y descartar las que superen el 20% de nodata o carezcan de señal (frames nocturnos). Verificar que ninguna ventana resultante contiene frames descartados.
- [x] 2.2 En `scripts/build_pytorch_dataset.py`, componer la ventana con una adquisición pre-incendio (la más reciente anterior a la hora de adquisición de la primera ronda CEMS de la serie) seguida de hasta seis adquisiciones post-incendio que terminan en el frame actual, rellenando con el frame más antiguo si no hay suficientes. Verificar que el primer frame de cada ventana es anterior a la primera ronda CEMS.
- [x] 2.3 Sustituir en `scripts/build_pytorch_dataset.py` los píxeles sin dato por la media de su banda dentro del frame, de modo que el tensor OLCI almacenado no contenga el centinela ni outliers de orden de magnitud. Verificar que ningún valor almacenado coincide con el centinela.

## 3. Métricas y ponderación

- [x] 3.1 En `src/burn_scar/baselines.py`, devolver NaN en `compute_metrics_numpy` cuando la verdad no contiene píxeles quemados por encima del umbral, y añadir un agregador por evento que ignore los NaN e informe del número de eventos evaluados. Verificar con una prueba unitaria que un target vacío no produce un F1 de 1,0.
- [x] 3.2 En `scripts/evaluate_baselines.py`, usar el agregador NaN-aware e informar de cuántos eventos han podido evaluarse. Verificar que la evaluación de B0 y B1 no reporta 100% sobre targets vacíos.
- [x] 3.3 En `src/burn_scar/models/propagation.py`, derivar el `pos_weight` de la proporción real de píxeles quemados del lote (acotado a un rango configurable) cuando el llamante no providea un valor explícito. Verificar con una prueba unitaria que el peso cambia con el desbalance del lote y que una sobrescritura explícita se respeta.
- [x] 3.4 En `scripts/train_propagation.py`, evaluar las métricas en un barrido de umbrales (0,1–0,7) y reportar el F1 a 0,5, el mejor F1 y el umbral que lo alcanza, usando el agregador NaN-aware. Verificar que el informe distingue el umbral por defecto del óptimo.

## 4. Umbral operativo y evaluación en el conjunto de test

- [x] 4.1 Guardar en cada checkpoint la configuración del modelo (`olci_channels`, `hidden_ch`, `scar_corruption_prob`) para poder reconstruirlo al evaluar sobre el conjunto de test. Verificar que el checkpoint contiene la clave `model_config`.
- [x] 4.2 Exponer en `evaluate_fold` las métricas por umbral (además del umbral por defecto y del óptimo) para poder evaluar con un umbral arbitrario ya congelado. Verificar que el informe de un fold incluye el F1 a 0,5, el mejor F1 y su umbral.
- [x] 4.3 Recoger el umbral óptimo de cada fold, consolidar un único umbral operativo como su mediana e informar de mínimo, máximo y dispersión, avisando si la dispersión supera 0,1. Verificar que el informe final muestra el umbral congelado y su rango entre folds.
- [x] 4.4 Evaluar el checkpoint de cada fold sobre el conjunto de test (fold -1) con el umbral ya congelado y reportar F1, IoU, precisión y recall. Verificar que las métricas de test se calculan con el umbral congelado y no con el óptimo de cada fold.
- [x] 4.5 Guardar el umbral operativo y las métricas de test en un JSON junto a los checkpoints para que la inferencia aplique el mismo valor. Verificar que el fichero se crea y contiene el umbral y las métricas.

## 5. Regeneración y verificación integral

- [x] 5.1 Regenerar el dataset con `uv run --extra rocm python scripts/build_pytorch_dataset.py` y verificar que al menos la mitad de las muestras contiene target ≥ 0,5, que ninguna ventana contiene frames nocturnos ni el centinela de nodata, y que el primer frame de cada ventana es pre-incendio. Verificar con un script de auditoría sobre el dataset regenerado.
- [x] 5.2 Reevaluar los baselines B0 y B1 sobre el dataset regenerado para fijar la referencia de comparación. Verificar que los valores publicados en el informe son distintos de 100% y que el número de eventos evaluados es coherente con la cobertura del dataset.
- [x] 5.3 Ejecutar la suite completa `uv run --extra rocm pytest tests/ -q` (24 tests en verde) y verificar el estilo con `uv run ruff check .`. Nota: en esta workstation NixOS el binario de ruff no arranca (carga glibc sin `nix-ld` instalado) — es una limitación del entorno, no del código; verificado por equivalente manual que ninguna línea añadida supera el límite de 100 caracteres ni deja imports sin usar.

## 6. Documentación

- [x] 6.1 Actualizar `docs/refinamiento_documentacion.md` para reflectar que la cicatriz se genera con el canal 0 (logit de quemado) de TMB-S3, que la ventana temporal es 1 frame pre-incendio + 6 post-incendio con filtrado de adquisiciones, y que los targets vacíos se reportan como NaN en vez de 1,0.
- [x] 6.2 Documentar en `docs/operacion.md` el procedimiento de regeneración del dataset (obligatorio antes de entrenar) y el uso de `nohup` para runs largos, incluyendo la nota de que el dataset no se versiona en Git.

## 7. Rendimiento de la carga de datos

- [x] 7.1 En `src/burn_scar/dataset.py`, añadir una función que resuelva `(num_workers, pin_memory)` según el backend (ROCm: 0 procesos auxiliares y sin memoria fijada; CUDA: 2 procesos auxiliares y memoria fijada) y usarla en `get_cv_dataloaders`. Verificar que en ROCm la época de entrenamiento baja de 74 s a menos de 2 s con el dataset completo.
- [x] 7.2 En `scripts/train_propagation.py`, permitir forzar la configuración con `--num-workers` y `--pin-memory`/`--no-pin-memory`, y aplicarla también al cargador de evaluación sobre el conjunto de test. Verificar que el entrenamiento completo (15 épocas × 5 folds) baja de ~2 h a menos de 3 min.
- [x] 7.3 En `src/burn_scar/models/propagation.py`, vectorizar la corrupción de la cicatriz para eliminar una sincronización de GPU por muestra del lote, conservando la decisión morfológica única por muestra. Verificar con una prueba unitaria que la corrupción conserva el rango [0,1] y no mezcla muestras.
- [x] 7.4 Documentar en `docs/operacion.md` la trampa del cargador de muestras en ROCm con la tabla de tiempos medidos y la forma de forzar la configuración.

## Workflow follow-up

- Regenerar el dataset y reentrenar la red con el dataset corregido; comparar el F1 obtenido contra los baselines B0/B1 y el umbral óptimo reportado.
- Archivar el cambio cuando los resultados del primer entrenamiento con el dataset corregido estén revisados por el equipo.

**Estado**: el primer entrenamiento completo con el dataset corregido ya se ejecutó (15 épocas × 5 folds, 131 s): F1 en CV 23,18% ± 8,33%, F1 en test (fold -1, umbral congelado 0,10) 33,80%, IoU 23,27%, recall 86,25%, frente a un baseline B1 de 10,80% de F1. Queda pendiente la revisión del equipo y el archivado del cambio.
