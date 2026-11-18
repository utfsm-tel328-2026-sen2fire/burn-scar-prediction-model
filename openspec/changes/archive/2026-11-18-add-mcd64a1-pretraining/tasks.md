# Tasks

## 1. Diagnóstico de desfase temporal (R19)

- [x] 1.1 Crear `scripts/audit_mcd64a1_lag.py` para consultar `MODIS/061/MCD64A1` sobre las series del manifiesto CEMS, calcular el desfase mediano (`BurnDate - observed_time`) en días y guardar un resumen JSON; verificar ejecutándolo sobre una muestra o el manifiesto CEMS y comprobando el reporte de mediana y percentiles.

## 2. Minería de eventos MCD64A1 y exclusión CEMS (BL4, D15)

- [x] 2.1 Crear `scripts/build_mcd64a1_manifest.py` para extraer eventos quemados mediterráneos (2018–2026) desde `MODIS/061/MCD64A1` en GEE, agruparlos con DBSCAN (`eps=150 km`, `30 días`), excluir cualquier solapamiento espacio-temporal con los 96 eventos CEMS y exportar `data/mcd_manifest.json` con soporte `--max-events`; verificar con `--max-events 20` que ningún evento generado se solapa con CEMS.
- [x] 2.2 Añadir tests unitarios en `tests/test_mcd64a1_manifest.py` que validen la lógica de exclusión espacio-temporal contra CEMS y la asignación de ventanas/horizontes; verificar con `pytest tests/test_mcd64a1_manifest.py`.

## 3. Descarga de parches y ensamblaje PyTorch (D12, D17, D18)

- [x] 3.1 Crear `scripts/download_gee_mcd64a1.py` (reutilizando utilidades de `scripts/download_gee_olci.py`) para descargar de forma incremental y reanudable los parches `64×64` a `300 m` de MCD64A1 (`BurnDate`, `Uncertainty`, `QA`), Sentinel-3 OLCI (conservando nubes/humo reales, filtrando solo nodata de swath/nocturno) y estáticas (`DEM`, `WorldCover`) en `data/raw_mcd/`; verificar ejecutando sobre 3 eventos de prueba y comprobando los `.npz` generados.
- [x] 3.2 Crear `scripts/build_mcd64a1_dataset.py` para generar `x_scar` con TMB-S3 congelado, construir `target` (`BurnDate ∈ (f, f+h]` con *shift* opcional si R19 lo requiere) y `mask` (`255` en `BurnDate < 0` o `QA` inválido) en `data/processed_mcd/dataset_index.json`; verificar que `BurnScarDataset("data/processed_mcd/dataset_index.json")` carga muestras con las mismas formas y dtypes que CEMS.

## 4. Preentrenamiento M-A y ajuste fino M-B (B10)

- [x] 4.1 Crear `scripts/pretrain_mcd64a1.py` para entrenar `BurnScarPropagationModel` sobre `data/processed_mcd/dataset_index.json` con split interno train/val sin fuga y guardar el checkpoint base `mcd_pretrained.pt`, registrando métricas en `ExperimentTracker` y evaluando **M-A** en cero-disparo sobre los folds CEMS; verificar con una corrida corta (`--epochs 1`) sobre el lote de prueba.
- [x] 4.2 Añadir `--init-checkpoint` a `scripts/train_propagation.py` para inicializar cada fold de CEMS desde `mcd_pretrained.pt` (Etapa **M-B**) sin romper el entrenamiento desde cero; verificar con `--epochs 1 --folds 0 --init-checkpoint ...` que carga los pesos y completa el fold.
- [x] 4.3 Documentar el flujo de comandos (`R19` → minería → descarga → preentrenamiento M-A → ajuste M-B) en `docs/operacion.md` (o `README.md`) y ejecutar `openspec validate add-mcd64a1-pretraining`.
