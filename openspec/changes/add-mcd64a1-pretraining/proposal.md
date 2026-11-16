# Proposal: Preentrenamiento con MCD64A1 (Etapa M-A → Ajuste M-B)

## Why

El conjunto limpio de CEMS cuenta actualmente con ~41–46 series mediterráneas (387 muestras en [`data/processed/dataset_index.json`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/data/processed/dataset_index.json)), lo que limita la capacidad de aprendizaje del modelo de propagación [`BurnScarPropagationModel`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/models/propagation.py) cuando se entrena desde cero. El diseño del proyecto (`B4`, `B6`, `B10`, `D12`, `D15`) contempla preentrenar el modelo con etiquetas diarias de `MODIS/061/MCD64A1` en Google Earth Engine (Etapa **M-A**) y luego inicializar con ese checkpoint el ajuste fino sobre los 5 folds de CEMS (Etapa **M-B**), pero esa tubería aún no está implementada en código.

## What Changes

- **Diagnóstico de desfase temporal (`R19`)**: Script de verificación sobre los eventos CEMS existentes para medir el desfase mediano entre `BurnDate` de MCD64A1 y `observed_time` de CEMS antes de descargar el pool masivo.
- **Minería de eventos MCD64A1 en GEE (`BL4`, `D15`)**: Extracción de eventos mediterráneos (2018–2026) agrupados por DBSCAN ($\varepsilon = 150\text{ km}$, ventana 30 días), excluyendo estrictamente los 96 eventos CEMS en espacio y tiempo para evitar contaminación de la CV (`R20`).
- **Descarga de parches y ensamblaje PyTorch (`D12`, `D17`, `D18`)**: Descarga de parches `64×64` a 300 m de MCD64A1 (`BurnDate`, `Uncertainty`, `QA`), Sentinel-3 OLCI y capas estáticas (DEM, WorldCover) —manteniendo nubes/humo reales en OLCI y filtrando solo por `QA`/nodata en la etiqueta— y generación de muestras compatibles con [`BurnScarDataset`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/dataset.py) usando TMB-S3 congelado.
- **Soporte de preentrenamiento M-A y ajuste M-B (`B10`)**: Modo o script de preentrenamiento sobre el índice MCD64A1 (evaluando **M-A** en cero-disparo sobre CEMS) y soporte en [`train_propagation.py`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/scripts/train_propagation.py) para inicializar los pesos desde un checkpoint preentrenado (`--init-checkpoint`) durante la CV de 5 folds (**M-B**).

## Capabilities

### New Capabilities
- `mcd64a1-pretraining`: Minería de eventos MCD64A1 en GEE sin solapamiento con CEMS, descarga de parches OLCI/estáticos/MCD64A1, construcción de muestras de preentrenamiento y flujo de entrenamiento en dos etapas (M-A → M-B).

### Modified Capabilities
<!-- No existing spec requirements are modified -->

## Impact

- **Nuevos scripts en `scripts/`**: diagnóstico `R19`, minería/descarga de eventos MCD64A1, ensamblaje de dataset de preentrenamiento y script/modo de preentrenamiento M-A.
- **Código existente en `src/burn_scar/` y `scripts/`**:
  - [`train_propagation.py`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/scripts/train_propagation.py): nuevo argumento `--init-checkpoint` (y soporte para evaluar M-A sin reentrenar o inicializar M-B).
  - [`dataset.py`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/dataset.py): reutilización directa de [`BurnScarDataset`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/dataset.py) mediante un índice separado `data/processed_mcd/dataset_index.json`.
- **Almacenamiento y GEE**: ~600–800 MB para un lote inicial de ~500 eventos (MCD64A1 pesa <25 MB en parches); requiere cuenta GEE autenticada y ejecución incremental con reanudación (*resume*) ante cuotas HTTP de GEE.
