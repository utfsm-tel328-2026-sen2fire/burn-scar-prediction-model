# mcd64a1-pretraining Specification

## Purpose
Define la tubería de preentrenamiento con `MODIS/061/MCD64A1` en Google Earth Engine: diagnóstico del desfase de detección (`R19`), minería de eventos mediterráneos sin contaminación con CEMS, ensamblaje de muestras compatibles con `BurnScarDataset` y entrenamiento en dos etapas (`M-A` → `M-B`).

## Requirements

### Requirement: Diagnóstico de desfase temporal R19
El sistema SHALL proporcionar una verificación cuantitativa del desfase mediano entre `BurnDate` de MCD64A1 y `observed_time` de CEMS sobre los eventos CEMS existentes antes de ejecutar la descarga masiva.

#### Scenario: Cálculo del desfase mediano sobre eventos CEMS
- **WHEN** se ejecuta el diagnóstico `R19` sobre el manifiesto CEMS
- **THEN** el sistema consulta `MODIS/061/MCD64A1` en GEE para las series CEMS válidas y reporta la mediana, percentiles y distribución del desfase en días respecto a `observed_time`

### Requirement: Minería de eventos MCD64A1 sin contaminación CEMS
El sistema SHALL descubrir y catalogar eventos de incendio mediterráneos en `MODIS/061/MCD64A1` (2018–2026) agrupados por DBSCAN (`ε = 150 km`, ventana `30 días`), excluyendo cualquier evento que se solape en espacio o tiempo con los 96 eventos de CEMS.

#### Scenario: Exclusión estricta de eventos CEMS
- **WHEN** un clúster candidato de MCD64A1 se solapa espacial o temporalmente con alguno de los 96 eventos CEMS
- **THEN** el evento candidato MUST ser excluido del catálogo de preentrenamiento

#### Scenario: Lote inicial acotado y reanudable
- **WHEN** se solicita construir el catálogo de eventos con un límite máximo (por ejemplo, `--max-events 500`)
- **THEN** el sistema selecciona hasta ese límite priorizando eventos mediterráneos con señal suficiente y guarda un manifiesto JSON reanudable

### Requirement: Descarga de parches y ensamblaje compatible con BurnScarDataset
El sistema SHALL descargar parches `64×64` a `300 m` (MCD64A1, Sentinel-3 OLCI, DEM y WorldCover) y generar muestras `.npz` con exactamente el mismo esquema de entradas que `BurnScarDataset`, usando TMB-S3 congelado para `x_scar`, manteniendo nubes/humo reales en OLCI y enmascarando con `255` los píxeles de etiqueta sin dato o con `QA` inválido.

#### Scenario: Generación de target y máscara de validez desde MCD64A1
- **WHEN** se construye una muestra para un frame `f` y horizonte `h` en `{1, 2, 3, 5, 7, 10, 14}` días
- **THEN** `target` contiene los píxeles remuestreados por *nearest* con `BurnDate ∈ (f, f + h]` descontando lo ya quemado, y `mask` marca `255` donde `BurnDate < 0` o el `QA` indica observaciones inválidas

#### Scenario: Preservación de nubosidad real en entradas OLCI
- **WHEN** un frame OLCI tiene nubosidad o humo parcial pero cumple `MAX_NODATA_FRAC <= 0.20` (borde de swath/nocturno)
- **THEN** el frame SHALL conservarse en `x_olci` sin filtrarse por porcentaje de nubes

### Requirement: Entrenamiento en dos etapas M-A y M-B
El sistema SHALL permitir preentrenar `BurnScarPropagationModel` sobre el índice MCD64A1 (Etapa **M-A**), evaluar **M-A** en cero-disparo sobre CEMS, e inicializar el ajuste fino en 5 folds sobre CEMS (Etapa **M-B**) a partir del checkpoint preentrenado.

#### Scenario: Inicialización de M-B desde checkpoint M-A
- **WHEN** se ejecuta el entrenamiento CV sobre CEMS indicando `--init-checkpoint` con un checkpoint M-A válido
- **THEN** cada fold carga los pesos preentrenados como inicialización antes de entrenar y registrar métricas **M-B** en el tracker
