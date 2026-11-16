# Proposal

## Why

El dataset vigente y las métricas asociadas no representan de forma fiable lo que el proyecto declara: el índice contiene 344 entradas pero solo 341 muestras únicas, una serie de test (`EMSR785_AOI01`) aporta WorldCover completamente vacío, no existe ninguna muestra de arranque en frío y la mayoría de los horizontes son menores de tres días. Como el evaluador consume directamente el índice, esas duplicaciones y la capa estática inválida contaminan los recuentos y las métricas publicadas; antes de publicar resultados hay que hacer reproducibles y honestos los datos, la cobertura y el protocolo de evaluación.

## What Changes

- **Identidad canónica de adquisiciones y muestras**: las entradas OLCI que representan la misma adquisición se colapsan de forma determinista; el índice exige identificadores y rutas de muestra únicos y el ensamblador aborta ante una colisión no resuelta.
- **Validación de capas estáticas**: WorldCover y DEM se validan antes de ensamblar; una capa ausente, vacía o fuera de las clases admitidas excluye la serie con un motivo explícito, en vez de producir entradas con ceros silenciosos. Se reintenta la recuperación de `EMSR785_AOI01` y, si la fuente no permite reconstruirla, se excluye de forma trazable.
- **Cobertura de arranque en frío explícita**: el índice registra el estado y la cobertura de `x_scar`; si no existen muestras genuinas con cicatriz vacía, el dataset y la evaluación declaran que el arranque en frío no está representado, sin fabricar ese dato alterando `x_scar` y el target.
- **Soporte de horizontes explícito**: se conserva `horizon_days` como valor continuo para todas las muestras observadas, sin imponer un máximo fijo ni descartar muestras por pertenecer a una banda; el índice publica la distribución empírica y la evaluación reporta métricas y recuentos por bandas, que se usan como estratos de diagnóstico y no filtran las métricas globales.
- **Métricas sobre muestras únicas y válidas**: la evaluación informa de entradas, muestras únicas, ficheros únicos y series; rechaza índices con duplicados y usa exactamente el mismo subconjunto para modelo y baselines.
- **Regeneración y republicación**: se regeneran los datasets `scar` y `cems`, se reentrenan las CV de producción y diagnóstico sobre el dataset canónico, se vuelve a congelar el umbral desde validación y se publica un único conjunto de métricas de test sin reutilizar cifras obsoletas.
- **BREAKING**: se sustituyen el dataset, los checkpoints y las métricas vigentes; los artefactos anteriores se conservan solo como histórico y se marcan como no publicables.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `propagation-dataset`: incorpora identidad única de adquisición/muestra, validación de capas estáticas, cobertura explícita de arranque en frío, soporte de horizontes y un inventario trazable del dataset.
- `propagation-evaluation`: evalúa únicamente muestras canónicas y válidas, agrega sobre identidades únicas, informa cobertura por horizonte y estado de cicatriz, y republica métricas y un umbral derivados del dataset corregido.

## Impact

- **Código**: `scripts/build_cems_manifest.py`, `scripts/download_gee_olci.py`, `scripts/build_pytorch_dataset.py`, `src/burn_scar/dataset.py`, `src/burn_scar/baselines.py`, `scripts/train_propagation.py` y `scripts/evaluate_baselines.py`.
- **Datos**: `data/cems_manifest_46.json`, `data/raw/static/`, `data/processed/` y `data/processed_target_cems/` se regeneran; el índice incluye un inventario de calidad y exclusiones.
- **Modelos y resultados**: se generan nuevas CV de producción y diagnóstico, con sus nuevos `operational_threshold.json`; los resultados actuales `prop_v4` dejan de ser la referencia publicable.
- **Tests**: se añaden invariantes de unicidad, validación estática, cobertura de cicatriz/horizonte, conteos de evaluación y regresión sobre el dataset regenerado.
- **Documentación**: `docs/dataset.md`, `docs/operacion.md`, `docs/decisiones.md` y `README.md` se sincronizan con la composición, el soporte y las métricas efectivamente verificadas.
