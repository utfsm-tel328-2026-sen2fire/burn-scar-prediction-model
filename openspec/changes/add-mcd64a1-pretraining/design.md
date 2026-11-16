# Design: Preentrenamiento con MCD64A1 (Etapa M-A → Ajuste M-B)

## Context

Ver `proposal.md` para la motivación. El código actual implementa la tubería completa de CEMS (`scripts/build_cems_manifest.py`, `scripts/download_gee_olci.py`, `scripts/build_pytorch_dataset.py`, `scripts/train_propagation.py`) y consume muestras `.npz` mediante [`BurnScarDataset`](file:///home/antonio/Documents/U/TEL328/Proyecto parte 2/burn-scar-prediction-model/src/burn_scar/dataset.py). Cada muestra ocupa ~84 KB en disco y las entradas son parches `64×64` a `300 m`.

## Goals / Non-Goals

**Goals:**
- Implementar la Etapa 1 (**M-A**) de preentrenamiento con `MODIS/061/MCD64A1` y su conexión a la Etapa 2 (**M-B**) en `train_propagation.py` sin alterar el formato de `BurnScarDataset`.
- Verificar `R19` (desfase `BurnDate` vs `observed_time`) antes de descargar cientos de eventos.
- Garantizar separación estricta respecto a los 96 eventos CEMS (`D15`/`R20`) y reanudación ante límites de GEE.

**Non-Goals:**
- Descargar todo el archivo global 2000–2026: se acota al Mediterráneo 2018–2026 y con un presupuesto inicial configurable (`--max-events`, p. ej. 300–500).
- Añadir nuevos canales de entrada en v1 (ERA5/FIRMS o Sentinel-1 quedan fuera de este cambio).
- Usar MCD64A1 en inferencia o como métrica de evaluación (`D16`).

## Decisions

1. **Minería de eventos server-side en GEE + lote acotado (`BL4`)**
   - *Decisión:* Consultar `MODIS/061/MCD64A1` en GEE filtrando por la cuenca mediterránea (2018–2026) y `BurnDate > 0`, extraer centroides/fechas de clústeres quemados para aplicar DBSCAN ($\varepsilon = 150\text{ km}$, 30 días) en local y descartar cualquier solapamiento espacio-temporal con los 96 eventos CEMS.
   - *Alternativa descartada:* Descargar los ~108 GeoTIFF mensuales completos del Mediterráneo (~0.35–1 GB) para agrupar todo localmente. Es más lento y prescindible si GEE entrega los puntos/vectores quemados o polígonos mensuales resumidos.

2. **Reutilización total del esquema `BurnScarDataset` en índice separado**
   - *Decisión:* Guardar el preentrenamiento en `data/raw_mcd/` y `data/processed_mcd/dataset_index.json` manteniendo exactamente las claves (`x_olci`, `x_dem`, `x_worldcover`, `x_scar`, `horizon_days`, `target`, `mask`). Así `BurnScarDataset` funciona sin cambios y `train_propagation.py` puede evaluar M-A directamente sobre `data/processed/dataset_index.json` (CEMS).
   - *Alternativa descartada:* Crear una clase `Dataset` paralela para MCD64A1, lo que duplicaría la lógica de normalización y arriesgaría inconsistencias entre M-A y M-B.

3. **Preservación de nubes en OLCI vs. filtrado `QA` en el target MCD64A1**
   - *Decisión:* Mantener `MAX_NODATA_FRAC = 0.20` solo para descartar bordes de swath/noches en `x_olci`, conservando nubes y humo reales. En cambio, en `target` se aplica `mask = 255` donde `BurnDate < 0` o `QA` indique observaciones insuficientes (bits 5–7), y opcionalmente se aplica un *shift* temporal si el diagnóstico `R19` muestra un desfase mediano `> 2` días.
   - *Alternativa descartada:* Filtrar escenas OLCI por nubosidad baja, lo que introduciría sesgo frente a operación real.

4. **Preentrenamiento M-A + inicialización `--init-checkpoint` en `train_propagation.py`**
   - *Decisión:* Añadir un script `scripts/pretrain_mcd64a1.py` (o submodo) que entrena `BurnScarPropagationModel` sobre `data/processed_mcd/dataset_index.json` (con split interno train/val por año/clúster para early stopping) y guarda `mcd_pretrained.pt`. Luego `train_propagation.py --init-checkpoint mcd_pretrained.pt` carga esos pesos al inicio de cada fold CEMS para obtener **M-B**, y permite evaluar **M-A** puro en cero-disparo sobre los mismos folds CEMS.
   - *Alternativa descartada:* Mezclar muestras MCD64A1 y CEMS en el mismo dataloader, lo cual diluiría la recalibración fraccional a 300 m de CEMS y rompería la comparación limpia `M-A` vs `M-B`.

## Risks / Trade-offs

- **[Cuotas/timeouts HTTP de GEE al descargar miles de frames OLCI]** → Descarga incremental con manifiesto persistente, reintentos con *backoff*, y presupuesto inicial acotado (`--max-events 300..500`).
- **[Desfase `R19` > 2 días en `BurnDate`]** → Ejecutar el diagnóstico `R19` como primera tarea; si el desfase es sistemático, aplicar un *shift* configurable (`--burndate-shift-days`) al muestrear `(f, f+h]`.
- **[Resolución gruesa 500 m → 300 m en preentrenamiento]** → Mitigado por la Etapa 2 (`M-B`) con ajuste fino sobre etiquetas fraccionales de 300 m de CEMS.
- **[Deriva orbital post-2020 `R7`]** → Guardar `year` en metadatos del índice MCD64A1 para poder auditar o filtrar por año si hiciera falta.
