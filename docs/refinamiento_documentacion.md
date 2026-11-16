# Documentación Técnica: Refinamiento y Construcción del Dataset de Propagación

> **Proyecto:** *burn-scar-prediction-model*  
> **Fecha:** Noviembre 2026  
> **Objetivo:** Predicción multi-horizonte ($h \in [0.1, 14]$ días) del crecimiento de cicatrices de incendios forestales en la cuenca mediterránea a 300 m de resolución, utilizando datos abiertos y modelos congelados.

---

## 1. Resumen Ejecutivo

Este documento detalla el trabajo realizado para desbloquear, descargar, procesar y estructurar el dataset completo del proyecto, dejando el entorno **100% operativo para entrenar redes neuronales en PyTorch**.

El flujo implementado transforma datos satelitales en bruto (imágenes ópticas multiespectrales, modelos digitales de elevación, coberturas de vegetación y polígonos vectoriales de emergencia) en tensores normalizados y alineados causalmente, organizados en **5 bloques de validación cruzada geográfica sin fuga espacial**:

```
[ CEMS Rapid Mapping ] ──> Vector GeoJSON (649 capas) ──> Rasterizador 300 m ──> Target Y [0..1]
[ Sentinel-3 OLCI L1 ] ──> 5 bandas (2,130 pasadas)   ──┐
[ Copernicus DEM 30m ] ──> Elevación y pendiente      ──┼──> TMB-S3 (GPU) ──> Entrada X (T=7)
[ ESA WorldCover 10m ] ──> Tipo de vegetación         ──┘      (scar f)
                                                                 │
                                                                 ▼
                                                  BurnScarDataset (PyTorch)
                                                  (295 muestras en 5 Folds)
```

---

## 2. Decisiones Bloqueantes Cerradas (BL1 a BL7)

Previo a la descarga de imágenes, se formalizaron y cerraron las siete decisiones bloqueantes en [`docs/decisiones.md` §1.3](decisiones.md#13-decisiones-para-la-recopilación-de-datos-bl1-bl7--cerradas):

| ID | Decisión | Definición acordada | Justificación técnica |
|---|---|---|---|
| **BL1** | Ventana temporal | **$T=7$ frames**: 6 pasadas consecutivas terminando en $f$, más 1 adquisición pre-incendio. | Cubre ~12 días de propagación (mediana CEMS) y fija la dimensión temporal del tensor para *batching*. |
| **BL2** | Checkpoint TMB-S3 | **ConvLSTM de 5 bandas** (`Oa04, Oa06, Oa08, Oa17, Oa21`). | Paridad estadística comprobada con 21 bandas; reduce la descarga de datos un **75%**. |
| **BL3** | Dominio temporal | **2018 a 2026** dentro del recuadro `GEO_MED` (30°–46.5°N, -6°–31°E). | Evita el cambio de dominio de iluminación previo a mayo 2018 (cuando solo operaba Sentinel-3A). |
| **BL4** | Agrupación MCD64A1 | Clúster **DBSCAN ($\varepsilon=150$ km, 30 días)** sobre píxeles quemados. | Idéntico al protocolo de TMB-S3 para comparabilidad en preentrenamiento. |
| **BL5** | Canales versión 1 | **Sentinel-1 (SAR) queda diferido a la v2.** Se incluye FIRMS / VIIRS en v1. | Evita descargar gigabytes a 10 m para submuestrear a 300 m en la fase inicial. |
| **BL6** | Cobertura regional | **33 series puramente mediterráneas en la CV principal**; las 8 de ultramar/atlántico quedan como test complementario fuera de dominio (`Fold -1`). | Preserva la homogeneidad ecológica del régimen de fuego mediterráneo. |
| **BL7** | Calidad y nubes | **Banderas de calidad nativas** de OLCI extraídas directamente desde GEE/CDSE. | Consistencia exacta con los datos de entrenamiento de TMB-S3. |

---

## 3. Auditoría de Splits y Censo de Muestras (P7 y R14)

### 3.1 Auditoría frente a TMB-S3 (Paso 0)
Ejecutada con [`scripts/audit_tmb_splits.py`](../scripts/audit_tmb_splits.py):
- Se cruzaron las 96 series mediterráneas de CEMS con las 770 cajas de entrenamiento de TMB-S3.
- **Resultado:** 50 series estaban en el conjunto `train`/`val` de TMB-S3 (pesos memorizados). Fueron **excluidas** según la regla **D13** para evitar que la red aprenda sobre mapas perfectos en lugar de estimaciones.
- Quedaron **44 series limpias utilizables** (34 nunca vistas y 12 del split de test de TMB-S3).

### 3.2 Censo Real de Muestras (P7 Resuelto)
Mediante [`scripts/build_cems_manifest.py`](../scripts/build_cems_manifest.py) se cruzaron los intervalos de observación entre rondas CEMS con las pasadas efectivas de Sentinel-3 OLCI:
- Se identificaron **411 muestras $(f, h)$** potenciales sobre 44 series, tras colapsar tres productos OLCI duplicados del catálogo CDSE.
- Con el inventario local de adquisiciones se ensamblan **295 muestras finales** en 40 series (el dataset canónico se regeneró el 2026-11-16 con identidad de adquisición única, validación de capas estáticas e inventario de cobertura; ver [`operacion.md`](operacion.md) §9).
- Distribución de horizontes objetivos: **mediana $h = 0.897$ días**, $p_{25}=0.027\text{ d}$, $p_{75}=1.042\text{ d}$.

### 3.3 Auditoría de Cambio de Sensor (R14 Medido)
- **116 de 188 transiciones (61.7%) cambiaron de sensor** entre rondas consecutivas (ej. Legion 50 cm $\leftrightarrow$ Sentinel-2 10 m $\leftrightarrow$ SPOT-6/7 1.5 m).
- **Conclusión metodológica:** Se justifica al 100% la regla **D4**: la evaluación debe estratificarse entre transiciones con el mismo sensor y transiciones con sensor distinto para no penalizar al modelo por diferencias de agudeza óptica entre satélites comerciales y abiertos.

---

## 4. Pipeline de Ingesta y Procesamiento en 5 Pasos

### Paso 1: Descarga de Vectores CEMS
* **Script:** [`scripts/download_cems_vectors.py`](../scripts/download_cems_vectors.py)
* **Entrada:** Enlaces directos a AWS S3 indexados en `data/cems_manifest_46.json`.
* **Resultado:** 649 archivos GeoJSON (2.78 GB) descargados en `data/raw/cems/{series_id}/`.
* **Capas:**
  - `rNN_observedEventA.geojson`: Polígonos de quemado trazados por analistas de CEMS.
  - `rNN_notAnalysedA.geojson`: Zonas con nubes, sombras o fuera de cobertura del sensor.

### Paso 2: Rasterización Fraccional a 300 m
* **Script:** [`scripts/rasterize_cems_labels.py`](../scripts/rasterize_cems_labels.py)
* **Especificación:**
  - **Ventana fija (D17):** $64 \times 64$ píxeles a 300 m ($19.2 \times 19.2\text{ km}$) en la proyección UTM local (`EPSG:326XX`).
  - **Fracción continua (D2):** Se rasteriza a una sub-rejilla de 30 m ($640 \times 640$ píxeles) y se aplica un promedio por bloques $10 \times 10$. Produce valores reales en $[0.0, 1.0]$.
  - **Máscara tri-estado (D3):** `valido = 1` para píxeles analizados y `valido = 255` para zonas no analizadas o fuera del AOI.
* **Resultado:** 41 series con 218 rondas guardadas en archivos comprimidos `.npz` en `data/processed/cems_labels/` (**0.18 MB total**).

### Paso 3: Generador de Cicatriz TMB-S3 Congelado
* **Módulo:** [`src/burn_scar/models/tmb_s3.py`](../src/burn_scar/models/tmb_s3.py)
* **Arquitectura:** ConvLSTM temporal (kernel $3\times 3$, 64 canales ocultos) + Backbone MiniSegFormer de 3 etapas con Efficient Self-Attention (SR-Ratios 4, 2, 1).
* **Pesos:** Checkpoint `convlstm_segformer_prepost_subset` descargado desde Hugging Face (`links-ads/tmb-s3`, 4.62 MB).
* **Salida correcta:** el canal **0** del modelo es el logit de "quemado" (upstream `bafep/modules/mean_teacher.py`: `_FIRE_CH = 0`, entrenado con BCE sobre un único logit). El canal **1** es la *cabeza de nubes* (`_CLOUD_CH = 1`), entrenada con peso `lambda_cloud = 0`, por lo que sus valores no son fiables. La cicatriz se obtiene con `sigmoid(logits[:, :, 0])`.
* **Comportamiento:** Se carga en la GPU (NVIDIA RTX 5060) en modo `eval()`, con todos los pesos congelados (`requires_grad = False`). Produce el mapa de probabilidad de quemado actual `scar` sobre parches de $64\times 64$.

### Paso 4: Descarga de Imágenes Satelitales y Relieve (GEE)
* **Script:** [`scripts/download_gee_olci.py`](../scripts/download_gee_olci.py)
* **Plataforma:** Google Earth Engine (`burn-scar-prediction-model`).
* **Volumen descargado:**
  - **Sentinel-3 OLCI (5 bandas):** 2,130 pasadas temporales guardadas en `data/raw/s3_olci/{series_id}/` (**24 MB en disco**).
  - **Copernicus DEM (30 m):** Elevación del terreno remuestreada a 300 m en `data/raw/static/{series_id}/`.
  - **ESA WorldCover 2021 (10 m):** Clases de combustible y cobertura vegetal en `data/raw/static/{series_id}/`.

### Paso 5: Ensamblaje del Dataset PyTorch y Validación Cruzada
* **Script:** [`scripts/build_pytorch_dataset.py`](../scripts/build_pytorch_dataset.py)
* **Módulo Dataset:** [`src/burn_scar/dataset.py`](../src/burn_scar/dataset.py)
* **Operación:**
  1. Compone la ventana temporal de 7 frames OLCI como **1 frame pre-incendio + 6 post-incendio** que terminan en $f$ (BL1). El corte pre-incendio es la hora de adquisición de la primera ronda CEMS de la serie.
  2. Descarta las adquisiciones no utilizables: las que superan el **20% de píxeles sin dato** (mismo criterio que el dataset upstream) y las **nocturnas** (sin ningún píxel válido con señal).
  3. Sustituye cada píxel sin dato por la **media de su banda** en el frame (nunca 0: en las bandas visibles el 0 parecería quemado).
  4. Ejecuta TMB-S3 en GPU para obtener $\text{scar}(f) \in [0, 1]$.
  5. Calcula el target causal:
     $$\text{Target}(f, h) = \text{clip}\left(\text{CEMS}(r+1) - \text{scar}_{\text{TMB3}}(f),\, 0,\, 1\right)$$
  6. Calcula además el **target de referencia**, el crecimiento real entre la última ronda
     CEMS conocida en el instante $f$ ($r_f$) y la ronda objetivo:
     $$\text{Target}_{\text{ref}}(f) = \text{clip}\left(\text{CEMS}(r+1) - \text{CEMS}(r_f),\, 0,\, 1\right)$$
     con la validez común a las dos rondas. El dataset guarda **las dos etiquetas**
     (`target` y `target_cems`); la de ensamblaje tiene 11,1× más superficie que la de
     referencia y una IoU de solo 0,083 con ella (ver [`operacion.md`](operacion.md) §9).
  7. Asigna cada muestra a su Fold geográfico correspondiente.
* **Modos de ensamblaje:** `--target-mode scar` (por defecto, el target de producción) y
  `--target-mode cems` (diagnóstico: el target de propagación pasa a ser el crecimiento
  real; **las entradas del modelo son idénticas en ambos modos**, solo cambia la etiqueta).
  El modo se registra en `dataset_index.json`.
* **Resultado:** 295 archivos `.npz` en `data/processed/samples/` e índice maestro `data/processed/dataset_index.json` (cada entrada incluye `frame_keys` y `frame_key` para auditar la ventana temporal, además de su banda de horizonte, su estado de cicatriz y la versión de esquema del índice). El dataset canónico se regeneró el 2026-11-16; las exclusiones y sus motivos están desglosados en [`operacion.md`](operacion.md) §9.
* **Verificación:** 256/270 muestras de validación tienen target con quemado (≥0,5) y ninguna ventana contiene frames nocturnos ni valores centinela. Los invariantes de la ventana (terminar en `f` y estar ordenadas en el tiempo) se comprueban en `tests/test_dataset_invariants.py`, y la codificación canónica de las máscaras sobre el fichero almacenado, en `tests/test_generator_version.py`. El dataset **no se versiona en Git** (`data/` está en `.gitignore`): hay que regenerarlo en local tras cualquier cambio en el generador o en la ventana temporal (ver [`operacion.md`](operacion.md) §8). El índice registra además la versión del generador, y el cargador avisa si el dataset almacenado no corresponde al código vigente.

---

## 5. Distribución de los 5 Folds Geográficos (D14)

Para impedir cualquier fuga espacial (**D8**), todos los AOIs pertenecientes a una misma activación CEMS permanecen estrictamente en el mismo fold:

```
[ Fold 0: Iberia Oeste ] ──────────> 7 series ·  33 muestras  (España Central / Occidental)
[ Fold 1: Pirineos / Levante ] ────> 5 series ·  25 muestras  (España Oriental / Sur Francia)
[ Fold 2: Central Med ] ───────────> 9 series · 119 muestras  (Túnez, Italia Norte, Croacia)
[ Fold 3: Balcanes ] ──────────────> 8 series · 105 muestras  (Montenegro, Albania, Serbia)
[ Fold 4: Egeo / Grecia ] ─────────> 4 series ·  32 muestras  (Grecia Continental e Islas)
──────────────────────────────────────────────────────────────────────────────────────────
[ Fold -1: Fuera de Dominio ] ─────> 8 series ·  76 muestras  (Nueva Caledonia, Kerguelen, etc.)
```

---

## 6. Uso en PyTorch (`BurnScarDataset`)

### 6.1 Estructura del Lote (Batch)
Al iterar con `DataLoader`, cada elemento contiene:

| Clave | Forma (Shape) | Tipo | Rango / Normalización | Descripción |
|---|---|---|---|---|
| `x_olci` | `(B, 7, 5, 64, 64)` | `float32` | $[0.0, 5.0]$ (rad / 1000) | Secuencia temporal de 7 pasadas con 5 bandas OLCI |
| `x_scar` | `(B, 1, 64, 64)` | `float32` | $[0.0, 1.0]$ | Fracción de cicatriz previa en $f$ (TMB-S3) |
| `x_dem` | `(B, 1, 64, 64)` | `float32` | $[0.0, 3.0]$ (altitud en km) | Elevación del terreno Copernicus DEM |
| `x_worldcover` | `(B, 1, 64, 64)` | `float32` | $[0.1, 1.0]$ (clase / 100) | Tipo de combustible / vegetación ESA WorldCover |
| `horizon_days` | `(B, 1)` | `float32` | $[0.0, 16.0]$ | Días hacia el objetivo $h$ |
| `target` | `(B, 1, 64, 64)` | `float32` | $[0.0, 1.0]$ | **Lo nuevo quemado** entre $f$ y $f+h$ (Ground Truth) |
| `mask` | `(B, 1, 64, 64)` | `uint8` | $\{1, 255\}$ | Máscara de validez (`1` = evaluable, `255` = ignorar) |
| `target_cems` | `(B, 1, 64, 64)` | `float32` | $[0.0, 1.0]$ | Target de referencia: crecimiento real entre rondas CEMS |
| `mask_cems` | `(B, 1, 64, 64)` | `uint8` | $\{1, 255\}$ | Validez común a las dos rondas CEMS (misma codificación que `mask`) |
| `has_reference` | `(B, 1)` | `float32` | $\{0.0, 1.0\}$ | `1` si la muestra trae target de referencia |

> **Codificación canónica de las máscaras:** `1` = píxel evaluable, `255` = píxel ignorado
> (no observado en la ronda). Las dos máscaras usan la misma convención, también en el
> modo de ensamblaje `cems`; el cargador normaliza los paquetes antiguos que usaban `0`
> para "no evaluable".

### 6.2 Código de Entrenamiento
```python
import torch
from burn_scar.dataset import get_cv_dataloaders

# Configurar Validación Cruzada (entrena en folds 1..4 y valida en fold 0)
# num_workers=None resuelve según el backend: 0 en ROCm (AMD), 2 en CUDA.
# Ver docs/operacion.md §8: con procesos auxiliares en AMD la época tarda 74 s
# en lugar de 1,7 s.
train_loader, val_loader = get_cv_dataloaders(val_fold=0, batch_size=8)

for batch in train_loader:
    x_olci = batch["x_olci"].cuda()       # (8, 7, 5, 64, 64)
    x_scar = batch["x_scar"].cuda()       # (8, 1, 64, 64)
    h      = batch["horizon_days"].cuda() # (8, 1)
    y_true = batch["target"].cuda()       # (8, 1, 64, 64)
    mask   = batch["mask"].cuda()         # (8, 1, 64, 64)

    # 1. Forward pass
    # y_pred = model(x_olci, x_scar, h)
    
    # 2. Pérdida enmascarada (solo píxeles donde mask == 1)
    # loss = bce_masked_loss(y_pred, y_true, mask)
```

---

## 7. Modelos de Referencia (Baselines) y Evaluación

Implementados en [`src/burn_scar/baselines.py`](../src/burn_scar/baselines.py) y evaluables con [`scripts/evaluate_baselines.py`](../scripts/evaluate_baselines.py):

* **B0 (Persistencia):** Predice avance cero ($\hat{\Delta} = 0$). Es el suelo trivial del problema.
* **B1 (Dilatación Radial):** Dilata la cicatriz actual de forma isótropa a razón de $0.6\text{ km/día}$ ($2\text{ px/día}$ a 300 m), expandiendo el perímetro proporcionalmente a $h$.
* **B2 (Advección por viento):** Desplaza el frente en la dirección del viento observada (criterio de aceptación: cualquier red neuronal debe superar a B2).

---

## 8. Verificación y Suite de Tests

El entorno cuenta con **19 pruebas automatizadas** que garantizan la integridad del sistema:

```bash
uv run pytest -v
```

```
tests/test_baselines.py .....                                             [ 17%]
tests/test_dataset.py ..                                                 [ 23%]
tests/test_env.py .....                                                  [ 40%]
tests/test_propagation.py ...                                            [ 50%]
tests/test_tmb_s3.py ...                                                 [ 63%]
tests/test_tracking.py ......                                            [100%]
============================= 24 passed in 8.11s ==============================
```

---

## 9. Instrucciones para Reproducir el Pipeline Completo

Para regenerar todo el dataset desde cero en cualquier máquina con `uv`:

```bash
# 1. Auditar colisiones de TMB-S3 y generar el censo
uv run python scripts/audit_tmb_splits.py --json runs/audit_results.json

# 2. Construir manifiesto cruzando CEMS con catálogo OData de CDSE
uv run python scripts/build_cems_manifest.py

# 3. Descargar geometrías CEMS desde AWS S3 (sin credenciales)
uv run python scripts/download_cems_vectors.py

# 4. Rasterizar etiquetas a 300 m (64x64 px)
uv run python scripts/rasterize_cems_labels.py

# 5. Descargar parches Sentinel-3 OLCI y capas estáticas desde GEE
uv run python scripts/download_gee_olci.py

# 6. Precalcular cicatriz con TMB-S3 y ensamblar muestras del dataset
#    (≈20 s en GPU; hay que repetirlo tras cualquier cambio en el generador
#     de cicatriz o en la ventana temporal: el dataset no se versiona en Git)
uv run python scripts/build_pytorch_dataset.py

# 7. Ejecutar evaluación de los baselines de referencia
uv run python scripts/evaluate_baselines.py

# 8. Entrenar el modelo de propagación (≈4 min; usar nohup/tmux si crece)
uv run python scripts/train_propagation.py --epochs 15 --tag prop_v2
```
