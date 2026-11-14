# Design

## Context

Ver `proposal.md` para la motivación. Actualmente:
- [`BurnScarPropagationModel`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/models/propagation.py#L48-L119) recibe `x_olci` `(B, T, 5, 64, 64)`, `x_dem`, `x_worldcover`, `x_scar` `(B, 1, 64, 64)` y `horizon_days`, devolviendo `logits` `(B, 1, 64, 64)`.
- [`train_propagation.py`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/scripts/train_propagation.py#L143-L152) guarda checkpoints `best_fold_<fold>.pt` con `{"fold", "epoch", "model_state_dict", "metrics"}`.
- [`ExperimentTracker`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/tracking.py#L147-L231) envuelve `SummaryWriter` pero solo expone `log_scalar` y `log_metrics`.
- TensorBoard (`tensorboard>=2.18`) ya incluye descarga individual de imágenes desde su pestaña **IMAGES** ("Save image" / botón de descarga en el visor), y `torchvision>=0.22` provee `torchvision.utils.save_image` para exportación por lotes a PNG.

## Goals / Non-Goals

**Goals:**
- Permitir inspeccionar cualquier checkpoint `best_fold_*.pt` sobre su split de validación correspondiente sin modificar el bucle de entrenamiento ni requerir nuevas dependencias.
- Soportar modos de selección `best`, `worst`, `both` (mejores + peores en una corrida), `series` (`--series-id`), `indices` (`--indices`) y `random`.
- Componer paneles legibles de `64x64` escalados (opcionalmente con factor entero nearest-neighbor para mejor legibilidad en TensorBoard/PNG) y permitir guardar copias PNG locales con `--save-dir`.

**Non-Goals:**
- No se construye una interfaz web propia (se aprovecha TensorBoard **IMAGES** y su exportación nativa, más `--save-dir` para lotes).
- No se modifican las métricas de entrenamiento ni el formato de checkpoints existentes.

## Decisions

### 1. Renderizado puro en PyTorch/NumPy (`src/burn_scar/visualization.py`)
- **Decisión**: Construir los paneles RGB `(3, H, W)` directamente con operaciones tensoriales de PyTorch/NumPy y exportar PNG opcionalmente con `torchvision.utils.save_image`.
- **Alternativa considerada**: Usar `matplotlib` para renderizar figuras con ejes y títulos. Descartada porque `matplotlib` no está en `pyproject.toml` y añade latencia/dependencias innecesarias para parches `64x64`.
- **Detalle visual**: Cada panel concatena horizontalmente 5 bloques separados por un borde gris de 2 px:
  1. **OLCI pseudo-RGB** (`x_olci[:, -1]` bandas `[2, 1, 0]` normalizadas por percentil/min-max robusto).
  2. **`x_scar`** (escala de grises/azul para cicatriz previa).
  3. **`target`** (verde sobre fondo oscuro; `mask == 255` en gris tenue).
  4. **`probs`** (`torch.sigmoid(logits)` como mapa de calor rojo-amarillo-blanco).
  5. **Overlay TP/FP/FN**: `x_scar` en azul/gris, `TP` en verde, `FP` en rojo, `FN` en amarillo, `mask == 255` atenuado.
  - Se aplicará `upsample` nearest-neighbor (p. ej. `scale=3` -> `192x192` por bloque) para que los píxeles `64x64` se vean nítidos al descargar desde TensorBoard o abrir el PNG.

### 2. Extensión compatible de `ExperimentTracker`
- **Decisión**: Añadir `log_image(tag, img_tensor, global_step=None, *, dataformats="CHW", flush=None)` en [`ExperimentTracker`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/tracking.py#L147-L231), reutilizando `self.flush()` para sincronización inmediata en NFS y permisos `0o664`.
- **Alternativa considerada**: Instanciar `SummaryWriter` directamente en el script. Descartada porque rompería la convención de aislamiento por usuario/namespace y permisos compartidos definida en `shared-experiment-tracking`.

### 3. Inferencia de `hidden_ch` desde el checkpoint
- **Decisión**: Inferir `hidden_ch` automáticamente inspeccionando la forma de `temporal_encoder.conv.weight` en `model_state_dict` si el usuario no pasa `--hidden-ch` explícitamente.
- **Rationale**: Los checkpoints actuales no guardan `hidden_ch`; inferirlo evita errores de carga cuando se prueban checkpoints entrenados con distintos tamaños de capa oculta.

## Risks / Trade-offs

- **[Peso de eventos en TensorBoard / NFS]** -> Limitar `--n-samples` por defecto (p. ej. `8`) para evitar archivos `tfevents` gigantes en el almacenamiento compartido.
- **[Muestras sin píxeles quemados en `target`]** -> En parches donde `target` es todo cero, `compute_metrics_numpy` puede dar F1=0 o F1=1 según el caso; el script mostrará `series_id`, `horizon_days` y `F1`/`IoU` en el tag/nombre de archivo para contextualizar el ranking.
