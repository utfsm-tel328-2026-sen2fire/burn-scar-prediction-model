# Proposal

## Why

Tras actualizar el modelo [`BurnScarPropagationModel`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/models/propagation.py#L48-L119) para incorporar correctamente la capa de cicatriz (`x_scar`), el entrenamiento solo registra métricas escalares en TensorBoard (`train_loss`, `val_loss`, `val_f1_event`, `val_iou_event`). No existe una forma de inspeccionar visualmente las predicciones espaciales de un checkpoint ya entrenado (`best_fold_*.pt`) ni de diagnosticar en qué casos el modelo acierta o falla sobre el terreno.

## What Changes

- Agregar soporte de registro de imágenes (`log_image`) en [`ExperimentTracker`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/tracking.py#L147-L231) con vaciado inmediato (`flush()`) y permisos compartidos para que las visualizaciones aparezcan en la pestaña **IMAGES** de TensorBoard.
- Crear un módulo de renderizado visual en `src/burn_scar/visualization.py` que construya paneles comparativos por muestra (`OLCI pseudo-RGB | x_scar | target | probabilidad | overlay TP/FP/FN` respetando `mask`).
- Crear un script CLI `scripts/visualize_predictions.py` que cargue un checkpoint (`best_fold_*.pt`), evalúe su split de validación sin fuga de datos, permita seleccionar muestras (**mejores**, **peores**, **por evento/índice** o **aleatorias**), publique los paneles en TensorBoard (exportables desde la propia interfaz web de TensorBoard) y opcionalmente guarde los PNG directamente en disco.

## Capabilities

### New Capabilities
- `prediction-visualization`: Inspección visual post-entrenamiento de checkpoints del modelo de propagación mediante paneles comparativos en TensorBoard (**IMAGES**) y exportación opcional a PNG, con modos de selección de muestras (`best`, `worst`, `series`, `indices`, `random`).

### Modified Capabilities
<!-- Ninguna capability existente cambia sus requisitos de contrato; ExperimentTracker se extiende de forma compatible. -->

## Impact

- `src/burn_scar/tracking.py`: nuevo método `log_image` en `ExperimentTracker`.
- `src/burn_scar/visualization.py`: utilidades puras PyTorch/NumPy para componer paneles RGB y overlays de error.
- `scripts/visualize_predictions.py`: nuevo script CLI de inferencia y visualización por checkpoint.
- `tests/test_visualization.py`: pruebas unitarias de composición de paneles, selección de muestras y logging de imágenes.
- Dependencias: sin nuevas dependencias externas (usa `torch`, `numpy`, `torchvision.utils.save_image` y `tensorboard` ya presentes en `pyproject.toml`).
