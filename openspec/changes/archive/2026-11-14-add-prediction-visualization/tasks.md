# Tasks

## 1. Soporte de imágenes en ExperimentTracker

- [x] 1.1 Agregar `log_image(tag, img_tensor, global_step=None, *, dataformats="CHW", flush=None)` en [`src/burn_scar/tracking.py`](file:///home/antonio/Documents/U/TEL328/Proyecto%20parte%202/burn-scar-prediction-model/src/burn_scar/tracking.py) usando `self.writer.add_image(...)` y `self.flush()`, verificando con un test en `tests/test_tracking.py` que escribe el evento de imagen y llama a `flush()`.

## 2. Módulo de renderizado de paneles y selección de muestras

- [x] 2.1 Crear `src/burn_scar/visualization.py` con funciones para construir el panel RGB de 5 bloques (`OLCI pseudo-RGB | x_scar | target | probs | overlay TP/FP/FN`) respetando `mask == 255` y escalado nearest-neighbor, verificando dimensiones `(3, H*scale, W_total*scale)` y rango `[0, 1]` con pruebas en `tests/test_visualization.py`.
- [x] 2.2 Implementar en `src/burn_scar/visualization.py` la selección y ordenamiento de muestras evaluadas (`best`, `worst`, `both`, `series`, `indices`, `random`) y la exportación a PNG con `torchvision.utils.save_image`, verificando cada modo y la creación de archivos `.png` mediante pruebas unitarias en `tests/test_visualization.py`.

## 3. Script CLI de visualización por checkpoint

- [x] 3.1 Crear `scripts/visualize_predictions.py` con soporte para `--checkpoint`, `--fold` (opcional, inferido del checkpoint), `--hidden-ch` (opcional, inferido de `state_dict`), `--mode` (`best`, `worst`, `both`, `series`, `indices`, `random`), `--n-samples`, `--series-id`, `--indices`, `--threshold`, `--scale`, `--tag` y `--save-dir`; verificar ejecución `--help` y prueba end-to-end sobre un checkpoint sintético en `tests/test_visualization.py`.
- [x] 3.2 Documentar el uso de `scripts/visualize_predictions.py` y la descarga desde la pestaña IMAGES de TensorBoard en `docs/experiment-tracking.md`, verificando que los comandos documentados coinciden con los flags del CLI.

## 4. Verificación integral

- [x] 4.1 Ejecutar `uv run pytest tests/test_tracking.py tests/test_visualization.py` y `uv run ruff check src/burn_scar/tracking.py src/burn_scar/visualization.py scripts/visualize_predictions.py tests/test_tracking.py tests/test_visualization.py`, confirmando 0 errores.
