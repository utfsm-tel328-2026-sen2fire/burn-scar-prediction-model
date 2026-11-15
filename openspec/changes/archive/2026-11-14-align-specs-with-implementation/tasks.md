# Tasks

## 1. Codificación canónica de la máscara de validez

- [x] 1.1 En `scripts/build_pytorch_dataset.py`, emitir la validez común entre rondas con la codificación canónica (`1` = evaluable, `255` = ignorado) y marcar íntegramente como `255` la validez común de una muestra sin ronda CEMS previa. Verificar con una prueba unitaria que un paquete generado en modo `cems` contiene solo `1` y `255` en ambas máscaras.
- [x] 1.2 En `src/burn_scar/dataset.py`, normalizar al cargar la máscara del target y la validez común a `{1, 255}` para tolerar paquetes antiguos. Verificar que un paquete con validez común en `{0, 1}` se carga con los píxeles no evaluables marcados como `255`.
- [x] 1.3 En `tests/test_dataset.py`, añadir la comprobación de la codificación canónica de las dos máscaras y del caso de muestra sin ronda previa. Verificar que las pruebas nuevas fallan con la codificación anterior y pasan con la nueva.

## 2. Recuentos de muestras evaluables en los informes

- [x] 2.1 En `src/burn_scar/baselines.py`, calcular en `aggregate_series_metrics` el número de muestras evaluables (muestra con métrica informativa) y el total de muestras, junto a los recuentos de series ya existentes. Verificar con una prueba que una serie con métricas no informativas aporta su total de muestras pero no suma ninguna evaluable.
- [x] 2.2 En `scripts/train_propagation.py`, propagar los recuentos de muestras a los dict por umbral (target de ensamblaje y target de referencia) y al resumen del conjunto de test. Verificar que el informe impreso muestra el recuento de muestras evaluables y el total para las dos etiquetas.
- [x] 2.3 En `tests/test_baselines.py` y `tests/test_propagation.py`, añadir la comprobación de los recuentos de muestras evaluables con un conjunto que incluye una verdad vacía. Verificar que las pruebas cubren el caso de una etiqueta no disponible.

## 3. Visualización coherente con la semántica NaN y el umbral de evaluación

- [x] 3.1 En `src/burn_scar/visualization.py`, evaluar la condición de "muestra con quemado" con el umbral de decisión en uso en lugar del `0,5` fijo. Verificar con una prueba que una muestra cuyo target tiene quemado por debajo de `0,5` pero por encima del umbral se cuenta como con quemado.
- [x] 3.2 En `src/burn_scar/visualization.py`, excluir del ranking `best`/`worst` las muestras cuya métrica no es informativa y reportar cuántas se han excluido. Verificar con una prueba que una muestra no informativa no aparece entre las mejores ni entre las peores.
- [x] 3.3 En `scripts/visualize_predictions.py`, calcular el promedio de F1 del fold solo sobre las muestras informativas e informar del recuento de informativas sobre el total evaluado. Verificar con una prueba que el promedio sigue siendo numérico con una sola muestra vacía en el fold.
- [x] 3.4 En `tests/test_visualization.py`, añadir las pruebas del tratamiento de las métricas no informativas en el ranking y en el promedio. Verificar que las pruebas fallan con el código anterior y pasan con el nuevo.

## 4. Documentación

- [x] 4.1 En `docs/refinamiento_documentacion.md`, documentar en la tabla del paquete de muestra las filas `target_cems` y `mask_cems`, con la codificación canónica de la máscara (`1` = evaluable, `255` = ignorar) y la nota de que el modo `cems` usa la validez común. Verificar que la tabla coincide con lo que emite el ensamblaje.
- [x] 4.2 En `docs/operacion.md`, documentar la codificación canónica de la máscara y precisar que "muestra evaluable" significa una muestra cuya métrica es informativa, no una muestra con quemado. Verificar que los informes descritos en el documento incluyen los recuentos nuevos.

## 5. Verificación integral

- [x] 5.1 Ejecutar `uv run pytest` y verificar que la suite completa pasa.
- [x] 5.2 Ejecutar `uv run ruff check` y verificar que no hay errores y que ninguna línea añadida supera el límite de 100 caracteres.
- [x] 5.3 Regenerar el dataset de diagnóstico con `--target-mode cems` y verificar que el paquete emitido usa la codificación canónica en ambas máscaras; comprobar además que el dataset de producción (`--target-mode scar`) queda byte a byte igual en sus máscaras.
- [x] 5.4 Ejecutar `openspec validate align-specs-with-implementation --strict` y `openspec validate --all --strict` y verificar que no hay hallazgos.

## Workflow follow-up

- Archivar el change con `openspec archive align-specs-with-implementation` para sincronizar los cuatro specs modificados.
- Reevaluar el run existente con `--eval-only` para comprobar los nuevos recuentos de muestras evaluables en el informe, sin reentrenar.
