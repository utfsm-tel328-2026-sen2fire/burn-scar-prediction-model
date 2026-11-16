# Tasks

## 1. Inventario completo de adquisiciones por serie

- [x] 1.1 Cambiar en `scripts/download_gee_olci.py` la consulta de la colección OLCI para que use la extensión del área de interés (el `bbox_wkt` que ya viaja en el manifiesto CEMS) en lugar del punto del centroide, de modo que la consulta tenga la misma cobertura geográfica que el catálogo de referencia. Verificar: una serie con escenarios de S3B ausentes recupera esas escenas en la descarga y el recuento de frames por serie aumenta.
- [x] 1.2 Subir el límite de resultados de la consulta de imágenes de 60 a un valor holgado y emitir un aviso por stdout cuando el número de imágenes devueltas alcance ese límite. Verificar: ejecutar el script sobre una serie que antes devolvía exactamente 60 imágenes y comprobar en la salida que descarga más de 60 ficheros y que el aviso aparece.
- [x] 1.3 Volver a descargar el inventario de las 44 series y contar, para cada serie, cuántas de las adquisiciones que el catálogo de referencia lista para su ventana temporal quedan disponibles en disco. Verificar: el recuento por serie se imprime y ninguna serie se queda en el tope de la consulta.
- [x] 1.4 Registrar en `docs/decisiones.md` que el inventario se consultaba por un punto mientras el catálogo de referencia lo hacía por el polígono del AOI, que esa discrepancia excluía las escenas de S3B y cuál es el criterio de intersección vigente. Verificar: la entrada de decisión cita el reparto de 125 de 128 ausencias en escenas de S3B.

## 2. Emparejamiento temporal del frame f

- [x] 2.1 Añadir en `src/burn_scar/dataset.py` una función pura que parsee la marca de tiempo de una clave de frame a `datetime` y otra que elija, entre una lista de adquisiciones utilizables, la más cercana en el tiempo a un instante dado dentro de una tolerancia de 2 minutos. Verificar: test unitario con dos adquisiciones separadas por 1 segundo que devuelve la más cercana y test con una separación de 10 minutos que devuelve `None`.
- [x] 2.2 Añadir la función pura que compone la ventana temporal a partir de las adquisiciones utilizables, el frame `f` ya emparejado y el corte pre-incendio, con las dos ramas de D4 y el relleno de frames. Verificar: test unitario que comprueba que la ventana resultante tiene siete frames, termina en `f` y está en orden temporal no decreciente.
- [x] 2.3 Sustituir en `scripts/build_pytorch_dataset.py` la comparación de cadenas `k <= f_key` por el emparejamiento por cercanía de 2.1 y la composición por 2.2. Verificar: el script sigue ejecutándose con los mismos argumentos y el `dataset_index.json` registra las mismas `frame_keys` para las muestras cuya ventana ya era correcta.

## 3. Descarte explícito de la muestra sin adquisición de f

- [x] 3.1 Cuando no exista adquisición utilizable que corresponda a `f`, omitir la muestra, imprimir un aviso con su `sample_id` y contabilizarla en un contador de descartes. Verificar: test con una serie sin adquisición de `f` que comprueba que no se escribe ningún `.npz` para esa muestra y que el contador aumenta.
- [x] 3.2 Comprobar que el descarte no elimina muestras que sí tienen adquisición de `f` pero carecen de frame pre-incendio utilizable. Verificar: test con una serie sin adquisiciones anteriores a la primera ronda CEMS que confirma que la muestra se conserva con la ventana de las últimas siete adquisiciones.

## 4. Contadores de control en el informe del ensamblaje

- [x] 4.1 Añadir al resumen del ensamblaje de `scripts/build_pytorch_dataset.py` tres cifras: ventanas que terminan en `f`, ventanas ordenadas cronológicamente y muestras descartadas por falta de adquisición de `f`. Verificar: ejecutar el ensamblaje y comprobar que las tres cifras aparecen en la salida y que las dos primeras coinciden con el total de muestras ensambladas.
- [x] 4.2 Añadir un test que verifique sobre el dataset generado que todas las ventanas registradas terminan en el frame `f` de su muestra y que están ordenadas. Verificar: el test lee `dataset_index.json`, comprueba la condición para todas las muestras y falla si alguna la incumple.

## 5. Regeneración del dataset

- [x] 5.1 Regenerar el dataset por defecto en modo `scar` con el código corregido. Verificar: el `dataset_index.json` resultante tiene `target_mode: scar` y los contadores de 4.1 salen íntegros.
- [x] 5.2 Regenerar el dataset en modo `cems` con el código corregido. Verificar: el `dataset_index.json` resultante tiene `target_mode: cems` y los contadores de 4.1 salen íntegros.
- [x] 5.3 Comparar el tamaño y la composición de los dos datasets regenerados con los anteriores y explicar en `docs/operacion.md` la diferencia de número de muestras. Verificar: la documentación recoge el recuento de muestras antes y después y el motivo del descarte.

## 6. Reentrenamiento, reevaluación y republicación de cifras

- [x] 6.1 Reentrenar los 5 folds sobre el dataset regenerado en modo `scar` y congelar el umbral operativo con la mediana de los umbrales óptimos por fold. Verificar: existe un `operational_threshold.json` junto a los checkpoints con el umbral, los umbrales por fold y la dispersión.
- [x] 6.2 Reevaluar el conjunto de test (fold -1) con el umbral congelado y recalcular los baselines al mismo umbral con `--operational-threshold-from`. Verificar: los informes de modelo y de baselines declaran el mismo umbral de origen y el mismo número de muestras evaluables contra la misma etiqueta.
- [x] 6.3 Actualizar `docs/operacion.md` con las cifras nuevas de CV, de test y de baselines, y marcar explícitamente como sustituidas las cifras derivadas del dataset anterior. Verificar: no queda ninguna cifra antigua de F1 sin marcar en el documento.
- [x] 6.4 Ejecutar la suite completa de tests y verificar que pasa entera. Verificar: `pytest tests/ -q` termina sin fallos y los tests nuevos de 2.1, 2.2, 3.1, 3.2 y 4.2 están entre ellos.

## Workflow follow-up

- **Tarea 1.3 ejecutada** (2026-11-15). El inventario se descargó con la consulta por la
  extension del AOI: 331 frames nuevos (2452 en total, 44/44 series sin errores, 6 min).
  Con el inventario completo, 412 de las 414 muestras del manifiesto tienen su adquisicion de
  `f` disponible (antes 262). El dataset se regenero a 344 muestras (antes 240) y se
  reentrenaron los dos modelos (`prop_v4`, `prop_cems_diag_v4`); las cifras estan en
  `docs/operacion.md` §9.
- Archivar el change una vez revisados los artefactos, con lo que los deltas de `propagation-dataset` se incorporated al spec principal.
- Verificar el resultado archivado con `openspec validate --all` y revisar que el spec principal recoge los requisitos de emparejamiento, descarte y orden cronológico.
