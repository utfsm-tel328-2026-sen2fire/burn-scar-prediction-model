# Tasks

> Este change se aplica **después** de `fix-propagation-window-alignment` (ver design.md D6): ambos tocan el generador y obligan a regenerar el dataset. Aplicados en ese orden, se regenera una sola vez.

## 1. Filtro de adquisiciones sin dato medido sobre píxeles

- [x] 1.1 Cambiar en `scripts/build_pytorch_dataset.py` el cálculo de la fracción de dato ausente para que se mida sobre los píxeles que no tienen ninguna banda con dato válido, en lugar de sobre las celdas de banda. Verificar: test unitario con un tensor sintético en el que un píxel tiene cuatro bandas válidas y una sin dato, que confirma que ese píxel no cuenta como ausente.
- [x] 1.2 Medir sobre el inventario real las adquisiciones cuyo veredicto de utilizabilidad cambia al pasar de celdas a píxeles, y dejar el recuento en la salida del ensamblaje. Verificar: el recuento de adquisiciones utilizables antes y después aparece en la salida y la diferencia está explicada.
- [x] 1.3 Actualizar en `docs/dataset.md` la regla de descarte de frames sin dato para que la unidad de medida (píxel sin ninguna banda válida) quede explícita y coincida con la regla D3. Verificar: la regla describe el mismo criterio que el que aplica el generador.

## 2. Codificación canónica en el paquete almacenado

- [x] 2.1 Añadir en `scripts/build_pytorch_dataset.py` una comprobación, antes de escribir cada paquete, de que `mask` y `mask_cems` contienen únicamente los valores `1` y `255`, abortando el ensamblaje si no es así. Verificar: test que fuerza una máscara con un valor inesperado y comprueba que el generador aborta en lugar de escribir el paquete.
- [x] 2.2 Añadir en `tests/test_dataset.py` un caso que abra el fichero `.npz` de una muestra almacenada, sin pasar por el cargador, y compruebe los conjuntos de valores de `mask` y `mask_cems`. Verificar: el test falla contra el dataset actual, que usa `0` en `mask_cems`, y pasa tras la regeneración de 4.1 y 4.2.
- [x] 2.3 Documentar en `src/burn_scar/dataset.py` que la normalización en lectura existe por compatibilidad con paquetes antiguos y no es un efecto colateral. Verificar: el docstring lo declara y el test de compatibilidad con paquetes antiguos sigue cubriendo el caso.

## 3. Versión del generador y detección de dataset obsoleto

- [x] 3.1 Añadir en `src/burn_scar/dataset.py` el cálculo de la huella del contenido de los ficheros que determinan el dataset y su lectura desde el índice. Verificar: test unitario que comprueba que la huella cambia al variar el contenido de uno de los ficheros enumerados y que es estable entre ejecuciones.
- [x] 3.2 Registrar `generator_version` y la lista de ficheros de origen en el `dataset_index.json`. Verificar: el índice regenerado contiene ambos campos y la lista enumera los ficheros que determinan el dataset.
- [x] 3.3 Emitir en el cargador de muestras un aviso cuando la `generator_version` registrada en el índice difiera de la vigente, distinguiendo el caso de un dataset sin versión registrada. Verificar: test que carga un índice con otra versión y comprueba el aviso, y test que carga un índice sin el campo y comprueba el aviso distinto.
- [x] 3.4 Añadir en `tests/test_dataset.py` un caso que compruebe que la lista de ficheros de origen registrada en el índice es la esperada. Verificar: el test falla si se añade o renombra un fichero que determina el dataset sin actualizar la lista.

## 4. Regeneración y verificación integral

- [x] 4.1 Regenerar el dataset por defecto en modo `scar` con el generador corregido. Verificar: el `dataset_index.json` resultante tiene `target_mode: scar` y `generator_version` presente, y el test de 2.2 pasa.
- [x] 4.2 Regenerar el dataset en modo `cems` con el generador corregido. Verificar: el `dataset_index.json` resultante tiene `target_mode: cems` y `generator_version` presente, y el test de 2.2 pasa.
- [x] 4.3 Comparar el tamaño y la composición de los dos datasets regenerados con los anteriores y registrar en `docs/operacion.md` el recuento de muestras, de descartes y el motivo. Verificar: la documentación recoge el antes y el después, y las cifras del documento coinciden con los índices regenerados.
- [x] 4.4 Ejecutar la suite completa de tests y verificar que pasa entera. Verificar: `pytest tests/ -q` termina sin fallos, incluidos los casos nuevos de 1.1, 2.1, 2.2, 3.1, 3.3 y 3.4.

## Workflow follow-up

- Regenerar, reentrenar, reevaluar y republicar las cifras **en el change `fix-propagation-window-alignment`**, para no repetir el ciclo de ensamblaje, entrenamiento y republicación. Este change debe quedar aplicado antes de esa republicación, pero sin generar por su cuenta.
- Archivar el change una vez revisados los artefactos, con lo que los deltas de `propagation-dataset` se incorporan al spec principal.
- Verificar el resultado archivado con `openspec validate --all`.
