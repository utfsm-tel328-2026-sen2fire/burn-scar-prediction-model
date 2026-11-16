# Tasks

> Alcance exclusivo: integridad del dataset y protocolo/métricas de evaluación. No se implementan componentes de serving, inferencia live, tiling ni nuevos canales de entrada.

## 1. Identidad canónica de adquisiciones y muestras

- [x] 1.1 Implementar en `scripts/build_cems_manifest.py` una clave canónica de adquisición `(satellite, timestamp UTC truncado al segundo)` y una deduplicación determinista de productos CDSE que conserven el producto representante y los identificadores alternativos. Verificar: test unitario con dos productos del mismo satélite y segundo y con dos productos de instantes distintos.
- [x] 1.2 Registrar en `data/cems_manifest_46.json` un resumen de productos colapsados por adquisición y de adquisiciones únicas. Verificar: el test del manifiesto comprueba que las tres colisiones conocidas se colapsan y que el número de adquisiciones únicas coincide con la suma de los recuentos por serie.
- [x] 1.3 Modificar `scripts/download_gee_olci.py` para agrupar productos GEE por timestamp de segundo, evitar sobrescrituras silenciosas y escribir cada raster de forma atómica. Verificar: test con dos imágenes del mismo segundo confirma una sola salida canónica y un test con una colisión de ruta irresoluble produce un error explícito.
- [x] 1.4 Añadir al ensamblador de `scripts/build_pytorch_dataset.py` una comprobación final de unicidad de `sample_id` y `file`, y una comprobación de correspondencia con los paquetes canónicos en disco. Verificar: test que inyecta un `sample_id` repetido y otro que inyecta una ruta repetida hacen abortar la generación antes de publicar el índice.
- [x] 1.5 Documentar la identidad canónica, la deduplicación y la política de colisión en `docs/dataset.md` y `docs/decisiones.md`. Verificar: la documentación usa la misma clave y el mismo criterio que los tests y el código.

## 2. Validación de las capas estáticas

- [x] 2.1 Implementar una función compartida de validación de DEM y WorldCover con forma, finitud, valores no nulos y catálogo de clases ESA WorldCover v200 admitido. Verificar: tests unitarios que aceptan una capa válida y rechazan WorldCover completamente cero, clases desconocidas, DEM con NaN y forma incorrecta.
- [x] 2.2 Modificar `scripts/download_gee_olci.py` para validar las capas estáticas antes de publicar `static_layers.npz`, escribir mediante fichero temporal y regenerar un fichero existente que no supere la validación. Verificar: test que preserva un fichero válido, reemplaza uno inválido y no deja un destino definitivo inválido tras un fallo.
- [x] 2.3 Modificar `scripts/build_pytorch_dataset.py` para validar las capas estáticas antes de ensamblar y registrar la exclusión de una serie con un motivo estructurado cuando no sean utilizables. Verificar: test con WorldCover vacío confirma que no se escribe ninguna muestra de esa serie y que su exclusión aparece en el inventario del índice.
- [x] 2.4 Documentar en `docs/dataset.md` y `docs/datos.md` las clases WorldCover admitidas, las condiciones de validación de DEM/WorldCover y la política de exclusión. Verificar: un test parametrizado de clases WorldCover usa la misma lista documentada.

## 3. Inventario y cobertura del índice

- [x] 3.1 Añadir al índice una versión de esquema y un inventario con entradas, muestras únicas, ficheros únicos, series, distribución por fold, huella SHA-256 del manifiesto de entrada y series excluidas con motivo. Verificar: test del esquema comprueba todos los campos y reconcilia los recuentos con el índice y el directorio de muestras.
- [x] 3.2 Registrar por muestra la cobertura de `x_scar` y su estado `cold_start`, y publicar en el índice el recuento de muestras y series de arranque en frío. Verificar: tests con cicatriz vacía y no vacía; el dataset real confirma que el estado no representado se informa sin fabricar una muestra.
- [x] 3.3 Publicar en el índice la distribución empírica de `horizon_days` y los recuentos de las bandas `(0,1]`, `(1,3]`, `(3,7]`, `(7,14]` y `(14,+inf)`, conservando todas las muestras con horizonte positivo observado sin máximo fijo ni descarte por banda. Verificar: tests de los límites de cada banda y del percentil calculado sobre un conjunto conocido, más un caso de horizonte de cola que permanece en el índice.
- [x] 3.4 Ampliar la huella `generator_version` para incluir los scripts que determinan manifiesto, descarga estática y ensamblaje, y registrar por separado la huella del manifiesto de entrada. Verificar: el test de generador espera la lista completa y falla al añadir o renombrar una fuente sin actualizar la lista.
- [x] 3.5 Hacer que `BurnScarDataset` valide la unicidad del índice al cargarlo y emita un error accionable antes de exponer muestras duplicadas. Verificar: test con IDs repetidos y test con rutas repetidas; el test de índices canónicos sigue pasando.
- [x] 3.6 Actualizar `docs/dataset.md` y `docs/decisiones.md` con la cobertura real de arranque en frío, las bandas de horizonte, el esquema del índice y la política de exclusiones. Verificar: los números descritos se generan desde el índice canónico y no se mantienen como constantes manuales divergentes.

## 4. Regeneración y auditoría del dataset canónico

- [x] 4.1 Regenerar `data/cems_manifest_46.json` con deduplicación e intentar una descarga estática limpia y validada de `EMSR785_AOI01`. Verificar: el manifiesto registra cero productos colapsados no resueltos y la capa de `EMSR785_AOI01` supera la validación o la serie aparece excluida con un motivo explícito.
- [x] 4.2 Regenerar `data/processed/` en modo `scar` con el generador canónico. Verificar: el índice no contiene IDs ni rutas repetidos, sus recuentos coinciden con los paquetes en disco y el inventario refleja horizontes, cicatriz y exclusiones.
- [x] 4.3 Regenerar `data/processed_target_cems/` en modo `cems` con el mismo generador y la misma entrada. Verificar: ambos índices comparten la misma composición de entradas y horizontes, y solo difieren en el target y sus máscaras.
- [x] 4.4 Añadir o extender una auditoría automatizada que compare índice, paquetes, folds, series, horizontes, WorldCover y estado de cicatriz, y ejecutar la auditoría sobre ambos datasets. Verificar: la auditoría falla con el dataset anterior, pasa con el canónico y deja un informe reproducible en `runs/`.
- [x] 4.5 Actualizar `docs/operacion.md` con la composición canónica antes/después, las exclusiones, la distribución de horizontes y la cobertura de arranque en frío. Verificar: cada cifra del documento se puede contrastar con los dos índices regenerados.

## 5. Evaluación sobre muestras únicas y estratos

- [x] 5.1 Actualizar `scripts/train_propagation.py` y `scripts/evaluate_baselines.py` para informar de entradas, muestras únicas, ficheros únicos, series y muestras evaluables sin colapsar denominadores distintos. Verificar: tests sintéticos con un índice duplicado y con un índice canónico comprueban los cuatro denominadores.
- [x] 5.2 Implementar el agrupamiento por banda de horizonte en las métricas de ensamblaje y referencia, con F1, IoU, precisión, recall y recuentos evaluables, manteniendo las métricas globales sobre todas las muestras únicas sin filtrar por banda ni máximo. Verificar: test con muestras sintéticas en varias bandas, test que una banda no informativa devuelve NaN, no cero, y test que los denominadores globales incluyen todas las bandas.
- [x] 5.3 Añadir al informe de evaluación el estado del estrato de arranque en frío y su advertencia cuando no haya muestras. Verificar: test con cero muestras de arranque en frío muestra “no representado” y no publica un desempeño artificial.
- [x] 5.4 Garantizar que modelo y baselines reciben el mismo índice, umbral, etiquetas y subconjunto único, y refactorizar la comparación para reutilizar un único descriptivo de conjunto evaluado. Verificar: test que ejecuta modelo y baseline sobre el mismo fixture confirma idénticos `n_samples_total` y `n_samples_evaluated`.
- [x] 5.5 Persistir en `operational_threshold.json` la versión del generador, huella SHA-256 y versión de esquema del índice, modo de target, semilla, recuento de muestras únicas/series y soporte de horizontes. Verificar: test que lee un artefacto generado comprueba todos los campos de procedencia y el origen/dispersión del umbral.
- [x] 5.6 Actualizar `docs/operacion.md` y `docs/dataset.md` con el protocolo de conteos únicos, bandas de horizonte diagnóstico, ausencia de máximo fijo en dataset/evaluación, estado de arranque en frío y procedencia obligatoria. Verificar: la documentación describe los mismos campos y bandas que implementan el código y los tests y diferencia el soporte observado de cualquier política operativa futura.

## 6. Reentrenamiento y republicación canónicos

- [x] 6.1 Añadir una semilla determinista al script de entrenamiento, aplicarla a Python, NumPy, PyTorch y DataLoader, y registrarla en el run y en `operational_threshold.json`. Verificar: dos ejecuciones de una época sobre el mismo fold y semilla producen la misma pérdida y el mismo hash de pesos.
- [x] 6.2 Entrenar `prop_v4_integrity` con cinco folds sobre `data/processed/dataset_index.json`, usando todas las muestras únicas con horizonte positivo observado sin filtrar por banda ni máximo, y seleccionando cada checkpoint solo con validación. Verificar: cada fold registra época, semilla, dataset hash, número de muestras de entrenamiento y F1 de validación; los cinco checkpoints son cargables y reproducibles.
- [x] 6.3 Congelar el umbral operativo desde los cinco folds de validación y evaluar el fold de test una sola vez con ese umbral. Verificar: `operational_threshold.json` contiene el umbral, su dispersión y métricas de test contra ambas etiquetas, sin que el test participase en la selección.
- [x] 6.4 Regenerar `prop_cems_diag_v4_integrity` sobre el dataset `cems` canónico si se conserva la comparación diagnóstica. Verificar: el run diagnóstico declara dataset hash, semilla, umbral propio y métricas no mezcladas con producción.
- [x] 6.5 Reevaluar B0, B1 y el trivial “todo quemado” al mismo umbral operativo y sobre el mismo índice canónico que el modelo. Verificar: los informes declaran los mismos denominadores y el mismo número de muestras evaluables para modelo y baselines.
- [x] 6.6 Publicar una tabla de resultados canónicos con F1, IoU, precisión, recall, métricas globales y soporte por banda de horizonte, y número de muestras/series evaluables contra ambas etiquetas; declarar las bandas como diagnóstico sin máximo fijo en este change y marcar `prop_v4` como histórico incompatible. Verificar: `docs/operacion.md` y `README.md` no presentan cifras antiguas como referencia publicable ni las bandas como filtros de entrenamiento o evaluación.

## 7. Verificación integral

- [x] 7.1 Ejecutar la suite completa de tests y las auditorías de ambos datasets. Verificar: `uv run --extra rocm pytest -q` termina sin fallos y las auditorías no encuentran duplicados, capas estáticas inválidas ni discrepancias entre índice y disco.
- [x] 7.2 Registrar las sumas de verificación de los índices, checkpoints, umbrales y datasets canónicos, y comprobar que el manifiesto del run permite reproducirlos. Verificar: un segundo intento de `sha256sum` produce los mismos valores y los hashes registrados coinciden con los artefactos.
- [x] 7.3 Validar el change y comprobar que los artefactos OpenSpec requeridos están completos. Verificar: `openspec validate fix-dataset-integrity-and-evaluation --strict` pasa sin avisos y `openspec status --change fix-dataset-integrity-and-evaluation` reporta todos los artefactos como completados.

## Workflow follow-up

- Archivar el change cuando el equipo haya revisado los nuevos datasets, checkpoints y métricas.
- Verificar el resultado archivado con `openspec validate --all`.
