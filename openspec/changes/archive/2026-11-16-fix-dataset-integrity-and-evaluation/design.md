# Design

## Context

Véase `proposal.md` para la motivación. El estado técnico relevante es el siguiente:

- CDSE devuelve, para tres instantes, dos registros de producto con el mismo satélite y una diferencia temporal de 0–125 microsegundos, pero con identificadores de producto distintos. `sample_id` trunca el tiempo al segundo, por lo que esos registros compiten por la misma clave y el ensamblador publica la misma ruta dos veces. El índice vigente contiene 344 entradas y 341 ficheros únicos.
- El downloader de GEE nombra los rasters por timestamp UTC al segundo. Si dos productos se asignan al mismo segundo, una descarga puede sobrescribir silenciosamente la anterior.
- `download_gee_olci.py` escribe `static_layers.npz` sin validar su contenido. `EMSR785_AOI01` tiene WorldCover íntegramente a cero y, aun así, el ensamblador generó 47 muestras de test con esa entrada.
- Ninguna de las 341 muestras únicas tiene `x_scar` completamente vacío; todas contienen al menos un píxel por encima de 0,5. La documentación que declara un arranque en frío incluido no describe el dataset almacenado.
- Los horizontes son continuos y muy asimétricos: mediana 0,982 días, p75 1,897 días y solo el 14,2 % por encima de tres días.
- `BurnScarDataset` consume el índice sin comprobar unicidad. `evaluate_baselines.py` y `train_propagation.py` recorren sus entradas, de modo que los recuentos y promedios actuales pueden repetir muestras.
- Los datasets no se versionan en Git y los checkpoints tampoco. La corrección solo es reproducible si se regenera, reevalúa y republica como un conjunto coherente.

Restricciones: no se modifica la arquitectura del modelo; deben mantenerse los modos de target `scar` y `cems`; la suite debe poder ejecutarse sin los datos presentes; el test fuera de dominio debe evaluarse únicamente después de congelar el umbral sobre validación.

## Goals / Non-Goals

**Goals:**

- Definir una identidad canónica y auditable para cada adquisición y muestra.
- Impedir que datos estáticos ausentes o inválidos entren en un tensor de modelo.
- Hacer visible la cobertura real de arranque en frío y de horizontes.
- Hacer que modelo, baselines y métricas usen el mismo conjunto de muestras únicas.
- Regenerar un dataset y una CV de producción canónicos, con métricas y umbral trazables al dataset corregido.

**Non-Goals:**

- No se implementan componentes de serving, inferencia en vivo, cola, tiling ni despliegue containerizado.
- No se añaden Sentinel-1, Sentinel-2, meteorología ni nuevos horizontes al entrenamiento.
- No se fabrica una muestra de arranque en frío ni se fuerza `x_scar` a cero.
- No se calibran probabilidades ni se redefine el modelo de propagación.
- No se implementa B2 ni se cambia la función de pérdida o el protocolo de agregación por evento.

## Decisions

### D1. La identidad canónica de una adquisición combina satélite y segundo UTC

**Elegido:** la clave canónica es `(satellite, timestamp_UTC_truncado_a_segundo)`. El catálogo CDSE se deduplica por esa clave antes de construir muestras. Si hay varios productos, se elige un representante de manera determinista —ordenando por identificador de producto— y se conservan en el manifiesto los identificadores de producto alternativos colapsados.

**Por qué:** las tres colisiones observadas tienen el mismo satélite y el mismo segundo, mientras que los ficheros descargados y las claves `frame_key` también tienen granularidad de segundo. La elección de segundo refleja la unidad que realmente puede materializarse en disco.

**Alternativa descartada:** incluir microsegundos o el `scene_id` en el nombre del paquete. Exigiría cambiar el emparejamiento CDSE–GEE, el formato de `frame_key` y todos los ficheros derivados sin aportar resolución espacial al raster.

**Colisión residual:** si dos adquisiciones canónicas distintas —por ejemplo, satélites diferentes en el mismo segundo— exigen la misma ruta, el ensamblador aborta. No se sobrescribe ni se fusiona información en silencio.

### D2. La deduplicación se aplica en origen y se vuelve a comprobar en el ensamblador

**Elegido:** `fetch_cdse_olci_scenes` devuelve una adquisición por clave canónica y un resumen de productos colapsados. El downloader agrupa por timestamp, no sobrescribe silenciosamente y escribe el fichero de forma atómica. Antes de publicar el índice, el ensamblador comprueba que `sample_id` y `file` sean únicos y que el número de rutas coincida con los paquetes canónicos.

**Por qué:** corregir solo el índice ocultaría que el catálogo sigue siendo ambiguo; corregir solo el catálogo no protegería contra futuras colisiones. Las dos barreras deben ser explícitas.

**Alternativa descartada:** deduplicar silenciosamente al cargar. Haría que los recuentos fueran correctos solo en memoria y ocultaría que el artefacto almacenado sigue duplicado.

### D3. Las capas estáticas se validan antes de que exista un paquete de muestra

**Elegido:** se introduce una función de validación compartida para las capas requeridas:

- DEM: forma `(64, 64)`, valores finitos y al menos una observación no nula.
- WorldCover: forma `(64, 64)`, valores dentro del catálogo ESA WorldCover v200 admitido y al menos una clase válida; un mapa completamente nulo se considera inválido.
- El downloader escribe primero en un fichero temporal, valida y solo entonces lo renombra al destino definitivo. Un fichero existente inválido se fuerza a regenerar.
- El ensamblador valida de nuevo las capas y excluye la serie, con un motivo estructurado, si no puede usarlas.

**Por qué:** el fallo actual se produce porque la ausencia de datos se representa con el mismo valor que una clase y el ensamblador no tiene una noción de “capa requerida no disponible”.

**Alternativa descartada:** rellenar ceros o mantener una máscara de entrada especial. El modelo actual consume WorldCover como escalar y no acepta una máscara estática; introducirla exigiría cambiar la arquitectura y el entrenamiento.

**Caso `EMSR785_AOI01`:** primero se intenta una descarga limpia y validada. Si la fuente sigue devolviendo una capa inválida, la serie se excluye; no se reintroduce el fichero vacío ni se cambia su pertenencia al fold de test para ocultar el problema.

### D4. El arranque en frío se documenta, no se sintetiza

**Elegido:** durante el ensamblaje se calcula para cada muestra la fracción de píxeles de `x_scar` mayores que cero y mayores o iguales a 0,5. Una muestra es `cold_start` únicamente si `x_scar` es completamente vacío. El índice publica los recuentos y el estado de cobertura.

**Por qué:** forzar `x_scar=0` en una muestra con quemado real cambia simultáneamente la entrada y la relación que define el target; dejaría de ser una observación de arranque en frío. Hasta que exista una adquisición genuina anterior a la detección, el estado correcto es “no representado”.

**Alternativa descartada:** generar una copia con la cicatriz a cero. Produciría un ejemplo artificial y una métrica no defendible.

### D5. Los horizontes conservan su valor continuo, no tienen máximo en este change y se evalúan por bandas canónicas

**Elegido:** `horizon_days` no se redondea ni se transforma en el dataset. Se conservan todas las muestras con horizonte positivo observado, sin imponer un máximo fijo ni filtrarlas por banda. El índice publica mínimo, p25, mediana, p75, p90, máximo y los recuentos de las bandas `(0,1]`, `(1,3]`, `(3,7]`, `(7,14]` y `(14,+inf)`. La evaluación publica las métricas al umbral operativo congelado para cada banda, contra ambas etiquetas, y mantiene las métricas globales sobre todas las muestras únicas.

**Por qué:** una única media global mezcla la región con soporte denso —horizontes cortos— y una cola escasa. Las bandas hacen visible que “7 días” no tiene el mismo respaldo que “1 día” sin inventar un umbral de producto. No hace falta filtrar porque cada muestra sigue contribuyendo a aprender representaciones espaciales y temporales compartidas, y la entrada explícita del horizonte permite condicionar la predicción al retraso real del objetivo.

**Límite operativo:** este change no fija un horizonte máximo. El dominio matemático del modelo es `h > 0` y el rango observado se publica como soporte empírico. Cualquier política operativa futura de máximo, advertencia de extrapolación o rango configurable se decidirá fuera de este change, sin requerir modificar la representación continua del horizonte.

**Umbral:** sigue siendo único y global y se calcula sobre todas las muestras únicas de validación. El barrido por banda puede ser informativo pero no selecciona el umbral operativo; el umbral se congela con los folds de validación, como ya fija el spec, y se aplica sin cambios al test y a los baselines.

### D6. El índice tiene una versión de esquema y un inventario de calidad

**Elegido:** el índice añade una versión de esquema y una sección de inventario con:

- entradas, muestras únicas, ficheros únicos, series y distribución por fold;
- distribución de horizontes y bandas;
- cobertura de cicatriz y recuento de arranque en frío;
- series excluidas y motivos;
- huella SHA-256 del manifiesto de entrada y versión del generador;
- productos de catálogo colapsados, si los hubo.

El generador conserva `generator_version` como huella corta del código que determina el dataset, pero amplía sus fuentes a los scripts que ahora condicionan el manifiesto y las capas estáticas.

**Por qué:** el índice es el contrato entre ensamblaje, loader, evaluación y consumidores externos. Sus recuentos deben poder reconciliarse con disco sin inferencias.

### D7. La evaluación rechaza índices no canónicos

**Elegido:** `BurnScarDataset` valida la unicidad al cargar y emite un error accionable si encuentra identificadores o rutas repetidos. Las funciones de evaluación calculan y persisten los denominadores canónicos. Modelo y baselines reciben el mismo objeto de dataset, índice, umbral y etiquetas.

**Por qué:** la corrección no debe depender de que cada consumidor recuerde deduplicar. La validación en el límite común evita que el entrenamiento y la evaluación diverjan.

**Compatibilidad:** los paquetes antiguos con máscaras no canónicas siguen siendo tolerados por el cargador. Los índices antiguos con duplicados dejan de ser aceptables para evaluación y se marcan como históricos.

### D8. El run canónico se reentrena y sus métricas se vuelven a publicar

**Elegido:** se genera un run lógico nuevo `prop_v4_integrity` con cinco folds sobre el dataset canónico. Se fija una semilla y se registra junto al run. El umbral se selecciona exclusivamente con los folds de validación, se congela y solo entonces se evalúa el fold de test. Se reevalúan B0, B1 y el trivial sobre el mismo índice. Si se conserva la comparación diagnóstica, se genera también `prop_cems_diag_v4_integrity` sobre el dataset `cems` canónico, con el mismo protocolo y sin mezclar sus métricas con las de producción.

**Por qué:** dos de las tres muestras duplicadas estaban en folds de entrenamiento. Reutilizar los pesos antiguos dejaría una discrepancia entre “dataset publicado” y “dataset usado para entrenar”. La reparación de WorldCover también cambia el test. La reproducibilidad exige regenerar el run completo.

**Alternativa descartada:** reevaluar `prop_v4` sin reentrenar. Es útil como comparación histórica, pero no puede ser el resultado canónico del dataset corregido.

### D9. Las métricas persistidas incluyen procedencia completa

**Elegido:** `operational_threshold.json` añade `dataset_generator_version`, `dataset_index_sha256`, `dataset_schema_version`, el recuento de muestras únicas y series, las bandas de horizonte y la cobertura de arranque en frío. Los resúmenes de CV y test distinguen entradas de muestras únicas.

**Por qué:** un resumen sin la huella del dataset no puede relacionarse de forma fiable con los datos que lo originaron.

### D10. La documentación antigua queda explícitamente sustituida

**Elegido:** `README.md` y `docs/operacion.md` dejan de presentar `prop_v4` como la referencia publicable. Las cifras antiguas permanecen como histórico con una advertencia de incompatibilidad con el índice canónico. `docs/dataset.md` refleja identidad, validación y cobertura; `docs/decisiones.md` registra las decisiones de este change.

**Por qué:** el repositorio ya ha sufrido deriva entre specs, documentos y artefactos. La corrección debe cerrar esa deriva en la misma entrega.

## Risks / Trade-offs

- **Disminuyen las muestras de entrenamiento al eliminar duplicados** → se registra el antes/después por fold y se reentrena; el run antiguo queda archivado, no se mezcla.
- **La recuperación de `EMSR785_AOI01` puede fallar** → se intenta una descarga limpia; si no valida, la serie se excluye con motivo y se reportan los denominadores del test resultantes.
- **El umbral puede seguir siendo inestable** → se conserva el procedimiento y se reporta la dispersión; este cambio no pretende convertir un umbral inestable en uno calibrado.
- **Una banda de horizonte puede quedar con pocos datos** → sus métricas se muestran como no disponibles cuando no son informativas y se acompaña del recuento; no se rellenan con ceros.
- **La dedupe por segundo podría ocultar productos realmente distintos** → se conservan todos los identificadores colapsados y se usa el segundo como contrato explícito; cualquier colisión de ruta no resoluble aborta.
- **Cambiar la huella del generador invalida los datasets en disco** → es intencional: obliga a regenerar antes de entrenar o evaluar y evita combinar datos de versiones incompatibles.
- **El test out-of-domain puede cambiar al excluir una serie** → se publica el número de series y muestras antes/después y se conserva el resultado histórico por separado.

## Migration Plan

1. Registrar una instantánea auditable del dataset actual: 344 entradas, 341 ficheros, duplicados identificados, WorldCover inválido y distribución de horizontes.
2. Implementar identidad/deduplicación, validación estática, inventario de cobertura y tests unitarios/invariantes.
3. Regenerar `data/cems_manifest_46.json` y corregir las capas estáticas; reintentar `EMSR785_AOI01` y registrar su resultado.
4. Regenerar `data/processed/` y `data/processed_target_cems/`; verificar que ambos índices tienen solo muestras únicas y el mismo inventario de cobertura.
5. Ejecutar la suite completa y las auditorías de dataset antes de entrenar.
6. Reentrenar `prop_v4_integrity` con los folds 0–4, congelar el umbral desde validación y evaluar el test una sola vez; regenerar también `prop_cems_diag_v4_integrity` si se conserva la comparación diagnóstica.
7. Reevaluar B0/B1/trivial al mismo umbral y publicar los resultados por horizonte y contra ambas etiquetas.
8. Actualizar la documentación, marcar `prop_v4` como histórico/no publicable y registrar las sumas de verificación del dataset y del run canónicos.
9. Rollback: los artefactos anteriores permanecen en disco y pueden restaurarse como histórico, pero no deben combinarse con el índice nuevo ni publicarse como resultados actuales.

## Open Questions

Ninguna. La reparación o exclusión explícita de `EMSR785_AOI01`, la ausencia de síntesis de arranque en frío, las bandas de horizonte y la obligatoriedad de reentrenar son decisiones cerradas por este design.
