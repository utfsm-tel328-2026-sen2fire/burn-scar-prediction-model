# prediction-visualization Specification

## Purpose

Define la visualización post-entrenamiento de checkpoints del modelo de propagación en TensorBoard (pestaña IMAGES) y la exportación opcional de paneles comparativos a PNG.

## Requirements

### Requirement: Checkpoint-scoped validation inference
El sistema SHALL cargar un checkpoint entrenado (`best_fold_*.pt`) y ejecutar inferencia en modo evaluación únicamente sobre el split de validación correspondiente al `fold` registrado en el checkpoint.

#### Scenario: Inferencia automática sobre el fold del checkpoint
- **WHEN** el usuario proporciona la ruta a un checkpoint que contiene `fold` y `model_state_dict`
- **THEN** el sistema instancia el modelo en modo evaluación, carga los pesos y evalúa únicamente las muestras del split de validación de ese fold

#### Scenario: Fold explícito o parámetro de arquitectura incompatible
- **WHEN** el checkpoint no coincide con el `--hidden-ch` configurado o carece de `fold` sin que el usuario indique `--fold`
- **THEN** el sistema informa el error de forma explícita antes de registrar imágenes

### Requirement: Flexible sample selection modes
El sistema SHALL permitir seleccionar las muestras a visualizar mediante modos configurables: mejores (`best`), peores (`worst`), por evento (`series`), por índices explícitos (`indices`) o muestreo aleatorio (`random`).

#### Scenario: Inspección de mejores y peores predicciones
- **WHEN** el usuario selecciona el modo `best` o `worst` indicando una cantidad `N`
- **THEN** el sistema ordena las muestras evaluadas según su F1 por muestra y selecciona las `N` mejores o las `N` peores

#### Scenario: Selección a medida por evento o índices
- **WHEN** el usuario especifica un identificador de serie (`--series-id`) o índices concretos (`--indices`)
- **THEN** el sistema filtra y visualiza únicamente las muestras correspondientes a esa selección

### Requirement: Multi-panel visual comparison with scar and mask awareness
El sistema SHALL componer un panel RGB horizontal por muestra que incluya contexto OLCI (pseudo-RGB), cicatriz previa (`x_scar`), verdad terreno (`target`), mapa de probabilidad predicha y overlay de aciertos/errores (`TP`, `FP`, `FN`) diferenciando píxeles ignorados (`mask == 255`).

#### Scenario: Composición de panel completo para una muestra
- **WHEN** se renderiza una muestra con tensores `x_olci`, `x_scar`, `target`, `mask` y probabilidades predichas
- **THEN** el sistema genera un tensor imagen RGB `(3, H, W_total)` con valores normalizados en `[0, 1]` donde las zonas con `mask == 255` quedan visualmente atenuadas

### Requirement: TensorBoard IMAGES logging and PNG export
El sistema SHALL registrar los paneles generados en TensorBoard para su visualización y descarga desde la pestaña IMAGES, y SHALL permitir exportarlos opcionalmente como archivos PNG en un directorio local.

#### Scenario: Registro en TensorBoard con vaciado inmediato
- **WHEN** se ejecuta la visualización sobre un checkpoint
- **THEN** los paneles quedan registrados mediante `ExperimentTracker` con `flush()` inmediato para aparecer en la pestaña IMAGES de TensorBoard

#### Scenario: Exportación directa a archivos PNG
- **WHEN** el usuario indica un directorio de salida mediante `--save-dir`
- **THEN** el sistema guarda cada panel como archivo `.png` nombrado con su ranking/índice, `series_id` y métrica F1 además de registrarlo en TensorBoard

### Requirement: Treatment of non-informative metrics in sample selection
El sistema SHALL excluir de la ordenación por F1 y del promedio de métricas del fold las muestras cuya métrica no es informativa, y SHALL informar del número de muestras informativas sobre el total evaluado, en lugar de sustituirlas por un valor numérico.

#### Scenario: El ranking ignora las muestras no informativas
- **WHEN** el fold de validación contiene muestras cuyo target no tiene quemado y cuya métrica no es informativa
- **THEN** la selección de mejores y peores y el promedio de F1 del fold se calculan solo sobre las muestras informativas, y el informe indica cuántas se han excluido

#### Scenario: El promedio no se anula por una sola muestra vacía
- **WHEN** al menos una muestra del fold tiene quemado en la verdad
- **THEN** el promedio de F1 publicado para el fold es un valor numérico, y no una ausencia de dato

#### Scenario: La detección de muestras con quemado usa el umbral de evaluación
- **WHEN** se evalúan las muestras del fold con un umbral de decisión distinto del de por defecto
- **THEN** la condición de muestra con quemado se evalúa con ese mismo umbral, de modo que la priorización de la selección sea coherente con las métricas mostradas
