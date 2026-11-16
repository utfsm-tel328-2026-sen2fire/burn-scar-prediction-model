# Spec Delta

## ADDED Requirements

### Requirement: Evaluation uses unique canonical samples
El evaluador SHALL rechazar índices con identificadores o rutas de muestra duplicados y SHALL informar de forma separada el número de entradas, muestras únicas, ficheros únicos y series evaluadas.

#### Scenario: El índice contiene duplicados
- **WHEN** se intenta evaluar un índice con identificadores o rutas repetidos
- **THEN** el evaluador aborta antes de calcular métricas e identifica los duplicados

#### Scenario: Se evalúa un índice canónico
- **WHEN** se evalúa un índice sin duplicados
- **THEN** los denominadores de métricas y los recuentos evaluables se expresan en muestras únicas y series únicas

### Requirement: Comparable model and baseline subsets
La evaluación del modelo y la de los baselines SHALL usar exactamente el mismo conjunto de muestras únicas y las mismas reglas de validez para que sus métricas sean comparables.

#### Scenario: Una muestra no es informativa
- **WHEN** una verdad no contiene quemado por encima del umbral y la métrica es NaN
- **THEN** la muestra se excluye del promedio del modelo y del baseline y ambos informes declaran el mismo recuento de muestras evaluables

#### Scenario: Se comparan modelo y baseline
- **WHEN** se reportan F1, IoU, precisión y recall para el modelo y para un baseline
- **THEN** ambos se calculan sobre el mismo índice, umbral, etiquetas y subconjunto de muestras únicas

### Requirement: Horizon-stratified metrics
El evaluador SHALL reportar F1, IoU, precisión, recall y recuentos evaluables por banda de horizonte contra las etiquetas de ensamblaje y de referencia, y SHALL incluir todas las muestras únicas del conjunto en las métricas globales sin filtrarlas por banda ni por un máximo de horizonte.

#### Scenario: Una banda contiene muestras evaluables
- **WHEN** una banda de horizonte contiene al menos una muestra informativa
- **THEN** el informe publica sus métricas y el número de muestras y series evaluables

#### Scenario: Una banda no contiene muestras informativas
- **WHEN** una banda de horizonte no contiene ninguna muestra informativa
- **THEN** sus métricas se reportan como no disponibles y no como un valor cero

#### Scenario: Se consulta el soporte global de horizontes
- **WHEN** se imprime el resumen de evaluación
- **THEN** incluye la distribución de horizontes y los recuentos por banda para que la agregación global no oculte la cola

#### Scenario: Las métricas globales no filtran horizontes
- **WHEN** el conjunto evaluado contiene muestras de varias o de todas las bandas
- **THEN** las métricas globales se calculan sobre todas las muestras únicas evaluables y las bandas solo añaden estratos de diagnóstico

### Requirement: Coverage limitations are reported
El evaluador SHALL informar del número de muestras de arranque en frío y, cuando no exista ninguna, SHALL declarar que el desempeño en ese escenario no está medido.

#### Scenario: Hay muestras de arranque en frío
- **WHEN** el dataset contiene muestras con `x_scar` completamente vacío
- **THEN** el informe publica el número de muestras y series de ese estrato y sus métricas cuando sean informativas

#### Scenario: No hay muestras de arranque en frío
- **WHEN** el estrato de arranque en frío contiene cero muestras
- **THEN** el informe declara que el escenario no está representado y no publica un desempeño artificial para él

### Requirement: Reproducible evaluation provenance
Todo resumen de evaluación SHALL identificar la versión del generador, la huella del índice, el modo de target, el run de checkpoints, el umbral operativo y los recuentos de muestras únicas y series.

#### Scenario: Se publica un resumen de test
- **WHEN** el evaluador genera métricas de un conjunto de test
- **THEN** el artefacto persistido incluye la procedencia del dataset y del modelo y el umbral con su origen y dispersión

#### Scenario: Se revisa un resultado histórico
- **WHEN** se consulta un run anterior a la corrección del dataset
- **THEN** queda identificado como no compatible con el índice canónico y no se mezcla con las métricas regeneradas del run corregido
