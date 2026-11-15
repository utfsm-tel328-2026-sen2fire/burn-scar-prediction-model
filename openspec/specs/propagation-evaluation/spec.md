# propagation-evaluation Specification

## Purpose

Define los requisitos de evaluación y de optimización del modelo de propagación: tratamiento de las métricas binarias cuando la verdad no contiene quemado, agregación por evento, reporte del umbral operativo óptimo y ponderación de la pérdida enmascarada por el desbalance real de clases.

## Requirements

### Requirement: Non-informative metrics when the truth contains no burn
El sistema SHALL devolver métricas no informativas (NaN) cuando la verdad de una muestra no contiene ningún píxel quemado por encima del umbral de evaluación, y SHALL NOT reportar un valor de 1,0 que premie predecir la ausencia de crecimiento.

#### Scenario: Métrica NaN sobre target sin quemado
- **WHEN** se evalúa una muestra cuyo target no tiene ningún píxel por encima del umbral y la predicción es de ausencia total de quemado
- **THEN** las métricas de esa muestra son NaN y no un F1 de 1,0

#### Scenario: La agregación ignora las muestras no informativas
- **WHEN** se agregan las métricas de varias muestras agrupadas por evento
- **THEN** el promedio por evento se calcula solo sobre las muestras con métricas informativas y el resultado informa de cuántos eventos han podido evaluarse frente al total

#### Scenario: Los baselines comparten la semántica de la métrica
- **WHEN** se evalúan los baselines de referencia con la misma implementación de métricas
- **THEN** un baseline que predice ausencia de crecimiento sobre un target vacío no obtiene una puntuación perfecta

### Requirement: Reporting of the optimal decision threshold
El sistema SHALL reportar, además de las métricas en el umbral operativo por defecto, el mejor F1 obtenido al recorrer un rango de umbrales y el umbral que lo alcanza.

#### Scenario: El informe distingue umbral por defecto y umbral óptimo
- **WHEN** se evalúa un fold de validación
- **THEN** el informe muestra el F1 a umbral 0,5, el mejor F1 del barrido y el umbral que lo alcanza

#### Scenario: Uso del mejor umbral para la selección de pesos
- **WHEN** el F1 de validación en el umbral óptimo del fold mejora respecto al fold anterior
- **THEN** el checkpoint seleccionado conserva las métricas de ese fold medidas en el umbral óptimo de ese fold, y el umbral operativo consolidado no interviene en esa elección

### Requirement: Class imbalance weighting derived from the batch
El sistema SHALL derivar el peso de la clase positiva de la pérdida enmascarada a partir de la proporción real de píxeles positivos y negativos válidos del lote, acotado a un rango configurable, y SHALL NOT depender de un valor fijo.

#### Scenario: El peso positivo se ajusta al desbalance del lote
- **WHEN** se calcula la pérdida sobre un lote con una proporción de píxeles quemados dada
- **THEN** el peso positivo aplicado es el cociente entre los píxeles válidos no quemados y los quemados de ese lote, acotado al rango configurado

#### Scenario: Sobrescritura explícita del peso
- **WHEN** el llamante proporciona un valor explícito de peso positivo
- **THEN** el sistema usa ese valor en lugar del derivado del lote

### Requirement: Frozen operational threshold and held-out evaluation
El sistema SHALL determinar un único umbral operativo a partir de los umbrales óptimos de los folds de validación, evaluarlo sobre el conjunto de test sin verlo durante la selección, y SHALL reportar tanto el umbral como su dispersión entre folds. El umbral operativo SHALL NOT intervenir en la selección de checkpoint ni de época, que se realizan con el F1 en el umbral óptimo de cada fold.

#### Scenario: El umbral operativo se consolida desde los folds de validación
- **WHEN** termina la validación cruzada con los umbrales óptimos de cada fold
- **THEN** el sistema fija un único umbral operativo como la mediana de esos umbrales e informa de su mínimo, máximo y dispersión

#### Scenario: El número publicable se mide sobre el conjunto de test
- **WHEN** se evalúa el conjunto de test con el umbral operativo ya congelado
- **THEN** el sistema reporta las métricas del conjunto de test al umbral congelado, sin que ese conjunto haya participado en la elección del umbral ni del checkpoint

#### Scenario: Un umbral inestable entre folds queda visible
- **WHEN** los umbrales óptimos de los folds difieren entre sí más allá de una diferencia de 0,1
- **THEN** el sistema advierte de que el umbral no es estable y lo refleja en el informe

#### Scenario: Persistencia del umbral para inferencia
- **WHEN** finaliza el entrenamiento
- **THEN** el sistema guarda el umbral operativo junto a los checkpoints para que la inferencia pueda aplicarlo sin recalcularlo

### Requirement: Reporting against both the assembly target and the reference target
El sistema SHALL reportar, en toda evaluación, las métricas calculadas contra el target de propagación del conjunto evaluado y las calculadas contra el target de referencia, e SHALL informar del número de muestras y de series evaluables en cada uno de los dos casos, entendiendo por evaluable una muestra cuya métrica es informativa.

#### Scenario: El informe distingue las dos etiquetas
- **WHEN** se evalúa un fold de validación o el conjunto de test
- **THEN** el informe muestra las métricas contra el target de ensamblaje y contra el target de referencia, con el recuento de muestras evaluables y de series evaluables de cada uno, y el total de muestras y de series del conjunto evaluado

#### Scenario: La distancia entre las dos etiquetas es visible
- **WHEN** el desempeño contra el target de ensamblaje y contra el target de referencia difieren de forma relevante
- **THEN** el informe señala esa diferencia de modo que el número no pueda interpretarse como desempeño en propagación

#### Scenario: Un conjunto sin target de referencia evaluable no rompe la evaluación
- **WHEN** el conjunto evaluado no dispone de target de referencia
- **THEN** la evaluación informa de las métricas del target de ensamblaje y marca el target de referencia como no disponible

### Requirement: Comparable evaluation of the reference baselines
El sistema SHALL permitir evaluar los baselines de referencia con el mismo umbral que se usa
para medir el modelo, incluido el umbral operativo congelado de un run ya entrenado, de modo
que la comparación se haga sobre la misma discretización de la verdad y el mismo conjunto de
muestras evaluables, y SHALL informar del umbral empleado y de su origen en el informe.

#### Scenario: Los baselines se miden al umbral operativo del run
- **WHEN** se solicita la evaluación de los baselines indicando el directorio de checkpoints de un run entrenado
- **THEN** los baselines se evalúan con el umbral operativo congelado de ese run, el mismo con el que se midió al modelo

#### Scenario: El informe declara el umbral usado
- **WHEN** se imprime el informe de los baselines
- **THEN** el informe indica el umbral de evaluación empleado y si procede de un valor explícito o del umbral operativo de un run

#### Scenario: La comparación con el modelo usa el mismo subconjunto de muestras
- **WHEN** los baselines se evalúan al umbral operativo del modelo
- **THEN** el número de muestras evaluables de los baselines coincide con el del modelo contra la misma etiqueta, de modo que los F1 comparados se apoyan en el mismo subconjunto de muestras

### Requirement: Meaning of the evaluation threshold for binary predictions
El sistema SHALL documentar que, cuando la predicción es binaria, el umbral de evaluación no
altera la predicción sino la discretización de la verdad, y SHALL reportar el recuento de
muestras evaluables junto a las métricas binarias.

#### Scenario: El umbral no cambia una predicción binaria
- **WHEN** se evalúan los baselines, cuyas predicciones solo toman los valores 0 y 1
- **THEN** el resultado no depende del umbral aplicado a la predicción, y el umbral solo determina qué píxeles de la verdad cuentan como quemado y qué muestras son evaluables
