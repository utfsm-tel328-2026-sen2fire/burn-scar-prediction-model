# Spec Delta

## ADDED Requirements

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
