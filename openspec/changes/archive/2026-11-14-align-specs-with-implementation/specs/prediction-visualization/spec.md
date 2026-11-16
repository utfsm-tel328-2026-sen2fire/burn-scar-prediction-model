# Spec Delta

## ADDED Requirements

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
