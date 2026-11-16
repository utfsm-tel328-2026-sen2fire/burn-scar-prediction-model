# Spec Delta

## ADDED Requirements

### Requirement: Reporting against both the assembly target and the reference target
El sistema SHALL reportar, en toda evaluación, las métricas calculadas contra el target de propagación del conjunto evaluado y las calculadas contra el target de referencia, e SHALL informar del número de muestras y de series evaluables en cada uno de los dos casos.

#### Scenario: El informe distingue las dos etiquetas
- **WHEN** se evalúa un fold de validación o el conjunto de test
- **THEN** el informe muestra las métricas contra el target de ensamblaje y contra el target de referencia, con el recuento de muestras y series evaluables de cada uno

#### Scenario: La distancia entre las dos etiquetas es visible
- **WHEN** el desempeño contra el target de ensamblaje y contra el target de referencia difieren de forma relevante
- **THEN** el informe señala esa diferencia de modo que el número no pueda interpretarse como desempeño en propagación

#### Scenario: Un conjunto sin target de referencia evaluable no rompe la evaluación
- **WHEN** el conjunto evaluado no dispone de target de referencia
- **THEN** la evaluación informa de las métricas del target de ensamblaje y marca el target de referencia como no disponible
