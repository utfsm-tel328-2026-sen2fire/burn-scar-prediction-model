# Spec Delta

## MODIFIED Requirements

### Requirement: Reporting of the optimal decision threshold
El sistema SHALL reportar, además de las métricas en el umbral operativo por defecto, el mejor F1 obtenido al recorrer un rango de umbrales y el umbral que lo alcanza.

#### Scenario: El informe distingue umbral por defecto y umbral óptimo
- **WHEN** se evalúa un fold de validación
- **THEN** el informe muestra el F1 a umbral 0,5, el mejor F1 del barrido y el umbral que lo alcanza

#### Scenario: Uso del mejor umbral para la selección de pesos
- **WHEN** el F1 de validación en el umbral óptimo del fold mejora respecto al fold anterior
- **THEN** el checkpoint seleccionado conserva las métricas de ese fold medidas en el umbral óptimo de ese fold, y el umbral operativo consolidado no interviene en esa elección

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
