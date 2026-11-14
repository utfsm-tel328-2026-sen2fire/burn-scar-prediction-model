# Spec Delta

## Purpose

Define los requisitos de construcción del dataset de propagación de cicatrices: generación de la cicatriz conocida a partir del checkpoint congelado de TMB-S3, composición de la ventana temporal multi-frame con adquisición pre-incendio, filtrado de adquisiciones no utilizables, tratamiento de los píxeles sin dato y semántica del target de nuevo quemado.

## ADDED Requirements

### Requirement: Known scar generated from the burned-class output of TMB-S3
El sistema SHALL generar el canal de cicatriz conocida a partir del logit de la clase "quemado" del checkpoint TMB-S3 congelado, que corresponde al canal 0 del modelo, y SHALL NOT usar el canal 1 (cabeza de nubes, entrenada con peso nulo).

#### Scenario: La cicatriz refleja el quemado real y no una superficie uniforme
- **WHEN** se calcula la cicatriz sobre una patch con quemado real y se compara con el polígono CEMS de la serie
- **THEN** la cicatriz toma valores en [0,1] con una fracción de píxeles por encima de 0,5 acotada y solapada con el polígono CEMS, en lugar de valer 1,0 en toda la patch

#### Scenario: Coherencia cuantitativa del generador de cicatriz
- **WHEN** se evalúa la cicatriz generada sobre el conjunto de muestras frente a los polígonos CEMS
- **THEN** la IoU media del generador de cicatriz es superior a 0,3

### Requirement: Temporal window with a pre-fire acquisition
El sistema SHALL componer la ventana temporal de cada muestra con exactamente una adquisición pre-incendio en la primera posición y hasta seis adquisiciones post-incendio que terminan en el frame actual f, usando como corte pre-incendio la hora de adquisición de la primera ronda de monitorización CEMS de la serie.

#### Scenario: La secuencia contiene un frame pre-incendio
- **WHEN** se ensambla una muestra de una serie con adquirido pre-incendio disponible
- **THEN** el primer frame de la secuencia temporal es la adquisición pre-incendio más reciente y los seis restantes son post-incendios, terminando en el frame actual f

#### Scenario: Relleno cuando no hay suficientes frames
- **WHEN** la serie dispone de menos de siete adquisiciones utilizables
- **THEN** el sistema completa la ventana repitiendo el frame más antiguo disponible y entrega igualmente siete frames

#### Scenario: Serie sin adquisición pre-incendio utilizable
- **WHEN** no existe ninguna adquisición pre-incendio utilizable antes de la primera ronda CEMS
- **THEN** el sistema ensambla la ventana con las seis adquisiciones post-incendio más recientes disponibles sin descartar la muestra

### Requirement: Filtering of unusable acquisitions
El sistema SHALL descartar, antes de componer la ventana temporal, toda adquisición con más del 20% de píxeles sin dato y toda adquisición sin ningún píxel válido con señal distinta de cero (adquisiciones nocturnas).

#### Scenario: Exclusión de frames nocturnos
- **WHEN** una adquisición presenta todos sus píxeles válidos a cero
- **THEN** el sistema la descarta y no la incluye en ninguna ventana temporal

#### Scenario: Exclusión de frames mayoritariamente sin dato
- **WHEN** una adquisición supera el 20% de píxeles sin dato
- **THEN** el sistema la descarta y no la incluye en ninguna ventana temporal

#### Scenario: Ninguna ventana contiene frames inutilizables
- **WHEN** se genera el dataset completo
- **THEN** ninguna de las ventanas temporales almacenadas contiene frames nocturnos ni frames con más del 20% de píxeles sin dato

### Requirement: Model inputs free of nodata sentinels
El sistema SHALL NOT almacenar el valor centinela de píxel sin dato en el tensor de entrada OLCI, y SHALL sustituir cada píxel sin dato por la media de su banda dentro del frame correspondiente.

#### Scenario: El tensor almacenado no contiene el centinela
- **WHEN** se inspecciona el tensor de entrada OLCI de una muestra generada
- **THEN** ningún valor del tensor equivale al centinela de sin dato y ningún valor presenta un orden de magnitud anómalo respecto al resto de la banda

### Requirement: Non-degenerate propagation target
El sistema SHALL definir el target de propagación como el área quemada nueva entre el frame actual y la ronda objetivo, calculada como el máximo entre cero y la diferencia entre el polígono CEMS de la ronda objetivo y la cicatriz en el frame actual, y SHALL garantizar que el dataset generado contenga una fracción no degenerada de muestras con píxeles quemados.

#### Scenario: El target de la mayoría de las muestras contiene quemado
- **WHEN** se genera el dataset completo y se inspecciona la distribución de los targets
- **THEN** al menos la mitad de las muestras contiene píxeles con valor de target igual o superior a 0,5

#### Scenario: El target es cero cuando no hay crecimiento
- **WHEN** la cicatriz estimada en el frame actual cubre ya el polígono de la ronda objetivo
- **THEN** el target de esa muestra es cero en todos sus píxeles, sin valores negativos
