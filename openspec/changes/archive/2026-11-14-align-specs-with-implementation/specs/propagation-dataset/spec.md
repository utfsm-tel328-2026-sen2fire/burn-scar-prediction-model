# Spec Delta

## MODIFIED Requirements

### Requirement: Temporal window with a pre-fire acquisition
El sistema SHALL componer la ventana temporal de cada muestra con exactamente una adquisición pre-incendio en la primera posición y hasta seis adquisiciones post-incendio que terminan en el frame actual f, usando como corte pre-incendio la hora de adquisición de la primera ronda de monitorización CEMS de la serie, y SHALL NOT descartar la muestra por carecer de adquisición pre-incendio utilizable.

#### Scenario: La secuencia contiene un frame pre-incendio
- **WHEN** se ensambla una muestra de una serie con adquirido pre-incendio disponible
- **THEN** el primer frame de la secuencia temporal es la adquisición pre-incendio más reciente y los seis restantes son post-incendios, terminando en el frame actual f

#### Scenario: Relleno cuando no hay suficientes frames
- **WHEN** la serie dispone de menos de siete adquisiciones utilizables
- **THEN** el sistema completa la ventana repitiendo el frame más antiguo disponible y entrega igualmente siete frames

#### Scenario: Serie sin adquisición pre-incendio utilizable
- **WHEN** no existe ninguna adquisición pre-incendio utilizable antes de la primera ronda CEMS
- **THEN** el sistema ensambla la ventana con las últimas siete adquisiciones utilizables disponibles hasta el frame actual f, sin descartar la muestra, y aplica el mismo relleno de frames del caso general si no hay siete

### Requirement: Filtering of unusable acquisitions
El sistema SHALL descartar, antes de componer la ventana temporal, toda adquisición en la que más del 20% de los píxeles carezca de dato válido en todas sus bandas y toda adquisición sin ningún píxel válido con señal distinta de cero (adquisiciones nocturnas). El porcentaje se computa sobre los píxeles del frame, no sobre las celdas de banda.

#### Scenario: Exclusión de frames nocturnos
- **WHEN** una adquisición presenta todos sus píxeles válidos a cero
- **THEN** el sistema la descarta y no la incluye en ninguna ventana temporal

#### Scenario: Exclusión de frames mayoritariamente sin dato
- **WHEN** una adquisición supera el 20% de píxeles sin dato válido en ninguna de sus bandas
- **THEN** el sistema la descarta y no la incluye en ninguna ventana temporal

#### Scenario: Ninguna ventana contiene frames inutilizables
- **WHEN** se genera el dataset completo
- **THEN** ninguna de las ventanas temporales almacenadas contiene frames nocturnos ni frames con más del 20% de píxeles sin dato

### Requirement: Model inputs free of nodata sentinels
El sistema SHALL NOT almacenar el valor centinela de píxel sin dato en el tensor de entrada OLCI, y SHALL sustituir cada celda de banda sin dato por la media de esa misma banda dentro del frame correspondiente, conservando las bandas válidas de los píxeles parcialmente observados.

#### Scenario: El tensor almacenado no contiene el centinela
- **WHEN** se inspecciona el tensor de entrada OLCI de una muestra generada
- **THEN** ningún valor del tensor equivale al centinela de sin dato y ningún valor presenta un orden de magnitud anómalo respecto al resto de la banda

#### Scenario: Un píxel parcialmente observado conserva sus bandas válidas
- **WHEN** una banda de un píxel carece de dato y otra banda del mismo píxel sí lo tiene
- **THEN** solo la celda de banda ausente se sustituye por la media de su banda, y el valor de la banda válida se conserva sin alterar

### Requirement: Non-degenerate propagation target
En el modo de ensamblaje por defecto (`scar`), el sistema SHALL definir el target de propagación como el área quemada nueva entre el frame actual y la ronda objetivo, calculada como el máximo entre cero y la diferencia entre el polígono CEMS de la ronda objetivo y la cicatriz en el frame actual. En el modo `cems` el target de propagación SHALL ser el crecimiento de referencia. En ambos modos SHALL garantizarse una fracción no degenerada de muestras con píxeles quemados.

#### Scenario: El target de la mayoría de las muestras contiene quemado
- **WHEN** se genera el dataset completo y se inspecciona la distribución de los targets
- **THEN** al menos la mitad de las muestras contiene píxeles con valor de target igual o superior a 0,5

#### Scenario: El target es cero cuando no hay crecimiento
- **WHEN** en modo `scar` la cicatriz estimada en el frame actual cubre ya el polígono de la ronda objetivo
- **THEN** el target de esa muestra es cero en todos sus píxeles, sin valores negativos

### Requirement: Reference growth target stored alongside the propagation target
El sistema SHALL almacenar, junto al target de propagación de cada muestra, un target de referencia con el crecimiento real de quemado entre la última ronda de monitorización CEMS conocida en el instante del frame actual y la ronda objetivo, junto con la máscara de píxeles válidos común a ambas rondas, y SHALL NOT alterar por defecto el target de propagación.

#### Scenario: La muestra incluye el crecimiento de referencia
- **WHEN** se ensambla una muestra para la que existe una ronda CEMS anterior o igual al instante de su frame actual
- **THEN** el paquete de la muestra incluye el crecimiento de referencia y la validez común a las dos rondas

#### Scenario: El target de propagación por defecto no cambia
- **WHEN** se ensambla el dataset sin indicar un modo de target
- **THEN** el target de propagación es el que ya se definía, la validez es la de la ronda objetivo y el dataset resultante es equivalente al anterior

#### Scenario: El ensamblaje puede usar el crecimiento de referencia como target
- **WHEN** se ensambla el dataset en modo de crecimiento de referencia
- **THEN** el target de propagación pasa a ser el crecimiento de referencia y las entradas del modelo (secuencia temporal, elevación, cobertura del suelo, cicatriz conocida y horizonte) son idénticas a las del modo por defecto

#### Scenario: Una muestra sin ronda CEMS previa no se descarta por eso
- **WHEN** una muestra no tiene ninguna ronda CEMS anterior al instante de su frame actual
- **THEN** el sistema conserva la muestra, deja su target de referencia sin definir y marca su validez común como no evaluable, en lugar de descartarla

## ADDED Requirements

### Requirement: Canonical encoding of the validity mask
El sistema SHALL codificar la máscara de validez de cada muestra con `1` = píxel evaluable y `255` = píxel ignorado, tanto en la máscara del target de ensamblaje como en la validez común entre rondas, y SHALL NOT almacenar ningún otro valor en las máscaras.

#### Scenario: Las dos máscaras comparten la misma codificación
- **WHEN** se inspecciona un paquete de muestra generado en cualquier modo de ensamblaje
- **THEN** tanto la máscara del target como la validez común entre rondas contienen únicamente los valores `1` y `255`, y los píxeles ignorados se distinguen de los evaluables

#### Scenario: La validez común marca como ignorados los píxeles no observados
- **WHEN** se ensambla el dataset en modo de crecimiento de referencia y un píxel no fue observado en alguna de las dos rondas
- **THEN** ese píxel queda marcado como ignorado (`255`) en la validez común, y no como evaluable
