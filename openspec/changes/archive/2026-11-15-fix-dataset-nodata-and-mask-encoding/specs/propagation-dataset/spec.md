# Spec Delta

## MODIFIED Requirements

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

#### Scenario: El umbral se mide sobre píxeles y no sobre celdas de banda
- **WHEN** una adquisición tiene píxeles parcialmente observados, con unas bandas válidas y otras sin dato
- **THEN** esos píxeles cuentan como observados, porque el umbral del 20% se mide sobre los píxeles que no tienen ninguna banda con dato válido

#### Scenario: Un frame con señal y muchos píxeles muertos se descarta
- **WHEN** una adquisición supera el 20% de sus píxeles sin dato válido en ninguna banda, aunque conserve señal distinta de cero
- **THEN** el sistema la descarta por ese criterio, con independencia de que tenga señal

### Requirement: Canonical encoding of the validity mask
El sistema SHALL codificar la máscara de validez de cada muestra con `1` = píxel evaluable y `255` = píxel ignorado, tanto en la máscara del target de ensamblaje como en la validez común entre rondas, y SHALL NOT almacenar ningún otro valor en las máscaras del paquete almacenado. El cargador de muestras SHALL tolerar paquetes antiguos que empleen otros valores, normalizándolos a la codificación canónica al leerlos.

#### Scenario: Las dos máscaras comparten la misma codificación
- **WHEN** se inspecciona un paquete de muestra generado en cualquier modo de ensamblaje
- **THEN** tanto la máscara del target como la validez común entre rondas contienen únicamente los valores `1` y `255`, y los píxeles ignorados se distinguen de los evaluables

#### Scenario: La validez común marca como ignorados los píxeles no observados
- **WHEN** se ensambla el dataset en modo de crecimiento de referencia y un píxel no fue observado en alguna de las dos rondas
- **THEN** ese píxel queda marcado como ignorado (`255`) en la validez común, y no como evaluable

#### Scenario: El paquete almacenado cumple la codificación sin pasar por el cargador
- **WHEN** se inspecciona el fichero de una muestra almacenada, sin pasar por el cargador de muestras
- **THEN** sus dos máscaras contienen únicamente los valores `1` y `255`, sin ningún valor heredado de codificaciones anteriores

#### Scenario: El cargador normaliza los paquetes antiguos
- **WHEN** se lee un paquete almacenado con un valor distinto de `1` en lugar de `255`
- **THEN** el cargador lo entrega ya normalizado a `255`, de modo que ningún consumidor de la muestra recibe un valor no canónico

## ADDED Requirements

### Requirement: Traceable dataset version
El sistema SHALL registrar en el índice del dataset la versión del generador con la que se construyó el conjunto de muestras, junto al modo de target y la marca de tiempo, de modo que se pueda detectar que un dataset almacenado fue producido por un generador distinto del vigente.

#### Scenario: El índice identifica la versión del generador
- **WHEN** se genera el dataset
- **THEN** el índice registra la versión del generador junto a la marca de tiempo y al modo de target

#### Scenario: Un dataset desactualizado es detectable
- **WHEN** se compara la versión del generador registrada en el índice de un dataset almacenado con la versión vigente
- **THEN** la diferencia queda a la vista antes de usar ese dataset para entrenar o evaluar
