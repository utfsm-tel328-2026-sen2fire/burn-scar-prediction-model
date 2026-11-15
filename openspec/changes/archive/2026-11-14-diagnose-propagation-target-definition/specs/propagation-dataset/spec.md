# Spec Delta

## ADDED Requirements

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
- **THEN** el sistema conserva la muestra y deja su target de referencia sin definir, en lugar de descartarla
