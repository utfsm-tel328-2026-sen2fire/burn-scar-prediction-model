# Spec Delta

## ADDED Requirements

### Requirement: Canonical acquisition and sample identity
El sistema SHALL identificar una adquisición OLCI por satélite y marca temporal UTC canónica, SHALL colapsar productos de catálogo que representen la misma adquisición y SHALL garantizar que cada `sample_id` y cada ruta de fichero aparezca una sola vez en el índice.

#### Scenario: Productos de catálogo duplicados representan una adquisición
- **WHEN** dos o más productos del catálogo tienen el mismo satélite y la misma marca temporal canónica
- **THEN** el sistema conserva una sola adquisición, selecciona un representante determinista y registra los identificadores de producto colapsados para auditoría

#### Scenario: Una colisión no puede resolverse canonicamente
- **WHEN** dos adquisiciones distintas requieren la misma clave de frame o la misma ruta de paquete
- **THEN** el ensamblador aborta antes de publicar el índice y informa de la colisión

#### Scenario: El índice mantiene una correspondencia uno a uno
- **WHEN** se genera el índice final
- **THEN** no contiene identificadores de muestra ni rutas de fichero repetidos y su recuento coincide con los paquetes canónicos almacenados

### Requirement: Validation of required static layers
El sistema SHALL validar DEM y WorldCover antes de ensamblar muestras y SHALL NOT crear muestras para una serie cuya capa estática requerida esté ausente, vacía, contenga valores no finitos o use clases WorldCover no admitidas.

#### Scenario: WorldCover está completamente vacío
- **WHEN** una serie tiene WorldCover con cero en todos sus píxeles
- **THEN** el sistema excluye la serie, registra el motivo de exclusión y no genera muestras para ella

#### Scenario: Una capa estática inválida ya existente no se reutiliza
- **WHEN** se encuentra un fichero estático almacenado que no supera la validación
- **THEN** el sistema no lo reutiliza, intenta obtener una copia válida y, si no puede, excluye la serie de forma explícita

#### Scenario: Las capas estáticas son válidas
- **WHEN** DEM y WorldCover tienen la forma esperada y valores admitidos
- **THEN** el sistema puede ensamblar las muestras de la serie usando esas capas

### Requirement: Explicit cold-start coverage
El sistema SHALL registrar el estado de cobertura de `x_scar` de cada muestra y SHALL publicar el número de muestras con cicatriz genuinamente vacía; si no existe ninguna, SHALL declarar el arranque en frío como no representado y SHALL NOT fabricarlo alterando `x_scar` o el target.

#### Scenario: El dataset contiene una muestra de cicatriz vacía
- **WHEN** una muestra almacenada tiene `x_scar` igual a cero en todos sus píxeles
- **THEN** el índice la clasifica como muestra de arranque en frío y la incluye en el recuento de cobertura correspondiente

#### Scenario: El dataset no contiene arranque en frío
- **WHEN** ninguna muestra almacenada tiene `x_scar` completamente vacío
- **THEN** el inventario declara que el arranque en frío no está representado y no genera una copia con la cicatriz forzada a cero

### Requirement: Horizon support is published
El índice SHALL publicar la distribución empírica de `horizon_days`, SHALL contar las muestras en las bandas `(0,1]`, `(1,3]`, `(3,7]`, `(7,14]` y `(14,+inf)` días y SHALL conservar todas las muestras con horizonte positivo observado sin imponer un máximo fijo ni descartarlas por su banda.

#### Scenario: El inventario resume el soporte temporal
- **WHEN** se genera el índice final
- **THEN** registra mínimo, percentiles 25, 50, 75 y 90, máximo y el recuento de muestras de cada banda de horizonte

#### Scenario: Un horizonte continuo cae en una banda inequívoca
- **WHEN** una muestra tiene un valor continuo de `horizon_days`
- **THEN** la muestra se cuenta en exactamente una banda y su valor original se conserva sin redondear ni agrupar

#### Scenario: Una muestra pertenece a una banda de cola
- **WHEN** una muestra tiene un horizonte positivo observado fuera de las bandas con mayor soporte
- **THEN** el índice la conserva, registra su banda y su valor continuo y no la excluye por ese motivo

### Requirement: Traceable dataset coverage inventory
El índice SHALL registrar el número de entradas, muestras únicas, ficheros únicos, series y distribución por fold, junto con las series excluidas y sus motivos.

#### Scenario: Una serie se excluye durante el ensamblaje
- **WHEN** el ensamblador excluye una serie por una validación o una falta de datos
- **THEN** el índice incluye su identificador y el motivo, y los recuentos publicados corresponden solo a las muestras incluidas

#### Scenario: El inventario permite reconciliar el dataset
- **WHEN** se compara el índice con el directorio de muestras
- **THEN** el número de muestras únicas del índice coincide con el número de paquetes canónicos y las exclusiones justifican la diferencia respecto al censo de entrada
