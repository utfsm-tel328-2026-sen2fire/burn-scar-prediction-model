# Spec Delta

## MODIFIED Requirements

### Requirement: Temporal window with a pre-fire acquisition
El sistema SHALL componer la ventana temporal de cada muestra con exactamente una adquisición pre-incendio en la primera posición y hasta seis adquisiciones post-incendio que terminan en el frame actual f, usando como corte pre-incendio la hora de adquisición de la primera ronda de monitorización CEMS de la serie, y SHALL NOT descartar la muestra por carecer de adquisición pre-incendio utilizable.

#### Scenario: La secuencia contiene un frame pre-incendio
- **WHEN** se ensambla una muestra de una serie con adquirido pre-incendio disponible
- **THEN** el primer frame de la secuencia temporal es la adquisición pre-incendio más reciente y los seis restantes son post-incendios, terminando en el frame actual f

#### Scenario: La adquisición de f se empareja por tiempo y no por marca truncada
- **WHEN** la marca de tiempo de la adquisición de f en el manifiesto de origen y la del fichero de imagen descargado difieren en menos de un minuto
- **THEN** esa adquisición se reconoce como el frame f y ocupa la última posición de la ventana temporal

#### Scenario: Una muestra sin adquisición utilizable de f no se ensambla
- **WHEN** no existe ninguna adquisición utilizable que corresponda al frame actual f
- **THEN** el sistema no construye la muestra, la informa, y no la sustituye por una adquisición anterior

#### Scenario: La ventana temporal está ordenada cronológicamente
- **WHEN** se ensambla una muestra
- **THEN** los siete frames de la ventana temporal están en orden temporal no decreciente

#### Scenario: Relleno cuando no hay suficientes frames
- **WHEN** la serie dispone de menos de siete adquisiciones utilizables
- **THEN** el sistema completa la ventana repitiendo el frame más antiguo disponible y entrega igualmente siete frames

#### Scenario: Serie sin adquisición pre-incendio utilizable
- **WHEN** no existe ninguna adquisición pre-incendio utilizable antes de la primera ronda CEMS
- **THEN** el sistema ensambla la ventana con las últimas siete adquisiciones utilizables disponibles hasta el frame actual f, sin descartar la muestra, y aplica el mismo relleno de frames del caso general si no hay siete

### Requirement: Non-degenerate propagation target
En el modo de ensamblaje por defecto (`scar`), el sistema SHALL definir el target de propagación como el área quemada nueva entre el frame actual y la ronda objetivo, calculada como el máximo entre cero y la diferencia entre el polígono CEMS de la ronda objetivo y la cicatriz en el frame actual. En el modo `cems` el target de propagación SHALL ser el crecimiento de referencia. En ambos modos SHALL garantizarse una fracción no degenerada de muestras con píxeles quemados.

#### Scenario: El target de la mayoría de las muestras contiene quemado
- **WHEN** se genera el dataset completo y se inspecciona la distribución de los targets
- **THEN** al menos la mitad de las muestras contiene píxeles con valor de target igual o superior a 0,5

#### Scenario: El target es cero cuando no hay crecimiento
- **WHEN** en modo `scar` la cicatriz estimada en el frame actual cubre ya el polígono de la ronda objetivo
- **THEN** el target de esa muestra es cero en todos sus píxeles, sin valores negativos

#### Scenario: La cicatriz sustraída es la del frame actual
- **WHEN** se calcula el target de propagación de una muestra
- **THEN** la cicatriz que se resta al polígono de la ronda objetivo es la estimada en el frame actual f de esa muestra, y no la de una adquisición anterior

## ADDED Requirements

### Requirement: Complete acquisition inventory per series
El sistema SHALL obtener todas las adquisiciones candidatas de cada serie dentro del rango temporal de interés, consultando la fuente de imágenes con la misma cobertura geográfica que el catálogo de referencia y sin truncar el resultado de la consulta por un límite de resultados, y SHALL informar de cualquier adquisición descartada por esos motivos.

#### Scenario: La consulta usa la cobertura geográfica del catálogo de referencia
- **WHEN** la colección de imágenes se consulta para una serie
- **THEN** la consulta cubre la extensión del área de interés de la serie, y no un punto, de modo que toda adquisición que el catálogo de referencia lista para esa área intersecte el resultado de la consulta

#### Scenario: Ninguna serie pierde adquisiciones por un límite de resultados
- **WHEN** una serie tiene más adquisiciones candidatas que el límite de resultados de la consulta de imágenes
- **THEN** todas ellas se descargan y quedan disponibles para el ensamblaje de muestras

#### Scenario: El descarte por tope de consulta queda visible
- **WHEN** la consulta de imágenes de una serie alcanza su límite de resultados
- **THEN** el proceso informa de que el inventario de adquisiciones de esa serie puede estar incompleto
