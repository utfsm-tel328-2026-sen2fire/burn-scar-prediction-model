# Spec Delta

## Purpose

Define los requisitos funcionales para la resolución de almacenamiento compartido (NFS/local), el aislamiento de experimentos por investigador, el registro seguro de eventos en red y la visualización centralizada con TensorBoard.

## ADDED Requirements

### Requirement: Configurable storage path resolution
El sistema SHALL resolver la ruta raíz de almacenamiento de experimentos (`runs/` y `checkpoints/`) desde variables de entorno o archivo `.env`, aplicando un fallback a directorios locales del proyecto cuando no esté configurada.

#### Scenario: Ruta compartida personalizada definida en el entorno
- **WHEN** el usuario define `BURNSCAR_SHARED_STORAGE=/ruta/compartida/TK_Stratego/Shared` en `.env` o en el entorno
- **THEN** el sistema resuelve los directorios de logs y checkpoints dentro de esa ruta compartida sin requerir modificaciones en el código fuente

#### Scenario: Fallback a almacenamiento local cuando no hay configuración
- **WHEN** no se define ninguna variable de entorno ni archivo `.env`
- **THEN** el sistema crea y utiliza carpetas locales `./runs` y `./checkpoints` dentro del repositorio sin arrojar error

### Requirement: Isolated experiment namespaces
El sistema SHALL generar nombres de espacio y directorios únicos para cada ejecución de entrenamiento estructurados con el identificador del investigador, el nombre del modelo o tag, y una marca temporal ISO/fecha.

#### Scenario: Creación de nombre de ejecución único
- **WHEN** un investigador con identificador `antonio` inicializa un experimento con tag `baseline_unet`
- **THEN** el sistema genera una ruta de log con formato `<base_runs>/antonio_baseline_unet_<timestamp>`

#### Scenario: Coexistencia de experimentos de múltiples investigadores
- **WHEN** dos investigadores ejecutan experimentos concurrentes sobre el mismo almacenamiento compartido
- **THEN** cada experimento escribe en un subdirectorio aislado evitando sobrescritura o bloqueo de archivos

### Requirement: Shared logging and checkpoint management
El sistema SHALL proveer una fábrica o envoltorio de logging que fuerce el vaciado de búfer (`flush`) a disco y garantice permisos de lectura y escritura para el grupo colaborativo.

#### Scenario: Vaciado inmediato de eventos de TensorBoard
- **WHEN** se invoca el método de registro de métricas tras un paso o época
- **THEN** el sistema ejecuta `flush()` para sincronizar los eventos de TensorBoard de inmediato en el sistema de archivos compartido

### Requirement: Unified TensorBoard launcher
El sistema SHALL proveer una utilidad de línea de comandos que resuelva automáticamente el directorio de logs configurado e inicie el servidor de TensorBoard.

#### Scenario: Lanzamiento sobre almacenamiento compartido
- **WHEN** el usuario ejecuta el comando o script de lanzamiento de TensorBoard
- **THEN** el servidor inicia apuntando a la ruta compartida efectiva permitiendo inspeccionar todas las curvas de ambos investigadores
