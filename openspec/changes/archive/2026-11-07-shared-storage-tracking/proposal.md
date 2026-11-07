# Proposal: Shared Storage Experiment Tracking

## Why

El entrenamiento de los modelos se realiza de forma colaborativa entre dos desarrolladores utilizando un almacenamiento compartido montado por NFS (`TK_Stratego/Shared`), cuyas rutas absolutas de montaje difieren entre máquinas (por ejemplo, `/home/antonio/...` vs `/home/<compañero>/...`). Sin un mecanismo desacoplado de configuración y convenciones claras de aislamiento, existe alto riesgo de rutas absolutas hardcodeadas en el código, colisiones al sobreescribir archivos de logs o pesos, demoras en el vaciado a disco en red (buffering de NFS) y problemas de permisos cruzados de lectura/escritura.

Esta propuesta establece una arquitectura estándar basada en configuración desacoplada (`.env` / variables de entorno), nombres de espacio aislados por desarrollador, gestión segura de permisos POSIX y utilidades para inicializar `SummaryWriter` y levantar TensorBoard apuntando al almacenamiento compartido de forma transparente.

## What Changes

- Soporte para configuración del entorno local mediante variables de entorno y archivo `.env` (con plantilla `.env.example`), permitiendo a cada desarrollador definir su propia ruta al montaje NFS (`SHARED_STORAGE_DIR`) y su identificador (`EXPERIMENT_USER`).
- Módulo centralizado de configuración y tracking (`src/burn_scar/utils/tracking.py` o módulo equivalente) que:
  - Resuelve las rutas a `runs/` (TensorBoard) y `checkpoints/` (pesos) en el NFS, con fallback automático a directorios locales si el NFS no está configurado o montado.
  - Genera namespaces únicos para cada experimento (`<user>_<model_or_tag>_<timestamp>`).
  - Provee una función fábrica para `SummaryWriter` que asegura el vaciado periódico a disco en red (`flush()`) y permisos de grupo compartidos.
- Actualización de `.gitignore` para ignorar `.env` y evitar subir rutas personales al repositorio Git.
- Script utilitario CLI (`scripts/launch_tensorboard.py` o comando vía `uv run`) que detecta la ruta compartida configurada y levanta el servidor de TensorBoard permitiendo visualizar de inmediato las curvas de ambos investigadores.
- Pruebas automatizadas unitarias que validan la resolución de rutas, fallback local y formato de nombres de espacio.

## Capabilities

### New Capabilities
- `shared-experiment-tracking`: Define los requerimientos para la resolución de almacenamiento compartido de experimentos, aislamiento de namespaces por investigador, persistencia de checkpoints y visualización agregada en TensorBoard.

### Modified Capabilities
*(Ninguna; no hay capacidades previas archivadas que cambien sus requisitos de comportamiento).*

## Impact

- **Nuevos archivos**: `.env.example`, módulo de utilidades de tracking (`src/burn_scar/tracking.py` o similar), `scripts/launch_tensorboard.py`, pruebas en `tests/test_tracking.py`.
- **Archivos modificados**: `.gitignore` (para ignorar `.env`).
- **Flujo de desarrollo**: Ambos desarrolladores copian `.env.example` a `.env`, configuran su ruta particular de NFS y su nombre de usuario, y ejecutan sus entrenamientos y TensorBoard sin necesidad de modificar código entre commits.
