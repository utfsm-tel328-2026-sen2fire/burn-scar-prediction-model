# Design: Shared Storage Experiment Tracking

## Context

El entrenamiento de modelos se efectúa colaborativamente en dos estaciones de trabajo independientes conectadas a un servidor NFS compartido (`TK_Stratego/Shared`). Las rutas locales de montaje difieren entre las máquinas de los investigadores (por ejemplo, `/home/antonio/TK_Stratego/Shared` vs `/home/otro_usuario/.../TK_Stratego/Shared`). Véase `proposal.md` para la motivación general.

## Goals / Non-Goals

**Goals:**
- Desacoplar completamente las rutas de archivo del código fuente mediante variables de entorno y archivo `.env` local (ignorado por Git).
- Proveer un módulo utilitario de configuración y tracking (`src/burn_scar/utils/tracking.py`) que abstraiga la creación de directorios, generación de namespaces por usuario y fecha, e inicialización de `SummaryWriter`.
- Implementar política de vaciado explícito (`flush()`) y permisos colaborativos de grupo (`0o775`) en sistemas de archivos en red.
- Proveer un script utilitario (`scripts/launch_tensorboard.py`) para lanzar TensorBoard apuntando al directorio compartido de forma automática.

**Non-Goals:**
- Implementar una base de datos centralizada de experimentos (MLflow Server, PostgreSQL).
- Manejar sincronización manual por red (NFS ya proporciona sincronización a nivel del sistema de archivos del kernel).
- Bloqueo transaccional de checkpoints entre usuarios; cada investigador escribe en su propio namespace aislado.

## Decisions

### Decisión 1: Configuración vía `.env` local y plantilla `.env.example`
- **Elección**: Utilizar un archivo `.env` local con precedencia de variables de entorno del sistema (`os.environ`), proveído con un cargador liviano sin dependencias extras en el módulo de tracking.
- **Razón**: Permite que Antonio configure `BURNSCAR_SHARED_STORAGE=/home/antonio/TK_Stratego/Shared` y `BURNSCAR_USER=antonio`, mientras que su compañero configura su propia ruta sin alterar archivos versionados en Git.
- **Alternativas consideradas**: Archivo `config.json` compartido (inadecuado porque generaría conflictos en Git ante cada cambio de ruta local).

### Decisión 2: Estructura de carpetas y namespaces en el almacenamiento compartido
- **Elección**: Definir una jerarquía estricta dentro del NFS:
  ```
  <BURNSCAR_SHARED_STORAGE>/burn_scar/
  ├── runs/
  │   └── <user>_<experiment_tag>_<YYYYMMDD_HHMMSS>/
  └── checkpoints/
      └── <user>_<experiment_tag>_<YYYYMMDD_HHMMSS>/
  ```
- **Razón**: Evita colisiones entre ejecuciones concurrentes de ambos desarrolladores. TensorBoard lee recursivamente todas las subcarpetas dentro de `runs/` y grafica las curvas de ambos usuarios superpuestas con leyendas automáticas.
- **Alternativas consideradas**: Una sola carpeta compartida donde ambos escriben (altísimo riesgo de sobrescritura de archivos `events.out.tfevents.*`).

### Decisión 3: Envoltorio `ExperimentTracker` sobre `SummaryWriter`
- **Elección**: Crear una clase envoltorio que administre la inicialización del directorio, la creación del `SummaryWriter` de PyTorch, el método `log_scalar` con vaciado periódico/forzado (`flush()`), y el guardado de checkpoints en la carpeta hermana `checkpoints/`.
- **Razón**: NFS emplea caché de escritura en el cliente; si un proceso no ejecuta `flush()`, los datos pueden tardar en hacerse visibles para el otro desarrollador.

### Decisión 4: Permisos de directorio colaborativos (`0o775`)
- **Elección**: Aplicar permisos de grupo de lectura/escritura (`0o775`) en la creación de carpetas mediante `os.makedirs(..., mode=0o775, exist_ok=True)`.
- **Razón**: Garantiza que si un investigador crea un experimento, el otro pueda inspeccionar sus logs o leer los checkpoints sin bloqueos de permisos a nivel de archivo Unix.

## Risks / Trade-offs

- **[Riesgo] El almacenamiento NFS no está montado al iniciar el entrenamiento** → Mitigación: El módulo verifica si la ruta existe. Si no existe o no se especificó, emite una advertencia visual clara y recurre al directorio local del proyecto (`./runs` y `./checkpoints`).
- **[Riesgo] Conexión de red inestable con el NFS durante el entrenamiento** → Mitigación: Los registros de TensorBoard son asíncronos y ligeros. Si se produce un error crítico de I/O en red, el wrapper permite capturar excepciones y notificar al operador.
