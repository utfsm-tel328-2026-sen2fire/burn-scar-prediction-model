# Tasks

## 1. Configuración de Entorno Local y Plantilla

- [x] 1.1 Crear `.env.example` documentando las variables clave (`BURNSCAR_SHARED_STORAGE`, `BURNSCAR_USER`, `BURNSCAR_TENSORBOARD_PORT`) con ejemplos para cada investigador, y verificar la sintaxis del archivo.
- [x] 1.2 Actualizar `.gitignore` asegurando la exclusión de `.env` y `.env.local` para prevenir la exposición de rutas personales de montaje, verificando con `git status`.

## 2. Módulo de Configuración y Experiment Tracker

- [x] 2.1 Crear el módulo `src/burn_scar/tracking.py` implementando la resolución de rutas hacia el almacenamiento compartido con fallback automático a directorios locales, la carga de configuración y la generación de namespaces únicos por investigador y fecha.
- [x] 2.2 Implementar la clase `ExperimentTracker` en `src/burn_scar/tracking.py` envolviendo `SummaryWriter` con vaciado forzado (`flush()`), asignación de permisos de grupo (`0o775`) y helper para guardado de checkpoints.
- [x] 2.3 Crear pruebas unitarias en `tests/test_tracking.py` para validar la resolución de rutas, el mecanismo de fallback y la generación de namespaces aislados, y verificar ejecutando `uv run --extra cuda pytest tests/test_tracking.py`.

## 3. Launcher de TensorBoard y Verificación Integral

- [x] 3.1 Crear `scripts/launch_tensorboard.py` para levantar automáticamente el servidor de TensorBoard apuntando a la ruta compartida efectiva resuelta por la configuración.
- [x] 3.2 Implementar una prueba de integración en `tests/test_tracking.py` que simule la escritura concurrente de métricas de dos investigadores distintos (`antonio` y `colaborador`) y verifique que ambos registros coexisten en el directorio de eventos sin colisiones.
- [x] 3.3 Ejecutar `uv run ruff check .` y verificar que todos los archivos nuevos y modificados cumplen con las normas de estilo y calidad de código.
