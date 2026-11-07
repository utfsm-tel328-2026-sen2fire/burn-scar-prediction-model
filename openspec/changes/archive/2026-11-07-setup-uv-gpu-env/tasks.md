# Tasks

## 1. Configuración de UV y Gestión de Dependencias

- [x] 1.1 Crear `pyproject.toml` configurando las dependencias base (`numpy`, `tensorboard`), grupo dev (`ruff`, `pytest`), extras (`cuda`, `rocm`, `cpu`), índices dedicados con `explicit = true` y matriz de conflictos en `[tool.uv.conflicts]`, verificando que el archivo es parseable por uv.
- [x] 1.2 Actualizar `.gitignore` agregando `.venv/`, `.pytest_cache/`, `.ruff_cache/` y directorios de logs de TensorBoard (`runs/`), verificando que no queden archivos de entorno pendientes en `git status`.

## 2. Sincronización del Entorno y Verificación de Conflictos

- [x] 2.1 Ejecutar `uv sync --extra cuda --group dev` para generar `uv.lock` e instalar PyTorch con soporte CUDA 12.8, verificando que la instalación culmina exitosamente sin advertencias de dependencias faltantes.
- [x] 2.2 Comprobar la regla de exclusión mutua ejecutando `uv sync --extra cuda --extra rocm` y verificando que uv aborta con un mensaje de conflicto entre extras.

## 3. Script de Diagnóstico e Instrumental de Métricas

- [x] 3.1 Crear `scripts/check_env.py` que detecte el acelerador activo, verifique la ejecución de cómputo matricial en GPU Blackwell (`sm_120`) y escriba un registro escalar de prueba mediante `SummaryWriter` de TensorBoard.
- [x] 3.2 Ejecutar `uv run --extra cuda python scripts/check_env.py` y verificar que detecta la GPU NVIDIA RTX 5060, ejecuta el tensor en GPU y crea el archivo de evento en el directorio `runs/`.

## 4. Pruebas Automatizadas y Calidad de Código

- [x] 4.1 Crear `tests/test_env.py` con tests unitarios que verifiquen la importabilidad de `torch`, `tensorboard` y `numpy`, y ejecutar `uv run --extra cuda pytest` verificando que todas las pruebas pasen.
- [x] 4.2 Ejecutar `uv run ruff check .` para verificar que los scripts y tests cumplen las directivas de formato y calidad de código.
