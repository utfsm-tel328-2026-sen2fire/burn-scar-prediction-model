# Proposal: Setup UV Multi-GPU Environment

## Why

El proyecto necesita una base ejecutable reproducible y de alto rendimiento para el entrenamiento de modelos de predicción de propagación de incendios. El hardware local de desarrollo cuenta con una GPU NVIDIA RTX 5060 (arquitectura Blackwell / `sm_120`), que requiere binarios específicos de PyTorch con CUDA 12.8 (`cu128`), mientras que en etapas posteriores se requerirá portabilidad y ejecución en hardware AMD (ROCm) y fallback a CPU sin modificar el código fuente del proyecto.

Establecer esta configuración tempranamente con `uv` evita conflictos de dependencias entre canales de paquetes, asegura reproducibilidad mediante lockfile universal e integra desde el inicio el instrumental de monitoreo básico (TensorBoard).

## What Changes

- Creación de la configuración de empaquetado y dependencias del proyecto en `pyproject.toml` usando `uv`.
- Configuración de targets de acelerador mutuamente excluyentes (`cuda`, `rocm`, `cpu`) usando extras y `[tool.uv.conflicts]`.
- Mapeo de índices dedicados de PyTorch (`pytorch-cu128`, `pytorch-rocm`, `pytorch-cpu`) con `explicit = true` para evitar filtraciones de dependencias desde PyPI.
- Inclusión de dependencias comunes iniciales: `numpy` y `tensorboard`.
- Inclusión del grupo de desarrollo (`dev`): `pytest` y `ruff`.
- Actualización de `.gitignore` para ignorar artefactos de entorno virtual (`.venv/`), caché de `uv` y directorios de logs de TensorBoard (`runs/`).
- Creación de un script de diagnóstico y verificación (`scripts/check_env.py`) para validar detección de GPU, ejecución de kernels en Blackwell (`sm_120`) y emisión de métricas a TensorBoard.

## Capabilities

### New Capabilities
- `execution-environment`: Define los requisitos del entorno de ejecución, selección de aceleradores de hardware (NVIDIA CUDA 12.8, AMD ROCm, CPU) e integración con el sistema de métricas y registro (TensorBoard).

### Modified Capabilities
*(Ninguna; no existen especificaciones previas en el proyecto).*

## Impact

- **Archivos nuevos/modificados**: `pyproject.toml`, `.gitignore`, `uv.lock` (generado tras sync), `scripts/check_env.py`.
- **APIs y dependencias**: PyTorch (`torch`, `torchvision`), `tensorboard`, `numpy`, `pytest`, `ruff`.
- **Flujo de desarrollo**: Los comandos de ejecución y pruebas pasarán a ejecutarse a través de `uv run --extra <acelerador>`.
