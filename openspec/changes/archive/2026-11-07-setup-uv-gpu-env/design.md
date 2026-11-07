# Design: Setup UV Multi-GPU Environment

## Context

El repositorio carece actualmente de configuración de entorno Python (`pyproject.toml` ausente). El equipo local dispone de una GPU NVIDIA RTX 5060 (Blackwell, `sm_120`), lo que impone la necesidad de utilizar canales CUDA 12.8 para PyTorch. En etapas posteriores, el código debe poder ejecutarse en máquinas con GPUs AMD (ROCm 6.x) o entornos CI sólo CPU sin divergencia de código fuente. Véase `proposal.md` para la motivación general.

## Goals / Non-Goals

**Goals:**
- Configurar un `pyproject.toml` estándar estructurado con `uv` para gestión declarativa de dependencias y lockfile determinista (`uv.lock`).
- Implementar extras mutuamente excluyentes (`cuda`, `rocm`, `cpu`) que apunten a los repositorios de wheels oficiales de PyTorch correspondientes.
- Proveer logging con TensorBoard disponible de forma agnóstica al hardware.
- Proporcionar un script de verificación técnica (`scripts/check_env.py`) que compruebe la operatividad del hardware y de TensorBoard.

**Non-Goals:**
- Instalar o configurar librerías geoespaciales o satelitales (`rasterio`, `geopandas`, `xarray`, APIs de GEE) en esta etapa.
- Configurar pipelines de compilación C++/CUDA locales (`torch.utils.cpp_extension`) o extensiones nativas personalizadas.
- Automatizar la detección dinámica de GPU en tiempo de instalación; la selección del backend se realiza explícitamente mediante el flag `--extra <backend>`.

## Decisions

### Decisión 1: Canales explícitos (`explicit = true`) en `tool.uv.index`
- **Elección**: Definir los índices `pytorch-cu128`, `pytorch-rocm` y `pytorch-cpu` con la directiva `explicit = true`.
- **Razón**: Por defecto, agregar un índice secundario en gestores de paquetes de Python puede provocar fugas de resolución donde paquetes estándar de PyPI se descarguen desde el índice secundario o viceversa. `explicit = true` obliga a que solo los paquetes explícitamente fijados en `[tool.uv.sources]` consulten dichos índices.
- **Alternativas consideradas**: Usar `--extra-index-url` en pip/requirements.txt (inseguro y susceptible a *dependency confusion*).

### Decisión 2: Exclusión mutua con `tool.uv.conflicts`
- **Elección**: Declarar `conflicts = [[{ extra = "cpu" }, { extra = "cuda" }, { extra = "rocm" }]]`.
- **Razón**: Las distribuciones de PyTorch para CUDA, ROCm y CPU comparten el mismo nombre de paquete (`torch`), pero difieren radicalmente en tamaño, binarios compartidos y librerías dinámicas (`libcudart` vs `librocblas`). Intentar resolverlas juntas corrompería el entorno. `uv` previene formalmente esta colisión y genera un lockfile universal consistente.
- **Alternativas consideradas**: Mantener múltiples ramas de git o archivos `requirements-cuda.txt` / `requirements-rocm.txt` separados (propenso a desincronización de dependencias base).

### Decisión 3: Versión de CUDA para Blackwell (RTX 5060)
- **Elección**: Apuntar al índice `https://download.pytorch.org/whl/cu128`.
- **Razón**: La arquitectura Blackwell (`sm_120`) de la RTX 5060 no está soportada por los kernels precompilados de CUDA 12.1/12.4 de PyTorch, causando errores `CUDA error: no kernel image is available`. CUDA 12.8 provee los binarios compatibles con `sm_120`.
- **Alternativas consideradas**: Compilar PyTorch desde código fuente con soporte `sm_120` (costoso en tiempo y complejidad) o usar `cu124` confiando en JIT PTX (inestable y con overhead alto al inicio).

## Risks / Trade-offs

- **[Riesgo] Tamaño de descarga de PyTorch CUDA 12.8 (> 2 GB)** → Mitigación: `uv` cuenta con caché global por hash de wheel; la descarga se realiza una única vez por máquina y la creación del virtualenv toma segundos.
- **[Riesgo] Diferencia de versiones menores entre CUDA y ROCm en los índices de PyTorch** → Mitigación: Se fijan rangos compatibles (`torch>=2.7.0`) en lugar de fijar un hash idéntico que podría no existir simultáneamente en el canal ROCm al mismo tiempo.
- **[Riesgo] Omisión accidental del flag de acelerador al ejecutar** → Mitigación: Documentar claramente el uso de `uv run --extra cuda` y configurar el script de diagnóstico para alertar al usuario si se ejecutó bajo CPU teniendo GPU presente.
