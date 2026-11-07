# execution-environment Specification

## Purpose
Proporciona un entorno de ejecución reproducible gestionado por uv con soporte desacoplado para aceleración por hardware (NVIDIA CUDA, AMD ROCm, CPU) y registro de métricas con TensorBoard.

## Requirements

### Requirement: Multi-hardware environment selection
El sistema SHALL permitir la sincronización de dependencias para aceleración por hardware diferenciando entre NVIDIA CUDA, AMD ROCm y CPU de manera mutuamente excluyente.

#### Scenario: Sincronización de entorno CUDA
- **WHEN** el usuario ejecuta la sincronización del entorno especificando el extra `cuda`
- **THEN** el sistema instala PyTorch con soporte CUDA sin incluir binarios ni librerías de ROCm

#### Scenario: Exclusión mutua entre aceleradores
- **WHEN** se intenta instalar simultáneamente los extras `cuda` y `rocm`
- **THEN** el gestor de paquetes `uv` rechaza la resolución reportando un conflicto explícito

### Requirement: Hardware acceleration verification
El sistema SHALL proveer una utilidad ejecutable que valide la disponibilidad de GPU, la correcta detección del dispositivo físico y la capacidad de ejecutar cómputo matricial en el dispositivo.

#### Scenario: Validación exitosa en arquitectura NVIDIA Blackwell
- **WHEN** se ejecuta el script de comprobación en una máquina con GPU NVIDIA RTX 5060
- **THEN** el script reporta `cuda_available=True`, identifica la tarjeta y completa una multiplicación de tensores en GPU sin error de kernel

#### Scenario: Validación en entorno sin acelerador GPU
- **WHEN** se ejecuta la comprobación en un host sin GPU compatible
- **THEN** el script advierte que no hay dispositivo CUDA/ROCm disponible e indica que el backend activo es CPU sin abortar abruptamente

### Requirement: Metric and experiment logging interface
El sistema SHALL proveer soporte para registro de métricas y artefactos de entrenamiento compatibles con TensorBoard sin importar el acelerador seleccionado.

#### Scenario: Registro de métricas escalares
- **WHEN** un script de validación o entrenamiento instancia un `SummaryWriter` y registra un valor escalar
- **THEN** se genera el archivo de log correspondiente en el directorio de eventos y puede ser leído por el servidor de TensorBoard
