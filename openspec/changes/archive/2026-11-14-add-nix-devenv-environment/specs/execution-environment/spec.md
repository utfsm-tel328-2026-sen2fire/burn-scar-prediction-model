# Spec Delta

## MODIFIED Requirements

### Requirement: Hardware acceleration verification
El sistema SHALL proveer una utilidad ejecutable que valide la disponibilidad de GPU, la correcta detección del dispositivo físico y la capacidad de ejecutar cómputo matricial en el dispositivo.

#### Scenario: Validación exitosa en arquitectura NVIDIA Blackwell
- **WHEN** se ejecuta el script de comprobación en una máquina con GPU NVIDIA RTX 5060
- **THEN** el script reporta `cuda_available=True`, identifica la tarjeta y completa una multiplicación de tensores en GPU sin error de kernel

#### Scenario: Validación exitosa en arquitectura AMD RDNA2
- **WHEN** se ejecuta el script de comprobación en una máquina con GPU AMD Radeon RX 6800
- **THEN** el script reporta el backend `rocm`, identifica la tarjeta y completa una multiplicación de tensores en GPU sin error de kernel

#### Scenario: Validación en entorno sin acelerador GPU
- **WHEN** se ejecuta la comprobación en un host sin GPU compatible
- **THEN** el script advierte que no hay dispositivo CUDA/ROCm disponible e indica que el backend activo es CPU sin abortar abruptamente
