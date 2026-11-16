# Spec Delta

## MODIFIED Requirements

### Requirement: Selección del acelerador dentro de la capa Nix
El sistema SHALL determinar el extra de `uv` correspondiente al acelerador de la máquina siguiendo el orden de precedencia NVIDIA, AMD y CPU, y permitir sobrescribirlo de forma explícita, rechazando valores no soportados.

#### Scenario: Detección de GPU AMD
- **WHEN** la máquina expone el device node `/dev/kfd` y no se indica ningún override
- **THEN** el shell selecciona el extra `rocm` y lo publica en la variable de entorno `BURNSCAR_ACCEL`

#### Scenario: Detección de GPU NVIDIA
- **WHEN** la máquina expone el device node `/dev/nvidiactl` y no se indica ningún override
- **THEN** el shell selecciona el extra `cuda` y lo publica en la variable de entorno `BURNSCAR_ACCEL`

#### Scenario: Host sin acelerador
- **WHEN** la máquina no expone device nodes de GPU y no se indica ningún override
- **THEN** el shell selecciona el extra `cpu` y lo publica en la variable de entorno `BURNSCAR_ACCEL`

#### Scenario: Precedencia cuando hay device nodes de varios aceleradores
- **WHEN** la máquina expone a la vez el device node de NVIDIA y el de AMD y no se indica ningún override
- **THEN** el shell selecciona el extra `cuda`, aplicando el orden de precedencia NVIDIA sobre AMD

#### Scenario: Override explícito del acelerador
- **WHEN** el usuario exporta `BURNSCAR_ACCEL` con un valor `cuda`, `rocm` o `cpu` antes de evaluar el shell
- **THEN** el shell selecciona ese extra, con independencia de los device nodes presentes en la máquina

#### Scenario: Valor de acelerador no soportado
- **WHEN** el usuario exporta `BURNSCAR_ACCEL` con un valor distinto de `cuda`, `rocm` o `cpu`
- **THEN** la evaluación del shell aborta con un mensaje que enumera los valores admitidos
