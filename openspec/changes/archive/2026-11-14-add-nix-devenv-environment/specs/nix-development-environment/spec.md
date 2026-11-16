# Spec Delta

## Purpose

Proporciona una capa opcional de provisión del entorno de desarrollo sobre Nix/devenv para
trabajar el proyecto en NixOS, con los inputs fijados, las librerías y cabeceras que
exigen los wheels de PyTorch, la selección del acelerador dentro de la capa nix y el control
explícito de la sincronización, sin alterar el flujo `uv` nativo del resto del equipo.

## ADDED Requirements

### Requirement: Shell de desarrollo Nix reproducible
El sistema SHALL proveer un shell de desarrollo Nix fijando sus inputs y exponiendo un
intérprete de Python 3.12 y el gestor `uv` utilizables sin intervención manual.

#### Scenario: Entrada al shell en un host NixOS
- **WHEN** el usuario ejecuta `devenv shell` en una máquina con Nix y flakes habilitados
- **THEN** el shell se construye con los inputs fijados en el lockfile y expone un
  intérprete de Python 3.12 y el ejecutable `uv` en el `PATH`

#### Scenario: Construcción del intérprete sin depender de versiones de parche externas
- **WHEN** se reconstruye el shell con los inputs fijados
- **THEN** la construcción del intérprete Python se completa sin errores de aplicación de
  parches sobre el tarball de CPython

#### Scenario: Entorno virtual aislado del proyecto
- **WHEN** el usuario ejecuta `uv sync` dentro del shell de devenv
- **THEN** el entorno virtual se crea en el directorio de estado de devenv y no en el
  directorio `.venv` de la raíz del proyecto

### Requirement: Aprovisionamiento de librerías y cabeceras del sistema
El sistema SHALL exponer en el shell las librerías de sistema y las rutas de cabeceras que
requieren los wheels de PyTorch para inicializarse y para compilar kernels en tiempo de
ejecución.

#### Scenario: Resolución de las dependencias dinámicas del wheel de PyTorch
- **WHEN** el usuario importa `torch` dentro del shell de devenv
- **THEN** las librerías de la familia de C++ y de OpenMP, la de numeración de memoria y las
  de compresión se resuelven sin error de enlazador

#### Scenario: Compilación JIT de kernels de convolución
- **WHEN** el runtime de MIOpen compila un kernel de convolución en tiempo de ejecución
- **THEN** el compilador embebido localiza las cabeceras de la librería estándar de C++ y
  las de libc, y la compilación termina sin error de cabecera no encontrada

#### Scenario: Ausencia de toolchain de ROCm gestionado por Nix
- **WHEN** el usuario inspecciona el shell de devenv
- **THEN** el runtime de ROCm procede del propio wheel de PyTorch y no de un paquete de ROCm
  instalado desde nixpkgs

### Requirement: Selección del acelerador dentro de la capa Nix
El sistema SHALL determinar el extra de `uv` correspondiente al acelerador de la máquina y
permitir sobrescribirlo de forma explícita, rechazando valores no soportados.

#### Scenario: Detección de GPU AMD
- **WHEN** la máquina expone el device node `/dev/kfd` y no se indica ningún override
- **THEN** el shell selecciona el extra `rocm` y lo publica en la variable de entorno
  `BURNSCAR_ACCEL`

#### Scenario: Detección de GPU NVIDIA
- **WHEN** la máquina expone el device node `/dev/nvidiactl` y no se indica ningún override
- **THEN** el shell selecciona el extra `cuda` y lo publica en la variable de entorno
  `BURNSCAR_ACCEL`

#### Scenario: Host sin acelerador
- **WHEN** la máquina no expone device nodes de GPU y no se indica ningún override
- **THEN** el shell selecciona el extra `cpu` y lo publica en la variable de entorno
  `BURNSCAR_ACCEL`

#### Scenario: Override explícito del acelerador
- **WHEN** el usuario exporta `BURNSCAR_ACCEL` con un valor `cuda`, `rocm` o `cpu` antes de
  evaluar el shell
- **THEN** el shell selecciona ese extra, con independencia de los device nodes presentes en
  la máquina

#### Scenario: Valor de acelerador no soportado
- **WHEN** el usuario exporta `BURNSCAR_ACCEL` con un valor distinto de `cuda`, `rocm` o
  `cpu`
- **THEN** la evaluación del shell aborta con un mensaje que enumera los valores admitidos

### Requirement: Sincronización explícita del entorno
El sistema SHALL sincronizar el entorno de forma explícita y bajo demanda, sin ejecutar
ninguna sincronización implícita al entrar al shell.

#### Scenario: Entrada al shell sin descarga implícita
- **WHEN** el usuario entra al shell de devenv por primera vez
- **THEN** no se descarga ningún wheel de PyTorch ni se modifica el entorno virtual

#### Scenario: Sincronización bajo demanda
- **WHEN** el usuario lanza la tarea de sincronización del proyecto
- **THEN** se ejecuta la sincronización de `uv` con el extra del acelerador seleccionado y el
  grupo de desarrollo

#### Scenario: Verificación del entorno bajo demanda
- **WHEN** el usuario lanza la tarea de verificación del entorno
- **THEN** se ejecuta la comprobación de acelerador con el extra del acelerador seleccionado

### Requirement: Preservación del flujo uv nativo
El sistema SHALL mantener el flujo de trabajo con `uv` sin cambios observables para quien no
utiliza Nix, y sin que la presencia de los ficheros de Nix altere la resolución de
dependencias del proyecto.

#### Scenario: Resolución idéntica con y sin capa Nix
- **WHEN** el usuario ejecuta `uv sync` con un extra de acelerador, con o sin Nix disponible
- **THEN** la resolución de dependencias, los índices consultados y el contenido del
  lockfile son los mismos

#### Scenario: Ejecución de la suite de tests sin Nix
- **WHEN** el usuario ejecuta la suite de tests con `uv` en una máquina sin Nix
- **THEN** la suite se ejecuta sin errores y sin referencia a ficheros de la capa Nix

#### Scenario: Hook de direnv inerte sin devenv
- **WHEN** un usuario con `direnv` instalado pero sin `devenv` entra en el directorio del
  proyecto
- **THEN** el hook no altera el entorno ni produce ningún error
