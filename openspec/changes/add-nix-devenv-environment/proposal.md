# Proposal: Entorno de desarrollo NixOS (devenv) para el proyecto

## Why

Uno de los dos integrantes del equipo trabaja sobre NixOS y no puede ejecutar el proyecto
con el flujo actual, que sólo cubre `uv` + Python del sistema. El intento previo de
provisionar el entorno con `devenv.nix` falla en la construcción del propio shell, no en el
proyecto: `devenv shell` aborta con

```
Failed to realize shell derivation: Cannot build '.../python3-3.12.16.drv'
  patching file Lib/mimetypes.py
  Hunk #1 FAILED at 40
```

porque `languages.python.version` resuelve el intérprete a través del flake `nixpkgs-python`,
que compila una *versión de parche concreta* de CPython (3.12.16) en lugar del `python312`
de nixpkgs, y el parche `mimetypes.patch` de nixpkgs ya no aplica sobre ese tarball.

Aun corrigiendo eso, el proyecto sigue sin ser ejecutable en NixOS: los wheels de PyTorch
para ROCm necesitan librerías de sistema que NixOS no expone por defecto
(`libstdc++.so.6`, `libgomp.so.1`, `libnuma.so.1`) y, sobre todo, el JIT de MIOpen compila
los kernels de convolución en tiempo de ejecución con el clang embebido en `libhiprtc.so`,
que busca las cabeceras de la librería estándar de C++ y de libc en `/usr/include/c++/<ver>/`
y `/usr/include/`, rutas que no existen en NixOS.

Esta propuesta añade una capa de provisión opcional sobre Nix/devenv que deja el flujo
`uv` intacto para el resto del equipo y hace que la GPU AMD (Radeon RX 6800, `gfx1030`)
del proyecto funcione con ROCm.

## What Changes

- **Capa Nix reproducible** (`devenv.nix`, `devenv.yaml`, `devenv.lock`) que provisiona
  Python 3.12, `uv` y las librerías de sistema que requieren los wheels de PyTorch, con
  los inputs fijados en `devenv.lock`.
- **Intérprete sin depender de `nixpkgs-python`**: se usa `languages.python.package =
  pkgs.python312` en lugar de `languages.python.version`, eliminando el punto de fallo
  (`mimetypes.patch` sobre CPython 3.12.16) y el input `nixpkgs-python` del `devenv.yaml`.
- **Cabeceras para el JIT de MIOpen**: `CPATH` apuntando a las cabeceras de C++ de `gcc` y a
  las de libc de `pkgs.glibc.dev`, para que la compilación JIT de MIOpen no aborte con
  `fatal error: 'type_traits' file not found` ni `'features.h' file not found`.
- **Librerías de sistema para los wheels de PyTorch** (`libstdc++`, `libgomp`, `libnuma`,
  `zlib`, `zstd`, `liblzma`, `libbz2`) declaradas en `languages.python.libraries`, sin
  instalar ROCm desde nixpkgs: el wheel de PyTorch ya incluye `libamdhip64`,
  `libhsa-runtime64`, `libmiopen` y `librocblas`, y del host sólo se necesita el driver
  del kernel (`amdgpu`/`kfd`) y los device nodes `/dev/kfd` y `/dev/dri`.
- **Selección del acelerador dentro de la capa nix**: detección en tiempo de evaluación
  (`/dev/nvidiactl` → `cuda`, `/dev/kfd` → `rocm`, si no `cpu`), con override explícito
  mediante la variable de entorno `BURNSCAR_ACCEL` y validación que aborta la evaluación
  con un mensaje claro ante valores inválidos.
- **Sincronización explícita**: `devenv shell` no ejecuta `uv sync` implícitamente (evita
  descargas de varios GB y re-resoluciones del entorno al cambiar de extra). La
  sincronización, la verificación del entorno y los tests se lanzan con las tareas
  `devenv tasks run burnscar:sync`, `burnscar:check` y `burnscar:test`, que usan el extra
  detectado.
- **`.envrc` versionado con guarda**: el archivo sólo actúa si `devenv` está instalado, de
  modo que `direnv` no rompe a los compañeros que no usan Nix. Se retira `.envrc` de
  `.gitignore` y se añade `.devenv/` (estado local del shell).
- **Documentación**: una guía nueva y autocontenida en `docs/nixos.md` (requisitos de
  sistema, flujo de trabajo, la variable `BURNSCAR_ACCEL`, dónde vive el entorno virtual
  y la advertencia de que ROCm no se instala desde nixpkgs), enlazada desde el índice de
  `docs/` y desde la tabla de documentación del `README` con **una línea en cada tabla**.
  El `README` no crece con detalles de NixOS: sólo apunta a la guía.

### No cambia

- `pyproject.toml` y `uv.lock` **no se tocan**: la resolución de dependencias, los extras
  `cuda`/`rocm`/`cpu`, los índices de PyTorch y la matriz de conflictos siguen siendo
  exactamente los mismos para todo el equipo.
- `src/`, `scripts/` y `tests/` no se modifican: `scripts/check_env.py` ya detecta ROCm
  (`torch.version.hip`) y funciona tal cual.

## Capabilities

### New Capabilities
- `nix-development-environment`: Capa opcional de provisión del entorno de desarrollo sobre
  Nix/devenv para NixOS: shell reproducible, provisión de librerías y cabeceras que
  necesitan los wheels de PyTorch, selección del acelerador y control explícito de la
  sincronización, sin alterar el flujo `uv` nativo.

### Modified Capabilities
- `execution-environment`: La verificación del entorno pasa a cubrir explícitamente la
  arquitectura AMD (Radeon RX 6800 / `gfx1030`) como segundo escenario de éxito, junto al
  escenario NVIDIA ya existente.

## Impact

- **Archivos nuevos**: `devenv.nix`, `devenv.yaml`, `devenv.lock`, y `.envrc` (hoy ignorado
  por `.gitignore`, pasa a versionarse con guarda).
- **Archivos modificados**: `.gitignore` (`.envrc` deja de ignorarse, se ignora `.devenv/`),
  `README.md` (una fila en la tabla de documentación), `docs/README.md` (una fila en la
  tabla de documentos), `docs/experiment-tracking.md` (nota sobre dónde vive el entorno
  virtual en cada flujo).
- **Documentación nueva**: `docs/nixos.md`, la guía de trabajo en NixOS.
- **Dependencias**: no se añaden dependencias de Python. En el lado Nix se incorporan
  `pkgs.python312`, `pkgs.uv`, `pkgs.gcc`, `pkgs.glibc.dev`, `numactl`, `zlib`, `zstd`,
  `xz` y `bzip2`.
- **Flujo de desarrollo**: para quien usa NixOS, `devenv shell` + `devenv tasks run
  burnscar:sync`; para el resto, `uv sync --extra <backend>` sin cambios respecto a hoy.
- **Requisito de sistema fuera del repositorio**: el host NixOS debe tener cargados los
  módulos del kernel `amdgpu` y `kfd` y exponer `/dev/kfd` y `/dev/dri`; se documenta, pero
  no lo provee este repositorio.
