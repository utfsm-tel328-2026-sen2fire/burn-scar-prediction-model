# Tasks

## 1. Shell Nix reproducible

- [x] 1.1 Escribir `devenv.nix` con `languages.python.enable = true`, `languages.python.package = pkgs.python312` (no `languages.python.version`), `languages.python.uv.enable = true`, `uv.sync.enable = false` y `venv.enable = false`, y verificar con `devenv shell -- bash -c 'python --version && uv --version'` que el shell construye y expone Python 3.12.x y `uv`.
- [x] 1.2 Quitar el input `nixpkgs-python` de `devenv.yaml` y regenerar `devenv.lock`, verificando que `grep -c nixpkgs-python devenv.lock` devuelve 0 y que `devenv shell` sigue evaluando sin errores.

## 2. Runtime del acelerador en NixOS

- [x] 2.1 Declarar en `languages.python.libraries` las librerías `zlib`, `zstd`, `xz`, `bzip2`, `gcc.lib` y `numactl`, y añadir `pkgs.gcc` a `packages` para tener `gcc`/`g++` en el `PATH`; verificar tras la sincronización que `ldd` sobre `libtorch_cpu.so`, `libtorch_hip.so`, `libamdhip64.so`, `libhsa-runtime64.so`, `libmiopen.so` y `librocblas.so` no reporta ninguna dependencia `not found`.
- [x] 2.2 Declarar `env.CPATH` con las cabeceras de C++ de `pkgs.gcc.cc` y las de libc de `pkgs.glibc.dev` (referenciando `pkgs.gcc.version` y `pkgs.stdenv.hostPlatform.config`, sin hashes del store), verificando que compila una convolución y un `torch.compile` con la caché de Triton vacía sin errores de cabecera no encontrada.
- [x] 2.3 Documentar en `docs/nixos.md` el requisito de sistema `programs.nix-ld.enable = true;` (sin él, el binario de `ruff` del wheel y el CPython gestionado por uv no arrancan por el `stub-ld` de NixOS), verificando que invocando el cargador real de glibc de Nix ambos binarios arrancan.

## 3. Selección del acelerador en la capa nix

- [x] 3.1 Implementar en `devenv.nix` la detección del acelerador (`builtins.getEnv "BURNSCAR_ACCEL"` como override; `/dev/nvidiactl` → `cuda`; `/dev/kfd` → `rocm`; en otro caso `cpu`), publicarla como `env.BURNSCAR_ACCEL` y añadir una `assertion` que aborte la evaluación con el mensaje de valores admitidos ante un valor inválido; verificar los cuatro casos con `devenv shell -- bash -c 'echo $BURNSCAR_ACCEL'` y con `BURNSCAR_ACCEL=cuda|devenv shell -- bash -c 'echo $BURNSCAR_ACCEL'`.
- [x] 3.2 Crear `docs/nixos.md` con la guía de trabajo en NixOS (requisitos de sistema, flujo de trabajo, `BURNSCAR_ACCEL`, ubicación del entorno virtual y el aviso de que ROCm no se instala desde nixpkgs) y documentar en ella la variable `BURNSCAR_ACCEL`, el valor detectado en la máquina de referencia y cómo forzar otro extra; verificando que los comandos que aparecen en la guía se ejecutan tal cual se escriben.

## 4. Sincronización explícita del entorno

- [x] 4.1 Definir en `devenv.nix` las tareas `burnscar:sync` (`uv sync --extra <detectado> --group dev`), `burnscar:check` (`uv run --extra <detectado> python scripts/check_env.py`) y `burnscar:test` (`uv run --extra <detectado> pytest`), usando el extra detectado en tiempo de evaluación y terminando cada `exec` con `>&2` para que `devenv tasks run` muestre la salida del comando; verificar con `devenv tasks list` que las tres aparecen y con `devenv tasks run burnscar:check` que se ve la salida completa del script.
- [x] 4.2 Verificar que entrar al shell no sincroniza nada: tras `devenv shell -- bash -c 'true'` el directorio `site-packages` del venv de devenv debe seguir vacío y no debe haberse descargado ningún wheel de PyTorch.
- [x] 4.3 Completar `docs/nixos.md` con el flujo completo de trabajo en NixOS (`devenv shell`, `devenv tasks run burnscar:sync`, `devenv tasks run burnscar:check`, `devenv tasks run burnscar:test`), la ubicación del entorno virtual y la necesidad de borrar `.devenv/state/venv` cuando cambie el intérprete del `devenv.nix`, verificando que la guía contiene los cuatro comandos y la nota del venv.

## 5. Preservación del flujo uv nativo

- [x] 5.1 Actualizar `.gitignore` para dejar de ignorar `.envrc` e ignorar `.devenv/`, verificando con `git check-ignore -v .envrc .devenv/state` que `.envrc` ya no está ignorado y `.devenv/` sí.
- [x] 5.2 Versionar un `.envrc` con guarda doble (`if command -v devenv >/dev/null 2>&1 && [ -f devenv.yaml ]; then eval "$(devenv direnvrc)"; use devenv; fi`), verificando que con `devenv` ausente el hook no altera el entorno ni emite error, y que con `devenv` presente `direnv allow` activa el shell correctamente.
- [x] 5.3 Verificar que `pyproject.toml` y `uv.lock` no han cambiado (`git diff --exit-code pyproject.toml uv.lock`) y que `uv lock --check` sigue dando la resolución por válida, para confirmar que la capa nix no altera la resolución de dependencias.
- [x] 5.4 Añadir en `docs/experiment-tracking.md` una nota sobre dónde vive el entorno virtual en cada flujo (`.venv` de uv nativo frente al venv de devenv) y sobre su consecuencia al lanzar TensorBoard, verificando que la nota remite a `docs/nixos.md`.
- [x] 5.5 Enlazar `docs/nixos.md` con una sola fila en la tabla de documentación del `README.md` y con una sola fila en la tabla de documentos de `docs/README.md`, verificando que el `README` no contiene ningún otro detalle de NixOS (ni comandos, ni variables, ni requisitos) más allá de ese enlace.

## 6. Verificación integral en la máquina de referencia (AMD)

- [x] 6.1 Ejecutar `devenv tasks run burnscar:sync` y después `devenv tasks run burnscar:check`, verificando que la comprobación reporta el backend `rocm`, identifica la Radeon RX 6800, completa la multiplicación de tensores en GPU y escribe el fichero de evento de TensorBoard.
- [x] 6.2 Ejecutar `devenv tasks run burnscar:test` y verificar que la suite completa pasa sin errores con el extra `rocm`.
