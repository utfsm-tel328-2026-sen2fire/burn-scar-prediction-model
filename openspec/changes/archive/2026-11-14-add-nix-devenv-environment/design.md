# Design: Entorno de desarrollo NixOS (devenv)

## Context

Ver `proposal.md` para la motivación. Este documento recoge el estado técnico y las
decisiones de implementación.

Estado verificado sobre el host de referencia (NixOS 26.11, devenv 2.4.0, AMD Radeon
RX 6800 / Navi 21 / `gfx1030`, kernel con `amdgpu` + `kfd` y `/dev/kfd` y `/dev/dri`
presentes):

- El `devenv.nix` inicial fallaba **antes de llegar al proyecto**: la construcción del
  intérprete abortaba con `Hunk #1 FAILED at 40` al aplicar `mimetypes.patch` sobre el
  tarball de CPython 3.12.16.
- `languages.python.version` no resuelve `pkgs.python312` de nixpkgs, sino
  `nixpkgs-python.packages.${system}.3.12`, que compila un *release de parche concreto*
  (3.12.16 en el pin actual) desde el tarball oficial de python.org. El parche
  `mimetypes.patch` de nixpkgs está escrito para la disposición previa de la lista
  `knownfiles` en `Lib/mimetypes.py`; upstream la movió a la rama no-Windows, de modo que
  el hunk ya no aplica. El propio flake `nixpkgs-python` lleva ya un override
  específico para `3.14.9` (`replacePatch "mimetypes.patch" ./patches/3.14.9-mimetypes.patch`),
  lo que confirma que el problema es conocido pero no cubre la rama 3.12.x.
- `pkgs.python312` en el pin actual de nixpkgs es 3.12.15, anterior al cambio upstream, y
  se construye sin problemas.
- Los wheels de PyTorch para ROCm (`torch 2.9.1+rocm6.4`) existen para `cp311`–`cp314`, por
  lo que Python 3.12 es una elección válida. Traen embebidos `libamdhip64`,
  `libhsa-runtime64`, `libmiopen` y `librocblas`.
- `nixpkgs` **no** ofrece ROCm utilizable como paquete (`pkgs.rocm` no está soportado), de
  modo que el runtime de ROCm tiene que venir del propio wheel.
- devenv fija `UV_PROJECT_ENVIRONMENT` a `<estado>/venv`, `UV_PYTHON_DOWNLOADS=never` y
  `UV_PYTHON_PREFERENCE=only-system` cuando `languages.python.uv.enable` está activo.
- La GPU del host se enumera correctamente y la arquitectura `gfx1030` figura en la lista
  de arquitecturas que soporta el build de ROCm (`arch_list` incluye `gfx1030`), por lo que
  no hace falta `HSA_OVERRIDE_GFX_VERSION`.
- El `libdrm_amdgpu.so` que trae el wheel exige `liblzma.so.5` y `libbz2.so.1`, que en
  NixOS no están presentes por defecto.
- `devenv tasks run` **captura la salida estándar** de la tarea y la descarta de la vista
  en vivo: sin una redirección explícita, el usuario no ve nada de lo que imprime el
  comando.
- Cambiar `languages.python.package` o `languages.python.libraries` cambia la ruta del
  store del intérprete, y el venv existente sigue apuntando al intérprete anterior porque
  `uv run` no lo reconstruye.
- El host tiene `environment.stub-ld` (el valor por defecto de NixOS), que sustituye el
  cargador de ELF por un stub que imprime un error. Eso impide arrancar el binario de
  `ruff` que trae el wheel y el CPython que `uv` descarga para sí mismo. La solución es
  `programs.nix-ld.enable = true;`, que es configuración del sistema y no del
  repositorio.

Restricción transversal: el repositorio lo usan dos personas y sólo una usa Nix. Todo lo
que se añada debe ser **inerte** para quien no lo usa.

## Goals / Non-Goals

**Goals:**
- Que `devenv shell` construya y entre sin errores en un host NixOS con GPU AMD.
- Que el wheel de PyTorch con ROCm pueda inicializarse y compilar kernels JIT dentro del
  shell.
- Que la elección del extra de `uv` viva en la capa nix y no afecte a `pyproject.toml`.
- Que la sincronización del entorno sea siempre explícita y bajo demanda.
- Que el flujo `uv` nativo conserve exactamente la misma resolución, lockfile y rutas.

**Non-Goals:**
- No se empaqueta el proyecto como derivación de Nix (`languages.python.import` / uv2nix):
  el wheel de PyTorch con ROCm no es reproducible de forma fiable desde nixpkgs y no es
  necesario para el flujo de trabajo.
- No se instala ROCm desde nixpkgs ni se configura el kernel desde el repositorio: el
  módulo `amdgpu`/`kfd` y sus device nodes son responsabilidad de la configuración del
  sistema del host.
- No se modifican `pyproject.toml`, `uv.lock`, `src/`, `scripts/` ni `tests/`.
- No se introduce autodetección de acelerador para los usuarios sin Nix: la lógica de
  selección se queda en la capa nix, según lo acordado.

## Decisions

### Decisión 1: `languages.python.package = pkgs.python312` en lugar de `languages.python.version`
- **Elección**: fijar el intérprete con `pkgs.python312` (3.12.15 en el pin actual) y no
  usar `languages.python.version`.
- **Razón**: `languages.python.version` delega en el flake `nixpkgs-python`, que compila un
  release de parche concreto de CPython y aplica los parches de nixpkgs sobre ese tarball.
  El `mimetypes.patch` de nixpkgs dejó de aplicar sobre las 3.12.x recientes, lo que
  aborta la construcción del shell antes de cualquier actividad del proyecto. `pkgs.python312`
  usa el tarball y el parche coherentes dentro del mismo commit de nixpkgs, y además está
  disponible en la caché binaria.
- **Consecuencia**: `nixpkgs-python` deja de ser necesario y se elimina del `devenv.yaml`;
  `devenv.lock` se regenera y queda sin ese input.
- **Alternativas consideradas**:
  - `devenv update` para subir el pin de `nixpkgs-python`: no determinista; el fix del
    parche puede no existir en la versión a la que se actualice, y un `devenv update`
    posterior volvería a exponer el mismo fallo.
  - Fijar `languages.python.version = "3.12.15"` (un release concreto): funcional, pero
    ata el proyecto a un parche de CPython que deja de recibir parches de seguridad.
  - Usar el `python3` por defecto de nixpkgs (3.14.8): también válido, pero 3.12 es lo que
    ya usan el resto de consumidores del proyecto y lo que cubre el rango de wheels de
    PyTorch con ROCm y CUDA del lockfile.

### Decisión 2: La detección del acelerador ocurre en la evaluación de nix, no en runtime
- **Elección**: `builtins.getEnv "BURNSCAR_ACCEL"` como override; si está vacío,
  `builtins.pathExists "/dev/nvidiactl"` → `cuda`, `builtins.pathExists "/dev/kfd"` →
  `rocm`, y `cpu` en caso contrario. El resultado se publica como `BURNSCAR_ACCEL` y se
  interpola en las tareas del proyecto.
- **Razón**: la presencia de los device nodes del driver es la señal más fiable y no
  exige instalar ROCm ni CUDA en el sistema, lo que es deseable: el runtime llega dentro
  del wheel. `builtins.getEnv` funciona en la evaluación de devenv porque ésta se realiza
  de forma impura; se verificó que el valor llega sin error al shell.
- **Alternativas consideradas**:
  - Detección en runtime dentro de `enterShell`: llegaría tarde, porque la tarea de
    sincronización de uv se ejecuta antes que `devenv:enterShell` y su `extra` debe estar
    decidido en tiempo de evaluación.
  - Interrogar `nvidia-smi` / `rocminfo`: obliga a instalar esas herramientas en el sistema
    para poder elegir el extra, que es justo lo que se quiere evitar.
  - Fijar el extra a mano en el `devenv.nix`: obligaría a editar un fichero versionado por
    cada cambio de máquina.

### Decisión 3: `uv sync` explícito, nunca implícito
- **Elección**: `languages.python.uv.sync.enable = false`; la sincronización, la
  verificación y los tests se lanzan con tareas `burnscar:sync`, `burnscar:check` y
  `burnscar:test`.
- **Razón**: el extra depende de la GPU y cambiarlo reinstala el wheel completo (varios GB
  para ROCm). Un `uv sync` implícito al entrar al shell provocaría descargas inesperadas la
  primera vez y re-resoluciones del entorno en cada cambio de extra. Además, el `extra`
  tendría que quedar escrito en la expresión nix.
- **Alternativas consideradas**:
  - `uv.sync.enable = true` con `uv.sync.extras = [ accel ]`: es la vía soportada por
    devenv y resolvería el extra desde nix, pero hace la descarga implícita y ata la
    ejecución del shell a la disponibilidad de red.
- **Consecuencia**: el venv de devenv vive en `<estado>/venv` y el de uv nativo en
  `./.venv`; son directorios distintos y no compiten entre sí.

### Decisión 4: `CPATH` para el compilador JIT embebido de MIOpen
- **Elección**: `CPATH = "${gcc}/include/c++/${gccVersion}:${gcc}/include/c++/${gccVersion}/${targetTriplet}:${pkgs.glibc.dev}/include"`.
- **Razón**: el wheel de ROCm embebe `libhiprtc.so`, que es el compilador JIT que MIOpen
  usa para compilar los kernels de convolución. Ese compilador va incluido dentro de una
  librería precompilada y **no** hereda los flags del `cc-wrapper` de nixpkgs, así que
  busca las cabeceras de la librería estándar de C++ en `/usr/include/c++/<ver>/` y las de
  libc en `/usr/include/`. En NixOS ninguna de esas rutas existe, y la compilación aborta
  primero con `fatal error: 'type_traits' file not found` y después con
  `'features.h' file not found`. `CPATH` le indica dónde están de verdad. Se referencian
  `pkgs.gcc.cc` y `pkgs.glibc.dev` en lugar de hashes del store para que las rutas se
  muevan solas con el pin de nixpkgs.
- **Verificado**: los compiladores externos del shell (el `gcc`/`g++` de nixpkgs y el que
  usa Triton) ya encuentran las cabeceras por su cuenta, y con `CPATH` presente compilan
  sin errores. Se ejercitaron `conv2d`, `conv2d` con grupos, `conv2d` depthwise,
  `convtranspose2d`, `batchnorm`, `relu`, `gelu`, `softmax`, pooling, `lstm`, `matmul`,
  `einsum` y `torch.compile` (con la caché de Triton vacía): todos pasan. El camino de
  JIT de MIOpen no llegó a dispararse con esas configuraciones, de modo que `CPATH` se
  mantiene como mitigación preventiva del error reportado, no como un requisito medido
  en el stack actual.
- **Alternativas consideradas**:
  - `NIX_CFLAGS_COMPILE` / `CPLUS_INCLUDE_PATH`: afectan a las compilaciones que lance el
    propio shell a través del `cc-wrapper`, pero no alcanzan al clang embebido en
    `libhiprtc.so`, que invoca su propia línea de órdenes.
  - Parchear el wheel en tiempo de instalación: ataría la solución a una versión concreta
    de torch y no es mantenible.

### Decisión 5: Las librerías de sistema entran por `languages.python.libraries`
- **Elección**: declarar `zlib`, `zstd`, `xz`, `bzip2`, `gcc.lib` y `numactl` en
  `languages.python.libraries`, y `pkgs.gcc` en `packages`.
- **Razón**:
  - `gcc.lib` aporta `libstdc++.so.6` y `libgomp.so.1`.
  - `numactl` aporta `libnuma.so.1`, de la que dependen `libhsa-runtime64.so` y las
    librerías de ROCm.
  - `zlib`/`zstd` aportan `libz.so.1` y `libzstd.so.1`.
  - `xz`/`bzip2` aportan `liblzma.so.5` y `libbz2.so.1`, que necesita el
    `libdrm_amdgpu.so` que trae el wheel.
  - `pkgs.gcc` en `packages` deja `gcc`/`g++` en el `PATH` del shell, que es lo que
    necesita `torch.utils.cpp_extension` y la compilación de extensiones nativas.
- **Verificado**: con la lista completa, `ldd` no reporta ninguna dependencia sin
  resolver en `libtorch_cpu.so`, `libtorch_hip.so`, `libamdhip64.so`,
  `libhsa-runtime64.so`, `libmiopen.so` ni `librocblas.so`. Sin `xz`/`bzip2` quedan
  `liblzma.so.5` y `libbz2.so.1` sin resolver.
- **Mecanismo**: `languages.python.libraries` se materializa como un prefijo de
  `LD_LIBRARY_PATH` en el *wrapper* del intérprete. Como el venv de uv enlaza a ese
  wrapper, las librerías quedan resolubles para el intérprete y para los procesos hijos
  que lance torch.
- **Alternativas consideradas**: `LD_LIBRARY_PATH` manual con rutas del store: frágil,
  porque las rutas cambian al mover el pin de nixpkgs.

### Decisión 6: ROCm no se instala desde nixpkgs
- **Elección**: no declarar ningún paquete de ROCm; el runtime proviene del wheel.
- **Razón**: nixpkgs no soporta ROCm como paquete distribuible, y el wheel de PyTorch
  ya incluye `libamdhip64`, `libhsa-runtime64`, `libmiopen` y `librocblas`. Del host sólo
  hacen falta el driver del kernel (`amdgpu`, `kfd`) y los device nodes `/dev/kfd` y
  `/dev/dri`, que son configuración del sistema y no del repositorio.
- **Consecuencia**: el requisito de sistema (módulos del kernel cargados y
  `programs.nix-ld.enable`) se documenta en `docs/nixos.md` y queda fuera del alcance del
  repositorio.

### Decisión 7: `.envrc` versionado con guarda de presencia de `devenv`
- **Elección**: versionar un `.envrc` que sólo actúa si `devenv` está en el `PATH` **y**
  existe un `devenv.yaml` en el directorio, y retirar `.envrc` de `.gitignore` (además de
  ignorar `.devenv/`).
- **Razón**: la guarda hace que el hook sea inerte para el resto del equipo, que usa
  `direnv` sin Nix: no se altera el entorno ni se emite ningún error. Versionarlo evita
  que cada persona de NixOS tenga que reconstruirlo a mano. La segunda condición
  (`devenv.yaml`) evita que un `devenv` instalado pero fuera de un proyecto de devenv
  provoque un error al invocar `use devenv`.
- **Verificado**: con `devenv` ausente, `direnv exec` no altera el entorno y no emite
  error; con `devenv` presente, `direnv allow` activa el shell y expone el intérprete y las
  variables de la capa nix.
- **Alternativas consideradas**:
  - Mantener `.envrc` ignorado y documentarlo: cero riesgo para el equipo, pero obliga a
    copiar un bloque de código a mano en cada máquina nueva.
  - Versionarlo sin guarda: rompería el `direnv` de quien no tenga `devenv` instalado.

### Decisión 8: Python 3.12 como intérprete del proyecto
- **Elección**: `pkgs.python312`.
- **Razón**: `pyproject.toml` declara `requires-python = ">=3.11"`, y el lockfile resuelve
  wheels `cp312` tanto para `torch 2.9.1+rocm6.4` como para la serie `cu128`. 3.12 es el
  punto medio entre lo que declara el proyecto y lo que cubren los wheels.

### Decisión 9: Las tareas dirigen su salida estándar a la salida de error
- **Elección**: terminar el `exec` de cada tarea con `>&2`.
- **Razón**: `devenv tasks run` captura la salida estándar del comando y la aparta de la
  vista en vivo; lo que se muestra es la salida de error. Sin la redirección, un
  `devenv tasks run burnscar:check` que termina correctamente no muestra ni el backend
  detectado ni el resultado del cálculo en GPU, que es justo lo que el usuario necesita
  ver. Redirigir la salida estándar a la de error mantiene el resultado visible y no
  altera el código de salida.
- **Verificado**: con la redirección, la tarea muestra la salida completa de
  `scripts/check_env.py`; sin ella, la tarea muestra únicamente un `{}` final.

### Decisión 10: Recrear el venv cuando cambia el intérprete
- **Elección**: documentar en `docs/nixos.md` que, tras cambiar `languages.python.package`
  o `languages.python.libraries`, hay que borrar el venv de devenv
  (`rm -rf .devenv/state/venv`) antes de volver a sincronizar.
- **Razón**: esas opciones se materializan en la ruta del store del intérprete. El venv
  que crea uv enlaza a esa ruta, y `uv run` **no** lo reconstruye aunque el intérprete
  haya cambiado: el venv sigue apuntando al intérprete anterior y las librerías
  declaradas en `libraries` no llegan a aplicarse. Es una trampa silenciosa: el entorno
  funciona pero con las librerías viejas.
- **Alternativas consideradas**: forzar la recreación desde el propio `devenv.nix`
  (por ejemplo, borrando el venv en `enterShell` cuando cambia el checksum del
  intérprete): automatiza el paso, pero obliga a reconstruir el venv en cada entrada al
  shell y a pagar el coste de sincronización con más frecuencia.

### Decisión 11: La guía de NixOS vive en `docs/nixos.md`, no en el `README`
- **Elección**: escribir la documentación operativa de NixOS en un documento propio,
  `docs/nixos.md`, y dejar el `README` con una única fila en su tabla de documentación que
  lo enlace.
- **Razón**: el `README` es el punto de entrada del proyecto y lo lee todo el equipo,
  incluido quien nunca va a usar Nix. Meter ahí los requisitos del kernel, el
  funcionamiento de las tareas de devenv, la ubicación del venv y los avisos sobre ROCm
  loataría el README con información que sólo le afecta a una persona. La guía se enlaza
  desde la tabla de documentación del `README` y desde el índice de `docs/`, de modo que
  sigue siendo alcanzable en un clic.
- **Alternativas consideradas**:
  - Sección larga en el `README`: más visible, pero ensucia la entrada del proyecto con
    detalles de una plataforma concreta.
  - Dejar la guía sólo en `docs/nixos.md` sin enlazarla desde el `README`: el `README`
    quedaría limpio, pero nadie sabría que la guía existe salvo por el índice de `docs/`.

## Risks / Trade-offs

- **[Riesgo] La detección ocurre en la máquina que evalúa, no en la que ejecuta** → Si el
  `devenv.nix` se evalúa en un host distinto del de trabajo (CI, una máquina de
  compilación), el extra detectado puede no coincidir con la GPU real. Mitigación: el
  override `BURNSCAR_ACCEL` explícito, y la tarea de sincronización muestra siempre el
  extra que va a usar.
- **[Riesgo] `devenv update` puede volver a romper la construcción del intérprete** → Si
  nixpkgs sube el tarball de CPython a una versión cuyo `mimetypes.py` vuelve a no casar
  con `mimetypes.patch`, `pkgs.python312` deja de construirse. Mitigación: el fallo es
  ruidoso y localizado en la fase de parcheo del intérprete, no en el proyecto; se
  revisa el parche upstream y, si procede, se fija el release concreto de CPython con
  `languages.python.version` como ocurre en el propio flake `nixpkgs-python`.
- **[Riesgo] Los wheels de ROCm ocupan varios GB** → El cache de `uv` es global y está
  indexado por hash, de modo que la descarga se hace una sola vez por máquina; la tarea de
  sincronización es explícita para que la descarga no sorprenda.
- **[Riesgo] `CPATH` es una variable global del shell** → Puede afectar a compilaciones
  que se lancen dentro del shell y esperen las cabeceras del sistema. Mitigación: las
  rutas añadidas son las de gcc y glibc del propio pin, que es lo que un usuario de NixOS
  espera encontrar; y el shell de devenv es el único ámbito afectado.
- **[Riesgo] El wheel puede seguir pidiendo cabeceras que la capa nix no expone** → Se
  detectó que el `libdrm_amdgpu.so` del wheel depende de `liblzma.so.5` y `libbz2.so.1`,
  y se cubrieron. Si una versión futura del wheel añade otra dependencia, aparecerá como
  `not found` en `ldd` y como error de carga en tiempo de ejecución. Mitigación: la
  comprobación de `ldd` sobre las librerías del wheel forma parte de la verificación del
  cambio, y `languages.python.libraries` es el punto único donde añadir el paquete que
  falte.
- **[Trade-off] Dos entornos virtuales conviviendo** → devenv usa `<estado>/venv` y uv
  nativo usa `./.venv`. A cambio, ninguno invalida al otro y la patología de disco es
  predecible. Se documenta en `docs/nixos.md` para que nadie busque el intérprete en el
  sitio equivocado.
- **[Trade-off] `libgomp` y `libstdc++` vienen del gcc de nixpkgs, no del del sistema** →
  Es la práctica habitual en Nix y evita inconsistencias de ABI, pero implica que el
  binario de Python del shell no es intercambiable con el del sistema.

## Migration Plan

1. No hay migración de datos ni cambios de comportamiento para el flujo actual: la capa
   Nix es aditiva.
2. Para el host NixOS: `devenv shell` y después `devenv tasks run burnscar:sync`.
3. Rollback: borrar `devenv.nix`, `devenv.yaml`, `devenv.lock` y `devenv/`; el flujo `uv`
   nativo no se ve afectado en ningún momento, porque `pyproject.toml` y `uv.lock` no se
   han tocado.

## Open Questions

- El camino de JIT de MIOpen (el compilador embebido en `libhiprtc.so`) no llegó a
  dispararse con las configuraciones de convolución probadas sobre el stack actual, de
  modo que `CPATH` queda como mitigación preventiva del error reportado y no como un
  requisito medido. Si apareciera un kernel que sí lo active, habría que confirmar que
  `CPATH` lo cubre o buscar la ruta de cabeceras que ese compilador espera. No afecta a
  los requisitos de este cambio.
