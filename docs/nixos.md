# Guía de trabajo en NixOS (devenv)

> **En una frase:** cómo ejecutar el proyecto en NixOS con un shell reproducible de
> [devenv](https://devenv.sh), sin tocar el flujo `uv` que usa el resto del equipo.

Esta guía es **opcional**. Si trabajas en macOS, Linux o Windows, no necesitas nada de
esto: el proyecto se ejecuta con `uv` y ya está documentado en el
[README](../README.md). Lo que sigue aplica únicamente a quien tiene Nix.

---

## 1. Requisitos del sistema

Esto lo configuras en tu sistema, **no** en el repositorio:

| Requisito | Por qué |
|---|---|
| Nix con flakes | `experimental-features = nix-command flakes` en `nix.conf` |
| `devenv` en el `PATH` | `nix profile install nixpkgs#devenv` |
| `direnv` (opcional) | activa el shell automáticamente al entrar en el directorio |
| Módulos del kernel `amdgpu` y `kfd` | sin ellos no hay GPU accessible desde el proceso |
| `programs.nix-ld.enable = true` | permite ejecutar binarios genéricos (ver más abajo) |

En NixOS, esto se resuelve en la configuración del sistema:

```nix
boot.kernelModules = [ "amdgpu" "kfd" ];
programs.nix-ld.enable = true;
```

`programs.nix-ld` sustituye el cargador stub de NixOS por el real, de modo que
arrancan los binarios que vienen enlazados contra una glibc genérica: el binario de
`ruff` que trae el wheel y el CPython que descarga `uv` para sí mismo. Sin esta opción
fallan con:

```text
Could not start dynamically linked executable: ruff
```

Aun con `nix-ld` activo, el shell de devenv sigue siendo la vía recomendada en NixOS:
aporta el intérprete y deja las librerías de ROCm resueltas (§6), mientras que el Python
gestionado por `uv` no las trae (ver §8).

Comprueba que el dispositivo existe antes de seguir:

```bash
ls -l /dev/kfd /dev/dri
```

> [!NOTE]
> En una máquina con GPU NVIDIA lo que aparece es `/dev/nvidiactl` en lugar de `/dev/kfd`.
> La capa de nix detecta cualquiera de los dos casos.

---

## 2. Primer arranque

El repositorio trae `devenv.nix`, `devenv.yaml` y `devenv.lock`. El lockfile fija los
inputs, así que el shell es reproducible:

```bash
devenv shell
```

Entra en el shell y ya tienes el intérprete y `uv`:

```bash
python --version   # Python 3.12.15
uv --version
```

> [!IMPORTANT]
> `devenv shell` **no** sincroniza nada. Los wheels de PyTorch pesan varios gigas y el
> extra depende de tu GPU, así que la sincronización es siempre explícita (§3).

Si usas `direnv`, el hook `.envrc` que trae el repositorio activa el shell al entrar en el
directorio. Está protegido con una guarda, de modo que **si no tienes `devenv` instalado
no hace nada y no da error** — el resto del equipo puede tener `direnv` sin miedo.

---

## 3. Flujo de trabajo

Las tareas de `devenv` ya eligen el extra del acelerador por ti:

```bash
devenv tasks run burnscar:sync    # uv sync --extra <detectado> --group dev
devenv tasks run burnscar:check   # comprueba GPU, cómputo en GPU y TensorBoard
devenv tasks run burnscar:test    # pytest
```

`burnscar:check` es la primera comprobación que deberías hacer en una máquina nueva:

```text
torch=2.9.1+rocm6.4 backend=rocm
cuda_available=True device=AMD Radeon Graphics
arch_list=['gfx900', ..., 'gfx1030', ...]
matmul en GPU OK (sum=-5786.518)
tensorboard: 5 archivo(s) de evento en runs/check_env
ENTORNO OK
```

---

## 4. Elegir el acelerador

La capa de nix elige el extra (`cuda`, `rocm` o `cpu`) al **evaluar** el shell, y lo publica
en la variable `BURNSCAR_ACCEL`:

```bash
echo $BURNSCAR_ACCEL   # rocm
```

| En la máquina | Extra detectado |
|---|---|
| `/dev/nvidiactl` presente | `cuda` |
| `/dev/kfd` presente | `rocm` |
| ninguno de los dos | `cpu` |

Para forzarlo:

```bash
BURNSCAR_ACCEL=cuda devenv shell
```

Un valor no soportado aborta la evaluación con un mensaje explícito en lugar de fallar más
tarde y de forma confusa.

> [!NOTE]
> La detección ocurre en la máquina que **evalúa** el shell, no en la que lo ejecuta. Si
> evalúas en un sitio distinto del que trabajas (CI, una máquina de compilación), fuerza el
> valor con `BURNSCAR_ACCEL`.

---

## 5. Dónde vive el entorno virtual

| Flujo | Entorno virtual |
|---|---|
| `uv` nativo (resto del equipo) | `./.venv` en la raíz del repositorio |
| `devenv shell` (esta guía) | `.devenv/state/venv` |

Son directorios distintos y **no se pisan**: puedes usar los dos en la misma máquina sin
que uno invalide al otro. `devenv` fija `UV_PROJECT_ENVIRONMENT` para que `uv` use su
propio entorno.

### La trampa: si cambias el intérprete, recrea el venv

Si editas `languages.python.package` o `languages.python.libraries` en `devenv.nix`, cambia
la ruta del store del intérprete. El venv que ya tienes **sigue apuntando al intérprete
anterior**, porque `uv run` no lo reconstruye: el entorno sigue funcionando, pero con las
librerías viejas y sin los cambios de `devenv.nix`.

```bash
rm -rf .devenv/state/venv
devenv tasks run burnscar:sync
```

---

## 6. Qué NO hace esta capa

- **No instala ROCm.** `nixpkgs` no soporta ROCm como paquete distribuible. El runtime
  (`libamdhip64`, `libhsa-runtime64`, `libmiopen`, `librocblas`) viene **dentro del wheel**
  de PyTorch. Del host sólo hacen falta los módulos del kernel y los device nodes (§1).
- **No toca `pyproject.toml` ni `uv.lock`.** La resolución de dependencias, los extras y
  los índices de PyTorch son los mismos para todo el equipo, con Nix o sin Nix.
- **No cambia cómo se ejecuta el proyecto.** El código es el mismo; sólo cambia quién
  proporciona el intérprete y las librerías.

---

## 7. Avisos conocidos

| Aviso | Qué significa |
|---|---|
| `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` | el `libdrm_amdgpu` del wheel busca una tabla de IDs de GPU en una ruta de ROCm que no existe en NixOS. Es inofensivo: la GPU se enumera igual. |
| `GPU arch does not support max_autotune_gemm mode usage` | aviso de Inductor sobre `max_autotune_gemm` en esta arquitectura. No afecta a la corrección. |

---

## 8. Volver al flujo `uv` nativo

Nada de lo anterior es obligatorio. Desde fuera del shell de devenv:

```bash
uv sync --extra rocm --group dev
uv run --extra rocm python scripts/check_env.py
```

Es exactamente el mismo flujo que usa el resto del equipo, con la misma resolución de
dependencias (el `uv.lock` es el mismo para todos).

> [!IMPORTANT]
> En NixOS, `uv` necesita un intérprete que pueda arrancar. Tienes dos opciones:
>
> * El **Python gestionado por `uv`**, que descarga de `python-build-standalone`. Es un
>   binario genérico: sólo arranca si has activado `programs.nix-ld` (§1).
> * Un **Python construido con Nix**, que es el que te da `devenv shell`.
>
> Con cualquiera de los dos la resolución de dependencias es la misma (`uv.lock` es
> compartido), pero el shell de devenv ya te pone el segundo delante y te deja las
> librerías de ROCm resueltas (§6).
