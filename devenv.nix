{ pkgs, config, lib, ... }:

let
  gcc = pkgs.gcc.cc;
  gccVersion = pkgs.gcc.version;
  targetTriplet = pkgs.stdenv.hostPlatform.config;

  # Seleccion del acelerador. Se evalua al construir el shell, no al entrar.
  # Override manual: BURNSCAR_ACCEL=cuda|rocm|cpu
  accelOverride = builtins.getEnv "BURNSCAR_ACCEL";
  hasNvidia = builtins.pathExists "/dev/nvidiactl";
  hasAmd = builtins.pathExists "/dev/kfd";
  accel =
    if accelOverride != "" then accelOverride
    else if hasNvidia then "cuda"
    else if hasAmd then "rocm"
    else "cpu";
  accelValidos = [ "cuda" "rocm" "cpu" ];
in
{
  languages.python = {
    enable = true;

    # pkgs.python312 y NO languages.python.version: este ultimo resuelve el
    # interprete a traves del flake nixpkgs-python, que compila un release de
    # parche concreto de CPython (3.12.16) cuyo "mimetypes.patch" de nixpkgs
    # ya no aplica -> la construccion del shell aborta antes de llegar al
    # proyecto. Ver docs/nixos.md.
    package = pkgs.python312;

    # uv gestiona su propio entorno (UV_PROJECT_ENVIRONMENT). El venv de devenv
    # es el que usa uv, no uno adicional.
    venv.enable = false;
    uv.enable = true;

    # Sin sincronizacion implicita: el extra depende de la GPU y cambiarlo
    # reinstala el wheel completo. Se sincroniza bajo demanda.
    uv.sync.enable = false;

    # Librerias de sistema que necesitan los wheels de PyTorch. devenv las
    # anade al LD_LIBRARY_PATH del interprete, y el venv de uv enlaza a ese
    # interprete, asi que quedan resolubles para torch y sus procesos hijos.
    libraries = with pkgs; [
      zlib # libz.so.1
      zstd # libzstd.so.1
      xz # liblzma.so.5 (lo pide el libdrm_amdgpu.so del wheel)
      bzip2 # libbz2.so.1 (idem)
      gcc.lib # libstdc++.so.6, libgomp.so.1
      numactl # libnuma.so.1 (dependencia del runtime de HSA/ROCm)
    ];
  };

  # gcc/g++ en el PATH: lo necesitan torch.utils.cpp_extension y la
  # compilacion de extensiones nativas.
  packages = [ pkgs.gcc ];

  env = {
    # Acelerador detectado, para que el shell y las tareas coincidan.
    BURNSCAR_ACCEL = accel;

    # MIOopen JIT-compila los kernels de conv en tiempo de ejecucion con el
    # clang embebido en libhiprtc.so. Ese compilador no hereda los flags del
    # cc-wrapper de nixpkgs y busca las cabeceras de la libreria estandar de
    # C++ en /usr/include/c++/<ver>/ y las de libc en /usr/include/, rutas que
    # no existen en NixOS. CPATH le dice donde estan de verdad.
    # Se referencian pkgs.gcc.cc y pkgs.glibc.dev, no hashes del store, para
    # que las rutas se muevan solas con el pin de nixpkgs.
    CPATH = "${gcc}/include/c++/${gccVersion}:${gcc}/include/c++/${gccVersion}/${targetTriplet}:${pkgs.glibc.dev}/include";
  };

  tasks = {
    # NB: cada exec termina en ">&2" porque devenv tasks run captura la
    # salida estandar y la aparta de la vista en vivo; lo que muestra es la
    # salida de error.
    "burnscar:sync" = {
      description = "Sincroniza el entorno uv con el extra del acelerador detectado";
      exec = "uv sync --extra ${accel} --group dev >&2";
    };
    "burnscar:check" = {
      description = "Verifica GPU, computo matricial y TensorBoard";
      exec = "uv run --extra ${accel} python scripts/check_env.py >&2";
    };
    "burnscar:test" = {
      description = "Ejecuta la bateria de tests";
      exec = "uv run --extra ${accel} pytest >&2";
    };
  };

  assertions = [
    {
      assertion = lib.elem accel accelValidos;
      message = "BURNSCAR_ACCEL debe ser uno de ${lib.concatStringsSep ", " accelValidos} (valor: '${accel}')";
    }
  ];
}
