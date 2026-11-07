"""Levanta el servidor de TensorBoard apuntando a la ruta compartida (o fallback local) resuelta."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burn_scar.tracking import resolve_storage_paths  # noqa: E402


def build_tensorboard_command(
    logdir: Path,
    port: int,
    host: str = "localhost",
) -> list[str]:
    """Construye el comando para ejecutar TensorBoard sobre el logdir indicado."""
    tb_bin = shutil.which("tensorboard")
    base_cmd = [tb_bin] if tb_bin else [sys.executable, "-m", "tensorboard.main"]
    return [
        *base_cmd,
        "--logdir",
        str(logdir),
        "--port",
        str(port),
        "--host",
        host,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inicia TensorBoard apuntando al directorio de runs compartido o local."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Puerto HTTP para TensorBoard (por defecto usa BURNSCAR_TENSORBOARD_PORT o 6006).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host donde escuchar (usa 0.0.0.0 para exponer en red local).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime la configuración resuelta y el comando sin iniciar el servidor.",
    )
    args = parser.parse_args(argv)

    storage = resolve_storage_paths()
    port = args.port if args.port is not None else storage.tensorboard_port
    mode_label = "NFS compartido" if storage.is_shared else "Fallback local"

    print(f"[TensorBoard] Modo: {mode_label}")
    print(f"[TensorBoard] Usuario activo: {storage.user}")
    print(f"[TensorBoard] Directorio de logs: {storage.runs_dir}")
    print(f"[TensorBoard] URL: http://{args.host}:{port}")

    cmd = build_tensorboard_command(logdir=storage.runs_dir, port=port, host=args.host)
    if args.dry_run:
        print(f"[TensorBoard] Comando: {' '.join(cmd)}")
        return 0

    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n[TensorBoard] Servidor detenido.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
