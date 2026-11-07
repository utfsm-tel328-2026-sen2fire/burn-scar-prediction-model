"""Resolución de almacenamiento compartido (NFS/local) y tracking colaborativo con TensorBoard."""

from __future__ import annotations

import getpass
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

SHARED_DIR_MODE = 0o775
SHARED_FILE_MODE = 0o664
DEFAULT_TENSORBOARD_PORT = 6006


@dataclass(frozen=True)
class StoragePaths:
    """Rutas resueltas para logs de TensorBoard y checkpoints de modelos."""

    base_dir: Path
    runs_dir: Path
    checkpoints_dir: Path
    is_shared: bool
    user: str
    tensorboard_port: int


def load_dotenv_file(env_path: Path | None = None) -> dict[str, str]:
    """Carga pares KEY=VALUE desde un archivo `.env` sin dependencias externas."""
    path = env_path if env_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        key = key.strip()
        val = raw_val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        values[key] = val
    return values


def ensure_shared_dir(path: Path, mode: int = SHARED_DIR_MODE) -> Path:
    """Crea un directorio (y sus padres) intentando asegurar permisos de grupo colaborativos."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(mode)
    except OSError:
        # Si el directorio pertenece a otro usuario del grupo en NFS, chmod puede fallar;
        # mientras sea escribible, se continúa normalmente.
        pass
    return path


def _sanitize_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", token.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def build_run_namespace(
    experiment_tag: str,
    user: str | None = None,
    timestamp: datetime | str | None = None,
) -> str:
    """Construye un identificador aislado `<user>_<experiment_tag>_<YYYYMMDD_HHMMSS>`."""
    resolved_user = _sanitize_token(user or getpass.getuser() or "researcher")
    resolved_tag = _sanitize_token(experiment_tag)
    if timestamp is None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif isinstance(timestamp, datetime):
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    else:
        ts_str = _sanitize_token(timestamp)
    return f"{resolved_user}_{resolved_tag}_{ts_str}"


def resolve_storage_paths(
    env_file: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
    fallback_root: Path | None = None,
) -> StoragePaths:
    """Resuelve las rutas de `runs/` y `checkpoints/` usando `.env` / entorno o fallback local."""
    merged_env: dict[str, str] = {}
    merged_env.update(load_dotenv_file(env_file))
    merged_env.update(os.environ)
    if env_overrides is not None:
        merged_env.update(env_overrides)

    raw_shared = merged_env.get("BURNSCAR_SHARED_STORAGE", "").strip()
    raw_user = merged_env.get("BURNSCAR_USER", "").strip()
    raw_port = merged_env.get("BURNSCAR_TENSORBOARD_PORT", "").strip()

    user = _sanitize_token(raw_user) if raw_user else _sanitize_token(getpass.getuser())
    try:
        port = int(raw_port) if raw_port else DEFAULT_TENSORBOARD_PORT
    except ValueError:
        port = DEFAULT_TENSORBOARD_PORT

    local_root = (fallback_root if fallback_root is not None else Path.cwd()).resolve()

    if raw_shared:
        shared_mount = Path(raw_shared).expanduser()
        if shared_mount.is_dir():
            base_dir = ensure_shared_dir(shared_mount / "burn_scar")
            runs_dir = ensure_shared_dir(base_dir / "runs")
            checkpoints_dir = ensure_shared_dir(base_dir / "checkpoints")
            return StoragePaths(
                base_dir=base_dir,
                runs_dir=runs_dir,
                checkpoints_dir=checkpoints_dir,
                is_shared=True,
                user=user,
                tensorboard_port=port,
            )
        warnings.warn(
            f"La ruta compartida configurada ({shared_mount}) no existe o no está montada. "
            f"Usando fallback local en {local_root}.",
            RuntimeWarning,
            stacklevel=2,
        )

    runs_dir = ensure_shared_dir(local_root / "runs")
    checkpoints_dir = ensure_shared_dir(local_root / "checkpoints")
    return StoragePaths(
        base_dir=local_root,
        runs_dir=runs_dir,
        checkpoints_dir=checkpoints_dir,
        is_shared=False,
        user=user,
        tensorboard_port=port,
    )


class ExperimentTracker:
    """Envoltorio colaborativo sobre SummaryWriter con vaciado inmediato a NFS y checkpoints."""

    def __init__(
        self,
        experiment_tag: str,
        *,
        user: str | None = None,
        timestamp: datetime | str | None = None,
        storage: StoragePaths | None = None,
        flush_always: bool = True,
    ) -> None:
        self.storage = storage if storage is not None else resolve_storage_paths()
        effective_user = user if user is not None else self.storage.user
        self.namespace = build_run_namespace(
            experiment_tag=experiment_tag,
            user=effective_user,
            timestamp=timestamp,
        )
        self.run_dir = ensure_shared_dir(self.storage.runs_dir / self.namespace)
        self.checkpoint_dir = ensure_shared_dir(self.storage.checkpoints_dir / self.namespace)
        self.flush_always = flush_always
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self._apply_event_file_permissions()

    def _apply_event_file_permissions(self) -> None:
        for event_file in self.run_dir.glob("events.out.tfevents.*"):
            try:
                event_file.chmod(SHARED_FILE_MODE)
            except OSError:
                pass

    def log_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        *,
        flush: bool | None = None,
    ) -> None:
        """Registra un escalar en TensorBoard y fuerza `flush()` para sincronizar en NFS."""
        self.writer.add_scalar(tag, scalar_value, global_step=global_step)
        should_flush = self.flush_always if flush is None else flush
        if should_flush:
            self.flush()

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        global_step: int | None = None,
        *,
        flush: bool | None = None,
    ) -> None:
        """Registra múltiples escalares en un mismo paso y vacía el búfer al final."""
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, global_step=global_step)
        should_flush = self.flush_always if flush is None else flush
        if should_flush:
            self.flush()

    def save_checkpoint(self, state: Any, filename: str = "checkpoint.pt") -> Path:
        """Guarda un diccionario de pesos/estado en la carpeta aislada del experimento."""
        target = self.checkpoint_dir / filename
        torch.save(state, target)
        try:
            target.chmod(SHARED_FILE_MODE)
        except OSError:
            pass
        return target

    def flush(self) -> None:
        """Fuerza el vaciado inmediato de eventos a disco/NFS."""
        self.writer.flush()
        self._apply_event_file_permissions()

    def close(self) -> None:
        """Vacía y cierra el SummaryWriter subyacente."""
        self.flush()
        self.writer.close()

    def __enter__(self) -> ExperimentTracker:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
