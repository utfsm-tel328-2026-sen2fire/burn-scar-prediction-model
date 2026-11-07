"""Verifica el entorno: acelerador, cómputo en GPU y escritura de logs de TensorBoard."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def detect_backend() -> str:
    """Devuelve 'cuda', 'rocm' o 'cpu' según el build de torch y el dispositivo disponible."""
    if not torch.cuda.is_available():
        return "cpu"
    return "rocm" if torch.version.hip else "cuda"


def check_accelerator() -> bool:
    backend = detect_backend()
    print(f"torch={torch.__version__} backend={backend}")
    if backend == "cpu":
        print("AVISO: no hay dispositivo CUDA/ROCm disponible; el backend activo es CPU.")
        return True

    print(f"cuda_available=True device={torch.cuda.get_device_name(0)}")
    print(f"arch_list={torch.cuda.get_arch_list()}")
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    result = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"matmul en GPU OK (sum={result:.3f})")
    return True


def check_tensorboard(base_dir: Path) -> bool:
    log_dir = base_dir / "check_env"
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("test/loss", 0.5, 1)
    writer.close()
    events = list(log_dir.glob("events.out.tfevents.*"))
    print(f"tensorboard: {len(events)} archivo(s) de evento en {log_dir}")
    return len(events) > 0


def main() -> int:
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    ok = check_accelerator()
    ok = check_tensorboard(runs_dir) and ok
    print("ENTORNO OK" if ok else "ENTORNO CON PROBLEMAS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
