from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import torch

from burn_scar.tracking import (
    ExperimentTracker,
    build_run_namespace,
    load_dotenv_file,
    resolve_storage_paths,
)


def test_load_dotenv_file_parses_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comentario\n"
        "BURNSCAR_SHARED_STORAGE=/mnt/shared/nfs\n"
        'BURNSCAR_USER="antonio"\n'
        "BURNSCAR_TENSORBOARD_PORT=6007\n",
        encoding="utf-8",
    )
    parsed = load_dotenv_file(env_file)
    assert parsed["BURNSCAR_SHARED_STORAGE"] == "/mnt/shared/nfs"
    assert parsed["BURNSCAR_USER"] == "antonio"
    assert parsed["BURNSCAR_TENSORBOARD_PORT"] == "6007"


def test_resolve_storage_paths_shared_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BURNSCAR_SHARED_STORAGE", raising=False)
    monkeypatch.delenv("BURNSCAR_USER", raising=False)

    fake_nfs = tmp_path / "TK_Stratego" / "Shared"
    fake_nfs.mkdir(parents=True)

    storage = resolve_storage_paths(
        env_file=tmp_path / "nonexistent.env",
        env_overrides={
            "BURNSCAR_SHARED_STORAGE": str(fake_nfs),
            "BURNSCAR_USER": "antonio",
            "BURNSCAR_TENSORBOARD_PORT": "6010",
        },
        fallback_root=tmp_path / "local",
    )

    assert storage.is_shared is True
    assert storage.user == "antonio"
    assert storage.tensorboard_port == 6010
    assert storage.base_dir == fake_nfs / "burn_scar"
    assert storage.runs_dir == fake_nfs / "burn_scar" / "runs"
    assert storage.checkpoints_dir == fake_nfs / "burn_scar" / "checkpoints"
    assert storage.runs_dir.is_dir()
    assert storage.checkpoints_dir.is_dir()


def test_resolve_storage_paths_fallback_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BURNSCAR_SHARED_STORAGE", raising=False)
    monkeypatch.delenv("BURNSCAR_USER", raising=False)

    local_root = tmp_path / "workspace"
    storage = resolve_storage_paths(
        env_file=tmp_path / "empty.env",
        env_overrides={"BURNSCAR_SHARED_STORAGE": "", "BURNSCAR_USER": "dev2"},
        fallback_root=local_root,
    )

    assert storage.is_shared is False
    assert storage.user == "dev2"
    assert storage.runs_dir == local_root / "runs"
    assert storage.checkpoints_dir == local_root / "checkpoints"
    assert storage.runs_dir.is_dir()
    assert storage.checkpoints_dir.is_dir()


def test_resolve_storage_paths_warns_and_falls_back_when_mount_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BURNSCAR_SHARED_STORAGE", raising=False)
    local_root = tmp_path / "fallback"

    with pytest.warns(RuntimeWarning, match="no existe o no está montada"):
        storage = resolve_storage_paths(
            env_file=tmp_path / "empty.env",
            env_overrides={"BURNSCAR_SHARED_STORAGE": str(tmp_path / "missing_nfs")},
            fallback_root=local_root,
        )

    assert storage.is_shared is False
    assert storage.runs_dir == local_root / "runs"


def test_build_run_namespace_formatting() -> None:
    fixed_dt = datetime(2026, 11, 7, 17, 30, 0)
    ns = build_run_namespace("baseline unet/v1", user="antonio", timestamp=fixed_dt)
    assert ns == "antonio_baseline_unet_v1_20261107_173000"


def test_concurrent_researchers_shared_tracking(tmp_path: Path) -> None:
    """Simula dos investigadores con puntos de montaje distintos al directorio compartido."""
    shared_server = tmp_path / "nfs_server_storage"
    shared_server.mkdir(parents=True)

    # Simulamos rutas de montaje distintas con symlinks (como en máquinas distintas)
    mount_antonio = tmp_path / "home_antonio" / "TK_Stratego" / "Shared"
    mount_antonio.parent.mkdir(parents=True)
    mount_antonio.symlink_to(shared_server, target_is_directory=True)

    mount_colaborador = tmp_path / "home_colaborador" / "otra_ruta" / "TK_Stratego" / "Shared"
    mount_colaborador.parent.mkdir(parents=True)
    mount_colaborador.symlink_to(shared_server, target_is_directory=True)

    storage_antonio = resolve_storage_paths(
        env_file=tmp_path / "none.env",
        env_overrides={
            "BURNSCAR_SHARED_STORAGE": str(mount_antonio),
            "BURNSCAR_USER": "antonio",
        },
    )
    storage_colaborador = resolve_storage_paths(
        env_file=tmp_path / "none.env",
        env_overrides={
            "BURNSCAR_SHARED_STORAGE": str(mount_colaborador),
            "BURNSCAR_USER": "colaborador",
        },
    )

    with (
        ExperimentTracker(
            "b0_advection",
            timestamp="20261107_180000",
            storage=storage_antonio,
        ) as tracker_a,
        ExperimentTracker(
            "b1_unet",
            timestamp="20261107_180000",
            storage=storage_colaborador,
        ) as tracker_b,
    ):
        for step in range(3):
            tracker_a.log_scalar("Loss/train", 1.0 / (step + 1), global_step=step)
            tracker_b.log_metrics(
                {"Loss/train": 0.8 / (step + 1), "IoU/val": 0.4 + 0.1 * step},
                global_step=step,
            )

        ckpt_a = tracker_a.save_checkpoint({"epoch": 2, "weights": torch.ones(4)}, "best.pt")
        ckpt_b = tracker_b.save_checkpoint({"epoch": 2, "weights": torch.zeros(4)}, "best.pt")

    # Verificamos en el servidor NFS que ambos experimentos coexisten sin colisión
    shared_runs = shared_server / "burn_scar" / "runs"
    run_dirs = sorted(p.name for p in shared_runs.iterdir() if p.is_dir())
    assert run_dirs == [
        "antonio_b0_advection_20261107_180000",
        "colaborador_b1_unet_20261107_180000",
    ]

    for run_name in run_dirs:
        events = list((shared_runs / run_name).glob("events.out.tfevents.*"))
        assert len(events) >= 1
        assert events[0].stat().st_size > 0

    assert ckpt_a.is_file()
    assert ckpt_b.is_file()
    assert ckpt_a.resolve() != ckpt_b.resolve()


def test_log_image_writes_and_flushes(tmp_path: Path) -> None:
    storage = resolve_storage_paths(
        env_file=tmp_path / "none.env",
        env_overrides={"BURNSCAR_SHARED_STORAGE": "", "BURNSCAR_USER": "antonio"},
        fallback_root=tmp_path / "local",
    )
    img = torch.rand(3, 16, 16, dtype=torch.float32)

    with ExperimentTracker("vis_test", timestamp="20261114_170000", storage=storage) as tracker:
        flush_calls = 0
        orig_flush = tracker.flush

        def counted_flush() -> None:
            nonlocal flush_calls
            flush_calls += 1
            orig_flush()

        tracker.flush = counted_flush  # type: ignore[method-assign]
        tracker.log_image("fold_0/sample_0", img, global_step=1)
        assert flush_calls == 1

    events = list(tracker.run_dir.glob("events.out.tfevents.*"))
    assert len(events) >= 1
    assert events[0].stat().st_size > 100

