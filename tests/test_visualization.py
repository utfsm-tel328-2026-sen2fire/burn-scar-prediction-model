from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from burn_scar.dataset import compute_generator_version
from burn_scar.visualization import (
    EvaluatedSample,
    build_comparison_panel,
    evaluate_validation_samples,
    export_panel_png,
    infer_hidden_ch_from_state_dict,
    informative_metric_mean,
    load_propagation_checkpoint,
    render_overlay_rgb,
    select_samples,
    split_informative_samples,
)


def _make_dummy_sample(
    idx: int,
    *,
    series_id: str = "EMSR001_01",
    f1: float = 0.5,
    iou: float = 0.35,
    burned_px: int = 20,
    horizon_days: float = 3.0,
    H: int = 16,
    W: int = 16,
) -> EvaluatedSample:
    x_olci = torch.rand(7, 5, H, W, dtype=torch.float32)
    x_scar = torch.zeros(1, H, W, dtype=torch.float32)
    x_scar[:, :4, :4] = 1.0
    target = torch.zeros(1, H, W, dtype=torch.float32)
    target[:, 4:8, 4:8] = 1.0
    probs = torch.clamp(target * f1 + 0.1, 0.0, 1.0)
    mask = torch.ones(1, H, W, dtype=torch.uint8)
    mask[:, -2:, -2:] = 255

    return EvaluatedSample(
        dataset_idx=idx,
        sample_id=f"sample_{idx:03d}",
        series_id=series_id,
        fold=0,
        horizon_days=horizon_days,
        f1=f1,
        iou=iou,
        precision=f1,
        recall=f1,
        target_burned_px=burned_px,
        x_olci=x_olci,
        x_scar=x_scar,
        target=target,
        probs=probs,
        mask=mask,
    )


def test_build_comparison_panel_dimensions_and_range() -> None:
    s = _make_dummy_sample(0, H=16, W=16)
    scale = 3
    sep_width = 2
    panel = build_comparison_panel(
        s.x_olci,
        s.x_scar,
        s.target,
        s.probs,
        s.mask,
        threshold=0.5,
        scale=scale,
        sep_width=sep_width,
    )

    w_total = 5 * 16 + 4 * sep_width
    assert panel.shape == (3, 16 * scale, w_total * scale)
    assert panel.dtype == torch.float32
    assert float(panel.min()) >= 0.0
    assert float(panel.max()) <= 1.0


def test_render_overlay_rgb_distinguishes_tp_fp_fn_and_ignored() -> None:
    H, W = 4, 4
    probs = torch.zeros(1, H, W)
    target = torch.zeros(1, H, W)
    x_scar = torch.zeros(1, H, W)
    mask = torch.ones(1, H, W, dtype=torch.uint8)

    # (0, 0) = TP
    probs[0, 0, 0] = 0.9
    target[0, 0, 0] = 1.0
    # (0, 1) = FP
    probs[0, 0, 1] = 0.9
    target[0, 0, 1] = 0.0
    # (0, 2) = FN
    probs[0, 0, 2] = 0.1
    target[0, 0, 2] = 1.0
    # (0, 3) = Ignored (mask == 255) even with high prob
    probs[0, 0, 3] = 0.95
    target[0, 0, 3] = 1.0
    mask[0, 0, 3] = 255

    ovl = render_overlay_rgb(probs, target, x_scar, mask, threshold=0.5)
    assert ovl.shape == (3, H, W)
    # TP -> green channel dominant
    assert ovl[1, 0, 0] > ovl[0, 0, 0] and ovl[1, 0, 0] > ovl[2, 0, 0]
    # FP -> red channel dominant
    assert ovl[0, 0, 1] > ovl[1, 0, 1] and ovl[0, 0, 1] > ovl[2, 0, 1]
    # FN -> yellow (red & green high, blue low)
    assert ovl[0, 0, 2] > 0.7 and ovl[1, 0, 2] > 0.7 and ovl[2, 0, 2] < 0.3
    # Ignored -> attenuated gray (~0.25)
    assert torch.allclose(ovl[:, 0, 3], torch.tensor([0.25, 0.25, 0.28]), atol=1e-2)


def test_select_samples_modes_and_export_png(tmp_path: Path) -> None:
    samples = [
        _make_dummy_sample(0, series_id="EMSR_A", f1=0.15, iou=0.10),
        _make_dummy_sample(1, series_id="EMSR_A", f1=0.85, iou=0.75),
        _make_dummy_sample(2, series_id="EMSR_B", f1=0.45, iou=0.30),
        _make_dummy_sample(3, series_id="EMSR_B", f1=0.92, iou=0.86),
    ]

    best = select_samples(samples, mode="best", n_samples=2)
    assert [s.dataset_idx for _, s in best] == [3, 1]
    assert all(g == "best" for g, _ in best)

    worst = select_samples(samples, mode="worst", n_samples=2)
    assert [s.dataset_idx for _, s in worst] == [0, 2]
    assert all(g == "worst" for g, _ in worst)

    both = select_samples(samples, mode="both", n_samples=1)
    assert [(g, s.dataset_idx) for g, s in both] == [("best", 3), ("worst", 0)]

    by_series = select_samples(samples, mode="series", series_id="EMSR_A", n_samples=5)
    assert [s.dataset_idx for _, s in by_series] == [0, 1]

    by_indices = select_samples(samples, mode="indices", indices=[2, 0])
    assert [s.dataset_idx for _, s in by_indices] == [2, 0]

    rnd = select_samples(samples, mode="random", n_samples=2, seed=123)
    assert len(rnd) == 2

    panel = build_comparison_panel(
        samples[3].x_olci,
        samples[3].x_scar,
        samples[3].target,
        samples[3].probs,
        samples[3].mask,
        scale=2,
    )
    png_path = export_panel_png(
        panel, tmp_path / "exports", group="best", rank=1, sample=samples[3]
    )
    assert png_path.is_file()
    assert png_path.suffix == ".png"
    assert png_path.stat().st_size > 100


def test_ranking_excluye_muestras_no_informativas() -> None:
    """Las muestras con F1 = NaN (verdad sin quemado) no ocupan puesto en el ranking."""
    samples = [
        _make_dummy_sample(0, f1=0.15, iou=0.10),
        _make_dummy_sample(1, f1=float("nan"), iou=float("nan")),
        _make_dummy_sample(2, f1=0.85, iou=0.75),
    ]

    informative, excluded = split_informative_samples(samples)
    assert len(informative) == 2
    assert excluded == 1

    best = select_samples(samples, mode="best", n_samples=2)
    assert [s.dataset_idx for _, s in best] == [2, 0]

    worst = select_samples(samples, mode="worst", n_samples=2)
    assert [s.dataset_idx for _, s in worst] == [0, 2]


def test_promedio_ignora_muestras_no_informativas() -> None:
    """Una sola muestra vacía no anula el promedio de F1 del fold."""
    samples = [
        _make_dummy_sample(0, f1=0.4, iou=0.2),
        _make_dummy_sample(1, f1=float("nan"), iou=float("nan")),
    ]

    assert informative_metric_mean(samples, "f1") == pytest.approx(0.4)
    assert informative_metric_mean(samples, "iou") == pytest.approx(0.2)


def test_promedio_sin_muestras_informativas_es_nan() -> None:
    """Sin ninguna muestra informativa el promedio es NaN, no 0."""
    samples = [_make_dummy_sample(0, f1=float("nan"), iou=float("nan"))]

    assert np.isnan(informative_metric_mean(samples, "f1"))


def test_muestras_con_quemado_se_evaluan_con_el_umbral_de_decision() -> None:
    """La condición de "muestra con quemado" usa el umbral de evaluación, no 0,5 fijo."""
    from burn_scar.models.propagation import BurnScarPropagationModel

    H = W = 64
    target = torch.zeros(1, 1, H, W)
    target[0, 0, 0, 0] = 0.3  # quemado por debajo de 0,5 pero por encima de 0,1

    batch = {
        "x_olci": torch.zeros(1, 7, 5, H, W),
        "x_dem": torch.zeros(1, 1, H, W),
        "x_worldcover": torch.zeros(1, 1, H, W),
        "x_scar": torch.zeros(1, 1, H, W),
        "horizon_days": torch.ones(1, 1),
        "target": target,
        "mask": torch.ones(1, 1, H, W, dtype=torch.uint8),
        "meta": {"sample_id": ["m1"], "series_id": ["serie_a"], "fold": [0]},
    }
    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=8, scar_corruption_prob=0.0)

    con_umbral_bajo = evaluate_validation_samples(model, [batch], device="cpu", threshold=0.1)
    assert con_umbral_bajo[0].target_burned_px == 1

    con_umbral_alto = evaluate_validation_samples(model, [batch], device="cpu", threshold=0.5)
    assert con_umbral_alto[0].target_burned_px == 0


def test_checkpoint_loading_and_validation_errors(tmp_path: Path) -> None:
    from burn_scar.models.propagation import BurnScarPropagationModel

    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=16, scar_corruption_prob=0.0)
    state_dict = model.state_dict()
    assert infer_hidden_ch_from_state_dict(state_dict) == 16

    ckpt_file = tmp_path / "best_fold_2.pt"
    torch.save({"fold": 2, "epoch": 4, "model_state_dict": state_dict, "metrics": {}}, ckpt_file)

    loaded_model, resolved_fold, _ = load_propagation_checkpoint(ckpt_file)
    assert resolved_fold == 2
    assert loaded_model.hidden_ch == 16
    assert not loaded_model.training

    # Error si hidden_ch explícito no coincide
    with pytest.raises(ValueError, match="no coincide con el checkpoint"):
        load_propagation_checkpoint(ckpt_file, hidden_ch=32)

    # Error si falta fold en el checkpoint y no se pasa --fold
    ckpt_no_fold = tmp_path / "no_fold.pt"
    torch.save({"model_state_dict": state_dict}, ckpt_no_fold)
    with pytest.raises(ValueError, match="no registra 'fold'"):
        load_propagation_checkpoint(ckpt_no_fold)


def test_visualize_predictions_cli_end_to_end(tmp_path: Path) -> None:
    import importlib.util
    import json
    import subprocess
    import sys

    import numpy as np

    from burn_scar.models.propagation import BurnScarPropagationModel
    from burn_scar.tracking import resolve_storage_paths

    # 1. Verificar ejecución de --help y cargar el módulo del script
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_predictions.py"
    spec = importlib.util.spec_from_file_location("visualize_predictions", script_path)
    assert spec is not None and spec.loader is not None
    vis_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis_mod)
    run_visualization = vis_mod.run_visualization

    help_proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--checkpoint" in help_proc.stdout
    assert "--save-dir" in help_proc.stdout

    # 2. Crear dataset sintético en tmp_path (fold 0 = val, fold 1 = train)
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir(parents=True)
    index_entries = []

    for i, fold in enumerate([0, 0, 1]):
        npz_path = samples_dir / f"sample_{i:03d}.npz"
        target = np.zeros((16, 16), dtype=np.float32)
        target[4:8, 4:8] = 1.0
        mask = np.ones((16, 16), dtype=np.uint8)
        mask[-2:, -2:] = 255
        np.savez_compressed(
            npz_path,
            x_olci=np.random.uniform(0, 500, size=(7, 5, 16, 16)).astype(np.float32),
            x_dem=np.random.uniform(0, 1000, size=(16, 16)).astype(np.float32),
            x_worldcover=np.full((16, 16), 10.0, dtype=np.float32),
            x_scar=np.zeros((16, 16), dtype=np.float32),
            horizon_days=np.float32(3.0),
            target=target,
            mask=mask,
        )
        index_entries.append(
            {
                "sample_id": f"sample_{i:03d}",
                "series_id": f"EMSR_TEST_{i % 2}",
                "fold": fold,
                "file": str(npz_path),
            }
        )

    index_file = tmp_path / "dataset_index.json"
    index_file.write_text(
        json.dumps({"generator_version": compute_generator_version(), "samples": index_entries}),
        encoding="utf-8",
    )

    # 3. Guardar checkpoint sintético para fold 0 con hidden_ch=16
    model = BurnScarPropagationModel(olci_channels=5, hidden_ch=16, scar_corruption_prob=0.0)
    ckpt_path = tmp_path / "best_fold_0.pt"
    torch.save(
        {
            "fold": 0,
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "metrics": {"f1_event": 0.5},
        },
        ckpt_path,
    )

    storage = resolve_storage_paths(
        env_file=tmp_path / "none.env",
        env_overrides={"BURNSCAR_SHARED_STORAGE": "", "BURNSCAR_USER": "tester"},
        fallback_root=tmp_path / "tracking_root",
    )
    png_out = tmp_path / "exported_pngs"

    rc = run_visualization(
        [
            "--checkpoint",
            str(ckpt_path),
            "--index",
            str(index_file),
            "--mode",
            "both",
            "--n-samples",
            "1",
            "--scale",
            "2",
            "--device",
            "cpu",
            "--tag",
            "e2e_vis",
            "--save-dir",
            str(png_out),
        ],
        storage=storage,
    )
    assert rc == 0

    png_files = sorted(png_out.glob("*.png"))
    assert len(png_files) == 2
    run_dirs = list((tmp_path / "tracking_root" / "runs").glob("tester_e2e_vis_*"))
    assert len(run_dirs) == 1
    events = list(run_dirs[0].glob("events.out.tfevents.*"))
    assert len(events) >= 1
    assert events[0].stat().st_size > 100

