"""
Utilidades de visualización e inspección post-entrenamiento para el modelo de propagación.

Construye paneles comparativos RGB de 5 bloques por muestra:
    [ OLCI pseudo-RGB | x_scar | target | probs | overlay TP/FP/FN ]
respetando la máscara de validez (`mask == 1` válido, `mask == 255` ignorado/atenuado),
permite filtrar/ordenar muestras (`best`, `worst`, `both`, `series`, `indices`, `random`)
y exportar paneles tanto a TensorBoard (pestaña IMAGES) como a archivos PNG.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import save_image

from burn_scar.baselines import compute_metrics_numpy
from burn_scar.models.propagation import BurnScarPropagationModel

VALID_SELECTION_MODES = ("best", "worst", "both", "series", "indices", "random")


@dataclass
class EvaluatedSample:
    """Resultado de inferencia y métricas para una muestra individual de validación."""

    dataset_idx: int
    sample_id: str
    series_id: str
    fold: int
    horizon_days: float
    f1: float
    iou: float
    precision: float
    recall: float
    target_burned_px: int
    x_olci: Tensor
    x_scar: Tensor
    target: Tensor
    probs: Tensor
    mask: Tensor


def _to_2d_float(tensor: Tensor | np.ndarray) -> Tensor:
    t = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu()
    while t.ndim > 2:
        t = t.squeeze(0)
    return t


def render_olci_rgb(x_olci: Tensor | np.ndarray, mask: Tensor | np.ndarray) -> Tensor:
    """
    Convierte el último frame OLCI `(5, H, W)` (o `(T, 5, H, W)`) en pseudo-RGB `(3, H, W)`.
    Atenúa las zonas donde `mask == 255`.
    """
    olci = torch.as_tensor(x_olci, dtype=torch.float32).detach().cpu()
    if olci.ndim == 4:
        olci = olci[-1]
    if olci.ndim != 3 or olci.shape[0] < 3:
        raise ValueError(
            f"Se esperaba x_olci con forma (C>=3, H, W), recibido: {tuple(olci.shape)}"
        )

    # Bandas (2, 1, 0) como composición falso color / pseudo-RGB
    rgb = olci[[2, 1, 0]].clone()
    msk = _to_2d_float(mask)
    valid = msk == 1

    for c in range(3):
        ch = rgb[c]
        valid_vals = ch[valid] if valid.any() else ch.reshape(-1)
        lo = torch.quantile(valid_vals, 0.02) if valid_vals.numel() > 1 else valid_vals.min()
        hi = torch.quantile(valid_vals, 0.98) if valid_vals.numel() > 1 else valid_vals.max()
        denom = (hi - lo).clamp(min=1e-6)
        rgb[c] = ((ch - lo) / denom).clamp(0.0, 1.0)

    invalid = (msk == 255).unsqueeze(0).expand_as(rgb)
    gray_bg = torch.full_like(rgb, 0.25)
    rgb = torch.where(invalid, 0.35 * rgb + 0.65 * gray_bg, rgb)
    return rgb.clamp(0.0, 1.0)


def render_scar_rgb(x_scar: Tensor | np.ndarray, mask: Tensor | np.ndarray) -> Tensor:
    """Renderiza la cicatriz conocida `x_scar` en azul acero sobre fondo oscuro `(3, H, W)`."""
    scar = _to_2d_float(x_scar).clamp(0.0, 1.0)
    msk = _to_2d_float(mask)
    H, W = scar.shape

    bg = torch.tensor([0.08, 0.10, 0.14], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    fg = torch.tensor([0.25, 0.72, 0.96], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    rgb = bg * (1.0 - scar.unsqueeze(0)) + fg * scar.unsqueeze(0)

    invalid = (msk == 255).unsqueeze(0).expand(3, H, W)
    gray = torch.full_like(rgb, 0.25)
    return torch.where(invalid, 0.35 * rgb + 0.65 * gray, rgb).clamp(0.0, 1.0)


def render_target_rgb(
    target: Tensor | np.ndarray,
    x_scar: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
) -> Tensor:
    """
    Renderiza el ground truth `target` (nueva quema en verde brillante, cicatriz previa en azul
    y píxeles ignorados `mask == 255` en gris atenuado).
    """
    tgt = (_to_2d_float(target) >= 0.5).float()
    scar = (_to_2d_float(x_scar) >= 0.5).float()
    msk = _to_2d_float(mask)
    H, W = tgt.shape

    base_bg = torch.tensor([0.08, 0.10, 0.12], dtype=torch.float32).view(3, 1, 1)
    rgb = base_bg.expand(3, H, W).clone()
    scar_color = torch.tensor([0.16, 0.34, 0.52], dtype=torch.float32).view(3, 1, 1)
    tgt_color = torch.tensor([0.15, 0.90, 0.35], dtype=torch.float32).view(3, 1, 1)

    rgb = torch.where(scar.unsqueeze(0) > 0.5, scar_color.expand(3, H, W), rgb)
    rgb = torch.where(tgt.unsqueeze(0) > 0.5, tgt_color.expand(3, H, W), rgb)

    invalid = (msk == 255).unsqueeze(0).expand(3, H, W)
    gray = torch.full_like(rgb, 0.25)
    return torch.where(invalid, gray, rgb).clamp(0.0, 1.0)


def render_probs_rgb(probs: Tensor | np.ndarray, mask: Tensor | np.ndarray) -> Tensor:
    """
    Renderiza el mapa continuo de probabilidad `[0, 1]` como mapa de calor
    y atenúa los píxeles con `mask == 255`.
    """
    p = _to_2d_float(probs).clamp(0.0, 1.0)
    msk = _to_2d_float(mask)
    H, W = p.shape

    r = (0.08 + 1.8 * p).clamp(0.0, 1.0)
    g = (0.08 + 1.8 * (p - 0.35).clamp(min=0.0)).clamp(0.0, 1.0)
    b = (0.14 * (1.0 - p) + 1.6 * (p - 0.75).clamp(min=0.0)).clamp(0.0, 1.0)
    rgb = torch.stack([r, g, b], dim=0)

    invalid = (msk == 255).unsqueeze(0).expand(3, H, W)
    gray = torch.full_like(rgb, 0.25)
    return torch.where(invalid, 0.35 * rgb + 0.65 * gray, rgb).clamp(0.0, 1.0)


def render_overlay_rgb(
    probs: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    x_scar: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
) -> Tensor:
    """
    Construye el mapa diagnóstico de aciertos/errores:
      - Azul acero: cicatriz previa (`x_scar`)
      - Verde:      Verdadero Positivo (TP)
      - Rojo:       Falso Positivo (FP)
      - Amarillo:   Falso Negativo (FN)
      - Gris:       Píxel ignorado (`mask == 255`)
    """
    p_bin = _to_2d_float(probs) >= threshold
    t_bin = _to_2d_float(target) >= 0.5
    s_bin = _to_2d_float(x_scar) >= 0.5
    msk = _to_2d_float(mask)
    valid = msk == 1
    invalid = msk == 255
    H, W = p_bin.shape

    base_bg = torch.tensor([0.08, 0.10, 0.12], dtype=torch.float32).view(3, 1, 1)
    rgb = base_bg.expand(3, H, W).clone()

    scar_color = torch.tensor([0.20, 0.48, 0.78], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    tp_color = torch.tensor([0.15, 0.90, 0.30], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    fp_color = torch.tensor([0.95, 0.20, 0.20], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    fn_color = torch.tensor([0.98, 0.82, 0.15], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    ign_color = torch.tensor([0.25, 0.25, 0.28], dtype=torch.float32).view(3, 1, 1).expand(3, H, W)

    rgb = torch.where(s_bin.unsqueeze(0), scar_color, rgb)
    tp = (p_bin & t_bin & valid).unsqueeze(0)
    fp = (p_bin & (~t_bin) & valid).unsqueeze(0)
    fn = ((~p_bin) & t_bin & valid).unsqueeze(0)

    rgb = torch.where(fn, fn_color, rgb)
    rgb = torch.where(fp, fp_color, rgb)
    rgb = torch.where(tp, tp_color, rgb)
    rgb = torch.where(invalid.unsqueeze(0), ign_color, rgb)
    return rgb.clamp(0.0, 1.0)


def build_comparison_panel(
    x_olci: Tensor | np.ndarray,
    x_scar: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    probs: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
    scale: int = 3,
    sep_width: int = 2,
) -> Tensor:
    """
    Compone el panel horizontal de 5 bloques:
        [ OLCI pseudo-RGB | x_scar | target | probs | overlay TP/FP/FN ]
    separados por `sep_width` píxeles y escalado por factor entero `scale` (nearest-neighbor).

    Retorna:
        Tensor float32 `(3, H * scale, W_total * scale)` en `[0, 1]`,
        donde `W_total = 5 * W + 4 * sep_width`.
    """
    if scale < 1:
        raise ValueError(f"scale debe ser >= 1, recibido: {scale}")
    if sep_width < 0:
        raise ValueError(f"sep_width debe ser >= 0, recibido: {sep_width}")

    b_olci = render_olci_rgb(x_olci, mask)
    b_scar = render_scar_rgb(x_scar, mask)
    b_tgt = render_target_rgb(target, x_scar, mask)
    b_prob = render_probs_rgb(probs, mask)
    b_ovl = render_overlay_rgb(probs, target, x_scar, mask, threshold=threshold)

    blocks = [b_olci, b_scar, b_tgt, b_prob, b_ovl]
    _, H, _ = b_olci.shape

    if sep_width > 0:
        sep = torch.full((3, H, sep_width), 0.85, dtype=torch.float32)
        parts: list[Tensor] = []
        for idx, blk in enumerate(blocks):
            if idx > 0:
                parts.append(sep)
            parts.append(blk)
        panel = torch.cat(parts, dim=2)
    else:
        panel = torch.cat(blocks, dim=2)

    if scale > 1:
        panel = F.interpolate(
            panel.unsqueeze(0),
            scale_factor=scale,
            mode="nearest",
        ).squeeze(0)

    return panel.clamp(0.0, 1.0)


def infer_hidden_ch_from_state_dict(state_dict: dict[str, Any]) -> int:
    """
    Infiere `hidden_ch` desde `temporal_encoder.conv.weight` (`shape = (4 * hidden_ch, ...)`).
    """
    key = "temporal_encoder.conv.weight"
    if key not in state_dict:
        raise KeyError(
            f"El checkpoint no contiene la clave '{key}' necesaria para inferir hidden_ch."
        )
    out_ch = int(state_dict[key].shape[0])
    if out_ch % 4 != 0:
        raise ValueError(f"Dimensión incompatible en '{key}': {out_ch} no es divisible por 4.")
    return out_ch // 4


def load_propagation_checkpoint(
    checkpoint_path: str | Path,
    *,
    fold: int | None = None,
    hidden_ch: int | None = None,
    device: str = "cpu",
) -> tuple[BurnScarPropagationModel, int, dict[str, Any]]:
    """
    Carga un checkpoint `best_fold_*.pt`, valida/infiere `fold` y `hidden_ch`,
    y retorna `(model_eval, resolved_fold, checkpoint_dict)`.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"El archivo {ckpt_path} no contiene un diccionario válido con 'model_state_dict'."
        )

    state_dict = ckpt["model_state_dict"]
    inferred_hidden = infer_hidden_ch_from_state_dict(state_dict)
    if hidden_ch is not None and hidden_ch != inferred_hidden:
        raise ValueError(
            f"El --hidden-ch={hidden_ch} indicado no coincide con el checkpoint "
            f"(hidden_ch={inferred_hidden})."
        )
    resolved_hidden = inferred_hidden if hidden_ch is None else hidden_ch

    resolved_fold = fold if fold is not None else ckpt.get("fold")
    if resolved_fold is None:
        raise ValueError(
            "El checkpoint no registra 'fold' y no se especificó --fold explícitamente."
        )
    resolved_fold = int(resolved_fold)
    if resolved_fold not in range(5):
        raise ValueError(f"El fold resuelto debe estar en [0..4], recibido: {resolved_fold}")

    model = BurnScarPropagationModel(
        olci_channels=5,
        hidden_ch=resolved_hidden,
        scar_corruption_prob=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolved_fold, ckpt


@torch.no_grad()
def evaluate_validation_samples(
    model: BurnScarPropagationModel,
    val_loader: torch.utils.data.DataLoader,
    *,
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[EvaluatedSample]:
    """
    Ejecuta inferencia sobre todas las muestras del `val_loader` y devuelve la lista
    de `EvaluatedSample` con métricas por muestra y tensores listos para visualizar.
    """
    model.eval()
    results: list[EvaluatedSample] = []
    running_idx = 0

    for batch in val_loader:
        x_olci = batch["x_olci"].to(device)
        x_dem = batch["x_dem"].to(device)
        x_wc = batch["x_worldcover"].to(device)
        x_scar = batch["x_scar"].to(device)
        h_days = batch["horizon_days"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        logits = model(x_olci, x_dem, x_wc, x_scar, h_days)
        probs = torch.sigmoid(logits)

        probs_cpu = probs.cpu()
        olci_cpu = x_olci.cpu()
        scar_cpu = x_scar.cpu()
        target_cpu = target.cpu()
        mask_cpu = mask.cpu()
        h_cpu = h_days.view(-1).cpu()

        meta = batch["meta"]
        batch_size = olci_cpu.shape[0]

        for i in range(batch_size):
            p_np = probs_cpu[i, 0].numpy()
            t_np = target_cpu[i, 0].numpy()
            m_np = mask_cpu[i, 0].numpy()
            metrics = compute_metrics_numpy(p_np, t_np, m_np, threshold=threshold)

            if "sample_id" in meta:
                sample_id = str(meta["sample_id"][i])
            else:
                sample_id = f"sample_{running_idx}"
            series_id = str(meta["series_id"][i]) if "series_id" in meta else "unknown"
            fold_val = int(meta["fold"][i]) if "fold" in meta else -1
            burned_px = int(((t_np >= threshold) & (m_np == 1)).sum())

            results.append(
                EvaluatedSample(
                    dataset_idx=running_idx,
                    sample_id=sample_id,
                    series_id=series_id,
                    fold=fold_val,
                    horizon_days=float(h_cpu[i].item()),
                    f1=float(metrics["f1"]),
                    iou=float(metrics["iou"]),
                    precision=float(metrics["precision"]),
                    recall=float(metrics["recall"]),
                    target_burned_px=burned_px,
                    x_olci=olci_cpu[i],
                    x_scar=scar_cpu[i],
                    target=target_cpu[i],
                    probs=probs_cpu[i],
                    mask=mask_cpu[i],
                )
            )
            running_idx += 1

    return results


def split_informative_samples(
    samples: Sequence[EvaluatedSample],
) -> tuple[list[EvaluatedSample], int]:
    """
    Separa las muestras con métrica informativa de las que no lo son.

    Una muestra es no informativa cuando su F1 es NaN, es decir cuando la verdad
    no contiene quemado por encima del umbral de evaluación (o no hay píxeles
    válidos). Devuelve las muestras informativas y cuántas se han descartado.
    """
    informative = [s for s in samples if not np.isnan(s.f1)]
    return informative, len(samples) - len(informative)


def informative_metric_mean(
    samples: Sequence[EvaluatedSample],
    field: str = "f1",
) -> float:
    """
    Promedio de una métrica sobre las muestras informativas (ignora las NaN).

    Devuelve NaN si no queda ninguna muestra informativa, para que el informe
    distinga "no evaluable" de un valor numérico.
    """
    informative, _ = split_informative_samples(samples)
    if not informative:
        return float("nan")
    values = np.array([getattr(s, field) for s in informative], dtype=float)
    return float(values.mean())


def select_samples(
    samples: Sequence[EvaluatedSample],
    *,
    mode: str = "both",
    n_samples: int = 8,
    series_id: str | None = None,
    indices: Sequence[int] | None = None,
    seed: int = 42,
) -> list[tuple[str, EvaluatedSample]]:
    """
    Filtra y ordena las muestras evaluadas según el modo elegido:
      - `best`:    las `n_samples` con mayor F1 (priorizando `target_burned_px > 0`).
      - `worst`:   las `n_samples` con menor F1 (priorizando `target_burned_px > 0`).
      - `both`:    hasta `n_samples` mejores y `n_samples` peores (`best` y `worst`).
      - `series`:  hasta `n_samples` muestras cuyo `series_id` coincida con `series_id`.
      - `indices`: las muestras cuyos `dataset_idx` estén en `indices`.
      - `random`:  muestreo aleatorio reproducible de hasta `n_samples` muestras.

    Las muestras con métrica no informativa (F1 = NaN) quedan fuera del ranking
    `best`/`worst`: sin quemado en la verdad su F1 no significa nada.

    Retorna una lista de pares `(group_label, EvaluatedSample)`.
    """
    if mode not in VALID_SELECTION_MODES:
        raise ValueError(f"Modo inválido '{mode}'. Opciones válidas: {VALID_SELECTION_MODES}")
    if not samples:
        return []

    if mode == "series":
        if not series_id:
            raise ValueError("El modo 'series' requiere especificar --series-id.")
        matched = [s for s in samples if s.series_id == series_id]
        if not matched:
            available = sorted({s.series_id for s in samples})
            raise ValueError(
                f"No se encontraron muestras para series_id='{series_id}'. "
                f"Disponibles en este fold: {available}"
            )
        return [(f"series_{series_id}", s) for s in matched[: max(1, n_samples)]]

    if mode == "indices":
        if not indices:
            raise ValueError(
                "El modo 'indices' requiere especificar al menos un índice en --indices."
            )
        by_idx = {s.dataset_idx: s for s in samples}
        selected: list[tuple[str, EvaluatedSample]] = []
        for idx in indices:
            if idx not in by_idx:
                raise ValueError(
                    f"Índice {idx} fuera de rango para el split de validación "
                    f"(0..{len(samples) - 1})."
                )
            selected.append(("index", by_idx[idx]))
        return selected

    if mode == "random":
        rng = np.random.default_rng(seed)
        k = min(max(1, n_samples), len(samples))
        chosen_pos = rng.choice(len(samples), size=k, replace=False)
        return [("random", samples[int(i)]) for i in chosen_pos]

    # Para best / worst / both se excluyen las muestras no informativas (F1 = NaN):
    # su métrica no significa nada y no deben ocupar puesto en el ranking.
    informative, _ = split_informative_samples(samples)
    pool = informative if informative else list(samples)
    k = min(max(1, n_samples), len(pool))

    best_sorted = sorted(pool, key=lambda s: (s.f1, s.iou, s.target_burned_px), reverse=True)
    worst_sorted = sorted(pool, key=lambda s: (s.f1, s.iou, -s.target_burned_px))

    if mode == "best":
        return [("best", s) for s in best_sorted[:k]]
    if mode == "worst":
        return [("worst", s) for s in worst_sorted[:k]]

    # mode == "both"
    out: list[tuple[str, EvaluatedSample]] = [("best", s) for s in best_sorted[:k]]
    out.extend(("worst", s) for s in worst_sorted[:k])
    return out


def export_panel_png(
    panel: Tensor,
    output_dir: str | Path,
    *,
    group: str,
    rank: int,
    sample: EvaluatedSample,
) -> Path:
    """
    Guarda un panel RGB `(3, H, W)` en disco como archivo `.png` descriptivo.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_series = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in sample.series_id)
    filename = (
        f"{group}_{rank:02d}_idx{sample.dataset_idx:03d}_{safe_series}"
        f"_h{int(round(sample.horizon_days))}d_f1_{int(round(sample.f1 * 100)):03d}.png"
    )
    target_path = out_dir / filename
    save_image(panel.clamp(0.0, 1.0), str(target_path))
    return target_path
