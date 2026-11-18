"""
Dataset PyTorch para Predicción de Propagación de Cicatrices (BurnScarDataset).

Carga las muestras empaquetadas en data/processed/samples/ con soporte
para particionado por Folds de Validación Cruzada (0..4 y -1).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_INDEX_PATH = Path("data/processed/dataset_index.json")

# Raíz del proyecto (este módulo vive en <raíz>/src/burn_scar/dataset.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ficheros cuyo contenido determina el dataset generado. Su huella conjunta
# identifica la versión del generador: si cambia alguno, el dataset almacenado
# queda obsoleto y hay que regenerarlo.
GENERATOR_SOURCES = (
    PROJECT_ROOT / "scripts" / "build_cems_manifest.py",
    PROJECT_ROOT / "scripts" / "download_gee_olci.py",
    PROJECT_ROOT / "scripts" / "build_pytorch_dataset.py",
    PROJECT_ROOT / "src" / "burn_scar" / "quality.py",
    PROJECT_ROOT / "src" / "burn_scar" / "dataset.py",
)

# Formato de la clave temporal de un frame OLCI ('20260912T091600').
FRAME_KEY_FORMAT = "%Y%m%dT%H%M%S"

# Tolerancia para emparejar el frame f con su fichero de imagen. La marca de
# tiempo del frame en el manifiesto de origen (CDSE) y la del fichero descargado
# (GEE) pueden diferir en sub-segundo porque uno trunca y el otro redondea; sin
# tolerancia, la adquisición de f quedaba fuera de su propia ventana.
FRAME_MATCH_TOLERANCE = dt.timedelta(minutes=2)

# Longitud de la ventana temporal: 1 frame pre-incendio + 6 post-incendio.
TEMPORAL_WINDOW_LEN = 7

# Codificación canónica de las máscaras de validez del paquete de muestra:
# 1 = píxel evaluable, 255 = píxel ignorado (no observado en la ronda).
# Es la misma convención que emite el rasterizador CEMS y que espera la
# visualización (src/burn_scar/visualization.py).
MASK_EVALUABLE = 1
MASK_IGNORED = 255


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normaliza una máscara de validez a la codificación canónica del proyecto:
    `1` = píxel evaluable, `255` = píxel ignorado.

    Tolera los paquetes antiguos, en los que la validez común entre rondas se
    emitía con `0` para los píxeles no evaluables. Esta normalización es
    compatibilidad con esos paquetes, no una garantía: el requisito de que el
    paquete almacenado use solo `1` y `255` se verifica sobre el fichero, en
    `scripts/build_pytorch_dataset.py` y en los tests.
    """
    m = np.asarray(mask).astype(np.uint8)
    return np.where(m == MASK_EVALUABLE, MASK_EVALUABLE, MASK_IGNORED).astype(np.uint8)


def compute_generator_version(sources: Sequence[Path] | None = None) -> str:
    """
    Huella corta (sha256 truncado) del contenido de los ficheros que
    determinan el dataset generado.

    Se usa en lugar de un número de versión mantenido a mano (que se olvida
    subir) y del hash de un commit (que no captura un árbol de trabajo sucio).
    """
    digest = hashlib.sha256()
    for source in (sources if sources is not None else GENERATOR_SOURCES):
        digest.update(Path(source).name.encode("utf-8"))
        path = Path(source)
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:12]


def generator_source_names(sources: Sequence[Path] | None = None) -> list[str]:
    """Rutas (relativas a la raíz del proyecto) de los ficheros del generador."""
    resolved = sources if sources is not None else GENERATOR_SOURCES
    out: list[str] = []
    for source in resolved:
        path = Path(source)
        try:
            out.append(str(path.resolve().relative_to(PROJECT_ROOT)))
        except ValueError:
            out.append(str(path))
    return out


def validate_unique_sample_index(samples: Sequence[dict]) -> None:
    """Rechaza índices con identificadores o rutas de muestra repetidos."""
    duplicate_ids = _duplicates(sample.get("sample_id") for sample in samples)
    duplicate_files = _duplicates(sample.get("file") for sample in samples)
    problems = []
    if duplicate_ids:
        problems.append(f"sample_id duplicados: {sorted(duplicate_ids)}")
    if duplicate_files:
        problems.append(f"rutas de fichero duplicadas: {sorted(duplicate_files)}")
    if problems:
        raise ValueError(
            "El índice del dataset no es canónico (" + "; ".join(problems) + "). "
            "Regenera el dataset antes de entrenar o evaluar."
        )


def _duplicates(values: Sequence[str | None]) -> list[str]:
    """Valores no nulos que aparecen más de una vez."""
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return [value for value, count in counts.items() if count > 1]


def _warn_if_dataset_is_stale(index_data: dict) -> None:
    """
    Avisa si el dataset cargado fue producido por un generador distinto del
    vigente.

    Es el punto de uso real del dataset (entrenar o evaluar), así que es donde
    la diferencia importa. Un dataset sin versión registrada se avisa aparte:
    es anterior a esta instrumentación, no necesariamente incompatible.
    """
    stored = index_data.get("generator_version")
    if stored is None:
        warnings.warn(
            "El dataset no registra la versión del generador; puede haber sido "
            "producido por una versión anterior del ensamblador. Regenera el "
            "dataset para estar seguro de que sus muestras cumplen el spec.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    current = compute_generator_version()
    if stored != current:
        warnings.warn(
            f"El dataset fue generado con la versión {stored} del generador y el "
            f"código actual produce la {current}. Regenera el dataset antes de "
            f"entrenar o evaluar con él.",
            RuntimeWarning,
            stacklevel=3,
        )


def resolve_loader_options(
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[int, bool]:
    """
    Decide (num_workers, pin_memory) según el backend de GPU.

    En ROCm (AMD) `pin_memory=True` con workers es patológico: cada lote
    reserva memoria fijada con hipHostMalloc (cientos de ms por tensor) y el
    fork de un proceso con contexto HIP inicializado cuesta segundos. Con este
    dataset (≈2 ms por muestra) los workers no aportan nada, así que en AMD
    se desactivan. En CUDA se mantienen los valores habituales.
    """
    is_rocm = bool(getattr(torch.version, "hip", None))
    if num_workers is None:
        num_workers = 0 if is_rocm else 2
    if pin_memory is None:
        pin_memory = False if is_rocm else torch.cuda.is_available()
    return int(num_workers), bool(pin_memory)


def frame_key_to_datetime(key: str) -> dt.datetime:
    """Convierte una clave de frame OLCI ('20260912T091600') en su instante."""
    return dt.datetime.strptime(key, FRAME_KEY_FORMAT)


def match_frame_key(
    keys: Sequence[str],
    instant: dt.datetime,
    tolerance: dt.timedelta = FRAME_MATCH_TOLERANCE,
) -> str | None:
    """
    Devuelve la clave de frame más cercana en el tiempo a `instant`, siempre que
    la diferencia no supere `tolerance`; en caso contrario, `None`.

    El emparejamiento es por cercanía temporal y no por comparación de cadenas
    porque la marca de tiempo del manifiesto de origen y la del fichero
    descargado pueden diferir en sub-segundo. Entre dos adquisiciones
    candidatas gana la más cercana, de modo que el emparejamiento es unívoco.

    Devolver `None` significa que no hay adquisición utilizable de ese frame.
    """
    best_key: str | None = None
    best_delta = float("inf")
    for key in keys:
        delta = abs((frame_key_to_datetime(key) - instant).total_seconds())
        if delta < best_delta:
            best_key, best_delta = key, delta
    if best_key is None or best_delta > tolerance.total_seconds():
        return None
    return best_key


def build_temporal_window(
    usable_keys: Sequence[str],
    frame_key: str,
    pre_fire_cut_key: str,
    window_len: int = TEMPORAL_WINDOW_LEN,
) -> list[str]:
    """
    Compone la ventana temporal de una muestra.

    - Con adquisiciones post-incendio utilizables: la adquisición pre-incendio
      más reciente en primera posición y hasta `window_len - 1` post-incendios
      que terminan en `frame_key`.
    - Sin post-incendio utilizable: las últimas `window_len` adquisiciones
      utilizables hasta `frame_key`.
    - Si no llegan, se rellena repitiendo el frame más antiguo.

    En todos los casos la ventana termina en `frame_key` y está ordenada
    cronológicamente.
    """
    past = sorted(
        (k for k in usable_keys if k <= frame_key),
        key=frame_key_to_datetime,
    )
    if not past:
        return []

    post = [k for k in past if k >= pre_fire_cut_key]
    if post:
        pre_candidates = [k for k in past if k < pre_fire_cut_key]
        if pre_candidates:
            selected = [pre_candidates[-1]] + post[-(window_len - 1):]
        else:
            selected = past[-window_len:]
    else:
        selected = past[-window_len:]

    while len(selected) < window_len:
        selected.insert(0, selected[0])
    return selected


def is_chronological(keys: Sequence[str]) -> bool:
    """Indica si una secuencia de claves de frame está ordenada en el tiempo."""
    instants = [frame_key_to_datetime(k) for k in keys]
    return all(a <= b for a, b in zip(instants, instants[1:], strict=False))


def resolve_sample_window(
    usable_keys: Sequence[str],
    frame_iso: str,
    pre_fire_cut_key: str,
    window_len: int = TEMPORAL_WINDOW_LEN,
) -> tuple[str | None, list[str]]:
    """
    Resuelve la ventana temporal de una muestra a partir de su frame `f`.

    Devuelve `(f_key, ventana)`. Si no hay ninguna adquisición utilizable que
    corresponda a `f`, devuelve `(None, [])` y la muestra debe descartarse: no
    se sustituye por una adquisición anterior, porque eso anclaría la cicatriz
    y el target a un instante distinto del que define el horizonte.
    """
    instant = dt.datetime.fromisoformat(frame_iso.replace("Z", ""))
    f_key = match_frame_key(usable_keys, instant)
    if f_key is None:
        return None, []
    return f_key, build_temporal_window(usable_keys, f_key, pre_fire_cut_key, window_len)


class BurnScarDataset(Dataset):
    """
    Dataset PyTorch para entrenamiento y evaluación multi-horizonte.
    
    Retorna por muestra:
        x_olci:       (7, 5, 64, 64) float32 (secuencia de 7 frames con 5 bandas espectrales)
        x_dem:        (1, 64, 64)    float32 (elevación normalizada)
        x_worldcover: (1, 64, 64)    float32 (clases de cobertura de suelo normalizadas)
        x_scar:       (1, 64, 64)    float32 (cicatriz TMB-S3 en el frame actual f)
        horizon_days: (1,)           float32 (horizonte h en días hacia el objetivo)
        target:       (1, 64, 64)    float32 (lo NUEVO quemado en [f, f+h], valores en [0..1])
        mask:         (1, 64, 64)    uint8   (1 = píxel válido, 255 = ignorado/no analizado)
        target_cems: (1, 64, 64)    float32 (crecimiento real entre rondas CEMS; referencia)
        mask_cems:   (1, 64, 64)    uint8   (validez común a las dos rondas; misma codificación)
        has_reference: (1,)          float32 (1 si la muestra trae target de referencia)
        meta:         dict           (sample_id, series_id, fold, etc.)
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        folds: Sequence[int] | int | None = None,
        transform=None,
    ) -> None:
        super().__init__()
        if index_path is None:
            index_path = DEFAULT_INDEX_PATH
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"No se encontró el índice del dataset en: {index_path}")

        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)

        _warn_if_dataset_is_stale(index_data)

        all_samples = index_data.get("samples", [])
        validate_unique_sample_index(all_samples)
        self.index_data = index_data
        self.total_entries = len(all_samples)
        self.total_unique_samples = len({s.get("sample_id") for s in all_samples})
        self.total_unique_files = len({s.get("file") for s in all_samples})
        self.total_series = len({s.get("series_id") for s in all_samples})
        if folds is not None:
            if isinstance(folds, int):
                folds = [folds]
            allowed_folds = set(folds)
            self.samples = [s for s in all_samples if s["fold"] in allowed_folds]
        else:
            self.samples = all_samples

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        info = self.samples[idx]
        file_path = info["file"]

        data = np.load(file_path)

        # 1. OLCI: normalizar radiancia (dividir entre 1000.0) -> shape (7, 5, 64, 64)
        x_olci = torch.from_numpy(data["x_olci"].astype(np.float32) / 1000.0)

        # 2. DEM: normalizar elevación en km (dividir entre 1000.0) -> shape (1, 64, 64)
        x_dem = torch.from_numpy(data["x_dem"].astype(np.float32) / 1000.0).unsqueeze(0)

        # 3. WorldCover: normalizar categorías -> shape (1, 64, 64)
        x_wc = torch.from_numpy(data["x_worldcover"].astype(np.float32) / 100.0).unsqueeze(0)

        # 4. Cicatriz TMB-S3 en f -> shape (1, 64, 64)
        x_scar = torch.from_numpy(data["x_scar"].astype(np.float32)).unsqueeze(0)

        # 5. Horizonte h en días -> shape (1,)
        h_days = torch.tensor([float(data["horizon_days"])], dtype=torch.float32)

        # 6. Target: incremento de quemado -> shape (1, 64, 64)
        target = torch.from_numpy(data["target"].astype(np.float32)).unsqueeze(0)

        # 7. Máscara de validez -> shape (1, 64, 64)
        #    Codificación canónica: 1 = evaluable, 255 = ignorado. Se normaliza
        #    al leer para tolerar paquetes antiguos (0 = no evaluable).
        mask = torch.from_numpy(_canonical_mask(data["mask"])).unsqueeze(0)

        # 8. Target de referencia: crecimiento real entre rondas CEMS, con su
        #    validez común. Puede no existir en datasets generados antes de
        #    esta capacidad; en ese caso se marca como no disponible.
        if "target_cems" in data.files:
            target_cems = torch.from_numpy(data["target_cems"].astype(np.float32)).unsqueeze(0)
            has_reference = 1.0
        else:
            target_cems = torch.zeros_like(target)
            has_reference = 0.0
        if "mask_cems" in data.files:
            mask_cems = torch.from_numpy(_canonical_mask(data["mask_cems"])).unsqueeze(0)
        else:
            mask_cems = torch.full_like(mask, MASK_IGNORED)

        sample = {
            "x_olci": x_olci,
            "x_dem": x_dem,
            "x_worldcover": x_wc,
            "x_scar": x_scar,
            "horizon_days": h_days,
            "target": target,
            "mask": mask,
            "target_cems": target_cems,
            "mask_cems": mask_cems,
            "has_reference": torch.tensor([has_reference], dtype=torch.float32),
            "meta": info,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_cv_dataloaders(
    val_fold: int,
    batch_size: int = 8,
    num_workers: int | None = None,
    index_path: str | Path | None = None,
    pin_memory: bool | None = None,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Retorna (train_loader, val_loader) para un fold de validación específico (0..4).
    Entrena en los 4 folds restantes y valida en val_fold (D14).

    `num_workers` y `pin_memory` en None se resuelven según el backend
    (ver `resolve_loader_options`).
    """
    if val_fold not in range(5):
        raise ValueError(f"val_fold debe estar en [0..4], recibido: {val_fold}")

    nw, pin = resolve_loader_options(num_workers, pin_memory)

    train_folds = [f for f in range(5) if f != val_fold]
    train_dataset = BurnScarDataset(index_path=index_path, folds=train_folds)
    val_dataset = BurnScarDataset(index_path=index_path, folds=[val_fold])

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )

    return train_loader, val_loader
