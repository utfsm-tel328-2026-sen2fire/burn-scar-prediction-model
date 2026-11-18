#!/usr/bin/env python3
"""
Minería de eventos MCD64A1 en el Mediterráneo (2018-2026) con clustering DBSCAN
(BL4: eps=150 km, 30 días) y exclusión estricta de los eventos CEMS (B6, D15, R20).

Exporta `data/mcd_manifest.json` listo para `scripts/download_gee_mcd64a1.py`
y `scripts/build_mcd64a1_dataset.py`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import re
import urllib.request
from pathlib import Path

import ee
import numpy as np
import rasterio

# Rejilla de horizontes de preentrenamiento (docs/dataset.md §8)
PRETRAIN_HORIZONS = [1, 2, 3, 5, 7, 10, 14]

# Cuenca mediterránea geográfica (docs/decisiones.md BL6)
MED_BBOX = (-9.5, 30.0, 36.0, 46.5)  # (lon_min, lat_min, lon_max, lat_max)
OVERVIEW_DIMS = (2275, 825)  # ~2 km por celda sobre la cuenca mediterránea

R_EARTH_KM = 6371.0
NUM_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distancia ortodrómica en kilómetros entre dos coordenadas WGS84."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * R_EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def haversine_matrix_km(coords: np.ndarray) -> np.ndarray:
    """Matriz de distancias Haversine (N, N) en km para un array (N, 2) de [lon, lat]."""
    lons = np.radians(coords[:, 0])[:, None]
    lats = np.radians(coords[:, 1])[:, None]
    dlon = lons - lons.T
    dlat = lats - lats.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats) * np.cos(lats.T) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def _parse_iso_date(ts: str | None) -> dt.date | None:
    if not ts:
        return None
    s = str(ts).replace("Z", "").split("+")[0]
    try:
        return dt.datetime.fromisoformat(s).date()
    except ValueError:
        return None


def _wkt_center(wkt: str) -> tuple[float, float] | None:
    pairs = [(float(m.group(1)), float(m.group(2))) for m in NUM_PAIR.finditer(wkt or "")]
    if not pairs:
        return None
    lons = [p[0] for p in pairs]
    lats = [p[1] for p in pairs]
    return (0.5 * (min(lons) + max(lons)), 0.5 * (min(lats) + max(lats)))


def load_cems_exclusion_records(
    cems_manifest_path: str | Path = "data/cems_manifest_46.json",
    tmb_manifest_path: str | Path = "data/tmb_manifest.json",
) -> list[dict]:
    """
    Carga los eventos de CEMS (tanto las 46 series limpias como las activaciones
    presentes en TMB-S3, cubriendo los 96 eventos mediterráneos y el resto de CEMS)
    para garantizar exclusión espacio-temporal completa (B6 / D15 / R20).
    """
    records: list[dict] = []

    p_cems = Path(cems_manifest_path)
    if p_cems.exists():
        with open(p_cems, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("series", []):
            center = s.get("center") or s.get("center_wgs84")
            if not center or len(center) < 2:
                continue
            dates = [
                _parse_iso_date(r.get("acquisition_time"))
                for r in s.get("rounds", [])
                if r.get("acquisition_time")
            ]
            dates = [d for d in dates if d is not None]
            if not dates:
                continue
            records.append(
                {
                    "id": s.get("series_id", s.get("code", "CEMS")),
                    "lon": float(center[0]),
                    "lat": float(center[1]),
                    "start_date": min(dates),
                    "end_date": max(dates),
                }
            )

    p_tmb = Path(tmb_manifest_path)
    if p_tmb.exists():
        with open(p_tmb, encoding="utf-8") as f:
            tmb_list = json.load(f)
        if isinstance(tmb_list, list):
            for item in tmb_list:
                center = _wkt_center(item.get("geometry_wkt", ""))
                if center is None:
                    continue
                d_evt = _parse_iso_date(item.get("event_time"))
                d_post = _parse_iso_date(item.get("delineation_time_post")) or d_evt
                if d_evt is None and d_post is None:
                    continue
                d0 = min(d for d in (d_evt, d_post) if d is not None)
                d1 = max(d for d in (d_evt, d_post) if d is not None)
                records.append(
                    {
                        "id": item.get("bbox_id", "TMB"),
                        "lon": float(center[0]),
                        "lat": float(center[1]),
                        "start_date": d0,
                        "end_date": d1,
                    }
                )

    return records


def is_overlapping_cems(
    lon: float,
    lat: float,
    start_date: dt.date,
    end_date: dt.date,
    cems_records: list[dict],
    spatial_radius_km: float = 50.0,
    temporal_buffer_days: int = 45,
) -> tuple[bool, str | None]:
    """
    Verifica si un evento candidato (lon, lat, [start_date, end_date]) se solapa
    en espacio (<= spatial_radius_km) y tiempo (+/- temporal_buffer_days) con algún
    evento CEMS.
    """
    w_start = start_date - dt.timedelta(days=temporal_buffer_days)
    w_end = end_date + dt.timedelta(days=temporal_buffer_days)

    for rec in cems_records:
        if w_end < rec["start_date"] or w_start > rec["end_date"]:
            continue
        dist = haversine_km(lon, lat, rec["lon"], rec["lat"])
        if dist <= spatial_radius_km:
            return True, rec["id"]
    return False, None


def dbscan_spatiotemporal(
    lons: np.ndarray,
    lats: np.ndarray,
    ordinals: np.ndarray,
    eps_km: float = 150.0,
    eps_days: int = 30,
    min_samples: int = 1,
) -> np.ndarray:
    """
    Agrupación DBSCAN espacio-temporal (distancia Haversine <= eps_km y
    |diferencia temporal| <= eps_days). Retorna etiquetas de clúster (0..K-1, -1=ruido).
    """
    n = len(lons)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    coords = np.column_stack([lons, lats])
    dist_km = haversine_matrix_km(coords)
    time_diff = np.abs(ordinals[:, None] - ordinals[None, :])
    adj = (dist_km <= eps_km) & (time_diff <= eps_days)
    neighbors_count = adj.sum(axis=1)
    is_core = neighbors_count >= min_samples

    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i] or not is_core[i]:
            continue
        visited[i] = True
        labels[i] = cluster_id
        queue = list(np.where(adj[i])[0])
        head = 0
        while head < len(queue):
            q = queue[head]
            head += 1
            if not visited[q]:
                visited[q] = True
                if is_core[q]:
                    for nb in np.where(adj[q])[0]:
                        if not visited[nb]:
                            queue.append(nb)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels


def build_event_window_and_horizons(
    start_date: dt.date,
    end_date: dt.date,
    horizons: list[int] | None = None,
    pre_fire_days: int = 14,
    post_fire_days: int = 1,
) -> dict:
    """
    Calcula la ventana de descarga OLCI (incluyendo 14 días pre-incendio por D18)
    y la rejilla de horizontes de muestreo h para el evento MCD64A1.
    """
    if horizons is None:
        horizons = list(PRETRAIN_HORIZONS)
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    olci_start = start_date - dt.timedelta(days=pre_fire_days)
    olci_end = end_date + dt.timedelta(days=post_fire_days)
    duration_days = (end_date - start_date).days + 1

    return {
        "burn_start_date": start_date.isoformat(),
        "burn_end_date": end_date.isoformat(),
        "duration_days": duration_days,
        "olci_start_date": olci_start.isoformat(),
        "olci_end_date": olci_end.isoformat(),
        "candidate_horizons_days": sorted(int(h) for h in horizons),
    }


def extract_events_from_overview(
    burn_doy: np.ndarray,
    year: int,
    bbox: tuple[float, float, float, float] = MED_BBOX,
    transform=None,
    local_eps_km: float = 12.0,
    local_eps_days: int = 14,
    min_overview_cells: int = 2,
) -> list[dict]:
    """
    A partir de un raster mensual de `BurnDate` sobre la cuenca mediterránea,
    agrupa celdas quemadas contiguas en parches candidatos de incendio (~19.2 km).
    """
    rows, cols = np.where(burn_doy > 0)
    if len(rows) == 0:
        return []

    h, w = burn_doy.shape
    if transform is not None:
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        lons = np.asarray(xs, dtype=np.float64)
        lats = np.asarray(ys, dtype=np.float64)
    else:
        lon_min, lat_min, lon_max, lat_max = bbox
        lons = lon_min + (cols + 0.5) * (lon_max - lon_min) / w
        lats = lat_max - (rows + 0.5) * (lat_max - lat_min) / h
    doys = burn_doy[rows, cols].astype(np.int32)

    base_ord = dt.date(year, 1, 1).toordinal() - 1
    ords = base_ord + doys

    # Si hay demasiadas celdas en un mes, submuestreamos o agrupamos por rejilla fina
    if len(lons) > 4000:
        idx = np.linspace(0, len(lons) - 1, 4000, dtype=np.int32)
        lons, lats, doys, ords = lons[idx], lats[idx], doys[idx], ords[idx]

    patch_labels = dbscan_spatiotemporal(
        lons,
        lats,
        ords,
        eps_km=local_eps_km,
        eps_days=local_eps_days,
        min_samples=min_overview_cells,
    )

    events: list[dict] = []
    for cid in sorted(set(patch_labels)):
        if cid < 0:
            continue
        mask = patch_labels == cid
        n_cells = int(mask.sum())
        if n_cells < min_overview_cells:
            continue
        c_lons = lons[mask]
        c_lats = lats[mask]
        mean_lon = float(np.mean(c_lons))
        mean_lat = float(np.mean(c_lats))
        # Elegir el medoide (celda quemada más cercana al centroide del parche)
        # para garantizar que la ventana 64x64 a 300 m caiga sobre píxeles quemados.
        medoid_idx = int(np.argmin((c_lons - mean_lon) ** 2 + (c_lats - mean_lat) ** 2))
        c_lon = float(c_lons[medoid_idx])
        c_lat = float(c_lats[medoid_idx])
        d_min = dt.date.fromordinal(int(np.min(ords[mask])))
        d_max = dt.date.fromordinal(int(np.max(ords[mask])))
        zone = int((c_lon + 180) // 6) + 1
        epsg_utm = (32600 + zone) if c_lat >= 0 else (32700 + zone)

        events.append(
            {
                "year": year,
                "center": [round(c_lon, 6), round(c_lat, 6)],
                "epsg_utm": epsg_utm,
                "start_date": d_min,
                "end_date": d_max,
                "overview_burned_cells": n_cells,
                "estimated_burned_ha": round(n_cells * 400.0, 1),
            }
        )
    return events


def download_monthly_overview(
    image_id: str, bbox: tuple[float, float, float, float] = MED_BBOX
) -> tuple[np.ndarray, rasterio.Affine]:
    """Descarga el mapa mensual de BurnDate sobre el Mediterráneo (~7 KB comprimido)."""
    ee_bbox = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
    img = ee.Image(image_id).select(["BurnDate"])
    url = img.getDownloadURL(
        {
            "region": ee_bbox,
            "dimensions": list(OVERVIEW_DIMS),
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )
    req = urllib.request.Request(url, headers={"User-Agent": "burn-scar-mcd-miner/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read(1), src.transform


def filter_and_cluster_candidates(
    candidates: list[dict],
    cems_records: list[dict],
    max_events: int | None = None,
    cluster_eps_km: float = 150.0,
    cluster_eps_days: int = 30,
    cems_exclusion_km: float = 50.0,
    cems_exclusion_days: int = 45,
) -> tuple[list[dict], dict]:
    """
    1. Agrupa los eventos candidatos con DBSCAN (BL4: eps=150 km, 30 días).
    2. Excluye cualquier evento (y cualquier clúster completo) que se solape en espacio
       y tiempo con alguno de los eventos CEMS (B6, D15, R20).
    3. Prioriza por área quemada estimada y asigna folds geográficos 0..4 sin fuga.
    """
    if not candidates:
        return [], {"total_candidates": 0, "excluded_cems": 0, "selected": 0}

    lons = np.array([c["center"][0] for c in candidates], dtype=np.float64)
    lats = np.array([c["center"][1] for c in candidates], dtype=np.float64)
    ords = np.array([c["start_date"].toordinal() for c in candidates], dtype=np.int32)

    cluster_ids = dbscan_spatiotemporal(
        lons, lats, ords, eps_km=cluster_eps_km, eps_days=cluster_eps_days, min_samples=1
    )

    # Identificar clústeres contaminados por CEMS
    contaminated_clusters: set[int] = set()
    direct_overlaps = 0
    for idx, cand in enumerate(candidates):
        overlaps, _ = is_overlapping_cems(
            cand["center"][0],
            cand["center"][1],
            cand["start_date"],
            cand["end_date"],
            cems_records,
            spatial_radius_km=cems_exclusion_km,
            temporal_buffer_days=cems_exclusion_days,
        )
        if overlaps:
            direct_overlaps += 1
            contaminated_clusters.add(int(cluster_ids[idx]))

    clean: list[dict] = []
    excluded_count = 0
    for idx, cand in enumerate(candidates):
        cid = int(cluster_ids[idx])
        if cid in contaminated_clusters:
            excluded_count += 1
            continue
        item = dict(cand)
        item["dbscan_cluster_id"] = cid
        clean.append(item)

    # Ordenar de mayor a menor señal (celdas quemadas) para priorizar en --max-events
    clean.sort(
        key=lambda x: (x.get("overview_burned_cells", 0), x["start_date"].toordinal()),
        reverse=True,
    )
    if max_events is not None and max_events > 0:
        clean = clean[:max_events]

    # Asignar fold interno (0..4) por clúster DBSCAN ordenado de Oeste a Este (D8)
    by_cluster: dict[int, list[dict]] = {}
    for ev in clean:
        by_cluster.setdefault(ev["dbscan_cluster_id"], []).append(ev)

    sorted_clusters = sorted(
        by_cluster.keys(),
        key=lambda c: float(np.mean([e["center"][0] for e in by_cluster[c]])),
    )
    n_clusters = max(1, len(sorted_clusters))
    cluster_to_fold = {
        c: min(int(i * 5 / n_clusters), 4) for i, c in enumerate(sorted_clusters)
    }

    final_events: list[dict] = []
    for i, ev in enumerate(clean, 1):
        win = build_event_window_and_horizons(ev["start_date"], ev["end_date"])
        sid = f"MCD_{ev['start_date'].strftime('%Y%m%d')}_{i:04d}"
        final_events.append(
            {
                "series_id": sid,
                "source": "MODIS/061/MCD64A1",
                "year": int(ev["year"]),
                "center": ev["center"],
                "epsg_utm": int(ev["epsg_utm"]),
                "dbscan_cluster_id": int(ev["dbscan_cluster_id"]),
                "fold": int(cluster_to_fold[ev["dbscan_cluster_id"]]),
                "overview_burned_cells": int(ev["overview_burned_cells"]),
                "estimated_burned_ha": float(ev["estimated_burned_ha"]),
                **win,
            }
        )

    stats = {
        "total_candidates": len(candidates),
        "direct_cems_overlaps": direct_overlaps,
        "excluded_by_cems_cluster": excluded_count,
        "selected_events": len(final_events),
        "unique_dbscan_clusters": len(sorted_clusters),
    }
    return final_events, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Construye el manifiesto de eventos MCD64A1 mediterráneos (2018-2026) "
            "excluyendo CEMS."
        )
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[6, 7, 8, 9],
        help="Meses a escanear por año (por defecto temporada alta mediterránea 6..9)",
    )
    parser.add_argument("--max-events", type=int, default=500)
    parser.add_argument("--cems-manifest", default="data/cems_manifest_46.json")
    parser.add_argument("--tmb-manifest", default="data/tmb_manifest.json")
    parser.add_argument("--output", default="data/mcd_manifest.json")
    args = parser.parse_args()

    ee.Initialize()

    cems_records = load_cems_exclusion_records(args.cems_manifest, args.tmb_manifest)
    print("=" * 72)
    print("MINERÍA DE EVENTOS MCD64A1 (MEDITERRÁNEO 2018-2026, BL4 + D15)")
    print(f"Registros de exclusión CEMS/TMB cargados: {len(cems_records)}")
    print(
        f"Años: {args.start_year}..{args.end_year} | "
        f"Meses: {args.months} | Max eventos: {args.max_events}"
    )
    print("=" * 72)

    col = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(f"{args.start_year}-01-01", f"{args.end_year}-12-31")
    )
    img_info_list = col.toList(200).getInfo()

    allowed_months = set(args.months)
    candidates: list[dict] = []

    # Recorrer de años recientes a antiguos (o todos los meses seleccionados)
    for item in reversed(img_info_list):
        img_id = item["id"]
        parts = img_id.split("/")[-1].split("_")
        if len(parts) < 2:
            continue
        yr, mo = int(parts[0]), int(parts[1])
        if yr < args.start_year or yr > args.end_year or mo not in allowed_months:
            continue

        try:
            arr, transform = download_monthly_overview(img_id)
            evts = extract_events_from_overview(arr, year=yr, transform=transform)
            candidates.extend(evts)
            print(
                f"  [{yr}-{mo:02d}] {len(evts):>3} parches quemados detectados "
                f"(acumulado={len(candidates)})"
            )
        except Exception as exc:
            print(f"  ! Error en {img_id}: {exc}")

        # Si ya hay suficientes candidatos limpios para cubrir max_events con holgura, parar
        if args.max_events and len(candidates) >= max(args.max_events * 3, 120):
            break

    final_events, stats = filter_and_cluster_candidates(
        candidates, cems_records, max_events=args.max_events
    )

    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_collection": "MODIS/061/MCD64A1",
        "bbox_wgs84": list(MED_BBOX),
        "dbscan_params": {"eps_km": 150.0, "eps_days": 30},
        "cems_exclusion_params": {"spatial_radius_km": 50.0, "temporal_buffer_days": 45},
        "summary": stats,
        "series": final_events,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("RESUMEN DEL MANIFIESTO MCD64A1")
    print("=" * 72)
    print(f"Candidatos totales:          {stats['total_candidates']}")
    print(f"Excluidos por solape CEMS:   {stats['excluded_by_cems_cluster']}")
    print(f"Eventos seleccionados:       {stats['selected_events']}")
    print(f"Clústeres DBSCAN (150 km):   {stats['unique_dbscan_clusters']}")
    print(f"Guardado en:                 {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
