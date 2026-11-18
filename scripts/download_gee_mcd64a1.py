#!/usr/bin/env python3
"""
Descarga incremental y reanudable de parches 64x64 a 300 m para eventos MCD64A1:
- MCD64A1 (`BurnDate`, `Uncertainty`, `QA` + `burn_ordinal`) en
  `data/raw_mcd/mcd64a1/{series_id}/mcd64a1_layers.npz`
- Capas estáticas (`DEM`, `WorldCover`) en `data/raw_mcd/static/{series_id}/static_layers.npz`
- Frames Sentinel-3 OLCI (5 bandas, conservando nubes/humo reales y descartando solo
  bordes de swath / adquisiciones nocturnas con `MAX_NODATA_FRAC > 0.20`) en
  `data/raw_mcd/s3_olci/{series_id}/{date_tag}.npz`
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import numpy as np

from scripts.download_gee_olci import S3_BANDS, download_gee_patch

MCD_BANDS = ["BurnDate", "Uncertainty", "QA"]
MAX_NODATA_FRAC = 0.20


def download_patch_with_retry(
    image: ee.Image,
    region: ee.Geometry,
    crs: str,
    dimensions: tuple[int, int] = (64, 64),
    retries: int = 3,
) -> np.ndarray:
    """Wrapper con backoff exponencial sobre download_gee_patch."""
    delay = 1.0
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            return download_gee_patch(image, region, crs, dimensions=dimensions)
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
            delay *= 2.0
    assert last_exc is not None
    raise last_exc


def _download_dem_patch(region: ee.Geometry, crs_str: str) -> np.ndarray:
    for asset_id in ("COPERNICUS/DEM/GLO30", "COPERNICUS/DEM/GLO30_2024_1"):
        try:
            dem_img = ee.ImageCollection(asset_id).select("DEM").mosaic()
            return download_patch_with_retry(dem_img, region, crs_str)[0].astype(np.float32)
        except Exception:
            continue
    raise RuntimeError("No se pudo descargar Copernicus DEM desde GEE.")


def process_mcd_event_gee(
    event_meta: dict,
    raw_root: str | Path = "data/raw_mcd",
    max_olci_frames: int = 45,
) -> dict:
    raw_root = Path(raw_root)
    sid = event_meta["series_id"]
    lon_c, lat_c = event_meta["center"]
    epsg_utm = event_meta.get("epsg_utm")
    if not epsg_utm:
        zone = int((lon_c + 180) // 6) + 1
        epsg_utm = (32600 + zone) if lat_c >= 0 else (32700 + zone)
    crs_str = f"EPSG:{epsg_utm}"

    point = ee.Geometry.Point([lon_c, lat_c])
    region = point.buffer(9600).bounds()

    # 1. Descargar estáticas (DEM + WorldCover) si no existen
    static_dir = raw_root / "static" / sid
    static_dir.mkdir(parents=True, exist_ok=True)
    static_file = static_dir / "static_layers.npz"

    if not static_file.exists():
        dem_arr = _download_dem_patch(region, crs_str)
        wc_img = ee.ImageCollection("ESA/WorldCover/v200").select("Map").mosaic()
        wc_arr = download_patch_with_retry(wc_img, region, crs_str)[0].astype(np.uint8)
        np.savez_compressed(static_file, dem=dem_arr, worldcover=wc_arr)

    # 2. Descargar parche MCD64A1 (BurnDate, Uncertainty, QA) a 64x64 (300 m, nearest)
    mcd_dir = raw_root / "mcd64a1" / sid
    mcd_dir.mkdir(parents=True, exist_ok=True)
    mcd_file = mcd_dir / "mcd64a1_layers.npz"

    burn_start = dt.date.fromisoformat(event_meta["burn_start_date"])
    burn_end = dt.date.fromisoformat(event_meta["burn_end_date"])
    m_start = dt.date(burn_start.year, burn_start.month, 1)
    m_end = burn_end + dt.timedelta(days=45)

    if not mcd_file.exists():
        mcd_col = (
            ee.ImageCollection("MODIS/061/MCD64A1")
            .filterBounds(point)
            .filterDate(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            .select(MCD_BANDS)
        )
        mcd_items = mcd_col.toList(6).getInfo()

        burn_ordinal = np.zeros((64, 64), dtype=np.int32)
        burn_doy = np.zeros((64, 64), dtype=np.int16)
        uncertainty = np.zeros((64, 64), dtype=np.uint8)
        qa = np.zeros((64, 64), dtype=np.uint8)

        for idx_m, item in enumerate(mcd_items):
            img_id = item["id"]
            props = item.get("properties", {})
            t_ms = props.get("system:time_start", 0)
            img_year = dt.datetime.fromtimestamp(t_ms / 1000.0, tz=dt.UTC).year
            arr = download_patch_with_retry(ee.Image(img_id).select(MCD_BANDS), region, crs_str)
            doy_m = arr[0].astype(np.int16)
            unc_m = np.clip(arr[1], 0, 255).astype(np.uint8)
            qa_m = np.clip(arr[2], 0, 255).astype(np.uint8)

            if idx_m == 0:
                burn_doy = doy_m.copy()
                uncertainty = unc_m.copy()
                qa = qa_m.copy()
                neg_mask = doy_m < 0
                burn_ordinal[neg_mask] = doy_m[neg_mask].astype(np.int32)

            pos_mask = (doy_m > 0) & (burn_ordinal <= 0)
            if np.any(pos_mask):
                base_ord = dt.date(img_year, 1, 1).toordinal() - 1
                burn_ordinal[pos_mask] = base_ord + doy_m[pos_mask].astype(np.int32)
                burn_doy[pos_mask] = doy_m[pos_mask]
                uncertainty[pos_mask] = unc_m[pos_mask]
                qa[pos_mask] = qa_m[pos_mask]

        np.savez_compressed(
            mcd_file,
            burn_ordinal=burn_ordinal,
            burn_doy=burn_doy,
            uncertainty=uncertainty,
            qa=qa,
        )

    # 3. Descargar frames Sentinel-3 OLCI (conservando nubes/humo reales)
    olci_dir = raw_root / "s3_olci" / sid
    olci_dir.mkdir(parents=True, exist_ok=True)

    olci_start = event_meta["olci_start_date"]
    olci_end = event_meta["olci_end_date"]

    s3_col = (
        ee.ImageCollection("COPERNICUS/S3/OLCI")
        .filterBounds(point)
        .filterDate(olci_start, olci_end)
        .select(S3_BANDS)
    )
    img_list = s3_col.toList(max_olci_frames).getInfo()

    saved_frames = 0
    skipped_nodata = 0

    for item in img_list:
        img_id = item["id"]
        props = item.get("properties", {})
        time_start = props.get("system:time_start", 0)
        frame_dt = dt.datetime.fromtimestamp(time_start / 1000.0, tz=dt.UTC)
        date_tag = frame_dt.strftime("%Y%m%dT%H%M%S")

        frame_path = olci_dir / f"{date_tag}.npz"
        if frame_path.exists():
            saved_frames += 1
            continue

        try:
            bands_arr = download_patch_with_retry(
                ee.Image(img_id).select(S3_BANDS), region, crs_str
            )
            valid = (bands_arr > 0) & (bands_arr < 32768)
            fill_frac = float((~valid).mean())
            has_signal = bool(valid.any()) and float(bands_arr[valid].max()) > 0.0
            # Solo descartamos bordes de swath / nocturnos sin señal; las nubes reales se conservan
            if fill_frac > MAX_NODATA_FRAC or not has_signal:
                skipped_nodata += 1
                continue

            np.savez_compressed(
                frame_path,
                olci_bands=bands_arr.astype(np.uint16),
                bands=S3_BANDS,
                timestamp=frame_dt.isoformat(),
                image_id=img_id,
            )
            saved_frames += 1
        except Exception as exc:
            print(f"   ! Error descargando OLCI {sid}/{date_tag}: {exc}")

    return {
        "series_id": sid,
        "frames_saved": saved_frames,
        "frames_skipped_swath_nodata": skipped_nodata,
        "total_olci_candidates": len(img_list),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descarga incremental de parches MCD64A1, OLCI y estáticas en data/raw_mcd/"
    )
    parser.add_argument("--manifest", default="data/mcd_manifest.json")
    parser.add_argument("--raw-root", default="data/raw_mcd")
    parser.add_argument("--limit", type=int, default=None, help="Límite de eventos a procesar")
    parser.add_argument(
        "--max-olci-frames",
        type=int,
        default=45,
        help="Máximo de escenas OLCI a consultar por evento",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Número de hilos paralelos para peticiones a GEE (por defecto 6)",
    )
    args = parser.parse_args()

    ee.Initialize()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    series_list = manifest.get("series", [])
    if args.limit:
        series_list = series_list[: args.limit]

    workers = max(1, int(args.workers))
    print("=" * 72, flush=True)
    print("DESCARGA DE PARCHES MCD64A1 + OLCI + ESTÁTICAS (64x64 a 300 m)", flush=True)
    print(
        f"Eventos: {len(series_list)} | Workers: {workers} | Destino: {args.raw_root}",
        flush=True,
    )
    print("=" * 72, flush=True)

    t0 = time.time()
    total_saved = 0
    errors = 0

    def _run_event(ev: dict) -> dict:
        return process_mcd_event_gee(
            ev, raw_root=args.raw_root, max_olci_frames=args.max_olci_frames
        )

    if workers == 1:
        for idx, ev in enumerate(series_list, 1):
            sid = ev["series_id"]
            try:
                res = _run_event(ev)
                total_saved += res["frames_saved"]
                print(
                    f"[{idx:>3}/{len(series_list)}] {sid:<20} -> "
                    f"{res['frames_saved']}/{res['total_olci_candidates']} frames OLCI válidos "
                    f"(swath descartados={res['frames_skipped_swath_nodata']})",
                    flush=True,
                )
            except Exception as exc:
                errors += 1
                print(f"[{idx:>3}/{len(series_list)}] {sid:<20} -> ! Error: {exc}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_run_event, ev): ev["series_id"] for ev in series_list}
            for idx, fut in enumerate(as_completed(future_map), 1):
                sid = future_map[fut]
                try:
                    res = fut.result()
                    total_saved += res["frames_saved"]
                    print(
                        f"[{idx:>3}/{len(series_list)}] {sid:<20} -> "
                        f"{res['frames_saved']}/{res['total_olci_candidates']} frames OLCI válidos "
                        f"(swath descartados={res['frames_skipped_swath_nodata']})",
                        flush=True,
                    )
                except Exception as exc:
                    errors += 1
                    print(f"[{idx:>3}/{len(series_list)}] {sid:<20} -> ! Error: {exc}", flush=True)

    print("=" * 72, flush=True)
    print(
        f"Completado en {time.time() - t0:.1f} s | "
        f"Frames OLCI guardados: {total_saved} | Errores: {errors}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
