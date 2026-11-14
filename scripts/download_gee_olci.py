#!/usr/bin/env python3
"""
Paso 4: Descarga de recortes Sentinel-3 OLCI (5 bandas) y capas estáticas vía GEE.

Descarga parches de 64x64 píxeles a 300 m directamente desde Google Earth Engine:
- Sentinel-3 OLCI: 5 bandas (Oa04, Oa06, Oa08, Oa17, Oa21) (~25 KB por frame).
- Copernicus DEM (30 m remuestreado a 300 m en 64x64 px).
- ESA WorldCover 2021 (10 m remuestreado a 300 m en 64x64 px).

Salida:
    data/raw/s3_olci/{series_id}/{frame_date}.npz
    data/raw/static/{series_id}/static_layers.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import ee
import numpy as np
import rasterio

# Las 5 bandas seleccionadas por TMB-S3 (BL2)
S3_BANDS = [
    "Oa04_radiance",  # 490 nm (azul)
    "Oa06_radiance",  # 620 nm (rojo)
    "Oa08_radiance",  # 665 nm (rojo oscuro)
    "Oa17_radiance",  # 865 nm (NIR)
    "Oa21_radiance",  # 1020 nm (NIR lejano)
]


def download_gee_patch(image: ee.Image, region: ee.Geometry, crs: str, dimensions: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Descarga una imagen de GEE como un array NumPy con dimensiones exactas."""
    url = image.getDownloadURL({
        "region": region,
        "dimensions": list(dimensions),
        "crs": crs,
        "format": "GEO_TIFF",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "burn-scar-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    with rasterio.open(io.BytesIO(data)) as src:
        arr = src.read()
    return arr


def process_series_gee(
    series_meta: dict,
    olci_out_dir: str,
    static_out_dir: str,
) -> dict:
    series_id = series_meta["series_id"]
    lon_c, lat_c = series_meta["center"]
    epsg_utm = series_meta.get("epsg_utm")
    if not epsg_utm:
        zone = int((lon_c + 180) // 6) + 1
        epsg_utm = (32600 + zone) if lat_c >= 0 else (32700 + zone)

    crs_str = f"EPSG:{epsg_utm}"

    # Región de 19.2 km (9600 m alrededor del centroide)
    point = ee.Geometry.Point([lon_c, lat_c])
    region = point.buffer(9600).bounds()

    # 1. Descargar capas estáticas (DEM y WorldCover) si no existen
    os.makedirs(os.path.join(static_out_dir, series_id), exist_ok=True)
    static_file = os.path.join(static_out_dir, series_id, "static_layers.npz")

    if not os.path.exists(static_file):
        try:
            # DEM GLO-30
            dem_img = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic()
            dem_arr = download_gee_patch(dem_img, region, crs_str)[0].astype(np.float32)

            # ESA WorldCover 2021
            wc_img = ee.ImageCollection("ESA/WorldCover/v200").select("Map").mosaic()
            wc_arr = download_gee_patch(wc_img, region, crs_str)[0].astype(np.uint8)

            np.savez_compressed(static_file, dem=dem_arr, worldcover=wc_arr)
        except Exception as e:
            print(f"   ! Error descargando estáticas de {series_id}: {e}")

    # 2. Descargar frames Sentinel-3 OLCI
    series_olci_dir = os.path.join(olci_out_dir, series_id)
    os.makedirs(series_olci_dir, exist_ok=True)

    rounds = series_meta.get("rounds", [])
    if not rounds:
        return {"series_id": series_id, "frames_downloaded": 0}

    # Intervalo de fechas: 14 días antes de la primera ronda hasta la última ronda + 1 día
    acq_times = [r["acquisition_time"] for r in rounds if r.get("acquisition_time")]
    if not acq_times:
        return {"series_id": series_id, "frames_downloaded": 0}

    acq_dts = [dt.datetime.fromisoformat(t.replace("Z", "")) for t in acq_times]
    start_dt = min(acq_dts) - dt.timedelta(days=14)
    end_dt = max(acq_dts) + dt.timedelta(days=1)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # Consultar colección OLCI filtrada por zona y fecha
    s3_col = (
        ee.ImageCollection("COPERNICUS/S3/OLCI")
        .filterBounds(point)
        .filterDate(start_str, end_str)
        .select(S3_BANDS)
    )

    # Obtener lista de imágenes con sus fechas
    img_list = s3_col.toList(60).getInfo()
    downloaded_frames = 0

    for item in img_list:
        img_id = item["id"]
        props = item.get("properties", {})
        time_start = props.get("system:time_start", 0)
        frame_dt = dt.datetime.fromtimestamp(time_start / 1000.0, tz=dt.timezone.utc)
        date_tag = frame_dt.strftime("%Y%m%dT%H%M%S")

        frame_path = os.path.join(series_olci_dir, f"{date_tag}.npz")
        if os.path.exists(frame_path):
            downloaded_frames += 1
            continue

        try:
            ee_img = ee.Image(img_id).select(S3_BANDS)
            bands_arr = download_gee_patch(ee_img, region, crs_str)
            # bands_arr shape: (5, 64, 64)
            np.savez_compressed(
                frame_path,
                olci_bands=bands_arr.astype(np.uint16),
                bands=S3_BANDS,
                timestamp=frame_dt.isoformat(),
                image_id=img_id,
            )
            downloaded_frames += 1
        except Exception as e:
            print(f"   ! Error descargando frame {date_tag}: {e}")

    return {
        "series_id": series_id,
        "frames_downloaded": downloaded_frames,
        "total_available": len(img_list),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paso 4: Descarga de Sentinel-3 OLCI vía GEE")
    parser.add_argument("--manifest", default="data/cems_manifest_46.json", help="Ruta al manifiesto")
    parser.add_argument("--olci-dir", default="data/raw/s3_olci", help="Directorio destino OLCI")
    parser.add_argument("--static-dir", default="data/raw/static", help="Directorio destino estáticas")
    parser.add_argument("--limit", type=int, default=None, help="Límite de series (para testing)")
    args = parser.parse_args()

    ee.Initialize()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    series_list = manifest.get("series", [])
    if args.limit:
        series_list = series_list[:args.limit]

    print("=" * 72)
    print(f"PASO 4: DESCARGA DE SENTINEL-3 OLCI (5 BANDAS) VÍA GEE")
    print(f"Series a procesar: {len(series_list)}")
    print(f"Destino OLCI: {args.olci_dir}")
    print(f"Destino Estáticas (DEM/WorldCover): {args.static_dir}")
    print("=" * 72)

    start_time = time.time()
    total_frames = 0

    for idx, s in enumerate(series_list, 1):
        sid = s["series_id"]
        res = process_series_gee(s, args.olci_dir, args.static_dir)
        total_frames += res["frames_downloaded"]
        print(f"[{idx:>2}/{len(series_list)}] {sid:<18} -> "
              f"{res['frames_downloaded']}/{res.get('total_available', 0)} frames OLCI descargados")

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("RESUMEN DE DESCARGA (PASO 4)")
    print("=" * 72)
    print(f"Series procesadas:               {len(series_list)}")
    print(f"Total de frames OLCI guardados:  {total_frames}")
    print(f"Tiempo total:                    {elapsed:.2f} s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
