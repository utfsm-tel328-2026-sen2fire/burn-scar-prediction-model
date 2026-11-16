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
from collections.abc import Mapping
from pathlib import Path
import ee
import numpy as np
import rasterio

# Límite de imágenes devueltas por la consulta de la colección OLCI. Es una
# salvaguarda, no un recorte con significado: si se alcanza, el inventario de la
# serie puede estar incompleto y se avisa por stdout.
IMAGE_QUERY_LIMIT = 500

# Reintentos de la consulta: la IAM de Google se propaga de forma gradual y
# puede devolver un error de permisos que se resuelve solo en unos segundos.
QUERY_RETRIES = 4
QUERY_RETRY_DELAY_S = 10

# Las 5 bandas seleccionadas por TMB-S3 (BL2)
S3_BANDS = [
    "Oa04_radiance",  # 490 nm (azul)
    "Oa06_radiance",  # 620 nm (rojo)
    "Oa08_radiance",  # 665 nm (rojo oscuro)
    "Oa17_radiance",  # 865 nm (NIR)
    "Oa21_radiance",  # 1020 nm (NIR lejano)
]


def wkt_to_geojson(wkt: str) -> dict:
    """
    Convierte una geometría WKT en un GeoJSON que `ee.Geometry` entienda.

    El manifiesto guarda el bbox del AOI en WKT (lo produce CDSE), y la API de
    Earth Engine espera GeoJSON **como diccionario** (su validador accede a
    `geo_json['type']`, así que una cadena JSON no vale).
    """
    from shapely.geometry import mapping
    from shapely.wkt import loads

    geojson = dict(mapping(loads(wkt)))
    # `mapping` devuelve tuplas; el GeoJSON de verdad usa listas.
    geojson["coordinates"] = _a_listas(geojson["coordinates"])
    return geojson


def _a_listas(valor):
    """Convierte recursivamente tuplas en listas (GeoJSON no admite tuplas)."""
    if isinstance(valor, (list, tuple)):
        return [_a_listas(v) for v in valor]
    return valor


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


def load_dotenv_file(env_path: Path | None = None) -> dict[str, str]:
    """
    Lee pares KEY=VALUE de un archivo `.env` sin dependencias externas.

    Las variables ya presentes en el entorno tienen prioridad (lo mismo que en
    `burn_scar.tracking`). Si una clave aparece repetida con valores distintos,
    gana el último y se avisa por stderr: si no, el valor efectivo es una
    sorpresa.
    """
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
        if key in values and values[key] != val:
            print(
                f"AVISO: {path}: '{key}' esta definida mas de una vez con valores "
                f"distintos; se usa el ultimo ('{val}').",
                file=sys.stderr,
            )
        values[key] = val
    return values


def resolve_ee_credentials(env: Mapping[str, str]) -> dict[str, str]:
    """
    Resuelve la configuracion de Earth Engine desde el entorno.

    - `EARTHENGINE_SERVICE_ACCOUNT`: ruta a la clave de una cuenta de servicio
      (JSON o PEM). Si esta definida, se autentica con ella.
    - `EARTHENGINE_PROJECT` (o `GOOGLE_CLOUD_PROJECT`): proyecto de Google
      Cloud con la API de Earth Engine activada.

    Sin ninguna de las dos, el comportamiento es el de siempre: credenciales
    persistentes del usuario (`earthengine authenticate`) y el proyecto que
    tenga configurado (`earthengine set_project`).
    """
    return {
        "project": env.get("EARTHENGINE_PROJECT") or env.get("GOOGLE_CLOUD_PROJECT") or "",
        "service_account": env.get("EARTHENGINE_SERVICE_ACCOUNT") or "",
    }


def init_earth_engine(env: Mapping[str, str]) -> None:
    """Inicializa Earth Engine con la configuracion resuelta del entorno."""
    config = resolve_ee_credentials(env)
    project = config["project"] or None
    key_file = config["service_account"] or None

    try:
        if key_file:
            if not Path(key_file).is_file():
                raise FileNotFoundError(
                    f"EARTHENGINE_SERVICE_ACCOUNT apunta a {key_file}, que no existe."
                )
            ee.Initialize(
                credentials=ee.ServiceAccountCredentials(key_file=key_file),
                project=project,
            )
        else:
            # project=None deja que la API use el proyecto de las credenciales
            # persistentes: identico a la llamada ee.Initialize() sin argumentos.
            ee.Initialize(project=project)
    except Exception as exc:
        raise SystemExit(
            "No se pudo inicializar Earth Engine.\n"
            "  - Autentica con tu cuenta:      earthengine authenticate\n"
            "  - O fija el proyecto:            earthengine set_project <proyecto>\n"
            "  - O usa una cuenta de servicio:  EARTHENGINE_SERVICE_ACCOUNT=<ruta.json>\n"
            "                                   EARTHENGINE_PROJECT=<proyecto>\n"
            f"Detalle: {exc}"
        ) from exc


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

    # Cobertura de la consulta: la extensión del área de interés (el bbox del
    # manifiesto), NO el punto del centroide. El catálogo de referencia (CDSE)
    # consulta por polígono y OLCI se distribuye en tiles fijos: filtrar por un
    # punto excluye las escenas de S3B cuyo tile no cubre el centroide aunque sí
    # el AOI, que es lo que dejaba sin adquisición de f a 125 de las 390 muestras.
    bbox_wkt = series_meta.get("bbox_wkt")
    query_geom = ee.Geometry(wkt_to_geojson(bbox_wkt)) if bbox_wkt else point

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
        .filterBounds(query_geom)
        .filterDate(start_str, end_str)
        .select(S3_BANDS)
    )

    # Obtener lista de imágenes con sus fechas. La IAM de Google puede estar
    # propagando y devolver un error transitorio de permisos, así que se reintenta
    # unas veces antes de renunciar con la serie.
    img_list = None
    for intento in range(QUERY_RETRIES):
        try:
            img_list = s3_col.toList(IMAGE_QUERY_LIMIT).getInfo()
            break
        except Exception as exc:
            if intento == QUERY_RETRIES - 1:
                raise
            print(
                f"   ! {series_id}: la consulta falló ({type(exc).__name__}); "
                f"reintento {intento + 2}/{QUERY_RETRIES}"
            )
            time.sleep(QUERY_RETRY_DELAY_S)

    if len(img_list) >= IMAGE_QUERY_LIMIT:
        print(
            f"   ! AVISO: {series_id} alcanzó el límite de {IMAGE_QUERY_LIMIT} imágenes "
            f"en la consulta; su inventario puede estar incompleto."
        )
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

    # El entorno real manda sobre el .env: las variables ya exportadas ganan.
    env = {**load_dotenv_file(), **os.environ}
    init_earth_engine(env)

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
    series_fallidas: list[str] = []

    for idx, s in enumerate(series_list, 1):
        sid = s["series_id"]
        # Una serie que falla no debe tumbar una descarga de 40 minutos: se
        # registra y se sigue con la siguiente.
        try:
            res = process_series_gee(s, args.olci_dir, args.static_dir)
        except Exception as exc:
            series_fallidas.append(sid)
            print(f"[{idx:>2}/{len(series_list)}] {sid:<18} -> ERROR: {type(exc).__name__}: "
                  f"{str(exc)[:200]}")
            continue
        total_frames += res["frames_downloaded"]
        print(f"[{idx:>2}/{len(series_list)}] {sid:<18} -> "
              f"{res['frames_downloaded']}/{res.get('total_available', 0)} frames OLCI descargados")

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("RESUMEN DE DESCARGA (PASO 4)")
    print("=" * 72)
    print(f"Series procesadas:               {len(series_list)}")
    print(f"Total de frames OLCI guardados:  {total_frames}")
    if series_fallidas:
        print(f"Series con error ({len(series_fallidas)}): {', '.join(series_fallidas)}")
        print("  Su inventario puede quedar incompleto; relanza el script para reintentarlas.")
    print(f"Tiempo total:                    {elapsed:.2f} s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
