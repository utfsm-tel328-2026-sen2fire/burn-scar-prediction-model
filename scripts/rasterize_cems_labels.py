#!/usr/bin/env python3
"""
Paso 2: Rasterización de etiquetas CEMS a la rejilla de 300 m (64x64 px).

Implementa las especificaciones exactas del proyecto:
- D17: Ventana fija de 64x64 px a 300 m (19.2 x 19.2 km) centrada en el fuego.
- D2: Etiqueta fraccional de quemado en [0..1] (promedio de subpíxeles de 30 m).
- D3: Máscara tri-estado de validez (1 = observado/válido, 255 = no analizado/fuera de AOI).

Salida:
    data/processed/cems_labels/{series_id}.npz
    data/processed/cems_labels/manifest_rasterized.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import numpy as np
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import shapely
from shapely.geometry import shape, box, Polygon, MultiPolygon


def get_utm_epsg(lon: float, lat: float) -> int:
    """Calcula el código EPSG de la zona UTM para (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def transform_geom_to_utm(geom: shapely.Geometry, transformer: pyproj.Transformer) -> shapely.Geometry | None:
    """Reproyecta y valida una geometría a UTM."""
    try:
        if geom.is_empty:
            return None
        valid_geom = shapely.make_valid(geom)
        if valid_geom.is_empty:
            return None
        return shapely.transform(
            valid_geom,
            lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1]))
        )
    except Exception:
        return None


def process_series(
    series_meta: dict,
    raw_dir: str,
    output_dir: str,
) -> dict | None:
    series_id = series_meta["series_id"]
    series_raw_path = os.path.join(raw_dir, series_id)

    if not os.path.exists(series_raw_path):
        return None

    lon_c, lat_c = series_meta["center"]
    epsg_utm = get_utm_epsg(lon_c, lat_c)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
    x_c, y_c = transformer.transform(lon_c, lat_c)

    # Ventana fija D17: 64 px a 300 m = 19,200 m (9600 m a cada lado del centroide)
    grid_size_px = 64
    res_m = 300.0
    half_width = (grid_size_px * res_m) / 2.0  # 9600.0 m

    x_min, x_max = x_c - half_width, x_c + half_width
    y_min, y_max = y_c - half_width, y_c + half_width

    # Sub-rejilla 10x10 para estimación fraccional precisa (subpíxeles de 30 m -> 640x640)
    sub_factor = 10
    sub_size_px = grid_size_px * sub_factor  # 640
    sub_res_m = res_m / sub_factor  # 30.0 m
    sub_transform = Affine(sub_res_m, 0.0, x_min, 0.0, -sub_res_m, y_max)
    grid_transform_300m = Affine(res_m, 0.0, x_min, 0.0, -res_m, y_max)

    grid_poly_utm = box(x_min, y_min, x_max, y_max)

    rounds = series_meta.get("rounds", [])
    if len(rounds) < 2:
        return None

    burnt_layers = []
    valid_layers = []
    round_numbers = []
    round_meta = []

    for r in rounds:
        r_num = r["monitoring_number"]
        event_path = os.path.join(series_raw_path, f"r{r_num:02d}_observedEventA.geojson")
        not_analysed_path = os.path.join(series_raw_path, f"r{r_num:02d}_notAnalysedA.geojson")

        if not os.path.exists(event_path):
            continue

        # 1. Cargar geometrías de fuego (observedEventA)
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                event_data = json.load(f)
        except Exception:
            continue

        event_geoms_utm = []
        for feat in event_data.get("features", []):
            g = feat.get("geometry")
            if g:
                utm_g = transform_geom_to_utm(shape(g), transformer)
                if utm_g and not utm_g.is_empty and utm_g.intersects(grid_poly_utm):
                    event_geoms_utm.append(utm_g)

        # Rasterizar fuego en sub-rejilla
        if event_geoms_utm:
            sub_burnt = rasterize(
                [(g, 1) for g in event_geoms_utm],
                out_shape=(sub_size_px, sub_size_px),
                transform=sub_transform,
                fill=0,
                dtype=np.uint8,
            )
        else:
            sub_burnt = np.zeros((sub_size_px, sub_size_px), dtype=np.uint8)

        # Promediar bloques 10x10 -> fracción quemada en [0..1] (D2)
        burnt_frac = sub_burnt.reshape(grid_size_px, sub_factor, grid_size_px, sub_factor).mean(axis=(1, 3))
        burnt_frac = np.clip(burnt_frac, 0.0, 1.0).astype(np.float16)

        # 2. Cargar geometrías de no analizado (notAnalysedA)
        not_analysed_geoms_utm = []
        if os.path.exists(not_analysed_path):
            try:
                with open(not_analysed_path, "r", encoding="utf-8") as f:
                    na_data = json.load(f)
                for feat in na_data.get("features", []):
                    g = feat.get("geometry")
                    if g:
                        utm_g = transform_geom_to_utm(shape(g), transformer)
                        if utm_g and not utm_g.is_empty and utm_g.intersects(grid_poly_utm):
                            not_analysed_geoms_utm.append(utm_g)
            except Exception:
                pass

        if not_analysed_geoms_utm:
            sub_na = rasterize(
                [(g, 1) for g in not_analysed_geoms_utm],
                out_shape=(sub_size_px, sub_size_px),
                transform=sub_transform,
                fill=0,
                dtype=np.uint8,
            )
            na_frac = sub_na.reshape(grid_size_px, sub_factor, grid_size_px, sub_factor).mean(axis=(1, 3))
            # Si más del 50% del píxel no fue analizado, marcar como 255 (D3)
            valido_mask = np.where(na_frac >= 0.5, 255, 1).astype(np.uint8)
        else:
            valido_mask = np.ones((grid_size_px, grid_size_px), dtype=np.uint8)

        burnt_layers.append(burnt_frac)
        valid_layers.append(valido_mask)
        round_numbers.append(r_num)
        round_meta.append({
            "monitoring_number": r_num,
            "sensor": r.get("sensor_name"),
            "acquisition_time": r.get("acquisition_time"),
            "burnt_pixels_gt0": int((burnt_frac > 0).sum()),
            "burnt_area_km2": float(round(burnt_frac.sum() * 0.09, 3)),
        })

    if len(burnt_layers) < 2:
        return None

    # Tensores apilados: shape (T_rondas, 64, 64)
    arr_burnt = np.stack(burnt_layers, axis=0)
    arr_valid = np.stack(valid_layers, axis=0)
    arr_rounds = np.array(round_numbers, dtype=np.int32)

    meta_payload = {
        "series_id": series_id,
        "name": series_meta.get("name"),
        "code": series_meta.get("code"),
        "aoi_number": series_meta.get("aoi_number"),
        "geo_mediterranean": series_meta.get("geo_mediterranean", True),
        "in_tmb_test": series_meta.get("in_tmb_test", False),
        "epsg_utm": epsg_utm,
        "center_wgs84": [lon_c, lat_c],
        "center_utm": [float(x_c), float(y_c)],
        "bbox_utm": [float(x_min), float(y_min), float(x_max), float(y_max)],
        "grid_shape": [grid_size_px, grid_size_px],
        "resolution_m": res_m,
        "affine_transform": list(grid_transform_300m[:6]),
        "rounds": round_meta,
    }

    # Guardar en npz comprimido
    out_file = os.path.join(output_dir, f"{series_id}.npz")
    np.savez_compressed(
        out_file,
        quemado=arr_burnt,
        valido=arr_valid,
        round_numbers=arr_rounds,
        meta=json.dumps(meta_payload),
    )

    return meta_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Paso 2: Rasterización de etiquetas CEMS a 300m")
    parser.add_argument("--manifest", default="data/cems_manifest_46.json", help="Ruta al manifiesto")
    parser.add_argument("--raw-dir", default="data/raw/cems", help="Directorio con los GeoJSON")
    parser.add_argument("--output-dir", default="data/processed/cems_labels", help="Directorio destino .npz")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: {args.manifest} no existe.", file=sys.stderr)
        return 1

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    series_list = manifest.get("series", [])
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print(f"PASO 2: RASTERIZACIÓN DE ETIQUETAS CEMS (300 m, 64x64 px)")
    print(f"Entrada: {args.raw_dir}")
    print(f"Destino: {args.output_dir}")
    print(f"Total series a procesar: {len(series_list)}")
    print("=" * 72)

    start_time = time.time()
    processed_series = []
    total_rounds_rasterized = 0

    for idx, s in enumerate(series_list, 1):
        series_id = s["series_id"]
        res = process_series(s, args.raw_dir, args.output_dir)
        if res:
            n_r = len(res["rounds"])
            total_rounds_rasterized += n_r
            processed_series.append(res)
            print(f"[{idx:>2}/{len(series_list)}] {series_id:<18} -> {n_r} rondas rasterizadas con éxito")
        else:
            print(f"[{idx:>2}/{len(series_list)}] {series_id:<18} -> ! Omitida (datos insuficientes)")

    elapsed = time.time() - start_time

    # Guardar índice de etiquetas rasterizadas
    manifest_out = os.path.join(args.output_dir, "manifest_rasterized.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_series": len(processed_series),
            "total_rounds": total_rounds_rasterized,
            "grid_spec": {
                "shape": [64, 64],
                "resolution_m": 300.0,
                "subgrid_resolution_m": 30.0,
                "fractional_quantization": "average_subgrid_10x10",
                "validity_unobserved_code": 255,
            },
            "series": processed_series,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("RESUMEN DE RASTERIZACIÓN (PASO 2)")
    print("=" * 72)
    print(f"Series procesadas y guardadas:  {len(processed_series)} / {len(series_list)}")
    print(f"Total de mapas rasterizados:   {total_rounds_rasterized} rondas")
    print(f"Archivos .npz generados en:    {args.output_dir}")
    print(f"Índice del dataset guardado en:{manifest_out}")
    print(f"Tiempo total:                  {elapsed:.2f} s")

    # Tamaño total en disco
    total_bytes = sum(os.path.getsize(os.path.join(args.output_dir, f))
                      for f in os.listdir(args.output_dir) if f.endswith(".npz"))
    print(f"Espacio total de etiquetas:    {total_bytes / (1024 * 1024):.2f} MB")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
