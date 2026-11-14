#!/usr/bin/env python3
"""
Paso 1: Descarga de capas vectoriales CEMS (GeoJSON).

Descarga directamente desde Amazon S3 (sin necesidad de autenticación)
todas las geometrías de las 44 series limpias especificadas en:
    data/cems_manifest_46.json

Guarda las capas en:
    data/raw/cems/{series_id}/r{round_number:02d}_{layer_name}.geojson
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def download_file(url: str, dest_path: str, retries: int = 4, timeout: int = 30) -> tuple[str, bool, str]:
    """Descarga un archivo con reintentos y validación de JSON."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10:
        return dest_path, True, "cached"

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = f"{dest_path}.tmp"
    delay = 1.0

    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "burn-scar-downloader/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                content = resp.read()

            # Validar que no esté vacío ni corrupto
            if len(content) < 5:
                raise ValueError("Archivo demasiado pequeño")

            # Comprobar que sea JSON válido
            json.loads(content.decode("utf-8"))

            with open(temp_path, "wb") as f:
                f.write(content)
            os.replace(temp_path, dest_path)
            return dest_path, True, "downloaded"

        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 1.5
            else:
                return dest_path, False, str(exc)

    return dest_path, False, "unknown error"


def extract_layer_key(layer_name: str) -> str:
    """Extrae una clave estandarizada para el nombre del archivo."""
    for key in ("observedEventA", "notAnalysedA", "observedEventP", "observedEventL"):
        if key in layer_name:
            return key
    parts = layer_name.split("_")
    return parts[-2] if len(parts) >= 2 else parts[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Paso 1: Descarga de capas vectoriales CEMS")
    parser.add_argument("--manifest", default="data/cems_manifest_46.json", help="Ruta al manifiesto")
    parser.add_argument("--output-dir", default="data/raw/cems", help="Directorio de destino")
    parser.add_argument("--workers", type=int, default=8, help="Número de hilos concurrentes")
    parser.add_argument("--limit", type=int, default=None, help="Límite de series (para testing)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: {args.manifest} no existe.", file=sys.stderr)
        return 1

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    series_list = manifest.get("series", [])
    if args.limit:
        series_list = series_list[:args.limit]

    tasks = []
    for s in series_list:
        series_id = s["series_id"]
        for r in s.get("rounds", []):
            round_num = r.get("monitoring_number", 0)
            for layer_name, url in r.get("vector_layers", {}).items():
                layer_key = extract_layer_key(layer_name)
                filename = f"r{round_num:02d}_{layer_key}.geojson"
                dest_path = os.path.join(args.output_dir, series_id, filename)
                tasks.append((url, dest_path, series_id, round_num, layer_key))

    print("=" * 72)
    print(f"PASO 1: DESCARGA DE GEOMETRÍAS CEMS ({len(tasks)} ARCHIVOS)")
    print(f"Destino: {args.output_dir}")
    print(f"Hilos concurrentes: {args.workers}")
    print("=" * 72)

    downloaded = 0
    cached = 0
    failed = 0
    errors = []

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(download_file, url, dest): (url, dest, sid, r_num, l_key)
            for url, dest, sid, r_num, l_key in tasks
        }

        for idx, future in enumerate(concurrent.futures.as_completed(future_to_task), 1):
            url, dest, sid, r_num, l_key = future_to_task[future]
            try:
                path, success, status = future.result()
                if success:
                    if status == "cached":
                        cached += 1
                    else:
                        downloaded += 1
                else:
                    failed += 1
                    errors.append((url, status))
            except Exception as e:
                failed += 1
                errors.append((url, str(e)))

            if idx % 50 == 0 or idx == len(tasks):
                elapsed = time.time() - start_time
                print(f"Progreso: {idx:>3}/{len(tasks)} archivos procesados "
                      f"({downloaded} descargados, {cached} en caché, {failed} fallidos) - {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 72)
    print("RESUMEN DE DESCARGA (PASO 1)")
    print("=" * 72)
    print(f"Total archivos planificados: {len(tasks)}")
    print(f"Archivos descargados con éxito: {downloaded}")
    print(f"Archivos previamente en caché:  {cached}")
    print(f"Archivos fallidos:              {failed}")
    print(f"Tiempo total:                   {elapsed_total:.2f} s")

    if errors:
        print("\nErrores detectados:")
        for url, err in errors[:10]:
            print(f"  - {url} -> {err}")

    # Verificar espacio ocupado
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(args.output_dir):
        for f in files:
            if f.endswith(".geojson"):
                file_count += 1
                total_bytes += os.path.getsize(os.path.join(root, f))

    print(f"\nDirectorio local: {args.output_dir}")
    print(f"Archivos GeoJSON presentes: {file_count}")
    print(f"Espacio total en disco:     {total_bytes / (1024 * 1024):.2f} MB")
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
