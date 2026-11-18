#!/usr/bin/env python3
"""
Construcción del manifiesto de muestras CEMS + Sentinel-3 OLCI (P7 y R14).

Cruza las 46 series limpias de CEMS (audit_results.json) con el catálogo
abierto de Copernicus Data Space Ecosystem (CDSE) para:
1. Censar el número real de muestras con la regla "una muestra por frame OLCI" (P7).
2. Auditar la continuidad de sensores entre rondas (R14 y D4).
3. Dejar listo el manifiesto estructurado para la descarga de imagen y rasterizado.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CEMS_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

NUM_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def get_json(url: str, tries: int = 5, timeout: int = 30) -> dict | None:
    """GET con reintentos y backoff exponencial."""
    delay = 1.0
    for _ in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "burn-scar-manifest/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                time.sleep(delay)
                delay *= 1.8
                continue
            return None
        except (URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(delay)
            delay *= 1.8
    return None


def parse_iso(ts_str: str) -> dt.datetime | None:
    """Parsea fecha ISO tolerando zona horaria o microsegundos."""
    if not ts_str:
        return None
    # Eliminar 'Z' o zona si existe para estandarizar en naive UTC
    s = ts_str.replace("Z", "").split("+")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def canonical_acquisition_key(satellite: str, instant: dt.datetime) -> str:
    """Clave canónica de una adquisición OLCI: satélite y segundo UTC."""
    return f"{satellite}:{instant.strftime('%Y%m%dT%H%M%S')}"


def deduplicate_olci_scenes(scenes: list[dict]) -> list[dict]:
    """
    Colapsa productos CDSE que representan la misma adquisición.

    La selección es determinista por identificador de producto y todos los IDs
    alternativos quedan en `product_ids` para auditoría.
    """
    groups: dict[str, list[dict]] = {}
    for scene in scenes:
        key = canonical_acquisition_key(scene["satellite"], scene["dt"])
        groups.setdefault(key, []).append(scene)

    canonical: list[dict] = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda scene: str(scene.get("id") or ""))
        representative = dict(candidates[0])
        product_ids = sorted(str(scene.get("id") or "") for scene in candidates)
        representative["acquisition_key"] = key
        representative["product_ids"] = product_ids
        representative["collapsed_product_ids"] = [
            product_id for product_id in product_ids if product_id != representative["id"]
        ]
        canonical.append(representative)

    canonical.sort(key=lambda scene: (scene["dt"], scene["acquisition_key"]))
    return canonical


def deduplicate_manifest_samples(series_list: list[dict]) -> dict[str, int]:
    """Colapsa entradas de muestra repetidas dentro de cada serie."""
    products_total = 0
    acquisitions_unique = 0
    acquisitions_collapsed = 0
    products_collapsed = 0

    for series in series_list:
        grouped: dict[str, list[dict]] = {}
        series_products_total = len(series.get("samples", []))
        for sample in series.get("samples", []):
            sample_id = str(sample.get("sample_id"))
            grouped.setdefault(sample_id, []).append(sample)
            products_total += 1

        # Se conserva el orden de aparición (cronológico) y, dentro de cada
        # `sample_id`, se elige un representante determinista.
        canonical_samples = []
        for _sample_id, candidates in grouped.items():
            chosen = sorted(
                candidates,
                key=lambda sample: str(sample.get("frame_scene_id") or sample.get("id") or ""),
            )[0]
            canonical_samples.append(chosen)
            products_collapsed += len(candidates) - 1

        series["samples"] = canonical_samples
        acquisitions_unique += len(canonical_samples)
        if len(canonical_samples) < series_products_total:
            acquisitions_collapsed += 1

    return {
        "products_total": products_total,
        "acquisitions_unique": acquisitions_unique,
        "acquisitions_collapsed": acquisitions_collapsed,
        "products_collapsed": products_collapsed,
    }


def summarize_olci_catalog(scenes: list[dict]) -> dict[str, int]:
    """Resumen auditable de productos CDSE frente a adquisiciones únicas."""
    products_total = sum(len(scene.get("product_ids", [scene.get("id")])) for scene in scenes)
    products_collapsed = sum(
        len(scene.get("collapsed_product_ids", [])) for scene in scenes
    )
    return {
        "products_total": products_total,
        "acquisitions_unique": len(scenes),
        "acquisitions_collapsed": sum(
            bool(scene.get("collapsed_product_ids")) for scene in scenes
        ),
        "products_collapsed": products_collapsed,
    }


def wkt_to_poly_coords(wkt: str) -> list[tuple[float, float]] | None:
    """Extrae las coordenadas de un POLYGON WKT."""
    pairs = NUM_PAIR.findall(wkt)
    if not pairs:
        return None
    return [(float(x), float(y)) for x, y in pairs]


def make_bbox_poly_wkt(coords: list[tuple[float, float]], buffer_deg: float = 0.05) -> str:
    """Genera un WKT POLYGON envolvente a partir de puntos."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x = min(xs) - buffer_deg, max(xs) + buffer_deg
    min_y, max_y = min(ys) - buffer_deg, max(ys) + buffer_deg
    return (
        f"POLYGON(({min_x:.4f} {min_y:.4f}, {max_x:.4f} {min_y:.4f}, "
        f"{max_x:.4f} {max_y:.4f}, {min_x:.4f} {max_y:.4f}, {min_x:.4f} {min_y:.4f}))"
    )


def fetch_cems_detail(code: str) -> dict | None:
    url = f"{CEMS_API}/public-activations/?code={code}"
    data = get_json(url)
    if data and data.get("results"):
        return data["results"][0]
    return None


def fetch_cdse_olci_scenes(
    poly_wkt: str,
    date_start: dt.datetime,
    date_end: dt.datetime,
) -> list[dict]:
    """Consulta las pasadas Sentinel-3 OLCI L1 EFR en CDSE OData."""
    t0_str = date_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    t1_str = date_end.strftime("%Y-%m-%dT%H:%M:%S.999Z")

    filter_expr = (
        f"Collection/Name eq 'SENTINEL-3' and contains(Name,'OL_1_EFR') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{poly_wkt}') "
        f"and ContentDate/Start gt {t0_str} and ContentDate/Start lt {t1_str}"
    )
    params = urllib.parse.urlencode(
        {"$filter": filter_expr, "$top": 100, "$orderby": "ContentDate/Start asc"}
    )
    url = f"{CDSE_ODATA}?{params}"

    data = get_json(url)
    if not data:
        return []

    scenes = []
    for item in data.get("value", []):
        t_acq = parse_iso(item.get("ContentDate", {}).get("Start"))
        if t_acq:
            scenes.append({
                "id": item.get("Id"),
                "name": item.get("Name"),
                "satellite": "S3A" if item.get("Name", "").startswith("S3A") else "S3B",
                "acquisition_time": t_acq.isoformat(),
                "dt": t_acq,
            })
    return deduplicate_olci_scenes(scenes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifiesto CEMS + Sentinel-3 OLCI (P7 y R14)")
    parser.add_argument(
        "--audit-json", default="runs/audit_results.json", help="Ruta a audit_results.json"
    )
    parser.add_argument(
        "--output", default="data/cems_manifest_46.json", help="Ruta del manifiesto de salida"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Límite de series a procesar (para testing)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.audit_json):
        print(
            f"Error: {args.audit_json} no existe. Ejecuta audit_tmb_splits.py primero.",
            file=sys.stderr,
        )
        return 1

    with open(args.audit_json, encoding="utf-8") as f:
        audit_data = json.load(f)

    all_series = audit_data.get("series", [])
    usable = [s for s in all_series if not s.get("in_tmb") or s.get("tmb_split") == "test"]
    if args.limit:
        usable = usable[:args.limit]

    print("=" * 72)
    print(f"PROCESANDO METADATOS PARA {len(usable)} SERIES UTILIZABLES")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    manifest_series = []
    sensor_counter: dict[str, int] = {}
    transition_sensor_changes = 0
    total_transitions = 0
    total_samples = 0
    horizons_days: list[float] = []

    # Cache de detalles de activación para evitar repetir llamadas al mismo EMSR con múltiples AOIs
    activation_cache: dict[str, dict] = {}

    for idx, s in enumerate(usable, 1):
        code = s["code"]
        aoi_num = s["aoi"]
        series_id = f"{code}_AOI{aoi_num:02d}"
        is_geo_med = s.get("geo_med", True)

        print(f"[{idx:>2}/{len(usable)}] {series_id} ({s.get('name')}) - GeoMed: {is_geo_med}")

        if code not in activation_cache:
            detail = fetch_cems_detail(code)
            activation_cache[code] = detail or {}
            time.sleep(0.3)
        detail = activation_cache.get(code, {})

        if not detail:
            print(f"   ! No se pudo obtener detalle para {code}")
            continue

        # Buscar el AOI correspondiente
        aoi_obj = next((a for a in detail.get("aois", []) if a.get("number") == aoi_num), None)
        if not aoi_obj:
            print(f"   ! No se encontró AOI {aoi_num} en {code}")
            continue

        extent_wkt = aoi_obj.get("extent", "")
        coords = wkt_to_poly_coords(extent_wkt)
        if not coords:
            print(f"   ! Sin geometría válida para AOI {aoi_num}")
            continue

        aoi_bbox_wkt = make_bbox_poly_wkt(coords, buffer_deg=0.08)

        # Extraer productos DEL (excluyendo FEP según D5)
        del_products = [p for p in (aoi_obj.get("products") or []) if p.get("type") == "DEL"]
        parsed_rounds = []
        for p in del_products:
            imgs = p.get("images", [])
            img0 = imgs[0] if imgs else {}
            sensor = img0.get("sensorName", "Unknown")
            sensor_counter[sensor] = sensor_counter.get(sensor, 0) + 1
            acq_dt = parse_iso(img0.get("acquisitionTime"))

            burnt_ha = 0.0
            stats = p.get("stats") or {}
            burnt_area = stats.get("Burnt area") or {}
            if isinstance(burnt_area, dict):
                none_dict = burnt_area.get("None") or {}
                if isinstance(none_dict, dict):
                    burnt_ha = float(none_dict.get("affected", 0.0) or 0.0)

            # Capas vectoriales
            vt_layers = {
                layer.get("name"): layer.get("json")
                for layer in p.get("layers", [])
                if layer.get("format") == "vt" and layer.get("json")
            }

            parsed_rounds.append({
                "monitoring_number": p.get("monitoringNumber", 0),
                "sensor_name": sensor,
                "resolution_class": img0.get("resolutionClass"),
                "acquisition_time": acq_dt.isoformat() if acq_dt else None,
                "dt": acq_dt,
                "burnt_ha": burnt_ha,
                "vector_layers": vt_layers,
            })

        # Ordenar rondas cronológicamente
        parsed_rounds = [r for r in parsed_rounds if r["dt"] is not None]
        parsed_rounds.sort(key=lambda r: r["dt"])

        if len(parsed_rounds) < 2:
            print(f"   ! Menos de 2 rondas DEL con fecha válida ({len(parsed_rounds)})")
            continue

        first_round_dt = parsed_rounds[0]["dt"]
        last_round_dt = parsed_rounds[-1]["dt"]

        # Pedimos OLCI desde 14 días antes de la primera ronda (para asegurar frame pre-incendio)
        # hasta la última ronda
        search_start = first_round_dt - dt.timedelta(days=14)
        search_end = last_round_dt + dt.timedelta(hours=6)

        olci_scenes = fetch_cdse_olci_scenes(aoi_bbox_wkt, search_start, search_end)
        time.sleep(0.4)  # backoff respetuoso con CDSE

        # Construir transiciones y muestras por frame OLCI
        transitions = []
        series_samples = []

        for r_idx in range(len(parsed_rounds) - 1):
            r_curr = parsed_rounds[r_idx]
            r_next = parsed_rounds[r_idx + 1]

            t_r = r_curr["dt"]
            t_r_next = r_next["dt"]
            dt_hours = (t_r_next - t_r).total_seconds() / 3600.0

            sensor_curr = r_curr["sensor_name"]
            sensor_next = r_next["sensor_name"]
            sensor_changed = (sensor_curr.lower() != sensor_next.lower())

            total_transitions += 1
            if sensor_changed:
                transition_sensor_changes += 1

            # Frames OLCI que caen en el intervalo (t_r, t_r_next]
            # Cumpliendo regla D9 y causalidad: t_r <= t_f <= t_r_next
            interval_frames = [
                s for s in olci_scenes
                if t_r <= s["dt"] <= t_r_next
            ]

            transition_info = {
                "r_from": r_curr["monitoring_number"],
                "r_to": r_next["monitoring_number"],
                "t_from": t_r.isoformat(),
                "t_to": t_r_next.isoformat(),
                "dt_hours": round(dt_hours, 1),
                "sensor_from": sensor_curr,
                "sensor_to": sensor_next,
                "sensor_changed": sensor_changed,
                "burnt_delta_ha": round(r_next["burnt_ha"] - r_curr["burnt_ha"], 1),
                "olci_frames_count": len(interval_frames),
            }
            transitions.append(transition_info)

            for f in interval_frames:
                t_f = f["dt"]
                # Horizonte h en días hacia la ronda siguiente r+1
                h_days = (t_r_next - t_f).total_seconds() / 86400.0
                horizons_days.append(round(h_days, 2))

                # Frames previos para el tensor temporal (hasta 6 frames pasados)
                past_frames = [s for s in olci_scenes if s["dt"] <= t_f]
                # Pre-fire: último frame anterior al primer indicio (search_start..first_round_dt)
                pre_fire_candidates = [s for s in olci_scenes if s["dt"] < first_round_dt]
                pre_fire_id = pre_fire_candidates[-1]["id"] if pre_fire_candidates else None

                sample_entry = {
                    "sample_id": f"{series_id}__F_{t_f.strftime('%Y%m%dT%H%M%S')}",
                    "frame_time": t_f.isoformat(),
                    "frame_scene_id": f["id"],
                    "frame_satellite": f["satellite"],
                    "target_round": r_next["monitoring_number"],
                    "target_time": t_r_next.isoformat(),
                    "horizon_days": round(h_days, 3),
                    "sensor_from": sensor_curr,
                    "sensor_to": sensor_next,
                    "sensor_changed": sensor_changed,
                    "past_frames_count": min(len(past_frames), 7),
                    "pre_fire_scene_id": pre_fire_id,
                }
                series_samples.append(sample_entry)
                total_samples += 1

        print(f"   -> {len(parsed_rounds)} rondas, {len(transitions)} transiciones, "
              f"{len(olci_scenes)} pasadas OLCI, {len(series_samples)} muestras (f, h)")

        manifest_series.append({
            "series_id": series_id,
            "code": code,
            "aoi_number": aoi_num,
            "name": s.get("name"),
            "geo_mediterranean": is_geo_med,
            "in_tmb_test": s.get("tmb_split") == "test",
            "center": s.get("center"),
            "bbox_wkt": aoi_bbox_wkt,
            "rounds": [
                {k: v for k, v in r.items() if k != "dt"}
                for r in parsed_rounds
            ],
            "transitions": transitions,
            "samples": series_samples,
        })

    # Resumen y estadísticas
    print("\n" + "=" * 72)
    print("INVENTARIO Y RESULTADOS DE AUDITORIA (P7 y R14)")
    print("=" * 72)
    print(f"Series procesadas:                 {len(manifest_series)}")
    print(f"Transiciones totales:              {total_transitions}")
    pct_sensor_change = (
        (100.0 * transition_sensor_changes / total_transitions) if total_transitions else 0.0
    )
    print(
        f"Transiciones con cambio de sensor: {transition_sensor_changes} "
        f"({pct_sensor_change:.1f}%) [R14]"
    )
    print(f"Total de muestras generadas (f, h):{total_samples} [P7 RESUELTO]")

    if horizons_days:
        h_sorted = sorted(horizons_days)
        p25 = h_sorted[int(0.25 * len(h_sorted))]
        p50 = h_sorted[int(0.50 * len(h_sorted))]
        p75 = h_sorted[int(0.75 * len(h_sorted))]
        print(f"Horizontes h (días):               min {min(horizons_days):.2f} · p25 {p25:.2f} · "
              f"mediana {p50:.2f} · p75 {p75:.2f} · max {max(horizons_days):.2f}")

    print("\nDistribución de sensores en rondas CEMS:")
    for sensor, count in sorted(sensor_counter.items(), key=lambda x: -x[1]):
        print(f"   - {sensor:<20}: {count:>3} rondas")

    # Guardar manifiesto
    manifest_catalog_summary = deduplicate_manifest_samples(manifest_series)
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "summary": {
            "total_series": len(manifest_series),
            "pure_mediterranean": sum(1 for s in manifest_series if s["geo_mediterranean"]),
            "overseas_or_atlantic": sum(1 for s in manifest_series if not s["geo_mediterranean"]),
            "tmb_test_control": sum(1 for s in manifest_series if s["in_tmb_test"]),
            "total_transitions": total_transitions,
            "sensor_change_transitions": transition_sensor_changes,
            "total_samples_olci": total_samples,
            "olci_catalog": manifest_catalog_summary,
        },
        "sensor_distribution": sensor_counter,
        "series": manifest_series,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nManifiesto guardado con éxito en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
