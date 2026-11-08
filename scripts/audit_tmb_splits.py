#!/usr/bin/env python3
"""
Auditoría de splits: ¿qué incendios mediterráneos de CEMS ha visto ya TMB-S3?

TMB-S3 se entrenó con CEMS y usó un split por cluster geográfico (DBSCAN, eps=150 km).
Antes de construir el dataset hay que saber cuáles de nuestras series están dentro de
su train/val, porque en ellas la cicatriz de entrada sería el perímetro CEMS *memorizado*
y el target cambiaría de significado (operacion.md §4 y decisiones.md §3).

Uso:
    python3 scripts/audit_tmb_splits.py                  # resumen por stdout
    python3 scripts/audit_tmb_splits.py --json out.json  # vuelca el detalle
    python3 scripts/audit_tmb_splits.py --paises-extra   # añade Chipre, Turquía, etc.

Solo biblioteca estándar. No descarga imagen: son metadatos.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------- constantes

CEMS_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "links-ads/tmb-s3"

# Países mediterráneos según el censo documentado (dataset.md §3): 14 países.
# Aviso: "France" incluye el norte de Francia y el territorio francés de ultramar
# (Nueva Caledonia, Kerguelen), que geográficamente no son mediterráneos.
# Se marcan aparte con GEO_MED (decisiones.md §1.3 BL6).
MED_COUNTRIES = {
    "Albania", "Bosnia and Herzegovina", "Bosnia and Herz.", "Bulgaria",
    "Croatia", "France", "Greece", "Italy", "Montenegro", "Portugal",
    "Serbia", "Slovenia", "Spain", "Syria", "Tunisia",
}

# Países del Mediterráneo oriental y del norte de África que quedan FUERA del
# censo documentado. Hoy no tienen ninguna activación de incendio en la API, pero
# si las hubiera el script dejaría de reproducir las cifras de dataset.md §3: por eso
# quedan aparte y solo se activan con --paises-extra.
MED_COUNTRIES_EXTRA = {
    "Cyprus", "Turkey", "Türkiye", "Morocco", "Egypt", "Libya",
    "Lebanon", "Israel", "Palestine",
}

# Recuadro que sí es "Mediterráneo" en sentido geográfico (criterio de 9.4.1).
GEO_MED_LAT = (30.0, 46.5)
GEO_MED_LON = (-6.0, 31.0)

R_KM = 6371.0
NUM_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


# ---------------------------------------------------------------- utilidades

def get_json(url: str, tries: int = 8, timeout: int = 60):
    """GET con reintentos y backoff (la API de CEMS devuelve 429 si te apresuras)."""
    delay = 1.0
    for _ in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "audit-tmb-splits/1.0"})
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


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distancia entre dos puntos (lon, lat) en km."""
    lon1, lat1 = a
    lon2, lat2 = b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_KM * math.asin(math.sqrt(h))


def wkt_centroid(wkt: str):
    """Centroide (lon, lat) de un POLYGON o POINT en WKT. None si no se puede."""
    if not wkt:
        return None
    pairs = NUM_PAIR.findall(wkt)
    if not pairs:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def is_geo_mediterranean(c: tuple[float, float]) -> bool:
    return (GEO_MED_LAT[0] <= c[1] <= GEO_MED_LAT[1]
            and GEO_MED_LON[0] <= c[0] <= GEO_MED_LON[1])


# ---------------------------------------------------------------- TMB-S3

def fetch_tmb_manifest() -> list:
    """Las 770 filas del manifest.parquet de TMB-S3 vía el servidor de datasets de HF."""
    rows: list = []
    offset = 0
    while True:
        url = (f"{HF_ROWS}?dataset={HF_DATASET}&config=manifest&split=train"
               f"&offset={offset}&length=100")
        data = get_json(url)
        if not data:
            break
        batch = [r["row"] for r in data.get("rows", [])]
        rows.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
    return rows


def index_tmb(rows: list) -> dict:
    """Código EMSR -> {split, cajas:[centro], centro:(lon,lat)}."""
    by_code: dict = {}
    for row in rows:
        code = row["bbox_id"].split("-")[0]
        entry = by_code.setdefault(code, {"split": row["split"], "boxes": []})
        c = wkt_centroid(row.get("geometry_wkt", ""))
        if c:
            entry["boxes"].append(c)
    for entry in by_code.values():
        boxes = entry["boxes"]
        entry["center"] = (
            (sum(b[0] for b in boxes) / len(boxes),
             sum(b[1] for b in boxes) / len(boxes))
            if boxes else None)
    return by_code


# ---------------------------------------------------------------- CEMS

def fetch_mediterranean_activations(countries: set[str]) -> list:
    """Activaciones de incendio mediterráneas según la API de CEMS."""
    activations: list = []
    url = f"{CEMS_API}/public-activations-info/?limit=500"
    while url:
        data = get_json(url)
        if not data:
            break
        activations.extend(data.get("results", []))
        url = data.get("next")
    return [
        a for a in activations
        if a.get("category") == "Wildfire"
        and any(c in countries for c in (a.get("countries") or []))
    ]


def fetch_series(activations: list) -> list:
    """Una 'serie' = un AOI con 2 o mas delineaciones DEL (senal de crecimiento)."""
    series: list = []
    for act in activations:
        code = act["code"]
        detail = get_json(f"{CEMS_API}/public-activations/?code={code}")
        if not detail or not detail.get("results"):
            print(f"  ! no se pudo leer {code}", file=sys.stderr)
            continue
        d = detail["results"][0]
        for aoi in d.get("aois", []):
            dels = [p for p in (aoi.get("products") or []) if p.get("type") == "DEL"]
            if len(dels) < 2:
                continue
            c = wkt_centroid(aoi.get("extent", ""))
            if not c:
                continue
            series.append({
                "code": code,
                "aoi": aoi.get("number"),
                "name": aoi.get("name"),
                "n_del": len(dels),
                "center": c,
                "geo_med": is_geo_mediterranean(c),
            })
    return series


# ---------------------------------------------------------------- principal

def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoria de splits TMB-S3 vs CEMS")
    ap.add_argument("--json", help="fichero donde volcar el detalle")
    ap.add_argument(
        "--paises-extra",
        action="store_true",
        help="incluir Chipre, Turqua, Marruecos, etc. (fuera del censo documentado)",
    )
    args = ap.parse_args()

    print("1/3 · manifest de TMB-S3")
    rows = fetch_tmb_manifest()
    if not rows:
        print("No se pudo leer el manifest. Abortando.", file=sys.stderr)
        return 1
    tmb = index_tmb(rows)
    split_count: dict = defaultdict(int)
    for r in rows:
        split_count[r["split"]] += 1
    print(f"    {len(rows)} cajas · {len(tmb)} activaciones · splits {dict(split_count)}")

    print("2/3 · activaciones mediterraneas de CEMS")
    countries = MED_COUNTRIES | (MED_COUNTRIES_EXTRA if args.paises_extra else set())
    acts = fetch_mediterranean_activations(countries)
    print(f"    {len(acts)} activaciones de incendio mediterraneas")

    print("3/3 · series con senal de crecimiento")
    series = fetch_series(acts)
    print(f"    {len(series)} series en {len({s['code'] for s in series})} activaciones\n")

    # --- cruce -------------------------------------------------------------
    tmb_centers = [(code, e["center"]) for code, e in tmb.items() if e["center"]]

    for s in series:
        entry = tmb.get(s["code"])
        s["in_tmb"] = entry is not None
        s["tmb_split"] = entry["split"] if entry else None
        # distancia al evento de TMB-S3 más cercano (con o sin coincidir el código)
        if tmb_centers:
            s["dist_event_km"] = round(min(
                haversine_km(s["center"], c) for _, c in tmb_centers), 1)
        else:
            s["dist_event_km"] = None
        # distancia al evento de TMB-S3 con el MISMO código, si existe
        s["dist_same_code_km"] = (
            round(haversine_km(s["center"], entry["center"]), 1)
            if entry and entry["center"] else None)

    direct = [s for s in series if s["in_tmb"]]
    by_split: dict = defaultdict(int)
    for s in direct:
        by_split[s["tmb_split"]] += 1

    # train+val = los pesos los aprendieron -> hay que excluirlos
    leaked = [s for s in direct if s["tmb_split"] in ("train", "val")]
    clean = [s for s in series if not s["in_tmb"] or s["tmb_split"] == "test"]
    clean_geo = [s for s in clean if s["geo_med"]]

    dists = sorted(s["dist_event_km"] for s in series
                   if s["dist_event_km"] is not None)

    def pct(p: float) -> float:
        if not dists:
            return float("nan")
        return dists[min(int(p * (len(dists) - 1)), len(dists) - 1)]

    print("=" * 68)
    print("RESULTADO DE LA AUDITORIA")
    print("=" * 68)
    print(f"Series mediterraneas                        {len(series)}")
    print(f"Actividades con senal                       {len({s['code'] for s in series})}")
    print(f"  de ellas, genuinamente mediterraneas      {sum(1 for s in series if s['geo_med'])}")
    print()
    print(f"Series cuyo codigo esta en TMB-S3           {len(direct)} "
          f"({100 * len(direct) / len(series):.0f}%)")
    for sp in ("train", "val", "test"):
        nota = "pesos entrenados -> EXCLUIR" if sp in ("train", "val") \
            else "pesos no los vieron"
        print(f"    en su {sp:<5} {by_split[sp]:>3}   <- {nota}")
    print()
    print(f"Distancia mediana al evento TMB-S3 mas cerca  {pct(0.50):>6} km")
    print(f"                                  p75 {pct(0.75):>6} km")
    print(f"                                  p90 {pct(0.90):>6} km")
    print()
    print("-" * 68)
    print(f"SERIES A EXCLUIR (train+val de TMB-S3)      {len(leaked)}")
    print(f"SERIES UTILIZABLES                          {len(clean)}  "
          f"({len(clean_geo)} genuinamente mediterraneas)")
    print(f"   +- nunca vistas por TMB-S3               "
          f"{sum(1 for s in clean if not s['in_tmb'])}")
    print(f"   +- en el test de TMB-S3 (control)         "
          f"{sum(1 for s in clean if s['in_tmb'])}")
    print("-" * 68)
    print()
    print("Subconjuntos por criterio de cercania (solo sobre las utilizables):")
    for thr in (50, 100, 150):
        n = sum(1 for s in clean if s["dist_event_km"] is not None
                and s["dist_event_km"] < thr)
        ng = sum(1 for s in clean_geo if s["dist_event_km"] is not None
                 and s["dist_event_km"] < thr)
        print(f"    < {thr:>3} km: {n:>3} series ({ng} mediterraneas)")
    # Complemento: decisiones.md §3 reporta las series a >=150 km de cualquier
    # evento de TMB-S3 (6 series, 2 mediterraneas en el censo del 2026-11-07).
    for thr in (150,):
        n = sum(1 for s in clean if s["dist_event_km"] is not None
                and s["dist_event_km"] >= thr)
        ng = sum(1 for s in clean_geo if s["dist_event_km"] is not None
                 and s["dist_event_km"] >= thr)
        print(f"    >= {thr:>3} km: {n:>3} series ({ng} mediterraneas)")
    print()
    print("Aviso: el criterio de 150 km deja muy pocas series; se usa como dato,")
    print("       no como criterio de reparto (decisiones.md §3).")
    print()

    if args.json:
        payload = {
            "tmb_splits": dict(split_count),
            "tmb_activations": len(tmb),
            "series_total": len(series),
            "series_in_tmb": len(direct),
            "by_tmb_split": dict(by_split),
            "excluir_train_val": len(leaked),
            "utilizables": len(clean),
            "utilizables_geo_med": len(clean_geo),
            "dist_p50_km": pct(0.50),
            "dist_p75_km": pct(0.75),
            "dist_p90_km": pct(0.90),
            "series": series,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"Detalle escrito en {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
