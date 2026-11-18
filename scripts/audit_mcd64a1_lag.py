#!/usr/bin/env python3
"""
Diagnóstico R19: Desfase temporal entre BurnDate de MCD64A1 y observed_time de CEMS.

Consulta MODIS/061/MCD64A1 en Google Earth Engine sobre los parches 64x64 (300 m)
de las series CEMS, compara la fecha de quema detectada por MODIS (`BurnDate`)
contra la fecha de observación de CEMS (`acquisition_time`) y guarda el resumen
estadístico (mediana, percentiles, incertidumbre media y recomendación de shift).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import urllib.request
from pathlib import Path

import ee
import numpy as np
import rasterio

MCD_BANDS = ["BurnDate", "Uncertainty", "QA"]


def _parse_iso_date(ts: str) -> dt.date:
    s = ts.replace("Z", "").split("+")[0]
    return dt.datetime.fromisoformat(s).date()


def _utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
    return f"EPSG:{epsg}"


def _download_mcd_patch(
    image: ee.Image, region: ee.Geometry, crs: str, dimensions: tuple[int, int] = (64, 64)
) -> np.ndarray:
    url = image.getDownloadURL(
        {
            "region": region,
            "dimensions": list(dimensions),
            "crs": crs,
            "format": "GEO_TIFF",
        }
    )
    req = urllib.request.Request(url, headers={"User-Agent": "burn-scar-r19-audit/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read()


def audit_series_lag(series_meta: dict, labels_dir: Path) -> dict:
    sid = series_meta["series_id"]
    lon_c, lat_c = series_meta["center"]
    crs_str = _utm_crs(lon_c, lat_c)
    point = ee.Geometry.Point([lon_c, lat_c])
    region = point.buffer(9600).bounds()

    rounds = [r for r in series_meta.get("rounds", []) if r.get("acquisition_time")]
    if not rounds:
        return {"series_id": sid, "status": "no_rounds"}

    round_dates = [_parse_iso_date(r["acquisition_time"]) for r in rounds]
    min_date = min(round_dates)
    max_date = max(round_dates)

    # Ventana de consulta mensual en MCD64A1: desde inicio del mes de min_date
    # hasta 45 días después de max_date para capturar posibles desfases de R19.
    start_month = dt.date(min_date.year, min_date.month, 1)
    end_query = max_date + dt.timedelta(days=45)

    col = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterBounds(point)
        .filterDate(start_month.strftime("%Y-%m-%d"), end_query.strftime("%Y-%m-%d"))
        .select(MCD_BANDS)
    )
    img_list = col.toList(12).getInfo()
    if not img_list:
        return {"series_id": sid, "status": "no_mcd64a1_images"}

    # Cargar etiquetas rasterizadas de CEMS si existen para comparar píxel a píxel
    label_path = labels_dir / f"{sid}.npz"
    cems_first_ord = None
    if label_path.exists():
        npz = np.load(label_path)
        quemado = npz["quemado"]  # (T, 64, 64)
        round_nums = list(npz["round_numbers"])
        round_num_to_date = {
            r["monitoring_number"]: _parse_iso_date(r["acquisition_time"])
            for r in rounds
            if "monitoring_number" in r
        }
        cems_first_ord = np.full((64, 64), fill_value=-1, dtype=np.int32)
        for t_idx, r_num in enumerate(round_nums):
            r_date = round_num_to_date.get(int(r_num), min_date)
            newly_seen = (quemado[t_idx] >= 0.25) & (cems_first_ord < 0)
            cems_first_ord[newly_seen] = r_date.toordinal()

    pixel_lags: list[float] = []
    uncertainties: list[float] = []
    mcd_burn_pixels = 0

    for item in img_list:
        img_id = item["id"]
        props = item.get("properties", {})
        time_start = props.get("system:time_start", 0)
        img_year = dt.datetime.fromtimestamp(time_start / 1000.0, tz=dt.UTC).year

        try:
            arr = _download_mcd_patch(ee.Image(img_id).select(MCD_BANDS), region, crs_str)
        except Exception as exc:
            return {"series_id": sid, "status": f"download_error: {exc}"}

        burn_doy = arr[0].astype(np.int32)
        unc = arr[1].astype(np.float32)

        valid_burn = burn_doy > 0
        if not np.any(valid_burn):
            continue

        mcd_burn_pixels += int(valid_burn.sum())
        base_ord = dt.date(img_year, 1, 1).toordinal() - 1
        burn_ord = base_ord + burn_doy

        if cems_first_ord is not None and np.any(valid_burn & (cems_first_ord > 0)):
            overlap = valid_burn & (cems_first_ord > 0)
            lags = (burn_ord[overlap] - cems_first_ord[overlap]).astype(np.float32)
            pixel_lags.extend(lags.tolist())
            uncertainties.extend(unc[overlap].tolist())
        else:
            ref_ord = min_date.toordinal()
            lags = (burn_ord[valid_burn] - ref_ord).astype(np.float32)
            pixel_lags.extend(lags.tolist())
            uncertainties.extend(unc[valid_burn].tolist())

    if not pixel_lags:
        return {
            "series_id": sid,
            "status": "no_burned_pixels_in_mcd64a1",
            "first_cems_date": min_date.isoformat(),
        }

    lags_np = np.array(pixel_lags, dtype=np.float32)
    unc_np = np.array(uncertainties, dtype=np.float32)
    return {
        "series_id": sid,
        "status": "ok",
        "first_cems_date": min_date.isoformat(),
        "n_mcd_burn_pixels": mcd_burn_pixels,
        "n_compared_pixels": int(lags_np.size),
        "lag_median_days": float(np.median(lags_np)),
        "lag_mean_days": float(np.mean(lags_np)),
        "lag_p25_days": float(np.percentile(lags_np, 25)),
        "lag_p75_days": float(np.percentile(lags_np, 75)),
        "uncertainty_mean_days": float(np.mean(unc_np)) if unc_np.size else 0.0,
        "_raw_lags": lags_np.tolist(),
    }


def compute_summary_stats(results: list[dict]) -> dict:
    ok_series = [r for r in results if r.get("status") == "ok"]
    series_medians = [r["lag_median_days"] for r in ok_series]
    all_pixel_lags = [lag for r in ok_series for lag in r.get("_raw_lags", [])]
    all_unc = [r["uncertainty_mean_days"] for r in ok_series]

    if not series_medians:
        return {
            "n_series_total": len(results),
            "n_series_with_mcd_burn": 0,
            "lag_series_median_days": None,
            "lag_pixel_median_days": None,
            "recommended_shift_days": 0,
            "r19_decision": "insufficient_data",
        }

    s_arr = np.array(series_medians, dtype=np.float32)
    p_arr = np.array(all_pixel_lags, dtype=np.float32)
    med_days = float(np.median(s_arr))
    # Regla R19 (docs/decisiones.md): se acepta si |desfase mediano| <= 2 días;
    # se corrige con un shift si es mayor.
    shift = 0 if abs(med_days) <= 2.0 else int(round(med_days))

    return {
        "n_series_total": len(results),
        "n_series_with_mcd_burn": len(ok_series),
        "n_pixels_compared": int(p_arr.size),
        "lag_series_median_days": med_days,
        "lag_series_mean_days": float(np.mean(s_arr)),
        "lag_series_p10_days": float(np.percentile(s_arr, 10)),
        "lag_series_p25_days": float(np.percentile(s_arr, 25)),
        "lag_series_p75_days": float(np.percentile(s_arr, 75)),
        "lag_series_p90_days": float(np.percentile(s_arr, 90)),
        "lag_pixel_median_days": float(np.median(p_arr)),
        "lag_pixel_p25_days": float(np.percentile(p_arr, 25)),
        "lag_pixel_p75_days": float(np.percentile(p_arr, 75)),
        "mean_uncertainty_days": float(np.mean(all_unc)) if all_unc else 0.0,
        "recommended_shift_days": shift,
        "r19_decision": (
            "accept_as_is (|median_lag| <= 2d)"
            if shift == 0
            else f"apply_shift ({shift:+d}d)"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnóstico R19: desfase entre BurnDate (MCD64A1) y observed_time (CEMS)"
    )
    parser.add_argument("--manifest", default="data/cems_manifest_46.json")
    parser.add_argument("--labels-dir", default="data/processed/cems_labels")
    parser.add_argument("--output", default="data/mcd64a1_lag_audit.json")
    parser.add_argument("--limit", type=int, default=None, help="Límite de series a consultar")
    args = parser.parse_args()

    ee.Initialize()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    series_list = manifest.get("series", [])
    if args.limit:
        series_list = series_list[: args.limit]

    print("=" * 72)
    print("DIAGNÓSTICO R19: DESFASE TEMPORAL MCD64A1 (BurnDate) vs CEMS")
    print(f"Series a auditar: {len(series_list)}")
    print("=" * 72)

    labels_dir = Path(args.labels_dir)
    results: list[dict] = []
    for idx, s in enumerate(series_list, 1):
        sid = s["series_id"]
        res = audit_series_lag(s, labels_dir)
        results.append(res)
        if res.get("status") == "ok":
            print(
                f"[{idx:>2}/{len(series_list)}] {sid:<16} -> "
                f"mediana={res['lag_median_days']:+5.1f} d "
                f"(IQR {res['lag_p25_days']:+5.1f}..{res['lag_p75_days']:+5.1f} d, "
                f"n={res['n_compared_pixels']} px, unc={res['uncertainty_mean_days']:.1f} d)"
            )
        else:
            print(f"[{idx:>2}/{len(series_list)}] {sid:<16} -> {res.get('status')}")

    summary = compute_summary_stats(results)
    clean_results = [{k: v for k, v in r.items() if k != "_raw_lags"} for r in results]

    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "manifest": args.manifest,
        "summary": summary,
        "series": clean_results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("RESUMEN R19 (BurnDate - CEMS observed_time)")
    print("=" * 72)
    print(
        f"Series con detección MCD64A1: "
        f"{summary['n_series_with_mcd_burn']}/{summary['n_series_total']}"
    )
    if summary["lag_series_median_days"] is not None:
        print(f"Desfase mediano por serie:    {summary['lag_series_median_days']:+.2f} días")
        print(
            f"Percentiles serie (p25..p75): {summary['lag_series_p25_days']:+.2f} .. "
            f"{summary['lag_series_p75_days']:+.2f} días"
        )
        print(
            f"Percentiles serie (p10..p90): {summary['lag_series_p10_days']:+.2f} .. "
            f"{summary['lag_series_p90_days']:+.2f} días"
        )
        print(f"Desfase mediano por píxel:    {summary['lag_pixel_median_days']:+.2f} días")
        print(f"Incertidumbre media MODIS:    {summary['mean_uncertainty_days']:.2f} días")
        print(f"Decisión R19:                 {summary['r19_decision']}")
    print(f"Reporte guardado en:          {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
