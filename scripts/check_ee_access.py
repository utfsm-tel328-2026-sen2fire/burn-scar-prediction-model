#!/usr/bin/env python3
"""
Verifica el acceso a Earth Engine y dice qué falta exactamente.

Se ejecuta antes de `scripts/download_gee_olci.py` (que es el paso que
necesita el acceso) y también sirve de diagnóstico cuando la descarga falla.

    python scripts/check_ee_access.py

Salida: 0 si todo está bien, 1 si falta algo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_gee_olci import (  # noqa: E402
    load_dotenv_file,
    resolve_ee_credentials,
    wkt_to_geojson,
)


def _titulo(ok: bool, texto: str) -> None:
    print(f"  [{'OK ' if ok else 'FALLA'}] {texto}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba el acceso a Earth Engine")
    parser.add_argument(
        "--manifest",
        default="data/cems_manifest_46.json",
        help="Manifiesto CEMS (para probar una consulta real de la colección)",
    )
    args = parser.parse_args()

    env = {**load_dotenv_file(), **os.environ}
    config = resolve_ee_credentials(env)
    proyecto = config["project"]
    cuenta = config["service_account"]

    print("=" * 72)
    print("COMPROBACIÓN DE ACCESO A EARTH ENGINE")
    print("=" * 72)
    print(f"  Proyecto:            {proyecto or '(sin EARTHENGINE_PROJECT)'}")
    print(f"  Cuenta de servicio:  {cuenta or '(no: se usan credenciales personales)'}")
    print()

    try:
        import ee
    except ImportError as exc:
        print(f"  [FALLA] earthengine-api no está instalado: {exc}")
        return 1

    # 1. La API de Earth Engine tiene que estar habilitada en el proyecto.
    #    Ojo: consultar el estado con la propia cuenta de servicio exige
    #    `serviceusage.services.use`, así que sin ese rol la comprobación no se
    #    puede hacer y no debeContar como fallo.
    api_habilitada = None
    if proyecto:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            import requests

            creds = service_account.Credentials.from_service_account_file(
                cuenta, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(Request())
            url = (
                f"https://serviceusage.googleapis.com/v1/projects/{proyecto}"
                "/services/earthengine.googleapis.com"
            )
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=30,
            )
            if resp.status_code == 200:
                api_habilitada = resp.json().get("state") == "ENABLED"
                _titulo(
                    api_habilitada,
                    f"API de Earth Engine habilitada en {proyecto}",
                )
            else:
                print(
                    "  [--] no se pudo consultar el estado de la API "
                    "(la cuenta de servicio no tiene serviceusage.services.use)"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [--] no se pudo consultar el estado de la API: {type(exc).__name__}")

    # 2. Inicializar la biblioteca.
    try:
        if cuenta:
            creds = ee.ServiceAccountCredentials(key_file=cuenta)
            ee.Initialize(credentials=creds, project=proyecto or None)
        else:
            ee.Initialize(project=proyecto or None)
        _titulo(True, "ee.Initialize() correcto")
    except Exception as exc:  # noqa: BLE001
        _titulo(False, f"ee.Initialize(): {str(exc)[:220]}")
        print()
        print("  Causa más probable y cómo se arregla:")
        print("    · Falta el rol 'Service Usage Consumer' "
              "(roles/serviceusage.serviceUsageConsumer) en la cuenta de servicio.")
        print("      https://console.cloud.google.com/iam-admin/iam?project="
              f"{proyecto or 'TU_PROYECTO'}")
        print("    · O falta registrar el proyecto para Earth Engine:")
        print(f"      https://code.earthengine.google.com/register?project={proyecto or 'TU_PROYECTO'}")
        print("    · Tras conceder un rol, la propagación tarda unos minutos.")
        return 1

    # 3. La cuenta de servicio debe poder leer el manifiesto para probar la
    #    consulta real de la colección OLCI.
    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        series = [s for s in manifest.get("series", []) if s.get("bbox_wkt")]
        if series:
            s = series[0]
            try:
                col = (
                    ee.ImageCollection("COPERNICUS/S3/OLCI")
                    .filterBounds(ee.Geometry(wkt_to_geojson(s["bbox_wkt"])))
                    .filterDate("2026-01-01", "2027-12-31")
                )
                n = len(col.toList(500).getInfo())
                _titulo(True, f"consulta de la colección OLCI sobre {s['series_id']}: {n} imágenes")
            except Exception as exc:  # noqa: BLE001
                _titulo(False, f"consulta de la colección: {str(exc)[:220]}")
                print()
                print("  Si el error menciona 'serviceusage.services.use', falta el rol")
                print("  'Service Usage Consumer' en la cuenta de servicio.")
                return 1

    print()
    print("  Todo listo para scripts/download_gee_olci.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
